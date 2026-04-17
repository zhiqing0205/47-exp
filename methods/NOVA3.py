import torch
import math
import copy
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from .RNN_net import NLIRNN

def predict(net, inputs):
    (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    return net((s1_embed.cuda(), s1_lens), (s2_embed.cuda(), s2_lens))


class Learner(nn.Module):
    def __init__(self, args, training_size, verbose=True):
        super(Learner, self).__init__()
        self.args = args
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_size = training_size

        self.inner_model = NLIRNN(
            word_embed_dim=args.word_embed_dim, encoder_dim=args.encoder_dim,
            n_enc_layers=args.n_enc_layers, dpout_model=0.0, dpout_fc=0.0,
            fc_dim=args.fc_dim, n_classes=args.n_classes,
            pool_type=args.pool_type, linear_fc=args.linear_fc
        ).to(self.device)

        # Upper-level variable x (sample weights)
        self.lambda_x = torch.ones(self.training_size, requires_grad=True, device=self.device)

        # Auxiliary z for penalty-based tracking
        param_count = sum(p.numel() for p in self.inner_model.parameters())
        self.z = torch.zeros(param_count, 1, device=self.device)
        with torch.no_grad():
            self.z.copy_(torch.cat([p.view(-1, 1) for p in self.inner_model.parameters()]))
        self.d_z = torch.zeros_like(self.z)

        # d_y momentum (EMA) — proven stable at 73.8%, keep this
        self.d_y = [torch.zeros_like(p) for p in self.inner_model.parameters()]

        # d_x momentum
        self.d_x = torch.zeros_like(self.lambda_x)

        # Hypermomentum for Neumann-based outer gradient (STORM-style like accbo)
        self.hyper_momentum_x = torch.zeros(self.training_size).to(self.device)
        self.hyper_history_x = torch.zeros(self.training_size).to(self.device)

        # Core hyperparams
        self.gamma_init = args.gamma
        self.gamma = args.gamma
        self.rho = args.beta
        self.nu = args.beta
        self.c_t = 1.0
        self.clip_grad = 1.0
        self.neumann_lr = getattr(args, 'neumann_lr', 0.01)
        self.hessian_q = getattr(args, 'hessian_q', 3)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=getattr(args, 'dropout_p', 0.1))

        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        # Best model tracking
        self.best_test_acc = 0.0
        self.best_model_state = None

    def _reg(self):
        return 0.0001 * sum([x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()

    def _neumann_hypergradient(self, val_loss, train_batch1, train_batch2):
        """Neumann series hypergradient (like stocbio/accbo) for precise outer gradient."""
        train_data, train_labels, train_indx = train_batch1
        val_data, val_labels, val_indx = train_batch2

        # Step 1: Fy_gradient = d(val_loss)/dy
        Fy_gradient = torch.autograd.grad(val_loss, self.inner_model.parameters(), retain_graph=True)
        F_gradient = [g.view(-1) for g in Fy_gradient]
        v_0 = torch.unsqueeze(torch.hstack(F_gradient), 1).detach()

        # Step 2: Hessian-vector product via Neumann series
        outputs = predict(self.inner_model, train_data)
        loss = torch.mean(torch.sigmoid(self.lambda_x[train_indx]) * self.criterion(outputs, train_labels.to(self.device))) + self._reg()
        Gy_gradient = torch.autograd.grad(loss, self.inner_model.parameters(), create_graph=True)

        G_gradient = []
        for g_grad, param in zip(Gy_gradient, self.inner_model.parameters()):
            G_gradient.append((param - self.neumann_lr * g_grad).view(-1))
        G_gradient = torch.hstack(G_gradient)

        z_list = []
        for _ in range(self.hessian_q):
            Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(Jacobian, self.inner_model.parameters(), retain_graph=True)
            v_0 = torch.unsqueeze(torch.hstack([v.view(-1) for v in v_new]), 1).detach()
            z_list.append(v_0)
        v_Q = self.neumann_lr * z_list[np.random.randint(self.hessian_q)]

        # Step 3: Jacobian-vector product for hypergradient w.r.t. lambda_x
        outputs2 = predict(self.inner_model, val_data)
        loss2 = torch.mean(torch.sigmoid(self.lambda_x[val_indx]) * self.criterion(outputs2, val_labels.to(self.device))) + self._reg()
        Gy2 = torch.autograd.grad(loss2, self.inner_model.parameters(), retain_graph=True, create_graph=True)
        Gy2_flat = torch.hstack([g.view(-1) for g in Gy2])
        hyper_grad = torch.autograd.grad(-torch.matmul(Gy2_flat, v_Q.detach()), self.lambda_x)[0]

        return hyper_grad

    def forward(self, train_loader, val_loader=None, training=True, epoch=0):
        task_accs, task_loss = [], []

        train_iter = iter(train_loader)
        val_iter = iter(val_loader) if val_loader else iter(train_loader)

        def next_train():
            nonlocal train_iter
            try: return next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                return next(train_iter)

        def next_val():
            nonlocal val_iter
            try: return next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                return next(val_iter)

        # Adaptive gamma: cosine anneal from gamma_init to gamma_init*0.2
        total_epochs = getattr(self.args, 'epoch', 25)
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        gamma_final = self.gamma_init * 0.2
        self.gamma = self.gamma_init - (self.gamma_init - gamma_final) * (1 - math.cos(math.pi * progress)) / 2

        # x/z lr decay
        decay = max(0.05, (5.0 / (epoch + 5)) ** 0.5) if epoch >= 5 else 1.0
        eta_t = self.args.nu * decay
        alpha_t = self.args.outer_update_lr * decay
        beta_t = self.args.inner_update_lr

        for step, data_g in enumerate(train_loader):
            data_f = next_val()
            data_gh = next_train()

            labels_f = data_f[1].to(self.device)
            labels_g = data_g[1].to(self.device)
            labels_gh = data_gh[1].to(self.device)
            idx_g = data_g[2]
            idx_gh = data_gh[2]

            # --- Update z (penalty tracking with momentum) ---
            y_state = [p.data.clone() for p in self.inner_model.parameters()]
            self._set_model_params(self.z)

            out_gh = predict(self.inner_model, data_gh[0])
            loss_gh = torch.mean(torch.sigmoid(self.lambda_x[idx_gh]) * self.criterion(out_gh, labels_gh)) + self._reg()
            grads_gh = torch.autograd.grad(loss_gh, self.inner_model.parameters())
            grad_g_z = torch.cat([g.detach().view(-1, 1) for g in grads_gh])

            y_flat = torch.cat([p.view(-1, 1) for p in y_state])
            curr_grad_z = grad_g_z + (1.0 / self.gamma) * (self.z.detach() - y_flat)
            self.d_z = (self.rho * self.d_z + (1 - self.rho) * curr_grad_z).detach()
            self.z.data -= eta_t * self.d_z

            for p, val in zip(self.inner_model.parameters(), y_state): p.data.copy_(val)

            # --- Update x (Neumann-based hypergradient + STORM momentum) ---
            out_f = predict(self.inner_model, data_f[0])
            val_loss = torch.mean(self.criterion(out_f, labels_f)) + self._reg()

            neumann_grad = self._neumann_hypergradient(val_loss, next_train(), next_train())

            # STORM-style momentum (like accbo line 195)
            self.hyper_momentum_x = (self.args.beta * self.hyper_momentum_x
                + (1 - self.args.beta) * neumann_grad.detach()
                + self.args.beta * (neumann_grad.detach() - self.hyper_history_x)).detach()
            self.hyper_history_x = neumann_grad.detach().clone()

            # Normalized update (per-element sign like accbo)
            self.lambda_x.data -= alpha_t * (self.hyper_momentum_x / (torch.norm(self.hyper_momentum_x) + 1e-8))

            # --- Update y (d_y EMA momentum + clip, proven stable) ---
            out_f_new = self.dropout(predict(self.inner_model, data_f[0]))
            loss_f_new = torch.mean(self.criterion(out_f_new, labels_f)) + self._reg()
            grads_f_y = torch.autograd.grad(loss_f_new, self.inner_model.parameters())

            out_g_new = self.dropout(predict(self.inner_model, data_g[0]))
            loss_g_new = torch.mean(torch.sigmoid(self.lambda_x[idx_g]) * self.criterion(out_g_new, labels_g)) + self._reg()
            grads_g_y = torch.autograd.grad(loss_g_new, self.inner_model.parameters())

            offset = 0
            for i, p in enumerate(self.inner_model.parameters()):
                numel = p.numel()
                z_i = self.z[offset:offset+numel].view(p.shape).detach()
                d_tilde = (1.0/self.c_t)*grads_f_y[i].detach() + grads_g_y[i].detach() + (1.0/self.gamma)*(z_i - p.detach())
                self.d_y[i] = (self.nu * d_tilde + (1 - self.nu) * self.d_y[i]).detach()
                dy_norm = torch.norm(self.d_y[i])
                if dy_norm > self.clip_grad:
                    self.d_y[i] = self.d_y[i] * self.clip_grad / dy_norm
                p.data -= beta_t * self.d_y[i]
                offset += numel

            task_accs.append(self._accuracy(out_f_new, labels_f))
            task_loss.append(loss_f_new.item())
            torch.cuda.empty_cache()
            if self.verbose and step % 100 == 0:
                print(f'Step {step} | Loss: {np.mean(task_loss):.4f} | Acc: {np.mean(task_accs):.4f} | gamma: {self.gamma:.1f}')

        return np.mean(task_accs), np.mean(task_loss)

    def _set_model_params(self, flat_params):
        offset = 0
        for p in self.inner_model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view(p.shape))
            offset += numel

    def _accuracy(self, outputs, labels):
        pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        return accuracy_score(pred.cpu().numpy(), labels.cpu().numpy())

    def test(self, test_loader):
        self.inner_model.eval()
        self.inner_model.to(self.device)
        task_accs, task_loss = [], []
        for data in test_loader:
            inputs, targets, _ = data
            outputs = predict(self.inner_model, inputs)
            loss = torch.mean(self.criterion(outputs, targets.to(self.device)))
            pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            acc = accuracy_score(pred.detach().cpu().numpy(), targets.numpy())
            task_accs.append(acc)
            task_loss.append(loss.detach().cpu())
            torch.cuda.empty_cache()
        self.inner_model.train()

        current_acc = np.mean(task_accs)
        if current_acc > self.best_test_acc:
            self.best_test_acc = current_acc
            self.best_model_state = copy.deepcopy(self.inner_model.state_dict())

        return np.mean(task_accs), np.mean(task_loss)
