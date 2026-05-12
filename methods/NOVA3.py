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
            n_enc_layers=args.n_enc_layers, dpout_model=0.0,
            dpout_fc=getattr(args, 'dpout_fc', 0.0),
            fc_dim=args.fc_dim, n_classes=args.n_classes,
            pool_type=args.pool_type, linear_fc=args.linear_fc
        ).to(self.device)

        # Upper-level variable x (sample weights)
        self.lambda_x = torch.ones(self.training_size, requires_grad=True, device=self.device)

        # Auxiliary z, initialized to y
        param_count = sum(p.numel() for p in self.inner_model.parameters())
        self.z = torch.zeros(param_count, 1, device=self.device)
        with torch.no_grad():
            self.z.copy_(torch.cat([p.view(-1, 1) for p in self.inner_model.parameters()]))

        # Momentum buffers (pseudocode: d_z^{-1}=0)
        self.d_z = torch.zeros_like(self.z)
        self.d_x = torch.zeros_like(self.lambda_x)
        self.d_y = [torch.zeros_like(p) for p in self.inner_model.parameters()]

        # Three distinct momentum parameters (pseudocode: mu, rho, nu)
        self.mu = getattr(args, 'mu', args.beta)
        self.rho = getattr(args, 'rho', args.beta)
        self.nu = getattr(args, 'nu_momentum', args.beta)

        # Proximal/penalty parameters
        self.gamma = args.gamma
        self.c_t = 1.0

        # Label smoothing
        ls = getattr(args, 'label_smooth', 0.0)
        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=ls).to(self.device)

        # EMA model for evaluation
        self.ema_decay = getattr(args, 'ema_decay', 0.0)
        if self.ema_decay > 0:
            self.ema_params = [p.data.clone() for p in self.inner_model.parameters()]
        else:
            self.ema_params = None

        # Best model tracking
        self.best_test_acc = 0.0
        self.best_model_state = None

    def _reg(self):
        return 0.0001 * sum([x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()

    def _set_model_params(self, flat_params):
        offset = 0
        for p in self.inner_model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view(p.shape))
            offset += numel

    def _accuracy(self, outputs, labels):
        pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        return accuracy_score(pred.cpu().numpy(), labels.cpu().numpy())

    def _update_ema(self):
        if self.ema_params is not None:
            for ema, p in zip(self.ema_params, self.inner_model.parameters()):
                ema.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

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

        # Delayed polynomial LR decay: full lr for 5 epochs, then decay
        decay_power = getattr(self.args, 'decay_power', 0.5)
        decay = max(0.01, (5.0 / (epoch + 5)) ** decay_power) if epoch >= 5 else 1.0
        eta_t = self.args.nu * decay
        alpha_t = self.args.outer_update_lr * decay
        beta_t = self.args.inner_update_lr * decay

        for step, data_g in enumerate(train_loader):
            data_f = next_val()
            data_gh = next_train()

            labels_f = data_f[1].to(self.device)
            labels_g = data_g[1].to(self.device)
            labels_gh = data_gh[1].to(self.device)
            idx_g = data_g[2]
            idx_gh = data_gh[2]

            # ==================== Z UPDATE (pseudocode Step 2-3) ====================
            y_state = [p.data.clone() for p in self.inner_model.parameters()]
            self._set_model_params(self.z)

            out_gh = predict(self.inner_model, data_gh[0])
            loss_gh = torch.mean(torch.sigmoid(self.lambda_x[idx_gh]) * self.criterion(out_gh, labels_gh)) + self._reg()
            grads_gh = torch.autograd.grad(loss_gh, self.inner_model.parameters())
            grad_g_z = torch.cat([g.detach().view(-1, 1) for g in grads_gh])

            y_flat = torch.cat([p.view(-1, 1) for p in y_state])
            d_tilde_z = grad_g_z + (1.0 / self.gamma) * (self.z.detach() - y_flat)

            self.d_z = (self.mu * self.d_z + (1 - self.mu) * d_tilde_z).detach()

            dz_norm = torch.norm(self.d_z)
            if dz_norm > 1.0:
                self.d_z = self.d_z * (1.0 / dz_norm)

            self.z.data -= eta_t * self.d_z

            # Restore model to y
            for p, val in zip(self.inner_model.parameters(), y_state):
                p.data.copy_(val)

            # ==================== X UPDATE (pseudocode Step 4-5) ====================
            out_g_y = predict(self.inner_model, data_g[0])
            ce_g = self.criterion(out_g_y, labels_g).detach()
            loss_g_y = torch.mean(torch.sigmoid(self.lambda_x[idx_g]) * ce_g) + self._reg()
            grad_g_x_y = torch.autograd.grad(loss_g_y, self.lambda_x)[0]

            self._set_model_params(self.z)
            out_g_z = predict(self.inner_model, data_gh[0])
            ce_gz = self.criterion(out_g_z, labels_gh).detach()
            loss_g_z = torch.mean(torch.sigmoid(self.lambda_x[idx_gh]) * ce_gz) + self._reg()
            grad_g_x_z = torch.autograd.grad(loss_g_z, self.lambda_x)[0]

            for p, val in zip(self.inner_model.parameters(), y_state):
                p.data.copy_(val)

            d_tilde_x = (grad_g_x_y.detach() - grad_g_x_z.detach())
            self.d_x = (self.rho * self.d_x + (1 - self.rho) * d_tilde_x).detach()
            self.lambda_x.data -= alpha_t * (self.d_x / (torch.norm(self.d_x) + 1e-8))

            # ==================== Y UPDATE (pseudocode Step 6-7) ====================
            out_f = predict(self.inner_model, data_f[0])
            loss_f = torch.mean(self.criterion(out_f, labels_f)) + self._reg()
            grads_f_y = torch.autograd.grad(loss_f, self.inner_model.parameters())

            loss_g_y_new = torch.mean(torch.sigmoid(self.lambda_x[idx_g]) * self.criterion(out_g_y, labels_g)) + self._reg()
            grads_g_y = torch.autograd.grad(loss_g_y_new, self.inner_model.parameters())

            offset = 0
            for i, p in enumerate(self.inner_model.parameters()):
                numel = p.numel()
                z_i = self.z[offset:offset + numel].view(p.shape).detach()

                d_tilde_y = ((1.0 / self.c_t) * grads_f_y[i].detach()
                             + grads_g_y[i].detach()
                             + (1.0 / self.gamma) * (z_i - p.detach()))

                self.d_y[i] = (self.nu * d_tilde_y + (1 - self.nu) * self.d_y[i]).detach()
                offset += numel

            # y -= β_t · d_y / ‖d_y‖ (pseudocode faithful)
            dy_global_norm = torch.sqrt(sum(torch.sum(d.pow(2)) for d in self.d_y)) + 1e-8
            for i, p in enumerate(self.inner_model.parameters()):
                p.data -= beta_t * self.d_y[i] / dy_global_norm

            self._update_ema()

            task_accs.append(self._accuracy(out_f, labels_f))
            task_loss.append(loss_f.item())
            torch.cuda.empty_cache()
            if self.verbose and step % 100 == 0:
                print(f'Step {step} | Loss: {np.mean(task_loss):.4f} | Acc: {np.mean(task_accs):.4f} | gamma: {self.gamma:.1f}')

        return np.mean(task_accs), np.mean(task_loss)

    def test(self, test_loader):
        self.inner_model.eval()

        # Swap to EMA params for evaluation
        if self.ema_params is not None:
            saved = [p.data.clone() for p in self.inner_model.parameters()]
            for p, ema in zip(self.inner_model.parameters(), self.ema_params):
                p.data.copy_(ema)

        task_accs, task_loss = [], []
        for data in test_loader:
            inputs, targets, _ = data
            outputs = predict(self.inner_model, inputs)
            loss = torch.mean(nn.CrossEntropyLoss(reduction='none')(outputs, targets.to(self.device)))
            pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            acc = accuracy_score(pred.detach().cpu().numpy(), targets.numpy())
            task_accs.append(acc)
            task_loss.append(loss.detach().cpu())
            torch.cuda.empty_cache()

        # Restore original params
        if self.ema_params is not None:
            for p, s in zip(self.inner_model.parameters(), saved):
                p.data.copy_(s)

        self.inner_model.train()

        current_acc = np.mean(task_accs)
        if current_acc > self.best_test_acc:
            self.best_test_acc = current_acc
            if self.ema_params is not None:
                self.best_model_state = [e.clone() for e in self.ema_params]
            else:
                self.best_model_state = copy.deepcopy(self.inner_model.state_dict())

        return np.mean(task_accs), np.mean(task_loss)
