import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from .RNN_net import NLIRNN

def predict(net, inputs):
    """ Get predictions for a single batch. """
    (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    outputs = net((s1_embed.cuda(), s1_lens), (s2_embed.cuda(), s2_lens))
    return outputs

class Learner(nn.Module):
    def __init__(self, args, training_size, verbose=True):
        super(Learner, self).__init__()
        self.args = args
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_size = training_size

        self.inner_model = NLIRNN(
            word_embed_dim=args.word_embed_dim,
            encoder_dim=args.encoder_dim,
            n_enc_layers=args.n_enc_layers,
            dpout_model=0.0, dpout_fc=0.0,
            fc_dim=args.fc_dim, n_classes=args.n_classes,
            pool_type=args.pool_type, linear_fc=args.linear_fc
        ).to(self.device)

        self.lambda_x = torch.ones(self.training_size, requires_grad=True, device=self.device)

        param_count = sum(p.numel() for p in self.inner_model.parameters())
        self.z = torch.zeros(param_count, 1, device=self.device)
        with torch.no_grad():
            self.z.copy_(torch.cat([p.view(-1, 1) for p in self.inner_model.parameters()]))

        self.d_z = torch.zeros_like(self.z)
        self.d_x = torch.zeros_like(self.lambda_x)
        self.d_y = [torch.zeros_like(p) for p in self.inner_model.parameters()]

        self.gamma = args.gamma
        self.rho = args.beta
        self.nu = args.beta
        self.c_t = 1.0
        self.clip_grad = 1.0
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def _set_model_params(self, flat_params):
        offset = 0
        for p in self.inner_model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset : offset + numel].view(p.shape))
            offset += numel

    def _reg(self):
        return 0.0001 * sum([x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()

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

        # Delayed polynomial LR decay (V5 best): keep full lr for first 5 epochs, then decay
        decay = max(0.05, (5.0 / (epoch + 5)) ** 0.5) if epoch >= 5 else 1.0
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

            # --- Update z (momentum) ---
            y_state = [p.data.clone() for p in self.inner_model.parameters()]
            self._set_model_params(self.z)

            out_gh = predict(self.inner_model, data_gh[0])
            loss_gh = torch.mean(torch.sigmoid(self.lambda_x[idx_gh]) * self.criterion(out_gh, labels_gh)) + self._reg()
            grads_gh = torch.autograd.grad(loss_gh, self.inner_model.parameters())
            grad_g_z_flat = torch.cat([g.detach().view(-1, 1) for g in grads_gh])

            y_flat = torch.cat([p.view(-1, 1) for p in y_state])
            curr_grad_z = grad_g_z_flat + (1.0 / self.gamma) * (self.z.detach() - y_flat)
            self.d_z = (self.rho * self.d_z + (1 - self.rho) * curr_grad_z).detach()
            dz_norm = torch.norm(self.d_z)
            if dz_norm > self.clip_grad:
                self.d_z = self.d_z * self.clip_grad / dz_norm
            self.z.data -= eta_t * self.d_z

            for p, val in zip(self.inner_model.parameters(), y_state): p.data.copy_(val)

            # --- Update x (normalized) ---
            out_f = predict(self.inner_model, data_f[0])
            loss_f = torch.mean(self.criterion(out_f, labels_f)) + self._reg()
            grad_f_x_tuple = torch.autograd.grad(loss_f, self.lambda_x, allow_unused=True)
            grad_f_x = grad_f_x_tuple[0] if grad_f_x_tuple[0] is not None else torch.zeros_like(self.lambda_x)

            out_g = predict(self.inner_model, data_g[0])
            loss_g = torch.mean(torch.sigmoid(self.lambda_x[idx_g]) * self.criterion(out_g, labels_g)) + self._reg()
            grad_g_x_y = torch.autograd.grad(loss_g, self.lambda_x)[0]

            self._set_model_params(self.z)
            out_g_z = predict(self.inner_model, data_g[0])
            loss_g_z = torch.mean(torch.sigmoid(self.lambda_x[idx_g]) * self.criterion(out_g_z, labels_g)) + self._reg()
            grad_g_x_z = torch.autograd.grad(loss_g_z, self.lambda_x)[0]

            for p, val in zip(self.inner_model.parameters(), y_state): p.data.copy_(val)

            d_tilde_x = (1.0 / self.c_t) * grad_f_x.detach() + grad_g_x_y.detach() - grad_g_x_z.detach()
            self.d_x = (self.rho * self.d_x + (1 - self.rho) * d_tilde_x).detach()
            self.lambda_x.data -= alpha_t * (self.d_x / (torch.norm(self.d_x) + 1e-8))

            # --- Update y (momentum + clip) ---
            out_f_new = predict(self.inner_model, data_f[0])
            loss_f_new = torch.mean(self.criterion(out_f_new, labels_f)) + self._reg()
            grads_f_y = torch.autograd.grad(loss_f_new, self.inner_model.parameters())

            out_g_new = predict(self.inner_model, data_g[0])
            loss_g_new = torch.mean(torch.sigmoid(self.lambda_x[idx_g]) * self.criterion(out_g_new, labels_g)) + self._reg()
            grads_g_y = torch.autograd.grad(loss_g_new, self.inner_model.parameters())

            offset = 0
            for i, p in enumerate(self.inner_model.parameters()):
                numel = p.numel()
                z_i = self.z[offset : offset + numel].view(p.shape).detach()
                d_tilde_y_i = (1.0 / self.c_t) * grads_f_y[i].detach() + grads_g_y[i].detach() + (1.0 / self.gamma) * (z_i - p.detach())
                self.d_y[i] = (self.nu * d_tilde_y_i + (1 - self.nu) * self.d_y[i]).detach()
                dy_norm = torch.norm(self.d_y[i])
                if dy_norm > self.clip_grad:
                    self.d_y[i] = self.d_y[i] * self.clip_grad / dy_norm
                p.data -= beta_t * self.d_y[i]
                offset += numel

            task_accs.append(self.get_accuracy(out_f_new, labels_f))
            task_loss.append(loss_f_new.item())
            torch.cuda.empty_cache()
            if self.verbose and step % 100 == 0:
                print(f'Step {step} | Task Loss: {np.mean(task_loss):.4f} | Acc: {np.mean(task_accs):.4f}')

        # === Diagnostic: lambda_x statistics ===
        with torch.no_grad():
            lx = self.lambda_x.detach()
            sx = torch.sigmoid(lx)
            lx_stats = {
                'lx_mean': lx.mean().item(),
                'lx_std': lx.std().item(),
                'lx_min': lx.min().item(),
                'lx_max': lx.max().item(),
                'sig_mean': sx.mean().item(),
                'sig_std': sx.std().item(),
                'sig_p10': torch.quantile(sx, 0.1).item(),
                'sig_p50': torch.quantile(sx, 0.5).item(),
                'sig_p90': torch.quantile(sx, 0.9).item(),
            }
            if self.verbose:
                print(f"  [lambda_x] lx: mean={lx_stats['lx_mean']:.4f} std={lx_stats['lx_std']:.4f} "
                      f"range=[{lx_stats['lx_min']:.4f}, {lx_stats['lx_max']:.4f}]")
                print(f"  [sigmoid]  mean={lx_stats['sig_mean']:.4f} std={lx_stats['sig_std']:.4f} "
                      f"p10={lx_stats['sig_p10']:.4f} p50={lx_stats['sig_p50']:.4f} p90={lx_stats['sig_p90']:.4f}")

        return np.mean(task_accs), np.mean(task_loss)

    def get_accuracy(self, outputs, labels):
        pre_label = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        return accuracy_score(pre_label.cpu().numpy(), labels.cpu().numpy())

    def test(self, test_loader):
        self.inner_model.eval()
        self.inner_model.to(self.device)
        task_accs, task_loss = [], []
        for data in test_loader:
            inputs, targets, _ = data
            outputs = predict(self.inner_model, inputs)
            loss = torch.mean(self.criterion(outputs, targets.to(self.device)))
            q_logits = F.softmax(outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1).detach().cpu().numpy().tolist()
            q_label_id = targets.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(loss.detach().cpu())
            torch.cuda.empty_cache()
        self.inner_model.train()
        return np.mean(task_accs), np.mean(task_loss)
