import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from .RNN_net import NLIRNN
import copy


def predict(net, inputs):
    """ Get predictions for a single batch. """
    (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    outputs = net((s1_embed.cuda(), s1_lens), (s2_embed.cuda(), s2_lens))
    return outputs


class Learner(nn.Module):
    def __init__(self, args, training_size):
        super(Learner, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_size = training_size

        # 1. 初始化内部模型 y
        self.inner_model = NLIRNN(
            word_embed_dim=args.word_embed_dim,
            encoder_dim=args.encoder_dim,
            n_enc_layers=args.n_enc_layers,
            dpout_model=0.0, dpout_fc=0.0,
            fc_dim=args.fc_dim, n_classes=args.n_classes,
            pool_type=args.pool_type, linear_fc=args.linear_fc
        ).to(self.device)

        # 2. 初始化超参数 x (lambda_x)
        self.lambda_x = torch.ones(self.training_size, requires_grad=True, device=self.device)

        # 3. 初始化辅助变量 z 及其动量 d_z
        param_count = sum(p.numel() for p in self.inner_model.parameters())
        self.z = torch.randn(param_count, 1, device=self.device)
        nn.init.xavier_uniform_(self.z)
        self.d_z = torch.zeros_like(self.z)  # z 的动量项

        # 4. 初始化 x 和 y 的动量缓存
        self.d_x = torch.zeros_like(self.lambda_x)
        self.d_y = [torch.zeros_like(p) for p in self.inner_model.parameters()]

        # 算法超参数配置
        self.gamma = args.gamma  # 近端参数
        self.rho = args.beta  # x 和 z 的动量系数
        self.nu_y = args.beta  # y 的动量系数
        self.c_t = 1.0  # 惩罚项系数 (通常设为1)
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def _get_z_slice(self, index):
        """ 辅助函数：获取扁平化 z 中对应第 i 个模型参数的切片 """
        offset = 0
        for i, p in enumerate(self.inner_model.parameters()):
            numel = p.numel()
            if i == index:
                return self.z[offset: offset + numel].view(p.shape)
            offset += numel
        return None

    def forward(self, train_loader, val_loader=None, training=True, epoch=0):
        task_accs, task_loss = [], []

        # 步长设置
        eta_t = self.args.nu  # z 的学习率
        alpha_t = self.args.outer_update_lr  # x 的学习率
        beta_t = self.args.inner_update_lr  # y 的学习率

        # 获取数据迭代器以支持采样三个独立的 Batch (S_f, S_g, S_gh)
        train_iter = iter(train_loader)
        val_iter = iter(val_loader) if val_loader is not None else iter(train_loader)

        # 显存优化：限制每轮迭代的步数，防止计算图堆积
        max_steps = len(train_loader) // 2
        for step in range(max_steps):
            try:
                data_f = next(train_iter)  # 用于计算外层损失 f
                data_g = next(val_iter)  # 用于计算内层损失 g (S_g)
                data_gh = next(train_iter)  # 用于更新 z 的采样 (S_g_hat)
            except StopIteration:
                break

            # --- Step A: 更新辅助变量 z (带动量) ---
            self.inner_model.zero_grad()
            inputs_gh, labels_gh, idx_gh = data_gh
            labels_gh = labels_gh.to(self.device)
            out_gh = predict(self.inner_model, inputs_gh)
            # 计算 ∇2 g(x, z) 的近似
            loss_gh = torch.mean(
                torch.sigmoid(self.lambda_x[idx_gh]) * self.criterion(out_gh, labels_gh))
            grads_gh = torch.autograd.grad(loss_gh, self.inner_model.parameters(), retain_graph=False)
            grad_g_z_flat = torch.cat([g.detach().view(-1) for g in grads_gh]).unsqueeze(1)

            # 获取当前 y 的扁平化
            y_flat = torch.cat([p.detach().view(-1) for p in self.inner_model.parameters()]).unsqueeze(1)

            # z 的当前梯度: ∇2 g + (1/gamma)*(z - y)
            curr_grad_z = grad_g_z_flat + (1.0 / self.gamma) * (self.z - y_flat)
            # 动量更新 z
            self.d_z = self.rho * self.d_z + (1 - self.rho) * curr_grad_z
            self.z = self.z - eta_t * self.d_z

            # 及时显存回收
            del out_gh, loss_gh, grads_gh, grad_g_z_flat, y_flat

            # --- Step B: 更新超参数 x (归一化更新) ---
            inputs_f, labels_f, _ = data_f
            labels_f = labels_f.to(self.device)
            out_f = predict(self.inner_model, inputs_f)
            loss_f = torch.mean(self.criterion(out_f, labels_f))

            # 获取 x 的梯度 (∇1 f)
            grad_f_x = torch.autograd.grad(loss_f, self.lambda_x, retain_graph= True,allow_unused=True)
            if grad_f_x[0] is not None:
                grad_f_x = grad_f_x[0]
            else:
                grad_f_x=torch.zeros_like(self.lambda_x)
            d_tilde_x = (1.0 / self.c_t) * grad_f_x
            # x 的动量更新
            self.d_x = self.rho * self.d_x + (1 - self.rho) * d_tilde_x
            # 执行归一化更新
            self.lambda_x.data -= alpha_t * (self.d_x / (torch.norm(self.d_x) + 1e-8))

            # --- Step C: 更新模型参数 y (归一化更新) ---
            # 获取 y 的梯度 (∇2 f)
            grads_f_y = torch.autograd.grad(loss_f, self.inner_model.parameters())

            for i, p in enumerate(self.inner_model.parameters()):
                z_slice = self._get_z_slice(i)
                # d_tilde_y = (1/c)*∇2 f + (1/gamma)*(z - y)
                d_tilde_y_i = (1.0 / self.c_t) * grads_f_y[i].detach() + \
                              (1.0 / self.gamma) * (z_slice.detach() - p.detach())

                # y 的动量更新
                self.d_y[i] = self.nu_y * self.d_y[i] + (1 - self.nu_y) * d_tilde_y_i
                # y 的归一化步长更新
                p.data -= beta_t * (self.d_y[i] / (torch.norm(self.d_y[i]) + 1e-8))

            # 统计与记录
            task_accs.append(self.get_accuracy(out_f, labels_f))
            task_loss.append(loss_f.item())
            torch.cuda.empty_cache()
            if step % 10 == 0:
                print(f'Step {step} | Task Loss: {np.mean(task_loss):.4f} | Acc: {np.mean(task_accs):.4f}')

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
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')
        self.inner_model.train()
        return np.mean(task_accs), np.mean(task_loss)