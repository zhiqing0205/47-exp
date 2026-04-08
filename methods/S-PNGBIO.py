import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from sklearn.metrics import accuracy_score


class Learner(nn.Module):
    def __init__(self, args, training_size):
        super(Learner, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_size = training_size

        # 1. 模型定义 (RNN 结构)
        from RNN_net import RNN
        self.inner_model = RNN(
            word_embed_dim=args.word_embed_dim,
            encoder_dim=args.encoder_dim,
            n_enc_layers=args.n_enc_layers,
            dpout_model=0.0,
            dpout_fc=0.0,
            fc_dim=args.fc_dim,
            n_classes=args.n_classes,
            pool_type=args.pool_type,
            linear_fc=args.linear_fc
        ).to(self.device)

        # 2. 上层变量 x (样本权重)
        self.lambda_x = torch.ones((self.training_size)).to(self.device)
        self.lambda_x.requires_grad = True

        # 3. 辅助变量 z (展平处理)
        param_count = sum(p.numel() for p in self.inner_model.parameters())
        self.z_params = torch.randn(param_count, 1).to(self.device)
        nn.init.xavier_uniform_(self.z_params)


        # 5. 超参数 (含罚函数系数 gamma 和差分系数 lambda_val)
        self.gamma = getattr(args, 'gamma', 0.1)
        self.lambda_val = getattr(args, 'lambda_val', 0.1)  # 差分梯度缩放系数

        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def get_loss(self, model, inputs, targets, data_indx=None):
        """计算基础分类损失 + 权重处理"""
        outputs = predict(model, inputs)
        if data_indx is not None:
            raw_loss = self.criterion(outputs, targets)
            weights=torch.sigmoid(self.lambda_x[data_indx])
            if weights.dim()!=raw_loss.dim():
                weights=weights.unsqueeze(1)
            loss = torch.mean(weights * raw_loss)
        else:
            loss = torch.mean(self.criterion(outputs, targets))
            # 样本权重 sigmoid 映射


        # L2 正则项
        reg = 0.0001 * sum([p.norm().pow(2) for p in model.parameters()]).sqrt()
        return loss + reg, outputs

    def __call__(self, train_loader, training=True, task_id=0):
        self.inner_model.train()
        task_accs, task_loss = [], []

        for step, data in enumerate(train_loader):
            inputs, targets, data_indx = data
            inputs = (inputs[0].to(self.device), inputs[1])
            targets = targets.to(self.device)

            # --- 第一阶段：梯度收集 ---
            # 1. 计算在 y 处的梯度 (f_y, f_x, g_y_y, g_y_x)
            loss_f, q_outputs = self.get_loss(self.inner_model, inputs, targets)
            grad_f_x_tuple = torch.autograd.grad(loss_f, self.lambda_x, retain_graph=True,allow_unused=True)
            grad_f_x=grad_f_x_tuple[0] if grad_f_x_tuple[0] is not None else torch.zeros_like(self.lambda_x)
            grad_f_y = torch.autograd.grad(loss_f, self.inner_model.parameters(), retain_graph=True)

            loss_g_y, _ = self.get_loss(self.inner_model, inputs, targets, data_indx)

            grad_g_y_x = torch.autograd.grad(loss_g_y, self.lambda_x, retain_graph=True)[0]
            grad_g_y_y = torch.autograd.grad(loss_g_y, self.inner_model.parameters(), retain_graph=True)

            # 2. 计算在 z 处的梯度 (g_z_z, g_z_x)
            original_y = [p.data.clone() for p in self.inner_model.parameters()]
            self._flatten_to_model(self.z_params)  # 切换模型到辅助变量 z

            loss_g_z, _ = self.get_loss(self.inner_model, inputs, targets, data_indx)
            grad_g_z_x = torch.autograd.grad(loss_g_z, self.lambda_x, retain_graph=True)[0]
            grad_g_z_z = torch.autograd.grad(loss_g_z, self.inner_model.parameters())
            grad_g_z_z_flat = torch.cat([g.view(-1) for g in grad_g_z_z]).unsqueeze(1)

            self._restore_model(original_y)  # 还原模型到 y

            # --- 第二阶段：动量更新与罚函数应用 ---
            with torch.no_grad():
                y_flat = torch.cat([p.data.view(-1) for p in self.inner_model.parameters()]).unsqueeze(1)

                # [Update z]: 罚函数导数 (1/gamma)*(z-y)
                # v_z = grad_g(z) + grad_penalty(z)
                v_z = grad_g_z_z_flat + (1.0 / self.gamma) * (self.z_params - y_flat)
                self.z_params -= self.args.inner_update_lr * v_z

                # [Update x]: 包含系数 lambda_val 的差分项
                # v_x = grad_f(x,y) + lambda * (grad_g(x,y) - grad_g(x,z))
                diff_x = grad_g_y_x - grad_g_z_x
                v_x = grad_f_x + self.lambda_val * diff_x


                d_x_hat=self.d_x / (torch.norm(v_x) + 1e-8)  # 归一化处理
                self.lambda_x.data -= self.args.outer_update_lr * d_x_hat
                self.lambda_x.data=torch.clamp(self.lambda_x.data, min=0.0)  # 限制在 [0, 1] 范围内

                # [Update y]: 罚函数关于 y 的导数为 (1/gamma)*(y-z)
                # v_y = grad_f(y) + grad_g(y) + grad_penalty(y)
                for i, p in enumerate(self.inner_model.parameters()):
                    z_p = self._get_z_part(i)
                    v_y = grad_f_y[i] + grad_g_y_y[i] + (1.0 / self.gamma) * (p.data - z_p)
                    d_y_hat = self.d_y[i] / (torch.norm(v_y) + 1e-8)
                    p.data -= self.args.inner_update_lr * d_y_hat

            # --- 统计与显存管理 ---
            pre_label = torch.argmax(q_outputs, dim=1).cpu().numpy()
            acc = accuracy_score(targets.cpu().numpy(), pre_label)
            task_accs.append(acc)
            task_loss.append(loss_f.item())

            if step % 10 == 0:
                print(f'Step {step} | Task Loss: {np.mean(task_loss):.4f} | Acc: {np.mean(task_accs):.4f}')
            torch.cuda.empty_cache()

        return np.mean(task_accs), np.mean(task_loss)

    def test(self, test_loader):
        self.inner_model.eval()
        task_accs, task_loss = [], []
        with torch.no_grad():
            for data in test_loader:
                inputs, targets, _ = data
                if isinstance(inputs[0], torch.Tensor):
                    inputs = (inputs[0].to(self.device), inputs[1])
                outputs = self.inner_model(inputs)
                loss = torch.mean(self.criterion(outputs, targets.to(self.device)))
                acc = accuracy_score(targets.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
                task_accs.append(acc);
                task_loss.append(loss.item())
        return np.mean(task_accs), np.mean(task_loss)

    def _flatten_to_model(self, flat_params):
        pointer = 0
        for p in self.inner_model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
            pointer += numel

    def _restore_model(self, param_list):
        for p, orig_p in zip(self.inner_model.parameters(), param_list):
            p.data.copy_(orig_p)

    def _get_z_part(self, index):
        pointer = 0
        for i, p in enumerate(self.inner_model.parameters()):
            numel = p.numel()
            if i == index: return self.z_params[pointer:pointer + numel].view_as(p)
            pointer += numel


def predict(net, inputs):
    return net(inputs)