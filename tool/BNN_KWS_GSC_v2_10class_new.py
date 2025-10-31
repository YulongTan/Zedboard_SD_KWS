from __future__ import absolute_import, division, print_function
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.autograd import Function

# =====================
# 0. 选择的10个类别（你想换就改这里）
# =====================
SELECTED_CLASSES = [
    "yes", "no", "go", "on", "wow",
    "happy", "follow", "off", "stop", "visual"
]

# =====================
# 1. 数据预处理
# =====================
DATASET_PATH = "./speech_commands"

TARGET_SR = 16000
N_MELS = 40
N_FFT = 400
HOP = 160
WIN = 400

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    win_length=WIN,
    hop_length=HOP,
    n_mels=N_MELS
)

def waveform_to_logmel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    # 到单声道 [1, T]
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 重采样到16k
    if sample_rate != TARGET_SR:
        resample = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
        waveform = resample(waveform)

    # 固定到1秒（16000点）
    TGT_LEN = TARGET_SR
    cur_len = waveform.shape[-1]
    if cur_len < TGT_LEN:
        pad_len = TGT_LEN - cur_len
        waveform = F.pad(waveform, (0, pad_len))
    elif cur_len > TGT_LEN:
        start = (cur_len - TGT_LEN) // 2
        waveform = waveform[:, start:start + TGT_LEN]

    # Mel & log  -> [1, 40, 98]
    mel = mel_transform(waveform).clamp(min=1e-6).log()
    return mel


def make_dataloaders(batch_size: int, data_root: str = None):
    dataset_path = data_root if data_root else DATASET_PATH

    train_set = torchaudio.datasets.SPEECHCOMMANDS(
        root=dataset_path, subset="training", download=True)
    val_set = torchaudio.datasets.SPEECHCOMMANDS(
        root=dataset_path, subset="validation", download=True)
    test_set = torchaudio.datasets.SPEECHCOMMANDS(
        root=dataset_path, subset="testing", download=True)

    # 我们只保留 SELECTED_CLASSES 里的样本
    selected = set(SELECTED_CLASSES)
    # 只给这10个类编号
    label2idx = {label: i for i, label in enumerate(SELECTED_CLASSES)}
    num_classes = len(SELECTED_CLASSES)

    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, base, l2i, selected):
            self.base = base
            self.l2i = l2i
            self.selected = selected

            # 过滤：只留下我们想要的10类
            self.indices = []
            for i in range(len(self.base)):
                lbl = self.base[i][2]
                if lbl in self.selected:
                    self.indices.append(i)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            waveform, sr, label, *_ = self.base[real_idx]
            x = waveform_to_logmel(waveform, sr)     # [1,40,98]
            y = self.l2i[label]                      # 0~9
            return x, y

    train_ds = _Wrap(train_set, label2idx, selected)
    val_ds   = _Wrap(val_set,   label2idx, selected)
    test_ds  = _Wrap(test_set,  label2idx, selected)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes


# =====================
# 2. 二值/三值函数
# =====================
class BinaryAct(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = torch.sign(x)
        out[out == 0] = -1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # tanh平滑梯度
        grad = (1 - torch.tanh(x * 0.1) ** 2) * grad_output
        return grad


class BinaryWeight(Function):
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        out = torch.sign(w)
        out[out == 0] = -1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        grad = (1 - torch.tanh(w * 0.1) ** 2) * grad_output
        return grad


# 激活量化模块
class ActBin(nn.Module):
    def __init__(self, A=2):
        super().__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.A == 2:
            return BinaryAct.apply(x)
        else:
            return self.relu(x)


# 权重量化模块
class WeightBin(nn.Module):
    def __init__(self, W=2):
        super().__init__()
        self.W = W

    def forward(self, w):
        if self.W == 2:
            return BinaryWeight.apply(w)
        else:
            return w


# =====================
# 3. 不用bias, 不用BN的量化卷积
# =====================
class QConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=2, padding=1, A=2, W=2):
        super().__init__()
        self.act_q = ActBin(A=A)
        self.w_q   = WeightBin(W=W)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k,
                              stride=stride, padding=padding,
                              bias=False)

    def forward(self, x):
        x = self.act_q(x)
        w = self.w_q(self.conv.weight)
        x = F.conv2d(x, w, bias=None,
                     stride=self.conv.stride,
                     padding=self.conv.padding,
                     dilation=self.conv.dilation,
                     groups=self.conv.groups)
        return x


# =====================
# 4. BNN KWS 网络
# =====================
class BNNKWS(nn.Module):
    def __init__(self, num_classes, A=2, W=2):
        super().__init__()
        # conv2 改为 stride=1, padding=1
        # conv3 改为 stride=1, padding=0
        self.conv1 = QConv2d(1,   32, k=3, stride=2, padding=1, A=A, W=W)   # [B,32,20,49]
        self.conv2 = QConv2d(32,  64, k=3, stride=1, padding=1, A=A, W=W)   # [B,64,20,49]
        self.conv3 = QConv2d(64,  64, k=3, stride=1, padding=0, A=A, W=W)   # [B,64,18,47]
        self.conv4 = QConv2d(64,  64, k=3, stride=2, padding=1, A=A, W=W)   # [B,64,9,24]

        self.act_out = ActBin(A=A)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(64, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x) + 1     # [B,32,20,49]
        x = self.conv2(x)         # [B,64,20,49]
        x = self.conv3(x)         # [B,64,18,47]
        x = self.conv4(x)         # [B,64,9,24]
        x = self.act_out(x)
        x = self.gap(x)           # [B,64,1,1]
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


# =====================
# 5. 训练 / 验证 / 测试
# =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


# =====================
# 6. main
# =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    train_loader, val_loader, test_loader, num_classes = make_dataloaders(batch_size, DATASET_PATH)

    # 这里的 num_classes 已经是 10 了
    model = BNNKWS(num_classes=num_classes, A=2, W=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    best_val_acc = 0.0
    best_state = None

    torch.nn.Module.load_state_dict(model, state_dict=torch.load("./bnn_kws_ckpt/bnn_kws_10cls_stride2_nobias_nobn.pt"), strict=True)

    EPOCHS = 60
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = eval_one_epoch(model, val_loader, criterion, device)

        print(f"[{epoch:02d}] train_loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # 测试
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"Test acc={test_acc*100:.2f}%")

    os.makedirs("./bnn_kws_ckpt", exist_ok=True)
    torch.save(model.state_dict(), "./bnn_kws_ckpt/bnn_kws_10cls_stride2_nobias_nobn.pt")
    print("model saved to ./bnn_kws_ckpt/bnn_kws_10cls_stride2_nobias_nobn.pt")

if __name__ == "__main__":
    main()
