import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 小型 CNN Encoder（最低要求，适合 CPU 训练）
# ============================================================
class SmallCNN(nn.Module):
    """轻量级 CNN，适合 CIFAR-10 32x32 输入 + CPU 快速实验

    结构：
        Conv(3->64) -> ReLU -> MaxPool
        Conv(64->128) -> ReLU -> MaxPool
        Conv(128->256) -> ReLU -> MaxPool
        Global AvgPool -> 256-dim 特征向量
    """

    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),      # 8 -> 1
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        feat = self.conv(x)                    # (B, 256, 1, 1)
        feat = feat.view(feat.size(0), -1)     # (B, 256)
        return feat


# ============================================================
# 2. ResNet-18 Encoder（推荐要求，适配 CIFAR-10 32x32）
# ============================================================

class BasicBlock(nn.Module):
    """ResNet 基本残差块：两个 3x3 卷积 + 跳跃连接"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """简化 ResNet-18，适配 CIFAR-10 32x32 输入

    相比标准 ImageNet ResNet-18 的修改：
    - 第 1 层: 3x3 conv (stride=1) 代替 7x7 conv (stride=2)  → 保留 32x32 分辨率
    - 去掉开头的 MaxPool → 防止过早丢失空间信息
    - 其余层与标准 ResNet-18 一致
    """

    def __init__(self, in_channels=3, feat_dim=512):
        super().__init__()
        self.in_planes = 64
        self.feat_dim = feat_dim

        # CIFAR-10 适配的第 1 层 (3x3, stride=1, 不接 MaxPool)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 4 个 layer，channel 逐层翻倍
        self.layer1 = self._make_layer(64, 2, stride=1)    # 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)   # 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)   # 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)   # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes, num_blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)                  # (B, 512, 1, 1)
        out = out.view(out.size(0), -1)          # (B, 512)
        return out


# ============================================================
# 3. Projection Head（MLP: Linear -> ReLU -> Linear）
# ============================================================

class ProjectionHead(nn.Module):
    """SimCLR 投影头：将 encoder 输出映射到对比学习空间

    g(h) = W₂ · ReLU(W₁ · h)

    论文发现 2 层 MLP 效果最优：
    - 恒等映射（不用 projection）→ 效果差
    - 线性投影 → 一般
    - 2 层 MLP → 最佳
    - 3 层 MLP → 没有进一步提升
    """

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. 完整 SimCLR 模型 = Encoder + ProjectionHead
# ============================================================

class SimCLR(nn.Module):
    """SimCLR 模型：encoder + projection head

    使用方法：
        model = SimCLR(encoder_name='resnet18')

        # 预训练：encoder + projection head 一起用
        z1 = model(x1)   # projection 后的向量，用于对比 loss
        z2 = model(x2)

        # Linear probe：冻结 encoder，只用 .encode()
        for p in model.encoder.parameters():
            p.requires_grad = False
        features = model.encode(images)  # encoder 原始输出
    """

    def __init__(self, encoder_name='resnet18', proj_hidden_dim=512, proj_out_dim=128):
        super().__init__()

        # 选择 encoder
        if encoder_name == 'cnn':
            self.encoder = SmallCNN(in_channels=3, feat_dim=256)
            enc_dim = 256
        elif encoder_name == 'resnet18':
            self.encoder = ResNet18(in_channels=3, feat_dim=512)
            enc_dim = 512
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}. Use 'cnn' or 'resnet18'.")

        self.projection_head = ProjectionHead(
            in_dim=enc_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
        )

    def forward(self, x):
        """返回 L2 归一化后的投影向量（用于对比 loss）"""
        h = self.encoder(x)                 # encoder 特征
        z = self.projection_head(h)         # 投影
        z = F.normalize(z, dim=1)           # L2 归一化
        return z

    def encode(self, x):
        """返回 encoder 原始特征向量（用于 linear probe）"""
        return self.encoder(x)


# ============================================================
# 5. 快速测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("测试 SmallCNN")
    cnn = SmallCNN(in_channels=3, feat_dim=256)
    x = torch.randn(4, 3, 32, 32)
    out = cnn(x)
    print(f"  输入: {x.shape}  ->  输出: {out.shape} (期望 [4, 256])")

    print("\n" + "=" * 60)
    print("测试 ResNet-18 (CIFAR-10 适配)")
    resnet = ResNet18(in_channels=3, feat_dim=512)
    out = resnet(x)
    print(f"  输入: {x.shape}  ->  输出: {out.shape} (期望 [4, 512])")

    print("\n" + "=" * 60)
    print("测试 ProjectionHead")
    proj = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128)
    h = torch.randn(4, 512)
    z = proj(h)
    print(f"  输入: {h.shape}  ->  输出: {z.shape} (期望 [4, 128])")

    print("\n" + "=" * 60)
    print("测试完整 SimCLR (resnet18)")
    simclr = SimCLR(encoder_name='resnet18')
    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)
    z1 = simclr(x1)
    z2 = simclr(x2)
    feat = simclr.encode(x1)
    print(f"  z1: {z1.shape}, z2: {z2.shape}, feat: {feat.shape}")
    print(f"  L2 norm of z1: {z1.norm(dim=1)[:4]} (应全为 1.0)")

    print("\n" + "=" * 60)
    print("测试完整 SimCLR (cnn)")
    simclr_cnn = SimCLR(encoder_name='cnn')
    z1 = simclr_cnn(x1)
    feat = simclr_cnn.encode(x1)
    print(f"  z1: {z1.shape}, feat: {feat.shape}")

    # 统计参数量
    print("\n" + "=" * 60)
    print("参数量统计:")
    for name, model in [('SmallCNN', cnn), ('ResNet18', resnet)]:
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} 参数")

    proj_params = sum(p.numel() for p in proj.parameters())
    print(f"  ProjectionHead: {proj_params:,} 参数")
    print(f"  SimCLR (resnet18): {sum(p.numel() for p in simclr.parameters()):,} 参数")
    print(f"  SimCLR (cnn): {sum(p.numel() for p in simclr_cnn.parameters()):,} 参数")
