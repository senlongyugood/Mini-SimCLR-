import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss

    这是 SimCLR 论文的核心损失函数。

    工作原理（以 batch_size=3 为例）:
        - 输入 3 张图片，每张做 2 种增广 → 共 6 个视图
        - 把 6 个视图拼接成 (6, 128) 的矩阵送入
        - 正样本对: (0,1) (2,3) (4,5)，即相邻两个是一对
        - 6 个视图中，每对正样本之外的所有视图都是负样本

    Args:
        temperature: 温度参数 τ，默认 0.5
                     τ 越小 → 分布越尖锐 → 模型更关注困难负样本
                     τ 越大 → 分布越平滑
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        """
        Args:
            z: L2 归一化后的投影向量，shape (2*N, D)
               其中 N 是 batch_size，D 是投影维度（通常 128）
               假设 z[0] 和 z[1] 是同一张图的两个增广，以此类推

        Returns:
            loss: 标量
        """
        # ---- 第 1 步：计算相似度矩阵 ----
        # z 已经 L2 归一化，所以点积就等于余弦相似度
        # sim[i][j] = z[i] · z[j]
        sim = z @ z.T                    # (2N, 2N)

        # ---- 第 2 步：除以温度 ----
        sim = sim / self.temperature

        # ---- 第 3 步：构造正样本对的标签 ----
        # batch 中: 视图 0 的正样本是 1，1 的正样本是 0
        #          视图 2 的正样本是 3，3 的正样本是 2，以此类推
        N = z.shape[0]                   # 2N
        batch_size = N // 2

        # 正样本索引：每个样本的配对样本位置
        # [1, 0, 3, 2, 5, 4, ...]
        pos_indices = torch.arange(N, device=z.device)
        pos_indices = pos_indices ^ 1    # 异或 1 翻转最低位 → 0↔1, 2↔3, 4↔5

        # ---- 第 4 步：抽取正样本相似度 ----
        # sim[i, pos_indices[i]] 就是每对正样本的相似度
        pos_sim = sim[torch.arange(N, device=z.device), pos_indices]  # (2N,)

        # ---- 第 5 步：计算 InfoNCE 的分子（正样本）和分母（所有样本） ----
        # 对每个 i，从 sim[i] 中排除自己（对角线），计算 log-sum-exp
        # 因为要排除自己，所以把对角线设为 -inf
        mask = torch.eye(N, device=z.device, dtype=torch.bool)
        sim_no_self = sim.masked_fill(mask, float('-inf'))

        # log-sum-exp：log(Σ exp(sim[i][k]))
        log_sum_exp = torch.logsumexp(sim_no_self, dim=1)   # (2N,)

        # ---- 第 6 步：计算每个视图的 loss ----
        # l(i) = -pos_sim[i] + log_sum_exp[i]
        loss_per_sample = -pos_sim + log_sum_exp             # (2N,)

        # ---- 第 7 步：取平均 ----
        loss = loss_per_sample.mean()

        return loss


def nt_xent_loss(z, temperature=0.5):
    """函数式接口，方便直接调用"""
    return NTXentLoss(temperature)(z)


# ============================================================
# 测试 & 教学用 demo
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("NT-Xent Loss 详细演示")
    print("=" * 60)

    # 模拟：batch_size=3，每张图 2 个增广 → 6 个视图
    # 假设 encoder 输出维度为 128
    N = 3
    D = 128
    z = F.normalize(torch.randn(2 * N, D), dim=1)

    print(f"\n1. 输入 shape: {z.shape}")
    print(f"   含义: {N} 张图 × 2 个增广 = {2*N} 个视图")
    print(f"   正样本对: (0,1) (2,3) (4,5)")

    # 手动一步步计算，帮助理解
    print(f"\n2. 相似度矩阵 (2N × 2N):")
    sim = z @ z.T
    print(f"   sim[i][j] = cos(z[i], z[j])")
    print(f"   sim[0][1] = {sim[0][1]:.4f}  ← 正样本对 #1")
    print(f"   sim[2][3] = {sim[2][3]:.4f}  ← 正样本对 #2")
    print(f"   sim[4][5] = {sim[4][5]:.4f}  ← 正样本对 #3")
    print(f"   对角线 sim[i][i] = 1.0（自己和自己的余弦距离）")

    print(f"\n3. 除以温度 τ=0.5 后计算 softmax 分布:")
    tau = 0.5
    sim_tau = sim / tau
    # 以视图 0 为例展示
    probs = (sim_tau[0] - sim_tau[0].max()).softmax(dim=0)
    print(f"   视图 0 对所有人的注意力权重:")
    print(f"   P(0→1, 正样本) = {probs[1]:.4f} ← 希望这个越大越好")
    print(f"   P(0→2) = {probs[2]:.4f}")
    print(f"   P(0→3) = {probs[3]:.4f}")
    print(f"   P(0→4) = {probs[4]:.4f}")
    print(f"   P(0→5) = {probs[5]:.4f}")

    print(f"\n4. 每个视图的 loss = -log(P(正样本))")
    print(f"   视图 0 的 loss = -log({probs[1]:.4f}) = {-torch.log(probs[1]):.4f}")
    print(f"   视图 1 的 loss = -log(P(1→0)) = ...")
    print(f"   总 loss = 所有 2N 个视图的平均")

    # 完整计算
    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(z)

    print(f"\n5. 完整 NT-Xent Loss = {loss.item():.4f}")

    # 验证：当所有向量相同时 loss 应该很小（好的情况）
    print(f"\n6. 边界情况测试:")
    print(f"   理想情况（所有正样本对完全一致）:")
    z_good = torch.cat([torch.ones(N, D), torch.ones(N, D)]) * 0.01
    z_good = F.normalize(z_good, dim=1)
    loss_good = criterion(z_good)
    print(f"   Loss = {loss_good.item():.4f}")

    print(f"\n   最差情况（随机向量，正样本碰巧接近）:")
    loss_random = criterion(z)
    print(f"   Loss = {loss.item():.4f}")

    print(f"\n   极端情况（所有向量完全相同 → 无法区分）:")
    z_all_same = F.normalize(torch.ones(2 * N, D), dim=1)
    loss_all_same = criterion(z_all_same)
    print(f"   Loss = {loss_all_same.item():.4f}")
    print(f"   → 所有相似度相同，区分不了正负样本，loss 高")
