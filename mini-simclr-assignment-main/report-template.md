# Mini-SimCLR 图像表征学习复现实验报告

## 1. 论文信息

- 论文名称：A Simple Framework for Contrastive Learning of Visual Representations
- 论文地址：https://arxiv.org/abs/2002.05709
- 官方代码参考：https://github.com/google-research/simclr

## 2. 任务说明

本实验复现的是 SimCLR 的核心思想：在没有人工标签参与预训练的情况下，通过对同一张图像生成两种随机增强视图，让模型学习图像表征。

```text
预训练输入：无标签 CIFAR-10 图像
预训练目标：同一图像的两个增强视图在表征空间中更接近，不同图像的表征相互区分
评估方式：丢弃 projection head，冻结 encoder，训练一个线性分类器，并报告 CIFAR-10 测试集准确率
```

## 3. 数据集

- 数据集名称：CIFAR-10
- 数据集地址：https://www.cs.toronto.edu/~kriz/cifar.html
- 实际使用预训练图像数：50000
- 实际使用 linear probe 训练图像数：50000
- 实际使用测试图像数：1000
- 使用设备：GPU（CUDA）
- 总训练耗时：5分钟

## 4. 数据增强

| 增强方法             | 参数设置                                                                              |
| -------------------- | ------------------------------------------------------------------------------------- |
| RandomResizedCrop    | size=32, scale=(0.2, 1.0)                                                             |
| RandomHorizontalFlip | p=0.5                                                                                 |
| ColorJitter          | brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2；通过 RandomApply 以 p=0.8 使用 |
| RandomGrayscale      | p=0.2                                                                                 |
| GaussianBlur         | 未使用                                                                                |

这些增强适合 SimCLR，因为它们可以在尽量保持图像语义类别不变的前提下改变图像的局部区域、颜色、方向和灰度信息。同一张图像经过两次独立随机增强后，模型不能只依赖像素级相似性，而需要学习更稳定的语义表征。RandomResizedCrop 迫使模型关注局部与整体的对应关系，ColorJitter 和 RandomGrayscale 降低模型对颜色的过拟合，RandomHorizontalFlip 增加常见视角变化。

## 5. 模型结构

```text
Image -> Two Augmented Views -> Shared Encoder -> Projection Head -> NT-Xent Loss
```

预训练阶段使用 encoder 和 projection head；评估阶段丢弃 projection head，只保留 encoder 输出的图像表征，并在其后训练一个线性分类器。

### 5.1 Encoder

- encoder 类型：小型 CNN
- 结构：Conv(3->64) + BN + ReLU + MaxPool；Conv(64->128) + BN + ReLU + MaxPool；Conv(128->256) + BN + ReLU + AdaptiveAvgPool
- 输出特征维度：256
- 是否使用预训练权重：linear probe 阶段加载自监督预训练得到的 encoder 权重

### 5.2 Projection Head

- MLP 层数：2 层 Linear
- hidden dimension：512
- output dimension：128
- 激活函数：ReLU
- BatchNorm：未在 projection head 中使用
- 输出处理：projection head 输出后进行 L2 normalize，用于计算余弦相似度

### 5.3 Linear Probe

- encoder 是否冻结：是
- linear classifier 输入维度：256
- 类别数：10
- 训练方式：只更新线性分类器参数，不更新 encoder

## 6. Loss 实现

本实验自行实现了 NT-Xent loss。

- batch size：128
- `2N` 个增强样本构造方式：每个 batch 中的图像分别生成两个增强视图 `x1` 和 `x2`，经过同一个 SimCLR 模型得到 `z1` 和 `z2`，再交替拼接为 `[z1_0, z2_0, z1_1, z2_1, ...]`
- 正样本索引确定方式：相邻两个样本为正样本对，使用 `pos_indices = torch.arange(N) ^ 1` 翻转最低位，例如 0 的正样本为 1，1 的正样本为 0
- temperature：0.5
- logits shape：对于 batch size 128，拼接后为 256 个视图，相似度矩阵 shape 为 `(256, 256)`，对角线表示自身相似度，会被 mask 掉

关键伪代码如下：

```python
sim = z @ z.T
sim = sim / temperature
pos_indices = torch.arange(z.shape[0], device=z.device) ^ 1
pos_sim = sim[torch.arange(z.shape[0], device=z.device), pos_indices]
sim_no_self = sim.masked_fill(torch.eye(z.shape[0], device=z.device).bool(), float("-inf"))
loss = (-pos_sim + torch.logsumexp(sim_no_self, dim=1)).mean()
```

## 7. 训练设置

### 7.1 自监督预训练

| 配置          |     数值 |
| ------------- | -------: |
| train images  |    50000 |
| epochs        |        3 |
| batch size    |      128 |
| optimizer     |    AdamW |
| learning rate |    0.001 |
| temperature   |      0.5 |
| encoder       | 小型 CNN |
| device        |      GPU |

### 7.2 Linear Probe

| 配置          |                                 数值 |
| ------------- | -----------------------------------: |
| train images  |                                50000 |
| test images   |                                 1000 |
| epochs        |                                    3 |
| batch size    |                                  128 |
| optimizer     | SGD, momentum=0.9, weight_decay=1e-4 |
| learning rate |                                  0.1 |
| device        |                                  GPU |

## 8. 训练过程

预训练 contrastive loss 日志如下：

| Epoch | Contrastive Loss | Learning Rate |
| ----- | ---------------: | ------------: |
| 1     |           4.9795 |     0.0007525 |
| 2     |           4.6762 |     0.0002575 |
| 3     |           4.5360 |     0.0000100 |

loss 在 3 个 epoch 中持续下降，从 4.9795 降到 4.5360，说明模型在对比学习目标上确实完成了优化。由于本实验只训练 3 个 epoch，loss 还没有完全收敛，但训练过程稳定，没有出现发散或 NaN。

Linear probe 训练日志如下：

| Epoch |   Loss | Train Acc | Test Acc |
| ----- | -----: | --------: | -------: |
| 1     | 1.7546 |    38.65% |   40.70% |
| 2     | 1.6049 |    39.40% |   47.00% |
| 3     | 1.5758 |    45.39% |   47.70% |

## 9. Linear Probe 结果

| 指标            |   结果 |
| --------------- | -----: |
| test accuracy   | 47.70% |
| random baseline |    10% |

最终测试准确率为 47.70%，明显高于 CIFAR-10 十分类随机猜测的 10%。这说明自监督预训练得到的 encoder 表征包含了可用于分类的有效信息。由于训练轮数较少、encoder 采用较浅的小型 CNN，并且测试集只取了 1000 张图像，因此准确率与完整 SimCLR 大规模实验相比仍有明显差距。继续增加 epoch、使用 ResNet-18、调节 batch size 和 temperature，预计可以进一步提升结果。

## 10. 预测结果展示

预测样例图已保存到：

```text
report/figures/prediction_samples.png
```

5 个测试样例的预测结果如下：

| 图片编号 | 真实类别 | 预测类别 | 是否正确 |
| -------- | -------- | -------- | -------- |
| 1        | cat      | frog     | 否       |
| 2        | ship     | ship     | 是       |
| 3        | ship     | ship     | 是       |
| 4        | airplane | ship     | 否       |
| 5        | frog     | frog     | 是       |

## 11. 问题与改进

实验过程中主要遇到的问题如下：

- 最开始脚本中的相对路径不稳定，从项目根目录运行时会找不到 `data` 和 `checkpoints`。
- `requirements.txt` 原本为空，换环境运行时不方便安装依赖。
- linear probe 最初容易和 SimCLR 预训练 dataloader 混用，监督分类阶段需要使用带标签的普通 CIFAR-10 dataloader。
- checkpoint 中的 encoder 类型需要和 linear probe 构建模型时保持一致，否则会出现权重 shape 不匹配。

解决方式：

- 将路径改为基于脚本所在目录自动解析，保证从项目根目录运行也能找到数据和模型。
- 补充 `requirements.txt`，加入 `torch`、`torchvision`、`tqdm`、`matplotlib`。
- 单独实现 linear probe dataloader，训练分类器时使用标签。
- linear probe 自动读取 checkpoint 中保存的 `encoder` 配置，避免手动填错。

后续改进方向：

- 增加预训练 epoch，例如从 3 提升到 20 或更多。
- 尝试 ResNet-18 encoder，提高表征能力。
- 对比不同 temperature，例如 0.1、0.5、1.0。
- 对比不同 batch size 对 NT-Xent loss 的影响。
- 加入 GaussianBlur，进一步贴近 SimCLR 原论文的数据增强策略。
- 使用完整测试集评估，并加入 t-SNE 或 nearest neighbor retrieval 可视化。

## 12. AI 对话过程记录

- 录制工具：entire
- 对话链接：
- 使用的 AI 模型：GPT deepseek
- 累计对话时长 / 会话数：

AI 主要帮助完成了代码运行排错、路径修复、linear probe 流程检查、实验日志整理和报告模板填写。

## 13. Git 提交记录

- 仓库地址：待填写
- 总 commit 数：5

`git log --oneline` 输出：

```text
14e175e 跑通代码
4f46474 现 NT-Xent loss教
134bd42 实现ResNet encoder
bae21d3 数据读取和增强
66be116 Initial commit: Mini-SimCLR assignment project structure
```
