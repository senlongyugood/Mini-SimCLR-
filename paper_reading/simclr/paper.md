# A Simple Framework for Contrastive Learning of Visual Representations

**论文元信息**

| 字段 | 值 |
|------|-----|
| 标题 | A Simple Framework for Contrastive Learning of Visual Representations |
| 作者 | Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton |
| 单位 | Google Research, Brain Team |
| 会议 | ICML 2020, Vienna, Austria, PMLR 119 |
| arXiv | 2002.05709v3 [cs.LG] |
| 代码 | https://github.com/google-research/simclr |

---

## 目录 / Page Index

- [Abstract / 摘要](#abstract) — p.1
- [1. Introduction / 引言](#sec1) — pp.1–2
- [2. Method / 方法](#sec2) — pp.2–3
- [3. Data Augmentation / 数据增强](#sec3) — pp.3–5
- [4. Architectures for Encoder and Head / 编码器与投影头架构](#sec4) — pp.5–6
- [5. Loss Functions and Batch Size / 损失函数与批量大小](#sec5) — pp.6–7
- [6. Comparison with State-of-the-art / 与前沿方法的比较](#sec6) — pp.7–8
- [7. Related Work / 相关工作](#sec7) — pp.8–9
- [8. Conclusion / 结论](#sec8) — p.9
- [Appendices / 附录](#appendices) — pp.12–20
- [术语表 / Terminology](#terminology)
- [阅读提示 / Critical Reading Notes](#reading-notes)

---

<a id="S001"></a>
**Source:** p.1 S001 — **Title & Authors**

> **Original:** A Simple Framework for Contrastive Learning of Visual Representations
>
> **中文:** 一个简单的视觉表征对比学习框架

> **Original:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton — Google Research, Brain Team
>
> **中文:** 陈霆, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton — Google Research, Brain Team

---

## Abstract / 摘要 <a id="sec-abstract"></a>

<a id="S002"></a>
**Source:** p.1 S002 — Abstract ¶1

**Original:** This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100× fewer labels.

**中文:** 本文提出 SimCLR：一个简单的视觉表征对比学习框架。我们简化了最近提出的对比自监督学习算法，无需专门的架构或记忆库。为了理解是什么使得对比预测任务能够学习到有用的表征，我们系统地研究了框架中的主要组件。我们表明：(1) 数据增强的组合在定义有效的预测任务中起着关键作用，(2) 在表征和对比损失之间引入可学习的非线性变换显著提升了所学表征的质量，(3) 与监督学习相比，对比学习受益于更大的批量大小和更多的训练步数。通过结合这些发现，我们在 ImageNet 上的自监督和半监督学习显著超越了先前的方法。在 SimCLR 学到的自监督表征上训练的线性分类器达到了 76.5% 的 top-1 准确率，相比先前的最先进方法有 7% 的相对提升，匹配了有监督 ResNet-50 的性能。在仅使用 1% 标签进行微调时，我们达到了 85.8% 的 top-5 准确率，以少 100 倍的标签超越了 AlexNet。

---

<a id="F001"></a>
**Figure 1** (p.1) — *ImageNet Top-1 accuracy of linear classifiers trained on representations learned with different self-supervised methods (pretrained on ImageNet). Gray cross indicates supervised ResNet-50. Our method, SimCLR, is shown in bold.*

**图 1:** 使用不同自监督方法（在 ImageNet 上预训练）学到的表征训练的线性分类器的 ImageNet Top-1 准确率。灰色叉号表示有监督 ResNet-50。我们的方法 SimCLR 以粗体显示。

---

## 1. Introduction / 引言 <a id="sec1"></a>

<a id="S003"></a>
**Source:** p.1 S003 — Introduction ¶1

**Original:** Learning effective visual representations without human supervision is a long-standing problem. Most mainstream approaches fall into one of two classes: generative or discriminative. Generative approaches learn to generate or otherwise model pixels in the input space (Hinton et al., 2006; Kingma & Welling, 2013; Goodfellow et al., 2014).

**中文:** 在无人工监督的情况下学习有效的视觉表征是一个长期存在的问题。大多数主流方法分为两类：生成式或判别式。生成式方法学习生成或以其他方式建模输入空间中的像素（Hinton et al., 2006; Kingma & Welling, 2013; Goodfellow et al., 2014）。

---

<a id="S004"></a>
**Source:** p.1 S004 — Introduction ¶2

**Original:** However, pixel-level generation is computationally expensive and may not be necessary for representation learning. Discriminative approaches learn representations using objective functions similar to those used for supervised learning, but train networks to perform pretext tasks where both the inputs and labels are derived from an unlabeled dataset. Many such approaches have relied on heuristics to design pretext tasks (Doersch et al., 2015; Zhang et al., 2016; Noroozi & Favaro, 2016; Gidaris et al., 2018), which could limit the generality of the learned representations. Discriminative approaches based on contrastive learning in the latent space have recently shown great promise, achieving state-of-the-art results (Hadsell et al., 2006; Dosovitskiy et al., 2014; Oord et al., 2018; Bachman et al., 2019).

**中文:** 然而，像素级生成的计算代价高昂，且对于表征学习可能并非必要。判别式方法使用类似于监督学习的目标函数来学习表征，但训练网络执行的是前置任务，其中输入和标签均来自无标签数据集。许多此类方法依赖启发式设计前置任务（Doersch et al., 2015; Zhang et al., 2016; Noroozi & Favaro, 2016; Gidaris et al., 2018），这可能限制了所学表征的泛化性。基于潜在空间中对比学习的判别式方法最近展现出了巨大潜力，取得了最先进的结果（Hadsell et al., 2006; Dosovitskiy et al., 2014; Oord et al., 2018; Bachman et al., 2019）。

---

<a id="S005"></a>
**Source:** p.1 S005 — Introduction ¶3

**Original:** In this work, we introduce a simple framework for contrastive learning of visual representations, which we call SimCLR. Not only does SimCLR outperform previous work (Figure 1), but it is also simpler, requiring neither specialized architectures (Bachman et al., 2019; Hénaff et al., 2019) nor a memory bank (Wu et al., 2018; Tian et al., 2019; He et al., 2019; Misra & van der Maaten, 2019).

**中文:** 在本文中，我们提出一个简单的视觉表征对比学习框架，称为 SimCLR。SimCLR 不仅超越了先前的工作（图 1），而且更加简单，既不需要专门的架构（Bachman et al., 2019; Hénaff et al., 2019），也不需要记忆库（Wu et al., 2018; Tian et al., 2019; He et al., 2019; Misra & van der Maaten, 2019）。

---

<a id="S006"></a>
**Source:** pp.1–2 S006 — Introduction ¶4 (key findings)

**Original:** In order to understand what enables good contrastive representation learning, we systematically study the major components of our framework and show that:

- Composition of multiple data augmentation operations is crucial in defining the contrastive prediction tasks that yield effective representations. In addition, unsupervised contrastive learning benefits from stronger data augmentation than supervised learning.
- Introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations.
- Representation learning with contrastive cross entropy loss benefits from normalized embeddings and an appropriately adjusted temperature parameter.
- Contrastive learning benefits from larger batch sizes and longer training compared to its supervised counterpart. Like supervised learning, contrastive learning benefits from deeper and wider networks.

**中文:** 为了理解是什么促成了良好的对比表征学习，我们系统地研究了框架中的主要组件，并表明：

- 多种数据增强操作的组合对于定义能产生有效表征的对比预测任务至关重要。此外，无监督对比学习比监督学习受益于更强的数据增强。
- 在表征和对比损失之间引入可学习的非线性变换显著提升了所学表征的质量。
- 使用对比交叉熵损失进行表征学习受益于归一化嵌入和适当调整的温度参数。
- 与监督学习相比，对比学习受益于更大的批量大小和更长的训练。与监督学习一样，对比学习也受益于更深更宽的网络。

---

<a id="S007"></a>
**Source:** p.2 S007 — Introduction ¶5 (results summary)

**Original:** We combine these findings to achieve a new state-of-the-art in self-supervised and semi-supervised learning on ImageNet ILSVRC-2012 (Russakovsky et al., 2015). Under the linear evaluation protocol, SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art (Hénaff et al., 2019). When fine-tuned with only 1% of the ImageNet labels, SimCLR achieves 85.8% top-5 accuracy, a relative improvement of 10% (Hénaff et al., 2019). When fine-tuned on other natural image classification datasets, SimCLR performs on par with or better than a strong supervised baseline (Kornblith et al., 2019) on 10 out of 12 datasets.

**中文:** 我们结合这些发现，在 ImageNet ILSVRC-2012（Russakovsky et al., 2015）上的自监督和半监督学习中取得了新的最先进水平。在线性评估协议下，SimCLR 达到 76.5% 的 top-1 准确率，相比先前最先进方法（Hénaff et al., 2019）有 7% 的相对提升。在仅使用 1% ImageNet 标签进行微调时，SimCLR 达到 85.8% 的 top-5 准确率，相对提升 10%（Hénaff et al., 2019）。在其他自然图像分类数据集上微调时，SimCLR 在 12 个数据集中的 10 个上与强监督基线（Kornblith et al., 2019）表现相当或更好。

---

## 2. Method / 方法 <a id="sec2"></a>

### 2.1 The Contrastive Learning Framework / 对比学习框架

<a id="S008"></a>
**Source:** p.2 S008 — §2.1 ¶1

**Original:** Inspired by recent contrastive learning algorithms (see Section 7 for an overview), SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. As illustrated in Figure 2, this framework comprises the following four major components.

**中文:** 受近期对比学习算法的启发（概述见第 7 节），SimCLR 通过在潜在空间中最大化同一数据样本的不同增强视图之间的一致性来学习表征。如图 2 所示，该框架包含以下四个主要组件。

---

<a id="F002"></a>
**Figure 2** (p.2) — *A simple framework for contrastive learning of visual representations. Two separate data augmentation operators are sampled from the same family of augmentations (t ∼ T and t′ ∼ T) and applied to each data example to obtain two correlated views. A base encoder network f(·) and a projection head g(·) are trained to maximize agreement using a contrastive loss. After training is completed, we throw away the projection head g(·) and use encoder f(·) and representation h for downstream tasks.*

**图 2:** 视觉表征对比学习的简单框架。从同一增强族（t ∼ T 和 t′ ∼ T）中采样两个独立的数据增强算子，并应用于每个数据样本以获得两个相关的视图。基础编码器网络 f(·) 和投影头 g(·) 被训练以使用对比损失最大化一致性。训练完成后，丢弃投影头 g(·)，使用编码器 f(·) 和表征 h 用于下游任务。

---

<a id="S009"></a>
**Source:** p.2 S009 — §2.1 ¶2 (four components)

**Original:**
- **A stochastic data augmentation module** that transforms any given data example randomly resulting in two correlated views of the same example, denoted x̃_i and x̃_j, which we consider as a positive pair. In this work, we sequentially apply three simple augmentations: random cropping followed by resize back to the original size, random color distortions, and random Gaussian blur. As shown in Section 3, the combination of random crop and color distortion is crucial to achieve a good performance.
- **A neural network base encoder f(·)** that extracts representation vectors from augmented data examples. Our framework allows various choices of the network architecture without any constraints. We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain h_i = f(x̃_i) = ResNet(x̃_i) where h_i ∈ R^d is the output after the average pooling layer.
- **A small neural network projection head g(·)** that maps representations to the space where contrastive loss is applied. We use a MLP with one hidden layer to obtain z_i = g(h_i) = W^(2)σ(W^(1)h_i) where σ is a ReLU nonlinearity. As shown in section 4, we find it beneficial to define the contrastive loss on z_i's rather than h_i's.
- **A contrastive loss function** defined for a contrastive prediction task. Given a set {x̃_k} including a positive pair of examples x̃_i and x̃_j, the contrastive prediction task aims to identify x̃_j in {x̃_k}_{k≠i} for a given x̃_i.

**中文:**
- **随机数据增强模块：** 对任意给定的数据样本进行随机变换，产生同一样本的两个相关视图，记为 x̃_i 和 x̃_j，视为一个正对。本文中，我们依次应用三种简单的增强：随机裁剪后调整回原始尺寸、随机颜色失真和随机高斯模糊。如第 3 节所示，随机裁剪与颜色失真的组合对于实现良好性能至关重要。
- **神经网络基础编码器 f(·)：** 从增强后的数据样本中提取表征向量。我们的框架允许各种网络架构选择，没有约束。为简单起见，我们采用常用的 ResNet（He et al., 2016），得到 h_i = f(x̃_i) = ResNet(x̃_i)，其中 h_i ∈ R^d 是平均池化层后的输出。
- **小型神经网络投影头 g(·)：** 将表征映射到应用对比损失的空间。我们使用带一个隐藏层的 MLP，得到 z_i = g(h_i) = W^(2)σ(W^(1)h_i)，其中 σ 是 ReLU 非线性函数。如第 4 节所示，我们发现在 z_i 上定义对比损失优于在 h_i 上。
- **对比损失函数：** 为对比预测任务而定义。给定包含正对样本 x̃_i 和 x̃_j 的集合 {x̃_k}，对比预测任务的目标是对于给定的 x̃_i，在 {x̃_k}_{k≠i} 中识别出 x̃_j。

---

<a id="S010"></a>
**Source:** p.2 S010 — §2.1 ¶3 (NT-Xent loss)

**Original:** We randomly sample a minibatch of N examples and define the contrastive prediction task on pairs of augmented examples derived from the minibatch, resulting in 2N data points. We do not sample negative examples explicitly. Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N−1) augmented examples within a minibatch as negative examples. Let sim(u, v) = u^⊤v / ∥u∥∥v∥ denote the dot product between ℓ₂ normalized u and v (i.e. cosine similarity). Then the loss function for a positive pair of examples (i, j) is defined as

ℓ_{i,j} = −log [ exp(sim(z_i, z_j) / τ) / Σ_{k=1}^{2N} 1_{[k≠i]} exp(sim(z_i, z_k) / τ) ],    (1)

where 1_{[k≠i]} ∈ {0,1} is an indicator function evaluating to 1 iff k ≠ i and τ denotes a temperature parameter. The final loss is computed across all positive pairs, both (i, j) and (j, i), in a mini-batch. This loss has been used in previous work (Sohn, 2016; Wu et al., 2018; Oord et al., 2018); for convenience, we term it NT-Xent (the normalized temperature-scaled cross entropy loss).

**中文:** 我们随机采样一个包含 N 个样本的小批量，并在从小批量派生的增强样本对上定义对比预测任务，得到 2N 个数据点。我们不显式采样负样本。相反，给定一个正对，类似于（Chen et al., 2017），我们将小批量中其他 2(N−1) 个增强样本视为负样本。令 sim(u, v) = u^⊤v / ∥u∥∥v∥ 表示 ℓ₂ 归一化后的 u 和 v 的点积（即余弦相似度）。则正对样本 (i, j) 的损失函数定义为

ℓ_{i,j} = −log [ exp(sim(z_i, z_j) / τ) / Σ_{k=1}^{2N} 1_{[k≠i]} exp(sim(z_i, z_k) / τ) ],    (1)

其中 1_{[k≠i]} ∈ {0,1} 是指示函数，当 k ≠ i 时值为 1，τ 表示温度参数。最终损失在小批量中的所有正对上计算，包括 (i, j) 和 (j, i)。该损失已在先前工作中使用（Sohn, 2016; Wu et al., 2018; Oord et al., 2018）；为方便起见，我们称其为 NT-Xent（归一化温度缩放交叉熵损失）。

---

<a id="S011"></a>
**Source:** p.3 S011 — Algorithm 1 (pseudocode)

**Original:** Algorithm 1 summarizes the proposed method.

**中文:** 算法 1 总结了所提出的方法。（详见原文第 3 页 Algorithm 1 伪代码）

---

### 2.2 Training with Large Batch Size / 大批量训练

<a id="S012"></a>
**Source:** p.3 S012 — §2.2 ¶1

**Original:** To keep it simple, we do not train the model with a memory bank (Wu et al., 2018; He et al., 2019). Instead, we vary the training batch size N from 256 to 8192. A batch size of 8192 gives us 16382 negative examples per positive pair from both augmentation views. Training with large batch size may be unstable when using standard SGD/Momentum with linear learning rate scaling (Goyal et al., 2017). To stabilize the training, we use the LARS optimizer (You et al., 2017) for all batch sizes. We train our model with Cloud TPUs, using 32 to 128 cores depending on the batch size.

**中文:** 为保持简单，我们不使用记忆库（Wu et al., 2018; He et al., 2019）训练模型。相反，我们将训练批量大小 N 从 256 变化到 8192。8192 的批量大小为每个正对提供了来自两个增强视图的 16382 个负样本。使用标准 SGD/Momentum 和线性学习率缩放（Goyal et al., 2017）时，大批量训练可能不稳定。为稳定训练，我们对所有批量大小使用 LARS 优化器（You et al., 2017）。我们使用 Cloud TPU 训练模型，根据批量大小使用 32 到 128 个核心。

---

<a id="S013"></a>
**Source:** p.3 S013 — §2.2 ¶2 (Global BN)

**Original:** **Global BN.** Standard ResNets use batch normalization (Ioffe & Szegedy, 2015). In distributed training with data parallelism, the BN mean and variance are typically aggregated locally per device. In our contrastive learning, as positive pairs are computed in the same device, the model can exploit the local information leakage to improve prediction accuracy without improving representations. We address this issue by aggregating BN mean and variance over all devices during the training. Other approaches include shuffling data examples across devices (He et al., 2019), or replacing BN with layer norm (Hénaff et al., 2019).

**中文:** **全局 BN。** 标准 ResNet 使用批归一化（Ioffe & Szegedy, 2015）。在数据并行的分布式训练中，BN 的均值和方差通常在各设备本地聚合。在我们的对比学习中，由于正对在同一设备上计算，模型可以利用局部信息泄漏来提高预测准确率，而无需改进表征。我们通过在训练期间跨所有设备聚合 BN 均值和方差来解决此问题。其他方法包括跨设备混洗数据样本（He et al., 2019），或用层归一化替代 BN（Hénaff et al., 2019）。

---

### 2.3 Evaluation Protocol / 评估协议

<a id="S014"></a>
**Source:** p.3 S014 — §2.3

**Original:** **Dataset and Metrics.** Most of our study for unsupervised pretraining (learning encoder network f without labels) is done using the ImageNet ILSVRC-2012 dataset (Russakovsky et al., 2015). Some additional pretraining experiments on CIFAR-10 (Krizhevsky & Hinton, 2009) can be found in Appendix B.9. We also test the pretrained results on a wide range of datasets for transfer learning. To evaluate the learned representations, we follow the widely used linear evaluation protocol (Zhang et al., 2016; Oord et al., 2018; Bachman et al., 2019; Kolesnikov et al., 2019), where a linear classifier is trained on top of the frozen base network, and test accuracy is used as a proxy for representation quality. Beyond linear evaluation, we also compare against state-of-the-art on semi-supervised and transfer learning.

**Default setting.** Unless otherwise specified, for data augmentation we use random crop and resize (with random flip), color distortions, and Gaussian blur (for details, see Appendix A). We use ResNet-50 as the base encoder network, and a 2-layer MLP projection head to project the representation to a 128-dimensional latent space. As the loss, we use NT-Xent, optimized using LARS with learning rate of 4.8 (= 0.3 × BatchSize/256) and weight decay of 10^−6. We train at batch size 4096 for 100 epochs. Furthermore, we use linear warmup for the first 10 epochs, and decay the learning rate with the cosine decay schedule without restarts (Loshchilov & Hutter, 2016).

**中文:** **数据集与指标。** 我们大部分无监督预训练（学习无标签的编码器网络 f）的研究使用 ImageNet ILSVRC-2012 数据集（Russakovsky et al., 2015）。CIFAR-10（Krizhevsky & Hinton, 2009）上的一些额外预训练实验见附录 B.9。我们还在广泛的数据集上测试了预训练结果的迁移学习性能。为评估所学表征，我们遵循广泛使用的线性评估协议（Zhang et al., 2016; Oord et al., 2018; Bachman et al., 2019; Kolesnikov et al., 2019），在冻结的基础网络之上训练线性分类器，以测试准确率作为表征质量的代理指标。除线性评估外，我们还在半监督和迁移学习上与最先进方法进行比较。

**默认设置。** 除非另有说明，数据增强我们使用随机裁剪和调整大小（带随机翻转）、颜色失真和高斯模糊（详见附录 A）。我们使用 ResNet-50 作为基础编码器网络，以及一个 2 层 MLP 投影头将表征投影到 128 维潜在空间。损失函数使用 NT-Xent，用 LARS 优化器优化，学习率为 4.8（= 0.3 × BatchSize/256），权重衰减为 10^−6。我们在批量大小 4096 下训练 100 个 epoch。此外，前 10 个 epoch 使用线性预热，并使用无重启的余弦衰减策略衰减学习率（Loshchilov & Hutter, 2016）。

---

<a id="F003"></a>
**Figure 3** (p.3) — *(a) Global and local views. (b) Adjacent views. Solid rectangles are images, dashed rectangles are random crops. By randomly cropping images, we sample contrastive prediction tasks that include global to local view (B→A) or adjacent view (D→C) prediction.*

**图 3:** (a) 全局与局部视图。(b) 相邻视图。实线矩形为图像，虚线矩形为随机裁剪。通过随机裁剪图像，我们采样包含全局到局部视图（B→A）或相邻视图（D→C）预测的对比预测任务。

---

## 3. Data Augmentation for Contrastive Representation Learning / 数据增强与对比表征学习 <a id="sec3"></a>

<a id="S015"></a>
**Source:** p.3–4 S015 — §3 intro

**Original:** Data augmentation defines predictive tasks. While data augmentation has been widely used in both supervised and unsupervised representation learning (Krizhevsky et al., 2012; Hénaff et al., 2019; Bachman et al., 2019), it has not been considered as a systematic way to define the contrastive prediction task. Many existing approaches define contrastive prediction tasks by changing the architecture. For example, Hjelm et al. (2018); Bachman et al. (2019) achieve global-to-local view prediction via constraining the receptive field in the network architecture, whereas Oord et al. (2018); Hénaff et al. (2019) achieve neighboring view prediction via a fixed image splitting procedure and a context aggregation network. We show that this complexity can be avoided by performing simple random cropping (with resizing) of target images, which creates a family of predictive tasks subsuming the above mentioned two, as shown in Figure 3. This simple design choice conveniently decouples the predictive task from other components such as the neural network architecture. Broader contrastive prediction tasks can be defined by extending the family of augmentations and composing them stochastically.

**中文:** 数据增强定义了预测任务。虽然数据增强已广泛用于监督和无监督表征学习（Krizhevsky et al., 2012; Hénaff et al., 2019; Bachman et al., 2019），但它尚未被视为定义对比预测任务的一种系统性方法。许多现有方法通过改变架构来定义对比预测任务。例如，Hjelm et al. (2018); Bachman et al. (2019) 通过约束网络架构中的感受野来实现全局到局部视图预测，而 Oord et al. (2018); Hénaff et al. (2019) 通过固定的图像分割过程和上下文聚合网络来实现相邻视图预测。我们表明，通过对目标图像执行简单的随机裁剪（带调整大小）即可避免这种复杂性，这创建了一族包含上述两种预测任务的任务，如图 3 所示。这种简单的设计选择方便地将预测任务与神经网络架构等其他组件解耦。通过扩展增强族并随机组合它们，可以定义更广泛的对比预测任务。

---

### 3.1 Composition of Data Augmentation / 数据增强的组合

<a id="S016"></a>
**Source:** pp.4–5 S016 — §3.1 ¶1

**Original:** To systematically study the impact of data augmentation, we consider several common augmentations here. One type of augmentation involves spatial/geometric transformation of data, such as cropping and resizing (with horizontal flipping), rotation (Gidaris et al., 2018) and cutout (DeVries & Taylor, 2017). The other type of augmentation involves appearance transformation, such as color distortion (including color dropping, brightness, contrast, saturation, hue) (Howard, 2013; Szegedy et al., 2015), Gaussian blur, and Sobel filtering. Figure 4 visualizes the augmentations that we study in this work.

**中文:** 为系统研究数据增强的影响，我们在此考虑几种常见的增强。一类增强涉及数据的空间/几何变换，如裁剪和调整大小（带水平翻转）、旋转（Gidaris et al., 2018）和 cutout（DeVries & Taylor, 2017）。另一类增强涉及外观变换，如颜色失真（包括颜色丢弃、亮度、对比度、饱和度、色调）（Howard, 2013; Szegedy et al., 2015）、高斯模糊和 Sobel 滤波。图 4 可视化了本文研究的增强。

---

<a id="F004"></a>
**Figure 4** (p.4) — *Illustrations of the studied data augmentation operators. Each augmentation can transform data stochastically with some internal parameters (e.g. rotation degree, noise level). Note that we only test these operators in ablation, the augmentation policy used to train our models only includes random crop (with flip and resize), color distortion, and Gaussian blur.*

**图 4:** 所研究的数据增强算子的示意图。每种增强可以通过一些内部参数（如旋转角度、噪声水平）随机变换数据。注意我们仅在消融实验中测试这些算子，用于训练模型的增强策略仅包括随机裁剪（带翻转和调整大小）、颜色失真和高斯模糊。

---

<a id="F005"></a>
**Figure 5** (p.4) — *Linear evaluation (ImageNet top-1 accuracy) under individual or composition of data augmentations, applied only to one branch. For all columns but the last, diagonal entries correspond to single transformation, and off-diagonals correspond to composition of two transformations (applied sequentially). The last column reflects the average over the row.*

**图 5:** 单独或组合数据增强（仅应用于一个分支）下的线性评估（ImageNet top-1 准确率）。除最后一列外，对角线对应单一变换，非对角线对应两个变换的组合（依次应用）。最后一列反映该行的平均值。

---

<a id="S017"></a>
**Source:** p.5 S017 — §3.1 key finding

**Original:** We observe that no single transformation suffices to learn good representations, even though the model can almost perfectly identify the positive pairs in the contrastive task. When composing augmentations, the contrastive prediction task becomes harder, but the quality of representation improves dramatically. Appendix B.2 provides a further study on composing broader set of augmentations.

One composition of augmentations stands out: random cropping and random color distortion. We conjecture that one serious issue when using only random cropping as data augmentation is that most patches from an image share a similar color distribution. Figure 6 shows that color histograms alone suffice to distinguish images. Neural nets may exploit this shortcut to solve the predictive task. Therefore, it is critical to compose cropping with color distortion in order to learn generalizable features.

**中文:** 我们观察到，没有任何单一变换足以学习到良好的表征，尽管模型几乎可以完美地在对比任务中识别正对。当组合增强时，对比预测任务变得更难，但表征质量显著提升。附录 B.2 提供了组合更广泛增强的进一步研究。

其中一种增强组合尤为突出：随机裁剪和随机颜色失真。我们推测，仅使用随机裁剪作为数据增强的一个严重问题是，图像的大多数补丁共享相似的颜色分布。图 6 显示仅颜色直方图就足以区分图像。神经网络可能利用这种捷径来解决预测任务。因此，将裁剪与颜色失真组合使用对于学习可泛化的特征至关重要。

---

<a id="F006"></a>
**Figure 6** (p.5) — *Histograms of pixel intensities (over all channels) for different crops of two different images (i.e. two rows). The image for the first row is from Figure 4. All axes have the same range.*

**图 6:** 两幅不同图像（即两行）不同裁剪区域的像素强度直方图（跨所有通道）。第一行的图像来自图 4。所有坐标轴具有相同的范围。

---

### 3.2 Contrastive Learning Needs Stronger Augmentation / 对比学习需要更强的增强

<a id="S018"></a>
**Source:** p.5 S018 — §3.2

**Original:** Stronger color augmentation substantially improves the linear evaluation of the learned unsupervised models. In this context, AutoAugment (Cubuk et al., 2019), a sophisticated augmentation policy found using supervised learning, does not work better than simple cropping + (stronger) color distortion. When training supervised models with the same set of augmentations, we observe that stronger color augmentation does not improve or even hurts their performance. Thus, our experiments show that unsupervised contrastive learning benefits from stronger (color) data augmentation than supervised learning.

**中文:** 更强的颜色增强显著提升了所学无监督模型的线性评估结果。在此背景下，AutoAugment（Cubuk et al., 2019）——一种通过监督学习找到的复杂增强策略——并不比简单的裁剪 +（更强的）颜色失真效果更好。当使用同一组增强训练监督模型时，我们观察到更强的颜色增强不会提升甚至会损害其性能。因此，我们的实验表明，无监督对比学习比监督学习受益于更强的（颜色）数据增强。

---

<a id="T001"></a>
**Table 1** (p.5) — *Top-1 accuracy of unsupervised ResNet-50 using linear evaluation and supervised ResNet-50, under varied color distortion strength and other data transformations.*

**表 1:** 在不同颜色失真强度和其他数据变换下，使用线性评估的无监督 ResNet-50 和有监督 ResNet-50 的 Top-1 准确率。

| Methods | 1/8 | 1/4 | 1/2 | 1 | 1 (+Blur) | AutoAug |
|---------|-----|-----|-----|---|-----------|---------|
| SimCLR | 59.6 | 61.0 | 62.6 | 63.2 | 64.5 | 61.1 |
| Supervised | 77.0 | 76.7 | 76.5 | 75.7 | 75.4 | 77.1 |

---

## 4. Architectures for Encoder and Head / 编码器与投影头架构 <a id="sec4"></a>

### 4.1 Bigger Models Help / 更大模型有益

<a id="S019"></a>
**Source:** p.5 S019 — §4.1

**Original:** Figure 7 shows, perhaps unsurprisingly, that increasing depth and width both improve performance. While similar findings hold for supervised learning (He et al., 2016), we find the gap between supervised models and linear classifiers trained on unsupervised models shrinks as the model size increases, suggesting that unsupervised learning benefits more from bigger models than its supervised counterpart.

**中文:** 图 7 显示——也许并不令人意外——增加深度和宽度都能提升性能。虽然类似的发现也适用于监督学习（He et al., 2016），但我们发现随着模型规模的增大，监督模型与在无监督模型上训练的线性分类器之间的差距在缩小，这表明无监督学习比监督学习更受益于更大的模型。

---

<a id="F007"></a>
**Figure 7** (p.5) — *Linear evaluation of models with varied depth and width. Models in blue dots are ours trained for 100 epochs, models in red stars are ours trained for 1000 epochs, and models in green crosses are supervised ResNets trained for 90 epochs (He et al., 2016).*

**图 7:** 不同深度和宽度模型的线性评估。蓝色圆点为我们训练 100 个 epoch 的模型，红色星号为我们训练 1000 个 epoch 的模型，绿色叉号为训练了 90 个 epoch 的有监督 ResNet（He et al., 2016）。

---

### 4.2 Nonlinear Projection Head / 非线性投影头

<a id="S020"></a>
**Source:** p.6 S020 — §4.2 ¶1–2

**Original:** We then study the importance of including a projection head, i.e. g(h). Figure 8 shows linear evaluation results using three different architecture for the head: (1) identity mapping; (2) linear projection, as used by several previous approaches (Wu et al., 2018); and (3) the default nonlinear projection with one additional hidden layer (and ReLU activation), similar to Bachman et al. (2019). We observe that a nonlinear projection is better than a linear projection (+3%), and much better than no projection (>10%). When a projection head is used, similar results are observed regardless of output dimension. Furthermore, even when nonlinear projection is used, the layer before the projection head, h, is still much better (>10%) than the layer after, z = g(h), which shows that the hidden layer before the projection head is a better representation than the layer after.

**中文:** 我们随后研究了包含投影头 g(h) 的重要性。图 8 显示了使用三种不同投影头架构的线性评估结果：(1) 恒等映射；(2) 线性投影，如几项先前工作所用（Wu et al., 2018）；(3) 默认的非线性投影，带一个额外的隐藏层（和 ReLU 激活），类似于 Bachman et al. (2019)。我们观察到非线性投影优于线性投影（+3%），且远优于无投影（>10%）。使用投影头时，无论输出维度如何，观察到相似的结果。此外，即使使用非线性投影，投影头之前的层 h 仍然远优于（>10%）投影之后的层 z = g(h)，这表明投影头之前的隐藏层是比之后更好的表征。

---

<a id="S021"></a>
**Source:** p.6 S021 — §4.2 ¶3

**Original:** We conjecture that the importance of using the representation before the nonlinear projection is due to loss of information induced by the contrastive loss. In particular, z = g(h) is trained to be invariant to data transformation. Thus, g can remove information that may be useful for the downstream task, such as the color or orientation of objects. By leveraging the nonlinear transformation g(·), more information can be formed and maintained in h. To verify this hypothesis, we conduct experiments that use either h or g(h) to learn to predict the transformation applied during the pretraining.

**中文:** 我们推测使用非线性投影之前的表征的重要性源于对比损失引起的信息丢失。具体来说，z = g(h) 被训练为对数据变换不变。因此，g 可能移除对下游任务有用的信息，如物体的颜色或朝向。通过利用非线性变换 g(·)，更多信息可以在 h 中形成和保持。为验证这一假设，我们进行了使用 h 或 g(h) 来学习预测预训练期间所应用变换的实验。

---

<a id="F008"></a>
**Figure 8** (p.6) — *Linear evaluation of representations with different projection heads g(·) and various dimensions of z = g(h). The representation h (before projection) is 2048-dimensional here.*

**图 8:** 使用不同投影头 g(·) 和不同 z = g(h) 维度的表征的线性评估。此处表征 h（投影前）为 2048 维。

---

<a id="T002"></a>
**Table 3** (p.6) — *Accuracy of training additional MLPs on different representations to predict the transformation applied.*

**表 3:** 在不同表征上训练额外的 MLP 来预测所应用变换的准确率。

| What to predict? | Random guess | h | g(h) |
|-----------------|-------------|-----|------|
| Color vs grayscale | 80 | 99.3 | 97.4 |
| Rotation | 25 | 67.6 | 25.6 |
| Orig. vs corrupted | 50 | 99.5 | 59.6 |
| Orig. vs Sobel filtered | 50 | 96.6 | 56.3 |

---

## 5. Loss Functions and Batch Size / 损失函数与批量大小 <a id="sec5"></a>

### 5.1 NT-Xent vs Alternatives / NT-Xent 与其他损失函数

<a id="S022"></a>
**Source:** p.6 S022 — §5.1 ¶1

**Original:** We compare the NT-Xent loss against other commonly used contrastive loss functions, such as logistic loss (Mikolov et al., 2013), and margin loss (Schroff et al., 2015). Table 2 shows the objective function as well as the gradient to the input of the loss function. Looking at the gradient, we observe 1) ℓ₂ normalization (i.e. cosine similarity) along with temperature effectively weights different examples, and an appropriate temperature can help the model learn from hard negatives; and 2) unlike cross-entropy, other objective functions do not weigh the negatives by their relative hardness. As a result, one must apply semi-hard negative mining (Schroff et al., 2015) for these loss functions.

**中文:** 我们将 NT-Xent 损失与其他常用的对比损失函数进行比较，如逻辑损失（Mikolov et al., 2013）和边界损失（Schroff et al., 2015）。表 2 显示了目标函数以及损失函数输入的梯度。观察梯度我们发现：1) ℓ₂ 归一化（即余弦相似度）与温度一起有效地对不同样本进行加权，适当的温度可以帮助模型从困难负样本中学习；2) 与交叉熵不同，其他目标函数不会根据负样本的相对难度对其进行加权。因此，对于这些损失函数必须应用半困难负样本挖掘（Schroff et al., 2015）。

---

<a id="T003"></a>
**Table 2** (p.6) — *Negative loss functions and their gradients. All input vectors, i.e. u, v⁺, v⁻, are ℓ₂ normalized. NT-Xent is an abbreviation for "Normalized Temperature-scaled Cross Entropy".*

**表 2:** 负损失函数及其梯度。所有输入向量 u, v⁺, v⁻ 均经过 ℓ₂ 归一化。NT-Xent 是"归一化温度缩放交叉熵"的缩写。

---

<a id="T004"></a>
**Table 4** (p.7) — *Linear evaluation (top-1) for models trained with different loss functions. "sh" means using semi-hard negative mining.*

**表 4:** 使用不同损失函数训练的模型的线性评估（top-1）。"sh" 表示使用半困难负样本挖掘。

| Margin | NT-Logi. | Margin (sh) | NT-Logi.(sh) | NT-Xent |
|--------|----------|-------------|--------------|---------|
| 50.9 | 51.6 | 57.5 | 57.9 | 63.9 |

---

<a id="S023"></a>
**Source:** p.7 S023 — §5.1 ¶2

**Original:** We next test the importance of the ℓ₂ normalization (i.e. cosine similarity vs dot product) and temperature τ in our default NT-Xent loss. Table 5 shows that without normalization and proper temperature scaling, performance is significantly worse. Without ℓ₂ normalization, the contrastive task accuracy is higher, but the resulting representation is worse under linear evaluation.

**中文:** 我们接下来测试 ℓ₂ 归一化（即余弦相似度 vs 点积）和温度 τ 在我们默认 NT-Xent 损失中的重要性。表 5 显示，没有归一化和适当的温度缩放，性能显著下降。没有 ℓ₂ 归一化时，对比任务准确率更高，但线性评估下所得表征更差。

---

<a id="T005"></a>
**Table 5** (p.7) — *Linear evaluation for models trained with different choices of ℓ₂ norm and temperature τ for NT-Xent loss.*

**表 5:** 使用不同 ℓ₂ 归一化和温度 τ 选择训练的模型的线性评估。

| ℓ₂ norm? | τ | Entropy | Contrastive acc. | Top 1 |
|----------|----|---------|-----------------|-------|
| Yes | 0.05 | 1.0 | 90.5 | 59.7 |
| Yes | 0.1 | 4.5 | 87.8 | 64.4 |
| Yes | 0.5 | 8.2 | 68.2 | 60.7 |
| Yes | 1 | 8.3 | 59.1 | 58.0 |
| No | 10 | 0.5 | 91.7 | 57.2 |
| No | 100 | 0.5 | 92.1 | 57.0 |

---

### 5.2 Larger Batch Sizes and Longer Training / 更大批量与更长训练

<a id="S024"></a>
**Source:** p.7 S024 — §5.2

**Original:** Figure 9 shows the impact of batch size when models are trained for different numbers of epochs. We find that, when the number of training epochs is small (e.g. 100 epochs), larger batch sizes have a significant advantage over the smaller ones. With more training steps/epochs, the gaps between different batch sizes decrease or disappear, provided the batches are randomly resampled. In contrast to supervised learning (Goyal et al., 2017), in contrastive learning, larger batch sizes provide more negative examples, facilitating convergence (i.e. taking fewer epochs and steps for a given accuracy). Training longer also provides more negative examples, improving the results.

**中文:** 图 9 显示了模型训练不同 epoch 数时批量大小的影响。我们发现，当训练 epoch 数较小时（例如 100 epochs），较大的批量大小相比小批量具有显著优势。随着更多的训练步数/epochs，不同批量大小之间的差距减小或消失，前提是批量被随机重采样。与监督学习（Goyal et al., 2017）相比，在对比学习中，更大的批量大小提供了更多的负样本，有助于收敛（即以更少的 epochs 和步数达到给定准确率）。更长的训练也提供了更多的负样本，改善了结果。

---

<a id="F009"></a>
**Figure 9** (p.7) — *Linear evaluation models (ResNet-50) trained with different batch size and epochs. Each bar is a single run from scratch.*

**图 9:** 使用不同批量大小和 epochs 训练的线性评估模型（ResNet-50）。每个柱状条代表一次从头开始的运行。

---

## 6. Comparison with State-of-the-art / 与前沿方法的比较 <a id="sec6"></a>

<a id="S025"></a>
**Source:** p.7 S025 — §6 intro

**Original:** In this subsection, similar to Kolesnikov et al. (2019); He et al. (2019), we use ResNet-50 in 3 different hidden layer widths (width multipliers of 1×, 2×, and 4×). For better convergence, our models here are trained for 1000 epochs.

**中文:** 在本节中，类似 Kolesnikov et al. (2019); He et al. (2019)，我们使用 3 种不同隐藏层宽度的 ResNet-50（宽度倍数为 1×、2× 和 4×）。为获得更好的收敛性，我们的模型在此训练 1000 个 epochs。

---

<a id="T006"></a>
**Table 6** (p.7) — *ImageNet accuracies of linear classifiers trained on representations learned with different self-supervised methods.*

**表 6:** 在不同自监督方法学到的表征上训练的线性分类器的 ImageNet 准确率。

| Method | Architecture | Param (M) | Top 1 | Top 5 |
|--------|-------------|-----------|-------|-------|
| Local Agg. | ResNet-50 | 24 | 60.2 | - |
| MoCo | ResNet-50 | 24 | 60.6 | - |
| PIRL | ResNet-50 | 24 | 63.6 | - |
| CPC v2 | ResNet-50 | 24 | 63.8 | 85.3 |
| **SimCLR (ours)** | **ResNet-50** | **24** | **69.3** | **89.0** |
| AMDIM | Custom-ResNet | 626 | 68.1 | - |
| CMC | ResNet-50 (2×) | 188 | 68.4 | 88.2 |
| MoCo | ResNet-50 (4×) | 375 | 68.6 | - |
| CPC v2 | ResNet-161 (*) | 305 | 71.5 | 90.1 |
| **SimCLR (ours)** | **ResNet-50 (2×)** | **94** | **74.2** | **92.0** |
| **SimCLR (ours)** | **ResNet-50 (4×)** | **375** | **76.5** | **93.2** |

---

<a id="S026"></a>
**Source:** p.7 S026 — §6 linear evaluation finding

**Original:** The best result obtained with our ResNet-50 (4×) can match the supervised pretrained ResNet-50.

**中文:** 使用我们的 ResNet-50 (4×) 获得的最佳结果可以匹配有监督预训练的 ResNet-50。

---

<a id="T007"></a>
**Table 7** (p.7) — *ImageNet accuracy of models trained with few labels.*

**表 7:** 使用少量标签训练的模型的 ImageNet 准确率。

| Method | Architecture | 1% Top 5 | 10% Top 5 |
|--------|-------------|----------|-----------|
| Supervised baseline | ResNet-50 | 48.4 | 80.4 |
| InstDisc | ResNet-50 | 39.2 | 77.4 |
| BigBiGAN | RevNet-50 (4×) | 55.2 | 78.8 |
| PIRL | ResNet-50 | 57.2 | 83.8 |
| CPC v2 | ResNet-161(*) | 77.9 | 91.2 |
| **SimCLR (ours)** | **ResNet-50** | **75.5** | **87.8** |
| **SimCLR (ours)** | **ResNet-50 (2×)** | **83.0** | **91.2** |
| **SimCLR (ours)** | **ResNet-50 (4×)** | **85.8** | **92.6** |

---

<a id="S027"></a>
**Source:** p.8 S027 — §6 semi-supervised finding

**Original:** Interestingly, fine-tuning our pretrained ResNet-50 (2×, 4×) on full ImageNet are also significantly better than training from scratch (up to 2%, see Appendix B.2).

**中文:** 有趣的是，在完整的 ImageNet 上微调我们预训练的 ResNet-50 (2×, 4×) 也显著优于从头开始训练（提升高达 2%，见附录 B.2）。

---

<a id="T008"></a>
**Table 8** (p.8) — *Comparison of transfer learning performance of our self-supervised approach with supervised baselines across 12 natural image classification datasets, for ResNet-50 (4×) models pretrained on ImageNet.*

**表 8:** 我们的自监督方法与监督基线在 12 个自然图像分类数据集上的迁移学习性能比较，使用在 ImageNet 上预训练的 ResNet-50 (4×) 模型。

---

<a id="S028"></a>
**Source:** p.8 S028 — §6 transfer learning finding

**Original:** When fine-tuned, our self-supervised model significantly outperforms the supervised baseline on 5 datasets, whereas the supervised baseline is superior on only 2 (i.e. Pets and Flowers). On the remaining 5 datasets, the models are statistically tied.

**中文:** 在微调时，我们的自监督模型在 5 个数据集上显著优于监督基线，而监督基线仅在 2 个数据集上（Pets 和 Flowers）领先。在其余 5 个数据集上，模型在统计上持平。

---

## 7. Related Work / 相关工作 <a id="sec7"></a>

<a id="S029"></a>
**Source:** pp.8–9 S029 — §7 intro

**Original:** The idea of making representations of an image agree with each other under small transformations dates back to Becker & Hinton (1992). We extend it by leveraging recent advances in data augmentation, network architecture and contrastive loss. A similar consistency idea, but for class label prediction, has been explored in other contexts such as semi-supervised learning (Xie et al., 2019; Berthelot et al., 2019).

**中文:** 使图像的表征在小变换下彼此一致的想法可以追溯到 Becker & Hinton (1992)。我们通过利用数据增强、网络架构和对比损失方面的最新进展来扩展它。类似的——但针对于类标签预测的——一致性思想已在其他上下文中被探索，如半监督学习（Xie et al., 2019; Berthelot et al., 2019）。

---

<a id="S030"></a>
**Source:** p.8 S030 — §7 — Handcrafted pretext tasks

**Original:** **Handcrafted pretext tasks.** The recent renaissance of self-supervised learning began with artificially designed pretext tasks, such as relative patch prediction (Doersch et al., 2015), solving jigsaw puzzles (Noroozi & Favaro, 2016), colorization (Zhang et al., 2016) and rotation prediction (Gidaris et al., 2018; Chen et al., 2019). Although good results can be obtained with bigger networks and longer training (Kolesnikov et al., 2019), these pretext tasks rely on somewhat ad-hoc heuristics, which limits the generality of learned representations.

**中文:** **手工设计的前置任务。** 自监督学习最近的复兴始于人工设计的前置任务，如相对补丁预测（Doersch et al., 2015）、解拼图（Noroozi & Favaro, 2016）、着色（Zhang et al., 2016）和旋转预测（Gidaris et al., 2018; Chen et al., 2019）。虽然通过更大的网络和更长的训练可以获得良好结果（Kolesnikov et al., 2019），但这些前置任务依赖于某种特定的启发式方法，限制了所学表征的泛化性。

---

<a id="S031"></a>
**Source:** p.8 S031 — §7 — Contrastive visual representation learning

**Original:** **Contrastive visual representation learning.** Dating back to Hadsell et al. (2006), these approaches learn representations by contrasting positive pairs against negative pairs. Along these lines, Dosovitskiy et al. (2014) proposes to treat each instance as a class represented by a feature vector (in a parametric form). Wu et al. (2018) proposes to use a memory bank to store the instance class representation vector, an approach adopted and extended in several recent papers (Zhuang et al., 2019; Tian et al., 2019; He et al., 2019; Misra & van der Maaten, 2019). Other work explores the use of in-batch samples for negative sampling instead of a memory bank (Doersch & Zisserman, 2017; Ye et al., 2019; Ji et al., 2019).

**中文:** **对比视觉表征学习。** 追溯到 Hadsell et al. (2006)，这些方法通过将正对与负对进行对比来学习表征。沿此思路，Dosovitskiy et al. (2014) 提出将每个实例视为由一个特征向量（以参数化形式）表示的类。Wu et al. (2018) 提出使用记忆库来存储实例类表征向量，该方法在最近的几篇论文中被采纳和扩展（Zhuang et al., 2019; Tian et al., 2019; He et al., 2019; Misra & van der Maaten, 2019）。其他工作探索使用批次内样本进行负采样而非记忆库（Doersch & Zisserman, 2017; Ye et al., 2019; Ji et al., 2019）。

---

<a id="S032"></a>
**Source:** p.9 S032 — §7 — uniqueness of SimCLR

**Original:** We note that almost all individual components of our framework have appeared in previous work, although the specific instantiations may be different. The superiority of our framework relative to previous work is not explained by any single design choice, but by their composition. We provide a comprehensive comparison of our design choices with those of previous work in Appendix C.

**中文:** 我们注意到，我们框架的几乎所有独立组件都已在先前工作中出现，尽管具体的实例化可能不同。我们框架相对于先前工作的优越性不能由任何单一设计选择来解释，而是由它们的组合来解释。我们在附录 C 中提供了我们的设计选择与先前工作的全面比较。

---

## 8. Conclusion / 结论 <a id="sec8"></a>

<a id="S033"></a>
**Source:** p.9 S033 — §8

**Original:** In this work, we present a simple framework and its instantiation for contrastive visual representation learning. We carefully study its components, and show the effects of different design choices. By combining our findings, we improve considerably over previous methods for self-supervised, semi-supervised, and transfer learning.

Our approach differs from standard supervised learning on ImageNet only in the choice of data augmentation, the use of a nonlinear head at the end of the network, and the loss function. The strength of this simple framework suggests that, despite a recent surge in interest, self-supervised learning remains undervalued.

**中文:** 在本文中，我们提出了一个用于对比视觉表征学习的简单框架及其实现。我们仔细研究了其各个组件，并展示了不同设计选择的效果。通过结合我们的发现，我们在自监督、半监督和迁移学习方面大幅超越了先前的方法。

我们的方法仅在数据增强的选择、网络末端非线性投影头的使用以及损失函数方面与 ImageNet 上的标准监督学习不同。这个简单框架的强大之处表明，尽管最近兴趣激增，自监督学习仍然被低估。

---

## Appendices / 附录 <a id="appendices"></a>

<a id="S034"></a>
**Source:** pp.12–20 S034 — Appendices overview

**Original (summary):**

**Appendix A** (p.12) — Data Augmentation Details: Provides pseudocode (TensorFlow and PyTorch) for the three augmentations: random crop and resize to 224×224, color distortion (color jittering + color dropping), and Gaussian blur.

**Appendix B** (pp.13–19) — Additional Experimental Results: Includes B.1 Batch Size and Training Steps (square root LR scaling), B.2 Broader augmentations (Sobel filtering, equalize, solarize, motion blur), B.3 Effects of Longer Training for Supervised Models, B.4 Understanding The Non-Linear Projection Head (eigenvalue analysis, t-SNE), B.5 Semi-supervised Fine-tuning, B.6 Linear Evaluation procedure, B.7 Correlation Between Linear Evaluation and Fine-Tuning, B.8 Transfer Learning (full methods and standard ResNet results), B.9 CIFAR-10 experiments, B.10 Tuning For Other Loss Functions.

**Appendix C** (pp.19–20) — Further Comparison to Related Methods: Detailed comparison with AMDIM, CPC, InstDisc/MoCo/PIRL, and CMC.

**中文（摘要）:**

**附录 A**（第 12 页）— 数据增强细节：提供了三种增强的伪代码（TensorFlow 和 PyTorch）：随机裁剪并调整大小至 224×224、颜色失真（颜色抖动 + 颜色丢弃）和高斯模糊。

**附录 B**（第 13–19 页）— 额外实验结果：包括 B.1 批量大小与训练步数（平方根学习率缩放）、B.2 更广泛的数据增强组合（Sobel 滤波、均衡化、曝光过度、运动模糊）、B.3 监督模型更长训练的效果、B.4 理解非线性投影头（特征值分析、t-SNE）、B.5 半监督微调、B.6 线性评估流程、B.7 线性评估与微调的相关性、B.8 迁移学习（完整方法和标准 ResNet 结果）、B.9 CIFAR-10 实验、B.10 其他损失函数的调参。

**附录 C**（第 19–20 页）— 与相关方法的进一步比较：与 AMDIM、CPC、InstDisc/MoCo/PIRL 和 CMC 的详细对比。

---

<a id="T009"></a>
**Table C.1** (p.19) — *A high-level comparison of design choices and training setup (for best result on ImageNet) for each method.*

**表 C.1:** 各方法的设计选择和训练设置（ImageNet 上最佳结果）的高层对比。

| Model | Data Aug. | Base Encoder | Projection Head | Loss | Batch Size | Epochs |
|-------|----------|-------------|----------------|------|-----------|--------|
| CPC v2 | Custom | ResNet-161 (modified) | PixelCNN | Xent | 512 | ~200 |
| AMDIM | Fast AutoAug. | Custom ResNet | Non-linear MLP | Xent w/ clip,reg | 1008 | 150 |
| CMC | Fast AutoAug. | ResNet-50 (2×, L+ab) | Linear layer | Xent w/ ℓ₂,τ | 156* | 280 |
| MoCo | Crop+color | ResNet-50 (4×) | Linear layer | Xent w/ ℓ₂,τ | 256* | 200 |
| PIRL | Crop+color | ResNet-50 (2×) | Linear layer | Xent w/ ℓ₂,τ | 1024* | 800 |
| **SimCLR** | **Crop+color+blur** | **ResNet-50 (4×)** | **Non-linear MLP** | **Xent w/ ℓ₂,τ** | **4096** | **1000** |

(* 使用记忆库)

---

## 术语表 / Terminology <a id="terminology"></a>

| 英文术语 | 中文翻译 | 说明 |
|---------|---------|------|
| contrastive learning | 对比学习 | 通过拉近正对、推远负对来学习表征 |
| representation | 表征 | 数据的向量化表示 |
| data augmentation | 数据增强 | 对数据施加随机变换以构造正样本对 |
| pretext task | 前置任务 | 为自监督学习设计的代理任务 |
| projection head | 投影头 | 将表征映射到对比损失空间的 MLP |
| NT-Xent | 归一化温度缩放交叉熵 | Normalized Temperature-scaled Cross Entropy |
| memory bank | 记忆库 | 存储历史表征向量的外部存储 |
| temperature parameter | 温度参数 | 控制 softmax 分布锐利度的超参数 |
| linear evaluation | 线性评估 | 冻结局表征训练线性分类器的评估协议 |
| semi-supervised learning | 半监督学习 | 使用少量标签和大量无标签数据的学习 |
| transfer learning | 迁移学习 | 将预训练模型应用于不同任务/数据集 |
| cosine similarity | 余弦相似度 | ℓ₂ 归一化向量间的点积 |
| batch size | 批量大小 | 每次训练的样本数量 |
| LARS | LARS 优化器 | Layer-wise Adaptive Rate Scaling 优化器 |
| global BN | 全局批归一化 | 跨设备聚合 BN 统计量 |
| hard negative | 难负样本 | 与正样本高度相似的负样本 |

---

## 阅读提示 / Critical Reading Notes <a id="reading-notes"></a>

1. **核心贡献定位：** 本文的贡献不是提出全新的组件，而是通过系统消融实验，揭示了各组件为何有效以及如何组合。四个关键发现——数据增强组合、非线性投影头、归一化嵌入加温度参数、大批量训练——各自在先前工作中都有雏形，但 SimCLR 首次将它们集成并系统验证。

2. **与 MoCo 的对比：** SimCLR 不使用记忆库，而是依赖大批量（4096–8192）提供足够负样本。这需要大量 TPU 计算资源（128 TPU v3 cores）。MoCo 通过动量编码器+队列实现相似效果但计算需求更低，是资源受限场景的更优选择。

3. **颜色失真的关键性：** 论文揭示了一个重要的捷径学习问题——当仅使用裁剪时，神经网络可以利用颜色分布匹配正对，而不学习语义特征。颜色失真通过强制模型忽略颜色线索来防止这种捷径。这解释了为何无监督对比学习比监督学习更需要强增强。

4. **投影头的作用机理：** 非线性投影头使得 h 可以保留更多下游任务有用的信息（如颜色、朝向），而 z = g(h) 被强制对数据增强不变。扔掉投影头的做法类似于"信息瓶颈"——让中间表征保留更多信息。

5. **实验注意事项：**
   - 默认 100 epoch 的训练并未达到最优性能，只是为了快速消融；最佳结果需要 1000 epoch
   - 平方根学习率缩放优于线性缩放（尤其对小批量）
   - 当使用多个加速器时必须使用全局 BN，否则模型会利用跨设备的信息泄漏

6. **已知局限：** 本文关注 ImageNet 分类任务。对密集预测任务（检测、分割）的表现未评估。对视频、医学影像等其他领域的泛化性需要进一步验证。

7. **潜在改进方向：**
   - 更广泛的数据增强组合可以进一步提升性能（见附录 B.2）
   - 移除 Gaussian blur 会降低 1.3% 的线性评估准确率
   - 在 CIFAR-10 上最大批量（4096）反而略微降低性能，暗示最优批量大小与数据集规模有关

---

*This bilingual reader was generated from arXiv:2002.05709v3. Figures are noted by location; actual images should be extracted from the original PDF. See `source_map.json` for the complete block-ID index and `translation_notes.md` for translation and extraction notes.*
