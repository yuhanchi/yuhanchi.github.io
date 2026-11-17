---
title: "只是套了个 Flow？Sundial你！"
date: "2026-11-17"
slug: "timeflow-sundial"
description: "拆解 Sundial 的 TimeFlow 输出头：为什么它的 Flow Matching 设计很平庸，以及为什么靠多次采样拉低 MASE 和 WQL 算不上真正的校准。"
category: "tech"
tags: ["AI", "time-series"]
math: true
preview_image: "/assets/img/posts/sundial-fig2.png"
preview_alt: "Sundial 整体架构（Figure 2）：连续 Patch 分词、因果 Transformer 底座以及 TimeFlow Loss 输出头。"
---
之前在小红书发完 [Chronos-2]({% post_url 2026-11-06-time-series-as-tokens %}) 的笔记后，有读者在评论区推荐了 [Sundial](https://arxiv.org/abs/2502.00816)（*Sundial: A Family of Highly Capable Time Series Foundation Models*，ICML 2025），用来回应我[上篇文章最后留下的那个问题]({% post_url 2026-11-06-time-series-as-tokens %}#6-possibly-open-questions)：能不能像分布强化学习那样，用 Flow Matching 头代替分位数头来建模预测分布？

乍一看，Sundial 的卖点很厉害：一个原生的、直接input连续值、Decoder-only 的时间序列基础模型，在万亿级时间点（`TimeBench`）上预训练，用基于 Flow Matching 的 **TimeFlow Loss** 替代离散分词和参数化先验分布。

<figure>
  <a href="{{ '/assets/img/posts/sundial-fig2.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/sundial-fig2.png' | relative_url }}" alt="Sundial 整体架构（Figure 2）：连续 Patch 分词、Pre-LN 因果 Transformer 底座与 TimeFlow Loss 输出头。" width="2057" height="596" loading="lazy"></a>
  <figcaption>Sundial 原文 Figure 2：连续 Patch 分词、带 RoPE 的 Pre-LN 因果 Transformer，以及 TimeFlow Loss 输出头。</figcaption>
</figure>

因为它的 Transformer 底座和 Patch 分词基本都是常规操作，整篇论文**唯一真正算得上新方法的地方**，就是这个 **TimeFlow** 输出头。下面我先简短交代一下它怎么工作的，再说说为什么我觉得这套设计和它所谓的“测试时校准（Test-Time Calibration）”站不住脚。

---

## 1. 唯一的新东西：TimeFlow 输出头

Sundial 的前半段非常标准：单变量输入先做两阶段实例归一化（`Re-Norm`），按长度 $$P = 16$$ 切成连续 Patch 并做嵌入，再送进一个带 RoPE、FlashAttention 和 KV Cache 的 Pre-LN 因果 Transformer，拿到每个 token 的隐藏表示 $$\mathbf{h}_i \in \mathbb{R}^D$$。

它真正改动的是最后一层 **TimeFlow Loss**（§4.1.3，式 6–9）：不让 $$\mathbf{h}_i$$ 直接回归未来预测向量 $$\hat{\mathbf{y}}_i \in \mathbb{R}^F$$ 或分位数，而是接一个 3 到 6 层的小 MLP（`FM-Net`），通过 AdaLN 把 $$\mathbf{h}_i$$ 当作条件喂进去，训练一个标准的条件 Flow Matching（CFM）：采样高斯噪声 $$\mathbf{y}_i^{(0)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_F)$$ 和时间 $$t \sim \mathcal{U}[0, 1]$$，在噪声和真实未来轨迹 $$\mathbf{y}_i \in \mathbb{R}^F$$ 之间做线性插值 $$\mathbf{y}_i^{(t)} = t \mathbf{y}_i + (1 - t)\mathbf{y}_i^{(0)}$$，回归常速度场：

$$
\mathcal{L}(\theta, \mathbf{h}_i) = \mathbb{E}_{t, \mathbf{y}_i^{(0)}, \mathbf{y}_i} \left[ \left\| \mathrm{FM\text{-}Net}\!\left(\mathbf{y}_i^{(t)}, t, \mathbf{h}_i\right) - \left(\mathbf{y}_i - \mathbf{y}_i^{(0)}\right) \right\|_2^2 \right].
$$

推理时固定 $$\mathbf{h}_i$$，从 $$\mathcal{N}(\mathbf{0}, \mathbf{I}_F)$$ 采一个初始噪声，跑 $$K = 50$$ 步均匀欧拉积分（$$\Delta t = 1/K$$）得到一条候选轨迹。为了做点预测和概率预测，模型换不同的初始噪声**重复采样 $$M$$ 次**（FEV 上 $$M = 20$$，GIFT-Eval 上 $$M = 100$$），然后对这 $$M$$ 条轨迹取中位数和经验分位数。

---

## 2. 为什么说这个 Flow Matching 头很 Trivial

把“生成式建模”这层包装剥掉之后，其实有四个硬伤非常明显。

### 2.1 教科书式 CFM + 照搬 MAR，几乎没有领域创新

数学上，式 (6)–(8) 就是最基础的线性插值条件 Flow Matching。结构上，在自回归 Transformer 上面挂一个带 AdaLN 的小 MLP 扩散/流匹配头，本质上就是把图像生成里的 **MAR**（*Autoregressive Image Generation without Vector Quantization*，[Li et al., 2024](https://arxiv.org/abs/2406.11838)）原封不动搬到了时序上——原文自己也引用了这篇。

哪怕预测长度拉到 $$F = 720$$，它的 `FM-Net` 也只是一个 3–6 层的普通 MLP（Table 5），里面没有任何针对时间序列的归纳偏置。把它叫做什么 “TimeFlow”，多少有点把通用连续输出头过度命名了。

### 2.2 论文自己的消融实验已经说明：真正起作用的是底座和 1T 数据

如果 TimeFlow 真的有那么关键，把它换成最土的 MSE 头，效果应该掉一大截才对。但看原文 **Table 3**（相同底座和 1T `TimeBench` 预训练，在 TSLib 上测零样本 MSE）：

<figure>
  <a href="{{ '/assets/img/posts/sundial-table3.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/sundial-table3.png' | relative_url }}" alt="Sundial 原文 Table 3：TimeFlow、Diffusion 与普通 MSE 在 ETTm1、ETTm2、ETTh1、ETTh2、ECL、Weather 上的零样本 MSE 对比。" width="1010" height="420" loading="lazy"></a>
  <figcaption>Sundial 原文 Table 3：同底座同数据规模下，TimeFlow、Diffusion 与纯 MSE 的零样本 MSE 对比。</figcaption>
</figure>

- 六个数据集平均 MSE：`TimeFlow` 是 **0.290**，纯 `MSE` 是 **0.296**，差了 0.006。
- 在 `ETTh1`（`0.404` vs `0.411`）和 `Weather`（`0.231` vs `0.234`）上，**不带 Flow Matching 的纯 `MSE` 反而赢了**。

更耐人寻味的是 **Table 7**（同样六个数据集加上 GIFT-Eval 的零样本 CRPS）：

<figure>
  <a href="{{ '/assets/img/posts/sundial-table7.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/sundial-table7.png' | relative_url }}" alt="Sundial 原文 Table 7：TimeFlow Loss、Diffusion Loss 与 MSE Loss 在六个 TSLib 数据集及 GIFT-Eval 上的零样本 CRPS 对比。" width="2054" height="372" loading="lazy"></a>
  <figcaption>Sundial 原文 Table 7：TimeFlow Loss、Diffusion Loss 与 MSE Loss 的零样本 CRPS 对比。</figcaption>
</figure>

- 纯 `MSE`（确定性预测，完全没有分布建模，此时 CRPS 退化成 MAE）在 TSLib 六个数据集上的得分是 `0.0063, 0.0040, 0.0058, 0.0032, 0.0080, 0.0023`，跟 `TimeFlow` 的 `0.0059, 0.0037, 0.0057, 0.0029, 0.0082, 0.0021` 几乎贴在一起，甚至在 `ECL` 上纯 `MSE` 还要更低。

所以我觉得很清楚：Sundial 能打，靠的是 **1 万亿时间点的 `TimeBench` 数据 + 连续 Patch + RoPE/Pre-LN 底座**，跟这个 Flow Matching 输出头关系不大。

### 2.3 “多次采样降低 MASE 和 WQL”是统计学常识，不是什么 Calibration

论文 §5.4（Figure 7）扫了一遍采样条数 $$M$$，看到 MASE 和 WQL 往下掉，就写了这么一段话：

> *"Generative modeling facilitates the flexibility to calibrate the final prediction during inference. Based on the median-based forecasting strategy, i.e., starting from multiple noise of a standard Gaussian and calculating the median of raw predictions, there are two configurations to calibrate final predictions..."*

这种说法其实站不住脚。设 $$\hat{\mathbf{y}}^{(1)}, \dots, \hat{\mathbf{y}}^{(M)} \stackrel{\text{i.i.d.}}{\sim} p_\theta(\mathbf{y} \mid \mathbf{x})$$ 是模型采出的 $$M$$ 条轨迹，从数学上看：

1. **MASE 为什么降**：对任意凸损失函数 $$\ell(\cdot, \mathbf{y})$$（比如 MASE 里的 $$L_1$$ 误差），用 $$M$$ 个独立样本的均值 $$\bar{\mathbf{y}}_M$$ 或逐点中位数 $$\tilde{\mathbf{y}}_M$$ 来代替单次采样 $$\hat{\mathbf{y}}^{(1)}$$，本来就会把蒙特卡洛方差以 $$\mathcal{O}(1/M)$$ 的速度压下去（由 Jensen 不等式，$$\mathbb{E}[\ell(\bar{\mathbf{y}}_M, \mathbf{y})] \le \mathbb{E}[\ell(\hat{\mathbf{y}}^{(1)}, \mathbf{y})]$$；样本中位数的渐近方差同样按 $$1/(4M f_\theta(\mathrm{med})^2)$$ 衰减）。哪怕你拿一个确定性的 MSE 模型，故意在输出上加点高斯噪声再采 $$M$$ 次求平均，画出来的曲线也一样单调下降。
2. **WQL 为什么降**：隐式流模型没法直接输出分位数，只能在每个时间步 $$t$$ 上先构造经验分布函数 $$\hat{F}_{M,t}(u) = \frac{1}{M}\sum_{m=1}^M \mathbf{1}\{\hat{y}_t^{(m)} \le u\}$$，再取经验分位数 $$\hat{Q}_{q,t}^{(M)} = \hat{F}_{M,t}^{-1}(q)$$ 去算 pinball loss $$\rho_q(y, u) = \max\big(q(y - u), (q - 1)(y - u)\big)$$。当 $$M = 1$$ 时，所有分位数全部坍缩成同一条采样轨迹（$$\hat{Q}_{0.1,t}^{(1)} = \dots = \hat{Q}_{0.9,t}^{(1)} = \hat{y}_t^{(1)}$$），预测区间宽度为 0，WQL 自然很难看。把 $$M$$ 从 1 拉到 20 或 100，只是在以 $$\mathcal{O}(M^{-1/2})$$ 的速率减小经验分位数 $$\hat{Q}_{q,t}^{(M)}$$ 相对于模型**自己已经学到的** $$F_{\theta,t}^{-1}(q)$$ 的有限样本抽样误差。这根本不是把 $$F_\theta$$ 校准到真实分布 $$F_{\text{true}}$$——如果模型本身学歪了，采一万次也只是更精确地算出了歪掉的分位数。

说白了，**只要是个靠采样做概率预测的模型——包括 [Chronos](https://arxiv.org/abs/2403.07815)——扫 $$M$$ 的时候都会出现一模一样的下降曲线**。而且在 FEV 榜单上，大家同样采样 $$M = 20$$ 次（Figure 4），Chronos-1 的 MASE 和 WQL 依然排在 Sundial 前面（Sundial 在那里的真正优势是 Patch 带来的 35× 推理加速，而不是分布建模更强）。至于 [Chronos-2]({% post_url 2026-11-06-time-series-as-tokens %})，人家直接用 Quantile Regression 头一次前向就吐出所有分位数，相当于直接站在 $$M \to \infty$$ 的无采样噪声极限上，根本不用跑 $$M \times K = 100 \times 50$$ 步 ODE。

### 2.4 缺失的消融与指标错位

最后还有两个明显的问题：

- **忘记和 Quantile Head 对比**：§5.3 里，Sundial 只拿 `TimeFlow` 去和 `MSE`（不能输出区间）以及 `DDPM Diffusion`（50 步采样本来就吃亏）比，却**没有**在同一个底座和 `TimeBench` 数据上，对比一个最直接的 **Multi-Quantile MLP Head（Pinball Loss）** 或参数化混合分布头。
- **一个联合生成器，却在考边缘考试题**：按理说，用 Flow Matching 一次生成整条向量 $$\hat{\mathbf{y}} \in \mathbb{R}^F$$，相比分位数头唯一真正的优势在于能学到跨时间步的**联合分布** $$p(\mathbf{y}_{t+1:t+F} \mid \mathbf{x}_{1:t})$$，而不是每个时间步各自独立的边缘分布。但整篇论文测的所有指标——MSE、MAE、MASE、WQL、逐点 CRPS——**全都是边缘指标**，没有任何一个指标去评估整条生成轨迹的联合质量（比如 Energy Score 或时间自相关结构）。这相当于自废武功，实在遗憾。

总结下来，Sundial 在工程上的工作——清洗出 `TimeBench` 并把连续 Patch 自回归 Transformer 训稳训大——是扎实的；但论文主打的 `TimeFlow` 头和“test-time calibration”叙事，包装的成分还是远大于实质。

---

## 3. 一个开放问题

到底什么样的建模方式，才算真正用上了生成式模型（它确实可能具备的）表达能力？如果推理到头来还是靠反复采样去凑逐点的中位数和分位数，那其实并没有发挥出生成模型真正擅长的东西。这个问题比较复杂，我要好好想想。

---

## References

- Liu, Y., Qin, G., Shi, Z., et al. (2025). [Sundial: A Family of Highly Capable Time Series Foundation Models](https://arxiv.org/abs/2502.00816). *ICML*.
- Ansari, A. F., et al. (2025). [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821).
- Ansari, A. F., Stella, L., Turkmen, C., et al. (2024). [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815). *TMLR*.
- Li, T., Tian, Y., Li, H., Deng, M., & He, K. (2024). [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838). *NeurIPS*. (MAR)

<details>
<summary>Cite this post</summary>
<p>If you find this post useful in your work, you can cite it as:</p>
<blockquote>
<p>Chi, Yuhan. “Just a Flow? Notes on Sundial.” November 17, 2026. <a href="https://yuhanchi.github.io/writing/timeflow-sundial/">https://yuhanchi.github.io/writing/timeflow-sundial/</a></p>
</blockquote>
<p>This post discusses and critiques the Sundial paper. To cite Sundial itself, please cite <a href="https://arxiv.org/abs/2502.00816">Liu et al., 2025</a>. To cite this post, use the following BibTeX entry:</p>
{% raw %}<pre><code class="language-bibtex">@misc{chi2026justaflow,
  author = {Chi, Yuhan},
  title = {{Just a Flow? Notes on Sundial}},
  year = {2026},
  month = nov,
  howpublished = {Blog post},
  url = {https://yuhanchi.github.io/writing/timeflow-sundial/}
}</code></pre>{% endraw %}
</details>
