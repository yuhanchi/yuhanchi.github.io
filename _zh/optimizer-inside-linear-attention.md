---
title: "线性注意力里是何种优化器？尺度对称性、写入与遗忘"
date: "2026-11-22"
slug: "optimizer-inside-linear-attention"
description: "归一化的读出不在乎状态的整体尺度，但每次的写入在乎。从这个差别出发，区分尺度校准与遗忘，理解 DeltaNet、Gated DeltaNet 和齐次更新。"
category: "tech"
tags: ["AI", "linear-attention", "optimization"]
math: true
published: true
preview_image: "/assets/img/posts/optimizer-inside-linear-attention/linear-attention.png"
preview_alt: "现代 Linear Attention（Gated Delta Rule）模块结构图：q、k 经 L2 归一化，状态读出经过 RMSNorm 与输出门控。"
---

我这几天读了些 Linear Attention 和 Pretrain Optimizer 的工作，让我开始想一个问题：既然 Linear Attention 可以看作 Test-Time Training，那它内部的优化器应该怎样处理尺度？更进一步地，我们这个 Linear Attention 是否具有某种同样好的性质，比如尺度或者对称？


## 1. 更新的尺度？

<figure>
  <a href="{{ '/assets/img/posts/optimizer-inside-linear-attention/linear-attention.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/optimizer-inside-linear-attention/linear-attention.png' | relative_url }}" alt="现代 Linear Attention（Gated Delta Rule）模块结构图：q、k 经卷积、SiLU 与 L2 归一化，与 v、α、β 一同进入 Gated Delta Rule，输出经过 Zero-Centered RMSNorm 与输出门控后再做线性投影。" width="613" height="576" loading="lazy"></a>
  <figcaption>现代 Linear Attention（Gated Delta Rule）模块结构：\(q, k\) 经 L2 归一化，状态读出后先经过 RMSNorm，再与输出门控相乘。架构示意图截取自 <a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking">Qwen3-Next-80B-A3B-Thinking</a> 模型结构图。</figcaption>
</figure>

仔细观察上图。状态是一个矩阵 $$S_t\in\mathbb R^{d_v\times d_k}$$。每一步接收一组键值 $$(k_t,v_t)$$，更新状态，再用查询 $$q_t$$ 读出：

$$
S_{t+1}=\alpha_t S_t(I-\beta_t k_tk_t^\top)+\gamma_t v_tk_t^\top,
\qquad
 y_t=\operatorname{RMSNorm}(S_{t+1}q_t).
\tag{1}
$$

先把 $$\alpha_t,\beta_t,\gamma_t$$ 看成三个独立系数，先不对它们做任何预设。但从一个基础语义上来说，我们有 $$\alpha_t$$ 用来控制旧状态的保留比例，$$\beta_t$$ 用以控制擦除强度，和 $$\gamma_t$$ 表示写入强度。把更新式展开：

$$
S_{t+1}
=
\underbrace{\alpha_t S_t}_{\text{旧状态的整体衰减}}
-
\underbrace{\alpha_t\beta_t(S_tk_t)k_t^\top}_{\text{沿当前键的擦除}}
+
\underbrace{\gamma_t v_tk_t^\top}_{\text{新写入}}.
\tag{2}
$$

如式 (2) 所言，我们称矩阵 $$S_t$$ 为状态，$$-\alpha_t\beta_t(S_tk_t)k_t^\top$$ 项为擦除，最后的 $$\gamma_t v_tk_t^\top$$ 项为写入；遗忘门 $$\alpha_t$$ 则作用在 $$S_t(I-\beta_t k_tk_t^\top)$$ 上。

现在把 $$S$$ 换成 $$\lambda S$$，其中 $$\lambda>0$$。不难注意到：实践中的读出满足

$$
\operatorname{RMSNorm}(\lambda Sq)
=
\operatorname{RMSNorm}(Sq).
\tag{3}
$$

所以读出具有尺度不变性。但从已有的更新递推式来看，状态矩阵的更新似乎并没有这种性质。

更具体地，我们会发现，式 (1) 中前两项似乎都会恰好随着 $$\lambda$$ 变化，但是最后的写入项不会。
严谨地写一下的话，就是记式 (1) 的更新算子为 $$\mathcal T$$，则

$$
\mathcal T(\lambda S)-\lambda\mathcal T(S)
=(1-\lambda)\gamma vk^\top.
\tag{4}
$$

因此，原本读出相同的两个状态，经过同一次更新后，读出通常就不再相同了。

关于 $$S$$，旧状态项是一次齐次的，新写入项是零次齐次的。直觉上就是说：**旧状态越大，同样一次写入越难改变它；旧状态越小，同样一次写入越容易改变它。** 状态的长度虽然不出现在当前输出里，却在控制下一步有多容易改变方向。

## 2. 两个 Goal？

沿着 [TTT](https://arxiv.org/abs/2407.04620) 和 [Test-Time Regression](https://arxiv.org/abs/2501.12352) 的视角，普通 DeltaNet 对应单步回归损失 $$\frac12\|Sk-v\|_2^2$$ 上的一步 SGD。但细想一下，写入端和读出端的优化目标其实并不一样。

对于写入对 $$(k,v)$$，我们希望 $$Sk$$ 尽可能贴近 $$v$$：

$$
\min_S \;\; \mathbb{E}\,\|Sk-v\|_2^2. \tag{Goal 1}
$$

记

$$
\Sigma=\mathbb E[kk^\top]\succ0,\qquad
C=\mathbb E[vk^\top],\qquad
P^*=C\Sigma^{-1}.
$$

而对于读出对 $$(q,o^\star)$$（这里 $$o$$ 既代表 *oracle* 又代表 *output*，小小的双关），因为 $$Sq$$ 紧接着就会经过 RMSNorm 把长度扔掉，我们真正关心的其实是输出**有没有指对方向**：

$$
\max_S \;\; \mathrm{sim}(Sq,\; o^\star). \tag{Goal 2}
$$

这里的“最大化相似度”可以写成余弦相似度 $$\max\cos\angle(Sq,o^\star)$$、归一化距离 $$\min\|\operatorname{RMSNorm}(Sq)-\operatorname{RMSNorm}(o^\star)\|_2^2$$，或者任何只依赖于正射线 $$\{a\,Sq : a>0\}$$ 的目标函数。

假设我们达成了 Goal 1，并且也认同 $$(q,o^\star)$$ 在某种意义上就是另一组 $$(k,v)$$（即理想读出满足 $$o^\star=P^*q$$），那么把 Goal 1 训好，自然会推导出 Goal 2。反过来呢？Goal 1 显得太强了：Goal 2 根本不在乎输出乘上任何正标量，直觉上似乎可以给每组键值分别乘上不同的系数。

真的能这样吗？取两个正交的键，设对应的值 $$v_1,v_2$$ 线性无关。如果状态满足

$$
Sk_1=2v_1,\qquad Sk_2=v_2,
\tag{5}
$$

单独查询 $$k_1$$ 或 $$k_2$$，输出方向都正确。但对于混合查询 $$q=(k_1+k_2)/\sqrt2$$，读出变成 $$2v_1+v_2$$ 的方向，而不是 $$v_1+v_2$$ 的方向。最后再归一化，也无法消除两部分之间的比例差异。

这对应一个简单的线性代数事实：

> **引理。** 设 $$\operatorname{rank}(M)\ge2$$。如果对所有 $$q\notin\ker M$$，$$Sq$$ 都是 $$Mq$$ 的正倍数，那么存在一个共同的 $$c>0$$，使得 $$S=cM$$。

<details markdown="1">
<summary>证明</summary>

写成 $$Sq=c(q)Mq$$。若 $$Mq_1,Mq_2$$ 线性无关，由

$$
S(q_1+q_2)=c(q_1)Mq_1+c(q_2)Mq_2
=c(q_1+q_2)(Mq_1+Mq_2)
$$

比较系数，可知这三个比例相同。若两个非零像向量共线，可以找第三个线性无关的像向量过渡，因而所有非零像都对应同一个 $$c$$。对 $$q_0\in\ker M$$，比较 $$q_1+q_0$$ 与 $$q_1$$，又得到 $$Sq_0=0$$，所以 $$S=cM$$。

</details>

受此启发，我们不难注意到，一旦要求 Goal 2 对**所有**查询都成立，逐查询的自由度就会立刻坍缩成唯一一个全局正尺度 $$c>0$$。因而，我们可以设计一个更加有趣的，或许也更加有潜力的目标 **Goal \***，其中的损失称为**尺度校准回归损失（Scale-Calibrated Regression，SCaR loss）**：

$$
\boxed{\ \min_{S,\; c>0}\; F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2\ }
\tag{Goal *}
$$

如果只看解集，Goal \* 其实很简单：令 $$P=cS$$，它就是普通最小二乘；固定任意 $$c>0$$，最优解都是 $$S=P^*/c$$，全部落在同一条射线 $$\{aP^* : a>0\}$$ 上。Goal \* 真正增加的是**几何**——它把读出层唯一不在乎的那个共同尺度 $$c$$ 显式写成了变量。

我们之后会看到，这个 Goal \* 是如何解决第 1 节和第 2 节的麻烦的。第 6 节还会说明，这个选择并不是任意的：在标量写入因子里，只有 Goal \* 给出的那个，才让总体方向不动点不带 ridge。

## 3. 把写入也一起缩放

式 (2) 的旧状态项已经满足

$$
\lambda S(I-\beta kk^\top)
=\lambda\big[S(I-\beta kk^\top)\big].
$$

缺的只是新写入。给它乘上一个满足 $$g(\lambda S)=\lambda g(S)$$ 的标量，就得到

$$
\boxed{
\mathcal T(S)=\alpha S(I-\beta kk^\top)+\gamma g(S)vk^\top,
\qquad
\mathcal T(\lambda S)=\lambda\mathcal T(S).
}
\tag{6}
$$

这叫正一次齐次，也就是这里需要的尺度等变性：状态放大多少，更新结果就放大多少。初态只差一个正倍数，后续状态也始终只差一个正倍数，因而每一步的归一化读出都相同。

现在，状态长度可以只作为一种表示约定，不再暗中控制后续写入的相对强度。

这也让标量门的作用变得清楚。只需要问两件事：**它只乘旧状态，还是连新写入一起乘？新写入会不会随状态一起缩放？**

| 标量乘在哪里？ | 写入大小不随 $$S$$ 缩放：$$vk^\top$$ | 写入随 $$S$$ 一起缩放：$$g(S)vk^\top$$ |
|---|---|---|
| **只乘旧状态**<br>$$\alpha S(I-\beta kk^\top)+\text{新写入}$$ | GDN 的遗忘门。旧状态相对新写入变小，同时影响状态长度与后续更新幅度；$$\alpha$$ 为常数时有 ridge 型偏差。 | 仍然降低旧状态相对新写入的权重，但效果不再依赖状态的整体长度。校准之后，仍可单独控制这种相对遗忘。 |
| **旧状态和新写入一起乘**<br>$$a\big[S(I-\beta kk^\top)+\text{新写入}\big]$$ | 当前读出不变，但改变了状态长度，因此会影响下一步写入的相对强度。整体校准和普通 DeltaNet 的球面投影都在这里。 | 只改变共同尺度。后续更新也随之缩放，所有归一化读出保持不变。 |

上面一行改变新旧内容之间的比例，下面一行改变整张状态的共同尺度。左边这一列里，共同尺度还会影响下一次更新；右边这一列里，这个影响被消除了。

性质最优良的或许会是右下角：如果更新已经一次齐次，那么每步末尾再乘一个正数，或者把状态投影到固定半径的球面上，都不会改变读出轨迹。证明只需归纳：若两条轨迹当前相差 $$\lambda$$ 倍，下一次更新后仍相差 $$\lambda$$ 倍，额外的归一化只会换一个比例。

注：这里说的遗忘，是整体调低旧状态相对于新写入的权重；至于沿当前键的擦除，则是针对单个方向的局部改写。

<details markdown="1">
<summary>另一种表达：衰减旧状态，等价于加大后续写入</summary>

还有一个等价的看法。对式 (1) 令 $$S_t=\kappa_tQ_t$$，取 $$\kappa_{t+1}=\alpha_t\kappa_t$$，就有

$$
Q_{t+1}=Q_t(I-\beta_tk_tk_t^\top)
+\frac{\gamma_t}{\kappa_{t+1}}v_tk_t^\top.
\tag{7}
$$

显式遗忘门消失了，效果变成后来写入的系数越来越大。若 $$\alpha_t<1$$，$$\kappa_{t+1}=\prod_{s\le t}\alpha_s$$ 会持续减小。于是，“不断衰减旧状态”和“不断加大后续写入”，表达的是同一条读出轨迹。这里的权重累乘了此前所有遗忘门的作用，靠单步的 $$\|v_t\|$$ 本身是替不了的。从数学直觉上来讲，这也一定会影响它的有效学习率（effective learning rate）。

</details>

表格区分了两种操作，但它们对读出究竟有什么影响？先看两个例子：一个有意衰减旧状态，另一个只是想固定状态长度，却都可能改变不同方向之间的比例。

## 4. 两种操作，怎样产生方向偏差？

为比较方向，以第 2 节的平稳回归解 $$P^*=C\Sigma^{-1}$$（即固定 $$c=1$$ 时普通回归 $$F(S,1)=\frac12\mathbb E\|Sk-v\|_2^2$$ 的最优解）为参照。下文谈到的偏差都以这个解为参照；真实序列的分布往往在变，未必需要一直保留这个静态解。

### GDN：低频方向受到更强的衰减

Gated DeltaNet 对应式 (1) 中的 $$\gamma=\beta$$：

$$
S_{t+1}=\alpha S_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top.
$$

在独立同分布样本和常数 $$\alpha,\beta$$ 下，均值不动点满足

$$
S_\star[(1-\alpha)I+\alpha\beta\Sigma]=\beta C,
$$

所以

$$
S_\star=\frac1\alpha C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=\frac{1-\alpha}{\alpha\beta}.
\tag{8}
$$

外面的 $$1/\alpha$$ 不影响归一化读出；里面的 ridge 项则通常会改变方向。

看一个两维例子：键以概率 $$p_1,p_2$$ 取 $$e_1,e_2$$，值与键相同，于是 $$\Sigma=C=\operatorname{diag}(p_1,p_2)$$，$$P^*=I$$。忽略共同尺度，不动点变成

$$
\operatorname{diag}\left(
\frac{p_1}{p_1+\lambda},\,
\frac{p_2}{p_2+\lambda}
\right).
\tag{9}
$$

取 $$p_1=0.8,p_2=0.2,\lambda=0.2$$，两个系数是 $$0.8$$ 和 $$0.5$$，混合查询 $$(e_1+e_2)/\sqrt2$$ 的输出就从 $$45^\circ$$ 偏到约 $$32^\circ$$。

低频键较少得到新的写入，却每步都在衰减，因此被压制得更多。这就是近因 ridge 偏差；若 $$\Sigma$$ 各向同性，ridge 只改变共同尺度。

GDN 与正则回归的关系在 [Test-Time Regression](https://arxiv.org/abs/2501.12352) 中已有明确推导，[MIRAS](https://arxiv.org/abs/2504.13173) 也把遗忘解释为保留正则化。这里进一步关心的是：哪些收缩会穿过归一化读出，表现为方向变化。

### 硬锁球面：尺度约定也可能制造同类偏差

普通 DeltaNet 每步更新后再投影到 $$\|S\|_F=R$$，并不能自动得到正确方向。对应的总体约束问题是

$$
\min_{\|S\|_F=R}\frac12\mathbb E\|Sk-v\|^2.
$$

拉格朗日驻点满足 $$S\Sigma-C+\lambda S=0$$，因此

$$
S_R=C(\Sigma+\lambda I)^{-1},\qquad \|S_R\|_F=R.
\tag{10}
$$

只需看 $$\lambda$$ 的符号。若 $$R<\|P^*\|_F$$，则 $$\lambda>0$$，低频方向被压制，和 GDN 一样（在上面的两维例子中取 $$R=0.5$$，混合读出偏到约 $$21^\circ$$）。若 $$R>\|P^*\|_F$$，则 $$\lambda<0$$，低频方向反而被放大。

两种操作产生了相似的方向偏差，表面上的原因却不同：GDN 有意降低旧内容的权重；球面投影只是想固定长度，却顺带改变了内容比例。（第 6 节会看到，它们背后其实是同一个机制。）

对球面投影而言，问题在于普通回归仍然拟合绝对长度。若想让长度只作为表示约定，就需要回到表格右边，让写入随状态一起缩放。接下来的问题是：应该放大多少？

## 5. 写入应该随状态放大多少？

满足一次齐次的 $$g$$ 有很多种，例如 $$g(S)=\|S\|_F$$。齐次性只要求写入与状态按相同比例缩放，没有确定二者应该保持什么比例。这时，第 2 节引入的 **Goal \***（$$F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2$$）正好用来定出这个比例。

它满足 $$F(\lambda S,c/\lambda)=F(S,c)$$。在 $$\lambda=1$$ 处对 $$\lambda$$ 求导，得到欧拉恒等式：

$$
\boxed{
\langle S,\nabla_SF\rangle_F
=c\,\partial_cF
=c\left(c\langle S,S\Sigma\rangle_F-\langle S,C\rangle_F\right).
}
\tag{11}
$$

等式左边是梯度沿状态自身方向的分量，也就是单纯让状态变长或变短的部分；它恰好由尺度校准导数 $$c\,\partial_c F$$ 控制。取当前最优尺度

$$
c^*(S)=\frac{\langle S,C\rangle_F}{\langle S,S\Sigma\rangle_F},
\qquad
 g^*(S)=\frac1{c^*(S)},
\tag{12}
$$

就有 $$\partial_cF=0$$，总体梯度与 $$S$$ 正交。以下在正尺度解存在的区域讨论这些公式。

代回更新，取新写入系数 $$\gamma=\beta$$、暂不加遗忘：

$$
\boxed{
S_{t+1}=S_t-\beta\big(S_tk_t-g^*(S_t)v_t\big)k_t^\top.
}
\tag{13}
$$

它同时具备两条性质。

一是 $$g^*(\lambda S)=\lambda g^*(S)$$，所以整个更新一次齐次。二是其总体增量满足

$$
\mathbb E[\Delta S\mid S]
=-\beta\big(S\Sigma-g^*(S)C\big),
\qquad
\langle S,\mathbb E[\Delta S\mid S]\rangle_F=0.
\tag{14}
$$

这里取常数步长，样本服从定义 $$C,\Sigma$$ 的同一分布。总体不动点满足 $$S\Sigma=g^*(S)C$$，也就是 $$S=g^*(S)P^*$$：落在目标射线上，没有额外的 ridge 项。

**齐次性消除对状态长度的依赖；这里的校准进一步让总体不动点落在目标射线上。** 此时再固定状态长度，也不会引入前面球面约束带来的额外 ridge。

### 状态长度不再控制方向步幅

上面确定了总体不动点的方向，再看每一步如何改变方向。这正是开头的尺度问题：状态放大以后，同样一次写入是否会变弱？

把 GDN 的状态写成 $$S_t=r_tU_t$$，其中 $$r_t=\|S_t\|_F$$、$$\|U_t\|_F=1$$。方向更新为

$$
U_{t+1}=N_F\left[
U_t(I-\beta_tk_tk_t^\top)
+\frac{\beta_t\|v_t\|_2}{\alpha_t r_t}\widehat v_tk_t^\top
\right],
\tag{15}
$$

其中 $$N_F(X)=X/\|X\|_F$$。因此 $$\beta_t\|v_t\|_2/(\alpha_t r_t)$$ 控制新写入相对于旧状态有多强。遗忘门既改变旧内容的权重，又通过状态长度影响后续更新。

齐次更新则不同。令 $$g(rU)=rg(U)$$，暂取 $$\alpha=1$$，式 (13) 变成

$$
U_{t+1}=N_F(U_t-\beta H_t),
\qquad
H_t=\big(U_tk_t-g(U_t)v_t\big)k_t^\top.
\tag{16}
$$

半径 $$r_t$$ 消失了。将 $$H_t$$ 分成径向与切向两部分，$$H_t=\xi_t U_t+H_t^\perp$$，其中 $$\xi_t=\langle U_t,H_t\rangle_F$$，则状态方向的转角为

$$
\theta_t=\operatorname{atan2}\left(
\beta\|H_t^\perp\|_F,\,1-\beta\xi_t
\right).
\tag{17}
$$

$$\beta$$ 是一个不依赖状态整体尺度的方向更新系数；转多少还取决于残差与对齐关系。精确总体校准使期望径向分量为零，但不要求每一步都纯切向。

这改变的是参数化方式，并没有增加表达能力：在稳态下，GDN 的 $$(\alpha,\beta)$$ 基本也能达到同样的 ridge 强度与方向步幅组合。真正的区别是，式 (15) 中 GDN 的步幅会通过 $$r_t$$ 依赖此前的历史：一次较大的 $$\|v_t\|_2$$，就可能让后面很多步的写入都变弱。不过，这个状态长度的反馈也有好处。从 $$S_0=0$$ 起步，最初较小的 $$r_t$$ 会带来较强的写入， 或许算一种很神奇像 warm-up的learning rate schedule。校准之后，可能就需要更加显式地设计这个 $$\beta$$ 以达到schedule的效果了。

<details markdown="1">
<summary>为什么尺度不变的目标还需要步长预条件？</summary>

从目标函数看，消去 $$c$$ 后的 $$G(S)=F(S,c^*(S))$$ 满足 $$G(\lambda S)=G(S)$$。不过，固定欧氏步长的梯度下降还不够：$$\nabla G(\lambda S)=\lambda^{-1}\nabla G(S)$$。式 (13) 实际对应

$$
\nabla G(S)=\big(c^*(S)\big)^2\big(S\Sigma-g^*(S)C\big),
\qquad
S_{t+1}=S_t-\frac{\beta}{\big(c^*(S_t)\big)^2}\nabla G(S_t).
\tag{18}
$$

正是这个步长预条件，让梯度更新也具备了所需的一次齐次性。

</details>

普通 DeltaNet 的内层回归并不尺度不变，所以遗忘会产生真实的回归收缩。校准后的齐次更新消除了状态长度对方向步幅的影响；若还需要降低旧内容的权重，可以另加遗忘门。

## 6. 三种校准方案，以及独立的遗忘门

上面的 $$c^*(S)$$ 给出了校准状态与目标之间全局尺度的参照。如何处理 Goal \* 中的尺度变量 $$c$$，有三种自然的选择。下面沿用第 2 节的记号，令有效状态 $$P_t=c_tS_t$$（由于 $$c_t>0$$，$$P_t$$ 与 $$S_t$$ 的归一化读出完全相同）；对单步损失 $$\ell_t(S,c)=\frac12\|cSk_t-v_t\|_2^2$$，统一对 $$S_t$$ 取预条件步长 $$\beta/c_t^2$$，就能把三种方案写成对齐的递推式。

**提前固定 $$c$$。** 这个该如何理解？首先我们知道，这个标量的取值是多少都不影响 $$P^*$$。并且这就有点类似于我们是上帝，提前 oracle 洞察了最后的 $$c$$ 是多少，有种先射箭再画靶的意思。令 $$c_t\equiv c>0$$ 为常数，更新递推为

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\frac{\beta}{c}\,v_tk_t^\top,
\qquad
P_{t+1}=P_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top.
\tag{19}
$$

换成 $$P_t=cS_t$$ 后，常数 $$c$$ 直接消掉了。取 $$c=1$$（此时 $$P_t=S_t$$）就是普通 DeltaNet，让状态长度自行适应回归目标；再对旧状态乘上衰减 $$\alpha_t$$，则得到 GDN（$$S_{t+1}=\alpha_t S_t(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$$）。优点是简单，代价是方向更新仍与状态长度耦合。

**在线学习 $$c$$。** 我们把 $$F$$ 的两个参数 $$S$$ 和 $$c$$ 同时进行在线学习——具体来说，在单步损失 $$\ell_t(S,c)$$ 上同时更新 $$S_t$$（步长 $$\beta/c_t^2$$）和 $$u_t=\log c_t$$（步长 $$\eta$$）。记预测 $$z_t=c_tS_tk_t=P_tk_t$$、残差 $$e_t=z_t-v_t$$，则 $$(S_t,c_t)$$ 的耦合递推为

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\frac{\beta}{c_t}\,v_tk_t^\top,
\qquad
c_{t+1}=a_t\,c_t,
\qquad
a_t=\exp(-\eta\,z_t^\top e_t).
\tag{20}
$$

将两式相乘，可以消去显式的 $$c_t$$，合并成单个有效状态 $$P_t=c_tS_t$$ 的递推：

$$
P_{t+1}=a_t\big[P_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top\big]
=a_t\big[P_t-\beta e_tk_t^\top\big].
\tag{21}
$$

这正是表格左下角：$$a_t$$ 同时缩放保留的旧状态和新写入。预测沿自身方向偏大时（$$z_t^\top e_t>0$$）缩小，偏小时放大。它调整整张状态与后续写入之间的尺度关系。

**实时算出 $$c$$。** 每一步直接取式 (12) 得到的瞬时最优尺度 $$c_t=c^*(S_t)$$（即 $$g^*(S_t)=1/c^*(S_t)$$），代入同样的预条件更新，就回到式 (13) 的齐次校准更新；若同样写成有效状态 $$P_t=c^*(S_t)S_t$$，则等价于先做一步普通更新 $$\widetilde P_{t+1}=P_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$$，再立刻按最优尺度重缩放：

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\beta\,g^*(S_t)\,v_tk_t^\top,
\qquad
P_{t+1}=c^*(\widetilde P_{t+1})\,\widetilde P_{t+1}.
\tag{22}
$$

总体更新不再拟合无关的状态长度，也不额外引入 ridge。AI 说可以用平滑统计量估计 $$g^*$$，例如 $$\hat g_t=\operatorname{EMA}(\|S_tk_t\|_2^2)/\operatorname{EMA}((S_tk_t)^\top v_t)$$，用来替换上式中的 $$g^*(S_t)$$，从而得到介于在线学习与总体最优响应之间的方案，我觉得合理。

这三种方案处理的是校准。是否额外衰减旧状态，是可以另行决定的事。

### 一个推论：校准之后仍可遗忘

再看表格右上角：使用精确校准的写入，同时只衰减旧状态，

$$
S_{t+1}=\alpha S_t(I-\beta k_tk_t^\top)+\beta g^*(S_t)v_tk_t^\top.
\tag{23}
$$

定义尺度无关的曲率

$$
\chi(S)=\frac{\langle S,S\Sigma\rangle_F}{\|S\|_F^2}.
$$

对总体更新寻找方向不动点 $$\mathbb E[S_{t+1}\mid S_t=S]=\mu S$$，可以得到

$$
\boxed{
S\propto C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=\frac{1-\alpha}{\alpha}\chi(S).
}
\tag{24}
$$

这个 ridge 与状态的整体长度无关，却依然改变不同方向之间的比例：校准去掉的是任意尺度带来的影响，而不是遗忘本身。

变化的是 $$\alpha$$ 的含义。式 (23) 一次齐次，整体乘一个正数不影响读出，所以它等价于

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\frac{\beta}{\alpha}\,g^*(S_t)v_tk_t^\top.
\tag{25}
$$

没有校准时，遗忘门把之后的写入**累积地**放大 $$1/\prod_{s\le t}\alpha_s$$ 倍（式 (7)）；校准之后，它只把**当前这一步**的写入放大 $$1/\alpha_t$$ 倍。它的 ridge 也不再依赖 $$\beta$$；在相同的 $$\alpha$$ 下，只有式 (8) 中 GDN ridge 的 $$\beta\chi$$ 倍（键单位化时 $$\beta\chi<1$$）。所以 $$\alpha$$ 不能直接沿用未校准的 GDN 中学到的值，需要重新学习。

### 各种 ridge 背后的同一个公式

式 (24) 只是一个特例。对式 (6) 形式的任意更新，定义**写入校准比**

$$
\omega(S)=\frac{\gamma\,g(S)}{\alpha\beta\,g^*(S)},
$$

也就是：在给定旧状态保留量的情况下，实际写入强度与 Goal \* 要求的写入强度之比。同样的计算给出

$$
\boxed{
S\propto C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=(\omega-1)\,\chi(S).
}
\tag{26}
$$

<details markdown="1">
<summary>推导</summary>

记 $$A=\langle S,S\Sigma\rangle_F$$、$$B=\langle S,C\rangle_F$$，于是 $$g^*B=A$$。方向不动点方程为

$$
\alpha S-\alpha\beta S\Sigma+\omega\alpha\beta g^*C=\mu S.
$$

与 $$S$$ 取内积，得 $$\mu=\alpha+\alpha\beta(\omega-1)\chi$$。代回原方程，再除以 $$\alpha\beta$$，便有

$$
S\big(\Sigma+(\omega-1)\chi I\big)=\omega g^*C.
$$

</details>

文章里出现的各种 ridge，都对应着 $$\omega$$ 的一种取值：

| 情形 | $$\omega$$ | $$\lambda$$ |
|---|---|---|
| 校准更新，式 (13) | $$1$$ | $$0$$ |
| 校准更新 + 遗忘门，式 (23) | $$1/\alpha$$ | $$(1-\alpha)\chi/\alpha$$，即式 (24) |
| GDN，式 (8) | $$1/(\alpha g^*(S_\star))$$ | $$(1-\alpha)/(\alpha\beta)$$，即式 (8) |
| 球面投影，式 (10) | $$1/g^*(S_R)$$ | $$(B-A)/R^2$$，即拉格朗日乘子 |

所以 $$\lambda=0$$ 当且仅当 $$\omega=1$$：在标量写入因子里，只有校准后的 $$g^*$$ 不带 ridge。其余情形只需看符号：$$\omega>1$$ 时 $$\lambda>0$$，低频方向被压制（GDN、小半径球面）；$$\omega<1$$ 时 $$\lambda<0$$，低频方向被放大（大半径球面）。因此，GDN 和球面约束背后是同一个机制——写入没有校准；区别只在于由谁决定 $$\omega$$：遗忘门，还是半径。

和 $$P^*$$ 本身一样，这些结论讨论的都是平稳分布下的总体不动点。

## 7. Takeaway

- **读出只留下一个全局尺度自由度，普通 DeltaNet 的写入却破坏了这个对称性**：RMSNorm 让读出对 $$S\mapsto\lambda S$$ 不变；只要查询会混合不同的键，引理就说明，读出能忽略的自由度只有一个全局正尺度 $$c>0$$，由此得到 Goal \*，$$F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2$$。但普通 DeltaNet 中旧状态是一次齐次的、新写入是零次齐次的，状态长度因此暗中控制了每次新写入有多容易改变方向。
- **方向偏差来自写入强度没有校准**：对标量门控的 delta 更新，总体方向不动点满足 $$S\propto C(\Sigma+\lambda I)^{-1}$$，其中 $$\lambda=(\omega-1)\chi(S)$$，$$\omega$$ 是实际写入强度与 Goal \* 要求的校准写入强度之比。只有 $$\omega=1$$ 时不带 ridge；GDN 的近因 ridge、球面约束产生的正负 ridge，以及校准之后的遗忘，都对应着 $$\omega$$ 的不同取值。
- **校准改变了遗忘门的含义**：没有校准时，$$\alpha_t$$ 相当于累积的写入放大 $$1/\prod_s\alpha_s$$，ridge 同时依赖 $$\alpha$$ 和 $$\beta$$。校准之后，整体衰减只是共同尺度的变化，$$\alpha_t$$ 留下的作用是当前步的写入放大 $$1/\alpha_t$$，ridge 变成不依赖 $$\beta$$ 的 $$(1-\alpha)\chi/\alpha$$，而 $$\beta$$ 成为尺度无关的方向步长系数。在同样的 $$\alpha$$ 下，遗忘强度大约弱了 $$1/(\beta\chi)$$ 倍，所以 $$\alpha$$ 需要重新参数化，不能直接沿用。这让参数化更清楚，并没有增加表达能力；GDN 通过状态长度反馈白送的 warm-up，也需要显式放进 $$\beta_t$$ 里。
- **所以，校准后的线性注意力内部，是尺度不变损失上的预条件梯度下降**：普通 DeltaNet 在一个并不尺度不变的内层损失上做 SGD；校准更新则是在式 (18) 的尺度不变目标 $$G$$ 上，用 $$(c^*)^{-2}$$ 做步长预条件。这与网络有效学习率理论研究的是同一类对象，而遗忘成为一个独立、显式的选择。

### 一个更通俗的类比：调配一杯混合饮料 （by Gemini）

如果暂时抛开矩阵和公式，可以把整件事想象成**往一壶饮料里不断加料**：

- **RMSNorm 只看方向** $$=$$ 喝的时候只在乎**配方比例**（比如糖和茶 1:1），不在乎这一壶总共是 1 升还是 2 升（共同尺度）；
- **第 2 节的引理** $$=$$ 单独尝糖水或茶水时，各自多倍少倍无所谓；可一旦混在一杯里喝，就不能只给糖翻倍而不给茶翻倍——唯一不毁掉风味的，只有整壶一起等比例放大或缩小；
- **第 1 节的对称性错配** $$=$$ 每次新倒进去一勺料（新写入），勺子大小是固定的：壶里原本只剩半杯时，倒一勺味道立刻大变；壶里已经装满一大桶时，倒同样一勺几乎尝不出变化——所以“壶里现在有多少（状态长度）”暗中决定了新信息的影响力；
- **齐次校准 vs 遗忘** $$=$$ **校准**是按当前壶里的总量等比例换大勺或小勺，让“总水量”真正无关紧要；**遗忘（$$\alpha<1$$）**则是每次加新料前先倒掉一部分旧饮料，让老味道主动变淡、新味道占更大比重。

## Future

我目前是基于数学和架构，展开了如上的观察与探索，讨论的也都是平稳分布下的总体不动点。我觉得（目前的见解），或许好的线性注意力机制设计应该把状态尺度、遗忘强度 $$\alpha$$ 和步长 $$\beta$$ 这几种作用解耦开来：状态尺度不应暗中影响更新；遗忘应当是一个显式的选择；$$\beta$$ 只负责每一步把方向转多少。这样每个参数各自做了什么，也更容易分析。



希望未来我自己或者其他人有机会用实验来检验这些想法。

---

## References

- Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., & Guestrin, C. (2024). [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620). *ICML*.
- Wang, K. A., Shi, J., & Fox, E. B. (2025). [Test-Time Regression: A Unifying Framework for Designing Sequence Models with Associative Memory](https://arxiv.org/abs/2501.12352). *arXiv:2501.12352*.
- Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). [It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/abs/2504.13173). *arXiv:2504.13173*. (MIRAS)
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). *NeurIPS*. (DeltaNet)
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). *ICLR*. (Gated DeltaNet)
- Qwen Team (2025). [Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking). *Hugging Face*（第 1 节 Gated Delta Rule 模块架构示意图来源）。
- Wen, K., Dang, X., Lyu, K., Ma, T., & Liang, P. (2026). [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899). *arXiv:2606.16899*.
- 苏剑林 (2026). [《从参数对称性到有效学习率》](https://spaces.ac.cn/archives/11985). *科学空间*.

<details>
<summary>Cite this post</summary>
<p>If you find this post useful in your work, you can cite it as:</p>
<blockquote>
<p>Chi, Yuhan. “What Optimizer Lives Inside Linear Attention? Scale Symmetry, Writing, and Forgetting.” November 22, 2026. <a href="https://yuhanchi.github.io/writing/optimizer-inside-linear-attention/">https://yuhanchi.github.io/writing/optimizer-inside-linear-attention/</a></p>
</blockquote>
<p>To cite this post, use the following BibTeX entry:</p>
{% raw %}<pre><code class="language-bibtex">@misc{chi2026optimizer,
  author = {Chi, Yuhan},
  title = {{What Optimizer Lives Inside Linear Attention? Scale Symmetry, Writing, and Forgetting}},
  year = {2026},
  month = nov,
  howpublished = {Blog post},
  url = {https://yuhanchi.github.io/writing/optimizer-inside-linear-attention/}
}</code></pre>{% endraw %}
</details>
