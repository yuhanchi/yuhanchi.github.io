---
title: "线性注意力内部住着什么优化器？尺度对称性、写入与遗忘"
date: "2026-11-22"
slug: "optimizer-inside-linear-attention"
description: "归一化读出不在乎状态的整体尺度，但下一次写入在乎。从这个差别出发，区分尺度校准与遗忘，理解 DeltaNet、Gated DeltaNet 和齐次更新。"
category: "tech"
tags: ["AI", "linear-attention", "optimization"]
math: true
published: true
---

这几天读了些 Linear Attention 和 Pretrain Optimizer 的工作。尤其是 Kaiyue Wen 的 [Hyperball Optimization](https://arxiv.org/abs/2606.16899) 和苏剑林的[《从参数对称性到有效学习率》](https://spaces.ac.cn/archives/11985)，让我开始想一个问题：既然线性注意力可以看作 Test-Time Training，那它内部的优化器应该怎样处理尺度？

问题的起点很简单：**把状态矩阵整体放大，归一化后的读出不变；但接下来用同一组键值更新，结果通常会变。**

原因就在更新式里。旧状态随缩放一起变大，新增的写入项却保持原来的大小。读出忽略的那个尺度，就这样重新进入了后续计算。

下面从这个差别出发，看看校准、遗忘和状态归一化各自在做什么。全文只假设键 $$k$$ 已经单位化，读出 $$Sq$$ 随后经过 RMSNorm，并把 RMSNorm 当作精确的尺度不变操作。

## 1. 读出不在乎尺度，更新为什么在乎？

状态是一个矩阵 $$S_t\in\mathbb R^{d_v\times d_k}$$。每一步接收一组键值 $$(k_t,v_t)$$，更新状态，再用查询 $$q_t$$ 读出：

$$
S_{t+1}=\alpha_t S_t(I-\beta_t k_tk_t^\top)+\gamma_t v_tk_t^\top,
\qquad
 y_t=\operatorname{RMSNorm}(S_{t+1}q_t).
\tag{1}
$$

先把 $$\alpha_t,\beta_t,\gamma_t$$ 看成三个独立系数，不预设它们之间的关系，也不让它们依赖状态的整体尺度。其中 $$0<\alpha_t\le1$$ 控制旧状态的保留比例，$$\beta_t$$ 控制擦除强度，$$\gamma_t$$ 控制写入强度。把更新式展开：

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

这里“状态”指矩阵 $$S$$，“更新”指整个递推，“写入”只指最后的新增项。擦除项减去旧状态在当前键上的一部分响应；遗忘门 $$\alpha_t$$ 则对整张旧状态作相同的衰减。

为什么叫写入？因为 $$\|k_t\|_2=1$$，加上 $$\gamma_t v_tk_t^\top$$，会让当前键上的预测 $$S_tk_t$$ 增加 $$\gamma_t v_t$$。对于其他查询 $$q$$，读出的变化是 $$\gamma_t v_t(k_t^\top q)$$：与当前键越接近，受到的影响越大。

现在把 $$S$$ 换成 $$\lambda S$$，其中 $$\lambda>0$$。读出满足

$$
\operatorname{RMSNorm}(\lambda Sq)
=
\operatorname{RMSNorm}(Sq).
\tag{3}
$$

但更新的前两项都放大了 $$\lambda$$ 倍，最后一项没有。记式 (1) 的更新算子为 $$\mathcal T$$，则

$$
\mathcal T(\lambda S)-\lambda\mathcal T(S)
=(1-\lambda)\gamma vk^\top.
\tag{4}
$$

因此，原本读出相同的两个状态，在同一次更新后通常不再读出相同。

关于 $$S$$，旧状态项是一次齐次的，新写入项是零次齐次的。**旧状态越大，同样一次写入越难改变它；旧状态越小，同样一次写入越容易改变它。** 状态的长度虽然不出现在当前输出里，却在控制下一步有多容易改变方向。

## 2. 只能忽略一个共同尺度

既然读出会归一化，能不能把每组键值分别乘上不同的系数？

取两个正交的键，设对应的值 $$v_1,v_2$$ 线性无关。如果状态满足

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

比较系数，得到三个比例相同。若两个非零像向量共线，可借第三个独立的像向量连接，因而所有非零像对应同一个 $$c$$。对 $$q_0\in\ker M$$，比较 $$q_1+q_0$$ 与 $$q_1$$，又得到 $$Sq_0=0$$，所以 $$S=cM$$。

</details>

这就是“每个向量都是特征向量，则线性算子只能是标量”的同类事实。在这里，它说明：**当查询会混合不同键时，能自由改变的是整张状态的共同尺度，而不是各组键值之间的相对比例。**

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
| **只乘旧状态** | GDN 的遗忘门。旧状态相对新写入变小，同时影响状态长度与后续更新幅度；常数门下有 ridge 型偏差。 | 仍然降低旧状态相对新写入的权重，但效果不再依赖状态的整体长度。精确校准时，可以算出尺度无关的 ridge 强度。 |
| **旧状态和新写入一起乘** | 当前读出不变，但改变了状态长度，因此会影响下一步写入的相对强度。整体校准和普通 DeltaNet 的球面投影都在这里。 | 只改变共同尺度。后续更新也随之缩放，所有归一化读出保持不变。 |

上面一行改变新旧内容之间的比例，下面一行改变整张状态的共同尺度。左边这一列里，共同尺度还会影响下一次更新；右边这一列里，这个影响被消除了。

尤其是右下角：如果更新已经一次齐次，那么每步末尾再乘一个正数，或者把状态投影到固定半径的球面上，都不会改变读出轨迹。证明只需归纳：若两条轨迹当前相差 $$\lambda$$ 倍，下一次更新后仍相差 $$\lambda$$ 倍，额外的归一化只会换一个比例。

这里说的遗忘，是额外降低旧状态相对新写入的权重。沿当前键的擦除也会覆盖旧关联，它是另一种更有针对性的修改。

还有一个等价的看法。对式 (1) 令 $$S_t=\kappa_tQ_t$$，取 $$\kappa_{t+1}=\alpha_t\kappa_t$$，就有

$$
Q_{t+1}=Q_t(I-\beta_tk_tk_t^\top)
+\frac{\gamma_t}{\kappa_{t+1}}v_tk_t^\top.
\tag{7}
$$

显式遗忘门消失了，作用变成后来写入的系数越来越大。若 $$\alpha_t<1$$，$$\kappa_{t+1}=\prod_{s\le t}\alpha_s$$ 会持续减小。于是，“不断衰减旧状态”和“不断加大后续写入”，可以表达同一条读出轨迹。这种相对权重包含此前所有遗忘门的累积作用，单步的 $$\|v_t\|$$ 无法独自表达它。

## 4. 齐次性解决尺度问题，校准决定学向哪里

满足一次齐次的 $$g$$ 有很多种，例如 $$g(S)=\|S\|_F$$。它们都能消除对状态整体尺度的依赖，但未必会得到我们想要的方向。要选择 $$g$$，还需要一个目标。

沿着 [TTT](https://arxiv.org/abs/2407.04620) 和 [Test-Time Regression](https://arxiv.org/abs/2501.12352) 的视角，DeltaNet 是单步回归损失 $$\frac12\|Sk-v\|_2^2$$ 上的一步 SGD。但细想一下，写入端和读出端的优化目标其实并不一样。

对于写入对 $$(k, v)$$，我们希望 $$S k$$ 尽可能贴近 $$v$$：

$$
\min_S \;\; \mathbb{E}\,\|S k - v\|_2^2. \tag{Goal 1}
$$

记

$$
\Sigma=\mathbb E[kk^\top]\succ0,\qquad
C=\mathbb E[vk^\top],\qquad
P^*=C\Sigma^{-1}.
$$

下文谈到方向偏差，都以这个平稳回归解 $$P^*$$ 为参照。真实序列需要追踪变化，未必应该始终保留这个解。

而对于读出对 $$(q, o^\star)$$（这里 $$o$$ 既代表 *oracle* 又代表 *output*），因为 $$S q$$ 紧接着就要经过 RMSNorm 把长度扔掉，我们真正关心的其实是输出**有没有指对方向**：

$$
\max_S \;\; \mathrm{sim}(S q,\; o^\star). \tag{Goal 2}
$$

这里的“最大化相似度”可以写成很多等价形式：

- 余弦相似度：$$\max\; \cos\angle(S q,\; o^\star)$$；
- 归一化距离：$$\min\; \|\operatorname{RMSNorm}(S q) - \operatorname{RMSNorm}(o^\star)\|_2^2$$；
- 或者任何只依赖于正射线 $$\{a\, S q : a > 0\}$$ 的目标函数。

假设我们达成了 Goal 1，并且也认同 $$(q, o^\star)$$ 在某种意义上就是另一组 $$(k, v)$$（即理想读出满足 $$o^\star = P^* q$$），那么把 Goal 1 训好，自然会推导出 Goal 2。

反过来呢？Goal 1 显然太强了：Goal 2 根本不在乎输出乘上任何正标量，直觉上每个查询似乎都可以接受各自不同的缩放因子。但这并不意味着没法从 Goal 2 往回推——第 2 节的引理已经说明，一旦要求 Goal 2 对**所有**查询都成立，逐查询的自由度就会瞬间坍缩成**唯一一个全局正尺度** $$c > 0$$。这就给出了带缩放校准的目标 **Goal \***：

$$
\boxed{\ \min_{S,\; c > 0}\; F(S, c) = \frac{1}{2}\,\mathbb{E}\,\|c\, S k - v\|_2^2\ } \tag{Goal *}
$$

如果只看解集，Goal \* 并没有什么新鲜的：令 $$P = c S$$，它就是普通最小二乘；固定任意 $$c > 0$$，最优解都是 $$S = P^* / c$$，彼此只差一个正标量，全部落在同一条射线 $$\{a P^* : a > 0\}$$ 上。Goal \* 真正增加的是**几何**——它把读出层唯一不在乎的那个共同尺度显式写成了变量。

它满足 $$F(\lambda S,c/\lambda)=F(S,c)$$。对 $$\lambda$$ 求导，得到欧拉恒等式：

$$
\boxed{
\langle S,\nabla_SF\rangle_F
=c\,\partial_cF
=c\left(c\langle S,S\Sigma\rangle_F-\langle S,C\rangle_F\right).
}
\tag{8}
$$

左边测量梯度沿状态自身方向的分量，也就是让状态变长或变短的部分。它恰好由尺度校准误差控制。取当前最优尺度

$$
c^*(S)=\frac{\langle S,C\rangle_F}{\langle S,S\Sigma\rangle_F},
\qquad
 g^*(S)=\frac1{c^*(S)},
\tag{9}
$$

就有 $$\partial_cF=0$$，总体梯度与 $$S$$ 正交。以下在正尺度解存在的区域讨论这些公式。

代回更新，取新写入系数 $$\gamma=\beta$$、暂不加遗忘：

$$
\boxed{
S_{t+1}=S_t-\beta\big(S_tk_t-g^*(S_t)v_t\big)k_t^\top.
}
\tag{10}
$$

它同时具备两条性质。

一是 $$g^*(\lambda S)=\lambda g^*(S)$$，所以整个更新一次齐次。二是其总体增量满足

$$
\mathbb E[\Delta S\mid S]
=-\beta\big(S\Sigma-g^*(S)C\big),
\qquad
\langle S,\mathbb E[\Delta S\mid S]\rangle_F=0.
\tag{11}
$$

这里取常数步长，样本来自定义 $$C,\Sigma$$ 的分布。总体不动点满足 $$S\Sigma=g^*(S)C$$，也就是 $$S=g^*(S)P^*$$：落在目标射线上，没有额外的 ridge 项。

**齐次性保证换一个状态长度不会改变算法；精确校准进一步保证总体更新沿着正确的回归方向。** 这是两件事。

## 5. Ridge 怎样改变读出方向？

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
\tag{12}
$$

外面的 $$1/\alpha$$ 不影响归一化读出；里面的 ridge 项则通常会改变方向。

看一个两维例子。键以概率 $$p_1,p_2$$ 分别取 $$e_1,e_2$$，值也分别为 $$e_1,e_2$$。于是 $$\Sigma=C=\operatorname{diag}(p_1,p_2)$$，理想状态为 $$P^*=I$$。加入 ridge 后，忽略共同尺度，状态变成

$$
\operatorname{diag}\left(
\frac{p_1}{p_1+\lambda},\,
\frac{p_2}{p_2+\lambda}
\right).
\tag{13}
$$

若 $$p_1=0.8,p_2=0.2,\lambda=0.2$$，两个系数分别是 $$0.8$$ 和 $$0.5$$。单独查任意一个键，方向仍然正确；但混合查询 $$(e_1+e_2)/\sqrt2$$ 的输出，从理想的 $$45^\circ$$ 偏到了 $$\arctan(0.5/0.8)\approx32^\circ$$。

低频键较少得到新的修正，却和其他方向一起持续衰减，因此受到更强的压制。这就是这里的近因 ridge 偏差。当 $$\Sigma$$ 各向同性时，ridge 只改变共同尺度；方向偏差来自不同方向受到不同程度的收缩。

GDN 与正则回归的关系在 [Test-Time Regression](https://arxiv.org/abs/2501.12352) 中已有明确推导，[MIRAS](https://arxiv.org/abs/2504.13173) 也把遗忘解释为保留正则化。这里进一步关心的是：哪些收缩会穿过归一化读出，表现为方向变化。

### 硬锁球面：尺度约定也可能制造同类偏差

普通 DeltaNet 每步更新后再投影到 $$\|S\|_F=R$$，并不能自动得到正确方向。对应的总体约束问题是

$$
\min_{\|S\|_F=R}\frac12\mathbb E\|Sk-v\|^2.
$$

拉格朗日驻点满足 $$S\Sigma-C+\lambda S=0$$，因此

$$
S_R=C(\Sigma+\lambda I)^{-1},\qquad \|S_R\|_F=R.
\tag{14}
$$

当半径小于无约束最优解的范数时，最优解对应正的 ridge 乘子。仍取上面的两维分布，令 $$R=0.5$$。方向正确的球面点是 $$(0.3536,0.3536)$$，但平方误差更偏爱 $$(0.4667,0.1795)$$：它把有限的长度更多分给了高频方向，混合读出约为 $$21^\circ$$。

所以，“只关心方向”并不意味着“把普通回归限制在任意球面上就行”。普通回归仍在拟合绝对长度，半径约束会迫使它牺牲不同方向之间的比例。修正写入的齐次性之后，投影才只是选择一个等价的状态长度。

### 校准以后，主动遗忘仍然存在

再看表格右上角：使用精确校准的写入，同时只衰减旧状态，

$$
S^+=\alpha S(I-\beta kk^\top)+\beta g^*(S)vk^\top.
\tag{15}
$$

定义尺度无关的曲率

$$
\chi(S)=\frac{\langle S,S\Sigma\rangle_F}{\|S\|_F^2}.
$$

对总体更新寻找方向不动点 $$\mathbb E[S^+\mid S]=\mu S$$，可以得到

$$
\boxed{
S\propto C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=\frac{1-\alpha}{\alpha}\chi(S).
}
\tag{16}
$$

<details markdown="1">
<summary>两步推导</summary>

记 $$A=\langle S,S\Sigma\rangle_F$$、$$B=\langle S,C\rangle_F$$，精确校准满足 $$g^*B=A$$。方向不动点方程为

$$
\alpha S-\alpha\beta S\Sigma+\beta g^*C=\mu S.
$$

与 $$S$$ 取内积，得 $$\mu=\alpha+\beta(1-\alpha)\chi$$。再整理原方程：

$$
S\left(\Sigma+\frac{\mu-\alpha}{\alpha\beta}I\right)
=\frac{g^*}{\alpha}C,
$$

便得到式 (16)。

</details>

这个 ridge 强度与状态的整体长度无关，却依然改变不同方向之间的比例。校准去掉了任意尺度带来的影响，并没有取消遗忘本身。

这也区分了上面两种偏差：GDN 中的额外衰减用于降低旧内容的权重；球面陷阱则是为了固定长度，顺带改变了内容比例。前者可能是我们需要的，后者未必。

## 6. 回到有效学习率

开头提到的有效学习率，现在可以接上了。

把 GDN 的状态写成 $$S_t=r_tU_t$$，其中 $$r_t=\|S_t\|_F$$、$$\|U_t\|_F=1$$。方向更新为

$$
U_{t+1}=N_F\left[
U_t(I-\beta_tk_tk_t^\top)
+\frac{\beta_t\|v_t\|_2}{\alpha_t r_t}\widehat v_tk_t^\top
\right],
\tag{17}
$$

其中 $$N_F(X)=X/\|X\|_F$$。因此 $$\beta_t\|v_t\|_2/(\alpha_t r_t)$$ 控制新写入相对于旧状态有多强。遗忘门既改变旧内容的权重，又通过状态长度影响后续更新。

齐次更新则不同。令 $$g(rU)=rg(U)$$，暂取 $$\alpha=1$$，式 (10) 变成

$$
U^+=N_F(U-\beta H),
\qquad
H=(Uk-g(U)v)k^\top.
\tag{18}
$$

半径 $$r$$ 消失了。将 $$H$$ 分成径向与切向两部分，$$H=\xi U+H^\perp$$，其中 $$\xi=\langle U,H\rangle_F$$，则状态方向的转角为

$$
\theta=\operatorname{atan2}\left(
\beta\|H^\perp\|_F,\,1-\beta\xi
\right).
\tag{19}
$$

$$\beta$$ 是一个不依赖状态整体尺度的方向更新系数；转多少还取决于残差与对齐关系。精确总体校准使期望径向分量为零，但不要求每一步都纯切向。

从目标函数看，消去 $$c$$ 后的 $$G(S)=F(S,c^*(S))$$ 满足 $$G(\lambda S)=G(S)$$。不过，固定欧氏步长的梯度下降还不够：$$\nabla G(\lambda S)=\lambda^{-1}\nabla G(S)$$。式 (10) 实际对应

$$
\nabla G(S)=(c^*)^2(S\Sigma-g^*C),
\qquad
S^+=S-\frac{\beta}{(c^*)^2}\nabla G(S).
\tag{20}
$$

是这个步长预条件，让更新也具有了需要的一次齐次性。

外层尺度不变网络中，weight decay 可以通过权重长度调节有效学习率。普通 DeltaNet 的内层回归并不尺度不变，所以遗忘还会产生真实的回归收缩。把内层目标和更新都与读出尺度对齐后，才能分清哪些变化来自校准，哪些来自有意降低旧内容的权重。

## 7. 三种处理校准尺度的方案

对于回归目标中的校准变量 $$c$$，有三种直接的选择。

**固定 $$c$$。** 取 $$c=1$$，就是普通 DeltaNet，让状态长度自行适应回归目标。加入旧状态衰减则得到 GDN。优点是简单，代价是方向更新仍与状态长度耦合。

**学习 $$c$$。** 在 $$F(S,c)$$ 上同时更新 $$S$$ 和 $$u=\log c$$。令 $$P=cS$$、$$z=Pk$$、$$e=z-v$$，对 $$S$$ 使用步长 $$\beta/c^2$$，对 $$u$$ 使用步长 $$\eta$$，可合并成

$$
P^+=a\big[P-\beta ek^\top\big],
\qquad
a=\exp(-\eta z^\top e).
\tag{21}
$$

这正是表格左下角：$$a$$ 同时缩放保留的旧状态和新写入。预测沿自身方向偏大时缩小，偏小时放大。它调整整张状态与后续写入之间的尺度关系。

**消去 $$c$$。** 取式 (9) 的 $$c^*(S)$$，使用式 (10) 的齐次校准更新。总体更新不再拟合无关的状态长度，也不额外引入 ridge。用平滑统计量估计 $$g^*$$，例如 $$\operatorname{EMA}(\|Sk\|^2)/\operatorname{EMA}((Sk)^\top v)$$，则得到介于在线学习与总体最优响应之间的方案。

这三种方案处理的是校准。是否额外衰减旧状态，是可以另行决定的事。

## 8. 这个视角还说明了什么？

逐通道门控会对不同特征施加不同衰减。它通常会改变混合查询的读出方向，因此属于内容修改，不能作为一个无关紧要的共同尺度消掉。这样做可以实现选择性遗忘，也需要相对于任务判断哪些方向应该保留。

同样，真实序列中的目标会变化，静态的 $$P^*$$ 只是分析参照。问题不在于消灭一切 ridge，而在于分清：某个方向的压制，是为了追踪新信息，还是仅仅由尺度处理顺带产生。

读出忽略共同尺度，更新也可以尊重这个自由度。**让旧状态与新写入随尺度一起变化，才使状态长度成为表示约定；新旧内容之间应该保留多少，则留给遗忘机制决定。**

---

## References

- Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., & Guestrin, C. (2024). [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620). *ICML*.
- Wang, K. A., Shi, J., & Fox, E. B. (2025). [Test-Time Regression: A Unifying Framework for Designing Sequence Models with Associative Memory](https://arxiv.org/abs/2501.12352). *arXiv:2501.12352*.
- Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). [It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/abs/2504.13173). *arXiv:2504.13173*. (MIRAS)
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). *NeurIPS*. (DeltaNet)
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). *ICLR*. (Gated DeltaNet)
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
