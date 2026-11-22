---
title: "What Optimizer Lives Inside Linear Attention? Scale Symmetry, Writing, and Forgetting"
date: "2026-11-22"
description: "Normalized readout ignores the overall scale of the state, yet the very next write does not. Starting from that mismatch, we separate scale calibration from forgetting across DeltaNet, Gated DeltaNet, and homogeneous updates."
category: "tech"
tags: ["AI", "linear-attention", "optimization"]
math: true
published: true
---

Lately I have been reading some work on Linear Attention and pretraining optimizers, which got me wondering: if Linear Attention can be viewed as Test-Time Training, how should its inner optimizer handle scale? Going a step further, does this Linear Attention block share some of the same nice properties—such as scale structure or symmetry?

My short answer, worked out below: plain DeltaNet runs SGD on an inner loss that is not scale-invariant; once the write is calibrated, the update becomes preconditioned gradient descent on a scale-invariant loss—the inner-loop analogue of the scale-invariant weights studied in effective-learning-rate theory ([Wen et al.](https://arxiv.org/abs/2606.16899); [Su](https://spaces.ac.cn/archives/11985)).

## 1. Scale in the Update?

<figure>
  <a href="{{ '/assets/img/posts/optimizer-inside-linear-attention/linear-attention.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/optimizer-inside-linear-attention/linear-attention.png' | relative_url }}" alt="Modern Linear Attention (Gated Delta Rule) block diagram: q and k pass through Linear, Conv, SiLU, and L2 normalization; v passes through Linear, Conv, and SiLU; alpha and beta come from linear projections; all feed into the Gated Delta Rule, followed by Zero-Centered RMSNorm, multiplication with the SiLU output gate, and a final Linear layer." width="613" height="576" loading="lazy"></a>
  <figcaption>Architecture of a modern Linear Attention (Gated Delta Rule) block: \(q, k\) are L2-normalized, and the state readout passes through RMSNorm before the output gate.</figcaption>
</figure>

Look closely at the diagram above. The state is a matrix $$S_t\in\mathbb R^{d_v\times d_k}$$. At each step, we receive a key-value pair $$(k_t,v_t)$$, update the state, and read it out with a query $$q_t$$:

$$
S_{t+1}=\alpha_t S_t(I-\beta_t k_tk_t^\top)+\gamma_t v_tk_t^\top,
\qquad
 y_t=\operatorname{RMSNorm}(S_{t+1}q_t).
\tag{1}
$$

First treat $$\alpha_t,\beta_t,\gamma_t$$ as three independent coefficients without making any assumptions about them yet. At a basic semantic level, $$\alpha_t$$ controls how much of the old state is retained, $$\beta_t$$ controls the erasure strength, and $$\gamma_t$$ represents the write strength. Expanding the update:

$$
S_{t+1}
=
\underbrace{\alpha_t S_t}_{\text{global decay of old state}}
-
\underbrace{\alpha_t\beta_t(S_tk_t)k_t^\top}_{\text{erasure along current key}}
+
\underbrace{\gamma_t v_tk_t^\top}_{\text{new write}}.
\tag{2}
$$

As labeled in Eq. (2), we call the matrix $$S_t$$ the state, the $$-\alpha_t\beta_t(S_tk_t)k_t^\top$$ term the erasure, and the final $$\gamma_t v_tk_t^\top$$ term the write; the forget gate $$\alpha_t$$ acts on $$S_t(I-\beta_t k_tk_t^\top)$$.

Now replace $$S$$ with $$\lambda S$$, where $$\lambda>0$$. In practice, the readout satisfies

$$
\operatorname{RMSNorm}(\lambda Sq)
=
\operatorname{RMSNorm}(Sq).
\tag{3}
$$

So this thought experiment makes sense. Notice that the first two terms in Eq. (1) scale right along with $$\lambda$$, whereas the final write term does not.
Writing this more formally: let $$\mathcal T$$ denote the update operator in Eq. (1), then

$$
\mathcal T(\lambda S)-\lambda\mathcal T(S)
=(1-\lambda)\gamma vk^\top.
\tag{4}
$$

Hence two states with identical readouts usually no longer read out the same after undergoing the same update.

In $$S$$, the old-state terms are degree-1 homogeneous, while the new write is degree-0 homogeneous. **The larger the old state, the harder it is for the same write to move it; the smaller the old state, the easier it is to move.** Even though the state's norm never appears in the current output, it quietly controls how easily the next step can rotate the state's direction.

## 2. Two Goals?

Following [TTT](https://arxiv.org/abs/2407.04620) and [Test-Time Regression](https://arxiv.org/abs/2501.12352), plain DeltaNet corresponds to one step of SGD on the per-step regression loss $$\frac12\|Sk-v\|_2^2$$. Look a little closer, though, and the writing side and the reading side do not actually share the same objective.

For a write pair $$(k,v)$$, we want $$Sk$$ as close to $$v$$ as possible:

$$
\min_S \;\; \mathbb{E}\,\|Sk-v\|_2^2. \tag{Goal 1}
$$

Let

$$
\Sigma=\mathbb E[kk^\top]\succ0,\qquad
C=\mathbb E[vk^\top],\qquad
P^*=C\Sigma^{-1}.
$$

For a read pair $$(q,o^\star)$$ (where $$o$$ stands for both *oracle* and *output*—a small pun), $$Sq$$ immediately passes through RMSNorm, which throws away its length. What we really care about is whether the output **points in the right direction**:

$$
\max_S \;\; \mathrm{sim}(Sq,\; o^\star). \tag{Goal 2}
$$

This "maximize similarity" can be written as cosine similarity $$\max\cos\angle(Sq,o^\star)$$, normalized distance $$\min\|\operatorname{RMSNorm}(Sq)-\operatorname{RMSNorm}(o^\star)\|_2^2$$, or any objective depending only on the positive ray $$\{a\,Sq : a>0\}$$.

Suppose we achieve Goal 1 and also accept that $$(q,o^\star)$$ is in some sense another $$(k,v)$$ pair (i.e., ideal readout satisfies $$o^\star=P^*q$$); then solving Goal 1 naturally implies Goal 2. What about the converse? Goal 1 looks too strong: Goal 2 does not care about multiplying the output by any positive scalar, so intuitively it might seem that each key-value pair could be scaled by a different factor.

Can it? Take two orthogonal keys with linearly independent values $$v_1,v_2$$, and suppose the state satisfies

$$
Sk_1=2v_1,\qquad Sk_2=v_2.
\tag{5}
$$

Querying $$k_1$$ or $$k_2$$ individually gives the right direction. But for the mixed query $$q=(k_1+k_2)/\sqrt2$$, the readout points along $$2v_1+v_2$$ rather than $$v_1+v_2$$. Normalizing at the end cannot remove the mismatched ratio between the two components.

This corresponds to a simple linear-algebra fact:

> **Lemma.** Suppose $$\operatorname{rank}(M)\ge2$$. If for every $$q\notin\ker M$$ the vector $$Sq$$ is a positive multiple of $$Mq$$, then there exists a single global scalar $$c>0$$ such that $$S=cM$$.

<details markdown="1">
<summary>Proof</summary>

Write $$Sq=c(q)Mq$$. Whenever $$Mq_1,Mq_2$$ are linearly independent, linearity gives

$$
S(q_1+q_2)=c(q_1)Mq_1+c(q_2)Mq_2
=c(q_1+q_2)(Mq_1+Mq_2),
$$

so comparing coefficients forces all three scalars to be equal. If two nonzero images are collinear, pick a third linearly independent image to bridge them; hence every nonzero image shares the same $$c$$. Finally, for $$q_0\in\ker M$$, comparing $$q_1+q_0$$ with $$q_1$$ gives $$Sq_0=0$$, so $$S=cM$$.

</details>

Inspired by this, it is easy to notice that once Goal 2 must hold for **all** queries, per-query freedom immediately collapses into a single global positive scale $$c>0$$. That lets us design a more interesting—and perhaps more promising—objective, **Goal \***:

$$
\boxed{\ \min_{S,\; c>0}\; F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2\ }
\tag{Goal *}
$$

As a set of solutions, Goal \* is actually very simple: setting $$P=cS$$ turns it into ordinary least squares; for any fixed $$c>0$$, the minimizer is $$S=P^*/c$$, all lying on the same ray $$\{aP^* : a>0\}$$. What Goal \* really adds is **geometry**—it promotes the one global scale that the readout ignores into an explicit variable $$c$$.

We will see shortly how Goal \* resolves the troubles from Sections 1 and 2. Section 6 will also show that this choice is not arbitrary: among scalar write factors, the one Goal \* prescribes is the only one whose population fixed direction carries no ridge.

## 3. Scale the Write Along with the State

The old-state terms in Eq. (2) already satisfy

$$
\lambda S(I-\beta kk^\top)
=\lambda\big[S(I-\beta kk^\top)\big].
$$

Only the new write breaks this symmetry. Multiplying the write by a scalar satisfying $$g(\lambda S)=\lambda g(S)$$ gives

$$
\boxed{
\mathcal T(S)=\alpha S(I-\beta kk^\top)+\gamma g(S)vk^\top,
\qquad
\mathcal T(\lambda S)=\lambda\mathcal T(S).
}
\tag{6}
$$

This is positive 1-homogeneity—the exact scale equivariance we need: scale the state by $$\lambda$$, and the updated state scales by $$\lambda$$ as well. Two initial states differing by a positive scalar remain scalar multiples of each other forever, so their normalized readouts stay identical at every step.

State length can now act purely as a representation convention, instead of silently governing the relative strength of future writes.

This also makes the role of scalar gates clear. We only need to ask two questions: **does the scalar multiply only the old state, or both old state and new write? And does the write scale with $$S$$?**

| What does the scalar multiply? | Write does not scale with $$S$$: $$vk^\top$$ | Write scales with $$S$$: $$g(S)vk^\top$$ |
|---|---|---|
| **Old state only**<br>$$\alpha S(I-\beta kk^\top)+\text{write}$$ | GDN's forget gate. Shrinks old state relative to new write while also changing state norm and future step sizes; introduces ridge bias when $$\alpha$$ is constant. | Still lowers old-state weight relative to new writes, without depending on overall state length. Even after calibration, relative forgetting remains independently controllable. |
| **Old state and new write together**<br>$$a\big[S(I-\beta kk^\top)+\text{write}\big]$$ | Leaves the current readout unchanged, but alters state length and therefore shifts the relative strength of the next write. Global calibration and plain DeltaNet's sphere projection live here. | Pure change of common scale. Future updates scale along with it, so every normalized readout stays completely unchanged. |

The top row alters the balance between old and new content; the bottom row alters the shared scale of the whole state. In the left column, that shared scale still leaks into the next update; in the right column, that leakage is gone.

Perhaps the best-behaved cell is the bottom right: once the update is 1-homogeneous, multiplying by a positive scalar at the end of every step—or projecting the state onto a fixed-radius sphere—does not change the readout trajectory at all. Induction proves it in one line: if two trajectories differ by a factor of $$\lambda$$, they still differ by $$\lambda$$ after the next update, and renormalizing only swaps one positive factor for another.

Note: "forgetting" here refers to globally lowering the weight of the old state relative to the new write; erasure along the current key is a targeted local edit along a single direction.

<details markdown="1">
<summary>An equivalent view: decaying the old state is boosting future writes</summary>

There is an equivalent way to read Eq. (1). Set $$S_t=\kappa_tQ_t$$ with $$\kappa_{t+1}=\alpha_t\kappa_t$$; then

$$
Q_{t+1}=Q_t(I-\beta_tk_tk_t^\top)
+\frac{\gamma_t}{\kappa_{t+1}}v_tk_t^\top.
\tag{7}
$$

The explicit forget gate vanishes, turning into an ever-growing multiplier on later writes. When $$\alpha_t<1$$, $$\kappa_{t+1}=\prod_{s\le t}\alpha_s$$ keeps shrinking. Under normalized readout, "continuously decaying the old state" and "continuously amplifying newer writes" trace out the exact same trajectory. Because this multiplier accumulates all past forget gates, the single-step norm $$\|v_t\|$$ alone cannot replace it.

</details>

The table separates the two operations, but how do they actually affect the readout? Let us look at two examples in the left column: one deliberately decays the old state, the other only tries to fix the state's length, yet both end up tilting the relative proportions across directions.

## 4. How Two Operations Produce Directional Bias

To compare directions, we use the stationary regression solution $$P^*=C\Sigma^{-1}$$ from Section 2—the minimizer of ordinary fixed-$$c=1$$ regression $$F(S,1)=\frac12\mathbb E\|Sk-v\|_2^2$$—as our reference. All biases discussed below are measured against this reference; real token streams drift over time, so keeping this static solution is not always what the network wants.

### GDN: Low-frequency directions suffer stronger decay

Gated DeltaNet corresponds to $$\gamma=\beta$$ in Eq. (1):

$$
S_{t+1}=\alpha S_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top.
$$

For i.i.d. samples and constant $$\alpha,\beta$$, the mean fixed point satisfies

$$
S_\star[(1-\alpha)I+\alpha\beta\Sigma]=\beta C,
$$

which gives

$$
S_\star=\frac1\alpha C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=\frac{1-\alpha}{\alpha\beta}.
\tag{8}
$$

The outer factor $$1/\alpha$$ is absorbed by RMSNorm, whereas the inner ridge term $$\lambda I$$ generally changes directions.

Take a 2D example: keys $$e_1,e_2$$ appear with probabilities $$p_1,p_2$$ and serve as their own values, so $$\Sigma=C=\operatorname{diag}(p_1,p_2)$$ and $$P^*=I$$. Up to a common scalar, the fixed point becomes

$$
\operatorname{diag}\left(
\frac{p_1}{p_1+\lambda},\,
\frac{p_2}{p_2+\lambda}
\right).
\tag{9}
$$

With $$p_1=0.8,p_2=0.2,\lambda=0.2$$, the entries are $$0.8$$ and $$0.5$$, so the mixed query $$(e_1+e_2)/\sqrt2$$ tilts from $$45^\circ$$ to about $$32^\circ$$.

Rare keys get fewer fresh writes but are decayed at every step, so they are suppressed more. This is the recency ridge bias; if $$\Sigma$$ is isotropic, the ridge only rescales the whole state.

The equivalence between GDN and regularized regression is established in [Test-Time Regression](https://arxiv.org/abs/2501.12352), and [MIRAS](https://arxiv.org/abs/2504.13173) similarly frames forgetting as retention regularization. What we highlight here is specifically which shrinkages pass through normalized readout as directional bias.

### Locking the sphere: a scale convention can create the same bias

Projecting plain DeltaNet onto $$\|S\|_F=R$$ after every step does not automatically preserve directions either. The corresponding population constrained problem is

$$
\min_{\|S\|_F=R}\frac12\mathbb E\|Sk-v\|^2.
$$

Its Lagrangian stationary condition is $$S\Sigma-C+\lambda S=0$$, giving

$$
S_R=C(\Sigma+\lambda I)^{-1},\qquad \|S_R\|_F=R.
\tag{10}
$$

Only the sign of $$\lambda$$ matters. If $$R<\|P^*\|_F$$, then $$\lambda>0$$ and rare directions are suppressed, as in GDN (in the 2D example with $$R=0.5$$, the mixed readout tilts to about $$21^\circ$$). If $$R>\|P^*\|_F$$, then $$\lambda<0$$ and rare directions are amplified instead.

Both operations produce a similar directional tilt, for seemingly different reasons: GDN deliberately down-weights older content, whereas sphere projection only tries to fix the norm yet accidentally distorts content ratios along the way. (Section 6 shows that underneath, they share one mechanism.)

For sphere projection, the culprit is that plain regression is still fitting absolute lengths. To make state norm a pure representation convention, we have to move to the right column of the table and scale the write with the state. The next question is: by how much?

## 5. How Much Should the Write Scale with the State?

Many functions $$g$$ are 1-homogeneous—for instance, $$g(S)=\|S\|_F$$. Homogeneity only demands that the write scale proportionally with the state; it does not say what ratio the two should maintain. Here **Goal \*** from Section 2 ($$F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2$$) pins that ratio down.

Because $$F(\lambda S,c/\lambda)=F(S,c)$$, differentiating with respect to $$\lambda$$ at $$\lambda=1$$ yields the Euler identity:

$$
\boxed{
\langle S,\nabla_SF\rangle_F
=c\,\partial_cF
=c\left(c\langle S,S\Sigma\rangle_F-\langle S,C\rangle_F\right).
}
\tag{11}
$$

The left-hand side is the component of the gradient along $$S$$ itself—the radial part that merely stretches or shrinks the state—and it is governed exactly by the scale derivative $$c\,\partial_c F$$. Choosing the optimal scale for the current state,

$$
c^*(S)=\frac{\langle S,C\rangle_F}{\langle S,S\Sigma\rangle_F},
\qquad
 g^*(S)=\frac1{c^*(S)},
\tag{12}
$$

sets $$\partial_cF=0$$ and makes the population gradient orthogonal to $$S$$. (Below we work in the region where a positive optimal scale exists.)

Plugging this into the update with $$\gamma=\beta$$ and no forgetting yet gives

$$
\boxed{
S_{t+1}=S_t-\beta\big(S_tk_t-g^*(S_t)v_t\big)k_t^\top.
}
\tag{13}
$$

It satisfies two properties at once.

First, $$g^*(\lambda S)=\lambda g^*(S)$$, so the entire update is 1-homogeneous. Second, its expected increment satisfies

$$
\mathbb E[\Delta S\mid S]
=-\beta\big(S\Sigma-g^*(S)C\big),
\qquad
\langle S,\mathbb E[\Delta S\mid S]\rangle_F=0.
\tag{14}
$$

With a constant step size and samples drawn from the distribution defining $$C,\Sigma$$, the population fixed point satisfies $$S\Sigma=g^*(S)C$$, i.e. $$S=g^*(S)P^*$$: it lands directly on the target ray with no extra ridge term.

**Homogeneity removes dependence on state length; this calibration ensures the population fixed point lies on the right regression ray.** Once both hold, fixing the state norm no longer injects the extra ridge seen in the sphere trap.

### State length no longer controls the angular step

Having settled the direction of the fixed point, let us check how each step moves the direction—bringing us back to our opening question of whether enlarging the state weakens the next write.

Write GDN's state as $$S_t=r_tU_t$$ with $$r_t=\|S_t\|_F$$ and $$\|U_t\|_F=1$$. Its direction evolves as

$$
U_{t+1}=N_F\left[
U_t(I-\beta_tk_tk_t^\top)
+\frac{\beta_t\|v_t\|_2}{\alpha_t r_t}\widehat v_tk_t^\top
\right],
\tag{15}
$$

where $$N_F(X)=X/\|X\|_F$$. The factor $$\beta_t\|v_t\|_2/(\alpha_t r_t)$$ controls how strongly the new write pushes against the old state. Thus the forget gate plays two roles at once: it down-weights older content and indirectly modulates future directional step sizes through $$r_t$$.

A homogeneous update behaves differently. Whenever $$g(rU)=rg(U)$$ (setting $$\alpha=1$$ for now), Eq. (13) reduces to

$$
U_{t+1}=N_F(U_t-\beta H_t),
\qquad
H_t=\big(U_tk_t-g(U_t)v_t\big)k_t^\top.
\tag{16}
$$

The radius $$r_t$$ drops out completely. Decomposing $$H_t=\xi_t U_t+H_t^\perp$$ into radial and tangential parts (with $$\xi_t=\langle U_t,H_t\rangle_F$$), the single-step turning angle is

$$
\theta_t=\operatorname{atan2}\left(
\beta\|H_t^\perp\|_F,\,1-\beta\xi_t
\right).
\tag{17}
$$

Now $$\beta$$ is a scale-free directional step coefficient; how far the state rotates depends only on $$\beta$$ and the residual geometry at $$U$$. Exact population calibration zeros out the expected radial component without requiring every single stochastic step to be purely tangential.

This is a change of parameterization rather than a gain in expressivity: at steady state, GDN's $$(\alpha,\beta)$$ can reach essentially the same pairs of ridge and step. The real difference is that GDN's step in Eq. (15) depends on its *history* through $$r_t$$: one burst of large $$\|v_t\|_2$$ weakens writes for many steps afterwards. That norm loop also does something useful, though. Starting from $$S_0=0$$, a small $$r_t$$ gives strong early writes—a free warm-up. After calibration, $$S=0$$ is a fixed point and $$\beta$$ is a constant angular step, so any warm-up or annealing schedule has to be put into $$\beta_t$$ explicitly.

<details markdown="1">
<summary>Why does a scale-invariant objective still need step-size preconditioning?</summary>

Eliminating $$c$$ gives the reduced objective $$G(S)=F(S,c^*(S))$$, which satisfies $$G(\lambda S)=G(S)$$. Plain fixed-step Euclidean gradient descent on $$G$$ is not enough, however, because $$\nabla G(\lambda S)=\lambda^{-1}\nabla G(S)$$. In population form, Eq. (13) actually corresponds to

$$
\nabla G(S)=\big(c^*(S)\big)^2\big(S\Sigma-g^*(S)C\big),
\qquad
S_{t+1}=S_t-\frac{\beta}{\big(c^*(S_t)\big)^2}\nabla G(S_t).
\tag{18}
$$

It is this step-size preconditioning by $$(c^*)^{-2}$$ that makes the gradient step 1-homogeneous.

</details>

In outer scale-invariant networks, weight decay regulates the effective learning rate through parameter norm without changing the optimal solution set. In plain DeltaNet, the inner regression is not scale-invariant, so forgetting also induces genuine regression shrinkage. Homogeneous calibrated updates strip state length out of the angular step size; if you still want to down-weight older memories, you can add a forget gate separately—though, as Section 6 shows, it no longer means quite the same thing.

## 6. Three Calibration Schemes, and an Independent Forget Gate

The $$c^*(S)$$ above gives a reference for calibrating the global scale between the state and the target. There are three natural ways to treat the scale variable $$c$$ in Goal \*. Carrying over the effective state $$P_t=c_tS_t$$ from Section 2 (since $$c_t>0$$, $$P_t$$ and $$S_t$$ share the exact same normalized readout) and using the preconditioned step size $$\beta/c_t^2$$ on $$S_t$$ for the per-step loss $$\ell_t(S,c)=\frac12\|cSk_t-v_t\|_2^2$$ puts all three schemes into the same recurrence notation.

**Fix $$c$$ in advance.** How should we think about this? First, whatever value this scalar takes does not affect $$P^*$$. It is a bit like playing God—knowing the final $$c$$ in advance like an oracle, almost shooting the arrow first and drawing the target around it. Fixing a constant $$c_t\equiv c>0$$ gives the recurrence

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\frac{\beta}{c}\,v_tk_t^\top,
\qquad
P_{t+1}=P_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top.
\tag{19}
$$

Rewritten in terms of $$P_t=cS_t$$, the constant $$c$$ cancels out. Setting $$c=1$$ (so $$P_t=S_t$$) gives plain DeltaNet, letting state length adapt on its own to the regression target; multiplying the old state by a decay $$\alpha_t$$ yields GDN ($$S_{t+1}=\alpha_t S_t(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top$$). It is simple, at the cost of keeping directional updates coupled to state length.

**Learn $$c$$ online.** We learn both parameters of $$F$$—$$S$$ and $$c$$—online at the same time: specifically, updating $$S_t$$ (step size $$\beta/c_t^2$$) and $$u_t=\log c_t$$ (step size $$\eta$$) jointly on the per-step loss $$\ell_t(S,c)$$. Letting $$z_t=c_tS_tk_t=P_tk_t$$ and $$e_t=z_t-v_t$$, the coupled recurrence for $$(S_t,c_t)$$ is

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\frac{\beta}{c_t}\,v_tk_t^\top,
\qquad
c_{t+1}=a_t\,c_t,
\qquad
a_t=\exp(-\eta\,z_t^\top e_t).
\tag{20}
$$

Multiplying the two updates eliminates $$c_t$$ explicitly and collapses them into a single-matrix recurrence for the effective state $$P_t=c_tS_t$$:

$$
P_{t+1}=a_t\big[P_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top\big]
=a_t\big[P_t-\beta e_tk_t^\top\big].
\tag{21}
$$

This sits in the bottom-left cell of our table: $$a_t$$ scales the retained state and the new write together—shrinking when the prediction overshoots along its own direction ($$z_t^\top e_t>0$$) and growing when it undershoots—recalibrating the scale relationship between the state and future writes.

**Compute $$c$$ on the fly.** At each step, take the instantaneous optimal scale $$c_t=c^*(S_t)$$ from Eq. (12) (so $$g^*(S_t)=1/c^*(S_t)$$). Plugging it into the same preconditioned step recovers the homogeneous calibrated update (13); written in terms of $$P_t=c^*(S_t)S_t$$, it is equivalent to taking one ordinary update $$\widetilde P_{t+1}=P_t(I-\beta k_tk_t^\top)+\beta v_tk_t^\top$$ and immediately rescaling by its optimal scale:

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\beta\,g^*(S_t)\,v_tk_t^\top,
\qquad
P_{t+1}=c^*(\widetilde P_{t+1})\,\widetilde P_{t+1}.
\tag{22}
$$

The population update stops fitting irrelevant state lengths and introduces no extra ridge. AI suggests estimating $$g^*$$ with smoothed statistics—for example, $$\hat g_t=\operatorname{EMA}(\|S_tk_t\|_2^2)/\operatorname{EMA}((S_tk_t)^\top v_t)$$ in place of $$g^*(S_t)$$ above—giving a middle ground between online learning and exact population best response; that seems reasonable to me.

All three schemes address calibration. Whether to additionally decay the old state is a separate decision.

### A corollary: calibration does not abolish forgetting

Consider the top-right cell of the table: combine exact calibrated writing with decay applied only to the old state,

$$
S_{t+1}=\alpha S_t(I-\beta k_tk_t^\top)+\beta g^*(S_t)v_tk_t^\top.
\tag{23}
$$

Define the scale-free directional curvature

$$
\chi(S)=\frac{\langle S,S\Sigma\rangle_F}{\|S\|_F^2}.
$$

Solving for a directional fixed point $$\mathbb E[S_{t+1}\mid S_t=S]=\mu S$$ of the population update gives

$$
\boxed{
S\propto C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=\frac{1-\alpha}{\alpha}\chi(S).
}
\tag{24}
$$

This ridge no longer depends on the state's length, yet it still changes the relative weights across directions: calibration removes artifacts of arbitrary scale, not forgetting itself.

What does change is the meaning of $$\alpha$$. Since Eq. (23) is 1-homogeneous, a global factor does not affect the readout, so Eq. (23) is equivalent to

$$
S_{t+1}=S_t(I-\beta k_tk_t^\top)+\frac{\beta}{\alpha}\,g^*(S_t)v_tk_t^\top.
\tag{25}
$$

Without calibration, the forget gate amplifies later writes cumulatively, by $$1/\prod_{s\le t}\alpha_s$$ (Eq. 7); with calibration, it only boosts the current write by $$1/\alpha_t$$. Its ridge also no longer depends on $$\beta$$, and at the same $$\alpha$$ it is only $$\beta\chi$$ times GDN's ridge in Eq. (8) ($$\beta\chi<1$$ for normalized keys). So $$\alpha$$ has to be re-parameterized, not reused.

### One formula behind every ridge

Eq. (24) is a special case. For any update of the form in Eq. (6), define the *write miscalibration*

$$
\omega(S)=\frac{\gamma\,g(S)}{\alpha\beta\,g^*(S)},
$$

that is, what we actually write divided by what Goal \* says we should write, given what we kept. The same calculation gives

$$
\boxed{
S\propto C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=(\omega-1)\,\chi(S).
}
\tag{26}
$$

<details markdown="1">
<summary>Derivation</summary>

Let $$A=\langle S,S\Sigma\rangle_F$$ and $$B=\langle S,C\rangle_F$$, so $$g^*B=A$$. The directional fixed-point equation is

$$
\alpha S-\alpha\beta S\Sigma+\omega\alpha\beta g^*C=\mu S.
$$

Taking the inner product with $$S$$ yields $$\mu=\alpha+\alpha\beta(\omega-1)\chi$$. Substituting back and dividing by $$\alpha\beta$$ gives

$$
S\big(\Sigma+(\omega-1)\chi I\big)=\omega g^*C.
$$

</details>

Every ridge in this post is one value of $$\omega$$:

| Case | $$\omega$$ | $$\lambda$$ |
|---|---|---|
| Calibrated, Eq. (13) | $$1$$ | $$0$$ |
| Calibrated + forget gate, Eq. (23) | $$1/\alpha$$ | $$(1-\alpha)\chi/\alpha$$, i.e. Eq. (24) |
| GDN, Eq. (8) | $$1/(\alpha g^*(S_\star))$$ | $$(1-\alpha)/(\alpha\beta)$$, i.e. Eq. (8) |
| Sphere projection, Eq. (10) | $$1/g^*(S_R)$$ | $$(B-A)/R^2$$, i.e. the Lagrange multiplier |

So $$\lambda=0$$ exactly when $$\omega=1$$: the calibrated $$g^*$$ is the only scalar write factor with no ridge. Otherwise, only the sign matters: $$\omega>1$$ gives $$\lambda>0$$ and suppresses rare directions (GDN, a small sphere); $$\omega<1$$ gives $$\lambda<0$$ and amplifies them (a large sphere). GDN and the sphere thus share one mechanism, a miscalibrated write; they differ only in what sets $$\omega$$—the gate or the radius.

Like $$P^*$$ itself, all of this concerns population fixed points under a stationary distribution.

## 7. Takeaway

- **Readout leaves one global scale free; plain DeltaNet's write breaks it.** RMSNorm makes the readout invariant to $$S\mapsto\lambda S$$, and once queries mix different keys, the lemma says this single global scale $$c>0$$ is the only freedom readout allows—hence Goal \*, $$F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2$$. In plain DeltaNet, however, the old state is degree-1 homogeneous and the write is degree-0 homogeneous, so state length quietly controls how strongly each new write moves the state.
- **Directional bias is write miscalibration.** For any scalar-gated delta update, population directional fixed points satisfy $$S\propto C(\Sigma+\lambda I)^{-1}$$ with $$\lambda=(\omega-1)\chi(S)$$, where $$\omega$$ is the actual write strength relative to the one Goal \* prescribes. Calibration ($$\omega=1$$) is the only ridge-free choice; GDN's recency ridge, the sphere trap (of either sign), and calibrated forgetting are all values of $$\omega$$.
- **Calibration changes what a forget gate is.** Without calibration, $$\alpha_t$$ acts as a cumulative write amplification $$1/\prod_s\alpha_s$$, and its ridge depends on both $$\alpha$$ and $$\beta$$. With it, global decay is pure gauge: $$\alpha_t$$ survives only as an instantaneous write emphasis $$1/\alpha_t$$, its ridge $$(1-\alpha)\chi/\alpha$$ no longer depends on $$\beta$$, and $$\beta$$ becomes a scale-free directional step size. At the same $$\alpha$$, it forgets about $$1/(\beta\chi)$$ times less, so $$\alpha$$ has to be re-parameterized, not reused. This is a cleaner parameterization rather than extra expressivity, and the free warm-up of GDN's norm loop must then be written into $$\beta_t$$ explicitly.
- **So the optimizer inside calibrated linear attention is preconditioned descent on a scale-invariant loss.** Plain DeltaNet is SGD on an inner loss that is not scale-invariant; the calibrated update is $$(c^*)^{-2}$$-preconditioned descent on the scale-invariant $$G$$ of Eq. (18)—the same kind of object studied in effective-learning-rate theory for outer networks, with forgetting as a separate, explicit choice.

All of these are statements about population fixed points under stationarity, the same frame in which $$P^*$$ is defined.

### A Simpler Analogy: Mixing a Drink

Setting matrices and equations aside for a moment, picture **repeatedly adding ingredients to a pitcher of mixed drink**:

- **RMSNorm only sees direction** $$=$$ when you taste the drink, you only care about the **recipe ratio** (say, 1:1 sugar to tea), not whether the pitcher holds 1 liter or 2 liters in total (the common scale);
- **The lemma in Section 2** $$=$$ if you taste sugar water or tea by themselves, doubling one does not matter; once they are mixed in the same cup, you cannot double the sugar without doubling the tea—the only change that preserves the flavor is scaling the entire pitcher uniformly;
- **The symmetry mismatch in Section 1** $$=$$ every new write adds one fixed-size scoop: when the pitcher only has half a cup left, one scoop dramatically shifts the flavor; when the pitcher is already full, the exact same scoop barely registers—so "how much liquid is currently in the pitcher (state norm)" silently governs how much each new piece of information moves the flavor;
- **Homogeneous calibration vs. forgetting** $$=$$ **calibration** swaps in a larger or smaller scoop proportional to the current volume in the pitcher so total volume genuinely stops mattering; **forgetting ($$\alpha<1$$)** pours out part of the old drink before adding the new scoop, deliberately fading old flavors so newer ones carry more weight.

---

## References

- Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., & Guestrin, C. (2024). [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620). *ICML*.
- Wang, K. A., Shi, J., & Fox, E. B. (2025). [Test-Time Regression: A Unifying Framework for Designing Sequence Models with Associative Memory](https://arxiv.org/abs/2501.12352). *arXiv:2501.12352*.
- Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). [It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization](https://arxiv.org/abs/2504.13173). *arXiv:2504.13173*. (MIRAS)
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). *NeurIPS*. (DeltaNet)
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). *ICLR*. (Gated DeltaNet)
- Wen, K., Dang, X., Lyu, K., Ma, T., & Liang, P. (2026). [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899). *arXiv:2606.16899*.
- Su, J. (2026). [From Parameter Symmetry to Effective Learning Rates (《从参数对称性到有效学习率》)](https://spaces.ac.cn/archives/11985). *Scientific Spaces*.

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
