---
title: "What Optimizer Lives Inside Linear Attention? Scale Symmetry, Writing, and Forgetting"
date: "2026-11-22"
description: "Normalized readout ignores the overall scale of the state, yet the very next write does not. Starting from that mismatch, we separate scale calibration from forgetting across DeltaNet, Gated DeltaNet, and homogeneous updates."
category: "tech"
tags: ["AI", "linear-attention", "optimization"]
math: true
published: true
---

Lately I have been reading recent work on linear attention and pretraining optimizers—especially Kaiyue Wen's [Hyperball Optimization](https://arxiv.org/abs/2606.16899) and Jianlin Su's [*From Parameter Symmetry to Effective Learning Rates*](https://spaces.ac.cn/archives/11985). That made me wonder: if linear attention can be read as Test-Time Training, how should its inner optimizer handle scale?

The starting observation is simple: **if you rescale the whole state matrix, the normalized readout does not change—yet updating both states with the same key-value pair usually gives different outputs.**

The reason sits right inside the recurrence. The old state scales up with the multiplier, while the newly written term keeps its original magnitude. The scale that the readout threw away sneaks straight back into the next step.

Below, we follow this single mismatch to see what calibration, forgetting, and state normalization are each doing. Throughout the post, we assume only that keys $$k$$ are unit-normalized, that the readout $$Sq$$ passes through RMSNorm, and that RMSNorm is treated as an exact scale-invariant map.

## 1. Readout Ignores Scale—Why Does the Update Care?

The state is a matrix $$S_t\in\mathbb R^{d_v\times d_k}$$. At each step, we receive a key-value pair $$(k_t,v_t)$$, update the state, and read it out with a query $$q_t$$:

$$
S_{t+1}=\alpha_t S_t(I-\beta_t k_tk_t^\top)+\gamma_t v_tk_t^\top,
\qquad
 y_t=\operatorname{RMSNorm}(S_{t+1}q_t).
\tag{1}
$$

For now, treat $$\alpha_t,\beta_t,\gamma_t$$ as three independent scalars—without tying them together or letting them depend on the state's overall scale. Here $$0<\alpha_t\le1$$ controls how much of the old state is retained, $$\beta_t$$ controls erasure strength, and $$\gamma_t$$ controls write strength. Expanding Eq. (1):

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

Here "state" means the matrix $$S$$, "update" means the full recurrence, and "write" refers only to the final additive term. The erasure term subtracts part of the old state's response along the current key; the forget gate $$\alpha_t$$ scales the entire old state uniformly.

Why call the last term a write? Because $$\|k_t\|_2=1$$, adding $$\gamma_t v_tk_t^\top$$ shifts the prediction $$S_tk_t$$ on the current key by $$\gamma_t v_t$$. For any other query $$q$$, the readout shifts by $$\gamma_t v_t(k_t^\top q)$$: the closer $$q$$ is to $$k_t$$, the larger the change.

Now replace $$S$$ with $$\lambda S$$ for some $$\lambda>0$$. The readout satisfies

$$
\operatorname{RMSNorm}(\lambda Sq)
=
\operatorname{RMSNorm}(Sq).
\tag{3}
$$

In the update, however, the first two terms scale by $$\lambda$$ while the last term does not. Letting $$\mathcal T$$ denote the update operator in Eq. (1), we have

$$
\mathcal T(\lambda S)-\lambda\mathcal T(S)
=(1-\lambda)\gamma vk^\top.
\tag{4}
$$

So two states with identical readouts generally no longer produce identical readouts after the very same update.

In $$S$$, the old-state terms are degree-1 homogeneous, while the new write is degree-0 homogeneous. **The larger the old state, the harder it is for the same write to move it; the smaller the old state, the easier it is to move.** Even though the state's norm never appears in the current output, it quietly controls how easily the next step can rotate the state's direction.

## 2. You Only Get One Global Scale

Following [TTT](https://arxiv.org/abs/2407.04620) and [Test-Time Regression](https://arxiv.org/abs/2501.12352), plain DeltaNet is one step of SGD on the per-step regression loss $$\frac12\|Sk-v\|_2^2$$. Look a little closer, though, and the writing side and the reading side do not share the same objective.

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

For a read pair $$(q,o^\star)$$ (where $$o$$ stands for both *oracle* and *output*), $$Sq$$ immediately passes through RMSNorm, which discards its length. What we actually care about is whether the output **points in the right direction**:

$$
\max_S \;\; \mathrm{sim}(Sq,\; o^\star). \tag{Goal 2}
$$

This "maximize similarity" can be written as cosine similarity $$\max\cos\angle(Sq,o^\star)$$, normalized distance $$\min\|\operatorname{RMSNorm}(Sq)-\operatorname{RMSNorm}(o^\star)\|_2^2$$, or any objective depending only on the positive ray $$\{a\,Sq : a>0\}$$.

Suppose we achieve Goal 1 and accept that $$(q,o^\star)$$ is in some sense another $$(k,v)$$ pair (i.e., ideal readout satisfies $$o^\star=P^*q$$); then solving Goal 1 automatically achieves Goal 2. What about the converse? Goal 1 looks too strong: Goal 2 ignores any positive rescaling of the output, so intuitively it might seem that each stored pair could carry its own scale factor.

Can it? Take two orthogonal keys with linearly independent values $$v_1,v_2$$, and suppose the state satisfies

$$
Sk_1=2v_1,\qquad Sk_2=v_2.
\tag{5}
$$

Querying $$k_1$$ or $$k_2$$ individually gives the right direction. Now ask the mixed query $$q=(k_1+k_2)/\sqrt2$$: the output points along $$2v_1+v_2$$ instead of $$v_1+v_2$$. Normalizing at the end cannot fix the wrong ratio between the two terms.

This reflects a simple linear-algebra fact:

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

This is the same fact as "a linear operator for which every vector is an eigenvector must be a scalar multiple of the identity." Here it says: **once queries mix different keys, the only thing you are free to change is a single global scale for the entire state, never the relative proportions across stored pairs.**

Consequently, requiring Goal 2 for **all** queries collapses per-query scale freedom into a single positive scalar $$c>0$$, giving the scale-calibrated objective **Goal \***:

$$
\boxed{\ \min_{S,\; c>0}\; F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2\ }
\tag{Goal *}
$$

As a set of solutions, Goal \* invents nothing new: substituting $$P=cS$$ turns it right back into ordinary least squares, and for any fixed $$c>0$$ the minimizer is $$S=P^*/c$$, all lying on the same ray $$\{aP^* : a>0\}$$. What Goal \* adds is **geometry**—it promotes the one global scale that the readout ignores into an explicit variable $$c$$.

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
| **Old state only** | GDN's forget gate. Shrinks old state relative to new write while also changing state norm and future step sizes; a constant gate introduces ridge bias. | Still lowers old-state weight relative to new writes, without depending on overall state length. Even after calibration, relative forgetting remains independently controllable. |
| **Old state and new write together** | Leaves the current readout unchanged, but alters state length and therefore shifts the relative strength of the next write. Global calibration and plain DeltaNet's sphere projection live here. | Pure change of common scale. Future updates scale along with it, so every normalized readout stays completely unchanged. |

The top row alters the balance between old and new content; the bottom row alters the shared scale of the whole state. In the left column, that shared scale still leaks into the next update; in the right column, that leakage is gone.

Notice the bottom-right cell in particular: once the update is 1-homogeneous, multiplying by a positive scalar at the end of every step—or projecting the state onto a fixed-radius sphere—does not change the readout trajectory at all. Induction proves it in one line: if two trajectories differ by a factor of $$\lambda$$, they still differ by $$\lambda$$ after the next update, and renormalizing only swaps one positive factor for another.

Throughout this post, "forgetting" refers to lowering the weight of the entire old state relative to the new write. Erasure along the current key also overwrites past associations, acting as a targeted local edit along one direction.

<details markdown="1">
<summary>An equivalent view: decaying the old state is boosting future writes</summary>

There is an equivalent way to read Eq. (1). Set $$S_t=\kappa_tQ_t$$ with $$\kappa_{t+1}=\alpha_t\kappa_t$$; then

$$
Q_{t+1}=Q_t(I-\beta_tk_tk_t^\top)
+\frac{\gamma_t}{\kappa_{t+1}}v_tk_t^\top.
\tag{7}
$$

The explicit forget gate vanishes, turning into an ever-growing multiplier on later writes. When $$\alpha_t<1$$, $$\kappa_{t+1}=\prod_{s\le t}\alpha_s$$ keeps shrinking. Under normalized readout, "continuously decaying the old state" and "continuously amplifying newer writes" trace out the exact same trajectory. Because this multiplier accumulates all past forget gates, the single-step norm $$\|v_t\|$$ alone cannot express it.

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

Take a 2D example. Keys $$e_1,e_2$$ appear with probabilities $$p_1,p_2$$, with matching values $$e_1,e_2$$. Then $$\Sigma=C=\operatorname{diag}(p_1,p_2)$$ and the ideal state is $$P^*=I$$. With the ridge term, up to a common scalar, the fixed point becomes

$$
\operatorname{diag}\left(
\frac{p_1}{p_1+\lambda},\,
\frac{p_2}{p_2+\lambda}
\right).
\tag{9}
$$

For $$p_1=0.8,p_2=0.2,\lambda=0.2$$, the two entries are $$0.8$$ and $$0.5$$. Querying $$e_1$$ or $$e_2$$ alone still points in the right direction, but the mixed query $$(e_1+e_2)/\sqrt2$$ tilts from the ideal $$45^\circ$$ down to $$\arctan(0.5/0.8)\approx32^\circ$$.

Rare keys receive fewer fresh writes while still getting decayed at every step alongside frequent keys, so they are suppressed more strongly. That is the recency ridge bias here. When $$\Sigma$$ is isotropic, the ridge only rescales the whole state; directional tilt comes from anisotropic shrinkage across frequencies.

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

Whenever the chosen radius $$R$$ is smaller than the unconstrained optimum's norm, the multiplier $$\lambda$$ is strictly positive. In the same 2D distribution with $$R=0.5$$, the direction-preserving point on the sphere is $$(0.3536,0.3536)$$ ($$45^\circ$$), yet squared error prefers $$(0.4667,0.1795)$$: it spends more of its limited norm budget on the high-frequency direction, tilting the mixed readout to about $$21^\circ$$.

Both operations produce a similar directional tilt, for completely different reasons: GDN deliberately down-weights older content, whereas sphere projection only tries to fix the norm yet accidentally distorts content ratios along the way.

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
U^+=N_F(U-\beta H),
\qquad
H=(Uk-g(U)v)k^\top.
\tag{16}
$$

The radius $$r$$ drops out completely. Decomposing $$H=\xi U+H^\perp$$ into radial and tangential parts (with $$\xi=\langle U,H\rangle_F$$), the single-step turning angle is

$$
\theta=\operatorname{atan2}\left(
\beta\|H^\perp\|_F,\,1-\beta\xi
\right).
\tag{17}
$$

Now $$\beta$$ is a scale-free directional step coefficient; how far the state rotates depends only on $$\beta$$ and the residual geometry at $$U$$. Exact population calibration zeros out the expected radial component without requiring every single stochastic step to be purely tangential.

<details markdown="1">
<summary>Why does a scale-invariant objective still need step-size preconditioning?</summary>

Eliminating $$c$$ gives the reduced objective $$G(S)=F(S,c^*(S))$$, which satisfies $$G(\lambda S)=G(S)$$. Plain fixed-step Euclidean gradient descent on $$G$$ is not enough, however, because $$\nabla G(\lambda S)=\lambda^{-1}\nabla G(S)$$. In population form, Eq. (13) actually corresponds to

$$
\nabla G(S)=(c^*)^2(S\Sigma-g^*C),
\qquad
S^+=S-\frac{\beta}{(c^*)^2}\nabla G(S).
\tag{18}
$$

It is this step-size preconditioning by $$(c^*)^{-2}$$ that makes the gradient step 1-homogeneous.

</details>

In outer scale-invariant networks, weight decay regulates the effective learning rate through parameter norm without changing the optimal solution set. In plain DeltaNet, the inner regression is not scale-invariant, so forgetting also induces genuine regression shrinkage. Homogeneous calibrated updates strip state length out of the angular step size; if you still want to down-weight older memories, you can add a forget gate separately.

## 6. Three Calibration Schemes, and an Independent Forget Gate

The optimal scale $$c^*(S)$$ gives us a reference point. There are three straightforward ways to treat the scale variable $$c$$ in Goal \*.

**Fix $$c$$.** Setting $$c=1$$ gives plain DeltaNet, letting state norm adapt on its own to the regression target; adding decay on the old state yields GDN. It is simple, at the cost of keeping directional updates coupled to state length.

**Learn $$c$$.** Update $$S$$ and $$u=\log c$$ jointly on $$F(S,c)$$. Letting $$P=cS$$, $$z=Pk$$, and $$e=z-v$$, using step size $$\beta/c^2$$ on $$S$$ and $$\eta$$ on $$u$$ collapses both updates into a single matrix:

$$
P^+=a\big[P-\beta ek^\top\big],
\qquad
a=\exp(-\eta z^\top e).
\tag{19}
$$

This sits in the bottom-left cell of our table: $$a$$ scales the retained state and the new write together—shrinking when the prediction overshoots along its own direction and growing when it undershoots—recalibrating the scale relationship between the state and future writes.

**Eliminate $$c$$.** Plugging $$c^*(S)$$ from Eq. (12) into Eq. (13) gives the homogeneous calibrated update, which stops fitting irrelevant absolute lengths and introduces no extra ridge. Estimating $$g^*$$ with smoothed statistics such as $$\operatorname{EMA}(\|Sk\|^2)/\operatorname{EMA}((Sk)^\top v)$$ yields a practical middle ground between one-step online learning and exact population calibration.

All three schemes address calibration. Whether to additionally decay the old state is a separate decision.

### A corollary: calibration does not abolish forgetting

Consider the top-right cell of the table: combine exact calibrated writing with decay applied only to the old state,

$$
S^+=\alpha S(I-\beta kk^\top)+\beta g^*(S)vk^\top.
\tag{20}
$$

Define the scale-free directional curvature

$$
\chi(S)=\frac{\langle S,S\Sigma\rangle_F}{\|S\|_F^2}.
$$

Solving for a directional fixed point $$\mathbb E[S^+\mid S]=\mu S$$ of the population update gives

$$
\boxed{
S\propto C(\Sigma+\lambda I)^{-1},
\qquad
\lambda=\frac{1-\alpha}{\alpha}\chi(S).
}
\tag{21}
$$

<details markdown="1">
<summary>Two-step derivation</summary>

Let $$A=\langle S,S\Sigma\rangle_F$$ and $$B=\langle S,C\rangle_F$$; exact calibration satisfies $$g^*B=A$$. The directional fixed-point equation is

$$
\alpha S-\alpha\beta S\Sigma+\beta g^*C=\mu S.
$$

Taking the inner product with $$S$$ yields $$\mu=\alpha+\beta(1-\alpha)\chi$$. Rearranging the equation as

$$
S\left(\Sigma+\frac{\mu-\alpha}{\alpha\beta}I\right)
=\frac{g^*}{\alpha}C
$$

immediately gives Eq. (21).

</details>

This ridge strength no longer depends on the state's overall length, yet it still alters the relative weights across directions. Calibration removes artifacts of arbitrary scale; it does not eliminate forgetting itself.

So the top-right cell does not force the model to keep all old content—it simply makes forgetting independent of whatever state norm happens to be chosen.

## 7. What Changes Should Be Left to Forgetting?

Channel-wise gates apply different decay rates to different features. By Section 2, this generally changes the readout direction on mixed queries, so it modifies stored content rather than acting as a harmless common scale. Its role is feature-selective forgetting; which directions ought to be kept depends on the task.

Likewise, targets drift in real sequences, and the stationary $$P^*$$ is only an analytical reference. The goal is not to eliminate every ridge, but to tell apart whether suppressing a direction serves to track new information or is merely an unintended byproduct of scale bookkeeping.

The readout ignores a common scale, and the update can respect that same symmetry. **Scaling the old state and the new write together turns state length into a pure convention, leaving the balance between old and new content to forgetting.**

## 8. Takeaway

- **One symmetry mismatch drives most of the behavior.** RMSNorm makes the readout invariant to $$S\mapsto\lambda S$$, whereas in plain DeltaNet the old state is degree-1 homogeneous and the write is degree-0 homogeneous—so state length quietly controls how strongly each new write moves the state.
- **Readout only leaves one global scale free (Goal \*).** Once queries mix different keys, per-memory scale differences leak into readout directions. All queries share only a single global scale $$c>0$$, leading directly to the calibrated regression objective $$F(S,c)=\frac12\mathbb E\|cSk-v\|_2^2$$.
- **The $$2\times 2$$ table separates calibration from forgetting.** Multiplying only the old state shifts the balance between old and new content (forgetting), while multiplying old and new together changes the shared scale; whether the write scales homogeneously as $$g(S)vk^\top$$ determines if that shared scale leaks into the next directional update.
- **Separate intentional forgetting from scale-bookkeeping artifacts.** A gate $$\alpha<1$$ on the old state alone always produces a ridge—even after exact homogeneous calibration, except its strength no longer depends on state norm. By contrast, the ridge from locking plain DeltaNet onto a sphere is purely an artifact of a non-homogeneous write; once the update is 1-homogeneous, sphere projection becomes a harmless gauge convention.
- **Effective learning rates decouple cleanly.** In GDN, $$\alpha_t$$ simultaneously regulates directional step size via state norm and decays older content. Aligning both the inner objective and the update with readout scale symmetry makes $$\beta$$ a scale-free directional step size, leaving the forget gate with the single job of deciding how much past content to retain.

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
