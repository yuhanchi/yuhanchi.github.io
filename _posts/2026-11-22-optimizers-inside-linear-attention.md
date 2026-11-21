---
title: "What Optimizer Lives Inside Linear Attention? A Scale-Invariance View of DeltaNet"
date: "2026-11-22"
description: "Starting from four assumptions every modern linear-attention block already makes, we use the global scale symmetry created by RMSNorm to view DeltaNet, Gated DeltaNet, and calibrated/homogeneous updates under one lens."
category: "tech"
tags: ["AI", "linear-attention", "optimization"]
math: true
published: true
---
Lately I spent a little over a day reading recent work on linear attention and pretraining optimizers, and noticed some surprisingly clean connections between the two. Working through them led to a few neat conclusions—and a few open questions I want to write down here.

Let us start from four assumptions that almost every modern linear-attention block already makes:

1. $$k$$ is normalized;
2. $$q$$ is normalized;
3. $$o = S q$$, and $$o$$ goes through a normalization (RMSNorm) before anything else touches it;
4. We may view Linear Attention as Test-Time Training (TTT).

After reading the posts by Kaiyue Wen ([Hyperball Optimization](https://arxiv.org/abs/2606.16899)) and Jianlin Su ([*From Parameter Symmetry to Effective Learning Rates*](https://spaces.ac.cn/archives/11985)) on optimizers and effective learning rates, a fun question comes up quite naturally: Linear Attention's Test-Time *Training* has the word *training* in its name, so **what does its optimizer actually look like?** And going one step further: can the special structure of the Linear Attention module itself suggest a better, or at least a more principled, optimization strategy?

This post does not propose a new architecture. It offers **a perspective**. Starting from nothing more than the four assumptions above, we will find that DeltaNet, Gated DeltaNet, and a few calibrated or homogeneous variants all show up as different answers to one small question.

---

## 1. Setup

The memory is a matrix $$S_t \in \mathbb{R}^{d_v \times d_k}$$. At each step we see a pair $$(k_t, v_t)$$, update the memory, and read it with $$q_t$$:

$$
S_{t+1} = S_t - \beta_t (S_t k_t - v_t) k_t^\top, \qquad
o_t = S_{t+1} q_t, \qquad
y_t = W_o\big[\mathrm{RMSNorm}(o_t) \odot \mathrm{SiLU}(g_t)\big].
$$

This update is exactly one SGD step on the per-token regression loss $$\frac{1}{2}\|S k_t - v_t\|_2^2$$. That is the TTT reading: the "inner model" is the linear map $$S$$, the "training data" is the stream of $$(k_t, v_t)$$, and the inner optimizer is SGD with step size $$\beta_t$$.

Notation used below: the key second moment $$\Sigma = \mathbb{E}[k k^\top] \succ 0$$, the cross-covariance $$C = \mathbb{E}[v k^\top]$$, the ordinary least-squares solution $$P^* = C\Sigma^{-1}$$, the old prediction $$p_t = S_t k_t$$, and the Frobenius normalization operator $$N_F(X) = X / \|X\|_F$$.

---

## 2. The Relationship Between $$q$$ and $$k$$

One thing I find genuinely curious is how we should think about $$q$$.

The usual story goes: $$S$$ *is* the memory. $$S k \approx v$$ means it has memorized something, and $$q$$ is a test vector that reads the memory back out. Think about that for a second and it feels slightly uncanny: in ordinary softmax attention we tend to feel that $$q$$ and $$k$$ do not really "live in the same linear space," yet here we write with $$k$$ and read with $$q$$ through the *same* linear map, and it works.

Think one step further and you notice that the write side and the read side do not even share the same objective.

For a write pair $$(k, v)$$, we want $$S k$$ to match $$v$$ as closely as possible:

$$
\min_S \;\; \mathbb{E}\,\|S k - v\|_2^2. \tag{Goal 1}
$$

For a read pair $$(q, o)$$ (where $$o$$ stands for both *oracle* and *output*, lol), what we really want is for the output to *point the right way*, because the very next thing that happens to $$o_t$$ is RMSNorm, which throws away its length:

$$
\max_S \;\; \mathrm{sim}(S q,\; o^\star). \tag{Goal 2}
$$

This "max similarity" can be written in several equivalent ways:

- cosine similarity: $$\max\; \cos\angle(S q,\; o^\star)$$;
- normalized distance: $$\min\; \|\mathrm{RMSNorm}(S q) - \mathrm{RMSNorm}(o^\star)\|_2^2$$;
- or anything that only depends on the positive ray $$\{a\, S q : a > 0\}$$.

Suppose we achieve Goal 1, and we also accept that $$(q, o)$$ is in some sense just another $$(k, v)$$ (i.e., the oracle satisfies $$o^\star = P^* q$$). Then solving Goal 1 well implies Goal 2.

What about the converse? Goal 1 is clearly too strong: Goal 2 does not care about any positive rescaling of the output. Each query could, in principle, accept its own scale. But that does not mean we cannot work backwards from Goal 2.

> **Side remark (assumption 2 is free).** Since $$o_t = S_{t+1} q_t$$ is linear in $$q_t$$ and ideal RMSNorm ($$\varepsilon \to 0$$) satisfies $$\mathrm{RMSNorm}(a\, o) = \mathrm{RMSNorm}(o)$$ for all $$a > 0$$, normalizing $$q$$ only multiplies $$o$$ by a positive scalar, which RMSNorm then eats. Under an ideal output norm, normalizing $$q$$ is redundant in both the forward and backward passes. Normalizing $$k$$ is a very different story, as we will see shortly.

---

## 3. A Lemma: You Only Get One Global Scale

Can each memory pick its own scale? Here is a two-second counterexample. Take two orthogonal keys $$k_1 \perp k_2$$ and store them with different multipliers:

$$
S k_1 = 2 v_1, \qquad S k_2 = 1 \cdot v_2.
$$

Each individual query reads back the right direction. Now ask the mixed query $$q = \frac{1}{\sqrt{2}}(k_1 + k_2)$$. The ideal output points along $$v_1 + v_2$$, yet we get $$2 v_1 + v_2$$. **The direction is wrong.** Per-memory scaling leaks into directions as soon as queries mix.

This is no accident:

> **Lemma (Projective uniqueness).** Let $$\operatorname{rank}(M) \ge 2$$. If for every $$q \notin \ker M$$ the vector $$S q$$ is a positive multiple of $$M q$$, then $$S = c M$$ for a unique global constant $$c > 0$$.

<details markdown="1">
<summary>Proof of the projective uniqueness lemma</summary>

Write $$S q = c(q)\, M q$$. Whenever $$M q_1, M q_2$$ are linearly independent, linearity gives

$$
c(q_1 + q_2)(M q_1 + M q_2) = S(q_1 + q_2) = c(q_1) M q_1 + c(q_2) M q_2,
$$

so comparing coefficients forces $$c(q_1) = c(q_2) = c(q_1 + q_2)$$. If $$M q_1, M q_2$$ are collinear, since $$\operatorname{rank}(M) \ge 2$$ we can bridge them through a third vector $$q_3$$ whose image $$M q_3$$ is independent of both, giving $$c(q_1) = c(q_3) = c(q_2)$$. Thus $$c$$ is constant on $$\mathbb{R}^{d_k} \setminus \ker M$$. Finally, for any $$q_0 \in \ker M$$, applying the same relation to $$q_1 + q_0$$ gives $$S(q_1 + q_0) = c M(q_1 + q_0) = c M q_1 = S q_1$$, so $$S q_0 = 0$$. Hence $$S = c M$$. $$\blacksquare$$

</details>

Once we require Goal 2 for *all* queries, per-query freedom collapses to **one global positive scale**. That gives the objective we will call Goal \*, or more descriptively the **Calibrated · Homogeneous · Invariant loss** (**CHI loss** for short):

$$
\boxed{\ \min_{S,\; c > 0}\; F(S, c) = \frac{1}{2}\,\mathbb{E}\,\|c\, S k - v\|_2^2\ } \tag{CHI / Goal *}
$$

Why those three words? Because $$c$$ explicitly **calibrates** the global scale between $$S k$$ and $$v$$; the pair $$(S, c)$$ forms a **homogeneous** scale family (and once $$c$$ is eliminated via $$c^*(S)$$, the reduced objective is homogeneous of degree 0 in $$S$$); and the readout goal is scale-**invariant** along rays $$\{a S : a > 0\}$$.

Let me be upfront about one thing, because it is the crux of this post. As a *set of solutions*, the CHI loss is nothing new: substitute $$P = c S$$ and you get ordinary least squares. What it adds is **geometry**. It turns the single direction the readout does not care about—the global scale—into an explicit variable. And once a symmetry becomes an explicit variable, we get to choose how the optimizer treats it.

---

## 4. Four Nice Properties of the CHI Loss

This objective behaves remarkably cleanly:

1. **Fix any $$c$$, and it is ordinary regression.** For fixed $$c > 0$$, $$F(\cdot, c)$$ is a strongly convex quadratic in $$S$$ with unique minimizer $$S^*(c) = P^* / c$$.
2. **It comes as a scale family.** For all $$a > 0$$, $$F(a S,\; c / a) = F(S, c)$$. Regardless of which $$c$$ we pick, optimal solutions differ only by a positive scalar and all lie on the same ray $$\{a P^* : a > 0\}$$.
3. **We can also optimize $$(S, c)$$ jointly.** Running SGD on $$(S, c)$$ together has no spurious interior stationary points: every interior stationary point satisfies $$c S = P^*$$. Moreover, as long as $$c \ge c_{\min} > 0$$, the gradient in $$S$$ satisfies a Polyak–Łojasiewicz (PL) inequality:

   $$
   \|\nabla_S F(S, c)\|_F^2 \ge 2 c_{\min}^2 \lambda_{\min}(\Sigma)\bigl(F(S, c) - F^*\bigr).
   $$

4. **Property 2 implies an Euler identity that governs the rest of the post.** Differentiate $$f(a) = F(a S,\; c / a) \equiv \text{const}$$ at $$a = 1$$:

   $$
   0 = f'(1) = \langle S,\; \nabla_S F \rangle_F - c\,\partial_c F
   \quad\Longrightarrow\quad
   \boxed{\ \langle S,\; \nabla_S F(S, c)\rangle_F = c\,\frac{\partial F}{\partial c} = c\big(c\,\langle S,\; S\Sigma\rangle_F - \langle S,\; C\rangle_F\big)\ } \tag{Euler}
   $$

<details markdown="1">
<summary>Proof of the PL inequality in Property 3</summary>

Set $$E = c S - P^*$$. Since $$C = P^* \Sigma$$, expanding around the least-squares solution $$P^*$$ eliminates the cross term and gives $$F(S, c) - F^* = \frac{1}{2}\operatorname{tr}(E \Sigma E^\top)$$, while $$\nabla_S F(S, c) = c E \Sigma$$. Because $$\Sigma \succ 0 \implies \Sigma^2 \succeq \lambda_{\min}(\Sigma)\Sigma$$, for any $$c \ge c_{\min} > 0$$ we have:

$$
\|\nabla_S F(S, c)\|_F^2 = c^2 \operatorname{tr}(E \Sigma^2 E^\top) \ge c^2 \lambda_{\min}(\Sigma)\operatorname{tr}(E \Sigma E^\top) \ge 2 c_{\min}^2 \lambda_{\min}(\Sigma)\bigl(F(S, c) - F^*\bigr). \quad \blacksquare
$$

</details>

Read the left-hand side of (Euler) geometrically. Picture $$S$$ as an arrow in matrix space. Its gradient splits orthogonally into a **tangential** component (rotating the arrow, changing directions, which is all RMSNorm sees) and a **radial** component (stretching or shrinking the arrow, which RMSNorm ignores). The Euler identity says that **the radial component of the $$S$$-gradient is exactly $$c\,\partial_c F$$.**

Setting $$\partial_c F = 0$$ gives the optimal scale for the current state (assuming positive numerator and denominator):

$$
c^*(S) = \frac{\langle S,\; C\rangle_F}{\langle S,\; S\Sigma\rangle_F} = \frac{\mathbb{E}\,\langle S k,\; v\rangle}{\mathbb{E}\,\|S k\|_2^2}.
$$

When $$c = c^*(S)$$, the expected gradient is **purely tangential**. When $$c$$ takes any other value, part of the gradient pushes along the radial direction that the output norm discards.

So everything boils down to one small question:

> **What do we do with $$c$$?**

There are three natural answers: **fix it**, **learn it**, or **eliminate it**. They correspond to the three mainlines below.

---

## 5. Mainline 1: Fix $$c \equiv 1$$ and Let the Norm Run Free (DeltaNet / Gated DeltaNet)

The simplest choice is to hard-code $$c \equiv 1$$ and put no constraint on $$\|S_t\|_F$$. That is plain DeltaNet.

Here the radial force from the Euler identity is not wasted—it acts as self-calibration. If $$S$$ is too short it gets stretched, if too long it gets shrunk, and the state settles at the unique point on the ray where $$c^*(S) = 1$$, namely $$S = P^*$$ (quick check: $$c^*(P^*) = \operatorname{tr}(P^*\Sigma P^{*\top}) / \operatorname{tr}(P^*\Sigma P^{*\top}) = 1$$).

### Why $$k$$ must be normalized (curvature calibration and normalized step size)

For one SGD step on $$\frac{1}{2}\|S k_t - v_t\|_2^2$$ with step size $$\eta_t$$, the new residual satisfies:

$$
S_{t+1} k_t - v_t = \big(1 - \eta_t \|k_t\|_2^2\big)(S_t k_t - v_t).
$$

Along this rank-one direction, the exact curvature is $$\|k_t\|_2^2$$. Whenever $$\eta_t \|k_t\|_2^2 \ge 2$$, the step stops shrinking—oscillating or diverging. Normalizing $$k_t$$ (equivalently, dividing by the exact one-step curvature $$\|k_t\|_2^2$$) pins the stable contraction interval to $$\beta_t \in (0, 2)$$: $$\beta_t \in (0, 1]$$ reduces the current key's residual monotonically ($$\beta_t = 1$$ clears it in one step), while $$\beta_t \in (1, 2)$$ overshoots with a sign flip while still contracting.

One precision check: $$\beta_t$$ is only the **Euclidean normalized step size** along the current key $$k_t$$. Successive keys are not orthogonal, and $$\beta_t$$ does not decompose the update against the tangent space of the existing state $$S_t$$, so it is not yet an "effective learning rate" for how much the memory's *direction* moves.

<details markdown="1">
<summary>Proof: why the single-step contraction interval is (0, 2)</summary>

For the instantaneous loss $$\ell_t(S) = \frac{1}{2}\|S k_t - v_t\|_2^2$$, the gradient is $$\nabla_S \ell_t(S_t) = (S_t k_t - v_t) k_t^\top$$. Right-multiplying $$S_{t+1} = S_t - \eta_t (S_t k_t - v_t) k_t^\top$$ by $$k_t$$ and subtracting $$v_t$$ yields the residual recurrence above. Taking squared norms gives:

$$
\ell_t(S_{t+1}) = \frac{1}{2}\|S_{t+1} k_t - v_t\|_2^2 = \big(1 - \eta_t \|k_t\|_2^2\big)^2 \ell_t(S_t).
$$

When $$S_t k_t \neq v_t$$, the loss strictly decreases in one step if and only if $$\lvert 1 - \eta_t \|k_t\|_2^2 \rvert < 1$$, i.e., $$0 < \eta_t \|k_t\|_2^2 < 2$$. At $$\eta_t \|k_t\|_2^2 = 1$$ the residual vanishes in one step (the exact rank-one Newton step); at $$\ge 2$$ it fails to shrink. Setting $$\|k_t\|_2 = 1$$ and $$\beta_t = \eta_t$$ decouples stability from key length and fixes it to $$\beta_t \in (0, 2)$$. $$\blacksquare$$

</details>

<details markdown="1">
<summary>Derivation: NLMS and one-step Newton curvature from physical-scale and orthogonal equivariance</summary>

Drop the outer projections for a moment and keep unnormalized nonzero keys $$k_t$$ without forgetting ($$\alpha_t = 1$$). Write $$e_t \triangleq S_t k_t - v_t$$ and $$G_t \triangleq \nabla_S \ell_t(S_t) = e_t k_t^\top$$. Under any rescaling and orthogonal change of coordinates $$k_t' = a U k_t$$, $$v_t' = b V v_t$$ ($$a, b > 0$$, $$U \in O(d_k)$$, $$V \in O(d_v)$$), consistency of $$S_t' k_t' = v_t'$$ requires $$S_t' = \frac{b}{a} V S_t U^\top$$. The error transforms as $$e_t' = b V e_t$$, whereas the gradient transforms as:

$$
G_t' = (S_t' k_t' - v_t')(k_t')^\top = a b\, V G_t U^\top.
$$

State and gradient scale differently ($$\frac{b}{a}$$ vs. $$a b$$), so a fixed-step Euclidean update picks up a spurious $$a^2$$ factor. Asking an update increment $$\Delta S = \phi(S, k, v)$$ to be equivariant—$$\phi(\frac{b}{a} V S U^\top, a U k, b V v) = \frac{b}{a} V \phi(S, k, v) U^\top$$—and plugging in $$U = I$$, $$V = I$$, $$a = 1/\|k\|_2$$, $$b = 1/\|v\|_2$$ yields the general equivariant form:

$$
\phi(S, k, v) = \frac{\|v\|_2}{\|k\|_2}\, F\!\left(\frac{\|k\|_2}{\|v\|_2} S,\; \hat{k},\; \hat{v}\right).
$$

Taking the simplest linear choice $$F(X, \hat{k}, \hat{v}) = \beta(\hat{v} - X \hat{k})\hat{k}^\top$$ immediately recovers normalized LMS (NLMS):

$$
S_{t+1} = S_t - \frac{\beta_t}{\|k_t\|_2^2}(S_t k_t - v_t) k_t^\top.
$$

Likewise, along the 1D slice $$\varphi(\zeta) \triangleq \ell_t(S_t + \zeta G_t) = \frac{1}{2}\|(1 + \zeta \|k_t\|_2^2) e_t\|_2^2$$, we have $$\varphi''(0) = \|k_t\|_2^4 \|e_t\|_2^2 = \|k_t\|_2^2 \|G_t\|_F^2$$, so the exact one-step Newton step size is also $$1 / \|k_t\|_2^2$$. Modern linear attention normalizes $$k_t$$ before writing, keeping outer $$W_k \mapsto a W_k$$ invariant and fixing the directional curvature at $$1$$. $$\blacksquare$$

</details>

### Relative write gain $$\rho_t$$ and the tangential effective step size

Gated DeltaNet (GDN) multiplies the old state by a scalar forget gate $$\alpha_t \in (0, 1]$$:

$$
S_{t+1} = \alpha_t S_t (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top.
$$

Split the nonzero state into norm and direction, $$r_t = \|S_t\|_F > 0$$ and $$U_t = S_t / r_t$$, with $$\hat{v}_t = v_t / \|v_t\|_2$$. Dividing by $$\alpha_t r_t$$ and renormalizing reveals the direction dynamics that the readout actually sees:

$$
U_{t+1} = N_F\!\Big(U_t (I - \beta_t k_t k_t^\top) + \rho_t\, \hat{v}_t k_t^\top\Big), \qquad
\boxed{\ \rho_t \triangleq \frac{\beta_t\, \|v_t\|_2}{\alpha_t\, r_t}\ }
$$

Here $$\rho_t$$ measures the **normalized gain of the new write relative to the old state**. When $$\beta_t \|v_t\|_2 / \alpha_t$$ is roughly comparable across steps, a smaller initial $$r_t$$ gives a larger $$\rho_t$$, making early writes shift the direction more readily—though $$\rho_t$$ alone does not guarantee a neat "automatic warm-up" schedule for the angular step. Conversely, when $$\alpha \in (0, 1)$$ is constant, decay holds the norm on a plateau: dropping the erasure and cross terms and treating $$\alpha, \beta, \|v\|_2$$ as constants gives the steady-state approximation $$\bar{\rho} \approx \sqrt{1 - \alpha^2} / \alpha \approx \sqrt{2(1 - \alpha)}$$ (for $$\alpha$$ close to 1). Intuitively, this echoes the weight-decay mechanism in Hyperball and Jianlin Su's analysis, where decay regulates the relative write amplitude via the equilibrium norm.

<details markdown="1">
<summary>Derivation: steady-state relative write gain approximation</summary>

With $$\|k_t\|_2 = 1$$, write GDN as $$S_{t+1} = \alpha_t S_t + (\beta_t v_t - \alpha_t \beta_t S_t k_t) k_t^\top$$. Using $$\langle S_t, w k_t^\top \rangle_F = \langle S_t k_t, w \rangle$$ to expand $$r_{t+1}^2 = \|S_{t+1}\|_F^2$$ gives the exact norm recurrence:

$$
r_{t+1}^2 = \alpha_t^2 r_t^2 + \beta_t^2 \|v_t\|_2^2 - \alpha_t^2 \beta_t (2 - \beta_t)\|S_t k_t\|_2^2 + 2\alpha_t \beta_t (1 - \beta_t)\langle S_t k_t, v_t\rangle.
$$

Dropping the erasure term ($$\|S_t k_t\|_2^2$$) and cross term ($$\langle S_t k_t, v_t\rangle$$) and treating $$\alpha \in (0, 1)$$, $$\beta$$, and $$\|v\|_2$$ as constants reduces this to $$r_{t+1}^2 \approx \alpha^2 r_t^2 + \beta^2 \|v\|_2^2$$. Setting $$r_{t+1} = r_t = r_\star$$ gives $$r_\star^2 \approx \frac{\beta^2 \|v\|_2^2}{1 - \alpha^2}$$. Substituting into $$\rho_t = \frac{\beta \|v\|_2}{\alpha r_t}$$ cancels both $$\beta$$ and $$\|v\|_2$$:

$$
\bar{\rho} \approx \frac{\beta \|v\|_2}{\alpha r_\star} = \frac{\sqrt{1 - \alpha^2}}{\alpha} = \frac{\sqrt{(1 - \alpha)(1 + \alpha)}}{\alpha} \approx \sqrt{2(1 - \alpha)} \qquad (\alpha \to 1^-). \quad \blacksquare
$$

</details>

Strictly speaking, however, $$\rho_t$$ is still **not** the angular effective learning rate. To see how far the direction actually rotates, we have to orthogonally decompose the **entire residual update** onto the tangent space at $$U_t$$.

For $$r_t > 0$$ and $$\|k_t\|_2 = 1$$, let $$u_t = U_t k_t$$, $$e_t = u_t - v_t / (\alpha_t r_t)$$, and $$H_t = e_t k_t^\top$$, so the update inside $$N_F(\cdot)$$ is $$U_t - \beta_t H_t$$. Projecting $$H_t$$ onto $$U_t$$ under the Frobenius inner product gives:

$$
\begin{aligned}
\xi_t &\triangleq \langle U_t, H_t \rangle_F = u_t^\top e_t, \qquad H_t^\perp \triangleq H_t - \xi_t U_t, \\
U_t - \beta_t H_t &= (1 - \beta_t \xi_t) U_t - \beta_t H_t^\perp, \qquad \langle U_t, H_t^\perp \rangle_F = 0.
\end{aligned}
$$

Whenever the radial coefficient $$1 - \beta_t \xi_t > 0$$, pulling it outside $$N_F(\cdot)$$ writes the update cleanly as "step along the projected residual, then normalize":

$$
U_{t+1} = N_F\!\left(U_t - \eta_t^{\mathrm{tan}} H_t^\perp\right), \qquad
\boxed{\ \eta_t^{\mathrm{tan}} = \frac{\beta_t}{1 - \beta_t \xi_t} = \frac{\beta_t}{1 - \beta_t \|u_t\|_2^2 + \rho_t\, u_t^\top \hat{v}_t}\ }
$$

This $$\eta_t^{\mathrm{tan}}$$ is the **tangential effective step size** relative to $$H_t^\perp$$: the radial components of both erasure and write enter the denominator, while their tangential components remain in $$H_t^\perp$$. Even then, the step-size coefficient is not the actual rotation angle. Letting $$\theta_t = \arccos\langle U_t, U_{t+1} \rangle_F$$, whenever the next state is nonzero we have:

$$
\|H_t^\perp\|_F^2 = \|e_t\|_2^2 - \xi_t^2, \qquad
\theta_t = \operatorname{atan2}\!\left(\beta_t \|H_t^\perp\|_F,\; 1 - \beta_t \xi_t\right).
$$

Only when $$1 - \beta_t \xi_t > 0$$ and $$\eta_t^{\mathrm{tan}} \|H_t^\perp\|_F \ll 1$$ do we have $$\theta_t \approx \eta_t^{\mathrm{tan}} \|H_t^\perp\|_F$$; when the radial factor is zero or negative, use $$\operatorname{atan2}$$ directly rather than reading $$\eta_t^{\mathrm{tan}}$$ as a small-step learning rate. Notice too that "effective" here is tied to the chosen residual scaling rather than being a universal scalar set solely by $$\alpha_t$$ or $$r_t$$ (so $$\bar{\rho}$$ cannot directly replace the steady-state turning angle). And at step $$r_t = 0$$, the direction decomposition is undefined and the state is simply seeded by the GDN update itself.

### The forget gate is a schedule for an auxiliary scale

This is my favorite identity in the post. Keep $$c \equiv 1$$ fixed, relabel GDN's actual state as $$P_t$$, and introduce an **auxiliary scale** $$\lambda_0 = 1$$, $$\lambda_{t+1} = \alpha_t \lambda_t > 0$$, along with an auxiliary state $$\widetilde{S}_t = P_t / \lambda_t$$. Dividing the GDN update by $$\lambda_{t+1}$$ gives:

$$
\widetilde{S}_{t+1} = \widetilde{S}_t (I - \beta_t k_t k_t^\top) + \frac{\beta_t}{\lambda_{t+1}}\, v_t k_t^\top.
$$

Substituting $$P_t = \lambda_t \widetilde{S}_t$$ back recovers:

$$
P_{t+1} = \frac{\lambda_{t+1}}{\lambda_t}\, P_t (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top, \qquad \frac{\lambda_{t+1}}{\lambda_t} = \alpha_t.
$$

In auxiliary coordinates, the exact same GDN trajectory reads as "no explicit forget gate, with write amplitudes rescaled over time." Let us be careful about what this is and is not: it is a state reparameterization, not a release of $$c \equiv 1$$. The underlying objective is still $$\frac{1}{2}\|P_t k_t - v_t\|_2^2 = \frac{1}{2}\|\lambda_t \widetilde{S}_t k_t - v_t\|_2^2$$, and the auxiliary write divides by $$\lambda_{t+1}$$ rather than $$\lambda_t$$, so this is not plain SGD on $$F(\widetilde{S}, \lambda_t)$$. It gives a scale-schedule interpretation of the forget gate, rather than eliminating forgetting or turning into the residual-driven calibration of the next section.

### What you get and what you pay

Because the recurrence is affine in $$S_t$$, it is natively parallel with chunkwise WY—that is why Gated DeltaNet is the workhorse. The price of a constant $$\alpha \in (0, 1)$$ is that rarely seen keys get discounted more often, so the mean fixed point carries a recency ridge:

$$
S_\star^{\mathrm{GDN}} = \frac{1}{\alpha}\, C\!\left(\Sigma + \frac{1 - \alpha}{\alpha \beta} I\right)^{-1}.
$$

<details markdown="1">
<summary>Proof of the GDN mean fixed point</summary>

Assume constant $$\alpha \in (0, 1)$$, $$\beta > 0$$, and i.i.d. pairs $$(k_t, v_t)$$ with finite second moments, $$\Sigma = \mathbb{E}[k_t k_t^\top]$$, $$C = \mathbb{E}[v_t k_t^\top]$$. Since the current sample is independent of $$S_t$$, taking expectations in the GDN recurrence gives:

$$
\mathbb{E}[S_{t+1}] = \alpha\, \mathbb{E}[S_t](I - \beta \Sigma) + \beta C.
$$

Setting $$\mathbb{E}[S_{t+1}] = \mathbb{E}[S_t] = S_\star^{\mathrm{GDN}}$$ yields $$S_\star^{\mathrm{GDN}}[(1 - \alpha)I + \alpha\beta \Sigma] = \beta C$$. Because $$\Sigma \succeq 0$$ and $$\alpha < 1$$, the matrix in brackets is positive definite and invertible, giving the formula above. (This characterizes the fixed point of the mean recurrence; actual convergence also requires step-size stability.) $$\blacksquare$$

</details>

In a drifting world, this ridge is the whole point of forgetting. In a stationary one, it is a systematic bias. Keep that tension in mind—it comes back at the end.

---

## 6. Mainline 2: Learn $$c$$ by Joint Online SGD on $$(S, \log c)$$

The per-token CHI loss $$\ell_t(S, c) = \frac{1}{2}\|c S k_t - v_t\|_2^2$$ already has two variables. **So why not run online SGD on both?**

Two clean choices make the update neat. First, the split $$(S, c) \mapsto (b S, c / b)$$ is arbitrary, so we precondition the $$S$$-step by $$\eta_t = \beta_t / c_t^2$$ to make the step invariant to that reparameterization. Second, to keep $$c > 0$$ for free, we update $$u = \log c$$ in log space:

$$
\begin{aligned}
z_t &= c_t S_t k_t, \qquad e_t = z_t - v_t, \\
S_{t+1} &= S_t - \frac{\beta_t}{c_t}\, e_t k_t^\top, \qquad
u_{t+1} = u_t - \gamma_t\, z_t^\top e_t, \qquad c_{t+1} = e^{u_{t+1}}.
\end{aligned}
$$

(Quick check: $$\nabla_S \ell_t = c_t\, e_t k_t^\top$$, and $$\partial_u \ell_t = c_t\,\partial_c \ell_t = z_t^\top e_t$$.)

Now define the effective state $$P_t = c_t S_t$$ and the step-to-step scale ratio $$a_t = c_{t+1} / c_t = \exp(-\gamma_t z_t^\top e_t)$$. Multiply the $$S$$-update by $$c_{t+1}$$, and **$$S$$ and $$c$$ disappear completely—we only need to maintain one matrix:**

$$
\boxed{\
z_t = P_t k_t, \quad e_t = z_t - v_t, \quad
a_t = \exp\!\big(-\gamma_t\, z_t^\top e_t\big), \quad
P_{t+1} = a_t\big[P_t - \beta_t e_t k_t^\top\big]\ }
$$

Three behaviors fall right out of this boxed rule:

1. **A two-sided gate driven by the residual.** $$z_t^\top e_t = \|z_t\|_2^2 - z_t^\top v_t$$ is a single-sample estimate of the radial gradient from the Euler identity. When the prediction overshoots along its own direction, $$a_t < 1$$ (shrink); when it undershoots, $$a_t > 1$$ (grow); at zero residual, $$a_t = 1$$. Unlike a forget gate, it can exceed 1.
2. **It multiplies the new write too.** GDN's $$\alpha_t$$ multiplies only the old memory, whereas $$a_t$$ sits outside the bracket and scales the retained memory and the new write together—because $$c$$ calibrates the *entire* linear map.
3. **One-step global calibration.** Although the current step's RMSNorm eats $$a_t$$, $$a_t$$ genuinely changes the next predictions $$z_{t+1}$$ and residuals $$e_{t+1}$$. Suppose the global scale of $$v$$ suddenly drifts. Plain DeltaNet only fixes keys as they happen to reappear, leaving the memory at *mixed* scales—running straight into the counterexample from Section 3. Here $$a_t$$ broadcasts the rescaling across the whole memory matrix in a single step.

**Engineering details (part of the method, not decoration).** Writing $$c = e^u > 0$$ does not by itself prevent $$c \to 0$$ or an exploding exponent. Just like the curvature argument for $$k$$ in Section 5, the local Gauss–Newton curvature along $$u = \log c$$ is $$\|z_t\|_2^2$$, so we normalize the step and clip the exponent:

$$
\tilde{\gamma}_t = \frac{\gamma_0}{\|z_t\|_2^2 + \|v_t\|_2^2 + \epsilon}, \qquad
a_t = \exp\!\Big(\operatorname{clip}\big(-\tilde{\gamma}_t\, z_t^\top e_t,\; -\delta_{\max},\; \delta_{\max}\big)\Big).
$$

One design note: from a single token alone, $$a_t$$ cannot tell "this token's value is unusually salient" from "the global scale has drifted." Normalization and clipping are what stop one loud token from rescaling the entire memory.

To first order, $$a_t \approx 1 - \gamma\,\langle \nabla_P \ell_t,\; P_t\rangle_F$$. So Mainline 2 is the delta rule plus one extra descent direction—the state matrix $$P_t$$ itself—acting as a rank-one preconditioner along the direction the readout ignores.

---

## 7. Mainline 3: Eliminate $$c$$ — Calibrated Hyperballs and Why You Never Needed the Sphere

The third answer is to plug in the instantaneous best scale $$c = c^*(S)$$ at every step. Then $$\partial_c F = 0$$, and by the Euler identity the expected gradient is **purely tangential**. In fact, eliminating $$c$$ turns the CHI loss into the reduced objective $$G(S) \triangleq F(S, c^*(S))$$, which is genuinely scale-invariant ($$G(a S) = G(S)$$): it depends only on direction, finally matching the $$L(cW) = L(W)$$ assumption in the outer Hyperball paper.

Before writing the calibrated fix, though, it helps to see why the tempting naive version fails.

### 7.1 The trap: hard-projecting plain DeltaNet onto a sphere

If we only care about directions, why not follow Hyperball naively and keep $$\|S\|_F = R$$ fixed by projecting after every step?

$$
S_{t+1} = R\, N_F\!\big(S_t - \beta_t (S_t k_t - v_t) k_t^\top\big).
$$

The problem is that the plain MSE loss $$\frac{1}{2}\|S k - v\|_2^2$$ is not scale-invariant, so the radial force in the Euler identity is real. Hard-projecting every step cuts off that radial force, which reappears in the stationary equation as a normal Lagrange multiplier $$-\lambda S$$ and pulls the state to a **hidden spherical ridge solution**:

$$
S_R = C(\Sigma + \lambda I)^{-1}, \qquad \|S_R\|_F = R.
$$

<details markdown="1">
<summary>Proof of the sphere-constrained ridge stationary point</summary>

For the population loss $$L(S) = \frac{1}{2}\mathbb{E}\|S k - v\|_2^2$$, the gradient is $$\nabla L(S) = S\Sigma - C$$. Under the constraint $$\|S\|_F = R > 0$$, the Lagrangian stationarity condition is:

$$
S\Sigma - C + \lambda S = 0, \qquad \|S\|_F = R.
$$

Whenever $$\Sigma + \lambda I$$ is invertible, this solves to $$S_R = C(\Sigma + \lambda I)^{-1}$$, with $$\lambda$$ fixed by $$\|S_R\|_F = R$$. (Singular cases require separate handling; also, this is the population stationary condition under the sphere constraint—constant-step stochastic iteration does not necessarily converge to this exact point, and $$\lambda > 0$$ corresponds to standard positive ridge regularization.) $$\blacksquare$$

</details>

A 2D numerical example makes it obvious why this bends directions. Suppose we store two orthogonal facts, a frequent one (80% of the time) and a rare one (20%), with ideal memory $$S = \operatorname{diag}(1, 1)$$ (norm $$\sqrt{2} \approx 1.414$$):

$$
L(s_1, s_2) = \frac{1}{2}\big[0.8\,(s_1 - 1)^2 + 0.2\,(s_2 - 1)^2\big].
$$

Lock the radius at $$R = 0.5$$. The unbiased point on the circle is $$(0.3536,\; 0.3536)$$ with $$L \approx 0.2089$$. Solving the Lagrange equations $$0.8(s_1 - 1) + \lambda s_1 = 0$$, $$0.2(s_2 - 1) + \lambda s_2 = 0$$, and $$s_1^2 + s_2^2 = 0.25$$ gives $$\lambda \approx 0.9143$$ and:

$$
s_1 \approx 0.4667, \qquad s_2 \approx 0.1795, \qquad L \approx 0.1811 \quad (\text{lower!}).
$$

The MSE contours are ellipses stretched by $$2\times$$ along $$s_2$$, so they touch the circle first near the $$s_1$$ axis. Under a tight norm budget, MSE would rather sacrifice the rare fact to reduce average squared error—leaving the frequent weight $$2.6\times$$ larger than the rare one. As $$R \to 0^+$$ the ratio approaches $$4 : 1$$, aligning with $$C$$: **plain linear attention with no erasure at all.** A mixed query that should read out at $$45^\circ$$ comes out near $$21^\circ$$ instead, and outer RMSNorm cannot repair a ratio that is already broken inside the memory.

> **Side remark (why calibrated spherical updates keep the residual magnitude instead of normalizing the update direction like outer Hyperball).** Outer Hyperball writes $$W_{t+1} = R\,N_F\big(W_t - \eta_t R\,N_F(u_t)\big)$$, normalizing both the weights and the update direction. Inside online linear attention, however, the gradient $$G_t = e_t k_t^\top$$ has norm $$\|e_t\|_2$$. Dividing out $$\|e_t\|_2$$ erases the signal of whether a fact has already been memorized, destroys the exact fixed point at $$e_t = 0$$, and makes stored memories jitter under small perturbations. Keeping the residual magnitude (clipping only an upper bound if needed) preserves the fixed point.

### 7.2 The fix: give the write term one power of $$S$$

The root cause takes one line to state:

$$
S_{t+1} = \underbrace{S_t (I - \beta_t k_t k_t^\top)}_{\text{degree 1 in } S_t} + \underbrace{\beta_t v_t k_t^\top}_{\text{degree 0 in } S_t}.
$$

The write term has no $$S_t$$ in it. Rescale $$S_t$$ and you immediately distort the ratio between old memory and new writes.

The fix is to multiply the write target by a positively 1-homogeneous scalar $$g(S_t)$$ (meaning $$g(a S) = a\, g(S)$$ for all $$a > 0$$):

$$
\boxed{\ S_{t+1} = S_t - \beta_t\big(S_t k_t - g(S_t)\, v_t\big) k_t^\top\ }
$$

The natural choice is $$g(S) = 1 / c^*(S) = \langle S, S\Sigma\rangle_F / \langle S, C\rangle_F$$: degree 2 over degree 1, hence degree 1 overall. This is literally Mainline 2 with $$c$$ replaced by its best response $$c^*(S)$$. (If you want an explicit capacity budget tied to the radius, you can also take $$g(S) = \tau_t \|S\|_F / \|v_t\|_2$$ with $$\tau_t = \|v_t\|_2 / (R c_t)$$.) Whenever samples are drawn from the distribution defining $$F$$ and the step size is independent of the current sample, the Euler identity makes the expected increment orthogonal to $$S_t$$—giving **zero ridge bias** under arbitrary anisotropic keys while letting $$\|v_t\|_2$$ continue to weight per-token salience.

In practice $$c^*(S)$$ has to be estimated online, giving a clean spectrum of estimators:

- **At one end, a single-token orthogonal projection:** $$\hat{g}_t = \|p_t\|_2^2 / (p_t^\top v_t)$$ (with $$p_t = S_t k_t$$). When $$p_t^\top v_t > 0$$, this scales $$v_t$$ along its positive ray until $$p_t$$ is the foot of the perpendicular, making even the single-step increment $$\Delta S_t$$ strictly orthogonal to $$S_t$$. It is clean, but noisy: the target norm becomes $$\|p_t\|_2 / \cos\theta$$, which washes out $$\|v_t\|_2$$ salience and writes weakly on keys the memory has never seen ($$p_t \approx 0$$).
- **A practical middle ground uses smoothed statistics:** for instance, $$\hat{g}_t = \mathrm{EMA}(\|p\|_2^2) / \mathrm{EMA}(p^\top v)$$. Tracking population $$1/c^*(S)$$ via exponential moving averages filters out single-step noise while keeping $$\|v_t\|_2$$ salience intact. If you occasionally rescale the state $$S \leftarrow S / z$$, just divide the numerator EMA by $$z^2$$ and the denominator EMA by $$z$$ so the whole operator remains strictly positively 1-homogeneous.

### 7.3 Why you never needed to lock the sphere

> **Lemma (Homogeneous commutation).** Assume initial states are positive multiples of each other and all intermediate states stay nonzero. If every update map $$\mathcal{T}_t$$ is positively 1-homogeneous ($$\mathcal{T}_t(a S) = a\,\mathcal{T}_t(S)$$ for all $$a > 0$$), then "never normalize" and "project back to the sphere of radius $$R$$ after every step" produce states that differ only by a positive scalar at every step $$t$$.

<details markdown="1">
<summary>Proof of homogeneous commutation by induction</summary>

Let $$X_{t+1} = \mathcal{T}_t(X_t)$$ be the unnormalized trajectory and $$Y_{t+1} = R\,\mathcal{T}_t(Y_t) / \|\mathcal{T}_t(Y_t)\|_F$$ be the sphere-projected trajectory. Assuming $$Y_t = \lambda_t X_t$$ with $$\lambda_t > 0$$, positive 1-homogeneity gives:

$$
Y_{t+1} = \frac{R\,\mathcal{T}_t(\lambda_t X_t)}{\|\mathcal{T}_t(\lambda_t X_t)\|_F} = \frac{R\,\lambda_t \mathcal{T}_t(X_t)}{\lambda_t \|\mathcal{T}_t(X_t)\|_F} = \frac{R}{\|X_{t+1}\|_F}\, X_{t+1}.
$$

The positive scalar $$\lambda_t$$ cancels out between numerator and denominator, completing the induction. $$\blacksquare$$

</details>

Through an ideal RMSNorm, both trajectories produce **the exact same outputs**. Hard-locking a radius on every step does not simplify the calibrated update; it just adds an extra full-matrix division per token. In practice you only rescale *occasionally*, for two down-to-earth engineering reasons:

1. **Overflow protection:** The discrete second-order term $$+\|\Delta S_t\|_F^2$$ makes the norm drift slowly upward, so dividing by a scalar every once in a while keeps values safely in bf16/fp16 without changing any direction.
2. **Chunkwise WY parallelism:** Normalizing at chunk boundaries and freezing a per-chunk constant $$g_m$$ (or $$\bar{c}_m$$) inside the chunk turns the intra-chunk recurrence back into the plain linear delta rule, compatible with fast chunkwise WY matrix multiplication.

---

## 8. One Formula, and Then One Knob

All three mainlines fit inside a single unified update (keys normalized $$\|k_t\|_2 = 1$$, old prediction $$p_t = S_t k_t$$):

$$
\boxed{\ S_{t+1} = a_t\Big[\alpha_t S_t - \beta_t\big(\alpha_t p_t - \tilde{v}_t\big) k_t^\top\Big], \qquad o_t = S_{t+1} q_t \xrightarrow{\;\mathrm{RMSNorm}\;} y_t\ }
$$

The roles split cleanly:

- $$\alpha_t$$ inside the bracket multiplies only the old memory: an **exogenous forget gate**;
- $$a_t$$ outside multiplies retained memory and new write together: an **endogenous residual calibration gate**;
- whether $$\tilde{v}_t$$ carries the positively 1-homogeneous factor $$g(S_t)$$ determines whether we sit in the **homogeneous / calibrated spherical world**.

| Mainline | $$(\alpha_t,\; a_t,\; \tilde{v}_t)$$ | What it does with $$c$$ | Strengths / costs |
|---|---|---|---|
| 1. Free-norm GDN | $$\alpha_t \in (0, 1],\; a_t = 1,\; \tilde{v}_t = v_t$$ | Fix $$c \equiv 1$$; gate acts as an exogenous schedule for auxiliary scale | Natively WY-parallel; recency ridge $$\frac{1-\alpha}{\alpha\beta}$$ |
| 2. Joint online SGD | $$\alpha_t = 1,\; a_t = e^{-\tilde{\gamma}_t z_t^\top e_t},\; \tilde{v}_t = v_t$$ | One gradient step per token on $$\log c$$ | One-step global recalibration; state-dependent gate needs a recurrent kernel (or per-chunk freezing) |
| 3. Homogeneous calibrated | $$\alpha_t = 1,\; a_t = 1,\; \tilde{v}_t = g(S_t) v_t$$ | Eliminate $$c$$ via $$c^*(S)$$ | No per-step projection, no ridge bias; chunk-boundary rescaling recovers WY |

We can push this unification one step further—and this is where the perspective earns its keep.

Make a simple change of coordinates: let $$S_t = \kappa_t Q_t$$ with $$\kappa_0 = 1$$ and $$\kappa_{t+1} = a_t \alpha_t \kappa_t$$. Substituting into the unified update and canceling $$\kappa_{t+1}$$ gives:

$$
\boxed{\ Q_{t+1} = Q_t (I - \beta_t k_t k_t^\top) + \beta_t\, \frac{\tilde{v}_t}{\alpha_t \kappa_t}\, k_t^\top\ }
$$

Because $$Q_t \propto S_t$$ at every step, **the readout through an ideal RMSNorm is completely unchanged.** That means under an ideal output norm, every member of the family is *plain, ungated DeltaNet with a per-token target scale* $$v_t / c_t$$. The three mainlines differ only in how they pick that single scalar sequence $$c_t$$ (corresponding to $$\alpha_t \kappa_t = \kappa_{t+1}$$ in Mainline 1, or the log-space $$c_t$$ in Mainline 2):

| How $$c_t$$ is chosen | Resulting algorithm |
|---|---|
| $$c_t \equiv 1$$ | DeltaNet |
| $$c_t = \prod_{s \le t} \alpha_s$$ (exogenous, input-dependent) | Gated DeltaNet |
| One online SGD step on $$\log c_t$$ toward $$c^*$$ | Mainline 2 (Joint SGD) |
| Exact best response $$c_t = c^*(S_t)$$ | Mainline 3 (population $$g$$) |
| Single-token or EMA estimate of $$c^*(S_t)$$ | Mainline 3 (practical $$g$$) |

(In floating point you would never actually run the $$Q$$-coordinates, because $$\prod \alpha_s^{-1}$$ explodes exponentially—which is why implementations keep the gated form. As a *way of thinking*, however, the equivalence is exact.)

It also explains hardware parallelism in one sentence: **chunkwise WY parallelism works when and only when $$c_t$$ is known before the chunk starts.** An exogenous $$c_t$$ (DeltaNet / GDN) is fine. A state-dependent $$c_t$$ (Mainlines 2 and 3) is not, unless you freeze it per chunk.

So here is the answer to our opening question: **the optimizer living inside linear attention is online SGD on the CHI loss (Goal \*), and the design space largely comes down to how we handle the global scale symmetry created by RMSNorm.** Fix it, learn it, or eliminate it.

---

## 9. Minimal Reference Code

```python
import torch

# Shapes: S [B, H, dv, dk]; q, k [B, H, dk] (k unit-norm, q un-normalized); v [B, H, dv]
def linear_attention_step(S, q, k, v, beta, mode="gdn", alpha=1.0,
                          gamma0=0.1, delta_max=0.5, g=None, eps=1e-6):
    p = torch.einsum("bhvk,bhk->bhv", S, k)                    # old prediction S_t k_t

    if mode == "gdn":            # Mainline 1: fix c=1, exogenous gate
        e = alpha * p - v
        S_next = alpha * S - (beta * e).unsqueeze(-1) * k.unsqueeze(-2)

    elif mode == "joint_sgd":    # Mainline 2: learn log c, two-sided gate a_t
        e = p - v
        pe = (p * e).sum(-1, keepdim=True)
        curv = (p * p).sum(-1, keepdim=True) + (v * v).sum(-1, keepdim=True) + eps
        a = torch.exp((-gamma0 * pe / curv).clamp(-delta_max, delta_max))
        S_next = a.unsqueeze(-1) * (S - (beta * e).unsqueeze(-1) * k.unsqueeze(-2))

    else:                        # Mainline 3: eliminate c, target g(S_t) v_t
        if g is None:            # single-token estimate 1/c* (pass a smoothed EMA g in practice)
            pv = (p * v).sum(-1, keepdim=True)
            pp = (p * p).sum(-1, keepdim=True)
            g = torch.where(pv > eps, pp / pv.clamp_min(eps), torch.ones_like(pv))
        e = p - g * v
        S_next = S - (beta * e).unsqueeze(-1) * k.unsqueeze(-2)
        # S=0 is a fixed point of a homogeneous map: seed with v_0 k_0^T.
        # Rescale by a scalar occasionally to avoid overflow; directions are unaffected.

    o = torch.einsum("bhvk,bhk->bhv", S_next, q)               # -> RMSNorm(o) * SiLU(gate)
    return S_next, o
```

---

## 10. Open Questions I Find Fun

- **What should $$c$$ track when the world drifts?** Everything above measured "bias" against the stationary $$P^* = C\Sigma^{-1}$$. Real token streams drift, and forgetting is a feature there. The natural next step is to ask which choice of $$c_t$$ is best for *tracking* rather than for stationary convergence.
- **Channel-wise gates.** The lemma in Section 3 says that a single global scale is the only freedom that never bends directions. Fine-grained, per-channel key decay (as in recent delta-attention variants) is per-direction scaling: in this picture it deliberately trades direction fidelity for selective forgetting. Can we make that trade-off quantitative?
- **Smoothed estimators of $$c^*$$.** Between "one gradient step" (Mainline 2) and "exact best response" (Mainline 3) lies a whole family of online estimators. Which one is cheap, stable, and chunk-friendly?

Four assumptions every block already makes, one symmetry, and one small question: **what do we do with $$c$$?**

<details>
<summary>Cite this post</summary>
<p>If you find this post useful in your work, you can cite it as:</p>
<blockquote>
<p>Chi, Yuhan. “What Optimizer Lives Inside Linear Attention? A Scale-Invariance View of DeltaNet.” November 22, 2026. <a href="https://yuhanchi.github.io/writing/optimizers-inside-linear-attention/">https://yuhanchi.github.io/writing/optimizers-inside-linear-attention/</a></p>
</blockquote>
<p>This post draws on ideas from Hyperball optimization (<a href="https://arxiv.org/abs/2606.16899">Wen et al.</a>) and parameter symmetry / effective learning rates (<a href="https://spaces.ac.cn/archives/11985">Jianlin Su</a>); please cite their original works when referring to those results. To cite this post, use the following BibTeX entry:</p>
{% raw %}<pre><code class="language-bibtex">@misc{chi2026optimizers,
  author = {Chi, Yuhan},
  title = {{What Optimizer Lives Inside Linear Attention? A Scale-Invariance View of DeltaNet}},
  year = {2026},
  month = nov,
  howpublished = {Blog post},
  url = {https://yuhanchi.github.io/writing/optimizers-inside-linear-attention/}
}</code></pre>{% endraw %}
</details>
