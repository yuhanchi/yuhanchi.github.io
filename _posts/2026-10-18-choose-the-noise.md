---
title: "Choose the Noise, Keep the Flow"
date: "2026-10-18"
description: "A gentle survey of noise-space policy steering: the VAE connection, five routes for learning the noise, and the limits of a frozen decoder."
category: "tech"
tags: ["AI", "reinforcement-learning", "flow-matching", "robotics"]
math: true
---
A recent survey by Nahim, Khan, and Ju, [A Survey of Flow Matching in Reinforcement Learning](https://arxiv.org/abs/2610.16275) (TMLR, 2026), organizes the field along two axes: *what* the flow models (actions, critics, dynamics) and *where* the RL signal enters (energy-weighted regression, policy gradients through the ODE, GRPO-style objectives). It is a useful map. I wanted to look more closely at one family that is not brought together as a group in that survey.

At the center are methods that leave the velocity field untouched. They keep a pretrained flow or diffusion policy frozen and train a small network that decides **which noise to feed it**. Given a state $$s$$, a proposer picks a point $$w$$ in noise space, and the frozen generator turns $$w$$ into an action. Imagine a robot policy that sometimes reaches around an obstacle on the left and sometimes on the right. Steering the noise changes how often it chooses each route, while reusing the generator that produces the movements.

The survey already covers an important relative. Its offline section discusses ReFORM (Zhang et al., 2026), which manipulates noise inside a bounded support so that the composed policy stays within the learned behavior model’s support. Its discussion of "freezing the flow and learning a correction" covers policies that *edit the action after it is generated*. DSRL (Wagenmaker et al., 2025) and related methods that *choose the input* instead of editing the output are absent from the version I read (v1). This post follows that thread, including a few neighbors that also train the decoder or learn from corrections rather than rewards.

This post collects that family and makes two points.

1. A **VAE-style inference view** helps connect the methods. The noise proposer plays the role of an encoder and the frozen generator that of a decoder. For a KL-regularized objective, this connection becomes an exact variational identity, with value taking the place of reconstruction quality. Other methods share the architecture without optimizing that same objective.
2. The most useful question to ask of each method is **where RL takes effect**. Action-space data can train a critic directly, but the proposer lives in noise space. Each method is defined largely by how it carries value from one space to the other.

<figure>
  <a href="{{ '/assets/img/posts/choose-the-noise/vae-correspondence.svg' | relative_url }}" title="Open scalable figure"><img src="{{ '/assets/img/posts/choose-the-noise/vae-correspondence.png' | relative_url }}" alt="Two aligned pipelines. Top, a VAE: x, encoder, latent z, decoder, reconstruction. Bottom, noise-space RL: state s, proposer, noise w, frozen flow, value Q over beta, with s also feeding the flow. A shared N(0, I) prior sits between the two latents, linked to each by a KL term." width="1420" height="510" loading="lazy"></a>
  <figcaption>A shared encoder–decoder pattern. Top: a VAE. Bottom: the Gaussian-prior, KL-regularized version of noise-space RL. The lower decoder produces an action, not a reconstruction of the state; value supplies the learning signal. The correspondence is between objectives, not identical generative models.</figcaption>
</figure>

## 1. The decoder is already trained

Run a flow policy with a deterministic sampler: the probability-flow ODE, DDIM with zero variance, or a one-step distilled model. It then becomes a fixed map

$$
a = \psi_\theta(s, w), \qquad w \sim \mathcal{N}(0, I),
$$

and all of its randomness enters through $$w$$. The policy is the pushforward of the Gaussian through this map, $$\pi_{\text{base}}(\cdot \mid s) = \psi_\theta(s, \cdot)_{\#}\, \mathcal{N}(0, I)$$. A noise-space method keeps $$\theta$$ fixed and replaces only the distribution of the input:

$$
\pi_\phi(\cdot \mid s) = \psi_\theta(s, \cdot)_{\#}\, q_\phi(\cdot \mid s).
$$

The proposer $$q_\phi$$ can be a Gaussian with a state-dependent mean and variance, a deterministic map, a residual added to fresh noise, or even a single constant vector. It is usually smaller than the generator, though not necessarily tiny: Noise PPO uses a roughly 187M-parameter proposer. The decoder $$\psi_\theta$$ can be very large (DSRL steers $$\pi_0$$, a 3.3B-parameter generalist policy) and is never updated.

DSRL states the construction cleanly. Define a new MDP whose actions are noises:

$$
P^{W}(\cdot \mid s, w) = P(\cdot \mid s, \psi_\theta(s, w)), \qquad r^{W}(s, w) = r(s, \psi_\theta(s, w)).
$$

The diffusion policy becomes part of the environment. Any RL algorithm can now run over $$w$$, using only forward passes of the generator, without its weights or its intermediate denoising steps.

The idea is older than diffusion policies. PLAS (Zhou et al., 2020) learned a policy in the latent space of a conditional VAE for offline RL. PARROT (Singh et al., 2021) trained a normalizing-flow behavioral prior and ran RL on its noise. It compared directly against a conditional-VAE latent space and argued that invertibility lets the agent keep full control over the final action. DSRL singles out PARROT as the precedent in control and makes the same point from the other side: diffusion models are not invertible by construction, so it is "much less clear" that steering their noise is expressive enough. In practice, it turned out to be.

## 2. Reading it as a VAE

Recall the evidence lower bound of an ordinary VAE (a conditional VAE additionally conditions its distributions on context):

$$
\log p(x) \;\ge\; \mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] \;-\; \mathrm{KL}\big(q_\phi(z \mid x) \,\Vert\, p(z)\big).
$$

To make the connection precise, consider the following KL-regularized policy-improvement objective:

$$
J(\phi) = \mathbb{E}_{s}\Big[\, \mathbb{E}_{w \sim q_\phi(\cdot \mid s)}\, Q\big(s, \psi_\theta(s, w)\big) \;-\; \beta\, \mathrm{KL}\big(q_\phi(\cdot \mid s) \,\Vert\, \mathcal{N}(0, I)\big) \Big].
$$

Here $$Q$$ is a fixed action-value estimate during the proposer update, $$\beta > 0$$ controls regularization, and states are sampled from a fixed training distribution. This is a local improvement objective, not by itself a complete solution to sequential RL. The two expressions line up as follows:

| VAE | Noise-space RL |
|---|---|
| observed input $$x$$ | desired optimality, with state $$s$$ as context |
| encoder $$q_\phi(z \mid x)$$ | noise proposer $$q_\phi(w \mid s)$$ |
| prior $$p(z) = \mathcal{N}(0, I)$$ | the Gaussian the flow was trained from |
| decoder $$p_\theta(x \mid z)$$ | frozen flow $$\psi_\theta(s, w)$$ |
| reconstruction $$\log p_\theta(x \mid z)$$ | value $$Q(s, \psi_\theta(s, w)) / \beta$$ |
| KL to the prior | KL to $$\mathcal{N}(0, I)$$: penalize departure from the training prior |
| decoder trained jointly | decoder pretrained by imitation, usually frozen |

The exact connection is a variational identity. Define $$Z(s) = \mathbb{E}_{w \sim \mathcal{N}(0,I)}[\exp(Q(s, \psi_\theta(s,w))/\beta)]$$ and assume it is finite. Then, for each state,

$$
\log Z(s) = \mathbb{E}_{q_\phi}[Q(s, \psi_\theta(s,w))/\beta] - \mathrm{KL}(q_\phi \Vert \mathcal{N}(0,I)) + \mathrm{KL}(q_\phi \Vert q^\star).
$$

The last term is nonnegative, so the first two terms form a lower bound. In the control-as-inference view (Levine, 2018), this can be read as an ELBO for a desired “optimality” observation. When $$Q$$ is bounded above, a state-dependent scaling makes the exponential a valid binary likelihood; the variational identity itself needs only a finite $$Z(s)$$. The unrestricted best proposer is

$$
q^\star(w \mid s) = \frac{\mathcal{N}(w; 0, I)\, \exp\big(Q(s, \psi_\theta(s, w))/\beta\big)}{Z(s)}.
$$

The difference from an ordinary VAE is what counts as the observation. The encoder is not asked to explain an action that was seen. It is asked to infer which noise *would have produced a good action*. The evidence is desired success, represented here by the value estimate. This is why “VAE-style” is more accurate than calling every method a conditional VAE. A deterministic proposer, for example, has infinite KL to a continuous Gaussian; box constraints, spherical latents, and supervised inversion require their own objectives.

Two facts make this reading useful rather than decorative. Eyring et al. (2025) state both for distilled text-to-image generators, and both carry over to policies without change.

**Tilting the noise tilts the policy.** Pushing the noise posterior through the frozen decoder gives exactly the reward-tilted base policy:

$$
\psi_\theta(s, \cdot)_{\#}\, q^\star(\cdot \mid s) \;=\; \pi^\star(\cdot \mid s) \;\propto\; \pi_{\text{base}}(\cdot \mid s)\, \exp\big(Q(s, \cdot)/\beta\big).
$$

The reason is that the tilt depends on $$w$$ only through $$a = \psi_\theta(s, w)$$.

<details markdown="1">
<summary>Why the tilt commutes with decoding</summary>

For a bounded test function $$f$$, suppressing the fixed state $$s$$,

$$
\begin{aligned}
\mathbb{E}_{q^\star}[f(\psi_\theta(w))]
&= Z^{-1}\mathbb{E}_{w\sim\mathcal{N}(0,I)}
   [f(\psi_\theta(w))e^{Q(\psi_\theta(w))/\beta}] \\
&= Z^{-1}\mathbb{E}_{a\sim\pi_{\text{base}}}
   [f(a)e^{Q(a)/\beta}].
\end{aligned}
$$

</details>

No invertibility is needed. This is the same target that KL-regularized fine-tuning of the whole flow aims for, but a small proposer can only approximate it as well as its distribution family and optimization allow. A single Gaussian, for example, need not represent a multimodal noise posterior.

<figure>
  <a href="{{ '/assets/img/posts/choose-the-noise/tilt-noise-tilts-policy.svg' | relative_url }}" title="Open scalable figure"><img src="{{ '/assets/img/posts/choose-the-noise/tilt-noise-tilts-policy.png' | relative_url }}" alt="Two rows of four panels for beta equal to infinity, 4, 1 and 0.1. Top: noise densities that concentrate in one wedge of the Gaussian as beta shrinks. Bottom: action samples that go from four equal clusters to a single cluster near the reward peak." width="1346" height="779" loading="lazy"></a>
  <figcaption>A 2D illustration of noise tilting and its decoded actions. Top: the tilted noise \(q^\star(w) \propto \mathcal{N}(w; 0, I)\, e^{Q(\psi(w))/\beta}\). Bottom: those noises pushed through the frozen flow (dots), over the target density \(\pi_{\text{base}}\, e^{Q/\beta}\) (shading). Dashed circles are contours of \(Q\), and the star is its peak. Large \(\beta\) gives back the base policy; small \(\beta\) favors the best mode in this example. This illustrates the ideal tilted distribution, not the result of training a restricted proposer.</figcaption>
</figure>

**The noise-space KL is a conservative proxy.** By the data processing inequality,

$$
\mathrm{KL}\big(\pi_\phi(\cdot \mid s) \,\Vert\, \pi_{\text{base}}(\cdot \mid s)\big) \;\le\; \mathrm{KL}\big(q_\phi(\cdot \mid s) \,\Vert\, \mathcal{N}(0, I)\big),
$$

with equality when $$\psi_\theta(s, \cdot)$$ is invertible. The left side can be expensive to evaluate: a continuous normalizing flow can compute densities through divergence integration, while a generic distilled decoder need not offer a tractable density. For a Gaussian proposer $$\mathcal{N}(\mu, \mathrm{diag}\,\sigma^2)$$, the right side is the familiar VAE term $$\tfrac12 \sum_j (\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2)$$. Noise PPO (Miao et al., 2025) uses exactly this KL penalty, together with an entropy bonus. Noise Hypernetworks and LP-DS (Simsir & Oguz, 2026) write the steered noise as a residual $$w = \epsilon + \Delta$$ and approximate the KL by $$\tfrac12\, \mathbb{E}\Vert\Delta\Vert^2$$, a surrogate justified under regularity and small-Jacobian assumptions. Bounding the residual’s magnitude alone does not establish those assumptions.

## 3. Where RL takes effect: carrying value into noise space

The decision variable for RL is now the noise, often parameterized by a Gaussian actor. The frozen decoder supplies a rich action distribution. This simplifies the policy interface, but does not guarantee an easy optimization landscape: nearby noises may have very different values. Some routes need only forward access to the decoder; others need its derivatives or an inverse.

Much of the available supervision, however, lives in the other space. An action-space critic $$Q(s, a)$$ is ordinary temporal-difference learning on $$(s, a, r, s')$$ transitions. Offline datasets, demonstrations, and human corrections all come labeled with actions, not noises. The proposer needs something different: a value of $$w$$, or a gradient with respect to $$w$$, of the composition

$$
w \;\longmapsto\; Q\big(s, \psi_\theta(s, w)\big).
$$

So the central design question of the family is how to **carry an action-space value back into noise space**. The papers suggest five useful routes, each with a different price. These are not disjoint categories: one-step backpropagation is a special case of solver backpropagation, and a method can combine inversion with a noise critic.

### Route 1: learn a critic in noise space

One option is to learn a noise-space critic directly from rollouts, as in DSRL-SAC. DSRL-NA ("noise-aliased") instead keeps two critics. It learns $$Q^{A}(s, a)$$ by temporal-difference learning in action space, then distills it into a noise-space critic by regressing $$Q^{W}(s, w)$$ onto $$Q^{A}(s, \psi_\theta(s, w))$$ with $$w \sim \mathcal{N}(0, I)$$. The decoder is only ever run forward. Because many noises decode to similar actions, a noise that was never played can still receive an accurate value. On Robomimic Square, plain SAC in noise space (DSRL-SAC) needs about twice as many samples. On real robots, DSRL reports cases where success rose from 20% to 90% within fewer than 50 online episodes. LP-DS keeps the same two-critic design. The price is an extra approximation: LPS argues that a distilled critic can match values well and still point its gradient in the wrong direction. A related forward-only idea appears in GoldenStart, which picks the highest-value noise for each state by decoding candidates and scoring them with the critic, then fits a conditional VAE to these noises as a learned prior. GoldenStart is not a pure frozen-decoder method, though: its flow teacher keeps training, and the deployed policy is a distilled one-step student.

### Route 2: invert the flow and move the data

Sometimes the signal is an action rather than a reward: a human correction, or a coarse suggestion from a vision-language model. A flow can be integrated backwards to recover the noise that would have produced that action, $$w^\star \approx \psi_\theta^{-1}(s, a^\star)$$. When the inversion is accurate, the corrected actions become noise-space training targets. A generic VAE decoder does not provide an inverse, so it usually relies on a learned encoder for inference. An exact ODE flow is invertible under suitable regularity assumptions; a coarse numerical sampler or a distilled decoder need not be. When inversion works, it supplies a noise that reconstructs the correction—not a sample from the reward-tilted posterior above. FlowDAgger (Murray et al., 2026) regresses the proposer onto inverted corrections. UniSteer (Lu et al., 2026) stores inverted transitions $$(s, \hat w, r, s')$$ in both a demonstration buffer and an RL buffer, and trains a noise-space critic on them by ordinary TD learning. Flow Reversal Steering (Tang et al., 2026) maps coarse reference actions to nearby modes of a generalist policy, then distills them or uses them to bootstrap DSRL. The price is inversion error. With the short ODE schedules of robot action heads, a naive reverse Euler step is not accurate enough, which is why FlowDAgger and UniSteer both use per-step fixed-point refinement for their flow action heads. Convergence requires conditions such as a sufficiently small step size relative to the velocity field’s Lipschitz constant; it is not automatic.

### Route 3: backpropagate through the sampling iterations

The chain rule gives the noise gradient exactly:

$$
\nabla_w\, Q\big(s, \psi_\theta(s, w)\big) = \Big(\frac{\partial \psi_\theta}{\partial w}\Big)^{\!\top} \nabla_a Q(s, a),
\qquad
\frac{\partial \psi_\theta}{\partial w} = \prod_{k=K-1}^{0} \Big(I + \Delta t\, \frac{\partial v_\theta}{\partial x}(x_k, t_k, s)\Big)
$$

for a $$K$$-step Euler sampler, with later-step Jacobians multiplying on the left. “Exact” here means the derivative of the implemented sampler and learned critic, not of the true environment return. With a reparameterized proposer $$w = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon$$, this is the VAE reparameterization trick with a frozen decoder. No second critic is needed, but the gradient has to pass through every step of the solver. DSRL calls this "computationally intensive and numerically unstable", and LPS calls it "expensive and unstable". D-Flow (Ben-Hamu et al., 2024) takes this route for per-sample noise optimization in image, audio, and molecule generation.

### Route 4: make the decoder one step

If $$\psi_\theta$$ is a single network evaluation, the product above has a single factor, and the critic gradient reaches the noise in one backward pass. LPS (Im et al., 2026) uses a one-step MeanFlow base policy for exactly this reason, and backpropagates $$\nabla_a Q$$ straight into the latent actor without a noise-space critic. It is a useful architectural neighbor, rather than a strictly frozen-decoder method: Algorithm 1 also updates the base policy by imitation during training. ReNO does the same with reward models on one-step image generators, and Noise Hypernetworks on distilled one- or few-step generators (SD-Turbo, SANA-Sprint, FLUX-Schnell). With a few steps the chain in Route 3 is short rather than trivial, which is still cheap. The price moves to the base policy: it must be one-step, which usually means it was trained or distilled for that purpose.

### Route 5: skip the transfer

A fifth option avoids the transfer entirely. For the return term, the likelihood-ratio policy gradient $$\nabla_\phi J_{\mathrm{return}} = \mathbb{E}\big[\nabla_\phi \log q_\phi(w \mid s)\, A(s, w)\big]$$ needs only $$\log q_\phi$$, which is exact for a Gaussian proposer: no ODE divergence, no SDE conversion, no surrogate likelihood. Much of the survey's online-RL section is about recovering a usable likelihood for flow policies; in noise space the problem does not arise. Noise PPO treats text-to-image generation as a one-step MDP whose action is the initial noise, and runs PPO with sampled rewards. KL and entropy regularizers contribute additional gradient terms. The price is the usual variance of score-function estimates.

### The five routes side by side

<figure>
  <a href="{{ '/assets/img/posts/choose-the-noise/five-routes.svg' | relative_url }}" title="Open scalable figure"><img src="{{ '/assets/img/posts/choose-the-noise/five-routes.png' | relative_url }}" alt="Five rows, each with noise space on the left, the frozen decoder in the middle, and action space on the right. 1: a regression arrow from the action critic to a noise critic. 2: a corrected action inverted back through the decoder. 3: a gradient passing through five solver steps. 4: a gradient passing through one step. 5: returns going straight to a likelihood-ratio gradient, bypassing the decoder." width="1420" height="1317" loading="lazy"></a>
  <figcaption>Five routes for training the noise proposer, including action supervision in Route 2 and direct return feedback in Route 5. The first row illustrates the two-critic variant of Route 1. Grey dashed arrows show forward decoding dependencies; orange arrows show what crosses into noise space.</figcaption>
</figure>

| Route | What crosses into noise space | Price |
|---|---|---|
| 1. learn a noise critic | TD targets or distilled values $$Q^{W}(s, w)$$ | critic error; distillation adds another approximation |
| 2. invert the flow | data: actions become noises | inversion error, worse with coarse steps |
| 3. backpropagate through the solver | exact gradients | memory and instability over many steps |
| 4. one-step decoder | exact gradients in one pass | needs a one-step base policy |
| 5. likelihood ratio | nothing; returns are used directly | variance |

## 4. A map of the family

The table groups the methods by their main learning route. It includes related designs with changing decoders, labeled explicitly. “Prior control” distinguishes actual constraints from initialization or supervision alone.

| Route | Method | Setting | Proposer $$q_\phi(w \mid s)$$ | Prior control |
|---|---|---|---|---|
| 1. noise critic | PARROT (2021; preprint 2020) | robot manipulation | RL policy over the noise of an invertible normalizing flow | prior initialization; no lasting proximity guarantee |
| 1. noise critic | DSRL (2025) | online and offline control; $$\pi_0$$ | state-conditioned noise actor; SAC, or an action critic distilled into a noise critic | noise box $$[-b, b]^d$$ |
| 1. noise critic | LP-DS (2026) | online control; $$\pi_0$$ | residual $$\epsilon + \Delta_\phi(s, \epsilon)$$ with noise-aliased critics | Lagrangian trust region on $$\mathbb{E}\Vert\Delta\Vert^2$$ |
| 1. forward scoring | GoldenStart (2026) | distilled one-step flow policies; teacher not frozen | conditional VAE fit to the highest-value noise per state | CVAE fitting; entropy control is on the action student |
| 2. inversion | FlowDAgger, UniSteer (2026) | human-in-the-loop VLA adaptation | noise policy regressed onto inverted corrections; UniSteer adds a noise critic | inverted targets; inversion alone does not ensure typical noise |
| 2. inversion | Flow Reversal Steering (2026) | generalist policies | auxiliary noise policy trained on inverted coarse actions | coarse references guide reversal; no generic KL bound |
| 3 / 4. backpropagation | ReFORM (2026) | offline RL; BC model and distilled decoder also trained | reflected flow generating bounded input noise | projected steps keep noise inside the source ball |
| 4. one-step | Noise Hypernetworks (2025) | distilled text-to-image | residual on fresh noise (LoRA), reward gradient through the generator | $$\tfrac12\Vert\Delta\Vert^2$$ as a KL surrogate |
| 4. one-step | LPS (2026) | offline RL; real robots; base also trained by imitation | latent actor on a sphere, $$\nabla_a Q$$ through a one-step MeanFlow | sphere of radius $$\sqrt{d}$$ |
| 5. likelihood ratio | Noise PPO (2025) | text-to-image | prompt-conditioned Gaussian trained by PPO | KL to $$\mathcal{N}(0, I)$$, entropy bonus |
| 5. likelihood ratio | GoRL (2025) | online control | Gaussian encoder; decoder refit in alternating phases | KL to $$\mathcal{N}(0, I)$$ |
| search | Golden Ticket (2026) | generative robot policies | one constant noise vector found by search | — |

ReFORM spans Routes 3 and 4: its original objective differentiates through the BC flow, and its practical design introduces one-step BC distillation to shorten this path. Its Algorithm 1 also continues fitting the BC model. D-Flow is a per-sample example of Route 3.

Three neighbors are worth knowing. PDP (Zhang et al., 2026) argues that Gaussian noise is a poorly conditioned control interface. Instead of steering it, PDP trains a literal VAE over whole demonstrations, with a geometry-alignment term, and modulates the denoiser with the resulting low-dimensional code. Condition-dependent source flow matching (Kim et al., 2026) and LAFM (Machado et al., 2026) learn the prior itself together with the flow at training time, which in VAE terms is a learned prior. D-Flow (Ben-Hamu et al., 2024) and ReNO (Eyring et al., 2024) optimize the noise separately for each sample instead of training a proposer.

## 5. What the VAE view explains

**The decoder fixes the map, not a safety guarantee.** With a frozen decoder, every steered action lies in the image of $$\psi_\theta(s, \cdot)$$. For finite $$Q$$ and positive $$\beta$$, the ideal tilted policy has the same support as the base policy. But support is not the same as likely or reliable behavior: a Gaussian-driven invertible flow can have full support, including actions far from the demonstrations. ReFORM makes a stronger construction by bounding the source support; relating its learned model’s support to the dataset still depends on how well that model fits. A truly unreachable action cannot be recovered by steering, while a very unlikely action might be. GoRL (Zhang et al., 2025) changes the map itself by training the decoder again. Its decoder starts as an identity-like map. Training then alternates between two phases: optimizing a Gaussian encoder with a KL penalty toward $$\mathcal{N}(0, I)$$ while the decoder is frozen, and refitting the decoder by flow matching on recent rollouts while the encoder is frozen.

**$$\beta$$ is a dial between the base policy and high-value actions.** For a fixed, well-behaved $$Q$$, as $$\beta \to \infty$$ the ideal proposer approaches the prior and the system returns to the base policy. This resembles posterior collapse in a VAE, though here returning to the base can be intentional. As $$\beta \to 0$$, mass concentrates near maximizers, when they exist. Multiple equally good actions—or multiple noises for one action—can remain; a single point mass is not inevitable. LP-DS reports that prior-replacing steering can lose modes, and uses a residual trust region whose multiplier plays a role similar to $$\beta$$, for its surrogate penalty. Golden Ticket (Patil et al., 2026) imposes a different restriction: one noise vector shared across states. It is not the general low-temperature limit of a state-conditioned proposer.

**Holes in the latent space.** A VAE can decode poorly in regions its training rarely visits. A Gaussian-driven flow can have the same problem: most training draws come from typical regions of the Gaussian. The family uses several ways to discourage atypical queries, with different strengths. DSRL bounds each noise coordinate in $$[-b,b]$$ and reports that $$b$$ between 1 and 3 works well online. A box bounds coordinates; it does not by itself enforce the Gaussian’s typical radius. LPS trains both its base policy and its latent actor on the sphere of radius $$\sqrt{d}$$, where high-dimensional Gaussian mass concentrates, to prevent what it calls "norm explosion"; strictly, its prior is uniform on that sphere rather than $$\mathcal{N}(0, I)$$. LP-DS bounds the squared norm of its residual, its stand-in for the KL.

**Aliasing makes some noise changes redundant.** Many noises can decode to similar actions. This is related to latent ambiguity, but distinct from the statistical non-identifiability of a VAE’s latent coordinates. DSRL-NA turns this weakness into data sharing across noises. The same fact explains why the noise KL is only an upper bound: when the decoder is genuinely many-to-one, part of the KL budget can be spent on redistributing noise among identical outputs. For an invertible flow the KLs remain equal, even if different noises produce very similar actions.

**Amortization has a gap.** Optimizing noise separately for each sample (D-Flow, ReNO) and training a proposer that predicts it (Noise Hypernetworks, whose subtitle is "Amortizing Test-Time Compute") are the policy-side versions of per-datapoint variational inference and an amortized encoder. The cost of amortization has a name in the VAE literature, the amortization gap (Cremer et al., 2018). Noise Hypernetworks report recovering a substantial portion of the gains of explicit test-time optimization at a fraction of its cost, which illustrates the quality–compute tradeoff. That comparison is not, by itself, a measurement of the formal variational amortization gap.

## 6. Open questions

- **When is a policy steerable?** DSRL notes that the base policy must be "steerable" for the method to work and studies when this holds empirically, but does not give a formal definition. Nothing in the construction guarantees useful steering in advance. It helps to separate three questions: can the decoder produce a good action, does the allowed proposer put enough probability on its preimages, and can the learning signal find them? Invertibility addresses only the first.
- **How does the number of sampling steps matter?** Noise PPO finds the largest gains at few inference steps; the benefit shrinks, but does not disappear, as the number of steps grows. Several robot papers rely on one-step or few-step decoders for different reasons: cheap Jacobians for LPS, low latency for robot control. Few steps are not free, however: FlowDAgger finds that coarse steps make naive inversion inaccurate.
- **When should the decoder be trained?** Freezing the decoder preserves the learned map and can improve sample efficiency; training it, as in GoRL, can change which useful behaviors are easy to represent. Where a pretrained policy should sit between the two is not settled.

**A practical reading rule.** If the decoder is available only through forward calls, start by considering a noise critic or a likelihood-ratio update. If corrections are available, inversion may turn them into direct supervision. If gradients are available and the sampler is short, direct backpropagation avoids critic distillation. Then check the paper’s actual prior constraint and whether the decoder stays frozen; the shared diagram does not settle either question.

The survey's second axis asks where the RL signal enters a flow-matching method. This family adds one more answer to that question: **the RL signal can enter through the input-noise distribution.**

## References

- Nahim, N. Z., Khan, F. N., & Ju, P. (2026). [A Survey of Flow Matching in Reinforcement Learning](https://arxiv.org/abs/2610.16275). *TMLR*.
- Wagenmaker, A., Nakamoto, M., Zhang, Y., et al. (2025). [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799). (DSRL)
- Singh, A., Liu, H., Zhou, G., Yu, A., Rhinehart, N., & Levine, S. (2021). [PARROT: Data-Driven Behavioral Priors for Reinforcement Learning](https://arxiv.org/abs/2011.10024). *ICLR*.
- Zhou, W., Bajracharya, S., & Held, D. (2020). [PLAS: Latent Action Space for Offline Reinforcement Learning](https://arxiv.org/abs/2011.07213). *CoRL*.
- Miao, Y., Loh, W., Poupart, P., & Kothawade, S. (2025). [A Minimalist Method for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2506.12036). (Noise PPO)
- Eyring, L., Karthik, S., Dosovitskiy, A., Ruiz, N., & Akata, Z. (2025). [Noise Hypernetworks: Amortizing Test-Time Compute in Diffusion Models](https://arxiv.org/abs/2508.09968).
- Eyring, L., Karthik, S., Roth, K., Dosovitskiy, A., & Akata, Z. (2024). [ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization](https://arxiv.org/abs/2406.04312). *NeurIPS*.
- Ben-Hamu, H., Puny, O., et al. (2024). [D-Flow: Differentiating through Flows for Controlled Generation](https://arxiv.org/abs/2402.14017). *ICML*.
- Zhang, C., Wan, Z., Chen, F., Yu, X., Tsang, I., & An, B. (2025). [GoRL: An Algorithm-Agnostic Framework for Online Reinforcement Learning with Generative Policies](https://arxiv.org/abs/2512.02581v1). The current arXiv version is titled *Training Diffusion Policies via Prior-Mapping Co-Evolution*; this post follows v1.
- Zhang, H., Sun, Y., & Xiong, H. (2026). [GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies](https://arxiv.org/abs/2603.14245). *ICLR*.
- Simsir, H., & Oguz, O. S. (2026). [Lagrangian Perturbation Diffusion Steering: Latent Reinforcement Learning for Generative Policies](https://arxiv.org/abs/2606.01151). *ICML*. (LP-DS)
- Im, H., Kolobov, A., Fu, J., & Lee, Y. (2026). [Latent Policy Steering through One-Step Flow Policies](https://arxiv.org/abs/2603.05296). *RSS*. (LPS)
- Zhang, S., So, O., Ahmad, H. M. S., Yu, E. Y., Cleaveland, M., Black, M., & Fan, C. (2026). [ReFORM: Reflected Flows for On-Support Offline RL via Noise Manipulation](https://arxiv.org/abs/2602.05051). *ICLR*.
- Murray, M., Chen, D., Bagaria, S., et al. (2026). [FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space](https://arxiv.org/abs/2607.08877).
- Lu, J., Qin, X., et al. (2026). [UniSteer: Unified Noise Steering for Efficient Human-Guided VLA Adaptation](https://arxiv.org/abs/2605.10821).
- Tang, A., Chen, W., Wagenmaker, A., Finn, C., & Levine, S. (2026). [Improving Robotic Generalist Policies via Flow Reversal Steering](https://arxiv.org/abs/2606.13675).
- Patil, O., Biza, O., Weng, T., et al. (2026). [You've Got a Golden Ticket: Improving Generative Robot Policies With A Single Noise Vector](https://arxiv.org/abs/2603.15757).
- Zhang, R., Fu, H., Jia, M., Konidaris, G., Du, Y., & da Silva, B. C. (2026). [From Noise to Control: Parameterized Diffusion Policies](https://arxiv.org/abs/2606.00336). *ICML*. (PDP)
- Kim, J., Park, J., Jeon, S., & Kim, S. (2026). [Better Source, Better Flow: Learning Condition-Dependent Source Distribution for Flow Matching](https://arxiv.org/abs/2602.05951).
- Machado, B., Chapin, A., Dellandréa, E., & Chen, L. (2026). [Flowing With Purpose: Latent Action Guided Flow Matching Policies for Robotic Manipulation](https://arxiv.org/abs/2606.23420). (LAFM)
- Kingma, D. P., & Welling, M. (2014). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). *ICLR*.
- Levine, S. (2018). [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909).
- Cremer, C., Li, X., & Duvenaud, D. (2018). [Inference Suboptimality in Variational Autoencoders](https://proceedings.mlr.press/v80/cremer18a.html). *ICML*.

<details>
<summary>Cite this post</summary>
<p>If you find this post useful in your work, you can cite it as:</p>
<blockquote>
<p>Chi, Yuhan. “Choose the Noise, Keep the Flow.” October 18, 2026. <a href="https://yuhanchi.github.io/writing/choose-the-noise/">https://yuhanchi.github.io/writing/choose-the-noise/</a></p>
</blockquote>
<p>Or use the following BibTeX entry:</p>
<pre><code class="language-bibtex">@misc{chi2026choosethenoise,
  author = {Chi, Yuhan},
  title = {{Choose the Noise, Keep the Flow}},
  year = {2026},
  month = oct,
  day = {18},
  howpublished = {Blog post},
  url = {https://yuhanchi.github.io/writing/choose-the-noise/}
}</code></pre>
<p>This citation refers to the blog’s explanation and synthesis. Please cite the original papers above for their methods and results.</p>
</details>
