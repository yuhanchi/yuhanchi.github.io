---
title: "Just a Flow? Notes on Sundial"
date: "2026-11-17"
description: "Notes on Sundial's TimeFlow head, why its flow-matching design and calibration are surprisingly trivial."
category: "tech"
tags: ["AI", "time-series"]
math: true
published: true
preview_image: "/assets/img/posts/timeflow-sundial/sundial-fig2.png"
preview_alt: "Overall architecture of Sundial (Figure 2): continuous patch tokenization, causal Transformer backbone, and the TimeFlow Loss flow-matching head."
---
After posting [Chronos-2]({% post_url 2026-11-06-time-series-as-tokens %}) on Xiaohongshu, a reader commented and recommended [Sundial](https://arxiv.org/abs/2502.00816) (*Sundial: A Family of Highly Capable Time Series Foundation Models*, ICML 2025) in response to my [open question on flow-matching heads]({% post_url 2026-11-06-time-series-as-tokens %}#6-possibly-open-questions). On the surface, Sundial's pitch is appealing: a native, continuous-valued, decoder-only time-series foundation model pre-trained on a trillion time points (`TimeBench`), replacing discrete tokenization and parametric priors with a generative **TimeFlow Loss** based on flow matching.

<figure>
  <a href="{{ '/assets/img/posts/timeflow-sundial/sundial-fig2.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/timeflow-sundial/sundial-fig2.png' | relative_url }}" alt="Overall architecture of Sundial (Figure 2): continuous patch tokenization, causal Transformer backbone, and the TimeFlow Loss flow-matching head." width="2057" height="596" loading="lazy"></a>
  <figcaption>Figure 2 from the Sundial paper: continuous patch tokenization, a Pre-LN causal Transformer with RoPE, and the TimeFlow Loss flow-matching head.</figcaption>
</figure>

Since the Transformer backbone and patch tokenization are standard, the **only genuinely new methodological claim** in Sundial is the **TimeFlow** output head. Below is a brief summary of how it works—followed by why this design, and the "test-time calibration" story around it, does not hold up well under scrutiny.

---

## 1. The Only New Piece: The TimeFlow Head

Sundial pairs a standard front end—two-stage instance normalization (`Re-Norm`), continuous patch embedding ($$P = 16$$), and a Pre-LN causal Transformer with RoPE, FlashAttention, and KV caching outputting $$\mathbf{h}_i \in \mathbb{R}^D$$—with **TimeFlow Loss** (§4.1.3, Eqs. 6–9).

Instead of mapping $$\mathbf{h}_i$$ directly to a point forecast $$\hat{\mathbf{y}}_i \in \mathbb{R}^F$$ or a set of quantiles, Sundial attaches a small MLP (`FM-Net`, 3–6 layers) conditioned on $$\mathbf{h}_i$$ via AdaLN and trains it with Conditional Flow Matching (CFM): draw $$\mathbf{y}_i^{(0)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_F)$$ and $$t \sim \mathcal{U}[0, 1]$$, form the linear interpolation $$\mathbf{y}_i^{(t)} = t \mathbf{y}_i + (1 - t)\mathbf{y}_i^{(0)}$$ toward the ground truth $$\mathbf{y}_i \in \mathbb{R}^F$$, and regress the velocity:

$$
\mathcal{L}(\theta, \mathbf{h}_i) = \mathbb{E}_{t, \mathbf{y}_i^{(0)}, \mathbf{y}_i} \left[ \left\| \mathrm{FM\text{-}Net}\!\left(\mathbf{y}_i^{(t)}, t, \mathbf{h}_i\right) - \left(\mathbf{y}_i - \mathbf{y}_i^{(0)}\right) \right\|_2^2 \right].
$$

At inference time, Sundial fixes $$\mathbf{h}_i$$, samples $$\hat{\mathbf{y}}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_F)$$, and runs $$K = 50$$ uniform Euler steps ($$\Delta t = 1/K$$) to generate one trajectory. Repeating this $$M$$ times ($$M = 20$$ on FEV, $$M = 100$$ on GIFT-Eval) with different noises yields the sample median and empirical quantiles.

---

## 2. Why This Flow-Matching Head Is Trivial

### 2.1 Textbook CFM + MAR, Zero Domain Innovation

Mathematically, Eqs. (6)–(8) are the most vanilla form of linear-interpolation Conditional Flow Matching. Architecturally, bolting a small AdaLN-conditioned MLP diffusion/flow head onto an autoregressive Transformer is a direct port of **MAR** (*Autoregressive Image Generation without Vector Quantization*, [Li et al., 2024](https://arxiv.org/abs/2406.11838))—which the paper itself cites.

Even when predicting $$F = 720$$ future steps at once, `FM-Net` is just a 3-to-6-layer MLP (Table 5) with no temporal structure or inductive bias. Calling this a bespoke "TimeFlow" mechanism overstates what is essentially a standard continuous-output wrapper.

### 2.2 Sundial's Own Ablations Expose That the Backbone and 1T Data Do the Heavy Lifting

If TimeFlow were doing something transformative, replacing it with a plain MSE head should hurt performance substantially. Look at **Table 3** (TSLib zero-shot MSE, same backbone and 1T `TimeBench` pre-training):

<figure>
  <a href="{{ '/assets/img/posts/timeflow-sundial/sundial-table3.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/timeflow-sundial/sundial-table3.png' | relative_url }}" alt="Table 3 from Sundial: zero-shot MSE across ETTm1, ETTm2, ETTh1, ETTh2, ECL, and Weather comparing TimeFlow, Diffusion, and MSE." width="1010" height="420" loading="lazy"></a>
  <figcaption>Table 3 from the Sundial paper: zero-shot MSE comparing TimeFlow, Diffusion, and plain MSE on the same backbone and pre-training scale.</figcaption>
</figure>

- Average MSE across six benchmarks: **0.290** (`TimeFlow`) vs. **0.296** (`MSE`)—a difference of 0.006.
- On `ETTh1` (`0.404` vs. `0.411`) and `Weather` (`0.231` vs. `0.234`), **plain `MSE` actually beats `TimeFlow`**.

Even more telling is **Table 7** (zero-shot CRPS across the same six TSLib datasets and GIFT-Eval):

<figure>
  <a href="{{ '/assets/img/posts/timeflow-sundial/sundial-table7.png' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/timeflow-sundial/sundial-table7.png' | relative_url }}" alt="Table 7 from Sundial: zero-shot CRPS across ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, and GIFT-Eval comparing TimeFlow Loss, Diffusion Loss, and MSE Loss." width="2054" height="372" loading="lazy"></a>
  <figcaption>Table 7 from the Sundial paper: zero-shot CRPS comparing TimeFlow Loss, Diffusion Loss, and MSE Loss.</figcaption>
</figure>

- Plain `MSE` (which has zero distributional modeling and reduces CRPS to MAE) scores `0.0063, 0.0040, 0.0058, 0.0032, 0.0080, 0.0023` on the six TSLib datasets—virtually indistinguishable from `TimeFlow`'s `0.0059, 0.0037, 0.0057, 0.0029, 0.0082, 0.0021`, and `MSE` even wins on `ECL`.

I suppose the main reason Sundial performs well is clearly the **1-trillion-point `TimeBench` dataset + continuous patching + RoPE/Pre-LN Transformer**, not the flow-matching head.

### 2.3 "Repeated Sampling Lowers MASE and WQL" Is Trivial Statistics, Not Calibration

In §5.4 (Figure 7), Sundial sweeps the number of sampled trajectories $$M$$ and writes:

> *"Generative modeling facilitates the flexibility to calibrate the final prediction during inference. Based on the median-based forecasting strategy, i.e., starting from multiple noise of a standard Gaussian and calculating the median of raw predictions, there are two configurations to calibrate final predictions..."*

This claim does not hold up. Mathematically, let $$\hat{\mathbf{y}}^{(1)}, \dots, \hat{\mathbf{y}}^{(M)} \stackrel{\text{i.i.d.}}{\sim} p_\theta(\mathbf{y} \mid \mathbf{x})$$ denote $$M$$ sampled trajectories:

1. **Why MASE drops**: For any convex error metric $$\ell(\cdot, \mathbf{y})$$ (such as the $$L_1$$ error in MASE), aggregating $$M$$ i.i.d. draws via the sample mean $$\bar{\mathbf{y}}_M$$ or coordinate-wise median $$\tilde{\mathbf{y}}_M$$ reduces Monte Carlo variance at rate $$\mathcal{O}(1/M)$$ (by Jensen's inequality, $$\mathbb{E}[\ell(\bar{\mathbf{y}}_M, \mathbf{y})] \le \mathbb{E}[\ell(\hat{\mathbf{y}}^{(1)}, \mathbf{y})]$$, and the asymptotic variance of the sample median scales as $$1 / (4 M f_\theta(\mathrm{med})^2)$$). Taking a deterministic MSE model, injecting artificial Gaussian noise, and averaging $$M$$ draws would produce the exact same downward curve.
2. **Why WQL drops**: An implicit flow head cannot output quantiles directly; at each step $$t$$ it must estimate the $$q$$-quantile from the empirical CDF $$\hat{F}_{M,t}(u) = \frac{1}{M}\sum_{m=1}^M \mathbf{1}\{\hat{y}_t^{(m)} \le u\}$$ as $$\hat{Q}_{q,t}^{(M)} = \hat{F}_{M,t}^{-1}(q)$$, and evaluate the pinball loss $$\rho_q(y, u) = \max\big(q(y - u), (q - 1)(y - u)\big)$$. At $$M = 1$$, every quantile collapses to the same single draw ($$\hat{Q}_{0.1,t}^{(1)} = \dots = \hat{Q}_{0.9,t}^{(1)} = \hat{y}_t^{(1)}$$), giving zero interval width and high pinball loss. Increasing $$M$$ simply reduces the $$\mathcal{O}(M^{-1/2})$$ finite-sample estimation error of $$\hat{Q}_{q,t}^{(M)}$$ relative to the model's already-learned $$F_{\theta,t}^{-1}(q)$$. It does *not* calibrate $$F_\theta$$ against the true distribution $$F_{\text{true}}$$; if $$F_\theta$$ is miscalibrated, 10,000 samples just estimate the wrong quantiles more precisely.

Mathematically, *any* sampling-based forecaster—including [Chronos](https://arxiv.org/abs/2403.07815)—gets the exact same free drop in MASE and WQL when sweeping $$M$$. In fact, at the same $$M = 20$$ on the FEV leaderboard (Figure 4), Chronos-1 still outperforms Sundial in MASE and WQL (Sundial's real win there is 35x faster inference from patching, not better probabilistic modeling). Meanwhile, [Chronos-2]({% post_url 2026-11-06-time-series-as-tokens %}) bypasses sampling altogether with a direct Quantile Regression head—effectively reaching the $$M \to \infty$$ limit in **one** forward pass without running $$M \times K = 100 \times 50$$ ODE steps.

### 2.4 A Missing Baseline and a Metric Mismatch

Two final gaps make the empirical case for TimeFlow even weaker:

- **No Quantile Head ablation**: In §5.3, Sundial compares `TimeFlow` only against `MSE` (no uncertainty) and `DDPM Diffusion` (harder to integrate in 50 steps). It never compares against a **Multi-Quantile MLP Head (Pinball Loss)** or a parametric mixture head on the exact same backbone and `TimeBench` corpus.
- **Joint generator, marginal metrics**: In principle, the one real advantage of generating full vectors $$\hat{\mathbf{y}} \in \mathbb{R}^F$$ via flow matching over a quantile head is capturing **joint multi-step dependencies** $$p(\mathbf{y}_{t+1:t+F} \mid \mathbf{x}_{1:t})$$ rather than per-step marginals. Yet every single metric reported in the paper—MSE, MAE, MASE, WQL, and per-step CRPS—is strictly **marginal**. Sundial never evaluates joint trajectory quality (e.g., Energy Score or temporal autocorrelation).

In short, I think Sundial's engineering contribution—curating `TimeBench` and scaling a clean continuous-patch autoregressive Transformer—is solid. Its flagship `TimeFlow` head and "test-time calibration" narrative, however, feel far more like packaging than substance.

---

## 3. (Possibly) Open Question

What kind of modeling can genuinely draw on the expressiveness (which generative models may well have!) for time series? If inference still boils down to repeated sampling just to extract marginal medians and quantiles, it does not exploit what generative models are actually good for.

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
