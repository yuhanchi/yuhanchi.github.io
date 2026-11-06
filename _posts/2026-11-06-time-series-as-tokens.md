---
title: "Time Series as Tokens: Notes on Chronos-2"
date: "2026-11-06"
description: "Notes on how Chronos-2 turns time series into tokens for universal forecasting."
category: "tech"
tags: ["AI", "time-series"]
math: true
preview_image: "/assets/img/posts/chronos2.webp"
preview_alt: "The complete Chronos-2 pipeline: robust scaling, tokenization, alternating time and group attention, and quantile forecasting."
---
[Chronos-2](https://arxiv.org/abs/2510.15821) ([GitHub code](https://github.com/amazon-science/chronos-forecasting), [Hugging Face weights](https://huggingface.co/amazon/chronos-2)) is a foundation model that turns time series into tokens and models them much like text. This is the first paper I have read in this field.

The paper is titled *Chronos-2: From Univariate to Universal Forecasting*, and the central idea is simple. Previous pretrained time-series models (such as the original [Chronos](https://arxiv.org/abs/2403.07815), [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting), [TimesFM](https://arxiv.org/abs/2310.10688), [Sundial](https://arxiv.org/abs/2502.00816), and [TiRex](https://arxiv.org/abs/2505.23719)) were mostly restricted to:

- **Univariate forecasting**: predicting a single series at a time while ignoring cross-series dependencies.
- **No known future information**: unable to condition on known future inputs (such as planned retail promotions, holidays, or weather forecasts).
- **No covariates**: ignoring exogenous variables altogether, or struggling to scale when variables are flattened into the context.

Chronos-2 moves from univariate to **universal** zero-shot forecasting: a single encoder-only foundation model that handles univariate, multivariate, and covariate-informed tasks (both past-only and known future covariates, numeric or categorical) via **in-context learning (ICL)** across groups of series, with $$\mathcal{O}(V)$$ memory scaling in the number of variates $$V$$.

<details markdown="1">
<summary><strong>Table 1.</strong> Capabilities of pretrained forecasting models (summarized from Table 1 of the paper).</summary>

| Model | Univariate | Multivariate | Past-Only Covariates | Known Covariates | Categorical Covariates | Cross Learning | Memory Scaling |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chronos-2** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | $$\mathcal{O}(V)$$ |
| [Toto-1.0](https://arxiv.org/abs/2505.14766) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | $$\mathcal{O}(V)$$ |
| [TabPFN-TS](https://arxiv.org/abs/2501.02945) | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | $$\mathcal{O}(V)$$ |
| [COSMIC](https://arxiv.org/abs/2506.03128) | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | $$\mathcal{O}(V^2)$$ |
| [Moirai-1.0](https://arxiv.org/abs/2402.02592) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | $$\mathcal{O}(V^2)$$ |
| Chronos-Bolt / Moirai-2.0 / Sundial / TimesFM-2.5 / TiRex | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | — |

</details>

---

## 1. Definitions

Given a historical time series $$\mathbf{Y}_{1:T} = [\mathbf{y}_1, \dots, \mathbf{y}_T]$$ of length $$T$$—where each observation $$\mathbf{y}_t \in \mathbb{R}^D$$ can be univariate ($$D = 1$$) or multivariate ($$D > 1$$)—the goal is to predict the next $$H$$ time steps $$\mathbf{Y}_{T+1:T+H}$$ over the forecast horizon $$H$$.

Forecasts may be supported by covariates (exogenous variables) $$\mathbf{X}_{1:T+H} = [\mathbf{x}_1, \dots, \mathbf{x}_{T+H}]$$ with $$\mathbf{x}_t \in \mathbb{R}^M$$. *Past-only covariates* are observed only in the historical context ($$t \le T$$); *known future covariates* are also available over the forecast horizon ($$t > T$$).

To unify targets and covariates into one matrix representation, Chronos-2 stacks the $$D$$ targets and $$M$$ covariates at each step:

- **Historical matrix** $$\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_T] \in \mathbb{R}^{(D+M) \times T}$$, where $$\mathbf{v}_t = [\mathbf{y}_t; \mathbf{x}_t] \in \mathbb{R}^{D+M}$$.
- **Future matrix** $$\mathbf{W} = [\mathbf{w}_{T+1}, \dots, \mathbf{w}_{T+H}] \in \mathbb{R}^{(D+M) \times H}$$, where entries for known future covariates contain their known values $$\mathbf{x}_t$$, while entries for targets and past-only covariates are marked as missing (`NaN`).

Now the problem becomes clear: how should we forecast?

---

## 2. Preprocessing and Tokenization

Since large language models have achieved great success, we naturally want to mimic them. First, however, we need to deal with some special properties of time series.

### 2.1 Quirky Data and Imbalanced Value Ranges

1. **Categorical covariates** (handled via **target encoding** from [Micci-Barreca, 2001](https://doi.org/10.1145/507533.507538) and [scikit-learn, Pedregosa et al., 2011](https://jmlr.org/papers/v12/pedregosa11a.html), or **ordinal encoding**):
   - When the target is univariate ($$D = 1$$), Chronos-2 converts each categorical covariate into a real number using smoothed per-series target encoding—mapping each category to the smoothed historical mean of the target when that category occurred.
   - When the target is multivariate ($$D > 1$$), it falls back to ordinal encoding (assigning a unique integer code to each category).

2. **Imbalanced value ranges** (handled via **standardization + $$\sinh^{-1}$$ variance-stabilizing transformation**, following econometrics in [Burbidge et al., 1988](https://doi.org/10.1080/01621459.1988.10478575) and electricity price forecasting in [Uniejewski & Weron, 2018](https://doi.org/10.3390/en11082039)):
   - Different series sit at arbitrary scales and often contain extreme spikes or outliers. For each dimension $$d \in \{1, \dots, D+M\}$$, Chronos-2 first computes the historical mean $$\mu_d$$ and standard deviation $$\sigma_d$$ (excluding missing values), standardizes both history $$\mathbf{V}$$ and future $$\mathbf{W}$$, and then applies the inverse hyperbolic sine $$\sinh^{-1}(z) = \ln\!\big(z + \sqrt{z^2 + 1}\big)$$:
     $$
     \tilde{v}_{t,d} = \sinh^{-1}\!\left(\frac{v_{t,d} - \mu_d}{\sigma_d}\right) \quad (t \le T), \qquad
     \tilde{w}_{t,d} = \sinh^{-1}\!\left(\frac{w_{t,d} - \mu_d}{\sigma_d}\right) \quad (t > T).
     $$
   - Because $$\sinh^{-1}(z) \approx z$$ near zero and $$\sinh^{-1}(z) \approx \operatorname{sgn}(z)\ln(2|z|)$$ for large $$|z|$$, this log-like transformation compresses extreme values while handling zeros and negative numbers. Concatenating and transposing the normalized history and future gives $$\mathbf{U} = [\tilde{\mathbf{V}}, \tilde{\mathbf{W}}]^\top \in \mathbb{R}^{(T+H) \times (D+M)}$$.

### 2.2 Time Embeddings (Like Positional Embeddings)

How does the model know where each step sits in time? Chronos-2 uses two positional signals:

1. **Explicit relative time-index meta feature $$\mathbf{j}$$**: For each univariate column $$\mathbf{u}_d \in \mathbb{R}^{T+H}$$, it constructs a normalized relative time index scaled by the maximum context length $$C$$:
   $$
   \mathbf{j} = \left[-\frac{T}{C},\; -\frac{T-1}{C},\; \dots,\; -\frac{1}{C},\; 0,\; \frac{1}{C},\; \dots,\; \frac{H-1}{C}\right].
   $$

2. **Rotary Position Embeddings ([RoPE, Su et al., 2024](https://arxiv.org/abs/2104.09864))**: Inside the transformer's temporal self-attention layers, Chronos-2 replaces T5's classic bucketed relative position bias with RoPE (following modern LLMs like [Llama 2, Touvron et al., 2023](https://arxiv.org/abs/2307.09288)).

### 2.3 Masking Future Data and Missing Values

In time series forecasting, we cannot rely on unknown future target data. In a decoder-only LLM, causal masking prevents tokens from attending to the future and generation proceeds autoregressively. Chronos-2 instead adopts an **encoder-only** architecture (based on the [T5 encoder, Raffel et al., 2020](https://jmlr.org/papers/v21/20-074.html)) and uses **feature masking** so it can predict multiple future patches in a single forward pass:

- For each dimension $$d$$, a binary mask $$\mathbf{m}_d \in \{0, 1\}^{T+H}$$ is set to $$1$$ where a value is observed and $$0$$ where it is missing or unknown.
- After building $$\mathbf{m}_d$$, all missing entries in $$\mathbf{u}_d$$ are filled with zeros.
- This single mask serves two roles simultaneously:
  1. It tells the model which historical observations ($$t \le T$$) were missing (`NaN`).
  2. In the future window ($$t > T$$), it distinguishes **targets and past-only covariates** ($$m_{t,d} = 0$$, zeroed out so no future target leaks) from **known future covariates** ($$m_{t,d} = 1$$, retaining their known future values $$\tilde{w}_{t,d}$$).

### 2.4 "Tokenizing" and "Embedding" via Patching

Instead of quantizing each scalar time step into a discrete vocabulary bin (as the first [Chronos](https://arxiv.org/abs/2403.07815) did), Chronos-2 follows [PatchTST (Nie et al., 2023)](https://arxiv.org/abs/2211.14730) and [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) by grouping adjacent time steps into non-overlapping **patches** of length $$P$$:

- The historical section (length $$T$$) and future section (length $$H$$) of $$\mathbf{u}_d$$, $$\mathbf{j}$$, and $$\mathbf{m}_d$$ are patched separately. If $$T$$ or $$H$$ is not a multiple of $$P$$, zero padding is applied on the **left** for history and on the **right** for the future.
- Let $$\bar{\mathbf{u}}_p, \bar{\mathbf{j}}_p, \bar{\mathbf{m}}_p \in \mathbb{R}^P$$ denote the $$p$$-th patch of the normalized series, time index, and mask. They are concatenated into a vector in $$\mathbb{R}^{3P}$$ and mapped into the transformer's hidden space $$\mathbb{R}^{D_{\text{model}}}$$ using an input residual network $$f^{\mathrm{in}}_\phi: \mathbb{R}^{3P} \to \mathbb{R}^{D_{\text{model}}}$$:
  $$
  \mathbf{h}_p = f^{\mathrm{in}}_\phi\big([\bar{\mathbf{u}}_p,\, \bar{\mathbf{j}}_p,\, \bar{\mathbf{m}}_p]\big).
  $$
- Between the historical context patches and the future patches, Chronos-2 inserts a learned special **`[REG]` token** (short for *register* token) that acts both as a context–future separator and as an **attention sink** ([Xiao et al., 2024](https://arxiv.org/abs/2309.17453)) to absorb excess attention mass.

---

## 3. Architecture and Code Walkthrough

<figure>
  <a href="{{ '/assets/img/posts/chronos2.webp' | relative_url }}" title="Open full size"><img src="{{ '/assets/img/posts/chronos2.webp' | relative_url }}" alt="The complete Chronos-2 pipeline: input series undergo robust scaling, meta-feature concatenation, patching, and patch embedding, then pass through N transformer blocks alternating Time Attention and Group Attention, and finally a quantile head and unscaling." width="1748" height="723" loading="lazy"></a>
  <figcaption><strong>Figure 1.</strong> The Chronos-2 pipeline: scaling, patch embedding, alternating time and group attention, quantile prediction, and unscaling (from <a href="https://arxiv.org/abs/2510.15821">Ansari et al., 2025</a>).</figcaption>
</figure>

As shown in Figure 1, the full pipeline flows from left to right:

$$
\text{Raw Input } (\mathbf{V}, \mathbf{W})
\;\xrightarrow{\text{Std} + \sinh^{-1}}\;
\mathbf{U}
\;\xrightarrow{+\,(\mathbf{j},\, \mathbf{m})}\;
\text{Patching \& } f^{\mathrm{in}}_\phi
\;\xrightarrow{+\,\texttt{[REG]}}\;
\bigl(\text{Time Attn} \to \text{Group Attn} \to \text{FFN}\bigr)_{\times N}
\;\xrightarrow{\text{Quantile Head}}\;
\hat{\mathbf{Z}}
\;\xrightarrow{\sinh}\;
\hat{\mathbf{Y}}.
$$

Let's look at the core attention and encoder code in [`src/chronos/chronos2/layers.py`](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/layers.py) and [`src/chronos/chronos2/model.py`](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/model.py).

### 3.1 Vanilla Multi-Head Attention (`MHA`)

In [`layers.py`](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/layers.py), `MHA` implements standard multi-head projections with an optional RoPE flag (`use_rope: bool = True`):

```python
class MHA(nn.Module):
    """Multi-head Attention Layer shared by both Time Attention and Group Attention."""

    def __init__(self, config: Chronos2CoreConfig, use_rope: bool = True):
        super().__init__()
        self.d_model: int = config.d_model          # Hidden dimension D_model
        self.kv_proj_dim: int = config.d_kv         # Per-head dimension d_kv
        self.n_heads: int = config.num_heads        # Number of attention heads H
        self.dropout: float = config.dropout_rate
        self.inner_dim: int = self.n_heads * self.kv_proj_dim
        self.config = config

        # Bias-free linear projections (T5 style)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # TimeSelfAttention sets use_rope=True; GroupSelfAttention sets use_rope=False
        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = Chronos2RotaryEmbedding(config=config)

    def _eager_attention(
        self,
        query_states: torch.Tensor,  # shape: (batch, n_heads, q_len, d_kv)
        key_states: torch.Tensor,    # shape: (batch, n_heads, kv_len, d_kv)
        value_states: torch.Tensor,  # shape: (batch, n_heads, kv_len, d_kv)
        mask: torch.Tensor,          # additive mask: 0.0 for valid tokens, -inf for masked tokens
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute raw dot-product attention scores: Q * K^T -> (batch, n_heads, q_len, kv_len)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # "bnqd,bnkd->bnqk"
        scores += mask  # Mask out padded patches or series from other groups
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights
```

### 3.2 Assembling the Encoder Block: Time Attention + Group Attention

Next, we assemble the layers into `Chronos2EncoderBlock` in [`model.py`](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/model.py). Each block alternates between **Time Self-Attention**, **Group Self-Attention**, and a **Feed-Forward Network**:

```python
class Chronos2EncoderBlock(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        assert not getattr(config, "is_decoder", False)  # Strictly an encoder-only block

        self.layer = nn.ModuleList()
        self.layer.append(TimeSelfAttention(config))   # Layer 0: attend across time patches (with RoPE)
        self.layer.append(GroupSelfAttention(config))  # Layer 1: attend across grouped series (no RoPE)
        self.layer.append(FeedForward(config))         # Layer 2: Pre-Norm MLP with residual connection

    def forward(
        self,
        hidden_states: torch.Tensor,   # shape: (B, T_patches, d_model), where B = total series/variates in batch
        *,
        position_ids: torch.Tensor,    # shape: (1, T_patches), patch indices [0, 1, ..., T_patches - 1]
        attention_mask: torch.Tensor,  # shape: (B, 1, 1, T_patches), masks padded context patches along time
        group_time_mask: torch.Tensor, # shape: (T_patches, 1, B, B), masks cross-group & padded entries
        output_attentions: bool = False,
    ) -> Chronos2EncoderBlockOutput:
        # Step 1: Time Self-Attention — each series independently aggregates information
        #         across its own historical patches, [REG] token, and future patches
        time_self_attn_outputs: AttentionOutput = self.layer[0](
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = time_self_attn_outputs[0]

        # Step 2: Group Self-Attention — at each patch index t, all series/variates/covariates
        #         sharing the same group_id exchange information across the batch axis B
        group_self_attn_outputs: AttentionOutput = self.layer[1](
            hidden_states, attention_mask=group_time_mask, output_attentions=output_attentions
        )
        hidden_states = group_self_attn_outputs[0]

        # Step 3: Pointwise Feed-Forward Network (RMSNorm -> Linear -> Act -> Dropout -> Linear + Residual)
        hidden_states = self.layer[2](hidden_states)

        return Chronos2EncoderBlockOutput(
            hidden_states=hidden_states,
            time_self_attn_weights=time_self_attn_outputs.attn_weights,
            group_self_attn_weights=group_self_attn_outputs.attn_weights,
        )
```

Let's compare `TimeSelfAttention` and `GroupSelfAttention` side by side in [`layers.py`](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/layers.py):

```python
class TimeSelfAttention(nn.Module):
    """Self-attention applied along the temporal (patch) axis for each individual series."""

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        # Uses RoPE because patches along the time axis have a strict sequential order
        self.self_attention = MHA(config, use_rope=True)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,   # shape: (batch, time, d_model)
        attention_mask: torch.Tensor,  # shape: (batch, 1, 1, time)
        position_ids: torch.Tensor,    # shape: (1, time)
        output_attentions: bool = False,
    ) -> AttentionOutput:
        # Pre-LayerNorm (T5-style RMSNorm) + MHA along dim=-2 (`time`) + Residual connection
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, position_ids=position_ids, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)

class GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis masked by the group attention mask."""

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        # Disable RoPE: variables/series within a group have no natural ordering (permutation-equivariant)
        self.self_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,   # Incoming shape: (batch, time, d_model)
        attention_mask: torch.Tensor,  # group_time_mask of shape: (time, 1, batch, batch)
        output_attentions: bool = False,
    ) -> AttentionOutput:
        # 1. Transpose `batch` and `time` axes so `batch` becomes the sequence axis (dim=-2)
        #    Shape: (batch, time, d_model) -> (time, batch, d_model)
        hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        normed_hidden_states = self.layer_norm(hidden_states)
        # 2. Run standard MHA: at each patch index `t`, all `batch` series attend to each other
        #    (restricted by `attention_mask` so only series with the same group_id interact)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 3. Transpose axes back to original layout: (time, batch, d_model) -> (batch, time, d_model)
        hidden_states = rearrange(hidden_states, "time batch d -> batch time d")

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)
```

Look at what `GroupSelfAttention` actually does—it is remarkably clean:

1. **Axis transposition (`batch time d -> time batch d`)**: Instead of concatenating multiple variates and covariates along the sequence dimension (which would blow up context length and scale quadratically across all patches and variates), Chronos-2 stacks all series, variates, and covariates along the **batch axis** (`batch`). Flipping `batch` and `time` makes the batch axis the sequence dimension (`dim=-2`) at each fixed patch index `time`, inspired by axial/cross-dimension attention in [Crossformer (Zhang & Yan, 2023)](https://openreview.net/forum?id=vSVLM2j9eie), [MSA Transformer (Rao et al., 2021)](https://proceedings.mlr.press/v139/rao21a.html), and [ViViT (Arnab et al., 2021)](https://arxiv.org/abs/2103.15691).
2. **No positional embedding (`use_rope=False`)**: Unlike patches along the time axis, series/variates/covariates inside a group have no natural ordering, so `GroupSelfAttention` disables RoPE and is permutation-equivariant along the group axis.
3. **Group-time masking (`_construct_and_invert_group_time_mask`)**: In a batch of size $$B$$, each row is assigned an integer group ID $$g_b$$ in a vector $$\mathbf{g} \in \mathbb{Z}^B$$. How does the encoder ensure information is only shared *within* the same group and only across *valid* (non-padded) patches? In `Chronos2Encoder`:

   ```python
   @staticmethod
   def _construct_and_invert_group_time_mask(
       group_ids: torch.Tensor,       # shape: (B,), integer group ID for each row in the batch
       attention_mask: torch.Tensor,  # shape: (B, T_patches), 1.0 if patch is valid, 0.0 if padded
       floating_type: torch.dtype,
   ) -> torch.Tensor:
       # 1. Pairwise group membership matrix of shape (B, B):
       #    group_mask[q, b] is True iff query series `q` and key series `b` belong to the same group
       group_mask = group_ids[:, None] == group_ids[None, :]

       # 2. Combine with temporal validity via outer product: (B_query, B_key) x (B_key, T_patches) -> (B_q, B_k, T)
       #    Series `q` can attend to series `b` at patch `t` iff they share a group AND `b` is observed at `t`
       group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)

       # 3. Rearrange to match GroupSelfAttention's score shape: (T_patches, n_heads=1, B_query, B_key)
       #    and convert binary mask {1, 0} into additive mask {0.0, -inf}
       group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
       group_time_mask = (1.0 - group_time_mask) * torch.finfo(floating_type).min
       return group_time_mask
   ```

### 3.3 One Unified Mechanism for Four Forecasting Setups

Because `GroupSelfAttention` is controlled purely by `group_ids` $$\mathbf{g}$$ and the future input matrix $$\mathbf{W}$$ (where `*` / `NaN` denotes masked future values), Chronos-2 solves four different forecasting setups without changing a single weight or layer:

| Task Type | Group IDs $$\mathbf{g}$$ | Future Inputs $$\mathbf{W}$$ | What `GroupSelfAttention` Does |
|---|---|---|---|
| **Univariate Forecasting** *(e.g., 3 independent series)* | $$\mathbf{g} = (1, 2, 3)$$ | All `*` (masked) | `group_mask` is the identity matrix; each series is predicted independently. |
| **Univariate Cross-Learning / Few-Shot ICL** *(3 related series in a batch)* | $$\mathbf{g} = (1, 1, 1)$$ | All `*` (masked) | Shares information across related univariate series in the batch (especially helpful for short histories / cold start). |
| **Multivariate Forecasting** *(3 co-evolving targets)* | $$\mathbf{g} = (1, 1, 1)$$ | All `*` (masked) | Jointly models cross-variate dynamics across the 3 target dimensions. |
| **Covariate-Informed Forecasting** *(1 target, 1 past-only cov., 2 known covs.)* | $$\mathbf{g} = (1, 1, 1, 1)$$ | Rows 1–2: `*`<br>Rows 3–4: $$x_{T+1:T+H}$$ | At future patch indices, masked target patches attend to the unmasked embeddings of known future covariates. |

### 3.4 Multi-Patch Quantile Head, Loss, and Unscaling

After passing through $$N$$ encoder blocks, Chronos-2 slices out the last `num_output_patches` hidden states (corresponding to the future patches) and maps them through a `ResidualBlock` (`self.output_patch_embedding`) to predict **21 quantiles** simultaneously across all future steps:

$$
\mathcal{Q} = \{0.01,\, 0.05,\, 0.1,\, 0.15,\, \dots,\, 0.85,\, 0.9,\, 0.95,\, 0.99\}.
$$

Compared to the standard 9-quantile decile grid $$\{0.1, \dots, 0.9\}$$, this grid includes more extreme tails ($$0.01, 0.05, 0.95, 0.99$$) for risk-sensitive forecasts.

- **Training objective**: Pinball / quantile regression loss averaged over valid target entries (known covariates and missing targets are masked out by `loss_mask = future_target_mask * inv_future_covariate_mask`):
  $$
  \mathcal{L}(\hat{z}, z) = \sum_{q \in \mathcal{Q}} \Big(q \cdot \max(z - \hat{z}^q, 0) + (1 - q) \cdot \max(\hat{z}^q - z, 0)\Big).
  $$
- **Two-stage curriculum**: Stage 1 pretrains with max context length $$C = 2048$$ and a small number of output patches; Stage 2 extends context length to $$C = 8192$$ and increases the number of output patches so the model can handle high-frequency long seasonalities and long horizons in a single forward pass.
- **Unscaling at inference**: Finally, `self.instance_norm.inverse` inverts the $$\sinh^{-1}$$ normalization on each predicted quantile $$\hat{z}^q_{t,d}$$:
  $$
  \hat{y}^q_{t,d} = \mu_d + \sigma_d \cdot \sinh\!\big(\hat{z}^q_{t,d}\big).
  $$

```python
# 1. Slice out the last `num_output_patches` hidden states corresponding to the future horizon
#    Shape: (batch_size, N_out_patches, d_model)
forecast_embeds = hidden_states[:, -num_output_patches:]

# 2. Project each future patch embedding through the output ResidualBlock:
#    R^(d_model) -> R^(num_quantiles * P), then reshape to (batch_size, 21, N_out_patches * P)
quantile_preds: torch.Tensor = self.output_patch_embedding(forecast_embeds)
quantile_preds = rearrange(
    quantile_preds,
    "b n (q p) -> b q (n p)",
    n=num_output_patches,
    q=self.num_quantiles,
    p=self.chronos_config.output_patch_size,
)

# 3. Unscale predictions back to the original data domain:
#    y_hat = mu_d + sigma_d * sinh(z_hat)
quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)")
quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
quantile_preds = rearrange(quantile_preds, "b (q h) -> b q h", q=self.num_quantiles)
```

---

## 4. Where Does the Training Data Come From? (Synthetic Multivariatizers)

A major bottleneck for universal time-series foundation models is data: public repositories ([Monash](https://arxiv.org/abs/2105.06643), [M4](https://doi.org/10.1016/j.ijforecast.2019.04.014), [GIFT-Eval](https://arxiv.org/abs/2410.10393), [Chronos corpus](https://arxiv.org/abs/2403.07815)) are overwhelmingly **univariate**. High-quality multivariate datasets and tasks with informative covariates are scarce.

Chronos-2's solution is striking: **its multivariate and covariate-informed capabilities are learned entirely from synthetic data.**

1. **Univariate data (real + synthetic)**: Selected real univariate datasets from Chronos and GIFT-Eval, combined with two synthetic univariate generators:
   - **TSI (Trend, Seasonality, and Irregularity)** ([Bahrpeyma et al., 2021](https://doi.org/10.1016/j.mex.2021.101459)): randomly composes trend, seasonal, and noise components.
   - **TCM (Temporal Causal Model)** ([Runge et al., 2023](https://doi.org/10.1038/s43017-023-00431-y)): samples random temporal causal graphs and generates series autoregressively.
2. **Multivariate & covariate data via *Multivariatizers***: A **multivariatizer** draws multiple independent series from base univariate generators (AR, ETS, TSI, and [KernelSynth](https://arxiv.org/abs/2403.07815)) and imposes cross-series structure on them:
   - **Cotemporaneous multivariatizers**: apply random linear or nonlinear transformations across series at the *same* time step $$t$$, inducing instantaneous cross-variate correlations.
   - **Sequential multivariatizers**: induce dependencies *across* time steps, creating lead–lag relationships (e.g., a covariate spike leading a target spike by several steps) and cointegration.
   - During training, subsets of these synthetic variates are randomly designated as targets ($$m_{t,d}=0$$ for $$t > T$$) or as known future covariates ($$m_{t,d}=1$$ for $$t > T$$), teaching `GroupSelfAttention` how to extract predictive signals from covariates in context.

---

## 5. Key Empirical Takeaways

1. **Benchmark performance**: The paper reports state-of-the-art results on fev-bench, GIFT-Eval, and Chronos Benchmark II.
2. **In-context learning**: Sharing information across related series and incorporating covariates improves forecasts.
3. **Ablations** ([Figure 8](https://arxiv.org/html/2510.15821v1#S5.F8)):
   - A small model (28M parameters) still performs well.
   - Training exclusively on synthetic data also works well, though the gap is larger on fev-bench.
   - Performance is already strong before long-context post-training, which brings further gains, especially on GIFT-Eval.

<figure>
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem;">
    <img src="{{ '/assets/img/posts/chronos2-model-size-skill.svg' | relative_url }}" alt="Skill scores comparing the 120M and 28M Chronos-2 models across three benchmarks." loading="lazy">
    <img src="{{ '/assets/img/posts/chronos2-synth-only-skill.svg' | relative_url }}" alt="Skill scores comparing Chronos-2 with a model trained only on synthetic data." loading="lazy">
    <img src="{{ '/assets/img/posts/chronos2-long-context-skill.svg' | relative_url }}" alt="Skill scores comparing Chronos-2 before and after long-context post-training." loading="lazy">
  </div>
  <figcaption><strong>Figure 8 from the paper.</strong> Ablations on model size, synthetic-only training, and long-context post-training (from <a href="https://arxiv.org/html/2510.15821v1#S5.F8">Ansari et al., 2025</a>).</figcaption>
</figure>

## 6. (Possibly) Open Questions

1. Where is the $$\sqrt{d_k}$$ factor in the attention calculation?
2. Can the tokenizer learn to adapt the patch length?
3. How long is the context window? Could the long-context problems seen in LLMs also arise here?
4. Why are two different time embeddings needed? If we already have RoPE, why add another?
5. As in distributional RL, could we use a flow-matching head to represent the forecast distribution instead of quantiles?
