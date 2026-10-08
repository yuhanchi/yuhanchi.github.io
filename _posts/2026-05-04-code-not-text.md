---
title: "When CoT Surface Features Transfer"
date: "2026-05-04"
description: "A technical note on measurement non-invariance across math, science, and coding reasoning traces."
category: "tech"
tags: ["AI", "code", "mathematics"]
---
This note summarizes my project [code-not-text](https://github.com/Chi-Shan0707/code-not-text). The interactive version is here: [demo](https://chi-shan0707.github.io/code-not-text/demo/). The full technical report is here: [PDF report](https://github.com/Chi-Shan0707/code-not-text/blob/main/project_report/project_report.pdf).

## Question

Can cheap summaries of a chain-of-thought trace predict whether a solution is correct?

I study one deliberately narrow feature family: hand-crafted CoT-surface and token-trajectory descriptors. The point is not to test every possible verifier. The point is to ask whether this cheap, interpretable feature family transfers across domains.

## Setup

The experiments use **DeepSeek-R1-0528-Qwen3-8B** at temperature 1.0 across three domains:

- **Math:** AIME24, AIME25, BRUMO25, and HMMT25; **7,680 aligned runs**
- **Science:** GPQA-style tasks; **12,672 runs**
- **Coding:** LiveCodeBench-v5 tasks; **10,688 runs**

The feature family includes token-confidence summaries, token-trajectory statistics, trajectory continuity, novelty, reflection count, and a small activation-derived descriptor. Evaluation uses problem-grouped splits. I report AoA (AUC-of-AUROC across trace anchors), AUROC@100%, and best-of-N reranking.

## Main Result

The same feature family behaves very differently across domains.

| Domain | AoA | AUROC@100% | Best-of-N=64 pass@1 |
|---|---:|---:|---:|
| Math | 0.958 | 0.982 | +10.0 pp |
| Science | 0.799 | 0.841 | +8.0 pp |
| Coding | 0.434 | 0.407 | -0.6 pp |

This is the central finding: the feature family is strong in math, narrower in science, and weak for coding correctness on unseen problems.

## Interpretation

The same feature names do not necessarily measure the same latent construct.

In math, features such as reflection count and trajectory continuity can track whether reasoning is converging. In science, the signal is more confidence-heavy. In coding, executable correctness is farther away from the surface form of the reasoning trace. A trace can look organized and confident while still implementing the wrong program logic.

I view this as a measurement non-invariance result: a feature family can be useful in one domain without being a general-purpose correctness proxy.

## Robustness Checks

The coding result is not just one failed probe. I tested several possible explanations:

- an 83-scalar coding-specific feature sweep,
- grouped feature ablations,
- a coding-specific CoT-only judge,
- nonlinear MLP classifiers,
- SSL pre-training on 42K unlabeled traces,
- token-level de-knotting.

None produced a strong generalizable coding verifier from this feature family.

## Semantic Knots

I also ran a smaller semantic analysis around what I call **semantic knots**: local places in the visible reasoning prose where the trace appears to lose state control and does not immediately repair itself. Ordinary uncertainty, normal exploration, and clean one-step self-correction do not count.

This analysis supports the same cross-domain story:

- **Math:** knots are common and strongly associated with incorrect answers.
- **Science:** knots appear less often, and the correctness relationship is weaker.
- **Coding:** knots are rare and do not meaningfully separate correct from incorrect solutions.

The de-knotting check is especially useful. Removing knot spans reduces math signal, which suggests that those spans carry failure information. In coding, de-knotting does not reveal a hidden verifier; it leaves the weak signal essentially weak. That is why I read the result as measurement non-invariance rather than a missing preprocessing trick.

## Boundary

This does **not** claim that all text-based verifiers fail. It also does not rule out code-aware models, execution-aware selectors, hidden-state probes, raw-text classifiers, or LLM judges.

The narrower conclusion is: cheap CoT-surface features are domain-specific measurement instruments, not general-purpose correctness proxies.
