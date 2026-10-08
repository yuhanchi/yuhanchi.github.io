---
title: From bandits to PPO, in one page
description: A compact path from the multi-armed bandit to the clipped objective behind modern policy optimization.
tags: [reinforcement learning, notes]
math: true
---

Reinforcement learning can look like a zoo of algorithms. This note follows one thread through it: *how do we improve a decision rule using only the rewards it produces?* Each step below fixes one problem left by the step before.

## 1. Bandits: choosing without a model

A $$K$$-armed bandit has unknown reward means $$\mu_1, \dots, \mu_K$$. At each round $$t$$ we pick an arm $$a_t$$ and observe a noisy reward. A natural yardstick is the **regret** after $$T$$ rounds,

$$
R_T \;=\; T\mu^\star - \mathbb{E}\Big[\sum_{t=1}^{T} \mu_{a_t}\Big], \qquad \mu^\star = \max_k \mu_k .
$$

Always pulling the arm that currently looks best can lock onto a bad arm forever. The *upper confidence bound* rule adds an optimism bonus that shrinks as an arm is tried more often:

$$
a_t = \arg\max_k \; \hat\mu_k + \sqrt{\frac{2\ln t}{n_k}} .
$$

This gives regret that grows only logarithmically in $$T$$. The lesson to carry forward: **exploration has to be paid for on purpose.**

## 2. MDPs: decisions that change the future

In a Markov decision process the action also moves us to a new state $$s'$$, so a choice can matter long after its immediate reward. A policy $$\pi(a \mid s)$$ is judged by its expected discounted return

$$
J(\pi) = \mathbb{E}_\pi\Big[\sum_{t \ge 0} \gamma^t r_t\Big], \qquad 0 \le \gamma < 1 .
$$

## 3. Policy gradients: follow the slope of $$J$$

Parameterize the policy as $$\pi_\theta$$. The policy gradient theorem says the gradient can be estimated from sampled trajectories, with no model of the environment:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A^{\pi_\theta}(s_t, a_t)\big].
$$

Here $$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$ is the *advantage*: how much better action $$a$$ is than the policy's average behaviour in state $$s$$. Subtracting $$V^{\pi}$$ does not change the expectation, but it can greatly reduce variance.

## 4. PPO: don't step too far

Plain gradient ascent has no sense of scale: a single large step can ruin a good policy. Proximal Policy Optimization reuses a batch of data collected by $$\pi_{\text{old}}$$ and maximizes a *clipped* surrogate. With the probability ratio $$r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$$,

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat A_t,\; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big].
$$

Once the ratio leaves $$[1-\epsilon, 1+\epsilon]$$ in the direction that would increase the objective, the gradient vanishes. The update can still improve the policy, but it is no longer rewarded for moving far from the data it learned from.

> Each step answers the same question with more structure: *how much should I trust what I have just seen?*

## A minimal sketch

```python
ratio = torch.exp(logp_new - logp_old)
unclipped = ratio * adv
clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
loss = -torch.min(unclipped, clipped).mean()
```

That is the whole trick — a few lines on top of the policy gradient.
