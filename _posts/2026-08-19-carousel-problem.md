---
title: "The Problem on a Carousel"
date: "2026-08-19"
description: "A stopping problem about placing luggage on a carousel with limited information and a hard deadline."
category: "tech"
tags: ["mathematics", "probability", "optimization"]
math: true
---
Waiting for luggage at the airport, I noticed something interesting around the carousel.<br>



## An airport observation

The pick-up place is a carousel. Baggage falls from a fixed chute onto the loop. Ahead of the chute sits a detector that can tell whether the short stretch of belt currently passing beneath it carries anything.<br> 


A natural idea suggests itself: when the detector says "empty," drop. The rule sounds very correct. But in fact, I stood there and found many suitcases bumping into others.

I was a little confused. So I try to use some assumptions and definitions to formalize this problem.

## Formalization

First fix a unit of length, so that every bag width is a positive integer.

Switch to the coordinate frame that moves with the belt and cut the belt into cells. The belt passes the detector at constant speed, which gives a clean correspondence:

> **Cell index = the moment that cell passes the detector.**

The belt is revealed cell by cell from left to right, one cell per second. As a result, the three original physical quantities — detector position, chute position, and drop delay — collapse into **one** parameter:

$$\ell = \text{the number of cells, among those the drop will cover, that you have already seen at the moment the drop command is issued}$$

$\ell$ is the field of view, measured in cells. It is tunable: moving the detector upstream, or using a wider sensing window, both increase $\ell$. The machine at the airport probably operated at $\ell = 1$ — it confirmed only the
first cell.

The belt is composed of $C$ cells, which is its circumference.

**Rule.** A bag arrives at the chute at time $r$ with width $w$, and must be dropped within $[r, r+T]$. At each moment you choose "drop" or "wait." If dropped at time $t$, it occupies cells $t, t+1, \dots, t+w-1$. If any of those cells is already occupied, that is a collision.

There are only five parameters: field of view $\ell$, width $w \sim \mathcal{W}$, deadline $T$, circumference $C$.

- The chute does not move, so you cannot choose *where* to place a bag; you can only choose *when* to drop, and the position follows.
This reduces a bin-packing problem to a stopping problem with deadlines.

## The trivial case, discarded

A very trivial strategy is to "drop whenever you see an empty cell", when<br>

1. $\ell \ge w$, <br>
2. with $w$ known, <br>
3. $T = \infty$, <br>
4. and only one bag.<br>



## An Interesting Case


### What is observed

Let $b_t(n) \in \{0,1\}$ be the occupancy of cell $n$ at time $t$. Define the **true free run** ahead of the drop position,

$$A_t \;=\; \min\{\,k \ge 0 \;:\; b_t(t+k) = 1\,\},$$

the number of consecutive empty cells starting at the cell about to be written.
The detector has a finite window, so what you actually see is the run
**right-censored at $\ell$**:

$$G_t \;=\; \min(A_t,\, \ell) \;\in\; \{0,1,\dots,\ell\}.$$

This is the crux. When $G_t < \ell$ you know the free run exactly. When
$G_t = \ell$ you know only that $A_t \ge \ell$, and nothing more. The observation
is informative about small gaps and mute about large ones.

### What is unknown

The width $W \sim \mathcal{W}$ on $\mathbb{Z}_{>0}$ is **not revealed until the bag
lands**. The distribution $\mathcal{W}$ is known; the realisation is not.


### The decision

A bag arrives at time $r$. Let $\mathcal{F}_t = \sigma(G_r, \dots, G_t)$ be the
information available at time $t$. A policy is a stopping time

$$\tau \;\in\; \{r,\, r+1,\, \dots,\, r+T\}, \qquad \tau \text{ adapted to } (\mathcal{F}_t).$$

The deadline is hard: if the bag has not been dropped by $r+T$ it is dropped then
regardless, i.e. $\tau = r+T$ on the event that no earlier drop occurred. There is
no option to abstain.

### The objective

A collision occurs iff the bag is wider than the true free run at the chosen moment:

$$\text{collision} \;=\; \{\,W > A_\tau\,\}.$$

The problem is

$$\min_{\tau} \;\; \mathbb{P}\big(W > A_\tau\big).$$

Note the asymmetry: **the collision event is governed by $A_\tau$, but the policy may only depend on the censored $G_\tau$**. You are optimizing against a quantity you can only partially observe.

### Bellman recursion

Write $k = r + T - t$ for the remaining slack and let $V_k(g)$ be the optimal collision probability given $k$ steps left and current observation $g$. The
immediate cost of stopping is

$$c(g) \;=\; \mathbb{P}\big(W > A \,\big|\, G = g\big)
\;=\;
\begin{cases}
\mathbb{P}(W > g), & g < \ell,\\[2pt]
\mathbb{P}\big(W > A \,\big|\, A \ge \ell\big), & g = \ell,
\end{cases}$$

and the recursion is

$$V_0(g) = c(g), \qquad
V_k(g) = \min\Big\{\, c(g), \;\; \mathbb{E}\big[V_{k-1}(G_{t+1}) \,\big|\, G_t = g\big] \Big\}.$$

The boundary condition $V_0(g) = c(g)$ is the hard deadline: the bag must be dropped immediately now! 





<!-- ### A minimal instance

To make every quantity explicit, take the belt in stationarity with each cell independently occupied with probability $p$. Then $A \sim \mathrm{Geom}(p)$ and

$$\mathbb{P}(G = g) = (1-p)^g p \ \ (g < \ell), \qquad \mathbb{P}(G = \ell) = (1-p)^\ell.$$

Take $W \sim \mathrm{Geom}(q)$ on $\{1,2,\dots\}$. Both $c(g)$ and the transition law of $G_t$ are then closed-form, and the recursion above can be iterated by hand.
This is the smallest instance in which all three effects — censoring, unknown
width, and the deadline — are simultaneously present. -->

<!-- ### Two questions

1. **Threshold structure.** Is the optimal policy of the form "drop iff
   $G_t \ge \theta_k$", with $\theta_k$ non-increasing as $k \to 0$? Intuitively you
   should grow less picky as the deadline approaches. Proving monotonicity of
   $\theta_k$ in $k$ would be the first real result.

2. **What one more cell of view is worth.** How does the optimal value
   $V_T^\ast(\ell)$ behave in $\ell$? I would guess it saturates once $\ell$ exceeds
   the bulk of $\mathcal{W}$ — past that point the extra cell is almost always
   censored away and buys nothing. Whether the saturation point sits at
   $\mathbb{E}[W]$, at a quantile of $\mathcal{W}$, or somewhere else is a clean and
   falsifiable question. -->

## A small numerical experiment

A small numerical test suggests a simple rule. Wait for a larger empty space when there is enough time. As the deadline gets closer, accept a smaller space. In other words, the best rule is patient at first and less picky at the end.
