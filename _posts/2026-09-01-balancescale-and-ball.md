---
title: "A Balance Scale with Balls"
date: "2026-09-01"
description: "A classic puzzle about finding one abnormal ball with a balance scale."
category: "musings"
tags: ["mathematics", "puzzle"]
math: true
---
Given 12 balls, with one ball heavier or lighter than the other 11, how many comparisons on a balance scale do you need to find the abnormal one?<br>

4 is undoubtedly enough, but 2 is of course too few. 3 seems feasible. Here is a strategy:<br>

<!-- Number the balls 1, 2, ..., 12. Even if we only want to find the abnormal ball, there are 12 possibilities. Each weighing has three possible results, so two weighings can distinguish at most $3^2=9$ possibilities. Therefore, at least three weighings are needed. -->

We now show that three are enough. First, compare {1,2,3,4} with {5,6,7,8}.

### Case 1: They balance

Then balls 1 through 8 are standard, and the abnormal ball is in {9,10,11,12}. Compare {9,10,11} with three standard balls.

- If they balance, ball 12 is abnormal. Compare it with a standard ball to tell whether it is heavier or lighter.
- If they do not balance, one of {9,10,11} is abnormal, and the direction tells us whether it is heavier or lighter. Compare 9 with 10. If they balance, it is 11; otherwise, the ball in the known direction is abnormal.

### Case 2: They do not balance

Without loss of generality, suppose {1,2,3,4} is heavier. Then one of {1,2,3,4} is heavy, or one of {5,6,7,8} is light. Balls 9 through 12 are standard.

Compare {1,5,6} with {2,7,8}.

- If the left side is heavier, the possibilities are: 1 is heavy, or 7 or 8 is light. Compare 7 with 8. If they balance, it is 1; otherwise, the lighter ball is abnormal.
- If the right side is heavier, the possibilities are: 2 is heavy, or 5 or 6 is light. Compare 5 with 6. If they balance, it is 2; otherwise, the lighter ball is abnormal.
- If they balance, either 3 or 4 is heavy. Compare them; the heavier one is abnormal.

If the right side was heavier in the first weighing, exchange the names of the two groups and use the same strategy. Thus three weighings always identify the abnormal ball and also tell whether it is heavier or lighter.


It is quite complicated, but we can get some insight from it:

1. When we get into Case 1, at first glance, we don't have any useful information about {9,10,11,12}. But actually, we do. We have the standard balls (1,2,3,4,5,6,7,8), which can show us the standard weight.


2. In Case 2, we have 8 balls and only 2 chances to use the balance scale, but we magically work it out! Actually, we can deal with at most 4 balls with only 2 opportunities. So how do we double that here? The fact that the scale is unbalanced seems trivial, but what is non-trivial is a "rank": which side is heavier and which side is lighter. And it truly brings something.


So now I have many questions lingering in my mind: what about 4 chances, or even k chances? Is there a general strategy? How can we describe or represent information like "{1,2,3,4}>{5,6,7,8}"? And, actually, there are 2 kinds of results: figure out the abnormal ball; or figure out the outlier and tell whether it is heavier or lighter.<br>

The problem is very intriguing and requires lots of logical reasoning. It is cerebral and can hone our reasoning skills. I think I should have started thinking about this problem when military training began.
