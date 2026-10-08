---
title: "Floating Point Notes in Computers"
date: "2026-07-11"
description: "Notes on how computers represent floating-point numbers and why rounding errors are unavoidable."
category: "tech"
tags: ["mathematics", "computer"]
math: true
---
Let's build the concept of floating-point numbers in computers from first principles.


<!-- We will follow a logical progression: from the physical limitations of computers, to the mathematical decisions, and finally to the engineering trade-offs that shaped the IEEE 754 standard, ultimately examining how modern CPUs execute common operations. -->


## 1. The Core Conflict: The Finite Cannot Represent the Infinite


The real number line is perfectly continuous and stretches to infinity. However, computers use fixed-size memory registers (e.g., 32 bits or 64 bits). A 32-bit sequence can only ever represent $$2^{32}$$ distinct states.


**First Principle:** You cannot map an infinite, continuous space into a finite, discrete space without losing information. We must accept that computer mathematics will inherently involve **approximations and rounding errors**.


## 2. Binary and the Need for Exponents


Computers operate in binary. Just as we use base-10 decimals, we can use binary fractions (e.g., $$1.01_2 = 1.25_{10}$$).


If we used a fixed-point system, we couldn't represent both cosmic scales ($$10^{20}$$) and quantum scales ($$10^{-10}$$) simultaneously. To cover both within just 32 bits, we borrow from scientific notation: $$a \times 10^b$$. In binary, this becomes **$$F \times 2^E$$**. The radix point "floats" depending on the exponent $$E$$.


## 3. Floating-Point Memory Layout (Bit Distribution)


In the IEEE 754 standard, a floating-point number is stored in three parts: **Sign (S)**, **Exponent (E)**, and **Fraction/Mantissa (F)**.


- **Single Precision (float)**: `1` sign bit + `8` exponent bits + `23` fraction bits = `32` bits
- **Double Precision (double)**: `1` sign bit + `11` exponent bits + `52` fraction bits = `64` bits


**Single Precision (32-bit) Memory Layout Diagram:**

```text
 31  30           23  22                                      0
 +--+---------------+-----------------------------------------+
 | S|    Exponent   |                Fraction                 |
 +--+---------------+-----------------------------------------+
 1-bit   8-bit                         23-bit
```


**Double Precision (64-bit) Memory Layout Diagram:**

```text
 63  62                     52  51                            0
 +--+-------------------------+-------------------------------+
 | S|        Exponent         |           Fraction            |
 +--+-------------------------+-------------------------------+
 1-bit        11-bit                         52-bit
```


## 4. Engineering Trade-offs: Implicit Bit, Gradual Underflow, and Biased Exponent


When designing the standard, engineers made several major trade-offs to optimize for efficiency, speed, and mathematical continuity:


### Trade-off 1: The Implicit Leading Bit (Efficiency)

In binary scientific notation, a normalized number always starts with `1` (e.g., $$1.011_2 \times 2^3$$). Since the digit before the radix point is *always* `1`, we don't store it; we simply assume its presence. This provides **one free bit of precision**.


**BUT**, there is a problem with the implicit `1`. If every number starts with `1.xxx`, how do we represent numbers very close to 0?

If the smallest exponent is $$-126$$, the smallest *normalized* positive number is $$1.0 \times 2^{-126}$$. Below this, the value would abruptly jump straight to 0. This huge gap breaks the mathematical property that $$x - y = 0$$ if and only if $$x = y$$.


**The Solution: Denormalization (Gradual Underflow)**

To fix this, the standard introduces a rule: if the Exponent field is all 0s, the implicit leading bit changes from `1.` to `0.`.

- These are called **Denormalized (or Subnormal) numbers**.
- Denormalized numbers do not have higher precision. They actually have **lower precision**, but they **can represent numbers much closer to 0**.
- As the value gets smaller, the number of leading zeros increases, and the significant bits drop from 24 down to 1. This **gradual underflow** bridges the gap to 0 and prevents abrupt jumps.


### Trade-off 2: Biased Exponent (Fast Comparison)

To rapidly compare two floating-point numbers, we want to feed their raw 32-bit patterns directly into standard **integer comparison circuits**.


If we used two's complement for negative exponents, a negative exponent (like `11111111`) would appear as a massive unsigned integer, breaking our fast integer comparison trick.


**Solution:** We use a **bias (offset)**. By adding a constant (e.g., 127 for single precision) to the actual exponent, all stored exponents become strictly positive. This preserves the integer sorting order!

- Single precision: $$\text{Stored } E = \text{True exponent} + 127$$
- To decode: $$\text{True exponent} = \text{Stored } E - 127$$
- Double precision: bias is `1023`.


## 5. Read the Bits Directly: Extreme Values in 32-bit Single Precision


This is the place to stop talking in the abstract and read real `0`/`1` patterns. Single precision rule recap:

- Layout: `S` (1 bit) \| `E` (8 bits) \| `F` (23 bits), Bias = `127`.
- If `E != 00000000` and `E != 11111111`: normalized, value = $$(-1)^S \times 1.F_2 \times 2^{E_{unsigned}-127}$$.
- If `E == 00000000`: denormalized / zero, value = $$(-1)^S \times 0.F_2 \times 2^{-126}$$. Note the exponent is fixed at $$-126$$ here, not $$0-127=-127$$.


Overview of the six extremes:

| # | Name | 32-bit pattern `S EEEEEEEE F...` | Hex | Approx. value |
|---|------|----------------------------------|-----|---------------|
| A | Largest finite positive | `0 11111110 11111111111111111111111` | `0x7F7FFFFF` | $$\approx +3.40 \times 10^{38}$$ |
| B | Smallest positive normalized | `0 00000001 00000000000000000000000` | `0x00800000` | $$\approx +1.18 \times 10^{-38}$$ |
| C | Smallest positive denormalized | `0 00000000 00000000000000000000001` | `0x00000001` | $$\approx +1.40 \times 10^{-45}$$ |
| D | Most negative (largest magnitude) | `1 11111110 11111111111111111111111` | `0xFF7FFFFF` | $$\approx -3.40 \times 10^{38}$$ |
| E | Negative closest to zero, normalized | `1 00000001 00000000000000000000000` | `0x80800000` | $$\approx -1.18 \times 10^{-38}$$ |
| F | Negative closest to zero, denormalized | `1 00000000 00000000000000000000001` | `0x80000001` | $$\approx -1.40 \times 10^{-45}$$ |


<div class="fp-example" markdown="1">

**Case A — Largest finite positive: `0 11111110 11111111111111111111111`**

```text
S = 0  (+)
E = 11111110
F = 11111111111111111111111  (23 ones)
Hex = 0x7F7FFFFF
```

- Exponent bits to unsigned: $$11111110_2 = 128+64+32+16+8+4+2+0 = 254_{10}$$.
- Minus bias: $$E_{true} = 254 - 127 = +127$$.
- Mantissa: $$1.F_2 = 1.111\dots111_2 = 2 - 2^{-23} \approx 1.99999988$$.
- Value: $$(2-2^{-23}) \times 2^{127} = 2^{128} - 2^{104} \approx 3.4028235 \times 10^{38}$$.

Why not `E = 11111111`? That pattern is reserved for `Inf` / `NaN`, so `11111110` is the largest finite exponent.

</div>


<div class="fp-example" markdown="1">

**Case B — Smallest positive normalized: `0 00000001 00000000000000000000000`**

```text
S = 0  (+)
E = 00000001
F = 00000000000000000000000  (23 zeros)
Hex = 0x00800000
```

- Exponent bits to unsigned: $$00000001_2 = 1_{10}$$.
- Minus bias: $$E_{true} = 1 - 127 = -126$$.
- Mantissa: $$1.000\dots000_2 = 1.0$$.
- Value: $$1.0 \times 2^{-126} \approx 1.17549435 \times 10^{-38}$$.

Anything smaller in magnitude cannot stay normalized; it must become denormalized.

</div>


<div class="fp-example" markdown="1">

**Case C — Smallest positive denormalized: `0 00000000 00000000000000000000001`**

```text
S = 0  (+)
E = 00000000  (special: denormalized)
F = 00000000000000000000001  (only LSB = 1)
Hex = 0x00000001
```

- Exponent field is all zeros, so the rule changes: leading bit is `0.`, true exponent is fixed at $$-126$$ (do **not** compute $$0-127=-127$$).
- Mantissa: $$0.F_2 = 0.000\dots001_2 = 2^{-23}$$.
- Value: $$2^{-23} \times 2^{-126} = 2^{-149} \approx 1.40129846 \times 10^{-45}$$.

This is the closest positive number to zero. One step smaller is exactly `+0`.

</div>


<div class="fp-example" markdown="1">

**Case D — Most negative (mirror of Case A): `1 11111110 11111111111111111111111`**

```text
S = 1  (-)
E = 11111110
F = 11111111111111111111111
Hex = 0xFF7FFFFF
```

- Exponent: $$11111110_2 = 254_{10}$$, $$254-127 = +127$$, identical to Case A.
- Value: $$-(2-2^{-23}) \times 2^{127} \approx -3.4028235 \times 10^{38}$$.

Only the sign bit flips; exponent and fraction decoding are unchanged.

</div>


<div class="fp-example" markdown="1">

**Case E — Negative closest to zero, normalized: `1 00000001 00000000000000000000000`**

```text
S = 1  (-)
E = 00000001
F = 00000000000000000000000
Hex = 0x80800000
```

- Exponent: $$00000001_2 = 1_{10}$$, $$1-127 = -126$$.
- Value: $$-1.0 \times 2^{-126} \approx -1.17549435 \times 10^{-38}$$.

</div>


<div class="fp-example" markdown="1">

**Case F — Negative closest to zero, denormalized: `1 00000000 00000000000000000000001`**

```text
S = 1  (-)
E = 00000000  (special: denormalized)
F = 00000000000000000000001
Hex = 0x80000001
```

- Same denormalized rule: $$0.F_2 \times 2^{-126} = 2^{-23} \times 2^{-126} = 2^{-149}$$.
- Value: $$\approx -1.40129846 \times 10^{-45}$$.

</div>


Reference for intuition: the largest denormalized number `0 00000000 11111111111111111111111` is $$(1-2^{-23}) \times 2^{-126} \approx 1.17549421 \times 10^{-38}$$, just one step below the smallest normalized number in Case B. That is gradual underflow in action.


## 6. Special Numbers: 0, Inf, NaN


Because our system is finite, operations can overflow or yield mathematically undefined results. To preserve our fast integer sorting trick, where should infinity live? At the very top (the largest possible exponent).


- **Exponent = All 1s**:
  - If **Fraction == 0**: **Infinity** (`+Inf` or `-Inf`).
  - If **Fraction != 0**: **NaN** (Not a Number, e.g., for $$\sqrt{-1}$$).
- **Exponent = All 0s**: Reserved for **Zero** (if the fraction is 0) and **Subnormals** (if the fraction is non-zero, see Cases C/F above).


## 7. Key Examples to Test Your Understanding


<div class="fp-example" markdown="1">

**Example 1: What is the smallest single-precision number strictly greater than 2?**

- To find this, we examine how $$2.0$$ is represented.
- $$2.0 = 1.0 \times 2^1$$.
- In Single Precision: Sign = `0`, Exponent = $$1 + 127 = 128$$, Fraction = $$000\dots000$$ (23 zeros).
- The next representable number simply increments the fraction by the smallest possible amount (setting the least significant bit, LSB, to 1).
- The Fraction becomes $$000\dots001$$.
- The value is $$1.000\dots001_2 \times 2^1 = (1 + 2^{-23}) \times 2 = 2 + 2^{-22}$$.
- Therefore, the smallest number greater than 2 is **$$2 + 2^{-22}$$** (approximately $$2 + 2.38 \times 10^{-7}$$).

</div>


<div class="fp-example" markdown="1">

**Example 2: What is the largest exact odd integer that can be represented?**

- To represent an exact integer, the fractional part must completely cover the integer bits without any fractional remainder.
- Single precision has 24 bits of precision (1 implicit + 23 explicit).
- The largest integer where the spacing between representable numbers is exactly 1 occurs when the exponent pushes the radix point perfectly to the end of the 23-bit fraction.
- This happens when $$E - 127 = 23 \implies \text{Exponent} = 150$$.
- The form is $$1.\text{Fraction} \times 2^{23}$$.
- The maximum value of this form is $$1.111\dots111_2 \times 2^{23} = 2^{24} - 1 = 16,777,215$$.
- What if we go larger? At $$E - 127 = 24$$, the value is $$1.F \times 2^{24}$$. The least significant bit now represents $$2^{-23} \times 2^{24} = 2^1 = 2$$. This means numbers in this range can only represent **EVEN** integers (the spacing is 2).
- So, the largest odd integer in single precision is **$$2^{24} - 1 = 16,777,215$$**. (For double precision, it is $$2^{53} - 1 = 9,007,199,254,740,991$$).

</div>


## 8. How Does the Computer Actually Calculate?


The underlying philosophy for calculations is to **reduce complex operations to integer math and polynomial approximations.**


### Addition (`+`) and Subtraction (`-`)

1. **Alignment:** The CPU right-shifts the fraction of the number with the smaller exponent until both exponents match.
2. **Integer Math:** The fractions are fed into the **exact same two's complement adder circuit** used for integers.
3. **Normalization:** The CPU shifts the fraction back into the $$1.\text{xxx}$$ format and adjusts the exponent accordingly.


*Warning:* Subtracting two very close numbers causes **catastrophic cancellation**, where significant bits cancel out, instantly destroying precision.


### Multiplication (`*`) and Division (`/`)

- **Multiply:** Multiply the fractions (using an integer multiplier) and **add** the exponents (subtracting one bias, since both exponents inherently included a bias).
- **Divide:** Divide the fractions and **subtract** the exponents (adding back one bias).


### Exponentiation ($$e^x$$) and Logarithm ($$\ln x$$)

The floating-point format ($$1.F \times 2^{E-\text{Bias}}$$) is *already* an exponential function!

- **Exponent ($$e^x$$):** Using the identity $$e^x = 2^{x / \ln 2}$$, the integer part of the result is **directly added to the exponent ($$E$$) field** (fast integer addition!). The tiny fractional part is calculated using a minimax polynomial.
- **Logarithm ($$\ln x$$):** Since $$\ln(x) = \ln(1.F) + (E-\text{Bias})\ln(2)$$, the $$(E-\text{Bias})$$ term is extracted via simple integer arithmetic. The CPU only needs to approximate $$\ln(1.F)$$ using polynomials.


### Power ($$x^y$$)

This is calculated as $$x^y = e^{y \cdot \ln(x)}$$ by calling the highly optimized hardware or software `log` and `exp` functions.


### Trigonometric Functions ($$\sin$$, $$\cos$$)

1. **Range Reduction:** The input is reduced (e.g., $$x \bmod (\pi/2)$$). This requires an incredibly precise stored value of $$\pi$$ to avoid massive errors for large inputs (e.g., using the Payne-Hanek algorithm).
2. **Polynomial Approximation:** The reduced input is evaluated using a minimax polynomial (like a Chebyshev series) via hardware **FMA (Fused Multiply-Add)** instructions, which calculate $$(A \times B) + C$$ in a single cycle without intermediate rounding errors.


## 9. Epilogue: Two's Complement and Hexadecimal


To fully appreciate floating-point design, we must contrast it with how standard integers are stored.


### Why do Integers use Two's Complement?

While floating-point exponents use a **Bias** to preserve their raw sorting order, standard signed integers use **Two's Complement**. Why?

- **Unified Adder/Subtractor:** The biggest advantage is that $$A - B$$ literally becomes $$A + (\text{Two's Complement of } B)$$. The CPU only needs to build a single fast addition circuit (ALU) to handle both addition and subtraction.
- **What about Multipliers and Dividers?** Do they share circuits too? Yes! Integer multipliers use arrays of adders (partial products). Two's complement allows the exact same core multiplier hardware to be used for signed numbers with only minor sign-extension tweaks (like Booth's multiplication algorithm). Dividers (like non-restoring division) also seamlessly leverage the same add/subtract ALU.
- **Single Zero:** Unlike "Sign-and-Magnitude" encoding, Two's complement has only one representation for $$0$$ (no $$+0$$ and $$-0$$), preventing bugs in software `if (x == 0)` checks.


### Why do we always read memory in Hexadecimal (Hex)?

You will often see 32-bit floats or integers printed as `0x40490FDB` instead of a massive 32-digit binary string.

- **Perfect mapping:** Binary is too long for humans to read, and converting back and forth between Base-10 and Base-2 is computationally annoying. But Base-16 (Hexadecimal) is the perfect middle ground because $$16 = 2^4$$.
- Exactly **4 binary bits map directly to exactly 1 Hex character** ($$0000_2 = \text{0}_{16}$$ through $$1111_2 = \text{F}_{16}$$).
- We can instantly break an 8-bit byte into two 4-bit "nibbles", represented cleanly by 2 Hex digits. Hex is not a new number format; it is simply a highly readable, compact way for humans to look at pure binary!
