# First-difference regularization operator

I want a linear operator $D$ for smoothness regularization of a 1-D signal
$x \in \mathbb{R}^n$. It takes first differences:

$$ (D x)_i = x_{i+1} - x_i, \qquad i = 0, \dots, n-2, $$

so $D : \mathbb{R}^n \to \mathbb{R}^{n-1}$ — it is **not** square. As a matrix it is bidiagonal:

$$
D = \begin{pmatrix}
-1 & 1 & & & \\
 & -1 & 1 & & \\
 & & \ddots & \ddots & \\
 & & & -1 & 1
\end{pmatrix} \in \mathbb{R}^{(n-1)\times n}.
$$

For efficiency I want the transpose written out explicitly rather than left to autodiff. The
adjoint $D^\top : \mathbb{R}^{n-1} \to \mathbb{R}^n$ acts as

$$
(D^\top y)_j =
\begin{cases}
-y_0 & j = 0 \\
y_{j-1} - y_j & 1 \le j \le n-2 \\
y_{n-2} & j = n-1.
\end{cases}
$$
