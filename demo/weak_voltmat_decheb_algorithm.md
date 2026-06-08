# `Weak_VoltMat_DE` 算法说明

本文档总结 `demo/weak_voltmat_decheb.jl` 中当前实现的弱奇异 Volterra 系数矩阵构造算法。该实现仍位于 `demo/` 层级，尚未并入 `src/` 主模块。

## 1. Overview

`Weak_VoltMat_DE` 模块构造 Liu--Xu 论文双指数变换下的弱奇异 Volterra 算子的 Chebyshev 系数矩阵。导出的核心函数为：

```julia
weak_voltmat_decheb
```

目标算子为

\[
(V_K u)(\phi(y))
=
\int_{-1}^{y}
(\phi(y)-\phi(\tau))^{\mu-1}
K(\phi(y),\phi(\tau))
u(\tau)
\phi'(\tau)\,\mathrm d\tau .
\]

返回的矩阵 \(A\) 是关于变量 \(y\) 的 \(N\times N\) Chebyshev 系数矩阵。若输入 \(u\) 的 Chebyshev 系数向量为 \(\hat u\)，则 \(A\hat u\) 近似给出 \(V_Ku\) 在 \(y\) 变量中的 Chebyshev 系数。

需要强调：返回的 \(A\) 是最终矩阵，已经包含外部因子 \((1+y)/2\)，不是中间矩阵 \(G\)。

## 2. Double-Exponential Transform

当前实现使用 Liu--Xu 论文中的双指数变换：

\[
\phi(y)=\tanh\!\left(\frac{\pi}{2}\sinh(\omega y)\right),
\]

\[
\phi'(y)
=
\frac{\pi\omega}{2}\cosh(\omega y)
\operatorname{sech}^2\!\left(\frac{\pi}{2}\sinh(\omega y)\right).
\]

端点约定为

\[
\phi(-1)=-1,\qquad \phi(1)=1.
\]

这个约定符合论文中选择 \(\omega\) 使得 \(\phi(\pm1)\) 在机器精度下近似 \(\pm1\) 的做法。

稳定计算是必要的。若直接计算

\[
\phi(y)-\phi(\tau),
\]

当 \(\tau\approx y\) 或 \(\phi\) 在端点附近饱和时，会产生严重的相消误差，甚至把应当为正的小量误判为零。因此 DE 路径中不会用直接相减来构造奇异商。

## 3. Duffy Transform and Transformed Kernel

令

\[
\tau(y,\eta)
=
\frac{y+1}{2}\eta+\frac{y-1}{2}.
\]

对基函数 \(T_k\)，Duffy 变换后有

\[
V_k(y)
=
\frac{1+y}{2}
\int_{-1}^{1}
(1-\eta)^{\mu-1}
\widetilde K_\phi(y,\eta)
T_k(\tau(y,\eta))\,\mathrm d\eta .
\]

其中

\[
\widetilde K_\phi(y,\eta)
=
\phi'(\tau)
\left(
\frac{\phi(y)-\phi(\tau)}{1-\eta}
\right)^{\mu-1}
K(\phi(y),\phi(\tau)).
\]

在 DE 变换路径中，\(\widetilde K_\phi\) 不通过直接计算 \(\phi(y)-\phi(\tau)\) 得到，而是使用稳定的对数公式。

## 4. Stable DE Evaluation of \(\widetilde K_\phi\)

实现中使用如下变量：

\[
t=\frac{1-\eta}{2},\qquad
\tau=y-(1+y)t.
\]

当 \(t\) 很小时，使用极限商：

\[
\log\_ratio
=
\log\left(\frac{1+y}{2}\right)+\log \phi'(y).
\]

当 \(t>0\) 时，定义

\[
A_y=\frac{\pi}{2}\sinh(\omega y),
\qquad
A_\tau=\frac{\pi}{2}\sinh(\omega \tau),
\]

\[
\xi_2=\frac{\omega(1+y)t}{2},
\]

\[
\xi_3
=
\pi\cosh\!\left(\frac{\omega(y+\tau)}{2}\right)\sinh(\xi_2).
\]

然后计算

\[
\begin{aligned}
\log\_\Delta/t
=\;&
\log\left(\frac{\pi\omega(1+y)}{2}\right)
+\log\cosh\!\left(\frac{\omega(y+\tau)}{2}\right)
+\log\sinhc(\xi_3)
+\log\sinhc(\xi_2) \\
&-\log\cosh(A_y)
-\log\cosh(A_\tau).
\end{aligned}
\]

由于

\[
\frac{\phi(y)-\phi(\tau)}{1-\eta}
=
\frac{\phi(y)-\phi(\tau)}{2t},
\]

所以

\[
\log\_ratio
=
\log\_\Delta/t-\log 2.
\]

同时

\[
\log \phi'(\tau)
=
\log\left(\frac{\pi\omega}{2}\right)
+\log\cosh(\omega\tau)
-2\log\cosh(A_\tau).
\]

最终

\[
\widetilde K_\phi(y,\eta)
=
\exp\!\left(
\log \phi'(\tau)+(\mu-1)\log\_ratio
\right)
K(\phi(y),\phi(\tau)).
\]

实现中使用的稳定辅助函数包括：

- `logcosh`
- `log1pexp`
- `sinhc`
- `log_sinhc`
- `log1p_phi`
- `log1m_phi`

这些函数用于避免端点饱和、指数溢出和相消。

## 5. Bivariate Chebyshev Expansion of \(\widetilde K_\phi\)

代码将平滑核因子近似为二维 Chebyshev 展开：

\[
\widetilde K_\phi(y,\eta)
\approx
\sum_{i=0}^{m}\sum_{j=0}^{n}
C_{ij}T_i(y)T_j(\eta).
\]

默认使用固定大小的系数矩阵：

\[
Csize_y\times Csize_\eta,
\]

当前常用默认值为 \(100\times100\)。采样使用第一类 Chebyshev 内点，因此不会在 \(\eta=1\) 这样的奇异端点直接采样。

数值诊断显示，将 \(C\) 从 \(65\times65\) 增大到约 \(100\times100\) 后，旧的 \(10^{-10}\) 量级核截断平台被消除；在当前测试中，\(100\times100\) 是 \(\omega=4,\mu=0.5\) 下较好的默认选择。

## 6. Definition of \(H\) Moments

定义核心矩：

\[
H_{j,r,k}
=
\int_{-1}^{1}
(1-\eta)^{\mu-1}
T_j(\eta)
\left(\frac{1+\eta}{2}\right)^r
P_{k-r}^{(-1,2r)}(\eta)
\,\mathrm d\eta,
\qquad 0\le r\le k.
\]

这些矩只依赖于 \(\mu\)，与核函数 \(K\) 和变换 \(\phi\) 无关。

## 7. Stable Computation of \(H_{0,0,k}\)

初始行

\[
H_{0,0,k}
=
\int_{-1}^{1}
(1-\eta)^{\mu-1}P_k^{(-1,0)}(\eta)\,\mathrm d\eta
\]

由已验证的稳定 DE 初始矩例程计算。

该例程避免在弱奇异端点直接评价被积函数，使用自动尾部截断和稳定求和。它不是 Liu--Xu 纸面变换 \(\phi\)；它是用于计算初始矩 \(H_{0,0,k}\) 的稳定双指数型积分例程。独立 Float64 高阶参考积分验证表明，该初始矩误差约为 \(10^{-14}\)。

## 8. \(r=0\) Layer Recurrence

令

\[
h_{j,k}=H_{j,0,k}.
\]

已知

\[
h_{0,k}=H_{0,0,k}.
\]

对 \(k\ge1\)，定义

\[
a_k=\frac{k+1}{2k+1},
\qquad
b_k=-\frac{1}{(2k-1)(2k+1)},
\qquad
c_k=\frac{k-1}{2k-1}.
\]

初始化：

\[
h_{1,0}=h_{0,0}+2h_{0,1},
\]

\[
h_{1,k}
=
a_k h_{0,k+1}
+b_k h_{0,k}
+c_k h_{0,k-1}.
\]

对 \(j\ge1\)，递推：

\[
h_{j+1,0}
=
2(h_{j,0}+2h_{j,1})-h_{j-1,0},
\]

\[
h_{j+1,k}
=
2(a_kh_{j,k+1}+b_kh_{j,k}+c_kh_{j,k-1})
-h_{j-1,k}.
\]

由于递推中需要 \(h_{j,k+1}\)，实现中在 \(k\) 方向保留 guard columns。

## 9. Fixed-\(j\) \((r,k)\) Recurrence

对每一个固定的 \(j\)，代码使用已验证的 fixed-\(j\) \((r,k)\) 递推构造所有

\[
0\le r\le k
\]

范围内的 \(H_{j,r,k}\)。

该递推的显式公式较长，详见 `demo/weak_voltmat_decheb.jl` 中的实现。关键点是：内部 \(r>0\) 区域不是通过完整三维张量沿 \(j\) 做前向递推得到的，而是对每个固定 \(j\) 独立构造。这避免了此前不稳定的三维 \(H[j,r,k]\) 前向递推。

## 10. Streaming Compression over \(j\)

定义压缩后的矩：

\[
Hcomp^{(i)}_{r,k}
=
\sum_j C_{ij}H_{j,r,k}.
\]

实现中一次只生成一个固定 \(j\) 的 \(H_{j,r,k}\) 层，然后立即累加到 \(Hcomp\)：

\[
Hcomp^{(i)}_{r,k}
\leftarrow
Hcomp^{(i)}_{r,k}+C_{ij}H_{j,r,k}.
\]

因此代码不显式存储完整的三维 \(H[j,r,k]\) 张量。

## 11. Assembly of \(G\) by the Restriction-Operator Formula

对 \(k=0\)，有

\[
G_{\ell,0}
=
Hcomp^{(\ell)}_{0,0},
\]

当 \(\ell\) 超出 \(C\) 的 \(i\) 方向范围时取零。

对 \(k>0\)，使用限制算子公式：

\[
\begin{aligned}
G_{\ell,k}
=
\frac{2-\delta_{\ell,0}}{2}
\sum_i
\Bigg[
&
\mathbf 1_{i+\ell\le k}
\frac{k}{k+i+\ell}
Hcomp^{(i)}_{i+\ell,k} \\
&+
\mathbf 1_{|i-\ell|\le k}
\frac{k}{k+|i-\ell|}
Hcomp^{(i)}_{|i-\ell|,k}
\Bigg].
\end{aligned}
\]

该公式来自 Chebyshev 乘积恒等式

\[
T_i(y)T_\ell(y)
=
\frac12\left(T_{i+\ell}(y)+T_{|i-\ell|}(y)\right)
\]

以及 Chebyshev 限制算子。实现中数学公式采用 0-based 索引，Julia 数组采用 1-based 索引。

## 12. Final Multiplication by \((1+y)/2\)

中间量满足

\[
V_k(y)=\frac{1+y}{2}g_k(y).
\]

因此最终矩阵为

\[
A=M\!\left[\frac{1+y}{2}\right]G.
\]

逐行公式为：

\[
A_{0,k}
=
\frac12G_{0,k}+\frac14G_{1,k},
\]

\[
A_{1,k}
=
\frac12G_{1,k}+\frac12G_{0,k}+\frac14G_{2,k},
\]

\[
A_{\ell,k}
=
\frac12G_{\ell,k}
+\frac14G_{\ell-1,k}
+\frac14G_{\ell+1,k},
\qquad \ell\ge2.
\]

因此 \(G\) 至少需要计算 \(N+1\) 行。最终返回的 \(A\) 是 \(N\times N\) 矩阵。

## 13. API

当前 API 为：

```julia
weak_voltmat_decheb(
    K;
    mu,
    N,
    omega=4.0,
    Csize_y=100,
    Csize_eta=100,
    tol=1e-14,
    return_info=false,
    output=:dense,
    kwargs...
)
```

其中 `K` 是二元函数 `K(x,t)`。

示例：

```julia
include("weak_voltmat_decheb.jl")
using .Weak_VoltMat_DE

A = weak_voltmat_decheb((x,t)->1.0; mu=0.5, N=96)

A, info = weak_voltmat_decheb(
    (x,t)->1.0;
    mu=0.5,
    N=96,
    return_info=true,
)
```

当 `return_info=true` 时，返回 `(A, info)`。`info` 中包含 `C_size`、`C`、`max_abs_C`、`last_row_max`、`last_col_max`、`tail_last_row_col_max`、`H00_finite`、`Hcomp_finite`、`G_finite`、`A_finite`、`max_abs_A`、`min_positive_abs_A`、`omega`、`mu`、`N`、`Csize_y`、`Csize_eta` 和 `stable_de_ktilde=true` 等诊断信息。

## 14. Diagnostics and Validation Status

当前验证结果可概括为：

- \(H_{0,0,k}\) 的误差约为 \(10^{-14}\)。
- \(r=0\) 层和内部 \(H\) 递推误差约为 \(10^{-16}\) 到 \(10^{-15}\)。
- 使用稳定 DE \(\widetilde K_\phi\) 和 \(C=100\times100\) 时，对多项式测试函数，在 \(N\ge96\) 时 action error 约为 \(5\times10^{-12}\)。
- 相对于有限 Chebyshev 核表示所定义的截断核算子，矩阵构造误差约为 \(10^{-14}\)。
- 当前剩余误差平台主要由精确稳定 \(\widetilde K_\phi\) 与其有限 Chebyshev 表示之间的差异以及参考积分误差主导，而不是 \(H\) 递推误差。

## 15. Current Demo Scripts

相关文件包括：

- `demo/weak_voltmat_decheb.jl`
- `demo/validate_weak_action_de.jl`
- `demo/diagnose_weak_de_error.jl`
- `demo/plot_logabs_spy_K1_mu0p5.jl`
- `demo/output/logabs_spy_K1_mu0p5_DEomega4_C100_N500.pdf`
- `demo/output/logabs_spy_K1_mu0p5_DEomega4_C100_N500.png`

## 16. Notes and Caveats

- 这是 demo-level 模块，尚未集成到 `src/`。
- 默认变换是 Liu--Xu 论文 DE 变换，不是 \(\tanh(\omega y)/\tanh(\omega)\)。
- 稳定的 \(\widetilde K_\phi\) 计算是必要的；直接计算 \(\phi(y)-\phi(\tau)\) 会导致灾难性相消。
- 更大的 `Csize_y` 和 `Csize_eta` 可以改善核近似，但会增加构造成本。
- 在当前 \(\omega=4,\mu=0.5\) 的测试中，`Csize_y=Csize_eta=100` 是一个较好的默认值。
