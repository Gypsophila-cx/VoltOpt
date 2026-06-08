# K=1 时快速带状弱奇异 Volterra 积分算子系数矩阵构造算法

本文档总结在 Liu--Xu 双指数变换下，针对 $K=1$ 的弱奇异 Volterra 积分算子，如何构造数值上有意义的带状 Chebyshev 系数矩阵。算法的目标是：

- 固定 DE 平滑因子的二维 Chebyshev 系数矩阵大小为 $100\times100$；
- 只构造数值上有意义的带状部分；
- 主矩递推从完整 $(r,k)$ 区域改为窄带 $(j,k)$ 递推；
- 递推初值使用关于 $N$ 线性的稳定算法计算；
- 最终组装利用 Toeplitz--Hankel 型卷积结构，并用 FFT 加速；
- 当带宽与 $100\times100$ 系数规模固定时，总构造复杂度为 $O(N)$。

---

## 1. 算子与 DE--Duffy 变换

考虑 $K=1$ 时的弱奇异 Volterra 算子

\begin{align*}
(Vu)(\phi(y))
=
\int_{-1}^{y}
(\phi(y)-\phi(\tau))^{\mu-1}
u(\tau)\phi'(\tau)\,\mathrm d\tau,
\qquad 0<\mu<1.
\end{align*}

这里使用 Liu--Xu 型双指数变换

\begin{align*}
\phi(y)
=
\tanh\!\left(\frac{\pi}{2}\sinh(\omega y)\right),
\end{align*}

并约定端点在双精度下满足 $\phi(-1)\approx -1$、$\phi(1)\approx 1$。

对积分区间 $[-1,y]$ 使用 Duffy 变换

\begin{align*}
\tau(y,\eta)
=
\frac{y+1}{2}\eta+\frac{y-1}{2},
\qquad \eta\in[-1,1].
\end{align*}

则对输入基函数 $T_k(\tau)$，有

\begin{align*}
V_k(y)
=
\frac{1+y}{2}
\int_{-1}^{1}
(1-\eta)^{\mu-1}
\widetilde K_\phi(y,\eta)
T_k(\tau(y,\eta))\,\mathrm d\eta,
\end{align*}

其中虽然原始核 $K=1$，DE--Duffy 后的平滑因子仍为非平凡二维函数

\begin{align*}
\widetilde K_\phi(y,\eta)
=
\phi'(\tau)
\left(
\frac{\phi(y)-\phi(\tau)}{1-\eta}
\right)^{\mu-1}.
\end{align*}

因此，$K=1$ 并不意味着二维 Chebyshev 系数矩阵退化为 $1\times1$。在当前 DE 设置下，仍需使用约 $100\times100$ 的系数矩阵来逼近 $\widetilde K_\phi$。

---

## 2. 固定大小的二维 Chebyshev 系数矩阵

算法固定使用

\begin{align*}
\widetilde K_\phi(y,\eta)
\approx
\sum_{i=0}^{99}\sum_{j=0}^{99}
C_{ij}T_i(y)T_j(\eta).
\end{align*}

记

\begin{align*}
I=99,
\qquad
J=99.
\end{align*}

即 $C\in\mathbb R^{100\times100}$。该矩阵只与 $\mu$、$\omega$ 和 DE 变换有关，与矩阵截断维数 $N$ 无关。因此在复杂度分析中，$I$ 和 $J$ 均视为固定常数。

在计算 $C$ 时，必须使用稳定的 DE 对数公式评价 $\widetilde K_\phi$，避免直接计算 $\phi(y)-\phi(\tau)$。直接相减在 $\tau\approx y$ 或端点饱和区域会造成严重相消，从而破坏后续矩阵构造精度。

---

## 3. 有效带状区域

令最终矩阵为 $A\in\mathbb R^{N\times N}$，行指标为 $\ell$，列指标为 $k$，并定义对角偏移

\begin{align*}
d=k-\ell.
\end{align*}

由 $C$ 的 $y$ 方向最高阶 $I=99$ 可知，中间矩阵 $G$ 的严格下带宽为 $I$，即

\begin{align*}
d_G=k-\ell\ge -I.
\end{align*}

最终矩阵满足

\begin{align*}
A
=
M\!\left[\frac{1+y}{2}\right]G.
\end{align*}

由于乘法矩阵 $M[(1+y)/2]$ 在 Chebyshev 系数空间中为三对角矩阵，最终矩阵 $A$ 的下带宽增加一格，所以

\begin{align*}
d_A\ge -(I+1)=-100.
\end{align*}

另一方面，针对 $K=1$ 的理论与数值观察表明，矩阵在上方具有数值带宽。设目标上带宽为 $B$，即当

\begin{align*}
d_A=k-\ell>B
\end{align*}

时，$A_{\ell,k}$ 已小于给定机器精度阈值。于是只需保留

\begin{align*}
-100\le d_A\le B.
\end{align*}

为正确计算最终 $A$ 的上带宽 $B$，中间矩阵 $G$ 需要多计算一条上带，即保留

\begin{align*}
-I\le d_G\le B+1.
\end{align*}

最后乘以 $M[(1+y)/2]$ 后再截断到 $d_A\le B$。

---

## 4. 核心矩与窄带变量

定义矩

\begin{align*}
H_{j,r,k}
=
\int_{-1}^{1}
(1-\eta)^{\mu-1}
T_j(\eta)
\left(\frac{1+\eta}{2}\right)^r
P_{k-r}^{(-1,2r)}(\eta)
\,\mathrm d\eta,
\qquad 0\le r\le k.
\end{align*}

它只依赖于 $\mu$，不依赖于 $K$、$\phi$ 或 $C$。

为了只构造带状部分，令

\begin{align*}
s=k-r,
\qquad
\mathcal H_{j,s}^{(r)}=H_{j,r,r+s}.
\end{align*}

在装配公式中，$s$ 只需要取到一个固定上界。设中间矩阵 $G$ 的目标上带宽为

\begin{align*}
B_G=B+1.
\end{align*}

由后续卷积结构可知，需要的最大 $s$ 为

\begin{align*}
S=I+B_G=I+B+1.
\end{align*}

由于 $j$ 递推会使用 $s+1$，并且需要推进到 $j=J$，初始层要多保留 guard 项。取

\begin{align*}
S_{\mathrm{guard}}=S+J+1.
\end{align*}

在当前固定 $100\times100$ 系数矩阵下，

\begin{align*}
I=J=99,
\qquad
S=B+100,
\qquad
S_{\mathrm{guard}}=B+200.
\end{align*}

当 $B$ 由机器精度阈值固定后，$S$ 与 $S_{\mathrm{guard}}$ 都与 $N$ 无关。

---

## 5. 递推初值的 $O(N)$ 稳定计算

主递推需要对每个固定 $r$ 先得到

\begin{align*}
\mathcal H_{0,s}^{(r)}=H_{0,r,r+s},
\qquad
\mathcal H_{1,s}^{(r)}=H_{1,r,r+s},
\qquad
0\le s\le S_{\mathrm{guard}}.
\end{align*}

### 5.1 计算 $H_{0,r,r+s}$

作变量代换

\begin{align*}
x=\frac{1+\eta}{2},
\qquad
\eta=2x-1.
\end{align*}

则

\begin{align*}
H_{0,r,r+s}
=
2^\mu
\int_0^1
(1-x)^{\mu-1}
x^r
P_s^{(-1,2r)}(2x-1)
\,\mathrm dx.
\end{align*}

使用有限超几何表达式

\begin{align*}
H_{0,r,r+s}
=
2^\mu
(-1)^s
\frac{(2r+1)_s}{s!}
B(r+1,\mu)
\sum_{q=0}^{s}
\frac{(-s)_q(s+2r)_q(r+1)_q}
{(2r+1)_q(r+\mu+1)_q q!}.
\end{align*}

该和式因为 $(-s)_q$ 而终止，长度只有 $s+1$。实际计算时不直接调用 Gamma 函数或 Pochhammer 符号，而使用递推比例。

首先按 $r$ 更新 Beta 因子：

\begin{align*}
B_0=B(1,\mu)=\frac{1}{\mu},
\end{align*}

\begin{align*}
B_{r+1}=B(r+2,\mu)
=
\frac{r+1}{r+\mu+1}B_r.
\end{align*}

对固定 $r$ 和 $s$，定义

\begin{align*}
A_{r,s}=(-1)^s\frac{(2r+1)_s}{s!}.
\end{align*}

它满足

\begin{align*}
A_{r,0}=1,
\qquad
A_{r,s+1}
=
-A_{r,s}\frac{2r+1+s}{s+1}.
\end{align*}

有限和中的项满足

\begin{align*}
t_0=1,
\end{align*}

\begin{align*}
\frac{t_{q+1}}{t_q}
=
\frac{(-s+q)(s+2r+q)(r+1+q)}
{(2r+1+q)(r+\mu+1+q)(q+1)}.
\end{align*}

于是

\begin{align*}
H_{0,r,r+s}
=
2^\mu B_r A_{r,s}\sum_{q=0}^{s}t_q.
\end{align*}

对有限和建议使用 Neumaier 或 Kahan 补偿求和。因为 $S_{\mathrm{guard}}$ 固定，所以对所有 $r=0,1,\dots,N-1$ 生成初始线的总复杂度为

\begin{align*}
O(N S_{\mathrm{guard}}^2)=O(N).
\end{align*}

当 $\mu=1$ 时，上述公式退化为更简单的相邻比值递推。例如

\begin{align*}
H_{0,r,r}=\frac{2}{r+1},
\qquad
H_{0,r,r+1}=-\frac{2(2r+1)}{(r+1)(r+2)},
\end{align*}

并且当 $s\ge2$ 时满足

\begin{align*}
H_{0,r,r+s}
=
-
\frac{s+2r}{s+2r-1}
\frac{r+s-2}{r+s+1}
H_{0,r,r+s-1}.
\end{align*}

但在弱奇异情形 $0<\mu<1$ 下，应使用上面的带 $\mu$ 的稳定有限和或其等价稳定递推。

### 5.2 由 $H_0$ 得到 $H_1$

由于 $T_1(\eta)=\eta$，利用 Jacobi 多项式乘 $\eta$ 的三项关系

\begin{align*}
\eta P_s^{(-1,2r)}(\eta)
=
a_{r,s}P_{s+1}^{(-1,2r)}(\eta)
+
b_{r,s}P_s^{(-1,2r)}(\eta)
+
c_{r,s}P_{s-1}^{(-1,2r)}(\eta).
\end{align*}

当 $s\ge1$ 时，

\begin{align*}
a_{r,s}
=
\frac{2(s+1)(s+2r)}{(2s+2r)(2s+2r+1)},
\end{align*}

\begin{align*}
b_{r,s}
=
\frac{4r^2-1}{(2s+2r-1)(2s+2r+1)},
\end{align*}

\begin{align*}
c_{r,s}
=
\frac{2(s-1)(s+2r)}{(2s+2r-1)(2s+2r)}.
\end{align*}

于是

\begin{align*}
H_{1,r,r+s}
=
a_{r,s}H_{0,r,r+s+1}
+
b_{r,s}H_{0,r,r+s}
+
c_{r,s}H_{0,r,r+s-1}.
\end{align*}

边界 $s=0$ 单独处理。因为

\begin{align*}
\eta P_0^{(-1,2r)}(\eta)
=
P_0^{(-1,2r)}(\eta)
+
\frac{2}{2r+1}P_1^{(-1,2r)}(\eta),
\end{align*}

所以

\begin{align*}
H_{1,r,r}
=
H_{0,r,r}
+
\frac{2}{2r+1}H_{0,r,r+1}.
\end{align*}

---

## 6. 以 $(j,k)$ 为主的窄带递推

对固定 $r$，已经得到 $j=0$ 和 $j=1$ 两层

\begin{align*}
H_{0,r,r+s},
\qquad
H_{1,r,r+s}.
\end{align*}

由 Chebyshev 递推

\begin{align*}
T_{j+1}(\eta)=2\eta T_j(\eta)-T_{j-1}(\eta),
\end{align*}

可得对 $j\ge1$，

\begin{align*}
H_{j+1,r,r+s}
=
2\left(
a_{r,s}H_{j,r,r+s+1}
+
b_{r,s}H_{j,r,r+s}
+
c_{r,s}H_{j,r,r+s-1}
\right)
-
H_{j-1,r,r+s}.
\end{align*}

这里仍采用 $s=k-r$，所以该递推可以看作在固定 $r$ 下沿窄带列指标 $k=r+s$ 推进的 $(j,k)$ 递推。实际只计算

\begin{align*}
0\le j\le 99,
\qquad
0\le s\le S.
\end{align*}

为保证递推中 $s+1$ 的访问不污染保留区，内部使用初始长度 $S_{\mathrm{guard}}$，但最终只保留 $s\le S$ 进入组装。

该主递推的总复杂度为

\begin{align*}
O(N J S_{\mathrm{guard}})=O(N).
\end{align*}

---

## 7. 装配公式与卷积结构

如果先定义压缩矩

\begin{align*}
Hcomp_{r,k}^{(i)}
=
\sum_{j=0}^{J}C_{ij}H_{j,r,k},
\end{align*}

则中间矩阵 $G$ 对 $k>0$ 的标准装配公式为

\begin{align*}
G_{\ell,k}
=
\frac{2-\delta_{\ell,0}}{2}
\sum_{i=0}^{I}
\left[
\mathbf 1_{i+\ell\le k}
\frac{k}{k+i+\ell}
Hcomp_{i+\ell,k}^{(i)}
+
\mathbf 1_{|i-\ell|\le k}
\frac{k}{k+|i-\ell|}
Hcomp_{|i-\ell|,k}^{(i)}
\right].
\end{align*}

为了降低常数，不显式形成 $Hcomp$，而是直接利用 $i$ 方向的卷积结构。令

\begin{align*}
d=k-\ell,
\qquad
r=k-s,
\end{align*}

并定义

\begin{align*}
Q_{j,s}^{(k)}
=
\frac{k}{k+r}H_{j,r,k}
=
\frac{k}{2k-s}H_{j,k-s,k}.
\end{align*}

### 7.1 Toeplitz 型部分

第一项满足 $r=i+\ell$。代入 $\ell=k-d$ 与 $r=k-s$，得到

\begin{align*}
d=i+s.
\end{align*}

因此该部分为标准卷积

\begin{align*}
U_d^{(k)}
=
\sum_{j=0}^{J}\sum_{i=0}^{I}
C_{ij}Q_{j,d-i}^{(k)}.
\end{align*}

即

\begin{align*}
U^{(k)}
=
\sum_{j=0}^{J}C_{:,j}*Q_{j,:}^{(k)}.
\end{align*}

### 7.2 Hankel / correlation 型部分

第二项中主要分支为 $r=\ell-i$，即 $\ell\ge i$。代入后得到

\begin{align*}
s=d+i.
\end{align*}

于是

\begin{align*}
L_d^{(k)}
=
\sum_{j=0}^{J}\sum_{i=0}^{I}
C_{ij}Q_{j,d+i}^{(k)}.
\end{align*}

这是一种相关型卷积。令 $\operatorname{rev}(C_{:,j})$ 表示反转后的 $i$ 方向系数，则

\begin{align*}
L_d^{(k)}
=
\sum_{j=0}^{J}
\left[\operatorname{rev}(C_{:,j})*Q_{j,:}^{(k)}\right]_{I+d}.
\end{align*}

第二项中的另一分支 $r=i-\ell$ 只会影响左上角有限小块。因为在 $i\le I$、$s\le S$、$d\le B_G$ 固定时，它要求

\begin{align*}
d=2k-i-s,
\end{align*}

从而 $k=O(I+S+B_G)$。该部分可以在左上角用直接公式修正，其代价与 $N$ 无关。

### 7.3 中间矩阵 $G$ 的带状列

对每一列 $k$，在忽略左上角有限修正后，保留带内有

\begin{align*}
G_{k-d,k}
=
\rho_{k-d}\left(U_d^{(k)}+L_d^{(k)}\right),
\qquad
\rho_{\ell}=\frac{2-\delta_{\ell,0}}{2}.
\end{align*}

其中 $d$ 只取

\begin{align*}
-I\le d\le B_G.
\end{align*}

---

## 8. FFT 卷积加速组装

令

\begin{align*}
L_{\mathrm{fft}}
=
\operatorname{nextpow2}(I+S+1).
\end{align*}

### 8.1 预处理 $C$ 的 FFT

对每个 $j=0,\dots,J$，预先计算

\begin{align*}
\widehat C_j^+
=
\operatorname{fft}(C_{:,j},L_{\mathrm{fft}}),
\end{align*}

\begin{align*}
\widehat C_j^-
=
\operatorname{fft}(\operatorname{rev}(C_{:,j}),L_{\mathrm{fft}}).
\end{align*}

这一步只与固定的 $100\times100$ 系数矩阵有关，不随 $N$ 增长。

### 8.2 每列 $k$ 的 FFT 组装

对固定列 $k$，已有

\begin{align*}
Q_{j,s}^{(k)},
\qquad
0\le j\le J,
\quad
0\le s\le S.
\end{align*}

对每个 $j$ 计算

\begin{align*}
\widehat Q_j^{(k)}
=
\operatorname{fft}(Q_{j,:}^{(k)},L_{\mathrm{fft}}).
\end{align*}

然后在频域求和：

\begin{align*}
\widehat U^{(k)}
=
\sum_{j=0}^{J}\widehat C_j^+\widehat Q_j^{(k)},
\end{align*}

\begin{align*}
\widehat L^{(k)}
=
\sum_{j=0}^{J}\widehat C_j^-\widehat Q_j^{(k)}.
\end{align*}

反变换得到

\begin{align*}
U^{(k)}=\operatorname{ifft}(\widehat U^{(k)}),
\qquad
L^{(k)}=\operatorname{ifft}(\widehat L^{(k)}).
\end{align*}

随后写入 $G$ 的带状列：

\begin{align*}
G_{k-d,k}
=
\rho_{k-d}
\left(
U_d^{(k)}+L_{I+d}^{(k)}
\right),
\qquad
-I\le d\le B_G.
\end{align*}

这里 $U_d^{(k)}$ 只在 $d\ge0$ 时有效；越界项按零处理。

---

## 9. 流式缓存策略

为了避免存储完整三维数组 $Q_{j,s}^{(k)}$，采用循环列缓存。

对每个固定 $r$，通过初值计算和 $(j,k)$ 递推生成

\begin{align*}
H_{j,r,r+s},
\qquad
0\le j\le J,
\quad
0\le s\le S.
\end{align*}

令 $k=r+s$，并写入对应列的缓存

\begin{align*}
Q_{j,s}^{(k)}
=
\frac{k}{k+r}H_{j,r,k}.
\end{align*}

由于每一列 $k$ 只需要来自 $r=k-s$、$0\le s\le S$ 的贡献，因此在按 $r=0,1,\dots,N-1$ 扫描时，只需保留 $S+1$ 个活动列。可以使用大小为

\begin{align*}
(S+1)\times(J+1)\times(S+1)
\end{align*}

的环形缓存。处理完第 $r$ 层后，列 $k=r$ 的所有 $s=0,\dots,S$ 贡献已经齐全，可以立即对该列执行 FFT 组装并清空缓存槽。

这种流式策略使工作内存与 $N$ 无关。

---

## 10. 最后乘以 $M[(1+y)/2]$

中间矩阵 $G$ 对应不含外部因子 $(1+y)/2$ 的部分。最终矩阵为

\begin{align*}
A
=
M\!\left[\frac{1+y}{2}\right]G.
\end{align*}

逐行公式为

\begin{align*}
A_{0,k}
=
\frac12G_{0,k}+\frac14G_{1,k},
\end{align*}

\begin{align*}
A_{1,k}
=
\frac12G_{1,k}+\frac12G_{0,k}+\frac14G_{2,k},
\end{align*}

\begin{align*}
A_{\ell,k}
=
\frac12G_{\ell,k}
+
\frac14G_{\ell-1,k}
+
\frac14G_{\ell+1,k},
\qquad \ell\ge2.
\end{align*}

实现时先构造 $G$ 的扩展带

\begin{align*}
-I\le d_G\le B+1,
\end{align*}

再得到最终 $A$ 的目标带

\begin{align*}
-(I+1)\le d_A\le B.
\end{align*}

最终矩阵可以用 packed diagonal band 或 packed lower-Hessenberg-like band 结构存储。

---

## 11. 算法伪代码

```julia
function build_K1_fast_banded_de_weak_voltmat(mu, N;
    omega = 4.0,
    Csize_y = 100,
    Csize_eta = 100,
    B = choose_upper_bandwidth(mu, omega),
)
    I = 99
    J = 99

    # 1. Stable Chebyshev expansion of the DE smooth factor
    C = compute_de_ktilde_cheb_coeffs_K1(mu, omega, 100, 100)

    # 2. Band parameters
    BG = B + 1
    S  = I + BG
    Sguard = S + J + 1
    Lfft = nextpow(2, I + S + 1)

    # 3. Precompute FFTs of C and reversed C
    Chat_plus  = precompute_fft_columns(C, Lfft)
    Chat_minus = precompute_fft_reversed_columns(C, Lfft)

    # 4. Allocate packed band for G and circular Q buffer
    Gband = PackedDiagonalBand(rows=N+1, cols=N, lower=I, upper=BG)
    Qbuf  = CircularColumnBuffer(num_slots=S+1, J=J, S=S)

    # 5. Scan r and generate narrow (j,k) moments
    beta_r = 1 / mu
    for r in 0:N-1
        # H0[s] = H_{0,r,r+s}, s=0:Sguard
        H0 = stable_initial_H0_line(mu, r, beta_r, Sguard)

        # H1[s] = H_{1,r,r+s}
        H1 = eta_times_H0_line(r, H0, Sguard)

        # j=0,1 layers
        write_Q_layers_to_buffer!(Qbuf, r, 0, H0, N, S)
        write_Q_layers_to_buffer!(Qbuf, r, 1, H1, N, S)

        Hjm1 = H0
        Hj   = H1

        # j recurrence
        for j in 1:J-1
            Hjnext = next_j_layer_by_eta_recurrence(r, Hjm1, Hj, Sguard)
            write_Q_layers_to_buffer!(Qbuf, r, j+1, Hjnext, N, S)
            Hjm1, Hj = Hj, Hjnext
        end

        # Column k=r is complete now; assemble it by FFT convolution.
        if r >= 1
            qcol = take_complete_column!(Qbuf, k=r)
            assemble_G_column_by_fft!(Gband, qcol, r, Chat_plus, Chat_minus,
                                      I, J, S, BG, Lfft)
            clear_column_slot!(Qbuf, k=r)
        end

        # Update beta factor for next r
        beta_r = ((r + 1) / (r + mu + 1)) * beta_r
    end

    # 6. Flush remaining columns near the end if needed
    flush_tail_columns!(Gband, Qbuf, Chat_plus, Chat_minus, N, I, J, S, BG, Lfft)

    # 7. Correct first column and finite left-top corner directly
    correct_k0_and_corner!(Gband, C, mu, N, I, J, S, BG)

    # 8. Final multiplication by (1+y)/2 and truncate to final band
    Aband = multiply_by_one_plus_y_over_2_and_truncate(Gband,
                                                       lower=I+1,
                                                       upper=B)
    return Aband
end
```

---

## 12. 复杂度分析

### 12.1 参数

固定

\begin{align*}
I=J=99.
\end{align*}

设最终上带宽为 $B$。则

\begin{align*}
S=I+B+1,
\qquad
S_{\mathrm{guard}}=S+J+1,
\qquad
L_{\mathrm{fft}}=\operatorname{nextpow2}(I+S+1).
\end{align*}

当 $B$ 由机器精度阈值固定时，$S$、$S_{\mathrm{guard}}$ 和 $L_{\mathrm{fft}}$ 均与 $N$ 无关。

### 12.2 系数矩阵 $C$ 的构造

$C$ 的大小固定为 $100\times100$。因此它的构造成本为

\begin{align*}
O(1)
\end{align*}

相对于 $N$ 而言是常数。

### 12.3 递推初值

每个 $r$ 需要计算 $s=0,\dots,S_{\mathrm{guard}}$ 的 $H_{0,r,r+s}$。用有限和直接计算时，成本为

\begin{align*}
O(S_{\mathrm{guard}}^2)
\end{align*}

每个 $r$。因此总成本为

\begin{align*}
O(NS_{\mathrm{guard}}^2)=O(N).
\end{align*}

若在特殊情形 $\mu=1$ 下使用相邻比值递推，则可降为

\begin{align*}
O(NS_{\mathrm{guard}}).
\end{align*}

但对一般弱奇异 $0<\mu<1$，有限和方法已经是关于 $N$ 的线性复杂度。

### 12.4 主 $(j,k)$ 递推

每个 $r$ 递推 $j=0,\dots,J$，并只保留 $s\le S_{\mathrm{guard}}$。总成本为

\begin{align*}
O(NJS_{\mathrm{guard}})=O(N).
\end{align*}

### 12.5 FFT 卷积组装

对每一列 $k$，需要对 $J+1$ 条 $Q_{j,:}^{(k)}$ 做长度 $L_{\mathrm{fft}}$ 的 FFT，并完成频域乘加。每列成本为

\begin{align*}
O\!\left(JL_{\mathrm{fft}}\log L_{\mathrm{fft}}\right).
\end{align*}

因此总成本为

\begin{align*}
O\!\left(NJL_{\mathrm{fft}}\log L_{\mathrm{fft}}\right)=O(N).
\end{align*}

如果不用 FFT，而直接做逐点散射或直接卷积，则主要装配成本为

\begin{align*}
O(NIJS).
\end{align*}

FFT 组装将每列中 $i$ 与 $s$ 的双重求和从 $O(IS)$ 降为 $O(L_{\mathrm{fft}}\log L_{\mathrm{fft}})$。在 $I=99$、$S\approx B+100$ 的情况下，这可以显著降低 $N$ 前的常数。

### 12.6 最终乘法与存储

最终乘以 $M[(1+y)/2]$ 只涉及带状矩阵的三条相邻行，成本为

\begin{align*}
O(N(I+B)).
\end{align*}

最终 packed band 存储量为

\begin{align*}
O(N(I+B)).
\end{align*}

循环缓存与 FFT 工作区大小为

\begin{align*}
O(JS^2+JL_{\mathrm{fft}}),
\end{align*}

与 $N$ 无关。

---

## 13. 总体复杂度

综合以上步骤，对固定 $C\in\mathbb R^{100\times100}$ 和固定目标上带宽 $B$，总构造成本为

\begin{align*}
O(N).
\end{align*}

更详细地写为

\begin{align*}
O\!\left(
N S_{\mathrm{guard}}^2
+
N J S_{\mathrm{guard}}
+
N J L_{\mathrm{fft}}\log L_{\mathrm{fft}}
+
N(I+B)
\right).
\end{align*}

其中

\begin{align*}
I=J=99,
\qquad
S=I+B+1,
\qquad
S_{\mathrm{guard}}=S+J+1.
\end{align*}

若 $B$ 固定，则全部括号内因子均为常数，因此整体为线性复杂度。

---

## 14. 数值实现注意事项

1. $K=1$ 时不能把 $C$ 设为 $1\times1$。DE--Duffy 后的 $\widetilde K_\phi$ 仍是非平凡二维函数，当前算法固定使用 $100\times100$ 系数矩阵。

2. 计算 $\widetilde K_\phi$ 时必须使用稳定对数公式，避免直接计算 $\phi(y)-\phi(\tau)$。

3. $G$ 要比最终 $A$ 多计算一条上带。若最终保留 $d_A\le B$，则中间 $G$ 应至少保留 $d_G\le B+1$。

4. $k=0$ 列应单独处理，因为 $k>0$ 的装配公式中含有因子 $k/(k+r)$。

5. 第二个绝对值分支中的 $r=i-\ell$ 只影响左上角有限小块，建议直接用原始公式修正，不放入 FFT 主流程。

6. 初始线 $H_{0,r,r+s}$ 的有限和应使用补偿求和；Beta 因子用递推更新，不直接使用 Gamma 函数。

7. FFT 长度应满足至少 $L_{\mathrm{fft}}\ge I+S+1$，并使用 zero padding 避免循环卷积污染线性卷积结果。

8. 实际代码中应预分配 FFT 工作区并使用 FFTW plan，避免在每列组装中频繁分配内存。

---

## 15. 推荐的最终接口

可以在 demo 层实现如下接口：

```julia
A, info = weak_voltmat_decheb_K1_banded_fft(
    ;
    mu = 0.5,
    N = 500,
    omega = 4.0,
    Csize_y = 100,
    Csize_eta = 100,
    upper_band = :auto,
    tol_band = eps(Float64),
    output = :packed_band,
    return_info = true,
)
```

其中 `info` 建议包含：

```julia
info = (
    C_size = (100, 100),
    I = 99,
    J = 99,
    upper_band = B,
    G_upper_band = B + 1,
    lower_band_A = 100,
    S = S,
    Sguard = Sguard,
    Lfft = Lfft,
    assembly = :fft_convolution,
    recurrence = :jk_banded,
    initial_moments = :stable_beta_terminating_hypergeom,
)
```

这样可以明确记录该矩阵是按照固定 $100\times100$ DE 系数、带状 $(j,k)$ 递推、FFT 卷积组装得到的快速 $K=1$ 弱奇异 Volterra 系数矩阵。
