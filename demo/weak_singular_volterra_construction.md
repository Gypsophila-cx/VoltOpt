# 弱奇异 Volterra 积分算子的谱构造思路

本文档整理当前用于构造弱奇异 Volterra 积分算子的谱方法框架。核心思想是先通过 Duffy 变换分离 Volterra 三角区域上的弱奇异因子，再将光滑核函数展开为二维 Chebyshev 级数，最后利用限制算子矩阵元及其矩量递推关系快速生成算子系数矩阵。

## 1. 问题形式

考虑弱奇异 Volterra 积分算子

$$
(\mathcal V_K u)(x)
=
\int_{-1}^{x} (x-t)^{\mu-1}K(x,t)u(t)\,\mathrm dt,
\qquad 0<\mu<1.
$$

若采用更常见的记号 $(x-t)^{-\alpha}$，则对应关系为 $\mu=1-\alpha$。

令输入函数用 Chebyshev 基表示为

$$
u(t)=\sum_{k\geq 0} u_k T_k(t).
$$

为了构造算子矩阵，只需研究基函数 $T_k$ 在该算子下的像。

## 2. Duffy 变换与弱奇异因子分离

对固定的外变量 $y\in[-1,1]$，令

$$
t=\frac{y+1}{2}\eta+\frac{y-1}{2},
\qquad \eta\in[-1,1].
$$

则 $t\in[-1,y]$，且

$$
y-t=\frac{y+1}{2}(1-\eta),
\qquad
\mathrm dt=\frac{y+1}{2}\,\mathrm d\eta.
$$

因此弱奇异核的主奇异性被转换为端点权重 $(1-\eta)^{\mu-1}$，并得到

$$
(\mathcal V_K T_k)(y)
=
\left(\frac{1+y}{2}\right)^\mu
\int_{-1}^{1}
(1-\eta)^{\mu-1}
\widetilde K_\varphi(y,\eta)
T_k\left(\frac{y+1}{2}\eta+\frac{y-1}{2}\right)
\,\mathrm d\eta.
$$

这里 $\widetilde K_\varphi$ 表示经过 Duffy 变换及可能的光滑因子吸收之后得到的光滑核函数。

## 3. 光滑核函数的二维 Chebyshev 展开

对光滑函数 $\widetilde K_\varphi(y,\eta)$ 作二维 Chebyshev 展开：

$$
\widetilde K_\varphi(y,\eta)
\approx
\sum_{i=0}^{I}\sum_{j=0}^{J} C_{ij}T_i(y)T_j(\eta).
$$

于是可将除去外部因子 $((1+y)/2)^\mu$ 后的部分写为

$$
g_k(y)
=
\sum_{i=0}^{I}\sum_{j=0}^{J}C_{ij}T_i(y)
\int_{-1}^{1}
(1-\eta)^{\mu-1}T_j(\eta)
T_k\left(\frac{y+1}{2}\eta+\frac{y-1}{2}\right)
\,\mathrm d\eta.
$$

最终有

$$
(\mathcal V_K T_k)(y)
=
\left(\frac{1+y}{2}\right)^\mu g_k(y).
$$

因此算法上可以先构造 $g_k$ 的 Chebyshev 系数矩阵，再左乘函数 $((1+y)/2)^\mu$ 对应的乘法矩阵。

## 4. 交换积分顺序与限制算子矩阵元

令 $g_k$ 的第 $\ell$ 个 Chebyshev 投影系数为

$$
\widehat g_{\ell k}
=
\frac{2-\delta_{\ell0}}{\pi}
\int_{-1}^{1}
\frac{g_k(y)T_\ell(y)}{\sqrt{1-y^2}}\,\mathrm dy.
$$

交换 $y$ 与 $\eta$ 的积分顺序后，出现核心积分

$$
\frac{1}{\pi}
\int_{-1}^{1}
\frac{T_r(y)T_k\left(\frac{1+\eta}{2}y+\frac{\eta-1}{2}\right)}{\sqrt{1-y^2}}\,\mathrm dy.
$$

这是将 $T_k$ 限制到区间 $[-1,\eta]$ 后再投影到 $T_r$ 上的限制算子矩阵元。对 $k\geq 1$，有

$$
\rho_{r,k}(\eta)
=
\frac{1}{\pi}
\int_{-1}^{1}
\frac{T_r(y)T_k\left(\frac{1+\eta}{2}y+\frac{\eta-1}{2}\right)}{\sqrt{1-y^2}}\,\mathrm dy
=
\frac{k}{k+r}
\left(\frac{1+\eta}{2}\right)^r
P_{k-r}^{(-1,2r)}(\eta),
$$

其中 $0\leq r\leq k$；当 $r>k$ 时，$\rho_{r,k}(\eta)=0$。

## 5. 核心弱奇异矩量

定义

$$
H_{j,r,k}^{(\mu)}
=
\int_{-1}^{1}
(1-\eta)^{\mu-1}T_j(\eta)
\left(\frac{1+\eta}{2}\right)^r
P_{k-r}^{(-1,2r)}(\eta)
\,\mathrm d\eta,
\qquad 0\leq r\leq k.
$$

并约定

$$
H_{j,r,k}^{(\mu)}=0,
\qquad r>k.
$$

利用乘积公式

$$
T_i(y)T_\ell(y)
=
\frac12\left(T_{i+\ell}(y)+T_{|i-\ell|}(y)\right),
$$

得到对 $k\geq1$ 的矩阵系数公式

$$
\widehat g_{\ell k}
=
\frac{2-\delta_{\ell0}}{2}
\sum_{i=0}^{I}\sum_{j=0}^{J} C_{ij}
\left[
\frac{k}{k+i+\ell}H_{j,i+\ell,k}^{(\mu)}
+
\frac{k}{k+|i-\ell|}H_{j,|i-\ell|,k}^{(\mu)}
\right].
$$

当 $k=0$ 时，$T_0=1$，因此

$$
\widehat g_{\ell0}
=
\sum_{j=0}^{J}C_{\ell j}m_j^{(\mu)},
$$

其中

$$
m_j^{(\mu)}
=
\int_{-1}^{1}(1-\eta)^{\mu-1}T_j(\eta)\,\mathrm d\eta.
$$

## 6. 初始层 $r=0$ 的五点递推

令

$$
h_{j,k}^{(\mu)}
:=
H_{j,0,k}^{(\mu)}
=
\int_{-1}^{1}(1-\eta)^{\mu-1}T_j(\eta)P_k^{(-1,0)}(\eta)\,\mathrm d\eta.
$$

### 6.1 初始行 $j=0$：双指数积分计算

本文实现中要求初始矩量

$$
h_{0,k}^{(\mu)}
=
H_{0,0,k}^{(\mu)}
=
\int_{-1}^{1}(1-\eta)^{\mu-1}P_k^{(-1,0)}(\eta)\,\mathrm d\eta
$$

使用双指数变换直接计算，而不是默认依赖闭式比值递推。闭式公式可以作为校验量，但默认数值路径采用 DE 积分，以便后续推广到更一般的初始矩量或带有额外光滑因子的情形。

采用标准 tanh--sinh 变换

$$
\eta=\psi(t)=\tanh\left(\frac{\pi}{2}\sinh t\right),
\qquad t\in(-\infty,\infty),
$$

则

$$
\psi'(t)
=
\frac{\pi}{2}\cosh t\,\operatorname{sech}^2\left(\frac{\pi}{2}\sinh t\right),
$$

并且

$$
h_{0,k}^{(\mu)}
=
\int_{-\infty}^{\infty}
(1-\psi(t))^{\mu-1}
P_k^{(-1,0)}(\psi(t))
\psi'(t)\,\mathrm dt .
$$

用步长 $h_{\mathrm{DE}}$ 的梯形公式离散为

$$
h_{0,k}^{(\mu)}
\approx
h_{\mathrm{DE}}
\sum_{m=-M_-}^{M_+}
(1-\psi(mh_{\mathrm{DE}}))^{\mu-1}
P_k^{(-1,0)}(\psi(mh_{\mathrm{DE}}))
\psi'(mh_{\mathrm{DE}}).
$$

关键实现要求是端点截断必须真正到达机器精度邻域。也就是说，需要选择

$$
T_+=M_+h_{\mathrm{DE}},
\qquad
T_-=M_-h_{\mathrm{DE}},
$$

使得

$$
0<1-\psi(T_+)\lesssim \varepsilon_{\mathrm{mach}},
\qquad
0<1+\psi(-T_-)\lesssim \varepsilon_{\mathrm{mach}}.
$$

在 Float64 中，应要求两侧截断处的映射点与 $\pm1$ 的距离达到机器精度量级。若 $\psi(T_+)$ 或 $\psi(-T_-)$ 在浮点中已经被舍入成精确的 $1$ 或 $-1$，则不应直接在该点计算奇异因子。实际实现可以采用以下任一方式：

- 将右端最后一个节点替换为 `prevfloat(1.0)`，将左端最后一个节点替换为 `nextfloat(-1.0)`；
- 或者在检测到 $\psi(t)=\pm1$ 时回退一个 DE 网格点，确保实际积分端点仍满足 $|1-|\psi(t)||=O(\varepsilon_{\mathrm{mach}})$ 且不等于零；
- 或者用稳定公式直接计算 $1-\psi(t)$ 与 $1+\psi(t)$，避免在端点处发生 $0^{\mu-1}\psi'(t)$ 的不定型。

这一条件非常重要。如果截断端点离 $\pm1$ 仍有可见距离，则 DE 积分实际只是在较短区间上积分，会系统性丢失端点弱奇异贡献；如果端点被舍入为精确 $\pm1$，则右端奇异因子可能产生 `Inf * 0` 或 `NaN`。因此正确实现应当保证“接近到机器精度，但不直接落在奇异端点上”。

在实现上，可一次生成所有 DE 节点 $\eta_m=\psi(mh_{\mathrm{DE}})$ 和权重

$$
w_m^{(\mu)}
=
h_{\mathrm{DE}}(1-\eta_m)^{\mu-1}\psi'(mh_{\mathrm{DE}}),
$$

然后对所有 $k$ 同时累加

$$
h_{0,k}^{(\mu)}
\approx
\sum_m w_m^{(\mu)}P_k^{(-1,0)}(\eta_m),
\qquad 0\leq k\leq N+J+1.
$$

$P_k^{(-1,0)}(\eta_m)$ 可以在每个 DE 节点上用 Jacobi 三项递推生成，从而避免逐个 $k$ 调用特殊函数。

闭式

$$
h_{0,0}^{(\mu)}=\frac{2^\mu}{\mu},
\qquad
h_{0,k}^{(\mu)}=-2^\mu\frac{(1-\mu)_{k-1}}{(\mu+1)_k},\quad k\geq1,
$$

以及比值关系

$$
h_{0,1}^{(\mu)}=-\frac{2^\mu}{\mu+1},
\qquad
h_{0,k+1}^{(\mu)}=\frac{k-\mu}{k+\mu+1}h_{0,k}^{(\mu)}
$$

可以保留为调试与验证工具，但不作为本文实现的默认初始矩量生成方式。

### 6.2 第二行 $j=1$

由于 $T_1(\eta)=\eta$，结合

$$
\eta P_k^{(-1,0)}(\eta)
=
a_kP_{k+1}^{(-1,0)}(\eta)
+b_kP_k^{(-1,0)}(\eta)
+c_kP_{k-1}^{(-1,0)}(\eta),
$$

其中对 $k\geq1$，

$$
a_k=\frac{k+1}{2k+1},
\qquad
b_k=-\frac{1}{(2k-1)(2k+1)},
\qquad
c_k=\frac{k-1}{2k-1},
$$

可得

$$
h_{1,k}^{(\mu)}
=
a_kh_{0,k+1}^{(\mu)}
+b_kh_{0,k}^{(\mu)}
+c_kh_{0,k-1}^{(\mu)},
\qquad k\geq1.
$$

边界 $k=0$ 为

$$
h_{1,0}^{(\mu)}
=
2h_{0,1}^{(\mu)}+h_{0,0}^{(\mu)}.
$$

### 6.3 五点递推

对 $j\geq1$，由

$$
\eta T_j(\eta)=\frac12\left(T_{j+1}(\eta)+T_{j-1}(\eta)\right)
$$

与 $P_k^{(-1,0)}$ 的三项递推相结合，得到

$$
h_{j+1,k}^{(\mu)}
=
2a_kh_{j,k+1}^{(\mu)}
+2b_kh_{j,k}^{(\mu)}
+2c_kh_{j,k-1}^{(\mu)}
-h_{j-1,k}^{(\mu)},
\qquad k\geq1.
$$

边界 $k=0$ 可以使用退化公式

$$
h_{j+1,0}^{(\mu)}
=
4h_{j,1}^{(\mu)}+2h_{j,0}^{(\mu)}-h_{j-1,0}^{(\mu)}.
$$

若 $j$ 的最大阶数为 $J$，目标矩阵维度为 $N$，则计算初始层时应当在 $k$ 方向多做 padding，例如计算到 $N+J+1$，以避免右边界截断误差影响目标区域。

## 7. 跨 $r$ 层的四点递推

定义

$$
Q_k^r(\eta)
=
\left(\frac{1+\eta}{2}\right)^r
P_{k-r}^{(-1,2r)}(\eta).
$$

则 $Q_k^r$ 满足点态四点关系

$$
(k+r+1)Q_{k+1}^{r+1}(\eta)
+(k-r-1)Q_k^{r+1}(\eta)
=
(k-r+1)Q_{k+1}^{r}(\eta)
+(k+r+1)Q_k^r(\eta).
$$

乘以 $(1-\eta)^{\mu-1}T_j(\eta)$ 并积分，得到

$$
(k+r+1)H_{j,r+1,k+1}^{(\mu)}
+(k-r-1)H_{j,r+1,k}^{(\mu)}
=
(k-r+1)H_{j,r,k+1}^{(\mu)}
+(k+r+1)H_{j,r,k}^{(\mu)}.
$$

等价地，

$$
H_{j,r+1,k+1}^{(\mu)}
=
H_{j,r,k}^{(\mu)}
+
\frac{k-r+1}{k+r+1}H_{j,r,k+1}^{(\mu)}
-
\frac{k-r-1}{k+r+1}H_{j,r+1,k}^{(\mu)}.
$$

边界为

$$
H_{j,r+1,r+1}^{(\mu)}
=
H_{j,r,r}^{(\mu)}
+
\frac{1}{2r+1}H_{j,r,r+1}^{(\mu)}.
$$

因此对每个固定的 $j$，可以从 $r=0$ 层出发，逐层生成所有 $0\leq r\leq k\leq N$ 的矩量。

## 8. 算子矩阵的组装流程

整体流程如下。

1. 对 Duffy 后的光滑核 $\widetilde K_\varphi(y,\eta)$ 作二维 Chebyshev 展开，得到系数矩阵 $C_{ij}$。
2. 用满足端点机器精度截断要求的 DE 双指数积分计算初始行 $h_{0,k}^{(\mu)}=H_{0,0,k}^{(\mu)}$。
3. 用五点递推生成初始层 $H_{j,0,k}^{(\mu)}$。
4. 对每个固定 $j$，用跨 $r$ 层的四点递推生成 $H_{j,r,k}^{(\mu)}$。
5. 将 $H_{j,r,k}^{(\mu)}$ 代入

   $$
   \widehat g_{\ell k}
   =
   \frac{2-\delta_{\ell0}}{2}
   \sum_{i=0}^{I}\sum_{j=0}^{J} C_{ij}
   \left[
   \frac{k}{k+i+\ell}H_{j,i+\ell,k}^{(\mu)}
   +
   \frac{k}{k+|i-\ell|}H_{j,|i-\ell|,k}^{(\mu)}
   \right]
   $$

   组装得到 $g_k$ 的 Chebyshev 系数矩阵 $G=(\widehat g_{\ell k})$。
6. 最后乘上外部因子 $((1+y)/2)^\mu$ 的 Chebyshev 乘法矩阵，得到原弱奇异 Volterra 算子的系数矩阵。

## 9. 复杂度与结构

若 $\widetilde K_\varphi$ 的二维 Chebyshev 展开大小为 $(I+1)\times(J+1)$，目标矩阵维度为 $N$，则：

- 初始矩量 $H_{0,0,k}^{(\mu)}$ 由 DE 积分生成。若 DE 节点数为 $M_{\mathrm{DE}}$，并用 Jacobi 三项递推在每个节点生成 $P_k^{(-1,0)}$，则成本约为 $O(M_{\mathrm{DE}}N)$；
- 初始层 $H_{j,0,k}^{(\mu)}$ 的五点递推复杂度约为 $O(JN)$，加上 padding 后仍为 $O(JN)$；
- 对所有 $j$ 生成三角区域 $0\leq r\leq k\leq N$ 的复杂度为 $O(JN^2)$；
- 直接代入求和组装矩阵的复杂度为 $O(IJN^2)$；
- 当核展开阶数 $I,J$ 固定时，总复杂度关于矩阵维度为 $O(N^2)$。

这里不应将其理解为严格带状矩阵的 $O(N)$ 构造。虽然限制算子导致矩阵具有下三角或 lower-Hessenberg 型结构，但弱奇异 Volterra 积分算子的有效系数区域通常仍包含 $O(N^2)$ 个自由元素。

## 10. 数值稳定性说明

该构造的稳定性主要来自三个方面：

1. 弱奇异权重 $(1-\eta)^{\mu-1}$ 只进入初始行 $H_{0,0,k}^{(\mu)}$，而该初始行使用 DE 双指数积分直接计算；DE 截断端点必须与 $\pm1$ 相差机器精度量级，同时避免在浮点中精确落到奇异端点；
2. 初始层 $r=0$ 的五点递推在典型情形 $J\ll N$ 下较稳定，因为高 $k$ 区域中递推近似为中性振荡传播；
3. 跨 $r$ 层的四点递推是由 Jacobi 多项式的精确相邻关系导出的点态恒等式，不引入额外数值积分误差。

实际实现中建议：

- 在 $k$ 方向为五点递推保留 padding；
- 生成 DE 节点时检查 $1-\eta_{\max}$ 与 $1+\eta_{\min}$ 是否达到机器精度量级；未达到则继续增大截断长度，若已舍入到精确端点则回退或使用 `prevfloat`/`nextfloat`；
- 对 $k=0$ 边界可以单独用一维矩量公式或高精度积分校正；
- 若 $J$ 较大，应监测初始层五点递推的舍入误差传播；
- 对最终矩阵可使用 lower-Hessenberg 或压缩三角格式存储。

## 11. 核心总结

当前构造的主线可以概括为

$$
\boxed{
\text{Duffy 变换}
\longrightarrow
\text{二维 Chebyshev 展开}
\longrightarrow
\text{限制算子矩阵元}
\longrightarrow
H_{j,r,k}^{(\mu)}\text{ 递推}
\longrightarrow
\text{矩阵组装}.
}
$$

其中最关键的递推结构是

$$
\boxed{
H_{0,0,k}^{(\mu)}
\longrightarrow
H_{j,0,k}^{(\mu)}
\longrightarrow
H_{j,r,k}^{(\mu)}.
}
$$

第一步用满足端点机器精度截断条件的 DE 双指数积分计算弱奇异初始矩量，第二步用 $(j,k)$ 五点递推生成初始层，第三步用 $(r,k)$ 四点递推生成所有限制算子矩量。最终，弱奇异 Volterra 积分算子的矩阵组装被转化为少量高精度初始积分、局部代数递推和有限求和，从而避免了对每个矩阵元进行独立的弱奇异数值积分。
