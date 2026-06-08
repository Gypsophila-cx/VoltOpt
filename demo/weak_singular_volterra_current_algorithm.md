# 当前弱奇异 Volterra 系数矩阵构造算法总结

本文档总结目前讨论得到的弱奇异 Volterra 积分算子系数矩阵构造流程，并给出一段可直接交给 Codex 的实现提示词。核心版本为：**固定 $j$ 的 $(r,k)$ 递推，初始 $r=0$ 层使用 $(j,k)$ 指标递推；其中初始矩 $H_{0,0,k}$ 使用附件文件中给出的稳定双指数积分算法计算。**

---

## 1. 目标算子与 Duffy 变换

考虑弱奇异 Volterra 型积分列函数

\begin{align*}
V_k(y)
=
\frac{1+y}{2}
\int_{-1}^{1}
(1-\eta)^{\mu-1}
\widetilde K_\varphi(y,\eta)
T_k\!\left(\frac{y+1}{2}\eta+\frac{y-1}{2}\right)
\,\mathrm d\eta .
\end{align*}

其中

\begin{align*}
\tau(y,\eta)
&=
\frac{y+1}{2}\eta+\frac{y-1}{2},
\\
\widetilde K_\varphi(y,\eta)
&=
\widetilde K_D(y,\eta)\widetilde K(y,\eta),
\\
\widetilde K_D(y,\eta)
&=
\varphi'(\tau(y,\eta))
\left(
\frac{\varphi(y)-\varphi(\tau(y,\eta))}{1-\eta}
\right)^{\mu-1},
\\
\widetilde K(y,\eta)
&=
K\bigl(\varphi(y),\varphi(\tau(y,\eta))\bigr).
\end{align*}

注意在 $\eta=1$ 附近必须稳定计算差商

\begin{align*}
\frac{\varphi(y)-\varphi(\tau(y,\eta))}{1-\eta}.
\end{align*}

当 $\eta\to 1$ 时，$\tau(y,\eta)\to y$，并且

\begin{align*}
y-\tau(y,\eta)=\frac{1+y}{2}(1-\eta),
\end{align*}

所以极限为

\begin{align*}
\lim_{\eta\to1}
\frac{\varphi(y)-\varphi(\tau(y,\eta))}{1-\eta}
=
\frac{1+y}{2}\varphi'(y).
\end{align*}

实现中不要在 $\eta\approx 1$ 时直接相减导致灾难性消去。

---

## 2. 光滑因子的二维 Chebyshev 展开

对光滑因子作二维 Chebyshev 展开：

\begin{align*}
\widetilde K_\varphi(y,\eta)
\approx
\sum_{i=0}^{m}\sum_{j=0}^{n}
C_{ij}T_i(y)T_j(\eta).
\end{align*}

于是定义

\begin{align*}
g_k(y)
=
\sum_{i=0}^{m}\sum_{j=0}^{n}
C_{ij}T_i(y)
\int_{-1}^{1}
(1-\eta)^{\mu-1}T_j(\eta)
T_k\!\left(\frac{y+1}{2}\eta+\frac{y-1}{2}\right)
\,\mathrm d\eta,
\end{align*}

并有

\begin{align*}
V_k(y)=\frac{1+y}{2}g_k(y).
\end{align*}

目标是构造矩阵 $A$，使得

\begin{align*}
V_k(y)\approx \sum_{\ell=0}^{N-1} A_{\ell k}T_\ell(y).
\end{align*}

---

## 3. 限制算子与核心矩 $H_{j,r,k}$

限制算子给出

\begin{align*}
T_k\!\left(\frac{1+\eta}{2}y+\frac{\eta-1}{2}\right)
=
\sum_{r=0}^{k}
L_{r,k}(\eta)T_r(y),
\end{align*}

其中对 $k>0$，

\begin{align*}
L_{r,k}(\eta)
=
\frac{k}{k+r}
\left(\frac{1+\eta}{2}\right)^r
P_{k-r}^{(-1,2r)}(\eta),
\qquad 0\le r\le k.
\end{align*}

定义核心矩

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

当前推荐的计算策略是：

1. 先用稳定双指数积分算法计算整条初始矩 $H_{0,0,k}$；
2. 用 $(j,k)$ 指标递推生成 $r=0$ 层 $H_{j,0,k}$；
3. 对每个固定 $j$，用稳定的 $(r,k)$ 递推生成 $H_{j,r,k}$；
4. 对 $j$ 即时压缩求和，避免显式存储三维数组。

---

## 4. 初始矩 $H_{0,0,k}$ 的稳定 DE 计算

初始矩为

\begin{align*}
H_{0,0,k}
=
\int_{-1}^{1}
(1-\eta)^{\mu-1}P_k^{(-1,0)}(\eta)
\,\mathrm d\eta.
\end{align*}

这一层必须使用附件文件中给出的稳定双指数积分算法。实现时需要特别避免之前出现过的 DE 坑：

- 不要在变换节点已经舍入到 $\eta=1$ 或 $\eta=-1$ 时继续评价被积函数；
- 不要直接计算会产生 `Inf`、`NaN` 或 $0^{\mu-1}$ 的端点表达式；
- 需要在节点距离端点达到机器精度时自动截断；
- 使用补偿求和或稳定求和；
- 尽量复用同一批 DE 节点和权重，一次性计算所有 $k$ 的 $P_k^{(-1,0)}(\eta)$ 递推值；
- 不要把这个初始层替换成普通 Gauss、Clenshaw--Curtis 或天真的 tanh-sinh 端点求值。

在 Codex 实现中，应当优先调用或移植附件文件中的稳定 DE routine，而不是重新写一个未经验证的版本。

---

## 5. $r=0$ 层的 $(j,k)$ 递推

记

\begin{align*}
h_{j,k}=H_{j,0,k}.
\end{align*}

已知初始行

\begin{align*}
h_{0,k}=H_{0,0,k}.
\end{align*}

利用 $P_k^{(-1,0)}$ 的乘法递推

\begin{align*}
\eta P_k^{(-1,0)}(\eta)
=
a_kP_{k+1}^{(-1,0)}(\eta)
+
b_kP_k^{(-1,0)}(\eta)
+
c_kP_{k-1}^{(-1,0)}(\eta),
\end{align*}

其中 $k\ge1$ 时

\begin{align*}
a_k=\frac{k+1}{2k+1},
\qquad
b_k=-\frac{1}{(2k-1)(2k+1)},
\qquad
c_k=\frac{k-1}{2k-1}.
\end{align*}

又因为

\begin{align*}
T_{j+1}(\eta)=2\eta T_j(\eta)-T_{j-1}(\eta),
\end{align*}

可由 $h_{0,k}$ 生成整个 $r=0$ 层。

首先

\begin{align*}
h_{1,0}=h_{0,0}+2h_{0,1},
\end{align*}

并对 $k\ge1$，

\begin{align*}
h_{1,k}=a_kh_{0,k+1}+b_kh_{0,k}+c_kh_{0,k-1}.
\end{align*}

然后对 $j\ge1$，

\begin{align*}
h_{j+1,0}
=
2\bigl(h_{j,0}+2h_{j,1}\bigr)-h_{j-1,0},
\end{align*}

并对 $k\ge1$，

\begin{align*}
h_{j+1,k}
=
2\bigl(a_kh_{j,k+1}+b_kh_{j,k}+c_kh_{j,k-1}\bigr)-h_{j-1,k}.
\end{align*}

由于递推中使用 $h_{j,k+1}$，实际需要在 $k$ 方向保留 guard columns。若需要 $0\le j\le n$、$0\le k\le N-1$，初始 $H_{0,0,k}$ 至少应计算到 $N+n+\texttt{guard}$。

---

## 6. 固定 $j$ 的 $(r,k)$ 递推

对每个固定 $j$，已经有边界层

\begin{align*}
H_{j,0,k}=h_{j,k}.
\end{align*}

然后使用之前已经验证过的固定 $j$ 的稳定 $(r,k)$ 递推，生成三角区域

\begin{align*}
\{H_{j,r,k}:0\le r\le k\le N_{\rm ext}\}.
\end{align*}

实现要求：

- 固定 $j$ 层之间彼此独立；
- 不要使用沿 $j$ 的前向递推来生成内部 $r>0$ 区域；
- 固定 $j$ 的 $(r,k)$ 递推应复用已有稳定公式或已有实现；
- 支持 triangular storage 或 packed storage；
- 可在生成当前 $j$ 层后立即压缩求和并释放。

---

## 7. 对 $j$ 即时压缩

最终组装只需要

\begin{align*}
\mathcal H^{(i)}_{r,k}
=
\sum_{j=0}^{n}C_{ij}H_{j,r,k}.
\end{align*}

所以计算每个固定 $j$ 层后，立即执行

\begin{align*}
\mathcal H^{(i)}_{r,k}
\leftarrow
\mathcal H^{(i)}_{r,k}
+
C_{ij}H_{j,r,k},
\qquad i=0,\dots,m.
\end{align*}

这样不需要存储完整 $H_{j,r,k}$ 三维数组。存储量从 $O(nN^2)$ 降为 $O(mN^2)+O(N^2)$。

---

## 8. 组装 $g_k$ 的 Chebyshev 系数矩阵 $G$

记

\begin{align*}
g_k(y)=\sum_{\ell\ge0}G_{\ell k}T_\ell(y).
\end{align*}

对 $k=0$，

\begin{align*}
G_{\ell,0}=\mathcal H^{(\ell)}_{0,0},
\end{align*}

其中 $\ell>m$ 时取零。

对 $k>0$，

\begin{align*}
G_{\ell k}
=
\frac{2-\delta_{\ell0}}{2}
\sum_{i=0}^{m}
\left[
\mathbf 1_{i+\ell\le k}
\frac{k}{k+i+\ell}
\mathcal H^{(i)}_{i+\ell,k}
+
\mathbf 1_{|i-\ell|\le k}
\frac{k}{k+|i-\ell|}
\mathcal H^{(i)}_{|i-\ell|,k}
\right].
\end{align*}

为了最后乘以 $(1+y)/2$，实际应至少计算 $G_{\ell k}$ 到 $\ell=N$，最后再截取 $N\times N$ 输出。

---

## 9. 乘外部因子并得到最终矩阵

由于

\begin{align*}
V_k(y)=\frac{1+y}{2}g_k(y),
\end{align*}

最终矩阵为

\begin{align*}
A=\mathcal M\!\left[\frac{1+y}{2}\right]G.
\end{align*}

逐项写为

\begin{align*}
A_{0k}
&=
\frac12G_{0k}+\frac14G_{1k},
\\
A_{1k}
&=
\frac12G_{1k}+\frac12G_{0k}+\frac14G_{2k},
\\
A_{\ell k}
&=
\frac12G_{\ell k}+\frac14G_{\ell-1,k}+\frac14G_{\ell+1,k},
\qquad \ell\ge2.
\end{align*}

最终输出

\begin{align*}
A_{0:N-1,0:N-1}.
\end{align*}

---

## 10. 推荐实现结构

推荐函数结构如下：

```julia
# 1. Build transformed kernel coefficients C
C, infoC = build_Kphi_cheb_coeffs(K, phi, dphi; mu, m, n, tol)

# 2. Stable DE initial moments, from attached/reference implementation
H00 = stable_de_H000_moments(mu, Kext; tol, h, maxlevel, compensated=true)

# 3. r = 0 layer via (j,k)-recurrence
H_r0 = build_r0_layer_from_H00(H00, n, Kneed; guard)

# 4. For each fixed j, build H[j,:,:] using stable (r,k)-recurrence
Hbar = zeros(m+1, triangular_size(Kneed))
for j in 0:n
    Hj = build_H_fixed_j_rk(H_r0[j+1, :], j, mu, Kneed)
    for i in 0:m
        axpy!(C[i+1,j+1], Hj, view_Hbar_i)
    end
end

# 5. Assemble G from Hbar
G = assemble_G_from_Hbar(Hbar, C, N, m)

# 6. Multiply by (1+y)/2 and truncate
A = multiply_by_one_plus_y_over_two(G, N)
```

---

## 11. Suggested validation tests

1. **No endpoint failure test**: for $\mu=0.5$ and large $K_{\rm ext}$, verify `stable_de_H000_moments` returns no `NaN` or `Inf`.
2. **Small-size brute force test**: compare selected $A_{\ell k}$ against direct high-order quadrature for small $N$.
3. **Identity map test**: set $\varphi(x)=x$ and simple smooth kernels such as $K(x,y)=1$ or $K(x,y)=\exp(-xy)$.
4. **Column $k=0$ test**: verify $G_{\ell,0}=\mathcal H^{(\ell)}_{0,0}$ and final multiplication by $(1+y)/2$ is correct.
5. **Structure test**: plot `log10.(abs.(A))` and check the expected Volterra lower-Hessenberg/top-dense-plus-banded structure.
6. **Regression test against previous implementation**: for moderate $N$, compare the new matrix to the older validated matrix entries where both are reliable.

---

## 12. Codex 提示词

```text
请重新实现弱奇异 Volterra 积分算子的 Chebyshev 系数矩阵构造函数，按照当前版本的算法，不要沿 j 前向递推内部矩。目标是构造矩阵 A，使得

V_k(y) = ((1+y)/2) ∫_{-1}^1 (1-η)^{μ-1} K̃_φ(y,η) T_k(((y+1)/2)η+(y-1)/2) dη
      ≈ ∑_ℓ A_{ℓk} T_ℓ(y)。

核心算法要求如下：

1. 先构造 τ(y,η)=((y+1)/2)η+(y-1)/2，并对
   K̃_φ(y,η)=φ'(τ(y,η))*((φ(y)-φ(τ(y,η)))/(1-η))^(μ-1)*K(φ(y),φ(τ(y,η)))
   做二维 Chebyshev 展开，得到 C[i,j]。在 η≈1 时必须稳定计算差商，使用极限
   ((1+y)/2)*φ'(y)，不要直接做会灾难性消去的相减。

2. 定义
   H[j,r,k] = ∫_{-1}^1 (1-η)^(μ-1) T_j(η) ((1+η)/2)^r P_{k-r}^{(-1,2r)}(η) dη,
   0≤r≤k。

3. 初始矩 H[0,0,k] 必须使用附件/参考文件中已经给出的稳定双指数积分算法计算。不要重新写一个天真的 tanh-sinh/DE 实现。特别注意：
   - transformed node 已经舍入到 η=1 或 η=-1 时不要继续评价被积函数；
   - 不要在端点评价 (1-η)^(μ-1)，避免 Inf/NaN；
   - 使用附件中的自动截断逻辑，节点距离端点达到机器精度时停止；
   - 使用补偿求和或附件中的稳定求和；
   - 最好复用同一批 DE 节点和权重，一次性递推计算所有 P_k^{(-1,0)}(η)，得到整条 H[0,0,k]；
   - 不要把这个初始层替换成普通 Gauss、Clenshaw--Curtis 或未经验证的端点求值版本。

4. 使用 (j,k) 指标递推生成 r=0 层 h[j,k]=H[j,0,k]。已知 h[0,k]=H[0,0,k]。对 k≥1，使用
   a_k=(k+1)/(2k+1), b_k=-1/((2k-1)*(2k+1)), c_k=(k-1)/(2k-1)。
   初始化：
   h[1,0]=h[0,0]+2*h[0,1]
   h[1,k]=a_k*h[0,k+1]+b_k*h[0,k]+c_k*h[0,k-1]
   递推：
   h[j+1,0]=2*(h[j,0]+2*h[j,1])-h[j-1,0]
   h[j+1,k]=2*(a_k*h[j,k+1]+b_k*h[j,k]+c_k*h[j,k-1])-h[j-1,k]。
   由于需要 h[j,k+1]，在 k 方向保留足够 guard columns；若需要 0≤j≤n 和 0≤k≤N-1，至少计算 H[0,0,k] 到 N+n+guard。

5. 对每个固定 j，以上一步得到的 H[j,0,k] 为边界层，使用已有稳定的固定 j 的 (r,k) 递推生成 H[j,r,k] 的三角区域。不要使用沿 j 前向递推生成 r>0 的内部区域。请复用当前代码库或参考推导中已经验证过的 fixed-j rk recurrence，支持 triangular/packed storage。

6. 不要显式存储完整三维 H[j,r,k]。每完成一个固定 j 层，就立即累加
   Hbar[i,r,k] += C[i,j] * H[j,r,k]
   对所有 i=0:m。最终只保留 Hbar[i,r,k]=∑_j C[i,j]H[j,r,k]。

7. 由 Hbar 组装 G。对 k=0，G[ℓ,0]=Hbar[ℓ,0,0]，ℓ>m 时为 0。对 k>0，使用
   G[ℓ,k] = (2-δ_{ℓ0})/2 * Σ_i [
       1_{i+ℓ≤k} * k/(k+i+ℓ) * Hbar[i,i+ℓ,k]
     + 1_{|i-ℓ|≤k} * k/(k+|i-ℓ|) * Hbar[i,|i-ℓ|,k]
   ]。
   为了之后乘以 (1+y)/2，G 至少计算到 row index ℓ=N。

8. 最后乘外部因子 (1+y)/2：
   A[0,k]=0.5*G[0,k]+0.25*G[1,k]
   A[1,k]=0.5*G[1,k]+0.5*G[0,k]+0.25*G[2,k]
   A[ℓ,k]=0.5*G[ℓ,k]+0.25*G[ℓ-1,k]+0.25*G[ℓ+1,k] for ℓ≥2。
   返回 A[0:N-1,0:N-1]。

9. 添加测试：
   - H[0,0,k] 的 DE 计算不允许出现 NaN/Inf；
   - 小 N 下和直接高精度数值积分的矩阵元素比较；
   - φ(x)=x 且 K(x,y)=1 或 exp(-x*y) 的 sanity check；
   - k=0 列和乘以 (1+y)/2 的索引检查；
   - 绘制 log10(abs(A)) 检查结构；
   - 和旧实现做中等 N 的回归比较。

请在实现中保留清晰的函数边界：build_Kphi_cheb_coeffs、stable_de_H000_moments、build_r0_layer_from_H00、build_H_fixed_j_rk、accumulate_Hbar、assemble_G_from_Hbar、multiply_by_one_plus_y_over_two。不要引入 BigFloat 作为默认路径；如果需要高精度验证，可以只在 test/reference 中使用。默认路径应为稳定的 Float64 实现。
```
