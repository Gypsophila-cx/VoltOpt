# 为什么 \(K=1\) 时 DE 弱奇异 Volterra 系数矩阵的带外元素足够小

本文只讨论当前 `Weak_VoltMat_DE` 构造中最特殊、也最容易观察到结构的情形：

\[
K(x,t)\equiv 1 .
\]

目标是解释如下数值现象：

> 在 Chebyshev 系数矩阵 \(A=(A_{\ell k})\) 中，当输出模态行号 \(\ell\) 比输入模态列号 \(k\) 大到一定程度时，矩阵元素 \(A_{\ell k}\) 的绝对值会低于双精度机器精度。因此矩阵在数值上表现为具有上带宽。

这里的“上带宽”按照矩阵索引理解为：若 \(d=\ell-k\) 足够大，则

\[
|A_{\ell k}|\lesssim \varepsilon_{\rm mach}.
\]

本文的核心结论是：

\[
\boxed{
|A_{\ell k}|
\;\lesssim\;
C_{\rm end}
\bigl(\phi'(-1)\bigr)^\mu
d^{-2\mu}
\frac{1}{1-\Lambda/d^2}
+
C_{\rm ana}\rho^{-d},
\qquad
d=\ell-k .
}
\]

其中 \(\rho>1\)，\(\Lambda>0\)，\(C_{\rm end},C_{\rm ana}\) 与 \(\ell,k\) 无关。对当前 DE 变换

\[
\phi(y)=\tanh\!\left(\frac{\pi}{2}\sinh(\omega y)\right),
\]

有

\[
\phi'(-1)
=
\frac{\pi\omega}{2}\cosh(\omega)
\operatorname{sech}^2\!\left(\frac{\pi}{2}\sinh\omega\right).
\]

当 \(\omega=4\) 时，

\[
\phi'(-1)\approx 4.01\times 10^{-35},
\qquad
\bigl(\phi'(-1)\bigr)^{1/2}\approx 6.33\times 10^{-18}.
\]

因此在常用参数 \(\mu=1/2,\omega=4\) 下，端点代数尾项的系数本身已经在机器精度附近，剩余解析项则随 \(d=\ell-k\) 几何衰减。这解释了为什么当 \(d\) 达到几十到一百左右时，带外元素通常已经小到双精度不可见。

---

## 1. 算子形式与最终核

当前算法在 Duffy 变换后，对输入基函数 \(T_k\) 计算

\[
V_k(y)
=
\frac{1+y}{2}
\int_{-1}^{1}
(1-\eta)^{\mu-1}
\widetilde K_\phi(y,\eta)
T_k(\tau(y,\eta))\,\mathrm d\eta ,
\]

其中

\[
\tau(y,\eta)
=
\frac{y+1}{2}\eta+\frac{y-1}{2}.
\]

当 \(K=1\) 时，

\[
\widetilde K_\phi(y,\eta)
=
\phi'(\tau)
\left(
\frac{\phi(y)-\phi(\tau)}{1-\eta}
\right)^{\mu-1}.
\]

为了分析最终矩阵 \(A\)，把外部因子 \((1+y)/2\) 合并进核中，定义

\[
B(y,\eta)
=
\frac{1+y}{2}
\phi'(\tau)
\left(
\frac{\phi(y)-\phi(\tau)}{1-\eta}
\right)^{\mu-1}.
\]

于是

\[
V_k(y)
=
\int_{-1}^{1}
(1-\eta)^{\mu-1}
B(y,\eta)
T_k(\tau(y,\eta))\,\mathrm d\eta .
\]

矩阵元素 \(A_{\ell k}\) 就是 \(V_k(y)\) 的第 \(\ell\) 个 Chebyshev 系数。

---

## 2. “数值上具有上带宽”的定义

设 \(d=\ell-k\)。称 \(A\) 在容差 \(\varepsilon\) 下具有数值上带宽 \(b\)，是指

\[
|A_{\ell k}|\le \varepsilon,
\qquad
\ell>k+b.
\]

本文要证明的不是严格有限带宽，而是说明存在一个可估计的 \(b\)，使得

\[
|A_{\ell k}|\le \varepsilon_{\rm mach},
\qquad
\ell>k+b.
\]

对当前问题来说，严格有限带宽一般并不成立，因为 \(B(y,\eta)\) 不是 \(y\) 的有限阶多项式。但由于其高阶 Chebyshev 尾项足够小，矩阵在双精度下看起来像上带状矩阵。

---

## 3. 带外元素只由 \(B\) 的 \(y\) 向高阶尾项贡献

固定 \(\eta\)。因为

\[
\tau(y,\eta)
=
\frac{1+\eta}{2}y+\frac{\eta-1}{2}
\]

关于 \(y\) 是一次函数，所以

\[
T_k(\tau(y,\eta))
\]

是 \(y\) 的次数不超过 \(k\) 的多项式。

令 \(\Pi_m^y B\) 表示 \(B\) 关于 \(y\) 的 \(m\) 阶 Chebyshev 截断，并定义

\[
E_m
=
\sup_{\eta\in[-1,1]}
\left\|
B(\cdot,\eta)-\Pi_m^yB(\cdot,\eta)
\right\|_{L^\infty(-1,1)} .
\]

取 \(m=d-1=\ell-k-1\)。则

\[
\Pi_{d-1}^yB(y,\eta)\,T_k(\tau(y,\eta))
\]

关于 \(y\) 的次数不超过

\[
(d-1)+k
=
\ell-1.
\]

因此它不可能贡献第 \(\ell\) 个 Chebyshev 模态。所以 \(A_{\ell k}\) 完全由尾项 \(B-\Pi_{d-1}^yB\) 贡献。

利用 \(|T_k|\le1\) 与

\[
\int_{-1}^{1}(1-\eta)^{\mu-1}\,\mathrm d\eta
=
\frac{2^\mu}{\mu},
\]

并使用粗略的 Chebyshev 系数估计 \(|[T_\ell]f|\le 2\|f\|_\infty\)，得到

\[
\boxed{
|A_{\ell k}|
\le
\frac{2^{\mu+1}}{\mu}
E_{\ell-k-1}.
}
\]

这一步是整个论证的基础。它说明：

> 带外元素的大小由最终核 \(B(y,\eta)\) 在 \(y\) 方向的 Chebyshev 尾项控制。

---

## 4. \(K=1\) 时最终核的左端点结构

令

\[
s=\frac{1+y}{2},
\qquad
q=\frac{1+\eta}{2}.
\]

则

\[
y=-1+2s,
\qquad
\tau=-1+2sq.
\]

又因为

\[
y-\tau
=
s(1-\eta),
\]

由平均值公式，

\[
\phi(y)-\phi(\tau)
=
(y-\tau)
\int_0^1
\phi'\!\left(\tau+\theta(y-\tau)\right)\,\mathrm d\theta .
\]

于是

\[
\frac{\phi(y)-\phi(\tau)}{1-\eta}
=
sR(s,q),
\]

其中

\[
R(s,q)
=
\int_0^1
\phi'\!\left(
-1+2s(q+(1-q)\theta)
\right)\,\mathrm d\theta .
\]

代入 \(B\)，得到

\[
B(y,\eta)
=
s\,\phi'(-1+2sq)
\left(sR(s,q)\right)^{\mu-1}.
\]

因此

\[
\boxed{
B(y,\eta)
=
s^\mu L(s,q),
}
\]

其中

\[
L(s,q)
=
\phi'(-1+2sq)R(s,q)^{\mu-1}.
\]

记

\[
g_0=\phi'(-1).
\]

当 \(s=0\) 时，

\[
R(0,q)=g_0,
\qquad
\phi'(-1+2sq)|_{s=0}=g_0.
\]

所以

\[
L(0,q)=g_0^\mu.
\]

因此

\[
\boxed{
B(y,\eta)
=
g_0^\mu s^\mu
+
s^\mu\bigl(L(s,q)-g_0^\mu\bigr).
}
\]

若 \(L\) 在 \(s=0\) 附近展开为

\[
L(s,q)
=
a_0(q)+a_1(q)s+a_2(q)s^2+\cdots,
\]

则

\[
a_0(q)=g_0^\mu,
\]

并且

\[
\boxed{
B(y,\eta)
=
g_0^\mu s^\mu
+
a_1(q)s^{\mu+1}
+
a_2(q)s^{\mu+2}
+\cdots .
}
\]

这里的高阶项不是一般大小的 \(s^{\mu+r}\)，而是继承了 DE 端点平坦性带来的小系数。

---

## 5. 为什么高阶项也带有小因子

将

\[
L(s,q)=g_0^\mu N(s,q)
\]

归一化，则

\[
N(0,q)=1.
\]

于是

\[
B(y,\eta)
=
g_0^\mu s^\mu N(s,q).
\]

如果 \(N(s,q)\) 在某个复邻域 \(|s|\le r_0\) 中关于 \(s\) 解析，并且存在常数 \(A\) 使得

\[
|N(s,q)|\le A,
\qquad
|s|\le r_0,\quad 0\le q\le1,
\]

那么由 Cauchy 估计，

\[
N(s,q)=\sum_{r=0}^{\infty}c_r(q)s^r,
\qquad
|c_r(q)|\le A r_0^{-r}.
\]

于是

\[
B(y,\eta)
=
g_0^\mu
\sum_{r=0}^{\infty}c_r(q)s^{\mu+r}.
\]

这说明所有端点幂型项都有共同的小因子

\[
\boxed{
g_0^\mu=(\phi'(-1))^\mu.
}
\]

对 DE 变换来说，这是关键因素。

---

## 6. 端点代数尾项的统一估计

对

\[
s^\alpha
=
\left(\frac{1+y}{2}\right)^\alpha,
\qquad
\alpha>0,
\]

它的 Chebyshev 系数满足

\[
|\widehat{s^\alpha}_n|
=
\frac{
2^{1-2\alpha}\Gamma(2\alpha+1)|\sin(\pi\alpha)|
}{\pi}
\frac{\Gamma(n-\alpha)}{\Gamma(n+\alpha+1)},
\qquad n\ge1,
\]

当 \(n\to\infty\) 时，

\[
|\widehat{s^\alpha}_n|
=
O(n^{-2\alpha-1}).
\]

因此 Chebyshev 尾和满足

\[
\sum_{n\ge d}|\widehat{s^\alpha}_n|
\le
C_\alpha d^{-2\alpha}.
\]

应用到

\[
B(y,\eta)
=
g_0^\mu
\sum_{r=0}^{\infty}c_r(q)s^{\mu+r},
\]

得到端点代数尾项的估计

\[
E_d^{\rm end}
\le
g_0^\mu
\sum_{r=0}^{\infty}
\sup_q |c_r(q)|\,C_{\mu+r}d^{-2(\mu+r)}.
\]

将 \(C_{\mu+r}\) 的温和增长吸收到常数中，存在常数 \(C_\mu,A,\Lambda\)，使得当 \(d^2>\Lambda\) 时

\[
\boxed{
E_d^{\rm end}
\le
C_\mu A g_0^\mu
d^{-2\mu}
\frac{1}{1-\Lambda/d^2}.
}
\]

于是由带外元素估计，

\[
\boxed{
|A_{\ell k}^{\rm end}|
\le
\frac{2^{\mu+1}}{\mu}
C_\mu A
g_0^\mu
d^{-2\mu}
\frac{1}{1-\Lambda/d^2},
\qquad
d=\ell-k .
}
\]

这个估计比单独写某个 \(d^{-2\mu-2}\) 项更合理，因为它保留了所有端点项共同的极小因子 \(g_0^\mu\)。

---

## 7. 解析内部部分的几何衰减

端点幂型展开解释了左端点 \(s=0\) 的代数尾项。另一方面，去掉端点奇性后，剩余部分在实区间附近解析，因此其 Chebyshev 尾项具有几何衰减。

具体地，若某个剩余部分在统一的 Bernstein 椭圆 \(E_\rho\) 中解析，并满足

\[
\sup_{q\in[0,1]}\sup_{z\in E_\rho}
|F(z,q)|
\le M_\rho,
\qquad
\rho>1,
\]

则 Chebyshev 尾项满足

\[
E_d^{\rm ana}
\le
\frac{2M_\rho}{\rho-1}\rho^{-d}.
\]

因此对应矩阵带外元素满足

\[
\boxed{
|A_{\ell k}^{\rm ana}|
\le
C_{\rm ana}\rho^{-d}.
}
\]

---

## 8. 合并估计

综合端点代数尾项与解析内部项，得到

\[
\boxed{
|A_{\ell k}|
\le
C_{\rm end}
g_0^\mu
d^{-2\mu}
\frac{1}{1-\Lambda/d^2}
+
C_{\rm ana}\rho^{-d},
\qquad
d=\ell-k,\quad d^2>\Lambda.
}
\]

其中

\[
g_0=\phi'(-1),
\]

而 \(C_{\rm end},C_{\rm ana},\Lambda,\rho\) 与 \(\ell,k\) 无关。

这就是理论上解释“带外元素足够小”的主要上界。

---

## 9. DE 变换下 \(g_0^\mu\) 极小

当前 DE 变换为

\[
\phi(y)
=
\tanh\!\left(
\frac{\pi}{2}\sinh(\omega y)
\right).
\]

其导数为

\[
\phi'(y)
=
\frac{\pi\omega}{2}
\cosh(\omega y)
\operatorname{sech}^2\!\left(
\frac{\pi}{2}\sinh(\omega y)
\right).
\]

因此

\[
g_0
=
\phi'(-1)
=
\frac{\pi\omega}{2}
\cosh(\omega)
\operatorname{sech}^2\!\left(
\frac{\pi}{2}\sinh\omega
\right).
\]

当 \(\omega\) 较大时，

\[
g_0
\approx
2\pi\omega\cosh(\omega)
\exp(-\pi\sinh\omega),
\]

从而

\[
g_0^\mu
\approx
\left(2\pi\omega\cosh\omega\right)^\mu
\exp(-\mu\pi\sinh\omega).
\]

这是一个随 \(\omega\) 双指数变小的量。

当 \(\omega=4\) 时，

\[
g_0
\approx
4.01\times10^{-35}.
\]

若 \(\mu=1/2\)，则

\[
g_0^\mu
\approx
6.33\times10^{-18}.
\]

这已经低于或接近双精度机器精度

\[
\varepsilon_{\rm mach}\approx2.22\times10^{-16}.
\]

因此端点代数尾项从一开始就非常小。

---

## 10. 为什么 \(d\approx100\) 时自然低于机器精度

令

\[
d=\ell-k.
\]

若 \(\mu=1/2,\omega=4\)，端点项满足粗略估计

\[
|A_{\ell k}^{\rm end}|
\lesssim
C_{\rm end}
(6.33\times10^{-18})
d^{-1}
\frac{1}{1-\Lambda/d^2}.
\]

当 \(d\) 达到几十或一百时，只要 \(\Lambda/d^2\) 不接近 \(1\)，该项就明显低于 \(10^{-16}\)。

解析项满足

\[
|A_{\ell k}^{\rm ana}|
\le
C_{\rm ana}\rho^{-d}.
\]

例如即便取一个不太大的椭圆参数 \(\rho=1.5\)，也有

\[
1.5^{-100}\approx 2.5\times10^{-18}.
\]

因此当 \(d\) 在 \(100\) 左右时，解析项也通常低于机器精度。这与当前使用 `Csize_y=100` 时观察到的数值带宽是一致的。

---

## 11. 与有限 Chebyshev 核展开的关系

如果在实现中直接把 \(\widetilde K_\phi(y,\eta)\) 的 \(y\) 向 Chebyshev 模态截断到 \(i\le m\)，那么由于

\[
T_k(\tau(y,\eta))
\]

关于 \(y\) 的次数不超过 \(k\)，中间核乘积的最高 \(y\) 次数不超过 \(k+m\)。再乘外部因子 \((1+y)/2\) 后，最高次数至多为 \(k+m+1\)。

因此有限展开矩阵严格满足

\[
A_{\ell k}^{(m)}=0,
\qquad
\ell>k+m+1.
\]

如果当前实现取 `Csize_y=100`，即 \(m=99\)，那么有限展开模型中严格有

\[
A_{\ell k}^{(m)}=0,
\qquad
\ell>k+100.
\]

真实算子并不严格有限带宽，但上面的尾项估计说明，真实矩阵与该有限带宽矩阵在带外的差异低于机器精度。因此数值上看到的是上带状结构。

---

## 12. 对多项式核的补充说明

若 \(K(x,t)\) 是多项式，则

\[
B_K(y,\eta)
=
s^\mu L(s,q)K(\phi(-1+2s),\phi(-1+2sq)).
\]

由于 \(K(\phi(-1+2s),\phi(-1+2sq))\) 关于 \(s,q\) 仍然光滑，端点结构仍是

\[
B_K(y,\eta)
=
s^\mu L_K(s,q).
\]

如果

\[
K(-1,-1)\ne0,
\]

则主导项变为

\[
(\phi'(-1))^\mu K(-1,-1)s^\mu.
\]

因此带外估计的主导项只多出一个因子 \(K(-1,-1)\)。如果 \(K\) 在 \((-1,-1)\) 有消失阶 \(r_*\)，则主导端点幂次变为

\[
s^{\mu+r_*},
\]

带外衰减进一步增强为

\[
d^{-2(\mu+r_*)}.
\]

所以多项式核不会破坏数值上带宽；它只改变常数和端点消失阶。

---

## 13. 最终结论

对 \(K=1\)，当前 DE 弱奇异 Volterra 系数矩阵的带外元素之所以足够小，原因可以概括为三点：

1. **带外矩阵元只由最终核 \(B(y,\eta)\) 的 \(y\) 向 Chebyshev 尾项产生。**

   \[
   |A_{\ell k}|
   \le
   \frac{2^{\mu+1}}{\mu}
   E_{\ell-k-1}.
   \]

2. **\(K=1\) 时最终核具有端点幂型结构。**

   \[
   B(y,\eta)
   =
   (\phi'(-1))^\mu
   \left(\frac{1+y}{2}\right)^\mu
   N\!\left(\frac{1+y}{2},\frac{1+\eta}{2}\right).
   \]

   其中 \(N(0,q)=1\)，并且 \(N\) 在实区间附近解析。

3. **DE 变换使 \((\phi'(-1))^\mu\) 极小。**

   对 \(\omega=4,\mu=1/2\)，

   \[
   (\phi'(-1))^\mu\approx6.33\times10^{-18}.
   \]

   因此端点代数尾项的系数已经接近机器精度，而剩余解析部分随 \(d=\ell-k\) 几何衰减。

最终可以用如下统一上界描述：

\[
\boxed{
|A_{\ell k}|
\le
C_{\rm end}
(\phi'(-1))^\mu
(\ell-k)^{-2\mu}
\frac{1}{1-\Lambda/(\ell-k)^2}
+
C_{\rm ana}\rho^{-(\ell-k)}.
}
\]

这说明当行列差 \(\ell-k\) 达到足够大时，带外元素必然低于机器精度。因此 \(K=1\) 时的系数矩阵在数值上具有上带宽。
