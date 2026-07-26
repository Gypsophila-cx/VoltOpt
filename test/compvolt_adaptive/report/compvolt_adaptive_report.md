# compVolt 自适应求值报告

## 1. 接口与网格语义

主要接口为：

```julia
compVolt_nonlinear_fast(K, cu;
    M=nothing,
    adaptive=isnothing(M) && isnothing(Nquad) && isnothing(grid),
    M0=nothing,
    tol=nothing,
    Mmax=nothing,
    max_refinements=8,
    return_info=false,
    truncate_output=false,
    include_u_factor=false,
    Nquad=nothing,
    grid=nothing,
)
```

`compVolt_nonlinear` 是具有相同关键字和返回语义的公开兼容别名。输入系数长度为
`L`，输入多项式次数为 `n=L-1`。`M` 是 Chebyshev--Lobatto 网格次数，每个
方向使用 `M+1` 个点。自适应加密为 `M -> 2M`，点数相应为
`M+1 -> 2M+1`；细网格 Julia 下标 `1:2:end` 是公共粗网格节点。

显式 `M`、`Nquad` 或 `grid` 选择固定网格。无这些关键字的默认调用现在自动
加密，默认完整输出长度为接受的 `M+1`，可能长于输入。`truncate_output=true`
仅将输出切片到 `length(cu)`，用于需要固定向量长度的外部迭代；它不是基于
容差的 chopping。非线性复合的有效高阶模态默认会保留。

## 2. 自适应判据与限制

在粗次数 `M` 和细次数 `2M` 上，使用

```text
eta = max_j |vM[j] - v2M[2j-1]| / max(1, max_j |v2M[j]|)
theta = max |c2M[M+2:2M+1]| / max(1, max_j |c2M[j]|)
```

其中 `theta` 的 Julia 下标 `M+2:2M+1` 对应数学模态 `M+1:2M`。两者均不大于
`tol` 时接受细网格。默认 `tol=100eps(T)`，`T` 由输入系数类型决定。
`max_refinements` 是次数加倍的最大次数；二维矩阵规模每次约增至四倍。默认
`Mmax=nothing`，大规模任务建议显式指定。达到限制时默认抛出异常；
`return_info=true` 则返回最细完成层级并令 `converged=false`。

返回信息字段为：`coefficients`、`M`、`eta`、`theta`、`refinements` 和
`converged`。固定模式的 `eta`、`theta` 为 `nothing`，`refinements=0`。

## 3. 非有限值诊断

实现会依次检查二维限制点值矩阵、每个核函数值、可选乘积后的被积函数值、
Clenshaw--Curtis 逐行和及节点积分值、DCT 输出系数，以及最终 `eta`、
`theta`。发现 `NaN` 或 `Inf` 时立即抛出错误，消息包含数值阶段、当前 `M`、
固定/自适应模式以及粗/细层级。测试验证了固定 `M=4` 的 `NaN` 和自适应细
网格 `M=4` 新增节点上的 `Inf` 均在核求值阶段提前失败。

## 4. Benchmark 字段和计时

`data/adaptive_oscillation.csv` 包含：

```text
case, omega, N, input_length, M0, accepted_M, accepted_npts, Mref,
eta, theta, refinements, converged, tol, Mmax, evaluation_time,
time_min, time_q25, time_median, samples_run, error_inf, relative_error
```

其中 `N` 明确为输入多项式次数，`input_length=N+1`，
`accepted_npts=accepted_M+1`。每个案例先执行一次完整预热，再由
BenchmarkTools 运行 7 个 `evals=1` 样本；`evaluation_time` 等于最小时间，
同时保留 lower quartile 和 median。计时闭包只调用自适应求值函数，不包含
输入系数构造、参考解或误差计算。

原固定网格 Clenshaw/compVolt 对比 benchmark 及其 CSV、MATLAB 脚本和图形已
原样迁移到同一目录；其数学问题、固定 `CompVoltGrid(N)` 调用和输出字段均未
改变。

## 5. 振荡实验

固定输入为 `u(x)=exp(x)`，在 17 点 Lobatto 网格上转换为固定长度 17 的
Chebyshev 系数，因此 `N=16`。核为

```text
K_omega(x,t,z) = cos(omega*(x-t))*z
```

频率为 `[0,5,10,20,40,80,160]`，`M0=16`、`tol=1e-10`、`Mmax=1024`、
`max_refinements=7`。固定参考次数为
`Mref=max(2*accepted_M,256)`。自适应和参考系数均在同一个 1025 点
Chebyshev--Lobatto 网格上用 Clenshaw 递推求值，并报告节点最大误差及相对
误差。

| omega | accepted_M | npts | refinements | eta | theta | Mref | error_inf | relative_error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 33 | 1 | 3.779e-16 | 1.182e-16 | 256 | 3.109e-15 | 1.323e-15 |
| 5 | 64 | 65 | 2 | 2.776e-16 | 5.318e-17 | 256 | 4.857e-16 | 3.364e-15 |
| 10 | 64 | 65 | 2 | 2.706e-16 | 4.059e-16 | 256 | 6.314e-16 | 1.076e-14 |
| 20 | 128 | 129 | 3 | 7.251e-16 | 3.784e-17 | 256 | 7.910e-16 | 3.174e-14 |
| 40 | 128 | 129 | 3 | 2.351e-14 | 9.091e-12 | 256 | 1.082e-15 | 1.005e-13 |
| 80 | 256 | 257 | 4 | 1.958e-15 | 9.986e-17 | 512 | 2.418e-15 | 4.822e-13 |
| 160 | 512 | 513 | 5 | 2.311e-15 | 7.285e-17 | 1024 | 3.499e-15 | 1.458e-12 |

全部案例收敛，接受时两个指标均低于容差。接受次数共有
`32,64,128,256,512` 五个层级；最高频的次数是最低频的 16 倍。因此实验
清楚观察到振荡增强时所需网格增长，同时所有参考误差仍远小于设定容差。

## 6. 运行与结果

实际运行：

```powershell
julia --project=. test\compvolt_adaptive\runtests.jl
julia --project=. test\runtests.jl
julia --project=. test\compvolt_adaptive\run_adaptive_oscillation.jl
matlab -batch "run('test/compvolt_adaptive/plot_adaptive_oscillation.m')"
matlab -batch "run('test/compvolt_adaptive/plot_nonlinear_bench.m')"
julia --project=. tools\audit_layout.jl
```

独立 compVolt 测试为 85/85 通过；总测试为 97/97 通过。振荡脚本生成 7 行
非空 CSV，无 `NaN`、`Inf` 或未收敛案例。MATLAB 成功生成两幅图的 PDF 和
PNG；仅报告了 `readtable` 将表头规范化为有效标识符的标准警告，不影响所需
字段读取。迁移后的原固定网格 benchmark 绘图脚本也成功重新生成四个既有
图形。布局审计为 0 failures（另有仓库原先缺少 `issue_list.md` 的 1 条警告）。

## 7. 未解决问题与性能瓶颈

未发现未解决的数值正确性问题。主要性能限制仍是二维点值矩阵的
`O(M^2)` 内存，以及当前解析 Clenshaw--Curtis 权重构造和逐点核求值的
`O(M^2)` 工作量；次数加倍会令这些成本约增长四倍。现有实现复用上一个细层
的积分值与系数，并且二维限制系数矩阵只构造一次，但没有实现偶奇块 DCT、
剪枝 DCT 或 NUFFT。
