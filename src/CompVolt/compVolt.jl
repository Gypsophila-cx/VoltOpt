include("../cheb2vals.jl")
include("../vals2cheb.jl")
include("multrestrict.jl")
using LinearAlgebra

function compVolt(K, cf)
    c = cf .* (-1).^(0:length(cf)-1)      # Adjust signs for Chebyshev coefficients

    N = length(c) - 1
    if isodd(N)
        N += 1  # Ensure N is even for Clenshaw-Curtis weights
        c = [c; 0.0]  # Pad an extra zero if needed
    end

    C = multrestrict(c)                 # Restrict coefficients
    V_f = cheb2vals_2d_fft(C)             # Convert to values on Chebyshev grid

    x = cos.(pi*(N:-1:0)/N)                 # Chebyshev points
    y = x

    KK(x,y) = K(x, (x+1)/2*y+(x-1)/2)   # modified kernel on square
    V_K = KK.(x', y)                     # Kernel values on Chebyshev grid
    V = V_K .* V_f                      # Element-wise multiplication

    w = clenshaw_curtis_analytical_weights(N)  # 计算 Clenshaw-Curtis 权重
    v = V'*w
    Fval = (x.+1)/2 .*v
    cF = vals2cheb(Fval)

    cF = cF .* (-1).^(0:length(cF)-1) 

    return cF
end

function compVolt_fast(K, cf)
    len_cf = length(cf)
    N = len_cf - 1
    
    # 提前判断是否需要补零，计算最终的 N
    N_even = isodd(N) ? N + 1 : N
    
    # 1. 预分配数组，替代昂贵的 (-1)^n 和 [c; 0.0] 拼接
    c = zeros(eltype(cf), N_even + 1)
    @inbounds for i in 1:len_cf
        # 奇数索引（1, 3, 5...对应 0, 2, 4次幂）保持原号，偶数索引变号
        c[i] = iseven(i) ? -cf[i] : cf[i]
    end

    # 外部调用（确保这两个函数也经过了优化）
    C = multrestrict(c)
    V_f = cheb2vals_2d_fft(C) 
    
    # 2. 计算 Chebyshev 点 (使用推导式避免广播带来的额外分配)
    x = [cos(pi * i / N_even) for i in N_even:-1:0]
    
    # 3. 核心优化：直接在循环中融合 Kernel 计算和乘法，消除中间矩阵 V_K
    # 注意：在原版 KK.(x', y) 中，列 j 对应 x' (即 x[j])，行 i 对应 y (即 x[i])
    V = Matrix{eltype(V_f)}(undef, N_even + 1, N_even + 1)
    @inbounds for j in 1:(N_even + 1)
        xj = x[j] # 原版中作为第一参数
        for i in 1:(N_even + 1)
            yi = x[i] # 原版中作为第二参数
            
            # 计算 modified kernel (修复了 x 和 y 传入的顺序)
            KK_val = K(xj, (xj + 1) / 2 * yi + (xj - 1) / 2)
            
            # 直接乘入 V 中
            V[i, j] = KK_val * V_f[i, j]
        end
    end

    # 4. 计算权重
    w = clenshaw_curtis_analytical_weights(N_even)
    
    # 5. 使用 adjoint (V') 替代 transpose，保证如果涉及复数时与原版严格一致
    # V' 在 Julia 中同样是惰性包装的 Adjoint 类型，不会分配新内存
    v = V' * w 
    
    # 6. 使用 @. 宏来融合所有的 element-wise 操作，只产生一次内存分配
    Fval = @. (x + 1) / 2 * v

    cF = vals2cheb(Fval)
    cF = cF .* (-1).^(0:length(cF)-1)
    
    return cF
end

function compVolt_matrix(K, CFs::AbstractMatrix)
    isempty(CFs) && return CFs
    
    L, M = size(CFs)
    N = L - 1
    N_even = isodd(N) ? N + 1 : N
    
    # 1. 预计算节点与权重
    x = [cos(pi * i / N_even) for i in N_even:-1:0]
    w = clenshaw_curtis_analytical_weights(N_even)
    
    # 2. 预计算“融合权重”的 Kernel 矩阵 (W)
    # 这一步直接将 w[i] 乘入 Kernel 中，省去了后续循环中 M*(N+1)^2 次乘法
    T_W = typeof(conj(K(x[1], x[1])) * w[1])
    W = Matrix{T_W}(undef, N_even + 1, N_even + 1)
    @inbounds for j in 1:(N_even + 1)
        xj = x[j]
        for i in 1:(N_even + 1)
            yi = x[i]
            KK_val = K(xj, (xj + 1) / 2 * yi + (xj - 1) / 2)
            W[i, j] = conj(KK_val) * w[i]
        end
    end

    # 3. 预分配工作区缓存 (所有列复用这几个向量，实现零额外分配)
    c_col = zeros(eltype(CFs), N_even + 1)
    v_col = zeros(T_W, N_even + 1)
    Fval  = zeros(T_W, N_even + 1)
    
    # ---------------------------------------------------------
    # 内部匿名函数：处理单列数据的核心逻辑
    # ---------------------------------------------------------
    function process_column!(m)
        fill!(c_col, zero(eltype(c_col)))
        @inbounds for i in 1:L
            # 直接标量索引提取矩阵元素，比切片 CFs[:, m] 速度更快且不分配内存
            c_col[i] = iseven(i) ? -CFs[i, m] : CFs[i, m]
        end
        
        # 外部转换 (假设这两个函数只能处理 1D 输入，生成 2D 输出 V_f)
        C = multrestrict(c_col)
        V_f = cheb2vals_2d_fft(C)
        
        # 核心积分：v_col = diag(W' * V_f) 的等价展开
        @inbounds for j in 1:(N_even + 1)
            sum_val = zero(T_W)
            for i in 1:(N_even + 1)
                sum_val += W[i, j] * conj(V_f[i, j])
            end
            v_col[j] = sum_val
            Fval[j] = (x[j] + 1) / 2 * v_col[j]
        end
        
        return vals2cheb(Fval)
    end
    # ---------------------------------------------------------

    # 4. 首列试探与结果矩阵预分配
    # 因为我们不确定外部函数 vals2cheb 确切的返回类型和长度
    # 我们先完整跑完第 1 列，获取类型与维度，从而精准分配 Results 矩阵
    cF_first = process_column!(1)
    Results = Matrix{eltype(cF_first)}(undef, length(cF_first), M)
    Results[:, 1] .= cF_first
    
    # 5. 高效循环处理剩余的列
    for m in 2:M
        Results[:, m] .= process_column!(m)
    end
    
    return Results
end

function compVolt_matrix_ultra(K, CFs::AbstractMatrix)
    isempty(CFs) && return CFs
    
    L, M = size(CFs)
    N = L - 1
    N_even = isodd(N) ? N + 1 : N
    
    # 1. 预计算节点、权重及缩放因子
    x = [cos(pi * i / N_even) for i in N_even:-1:0]
    w = clenshaw_curtis_analytical_weights(N_even)
    
    # 【优化1】预计算 x 的仿射变换缩放因子，消除两处热循环中的冗余加法与除法
    x_scale = [(xj + 1) / 2 for xj in x] 
    
    # 2. 预计算“融合权重”的 Kernel 矩阵 (W)
    T_W = typeof(conj(K(x[1], x[1])) * w[1])
    W = Matrix{T_W}(undef, N_even + 1, N_even + 1)
    @inbounds for j in 1:(N_even + 1)
        xj = x[j]
        scale_j = x_scale[j]
        for i in 1:(N_even + 1)
            yi = x[i]
            # 这里的数学等价于 (xj + 1)/2 * yi + (xj - 1)/2
            KK_val = K(xj, scale_j * yi + (xj - 1) / 2) 
            W[i, j] = conj(KK_val) * w[i]
        end
    end

    # ---------------------------------------------------------
    # 【优化2】重构单列处理逻辑，解耦缓存以支持并发，并引入 SIMD
    # ---------------------------------------------------------
    function process_column(m)
        # 为每个任务分配局部内存。只要内部的黑盒谱方法也在分配，这里的开销就微乎其微
        c_col = zeros(eltype(CFs), N_even + 1)
        @inbounds for i in 1:L
            c_col[i] = iseven(i) ? -CFs[i, m] : CFs[i, m]
        end
        
        C = multrestrict(c_col)
        V_f = cheb2vals_2d_fft(C)
        
        Fval = zeros(T_W, N_even + 1)
        @inbounds for j in 1:(N_even + 1)
            sum_val = zero(T_W)
            # 【优化3】强制启用 SIMD (单指令多数据流) 向量化内积操作
            @simd for i in 1:(N_even + 1)
                sum_val += W[i, j] * conj(V_f[i, j])
            end
            # 直接使用预计算的 x_scale
            Fval[j] = x_scale[j] * sum_val
        end
        
        return vals2cheb(Fval)
    end
    # ---------------------------------------------------------

    # 3. 首列试探与结果矩阵精准分配
    cF_first = process_column(1)
    Results = Matrix{eltype(cF_first)}(undef, length(cF_first), M)
    Results[:, 1] .= cF_first
    
    # 【优化4】多线程无锁并行处理剩余列
    # 注意：启动 Julia 时需设置线程数，例如 `julia -t auto` 或在 VSCode 设置中调整
    Threads.@threads :static for m in 2:M
        Results[:, m] .= process_column(m)
    end
    
    return Results
end

function clenshaw_curtis_fft_weights(N::Int)
    @assert iseven(N) "为保证算法对称性简介，此函数要求 N 为偶数"
    
    # 1. 构造频域系数向量 A
    A = zeros(Float64, N)
    
    # 填充 1 到 N/2-1 的系数，并利用实偶对称性镜像填充到后半段
    for j in 1:(div(N, 2) - 1)
        A[j + 1] = 1.0 / (4 * j^2 - 1)
        A[N - j + 1] = A[j + 1] 
    end
    # 填充中心对称点 N/2 的系数
    A[div(N, 2) + 1] = 1.0 / (N^2 - 1)
    
    # 2. 核心魔法：通过 IFFT 进行极速并行求和
    # Julia 的 ifft 默认带有 1/N 的缩放因子，为了得到纯粹的求和结果，我们需要乘回 N
    S = N .* real(ifft(A))
    
    # 3. 组装最终的权重 w
    w = zeros(Float64, N + 1)
    for k in 0:(N - 1)
        c_k = (k == 0) ? 1.0 : 2.0
        w[k + 1] = (c_k / N) * (1.0 - S[k + 1])
    end
    w[N + 1] = w[1] # 根据对称性，右端点权重恒等于左端点
    
    return w
end

function clenshaw_curtis_analytical_weights(N::Int)
    w = zeros(N + 1)
    for k in 0:N
        c_k = (k == 0 || k == N) ? 1.0 : 2.0
        sum_val = 0.0
        for j in 1:floor(Int, N/2)
            b_j = (j == N/2) ? 1.0 : 2.0
            sum_val += (b_j / (4j^2 - 1)) * cos(2 * j * k * pi / N)
        end
        w[k+1] = (c_k / N) * (1 - sum_val)
    end
    return w
end
# ============================================================================
# Matrix-free nonlinear Volterra and Newton--GMRES extensions
# ============================================================================
# This block is appended to the original compVolt.jl without changing the
# original functions above.
#
# Main use cases:
#
#   1. Nonlinear Volterra action
#
#          F(x) = ∫_{-1}^x K(x,y,u(y)) dy.
#
#      Use:
#
#          cF = compVolt_nonlinear_fast(K, cu)
#
#   2. Matrix-free Newton Jacobian-vector product for
#
#          R[u](x) = u(x) - g(x) - λ∫_{-1}^x K(x,y,u(y)) dy.
#
#      If Ku = ∂K/∂u, then
#
#          J[u]v = v - λ∫_{-1}^x Ku(x,y,u(y))v(y) dy.
#
#      The efficient GMRES workflow is:
#
#          cache = CompVoltNewtonGMRESCache(Ku, cu; λ=λ)
#          compVolt_newton_jv!(out, cache, cv)
#
#      During a fixed Newton step, cache stores the restricted values of u and
#      the weighted kernel derivative
#
#          WKu[i,j] = w[i] * Ku(x[j], y_ij, u(y_ij)),
#
#      so each GMRES matvec only recomputes the restricted values of v and then
#      performs a fused weighted summation.
# ============================================================================

# ----------------------------
# Small internal utilities
# ----------------------------

_cv_even_degree(N::Integer) = isodd(N) ? Int(N) + 1 : Int(N)

function _cv_check_grid_degree(N::Int)
    N >= 1 || error("The Chebyshev degree must be at least 1.")
    iseven(N) || error("The Chebyshev degree used for Clenshaw--Curtis must be even.")
    return nothing
end

function _cv_signpad!(dst::AbstractVector, src::AbstractVector)
    length(dst) >= length(src) || error("Destination buffer is shorter than source coefficients.")
    fill!(dst, zero(eltype(dst)))
    @inbounds for i in eachindex(src)
        # Julia index i corresponds to Chebyshev mode i-1.
        # The original code uses cf .* (-1).^(0:length(cf)-1).
        dst[i] = iseven(i) ? -src[i] : src[i]
    end
    return dst
end

function _cv_signflip_back!(c::AbstractVector)
    @inbounds for i in eachindex(c)
        if iseven(i)
            c[i] = -c[i]
        end
    end
    return c
end

# Compute the two-dimensional restricted values
#
#     V[i,j] = f((1+x[j])*(1+x[i])/2 - 1),
#
# where x[j] is the outer Volterra endpoint and x[i] is the inner
# Clenshaw--Curtis variable.
function _cv_restricted_vals_from_pad(cpad::AbstractVector)
    C = multrestrict(cpad)
    return cheb2vals_2d_fft(C)
end

# ----------------------------
# Reusable grid data
# ----------------------------

struct CompVoltGrid{TX<:AbstractVector,TW<:AbstractVector}
    N::Int
    x::TX
    w::TW
    scale::TX
    shift::TX
end

function CompVoltGrid(N::Int)
    N_even = _cv_even_degree(N)
    _cv_check_grid_degree(N_even)

    x = [cos(pi * i / N_even) for i in N_even:-1:0]
    w = clenshaw_curtis_analytical_weights(N_even)

    scale = similar(x)
    shift = similar(x)
    @inbounds for j in eachindex(x)
        scale[j] = (x[j] + 1) / 2
        shift[j] = (x[j] - 1) / 2
    end

    return CompVoltGrid(N_even, x, w, scale, shift)
end

function _cv_choose_grid(cf::AbstractVector, Nquad::Union{Nothing,Int}, grid::Union{Nothing,CompVoltGrid})
    if grid !== nothing
        length(cf) - 1 <= grid.N || error("grid.N must be at least length(cf)-1.")
        return grid
    end

    N0 = length(cf) - 1
    Nq = Nquad === nothing ? N0 : Nquad
    Nq = _cv_even_degree(Nq)
    Nq >= N0 || error("Nquad must be at least length(cf)-1.")
    return CompVoltGrid(Nq)
end

# ----------------------------
# Nonlinear Volterra action
# ----------------------------

function compVolt_nonlinear_fast(
    K,
    cu::AbstractVector;
    Nquad::Union{Nothing,Int}=nothing,
    grid::Union{Nothing,CompVoltGrid}=nothing,
    truncate_output::Bool=false,
    include_u_factor::Bool=false,
)
    isempty(cu) && return similar(cu, 0)

    g = _cv_choose_grid(cu, Nquad, grid)
    N = g.N

    cpad = zeros(eltype(cu), N + 1)
    _cv_signpad!(cpad, cu)
    Uvals = _cv_restricted_vals_from_pad(cpad)

    # Use x = 1, η = -1 as a safe type sample.
    sample_x = g.x[end]
    sample_y = g.x[1]
    sample_u = Uvals[1, end]
    sample = K(sample_x, sample_y, sample_u)
    if include_u_factor
        sample *= sample_u
    end

    T = typeof(sample * g.w[1] * g.scale[end])
    Fval = zeros(T, N + 1)

    @inbounds for j in 1:(N + 1)
        # x[1] = -1, so the Volterra interval has zero length.
        if j == 1
            Fval[j] = zero(T)
            continue
        end

        xj = g.x[j]
        sc = g.scale[j]
        sh = g.shift[j]

        s = zero(T)
        @simd for i in 1:(N + 1)
            yij = sc * g.x[i] + sh
            uij = Uvals[i, j]
            val = K(xj, yij, uij)
            if include_u_factor
                val *= uij
            end
            s += g.w[i] * val
        end
        Fval[j] = sc * s
    end

    cF = vals2cheb(Fval)
    _cv_signflip_back!(cF)

    return truncate_output ? cF[1:length(cu)] : cF
end

# Backward-compatible alias with the name used in previous discussion.
function compVolt_nonlinear(args...; kwargs...)
    return compVolt_nonlinear_fast(args...; kwargs...)
end

# ----------------------------
# Newton--GMRES cache
# ----------------------------

mutable struct CompVoltNewtonGMRESCache{TK,TG,TL,TCU,TCV,TU,TW,TF}
    Ku::TK
    grid::TG
    λ::TL
    cu_pad::TCU
    cv_pad::TCV
    Uvals::TU
    WKu::TW
    Fval::TF
end

function _cv_fill_weighted_Ku!(WKu::AbstractMatrix, Ku, g::CompVoltGrid, Uvals::AbstractMatrix)
    N = g.N
    size(WKu, 1) == N + 1 && size(WKu, 2) == N + 1 || error("WKu has incompatible size.")
    size(Uvals, 1) == N + 1 && size(Uvals, 2) == N + 1 || error("Uvals has incompatible size.")

    @inbounds for j in 1:(N + 1)
        xj = g.x[j]
        sc = g.scale[j]
        sh = g.shift[j]
        for i in 1:(N + 1)
            yij = sc * g.x[i] + sh
            WKu[i, j] = g.w[i] * Ku(xj, yij, Uvals[i, j])
        end
    end
    return WKu
end

function CompVoltNewtonGMRESCache(
    Ku,
    cu::AbstractVector;
    v_template::AbstractVector=cu,
    λ=1.0,
    Nquad::Union{Nothing,Int}=nothing,
    grid::Union{Nothing,CompVoltGrid}=nothing,
)
    isempty(cu) && error("cu must be nonempty.")
    length(v_template) == length(cu) || error("v_template and cu must have the same length.")

    g = _cv_choose_grid(cu, Nquad, grid)
    N = g.N

    cu_pad = zeros(eltype(cu), N + 1)
    cv_pad = zeros(eltype(v_template), N + 1)

    _cv_signpad!(cu_pad, cu)
    Uvals = _cv_restricted_vals_from_pad(cu_pad)

    # Type sampling. x=end is x=1, i=1 is y=-1.
    sample_WKu = g.w[1] * Ku(g.x[end], g.x[1], Uvals[1, end])
    WKu = Matrix{typeof(sample_WKu)}(undef, N + 1, N + 1)
    _cv_fill_weighted_Ku!(WKu, Ku, g, Uvals)

    # Fval stores values of ∫ Ku(x,y,u(y))v(y)dy before vals2cheb.
    # Determine its type using the same restriction pipeline as a real GMRES
    # vector. This is slightly more expensive at construction time, but avoids
    # type surprises if cheb2vals_2d_fft returns complex values.
    _cv_signpad!(cv_pad, v_template)
    Vsample = _cv_restricted_vals_from_pad(cv_pad)
    sample_F = g.scale[end] * WKu[1, end] * Vsample[1, end]
    Fval = zeros(typeof(sample_F), N + 1)

    return CompVoltNewtonGMRESCache(Ku, g, λ, cu_pad, cv_pad, Uvals, WKu, Fval)
end

# Refresh the cache when the Newton iterate u changes. This is called once per
# Newton step, not once per GMRES iteration.
function update_newton_cache!(cache::CompVoltNewtonGMRESCache, cu::AbstractVector)
    length(cu) <= length(cache.cu_pad) || error("cu is longer than the cache degree.")

    _cv_signpad!(cache.cu_pad, cu)
    cache.Uvals = _cv_restricted_vals_from_pad(cache.cu_pad)
    _cv_fill_weighted_Ku!(cache.WKu, cache.Ku, cache.grid, cache.Uvals)

    return cache
end

# Compute only the linearized Volterra integral
#
#     out ≈ coefficients of ∫_{-1}^x Ku(x,y,u(y))v(y)dy.
#
# The output is truncated to length(out), which is usually length(cv) in GMRES.
function compVolt_linearized_integral!(
    out::AbstractVector,
    cache::CompVoltNewtonGMRESCache,
    cv::AbstractVector,
)
    length(cv) <= length(cache.cv_pad) || error("cv is longer than the cache degree.")
    length(out) <= length(cache.Fval) || error("out is longer than the cache degree.")

    g = cache.grid
    N = g.N

    _cv_signpad!(cache.cv_pad, cv)
    Vvals = _cv_restricted_vals_from_pad(cache.cv_pad)

    Fval = cache.Fval

    @inbounds for j in 1:(N + 1)
        if j == 1
            Fval[j] = zero(eltype(Fval))
            continue
        end

        s = zero(eltype(Fval))
        @simd for i in 1:(N + 1)
            s += cache.WKu[i, j] * Vvals[i, j]
        end
        Fval[j] = g.scale[j] * s
    end

    cF = vals2cheb(Fval)
    _cv_signflip_back!(cF)

    @inbounds for i in eachindex(out)
        out[i] = cF[i]
    end

    return out
end

# Matrix-free Newton Jacobian-vector product for
#
#     R[u](x) = u(x) - g(x) - λ∫_{-1}^x K(x,y,u(y))dy.
#
# Hence
#
#     J[u]v = v - λ∫_{-1}^x Ku(x,y,u(y))v(y)dy.
#
# This is the function to pass to GMRES.
function compVolt_newton_jv!(
    out::AbstractVector,
    cache::CompVoltNewtonGMRESCache,
    cv::AbstractVector,
)
    length(out) == length(cv) || error("out and cv must have the same length.")

    compVolt_linearized_integral!(out, cache, cv)

    λ = cache.λ
    @inbounds for i in eachindex(out)
        out[i] = cv[i] - λ * out[i]
    end

    return out
end

# Convenience constructor returning a GMRES-compatible closure and its cache.
# Example:
#
#     Jv!, cache = compVolt_newton_jv_operator(Ku, cu; λ=λ)
#     Jv!(out, v)
#
# When u changes in the next Newton step, call
#
#     update_newton_cache!(cache, cu_new)
#
# and reuse the same Jv! closure.
function compVolt_newton_jv_operator(
    Ku,
    cu::AbstractVector;
    v_template::AbstractVector=cu,
    λ=1.0,
    Nquad::Union{Nothing,Int}=nothing,
    grid::Union{Nothing,CompVoltGrid}=nothing,
)
    cache = CompVoltNewtonGMRESCache(
        Ku,
        cu;
        v_template=v_template,
        λ=λ,
        Nquad=Nquad,
        grid=grid,
    )

    function Jv!(out::AbstractVector, v::AbstractVector)
        return compVolt_newton_jv!(out, cache, v)
    end

    return Jv!, cache
end

# ----------------------------
# Optional: residual helper
# ----------------------------

# Residual coefficients for
#
#     R[u] = u - g - λ*V_K[u],
#
# where V_K[u](x) = ∫_{-1}^x K(x,y,u(y))dy.
function compVolt_nonlinear_residual(
    K,
    cu::AbstractVector,
    cg::AbstractVector;
    λ=1.0,
    Nquad::Union{Nothing,Int}=nothing,
    grid::Union{Nothing,CompVoltGrid}=nothing,
)
    length(cu) == length(cg) || error("cu and cg must have the same length.")

    Vcu = compVolt_nonlinear_fast(
        K,
        cu;
        Nquad=Nquad,
        grid=grid,
        truncate_output=true,
    )

    T = promote_type(eltype(cu), eltype(cg), eltype(Vcu))
    R = Vector{T}(undef, length(cu))
    @inbounds for i in eachindex(cu)
        R[i] = cu[i] - cg[i] - λ * Vcu[i]
    end
    return R
end
