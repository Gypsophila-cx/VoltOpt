using FFTW

# =======================================================================================
"""
    cheb2_vals2coeffs(V::AbstractMatrix{T}; tol=eps()) where T<:Real

给定二维 Chebyshev 极值网格上的点值矩阵 `V`，计算其二维 Chebyshev 系数矩阵 `C`。
通过截断尾部趋于零的高频系数，使得返回的矩阵尺寸 `n x m` 尽可能地小。
"""
function cheb2_vals2coeffs(V::AbstractMatrix{T}; tol=eps()) where T<:Real
    Nx, Ny = size(V)
    if Nx < 2 || Ny < 2
        error("输入矩阵的维度必须至少为 2x2。")
    end
    
    # 1. 直接应用 2D DCT-I 变换，无需提前修改 V
    C_raw = FFTW.r2r(V, FFTW.REDFT00)
    
    # 2. 全局归一化
    C = C_raw ./ ((Nx - 1) * (Ny - 1))
    
    # 3. 频域边界项调整：根据标准 Chebyshev 展开式，边界项系数需要减半
    C[1, :] ./= 2.0
    C[end, :] ./= 2.0
    C[:, 1] ./= 2.0
    C[:, end] ./= 2.0
    
    # 4. 矩阵自适应截断
    n_eff, m_eff = 1, 1
    
    # 寻找 x 方向最后一个有效系数所在的行
    for i in Nx:-1:1
        if maximum(abs, @view C[i, :]) > tol
            n_eff = i
            break
        end
    end
    
    # 寻找 y 方向最后一个有效系数所在的列
    for j in Ny:-1:1
        if maximum(abs, @view C[:, j]) > tol
            m_eff = j
            break
        end
    end
    
    # 返回截断后的紧凑系数矩阵
    return C[1:n_eff, 1:m_eff]
end

# =======================================================================================
"""
基于 FFTW 的自适应二维切比雪夫插值，返回核函数 K 的插值系数矩阵 C。
"""
function adaptive_cheb2_coeffmat(K::Function; tol=eps(), max_N=1024, initial_N=16)
    N = initial_N
    
    while N <= max_N
        # 1. 生成 Chebyshev-Lobatto 节点 (极值点)
        # 节点范围 [-1, 1]，总共 N 个点
        x = cos.( (0:N-1) .* (π / (N-1)) )
        y = x
        
        # 2. 在张量网格上评估二维核函数 K(x, y)
        V = [K(xi, yj) for xi in x, yj in y]
        
        # 3. 使用 FFTW 的二维 DCT-I (REDFT00) 计算切比雪夫系数
        C = FFTW.r2r(V, FFTW.REDFT00)
        
        # 4. 系数标准化
        # 抵消 FFTW REDFT00 变换自带的缩放因子
        C ./= (N - 1)^2
        
        # 调整边界系数以匹配标准切比雪夫多项式定义:
        # K(x,y) ≈ sum( C[i,j] * T_{i-1}(x) * T_{j-1}(y) )
        C[1, :] ./= 2; C[N, :] ./= 2
        C[:, 1] ./= 2; C[:, N] ./= 2
        
        # 5. 检查高频项的衰减以判断是否满足自适应容差
        max_c = maximum(abs, C)
        tail_error = max(maximum(abs, C[N-1:N, :]), maximum(abs, C[:, N-1:N]))
        
        if tail_error <= tol * max_c || tail_error <= eps()
            return truncate_coeffs_robust(C, tol)
        end
        
        # 精度不足则将网格节点翻倍
        N *= 2
    end
    
    @warn "已达到最大网格数 max_N=$(max_N)，系数高频衰减可能未完全满足容差tol=$(tol)的要求。"
    return C
end

"""
裁剪系数矩阵中的高频全零/噪声部分，动态缩小矩阵维度。
"""
function truncate_coeffs_robust(C::Matrix{Float64}, tol::Float64)
    max_c = maximum(abs, C)
    if max_c == 0 
        return C[1:1, 1:1] 
    end
    
    n, m = size(C)
    
    # 1. 提取每行和每列的无穷范数 (最大绝对值)
    # 使用 @view 避免不必要的内存分配
    row_norms = [maximum(abs, @view C[i, :]) for i in 1:n]
    col_norms = [maximum(abs, @view C[:, j]) for j in 1:m]
    
    # 2. 分别计算行和列的一维截断索引
    row_cut = find_chop_index(row_norms, max_c, tol)
    col_cut = find_chop_index(col_norms, max_c, tol)
    
    return C[1:row_cut, 1:col_cut]
end

"""
一维包络线平原检测核心逻辑
"""
function find_chop_index(norms::Vector{Float64}, max_c::Float64, tol::Float64)
    N = length(norms)
    
    # 1. 构造单调递减的包络线 (Reverse Cumulative Maximum)
    # 这一步极其关键：它能填平切比雪夫系数中由于函数奇偶性导致的 "锯齿状" 零值跌落
    env = reverse(accumulate(max, reverse(norms)))
    
    # 2. 计算底线阈值：结合传入的容差和机器精度的安全底线
    abs_threshold = max(tol * max_c, eps() * max_c)
    
    cut_idx = N
    for k in N:-1:1
        tail_max = env[k]
        tail_min = env[end] + eps() * max_c # 加上 eps 防止除零错误
        
        # 截断条件 A：已经严格低于我们设定的绝对容差底线
        is_below_tol = tail_max <= abs_threshold
        
        # 截断条件 B：Chebfun 式平原检测
        # - 条件 B1：必须处于极低的量级 (例如全局峰值的 1e-11 以下，防止误砍真实缓慢衰减的高频区)
        # - 条件 B2：尾部波动平坦 (最大值与最小值的比值在两个数量级以内，说明已无衰减趋势)
        is_flat_plateau = (tail_max < max_c * 1e-11) && (tail_max / tail_min < 100.0)
        
        if is_below_tol || is_flat_plateau
            cut_idx = k - 1
        else
            break # 遇到真实的显著衰减区，立即停止截断
        end
    end
    
    return max(1, cut_idx)
end


# =======================================================================================
"""
    cheb_intmat(N::Int; square::Bool=false)

构造用于 Chebyshev 系数向量的积分矩阵 S，对应算子 I[f](x) = \\int_{-1}^x f(t) dt。
输入 N 为切比雪夫展开的项数（即对应 T_0 到 T_{N-1}）。

默认返回大小为 (N+1) x N 的矩阵，以保留积分增加的最高阶项。
若设置 square=true，则截断最高阶，返回 N x N 的方阵。
"""
function cheb_intmat(N::Int; square::Bool=false)
    if N < 1
        error("N 必须大于等于 1")
    end
    
    # 预分配 (N+1) x N 矩阵
    S = zeros(Float64, N + 1, N)
    
    # 1. 构造 C_n (n >= 1) 的映射系数
    # 对应 C_1 = c_0 - c_2 / 2 (注意 Julia 索引从 1 开始: C_1 对应行 2, c_0 对应列 1)
    if N >= 1
        S[2, 1] = 1.0
    end
    if N >= 3
        S[2, 3] = -0.5
    end
    
    # 对应 C_n = (c_{n-1} - c_{n+1}) / (2n) , n >= 2
    for n in 2:N
        S[n+1, n] = 1.0 / (2.0 * n)
        if n + 2 <= N
            S[n+1, n+2] = -1.0 / (2.0 * n)
        end
    end
    
    # 2. 构造 C_0 (积分常数) 
    # 由 F(-1) = 0 决定: C_0 = sum_{n=1}^N (-1)^{n+1} C_n
    for n in 1:N
        sign_val = (n % 2 == 1) ? 1.0 : -1.0
        S[1, :] .+= sign_val .* S[n+1, :]
    end
    
    # 3. 根据参数决定是否截断返回方阵
    if square
        return S[1:N, :]
    else
        return S
    end
end

# =======================================================================================
"""
    cheb_multmat(a::AbstractVector{T}, N::Int) where T <: Number

给定以 Chebyshev 基展开的函数 f(x) = ∑ a_k T_k(x) 的系数向量 `a`，
构造其作为乘法算子的截断矩阵 M。

返回的矩阵是一个 N × N 的方阵，这意味着所有超过 T_{N-1} 的高阶乘积项都会被丢弃。
"""
function cheb_multmat(a::AbstractVector{T}, N::Int) where T <: Number
    n = length(a) - 1 # f(x) 的最高阶数
    
    # 预分配 N x N 截断方阵
    M = zeros(T, N, N)
    
    for k in 0:n
        for j in 0:N-1
            # 贡献 1：T_{k+j}
            d1 = k + j
            if d1 < N
                M[d1 + 1, j + 1] += a[k + 1] / 2.0
            end
            
            # 贡献 2：T_{|k-j|}
            d2 = abs(k - j)
            if d2 < N
                M[d2 + 1, j + 1] += a[k + 1] / 2.0
            end
        end
    end
    
    return M
end