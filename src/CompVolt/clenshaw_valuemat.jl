using LinearAlgebra

@inline function clenshaw_inline(z::Float64, c::AbstractVector{Float64})
    N = length(c)
    if N == 0 return 0.0 end
    if N == 1 return c[1] end
    
    bk1 = 0.0
    bk2 = 0.0
    z2 = 2.0 * z
    
    @inbounds for k in N:-1:2
        bk = c[k] + z2 * bk1 - bk2
        bk2 = bk1
        bk1 = bk
    end
    
    @inbounds return c[1] + z * bk1 - bk2
end

function clenshaw_valuemat(c::Vector{Float64}, Mx::Int, My::Int)
    # 1. 生成 Chebyshev-Lobatto 节点
    x_nodes = [-cos(π * i / (Mx - 1)) for i in 0:(Mx - 1)]
    y_nodes = [-cos(π * j / (My - 1)) for j in 0:(My - 1)]
    
    # 2. 【优化2】使用 undef 替代 zeros，省去大矩阵内存填零的时间
    U_grid = Matrix{Float64}(undef, Mx, My)
    
    # 3. 【优化3】代数预计算：将 x + 1.0 提出来，省去 Mx*My 次加法指令
    x_plus_1 = x_nodes .+ 1.0
    
    # 4. 【优化4】列主序 (Column-major) 遍历
    # Julia 矩阵在内存中是按列连续存储的。
    # 我们让 j (列) 在外层，i (行) 在内层，确保 CPU 缓存命中率达到 100%
    for j in 1:My
        # 提取当前列的公共因子，内层只需一次乘法和一次减法
        y_mapped = (y_nodes[j] + 1.0) / 2.0
        
        # 5. 【优化5】SIMD 向量化
        # @inbounds 关掉边界检查，@simd 提示 LLVM 编译器使用 AVX/SSE 寄存器一次性打包计算多个 z
        @inbounds @simd for i in 1:Mx
            z = x_plus_1[i] * y_mapped - 1.0
            U_grid[i, j] = clenshaw_inline(z, c)
        end
    end
    
    return x_nodes, y_nodes, U_grid
end