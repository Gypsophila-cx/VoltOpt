using FastTransforms, LinearAlgebra

function multrestrict(c::AbstractVector{<:Real})
    # 统一转换为浮点数类型以避免类型不稳定
    c_float = float.(c)
    T = eltype(c_float)
    N = length(c_float) - 1
    C = zeros(T, N + 1, N + 1)

    # 1. 预处理系数 c
    c_mod = copy(c_float)
    c_mod[1] *= 2.0
    c_new = similar(c_float)
    @inbounds for i in 1:N
        c_new[i] = (c_mod[i] + c_mod[i+1]) / 2.0
    end
    c_new[N+1] = c_mod[N+1] / 2.0

    # 使用 FastTransforms 中的 leg2cheb 函数进行勒让德到切比雪夫的转换
    ll = leg2cheb(c_new)                           
    C[1, :] .= ll

    # 2. 定义 S0 和 K 的非零对角线解析通项 (内联，剔除所有恒为0的项)
    @inline S0_diag0(I) = I == 1 ? 1.0 : 0.5
    @inline S0_diag2(I) = -0.5

    @inline K_diag0(I) = -0.5 * (I - 1)
    @inline K_diag1(I) = 1.0 * I
    @inline K_diag2(I) = -0.5 * (I + 1)

    # 3. 核心循环：无内存分配的上三角系统向后代换
    @inbounds for k in 1:N
        # 边界 1：总是执行 (对应全局索引 I = N + 1)
        let I = N + 1
            u_val = ((k - 1) * S0_diag0(I) + K_diag0(I)) * C[k, I]
            D0_LHS = k * S0_diag0(I) - K_diag0(I)
            C[k+1, I] = u_val / D0_LHS
        end

        # 边界 2：当 k < N 时执行 (对应全局索引 I = N)
        if k < N
            let I = N
                # S0_diag1 恒为 0，已在此处代数化简
                u_val = ((k - 1) * S0_diag0(I) + K_diag0(I)) * C[k, I] + 
                        K_diag1(I) * C[k, I+1]
                
                rhs_eff = u_val + K_diag1(I) * C[k+1, I+1]
                D0_LHS = k * S0_diag0(I) - K_diag0(I)
                C[k+1, I] = rhs_eff / D0_LHS
            end
        end

        # 主干循环：I 从 N-1 递减到 k+1 (对应向后代换的主体部分)
        for I in N-1:-1:k+1
            u_val = ((k - 1) * S0_diag0(I) + K_diag0(I)) * C[k, I] +
                    K_diag1(I) * C[k, I+1] +
                    ((k - 1) * S0_diag2(I) + K_diag2(I)) * C[k, I+2]

            rhs_eff = u_val + K_diag1(I) * C[k+1, I+1] -
                      (k * S0_diag2(I) - K_diag2(I)) * C[k+1, I+2]

            D0_LHS = k * S0_diag0(I) - K_diag0(I)
            C[k+1, I] = rhs_eff / D0_LHS
        end
    end

    # 4. 后处理
    @views C[2:end, :] .*= 2.0

    sum_c1 = zero(T)
    @inbounds @simd for j in 2:N+1
        sum_c1 += C[1, j]
    end
    C[1, 1] = sum(c_float) - sum_c1

    # 5. 矩阵对称化 (等效于 C = C + triu(C, 1)')
    # 重构了嵌套顺序：i 位于外层，j 位于内层，使得写入操作 C[j, i] 沿内存连续方向(列)进行
    @inbounds for i in 1:N+1
        for j in i+1:N+1
            C[j, i] = C[i, j]
        end
    end

    return C
end