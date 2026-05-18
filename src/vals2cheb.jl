function vals2cheb(v::AbstractVector)
    N = length(v) - 1
    
    # 1. 构造偶对称序列：将原向量和去头去尾的逆序向量拼接，长度变为 2N
    # 比如 v = [v0, v1, v2, v3]，拼接后变成 [v0, v1, v2, v3, v2, v1]
    V = vcat(v, v[end-1:-1:2])
    
    # 2. 调用 FFT，除以 N 进行缩放，并提取实部
    C = real.(fft(V)) ./ N
    
    # 3. 截取前 N+1 个系数，并对首尾系数进行数学定义上的折半处理
    coeffs = C[1:N+1]
    coeffs[1] /= 2.0
    coeffs[end] /= 2.0
    
    return coeffs
end