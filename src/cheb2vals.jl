using FFTW

function cheb2vals_lobatto_fft(C::AbstractMatrix)
    N = size(C, 1)
    
    # FFTW 的 REDFT00 会对首尾项和中间项使用特定的权重。
    # 为了匹配标准的 Chebyshev 求和公式，我们需要对系数矩阵进行轻微的缩放：
    C_scaled = copy(C)
    C_scaled[1, :] .*= 2.0     # 缩放第 0 阶系数 (k=0)
    C_scaled[end, :] .*= 2.0   # 缩放最高阶系数 (k=N-1)
    
    # 沿着列（维度 1）进行 DCT-I 变换，并除以 2 进行归一化
    V = FFTW.r2r(C_scaled, FFTW.REDFT00, 1) ./ 2.0
    
    return V
end

function cheb2vals_2d_fft(C::AbstractMatrix)
    N = size(C, 1)
    
    # 1. 复制矩阵以防修改原数据
    C_scaled = copy(C)
    
    # 2. 匹配 2D Chebyshev 展开的权重
    # 对第一维度（行边界）进行缩放
    C_scaled[1, :] .*= 2.0
    C_scaled[end, :] .*= 2.0
    
    # 对第二维度（列边界）进行缩放
    C_scaled[:, 1] .*= 2.0
    C_scaled[:, end] .*= 2.0
    
    # （注：此时四个角点的系数已经被自动乘以 4.0，这在数学上完全正确）
    
    # 3. 对所有维度执行 2D REDFT00
    # r2r 默认对传入数组的所有维度进行指定的变换
    V = FFTW.r2r(C_scaled, FFTW.REDFT00)
    
    # 4. 2D 归一化：每个维度除以 2，因此总共除以 4
    V ./= 4.0
    
    return V
end