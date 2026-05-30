using SparseArrays
using Random
using Plots
using LaTeXStrings

gr()

const SCRIPT_DIR   = @__DIR__
const PROJECT_ROOT = dirname(SCRIPT_DIR)
const FIG_DIR      = joinpath(SCRIPT_DIR, "figures")
mkpath(FIG_DIR)

# ============================================================
# 载入项目模块
# ============================================================

include(joinpath(PROJECT_ROOT, "src", "VoltOpt.jl"))
using .VoltOpt

# ============================================================
# 参数设置
# ============================================================

const Nplot = 200
const Nmat  = 300

K(x, y) = exp(-x * y)

# ============================================================
# 工具函数
# ============================================================

function extract_matrix(B_raw)
    return B_raw isa Tuple ? B_raw[1] : B_raw
end

function detect_raw_top_dense_rows(A::AbstractMatrix; density_threshold::Float64=0.8)
    nrow, ncol = size(A)
    top_rows = 0

    for i in 1:nrow
        row_nnz = count(!iszero, @view A[i, :])
        if row_nnz >= density_threshold * ncol
            top_rows += 1
        else
            break
        end
    end

    return top_rows
end

function detect_bandwidth(A::AbstractMatrix; exclude_dense_top::Bool=true, default_bw::Int=10)
    nrow, ncol = size(A)

    raw_top_rows = exclude_dense_top ? detect_raw_top_dense_rows(A) : 0

    rows = Int[]
    cols = Int[]

    for j in 1:ncol
        for i in 1:nrow
            if i > raw_top_rows && !iszero(A[i, j])
                push!(rows, i)
                push!(cols, j)
            end
        end
    end

    if isempty(rows)
        @warn "No nonzeros found after excluding raw dense top rows. Use default bandwidth."
        return default_bw, raw_top_rows
    end

    bw = maximum(abs.(rows .- cols))
    bw = clamp(Int(bw), 0, min(nrow, ncol) - 1)

    return bw, raw_top_rows
end

function add_machine_perturbation(
    A::AbstractMatrix;
    bw::Int,
    add_top_dense_rows::Bool,
    use_random_perturb::Bool=true,
    seed::Int=1234,
)
    nrow, ncol = size(A)

    scale = maximum(abs.(A))
    scale = max(scale, 1.0)

    δ = eps(Float64)

    top_rows = add_top_dense_rows ? min(bw, nrow) : 0

    I = Int[]
    J = Int[]
    V = Float64[]

    sizehint!(I, top_rows * ncol + ncol * (2bw + 1))
    sizehint!(J, top_rows * ncol + ncol * (2bw + 1))
    sizehint!(V, top_rows * ncol + ncol * (2bw + 1))

    rng = MersenneTwister(seed)

    for j in 1:ncol
        # --------------------------------------------------------
        # 1. 顶部稠密行扰动
        #
        # Jacobi / Chebyshev 使用；
        # Legendre 不使用。
        # --------------------------------------------------------

        if add_top_dense_rows
            for i in 1:top_rows
                push!(I, i)
                push!(J, j)

                if use_random_perturb
                    push!(V, δ * (1.0 + rand(rng)))
                else
                    push!(V, δ)
                end
            end
        end

        # --------------------------------------------------------
        # 2. 带状区域扰动
        # --------------------------------------------------------

        i_min = add_top_dense_rows ? max(top_rows + 1, j - bw) : max(1, j - bw)
        i_max = min(nrow, j + bw)

        if i_min <= i_max
            for i in i_min:i_max
                push!(I, i)
                push!(J, j)

                if use_random_perturb
                    push!(V, δ * (1.0 + rand(rng)))
                else
                    push!(V, δ)
                end
            end
        end
    end

    return sparse(I, J, V, nrow, ncol)
end

# ============================================================
# 绘图函数
# ============================================================

function plot_voltmat_spy(
    B_raw;
    title_string,
    output_name,
    basis_name,
    add_top_dense_rows::Bool,
    Nplot::Int=200,
    seed::Int=1234,
)
    B = extract_matrix(B_raw)

    size(B, 1) >= Nplot && size(B, 2) >= Nplot ||
        error("Matrix size $(size(B)) is smaller than $(Nplot) × $(Nplot).")

    # ------------------------------------------------------------
    # 1. 提取前 Nplot × Nplot 子块
    # ------------------------------------------------------------

    A = Matrix(B[1:Nplot, 1:Nplot])

    # ------------------------------------------------------------
    # 2. 检测带宽
    #
    # 对 Jacobi / Chebyshev，如果原矩阵已经有真实顶部稠密行，
    # 检测带宽时先排除这些顶部稠密行，避免把顶部稠密行误判成超大带宽。
    # ------------------------------------------------------------

    bw, raw_top_rows = detect_bandwidth(A; exclude_dense_top=add_top_dense_rows)

    println("============================================================")
    println("Basis: ", basis_name)
    println("Matrix size: ", size(B))
    println("Plotted block size: ", Nplot, " × ", Nplot)
    println("Detected raw dense top rows = ", raw_top_rows)
    println("Detected bandwidth = ", bw)

    # ------------------------------------------------------------
    # 3. 添加机器精度扰动
    # ------------------------------------------------------------

    Pert = add_machine_perturbation(
        A;
        bw = bw,
        add_top_dense_rows = add_top_dense_rows,
        seed = seed,
    )

    A_abs = abs.(sparse(A)) + Pert

    # ------------------------------------------------------------
    # 4. 取 log10
    # ------------------------------------------------------------

    A_log = copy(A_abs)
    A_log.nzval .= log10.(A_log.nzval)

    rows, cols, vals = findnz(A_log)

    isempty(vals) && error("The plotted matrix block has no nonzero entries.")

    # ------------------------------------------------------------
    # 5. 诊断前若干行
    # ------------------------------------------------------------

    n_diag_rows = min(20, Nplot)

    println("Diagnostic for the first $(n_diag_rows) rows after perturbation:")

    for i in 1:n_diag_rows
        row_abs = abs.(A_abs[i, :])
        row_nnz = count(!iszero, row_abs)
        row_max = maximum(row_abs)
        row_min_nz = row_nnz == 0 ? 0.0 : minimum(row_abs[row_abs .!= 0])

        println(
            "  row ", lpad(i, 3),
            " | nnz = ", lpad(row_nnz, 4),
            " / ", Nplot,
            " | min nonzero = ", row_min_nz,
            " | max = ", row_max,
        )
    end

    println("Total plotted nonzeros: ", length(vals))

    # ------------------------------------------------------------
    # 6. Colorbar 刻度
    # ------------------------------------------------------------

    min_val, max_val = extrema(vals)
    tick_positions = collect(floor(min_val):1.0:ceil(max_val))
    tick_labels = ["10^$(Int(p))" for p in tick_positions]

    # ------------------------------------------------------------
    # 7. 手动画 spy 图
    #
    # yflip=true 让第 1 行显示在顶部。
    # ------------------------------------------------------------

    p = scatter(
        cols,
        rows;
        marker_z = vals,
        seriestype = :scatter,
        markersize = 1,
        markerstrokewidth = 0,
        marker = :rect,
        legend = false,
        colorbar = true,
        colorbar_ticks = (tick_positions, tick_labels),
        xlabel = "column index",
        ylabel = "row index",
        title = title_string,
        xlims = (1, Nplot),
        ylims = (1, Nplot),
        yflip = true,
        framestyle = :box,
        aspect_ratio = :equal,
        grid = false,
        minorgrid = false,
        xgrid = false,
        ygrid = false,
    )

    display(p)

    output_path = joinpath(FIG_DIR, output_name)
    savefig(p, output_path)

    println("Figure saved to: ", output_path)
    println("============================================================")
    println()

    return p, bw
end

# ============================================================
# 构造三个矩阵
# ============================================================

println("Constructing Jacobi Volterra matrix...")
B_jac_raw = VoltOpt.voltmat_jacobi(K, Nmat, 0.3, 0.2)

println("Constructing Chebyshev Volterra matrix...")
B_cheb_raw = VoltOpt.voltmat_cheb(K, Nmat; tol=1e-20)

println("Constructing Legendre Volterra matrix...")
B_leg_raw = VoltOpt.voltmat_leg(K, Nmat; tol=1e-20)

# ============================================================
# 绘图并输出
# ============================================================

plot_voltmat_spy(
    B_jac_raw;
    title_string = L"P_n^{(0.3,0.2)}(x)",
    output_name = "voltmat_jacobi_spy_exp_negxy.pdf",
    basis_name = "Jacobi(0.3, 0.2)",
    add_top_dense_rows = true,
    Nplot = Nplot,
    seed = 1234,
)

plot_voltmat_spy(
    B_cheb_raw;
    title_string = L"T_n(x)",
    output_name = "voltmat_cheb_spy_exp_negxy.pdf",
    basis_name = "Chebyshev",
    add_top_dense_rows = true,
    Nplot = Nplot,
    seed = 2345,
)

plot_voltmat_spy(
    B_leg_raw;
    title_string = L"P_n(x)",
    output_name = "voltmat_leg_spy_exp_negxy.pdf",
    basis_name = "Legendre",
    add_top_dense_rows = false,
    Nplot = Nplot,
    seed = 3456,
)

println("All figures have been written to:")
println("  ", joinpath(FIG_DIR, "voltmat_jacobi_spy_exp_negxy.pdf"))
println("  ", joinpath(FIG_DIR, "voltmat_cheb_spy_exp_negxy.pdf"))
println("  ", joinpath(FIG_DIR, "voltmat_leg_spy_exp_negxy.pdf"))