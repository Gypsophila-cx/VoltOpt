using FFTW
using Printf
using SparseArrays

# ============================================================
# Paths
# ============================================================

script_dir = @__DIR__
project_root = dirname(script_dir)
data_dir = joinpath(project_root, "test", "data")
mkpath(data_dir)

include(joinpath(project_root, "src", "VoltOpt.jl"))
using .VoltOpt

# ============================================================
# Settings
# ============================================================

const MATRIX_TOL = 64eps(Float64)
const DEFAULT_REPEATS = 3
const TIMING_MODE_VARY_N = "fixed_bandwidth_varyN"
const TIMING_MODE_VARY_BANDWIDTH = "fixedN_vary_bandwidth"
const TIMING_MODE = TIMING_MODE_VARY_BANDWIDTH

K(x, y) = exp(-x * y)

make_N_grid() = log_integer_grid(100, 10_000, 30)
make_bandwidth_grid() = log_integer_grid(2, 300, 30)

function log_integer_grid(a::Int, b::Int, n::Int)
    a > 0 || throw(ArgumentError("a must be positive"))
    b >= a || throw(ArgumentError("b must be greater than or equal to a"))
    n > 0 || throw(ArgumentError("n must be positive"))

    if n == 1
        a == b || error("cannot include distinct endpoints with one grid point")
        return [a]
    end

    b - a + 1 >= n || error("cannot generate $n unique integers in [$a, $b]")

    loga = log(Float64(a))
    logb = log(Float64(b))
    vals = Vector{Int}(undef, n)

    for i in 1:n
        theta = (i - 1) / (n - 1)
        vals[i] = round(Int, exp((1 - theta) * loga + theta * logb))
    end

    vals[1] = a
    vals[end] = b

    for i in 2:n
        vals[i] = max(vals[i], vals[i - 1] + 1)
    end

    for i in (n - 1):-1:1
        vals[i] = min(vals[i], vals[i + 1] - 1)
    end

    vals[1] = a
    vals[end] = b

    if length(unique(vals)) != n || first(vals) != a || last(vals) != b ||
       any(vals[i] >= vals[i + 1] for i in 1:(n - 1))
        error("failed to generate a monotone unique integer log grid")
    end

    return vals
end

# ============================================================
# Coefficients and construction
# ============================================================

function cheb_duffy_kernel_coeffs(Kfun::Function, input_bandwidth::Int)
    input_bandwidth >= 2 ||
        throw(ArgumentError("input_bandwidth must be at least 2"))

    nodes = cos.((0:(input_bandwidth - 1)) .* (pi / (input_bandwidth - 1)))
    values = Matrix{Float64}(undef, input_bandwidth, input_bandwidth)

    @inbounds for j in 1:input_bandwidth
        y = nodes[j]
        for i in 1:input_bandwidth
            x = nodes[i]
            t = (1 + x) * (1 + y) / 2 - 1
            values[i, j] = Kfun(x, t)
        end
    end

    coeffs = FFTW.r2r(values, FFTW.REDFT00)
    coeffs ./= (input_bandwidth - 1)^2
    coeffs[1, :] ./= 2.0
    coeffs[end, :] ./= 2.0
    coeffs[:, 1] ./= 2.0
    coeffs[:, end] ./= 2.0

    return Matrix{Float64}(coeffs)
end

function construct_voltmat_cheb_from_coeffs(coeffs::Matrix{Float64}, N::Int)
    return VoltOpt.voltmat_square_from_coeffmat_fft_convolution(
        coeffs,
        N;
        tol = MATRIX_TOL,
        use_fft_top = true,
    )
end

# ============================================================
# Timing and matrix diagnostics
# ============================================================

function time_min(f; repeats::Int = DEFAULT_REPEATS)
    best = Inf
    for _ in 1:repeats
        GC.gc()
        t = @elapsed f()
        best = min(best, t)
    end
    return best
end

function time_min_with_result(f; repeats::Int = DEFAULT_REPEATS)
    best = Inf
    best_value = nothing

    for _ in 1:repeats
        GC.gc()
        value_ref = Ref{Any}()
        t = @elapsed begin
            value_ref[] = f()
        end

        if t < best
            best = t
            best_value = value_ref[]
        end
    end

    return best, best_value
end

function warmup()
    Klocal(x, y) = exp(-x * y)
    coeffs = cheb_duffy_kernel_coeffs(Klocal, 4)
    construct_voltmat_cheb_from_coeffs(coeffs, 32)
    return nothing
end

function detect_bandwidth(A; tol = 0.0)
    S = sparse(A)
    rows, cols, vals = findnz(S)
    lower_bw = 0
    upper_bw = 0

    for (i, j, v) in zip(rows, cols, vals)
        if abs(v) > tol
            lower_bw = max(lower_bw, i - j)
            upper_bw = max(upper_bw, j - i)
        end
    end

    return lower_bw, upper_bw
end

# ============================================================
# CSV
# ============================================================

const CSV_HEADERS = [
    "case_name",
    "N",
    "input_bandwidth",
    "bandwidth_source",
    "kernel_coeff_time_min",
    "construct_from_coeffs_time_min",
    "total_time_sum_min",
    "end_to_end_time_min",
    "timing_mode",
    "matrix_size",
    "nnz",
    "lower_bw",
    "upper_bw",
    "status",
    "error_message",
]

function csv_escape(x)
    x === nothing && return ""
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    else
        return s
    end
end

function write_csv(csv_file::AbstractString, rows::Vector{<:NamedTuple})
    open(csv_file, "w") do io
        println(io, join(CSV_HEADERS, ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, Symbol(h))) for h in CSV_HEADERS), ","))
        end
    end
end

# ============================================================
# Benchmark row
# ============================================================

function failed_row(
    N::Int,
    input_bandwidth::Int;
    bandwidth_source::String,
    timing_mode::String = TIMING_MODE,
    error_message::String,
)
    return (
        case_name = "cheb_N$(N)_bandwidth$(input_bandwidth)",
        N = N,
        input_bandwidth = input_bandwidth,
        bandwidth_source = bandwidth_source,
        kernel_coeff_time_min = "",
        construct_from_coeffs_time_min = "",
        total_time_sum_min = "",
        end_to_end_time_min = "",
        timing_mode = timing_mode,
        matrix_size = "",
        nnz = "",
        lower_bw = "",
        upper_bw = "",
        status = "failed",
        error_message = error_message,
    )
end

function benchmark_one_cheb(
    N::Int,
    input_bandwidth::Int;
    bandwidth_source::String = "manual",
    repeats::Int = DEFAULT_REPEATS,
    precomputed_coeffs::Union{Nothing,Matrix{Float64}} = nothing,
    precomputed_kernel_coeff_time_min::Union{Nothing,Float64} = nothing,
    timing_mode::String = TIMING_MODE,
)
    try
        kernel_coeff_time_min, coeffs =
            if precomputed_coeffs === nothing
                time_min_with_result(
                    () -> cheb_duffy_kernel_coeffs(K, input_bandwidth);
                    repeats = repeats,
                )
            else
                (
                    precomputed_kernel_coeff_time_min === nothing ?
                    0.0 :
                    precomputed_kernel_coeff_time_min,
                    precomputed_coeffs,
                )
            end

        construct_from_coeffs_time_min, A = time_min_with_result(
            () -> construct_voltmat_cheb_from_coeffs(coeffs, N);
            repeats = repeats,
        )

        end_to_end_time_min = time_min(
            () -> begin
                C = cheb_duffy_kernel_coeffs(K, input_bandwidth)
                construct_voltmat_cheb_from_coeffs(C, N)
            end;
            repeats = repeats,
        )

        S = sparse(A)
        matrix_size = "$(size(A, 1))x$(size(A, 2))"
        nnz_A = nnz(S)
        lower_bw, upper_bw = detect_bandwidth(S)

        return (
            case_name = "cheb_N$(N)_bandwidth$(input_bandwidth)",
            N = N,
            input_bandwidth = input_bandwidth,
            bandwidth_source = bandwidth_source,
            kernel_coeff_time_min = kernel_coeff_time_min,
            construct_from_coeffs_time_min = construct_from_coeffs_time_min,
            total_time_sum_min = kernel_coeff_time_min + construct_from_coeffs_time_min,
            end_to_end_time_min = end_to_end_time_min,
            timing_mode = timing_mode,
            matrix_size = matrix_size,
            nnz = nnz_A,
            lower_bw = lower_bw,
            upper_bw = upper_bw,
            status = "success",
            error_message = "",
        )
    catch err
        return failed_row(
            N,
            input_bandwidth;
            bandwidth_source = bandwidth_source,
            timing_mode = timing_mode,
            error_message = sprint(showerror, err),
        )
    end
end

# ============================================================
# Main benchmarks
# ============================================================

function assert_grids()
    @assert length(make_N_grid()) == 30
    @assert length(unique(make_N_grid())) == 30
    @assert first(make_N_grid()) == 100
    @assert last(make_N_grid()) == 10_000

    @assert length(make_bandwidth_grid()) == 30
    @assert length(unique(make_bandwidth_grid())) == 30
    @assert first(make_bandwidth_grid()) == 2
    @assert last(make_bandwidth_grid()) == 300

    return nothing
end

function run_benchmark_varyN(;
    input_bandwidth::Int = 20,
    bandwidth_source::String = "existing_default_kerneldepth",
    Ns::Vector{Int} = make_N_grid(),
    repeats::Int = DEFAULT_REPEATS,
    output_csv::String = joinpath(data_dir, "voltmat_cheb_fixed_bandwidth20_varyN_time.csv"),
)
    assert_grids()
    warmup()

    rows = NamedTuple[]

    coeff_time = NaN
    coeffs = nothing
    coeff_error = nothing

    try
        coeff_time, coeffs = time_min_with_result(
            () -> cheb_duffy_kernel_coeffs(K, input_bandwidth);
            repeats = repeats,
        )
    catch err
        coeff_error = sprint(showerror, err)
    end

    for N in Ns
        @printf("N = %d, input_bandwidth = %d\n", N, input_bandwidth)

        row = if coeff_error === nothing
            benchmark_one_cheb(
                N,
                input_bandwidth;
                bandwidth_source = bandwidth_source,
                repeats = repeats,
                precomputed_coeffs = coeffs,
                precomputed_kernel_coeff_time_min = coeff_time,
                timing_mode = TIMING_MODE_VARY_N,
            )
        else
            failed_row(
                N,
                input_bandwidth;
                bandwidth_source = bandwidth_source,
                timing_mode = TIMING_MODE_VARY_N,
                error_message = coeff_error,
            )
        end

        push!(rows, row)

        if row.status == "success"
            @printf(
                "status = %s, kernel = %.6g s, construct = %.6g s, end_to_end = %.6g s, nnz = %s, bw = (%s,%s)\n",
                row.status,
                row.kernel_coeff_time_min,
                row.construct_from_coeffs_time_min,
                row.end_to_end_time_min,
                string(row.nnz),
                string(row.lower_bw),
                string(row.upper_bw),
            )
        else
            println("status = failed, error = ", row.error_message)
        end
    end

    write_csv(output_csv, rows)

    successful_rows = count(row -> row.status == "success", rows)
    failed_rows = length(rows) - successful_rows
    saved_path = replace(relpath(output_csv, project_root), '\\' => '/')

    println("Saved Chebyshev varying-N benchmark to ", saved_path)
    println("rows = ", length(rows))
    println("successful rows = ", successful_rows)
    println("failed rows = ", failed_rows)
    println("N grid length = ", length(Ns))

    return rows
end

function run_benchmark_varyn(;
    N::Int = 1000,
    bandwidths::Vector{Int} = make_bandwidth_grid(),
    bandwidth_source::String = "log_grid_2_300_30",
    repeats::Int = DEFAULT_REPEATS,
    output_csv::String = joinpath(data_dir, "voltmat_cheb_fixedN1000_vary_bandwidth_time.csv"),
)
    assert_grids()
    warmup()

    rows = NamedTuple[]

    for input_bandwidth in bandwidths
        @printf("N = %d, input_bandwidth = %d\n", N, input_bandwidth)

        row = benchmark_one_cheb(
            N,
            input_bandwidth;
            bandwidth_source = bandwidth_source,
            repeats = repeats,
            timing_mode = TIMING_MODE_VARY_BANDWIDTH,
        )

        push!(rows, row)

        if row.status == "success"
            @printf(
                "status = %s, kernel = %.6g s, construct = %.6g s, end_to_end = %.6g s, nnz = %s, bw = (%s,%s)\n",
                row.status,
                row.kernel_coeff_time_min,
                row.construct_from_coeffs_time_min,
                row.end_to_end_time_min,
                string(row.nnz),
                string(row.lower_bw),
                string(row.upper_bw),
            )
        else
            println("status = failed, error = ", row.error_message)
        end
    end

    write_csv(output_csv, rows)

    successful_rows = count(row -> row.status == "success", rows)
    failed_rows = length(rows) - successful_rows
    saved_path = replace(relpath(output_csv, project_root), '\\' => '/')

    println("Saved Chebyshev fixed-N varying-bandwidth benchmark to ", saved_path)
    println("rows = ", length(rows))
    println("successful rows = ", successful_rows)
    println("failed rows = ", failed_rows)
    println("bandwidth grid length = ", length(bandwidths))

    return rows
end

function run_benchmark(; mode::Symbol = :auto, kwargs...)
    selected_mode = if mode === :auto
        requested_script_mode()
    else
        mode
    end

    if selected_mode === :varyN
        return run_benchmark_varyN(; kwargs...)
    elseif selected_mode === :varyn || selected_mode === :vary_bandwidth
        return run_benchmark_varyn(; kwargs...)
    end

    throw(ArgumentError("mode must be :auto, :varyN, :varyn, or :vary_bandwidth"))
end

function requested_script_mode()
    name = basename(PROGRAM_FILE)
    return occursin("varyN", name) ? :varyN : :varyn
end

function is_main_script()
    isempty(PROGRAM_FILE) && return false
    try
        return samefile(PROGRAM_FILE, @__FILE__)
    catch
        return lowercase(abspath(PROGRAM_FILE)) == lowercase(abspath(@__FILE__))
    end
end

if is_main_script()
    run_benchmark()
end
