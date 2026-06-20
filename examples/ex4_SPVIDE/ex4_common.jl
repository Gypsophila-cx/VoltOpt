using Printf

function ex4_env_int(name::AbstractString, default::Int)
    value = get(ENV, name, "")
    isempty(value) && return default
    parsed = tryparse(Int, value)
    parsed === nothing && error("Environment variable $(name) must be an integer, got $(value).")
    return parsed
end

function ex4_env_float(name::AbstractString, default::Float64)
    value = get(ENV, name, "")
    isempty(value) && return default
    parsed = tryparse(Float64, value)
    parsed === nothing && error("Environment variable $(name) must be a Float64, got $(value).")
    return parsed
end

function ex4_env_bool(name::AbstractString, default::Bool=false)
    value = lowercase(strip(get(ENV, name, "")))
    isempty(value) && return default
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    error("Environment variable $(name) must be boolean-like, got $(value).")
end

function ex4_env_int_list(name::AbstractString)
    value = strip(get(ENV, name, ""))
    isempty(value) && return nothing

    out = Int[]
    for item in split(value, ",")
        token = strip(item)
        isempty(token) && continue
        parsed = tryparse(Int, token)
        parsed === nothing && error("Environment variable $(name) must be a comma-separated integer list, got $(value).")
        push!(out, parsed)
    end

    isempty(out) && error("Environment variable $(name) did not contain any integer values.")
    return out
end

function ex4_evenly_spaced_integer_nodes(lo::Int, hi::Int, count::Int)
    lo <= hi || error("lo must not exceed hi.")
    count >= 2 || error("count must be at least 2.")

    nodes = round.(Int, range(lo, hi; length=count))
    length(unique(nodes)) == count || error("Rounded node list contains duplicates.")
    return nodes
end

function ex4_default_n_list(n_final::Int, n_count::Int)
    explicit = ex4_env_int_list("EX4_N_LIST")
    explicit !== nothing && return explicit
    return ex4_evenly_spaced_integer_nodes(2, n_final, n_count)
end

function ex4_reference_degree(N::Int; factor::Float64=1.01)
    return max(N + 1, round(Int, factor * N))
end

function ex4_collect_needed_degrees(N_list; n_final::Union{Nothing,Int}=nothing, factor::Float64=1.01)
    needed_N = Set{Int}()
    for N in N_list
        push!(needed_N, N)
        push!(needed_N, ex4_reference_degree(N; factor=factor))
    end
    n_final !== nothing && push!(needed_N, n_final)
    return sort(collect(needed_N))
end

function ex4_diagnostic_x_grid(M::Int)
    M >= 2 || error("Diagnostic grid size M must be at least 2.")
    return cos.(pi .* (0:M) ./ M)
end

function ex4_diagnostic_physical_grid(M::Int)
    x = ex4_diagnostic_x_grid(M)
    return 0.5 .* (x .+ 1.0)
end

function ex4_shape_x_grid(num_points::Int)
    num_points >= 2 || error("Shape grid must contain at least two points.")
    return collect(range(0.0, 1.0; length=num_points))
end

function ex4_csv_cell(value)
    s = string(value)
    needs_quote = occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
    s = replace(s, "\"" => "\"\"")
    return needs_quote ? "\"$(s)\"" : s
end

function ex4_write_csv_header(path::AbstractString, columns)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(columns, ","))
        flush(io)
    end
    return path
end

function ex4_append_csv_row(path::AbstractString, row)
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, join(ex4_csv_cell.(row), ","))
        flush(io)
    end
    return path
end

function ex4_status_from_error(err)
    msg = sprint(showerror, err)
    msg = replace(msg, "\r" => " ", "\n" => " ")
    return "failed:$(typeof(err)):$(msg)"
end

function ex4_pair_status(status_N::AbstractString, status_ref::AbstractString)
    status_N == "ok" && status_ref == "ok" && return "ok"
    return "failed:N=$(status_N);N_ref=$(status_ref)"
end

function ex4_spvide_output_tag(eps_value::Float64, omega_value::Float64; kerneldepth=nothing)
    tag = kerneldepth === nothing ?
        @sprintf("eps%.0e_omega%.0f", eps_value, omega_value) :
        @sprintf("eps%.0e_omega%.0f_kdepth%d", eps_value, omega_value, Int(kerneldepth))
    return replace(tag, "+" => "", "-" => "minus")
end

ex4_result_columns = [
    "method",
    "N",
    "N_ref",
    "omega",
    "cauchy_coeff_error",
    "cauchy_grid_error",
    "tail_coeff_norm",
    "build_operator_sec",
    "build_rhs_sec",
    "solve_sec",
    "total_time_sec",
    "nnzA",
    "nnzVK",
    "residual_inf",
    "status",
]

ex4_test_result_columns = [
    "method",
    "N",
    "N_ref",
    "cauchy_coeff_error",
    "cauchy_grid_error",
    "total_time_sec",
    "status",
]

ex4_shape_columns = [
    "method",
    "N",
    "x",
    "uN",
    "uNref",
    "diff",
]
