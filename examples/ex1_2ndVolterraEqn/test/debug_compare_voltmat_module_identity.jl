using InteractiveUtils
using SparseArrays
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SRC_DIR = joinpath(ROOT, "src")
const N = 1000
const tol = eps(Float64)
const OUT_TXT = joinpath(@__DIR__, "debug_compare_voltmat_module_identity.txt")
const OUT_CSV = joinpath(@__DIR__, "debug_compare_voltmat_module_identity.csv")

const COLUMNS = [
    :source,
    :N,
    :tol,
    :module_name,
    :module_objectid,
    :function_objectid,
    :parent_module,
    :method_info,
    :K_type,
    :K_method,
    :sample_max_abs_diff_vs_aging_K,
    :t_VK,
    :VK_type,
    :VK_rows,
    :VK_cols,
    :nnz,
    :status,
    :notes,
]

function csv_escape(x)
    x === missing && return ""
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, join(string.(COLUMNS), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, col)) for col in COLUMNS), ","))
        end
    end
end

function row(; kwargs...)
    data = Dict{Symbol,Any}(col => missing for col in COLUMNS)
    data[:N] = N
    data[:tol] = tol
    data[:sample_max_abs_diff_vs_aging_K] = NaN
    data[:t_VK] = NaN
    data[:status] = "not_run"
    data[:notes] = ""
    for (key, value) in kwargs
        data[key] = value
    end
    return NamedTuple{Tuple(COLUMNS)}(Tuple(data[col] for col in COLUMNS))
end

function encode_field(x)
    s = string(x)
    s = replace(s, "%" => "%25")
    s = replace(s, "\t" => "%09")
    s = replace(s, "\n" => "%0A")
    s = replace(s, "\r" => "%0D")
    return s
end

function decode_field(s)
    s = replace(s, "%0D" => "\r")
    s = replace(s, "%0A" => "\n")
    s = replace(s, "%09" => "\t")
    s = replace(s, "%25" => "%")
    return s
end

function row_to_marker(row)
    return "__VOLT_IDENTITY_ROW__\t" * join((encode_field(getproperty(row, col)) for col in COLUMNS), "\t")
end

function marker_to_row(line)
    prefix = "__VOLT_IDENTITY_ROW__\t"
    startswith(line, prefix) || error("not a row marker")
    fields = split(line[length(prefix) + 1:end], '\t'; keepempty=true)
    length(fields) == length(COLUMNS) || error("bad field count: $(length(fields))")
    values = map(decode_field, fields)
    return NamedTuple{Tuple(COLUMNS)}(Tuple(values))
end

# ---------------------------------------------------------------------------
# aging_direct_include style.
# ---------------------------------------------------------------------------

include(joinpath(SRC_DIR, "VoltMat", "voltmat_cheb.jl"))
using .VoltMat_Cheb

theta(x) = 1.0 + tanh(10.0 * x)
tau_xy(x, y) = 1.0 / theta((x + y) / 2)

function K(x, y)
    tau = tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

function collect_identity(source, Kfun; sample_diff=NaN, notes="")
    local VK
    try
        method_info = sprint(show, @which VoltMat_Cheb.voltmat_cheb_square(Kfun, N - 1; tol=tol))
        k_method = sprint(show, @which Kfun(0.1, -0.2))
        GC.gc()
        t_vk = @elapsed begin
            VK = VoltMat_Cheb.voltmat_cheb_square(Kfun, N - 1; tol=tol)
        end
        return row(
            source=source,
            module_name=string(VoltMat_Cheb),
            module_objectid=string(objectid(VoltMat_Cheb)),
            function_objectid=string(objectid(VoltMat_Cheb.voltmat_cheb_square)),
            parent_module=string(parentmodule(VoltMat_Cheb.voltmat_cheb_square)),
            method_info=method_info,
            K_type=string(typeof(Kfun)),
            K_method=k_method,
            sample_max_abs_diff_vs_aging_K=sample_diff,
            t_VK=t_vk,
            VK_type=string(typeof(VK)),
            VK_rows=size(VK, 1),
            VK_cols=size(VK, 2),
            nnz=nnz(VK),
            status="success",
            notes=notes,
        )
    catch err
        return row(
            source=source,
            module_name=isdefined(Main, :VoltMat_Cheb) ? string(VoltMat_Cheb) : missing,
            module_objectid=isdefined(Main, :VoltMat_Cheb) ? string(objectid(VoltMat_Cheb)) : missing,
            function_objectid=isdefined(Main, :VoltMat_Cheb) ? string(objectid(VoltMat_Cheb.voltmat_cheb_square)) : missing,
            parent_module=isdefined(Main, :VoltMat_Cheb) ? string(parentmodule(VoltMat_Cheb.voltmat_cheb_square)) : missing,
            K_type=string(typeof(Kfun)),
            status="failed",
            notes="$(notes) error=$(sprint(showerror, err))",
        )
    end
end

rows = NamedTuple[]
push!(rows, collect_identity(
    "aging_direct_include",
    K;
    sample_diff=0.0,
    notes="Parent process direct include of src/VoltMat/voltmat_cheb.jl using the requested aging-style K.",
))

# ---------------------------------------------------------------------------
# common include style in a child process.
# ---------------------------------------------------------------------------

const CHILD_CODE = raw"""
using InteractiveUtils
using SparseArrays

const N = 1000
const tol = eps(Float64)
const CHILD_DEBUG_DIR = get(ENV, "EX1_DEBUG_DIR", @__DIR__)
include(joinpath(CHILD_DEBUG_DIR, "ex1_2nd_volterra_common.jl"))

aging_theta(x) = 1.0 + tanh(10.0 * x)
aging_tau_xy(x, y) = 1.0 / aging_theta((x + y) / 2)

function aging_K(x, y)
    tau = aging_tau_xy(x, y)
    return exp(-(x - y) / tau) / tau
end

const COLUMNS = [
    :source,
    :N,
    :tol,
    :module_name,
    :module_objectid,
    :function_objectid,
    :parent_module,
    :method_info,
    :K_type,
    :K_method,
    :sample_max_abs_diff_vs_aging_K,
    :t_VK,
    :VK_type,
    :VK_rows,
    :VK_cols,
    :nnz,
    :status,
    :notes,
]

function encode_field(x)
    s = string(x)
    s = replace(s, "%" => "%25")
    s = replace(s, "\t" => "%09")
    s = replace(s, "\n" => "%0A")
    s = replace(s, "\r" => "%0D")
    return s
end

function row(; kwargs...)
    data = Dict{Symbol,Any}(col => "" for col in COLUMNS)
    data[:N] = N
    data[:tol] = tol
    data[:sample_max_abs_diff_vs_aging_K] = NaN
    data[:t_VK] = NaN
    data[:status] = "not_run"
    data[:notes] = ""
    for (key, value) in kwargs
        data[key] = value
    end
    return NamedTuple{Tuple(COLUMNS)}(Tuple(data[col] for col in COLUMNS))
end

function emit(row)
    println("__VOLT_IDENTITY_ROW__\t" * join((encode_field(getproperty(row, col)) for col in COLUMNS), "\t"))
end

function collect_identity(source, Kfun; sample_diff=NaN, notes="")
    local VK
    try
        method_info = sprint(show, @which VoltMat_Cheb.voltmat_cheb_square(Kfun, N - 1; tol=tol))
        k_method = sprint(show, @which Kfun(0.1, -0.2))
        GC.gc()
        t_vk = @elapsed begin
            VK = VoltMat_Cheb.voltmat_cheb_square(Kfun, N - 1; tol=tol)
        end
        return row(
            source=source,
            module_name=string(VoltMat_Cheb),
            module_objectid=string(objectid(VoltMat_Cheb)),
            function_objectid=string(objectid(VoltMat_Cheb.voltmat_cheb_square)),
            parent_module=string(parentmodule(VoltMat_Cheb.voltmat_cheb_square)),
            method_info=method_info,
            K_type=string(typeof(Kfun)),
            K_method=k_method,
            sample_max_abs_diff_vs_aging_K=sample_diff,
            t_VK=t_vk,
            VK_type=string(typeof(VK)),
            VK_rows=size(VK, 1),
            VK_cols=size(VK, 2),
            nnz=nnz(VK),
            status="success",
            notes=notes,
        )
    catch err
        return row(
            source=source,
            module_name=isdefined(Main, :VoltMat_Cheb) ? string(VoltMat_Cheb) : "",
            module_objectid=isdefined(Main, :VoltMat_Cheb) ? string(objectid(VoltMat_Cheb)) : "",
            function_objectid=isdefined(Main, :VoltMat_Cheb) ? string(objectid(VoltMat_Cheb.voltmat_cheb_square)) : "",
            parent_module=isdefined(Main, :VoltMat_Cheb) ? string(parentmodule(VoltMat_Cheb.voltmat_cheb_square)) : "",
            K_type=string(typeof(Kfun)),
            status="failed",
            notes="$(notes) error=$(sprint(showerror, err))",
        )
    end
end

emit(collect_identity(
    "common_include",
    aging_K;
    sample_diff=0.0,
    notes="Child process include of ex1_2nd_volterra_common.jl using aging_K.",
))

sample_points = [(0.1, -0.2), (0.5, 0.0), (-0.5, -0.8)]
sample_diff = maximum(abs(aging_K(x, y) - ex1_kernel(x, y)) for (x, y) in sample_points)
emit(collect_identity(
    "common_ex1_kernel",
    ex1_kernel;
    sample_diff=sample_diff,
    notes="Child process include of ex1_2nd_volterra_common.jl using common ex1_kernel.",
))
"""

function collect_child_rows()
    cmd = setenv(
        `$(Base.julia_cmd()) --project=$(ROOT) -e $(CHILD_CODE)`,
        "EX1_DEBUG_DIR" => @__DIR__,
    )
    output = read(cmd, String)
    child_rows = NamedTuple[]
    for line in split(output, '\n')
        if startswith(line, "__VOLT_IDENTITY_ROW__\t")
            push!(child_rows, marker_to_row(line))
        end
    end
    isempty(child_rows) && error("child process produced no identity rows. Raw stdout:\n$(output)")
    return child_rows
end

append!(rows, collect_child_rows())

write_csv(OUT_CSV, rows)

function find_source(rows, source)
    idx = findfirst(row -> row.source == source, rows)
    idx === nothing && error("missing source $(source)")
    return rows[idx]
end

aging = find_source(rows, "aging_direct_include")
common = find_source(rows, "common_include")
common_kernel = find_source(rows, "common_ex1_kernel")

same_method = aging.method_info == common.method_info
same_module_name = aging.module_name == common.module_name
same_module_objectid = aging.module_objectid == common.module_objectid
same_function_objectid = aging.function_objectid == common.function_objectid
same_parent = aging.parent_module == common.parent_module

aging_t = parse(Float64, string(aging.t_VK))
common_t = parse(Float64, string(common.t_VK))
t_ratio = max(aging_t, common_t) / min(aging_t, common_t)

open(OUT_TXT, "w") do io
    println(io, "debug_compare_voltmat_module_identity")
    println(io, "N = $(N)")
    println(io, "tol = $(tol)")
    println(io)
    println(io, "Rows:")
    for r in rows
        println(io, "- source=$(r.source)")
        println(io, "  module_name=$(r.module_name)")
        println(io, "  module_objectid=$(r.module_objectid)")
        println(io, "  function_objectid=$(r.function_objectid)")
        println(io, "  parent_module=$(r.parent_module)")
        println(io, "  method_info=$(r.method_info)")
        println(io, "  K_type=$(r.K_type)")
        println(io, "  K_method=$(r.K_method)")
        println(io, "  sample_max_abs_diff_vs_aging_K=$(r.sample_max_abs_diff_vs_aging_K)")
        println(io, "  t_VK=$(r.t_VK)")
        println(io, "  VK_type=$(r.VK_type)")
        println(io, "  size=$(r.VK_rows)x$(r.VK_cols)")
        println(io, "  nnz=$(r.nnz)")
        println(io, "  status=$(r.status)")
        println(io, "  notes=$(r.notes)")
    end
    println(io)
    println(io, "Judgement:")
    println(io, "1. aging_direct_include and common_include method_info identical: $(same_method)")
    println(io, "2. module_name identical: $(same_module_name)")
    println(io, "3. module_objectid identical: $(same_module_objectid)")
    println(io, "4. function_objectid identical: $(same_function_objectid)")
    println(io, "5. parent_module identical: $(same_parent)")
    println(io, "6. t_VK ratio max/min: $(@sprintf("%.3f", t_ratio))")
    println(io, "7. common_ex1_kernel sample_max_abs_diff_vs_aging_K: $(common_kernel.sample_max_abs_diff_vs_aging_K)")
    println(io)
    println(io, "Interpretation:")
    if same_method
        println(io, "- aging_direct_include and common_include resolve to the same source method string.")
    else
        println(io, "- aging_direct_include and common_include resolve to different method strings.")
    end
    println(io, "- The common row is collected in a child Julia process to avoid replacing Main.VoltMat_Cheb in the parent process.")
    println(io, "- Therefore module_objectid/function_objectid are process-local object identities; equality across these two rows is not expected and cannot prove shared object identity.")
    println(io, "- If method_info is identical but objectids differ, the scripts are using independent module instances loaded from the same source method.")
    println(io, "- If t_VK differs while method_info is identical, check K_type, kernel values, package/global state, and process-local module state.")
end

println("Wrote ", OUT_TXT)
println("Wrote ", OUT_CSV)
