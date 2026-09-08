using Profile

include(joinpath(@__DIR__, "..", "..", "src", "VoltOpt.jl"))
const W = VoltOpt.Weak_VoltMat_Cheb

N = parse(Int, get(ENV, "WEAK_PROFILE_N", "4000"))
I = parse(Int, get(ENV, "WEAK_PROFILE_I", "31"))
J = parse(Int, get(ENV, "WEAK_PROFILE_J", string(I)))
C = ones(Float64, I + 1, J + 1)
M0 = W.initial_moments_fast(0.5, 0.0, 0.5, J, N)
Mcomb = W._precombine_kernel_moments(C, M0, N, J)

function assemble(Mcomb, N, I)
    return W._assemble_eta1_ellmajor_scatter_from_Mcomb(
        copy(Mcomb), N, 0.0;
        output_format=:packed_lower_hessenberg,
        lower_bw=I + 1,
    )
end

assemble(Mcomb, N, I)
Profile.clear()
Profile.@profile assemble(Mcomb, N, I)
Profile.print(format=:flat, sortedby=:count, mincount=20)
