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
    x_nodes = [-cos(π * i / (Mx - 1)) for i in 0:(Mx - 1)]
    y_nodes = [-cos(π * j / (My - 1)) for j in 0:(My - 1)]
    return clenshaw_valuemat(c, x_nodes, y_nodes)
end

function clenshaw_valuemat(
    c::Vector{Float64},
    x_nodes::AbstractVector{Float64},
    y_nodes::AbstractVector{Float64},
)
    Mx = length(x_nodes)
    My = length(y_nodes)
    U_grid = Matrix{Float64}(undef, Mx, My)
    x_plus_1 = x_nodes .+ 1.0

    # The first matrix index is the outer x node; the second is the reference
    # integration node. Keep the second index outermost for column-major SIMD.
    for j in 1:My
        y_mapped = (y_nodes[j] + 1.0) / 2.0
        @inbounds @simd for i in 1:Mx
            z = x_plus_1[i] * y_mapped - 1.0
            U_grid[i, j] = clenshaw_inline(z, c)
        end
    end

    return x_nodes, y_nodes, U_grid
end
