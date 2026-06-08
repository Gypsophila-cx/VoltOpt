using LinearAlgebra
using SparseVolterraNew
using Test

Kcorrectness(x, y) = exp(-x * y)

@testset "strict-banded target matrix agrees with dense Clenshaw construction" begin
    for N in (16, 32, 64)
        for kerneldepth in (4, 6)
            dense = triVolterraFullKernelOpP01(Kcorrectness, N, true, kerneldepth)
            banded = triVolterraFullKernelOpP01_strictband(Kcorrectness, N, true, kerneldepth)

            @test size(banded) == size(dense) == (N, N)
            @test all(isfinite, banded)
            @test all(isfinite, dense)

            # The strict-banded code is a compatibility migration of the
            # Gutleb--Olver banded assembly target.  This comparison checks
            # that it reproduces the original dense Clenshaw assembly for the
            # same projected kernel, rather than requiring equality to an
            # independent infinite-dimensional integral operator.
            err = norm(Matrix(banded) - Matrix(dense), Inf)
            ref = max(norm(Matrix(dense), Inf), eps(Float64))
            @test err / ref < 1.0e-8
        end
    end
end
