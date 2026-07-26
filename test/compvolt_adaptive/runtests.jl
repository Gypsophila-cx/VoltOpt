using Test
using VoltOpt

include(joinpath(@__DIR__, "oscillation_common.jl"))
using .CompVoltAdaptiveOscillation

function legacy_compvolt_nonlinear_fixed(K, cu::AbstractVector, M::Int)
    cpad = zeros(eltype(cu), M + 1)
    @inbounds for i in eachindex(cu)
        cpad[i] = iseven(i) ? -cu[i] : cu[i]
    end

    C = VoltOpt.multrestrict(cpad)
    Uvals = VoltOpt.cheb2vals_2d_fft(C)
    x = [cos(pi * i / M) for i in M:-1:0]
    w = VoltOpt.clenshaw_curtis_analytical_weights(M)
    Fval = zeros(Float64, M + 1)

    @inbounds for j in 2:(M + 1)
        scale = (x[j] + 1) / 2
        shift = (x[j] - 1) / 2
        s = 0.0
        for i in 1:(M + 1)
            y = scale * x[i] + shift
            s += w[i] * K(x[j], y, Uvals[i, j])
        end
        Fval[j] = scale * s
    end

    coefficients = VoltOpt.vals2cheb(Fval)
    @inbounds for i in 2:2:length(coefficients)
        coefficients[i] = -coefficients[i]
    end
    return coefficients
end

function captured_exception(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

@testset "adaptive nonlinear CompVolt" begin
    @testset "backward-compatible call and fixed M" begin
        constant_kernel(x, y, u) = one(x)
        cu = [1.0, 0.0, 0.0]

        legacy_default = legacy_compvolt_nonlinear_fixed(constant_kernel, cu, 2)
        default_result = compVolt_nonlinear_fast(constant_kernel, cu)
        @test default_result isa Vector{Float64}
        @test default_result[1:length(legacy_default)] ≈ legacy_default atol=5e-14 rtol=5e-14
        @test maximum(abs, default_result[(length(legacy_default) + 1):end]) <= 5e-14

        default_info = compVolt_nonlinear_fast(
            constant_kernel, cu; return_info=true,
        )
        @test propertynames(default_info) == (
            :coefficients, :M, :eta, :theta, :refinements, :converged,
        )
        @test default_info.refinements >= 1
        @test length(default_info.coefficients) == default_info.M + 1
        @test length(default_info.coefficients) > length(cu)
        @test default_info.M + 1 == length(default_info.coefficients)

        truncated = compVolt_nonlinear_fast(
            constant_kernel, cu; truncate_output=true,
        )
        @test length(truncated) == length(cu)
        # truncate_output is a fixed-length slice, not tolerance-based chopping.
        @test truncated ≈ default_result[1:length(cu)] atol=5e-14 rtol=5e-14

        K(x, y, u) = exp(0.2x - 0.1y) + u^2
        cu_nonlinear = [0.5, -0.2, 0.1, 0.05]
        M = 8
        legacy_fixed = legacy_compvolt_nonlinear_fixed(K, cu_nonlinear, M)
        fixed = compVolt_nonlinear_fast(K, cu_nonlinear; M=M, return_info=true)

        @test fixed.M == M
        @test length(fixed.coefficients) == M + 1
        @test fixed.refinements == 0
        @test fixed.eta === nothing
        @test fixed.theta === nothing
        @test fixed.converged
        @test fixed.coefficients ≈ legacy_fixed atol=2e-13 rtol=2e-13

        legacy_keyword = compVolt_nonlinear_fast(K, cu_nonlinear; Nquad=M)
        legacy_grid = compVolt_nonlinear_fast(K, cu_nonlinear; grid=CompVoltGrid(M))
        alias_result = compVolt_nonlinear(K, cu_nonlinear; M=M)
        @test legacy_keyword ≈ legacy_fixed atol=2e-13 rtol=2e-13
        @test legacy_grid ≈ legacy_fixed atol=2e-13 rtol=2e-13
        @test alias_result ≈ fixed.coefficients atol=2e-13 rtol=2e-13

        odd_fixed = compVolt_nonlinear_fast(K, cu_nonlinear; M=5, return_info=true)
        legacy_odd = compVolt_nonlinear_fast(
            K, cu_nonlinear; Nquad=5, return_info=true,
        )
        @test odd_fixed.M == 5
        @test length(odd_fixed.coefficients) == 6
        @test legacy_odd.M == 6
        @test length(legacy_odd.coefficients) == 7

        @test_throws ArgumentError compVolt_nonlinear_fast(K, cu_nonlinear; M=1)
        @test_throws ArgumentError compVolt_nonlinear_fast(
            K, cu_nonlinear; M=M, Nquad=M,
        )
    end

    @testset "smooth kernel accepts the first comparison" begin
        calls = Ref(0)
        function counted_constant_kernel(x, y, u)
            calls[] += 1
            return one(x)
        end

        tol = 1e-10
        info = compVolt_nonlinear_fast(
            counted_constant_kernel,
            [1.0, 0.0, 0.0];
            M0=2,
            tol=tol,
            Mmax=16,
            return_info=true,
        )

        @test info.converged
        @test info.M == 4
        @test info.refinements == 1
        @test info.eta <= tol
        @test info.theta <= tol
        @test info.coefficients ≈ [1.0, 1.0, 0.0, 0.0, 0.0] atol=5e-14 rtol=5e-14
        # One M=2 evaluation and one M=4 evaluation; the M=2 level is not repeated.
        @test calls[] == (1 + 2 * 3) + (1 + 4 * 5)
    end

    @testset "oscillatory kernel refines and matches fixed references" begin
        oscillatory_kernel(x, y, u) = exp(8y)
        cu = [1.0]
        adaptive = compVolt_nonlinear_fast(
            oscillatory_kernel,
            cu;
            M0=1,
            tol=1e-9,
            Mmax=128,
            return_info=true,
        )

        @test adaptive.converged
        @test adaptive.refinements > 1
        @test adaptive.M > 2
        @test max(adaptive.eta, adaptive.theta) <= 1e-9

        same_M = compVolt_nonlinear_fast(oscillatory_kernel, cu; M=adaptive.M)
        high_reference = compVolt_nonlinear_fast(oscillatory_kernel, cu; M=128)
        @test adaptive.coefficients ≈ same_M atol=2e-11 rtol=2e-12
        @test adaptive.coefficients ≈ high_reference[1:(adaptive.M + 1)] atol=2e-10 rtol=2e-11
        @test maximum(abs, high_reference[(adaptive.M + 2):end]) <= 2e-10
    end

    @testset "limits report nonconvergence" begin
        oscillatory_kernel(x, y, u) = exp(8y)
        kwargs = (; M0=1, tol=0.0, Mmax=2)

        failed = compVolt_nonlinear_fast(
            oscillatory_kernel, [1.0]; kwargs..., return_info=true,
        )
        @test !failed.converged
        @test failed.M == 2
        @test failed.refinements == 1
        @test failed.eta !== nothing
        @test failed.theta !== nothing

        err = captured_exception(
            () -> compVolt_nonlinear_fast(oscillatory_kernel, [1.0]; kwargs...),
        )
        @test err isa ErrorException
        message = sprint(showerror, err)
        @test occursin("final M=2", message)
        @test occursin("eta=", message)
        @test occursin("theta=", message)

        refinement_limited = compVolt_nonlinear_fast(
            oscillatory_kernel,
            [1.0];
            M0=1,
            tol=0.0,
            max_refinements=1,
            return_info=true,
        )
        @test !refinement_limited.converged
        @test refinement_limited.refinements == 1
    end

    @testset "non-finite values fail at their numerical stage" begin
        nan_error = captured_exception(
            () -> compVolt_nonlinear_fast((x, y, u) -> NaN, [1.0]; M=4),
        )
        @test nan_error isa ErrorException
        nan_message = sprint(showerror, nan_error)
        @test occursin("kernel function value", nan_message)
        @test occursin("fixed-grid mode", nan_message)
        @test occursin("M=4", nan_message)

        new_fine_node = sqrt(0.5)
        function inf_on_fine_grid(x, y, u)
            if abs(abs(x) - new_fine_node) <= 32eps(Float64)
                return Inf
            end
            return one(x)
        end
        inf_error = captured_exception(
            () -> compVolt_nonlinear_fast(
                inf_on_fine_grid,
                [1.0, 0.0, 0.0];
                M0=2,
                tol=1e-10,
                Mmax=16,
            ),
        )
        @test inf_error isa ErrorException
        inf_message = sprint(showerror, inf_error)
        @test occursin("kernel function value", inf_message)
        @test occursin("adaptive fine-grid", inf_message)
        @test occursin("fine-grid M=4", inf_message)
    end

    @testset "floating-point backend" begin
        info32 = compVolt_nonlinear_fast(
            (x, y, u) -> one(x), Float32[1]; M=4, return_info=true,
        )
        @test eltype(info32.coefficients) === Float32

        err = captured_exception(
            () -> compVolt_nonlinear_fast((x, y, u) -> u, BigFloat[1, 2]; M=2),
        )
        @test err isa ArgumentError
        @test occursin("BigFloat is not supported", sprint(showerror, err))
    end

    @testset "oscillation-grid regression" begin
        cu = CompVoltAdaptiveOscillation.input_coefficients()
        regression_omegas = (0, 40, 160)
        infos = NamedTuple[]
        relative_errors = Float64[]

        @test length(cu) == CompVoltAdaptiveOscillation.INPUT_LENGTH
        for omega in regression_omegas
            info = CompVoltAdaptiveOscillation.adaptive_solution(omega, cu)
            Mref, reference = CompVoltAdaptiveOscillation.fixed_reference(
                omega, info.M, cu,
            )
            error_inf, relative_error = CompVoltAdaptiveOscillation.comparison_error(
                info.coefficients, reference,
            )
            push!(infos, info)
            push!(relative_errors, relative_error)

            @test info.converged
            @test info.M + 1 == length(info.coefficients)
            @test max(info.eta, info.theta) <= CompVoltAdaptiveOscillation.TOLERANCE
            @test Mref > info.M
            @test isfinite(error_inf)
            @test relative_error <= 5e-10
        end

        accepted_degrees = [info.M for info in infos]
        @test length(unique(accepted_degrees)) >= 2
        @test accepted_degrees[end] > accepted_degrees[1]
        @test infos[end].refinements > 1
        @test all(isfinite, relative_errors)
    end
end
