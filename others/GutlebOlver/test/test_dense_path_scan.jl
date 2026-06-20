using Test

@testset "dense path scan" begin
    pkgroot = normpath(joinpath(@__DIR__, ".."))
    dirs = (
        joinpath(pkgroot, "src"),
        joinpath(pkgroot, "examples"),
        joinpath(pkgroot, "test"),
    )
    banned = r"\b(Matrix|Array|collect)\("
    violations = String[]

    for dir in dirs
        for (root, _, files) in walkdir(dir)
            for file in files
                endswith(file, ".jl") || continue
                path = joinpath(root, file)
                text = read(path, String)
                if !isnothing(match(banned, text))
                    push!(violations, path)
                end
            end
        end
    end

    @test isempty(violations)
end
