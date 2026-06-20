using GutlebOlver

function main()
    N = 32
    println("Manufactured K=1 second-kind strict-band sparse solve")
    println("Equation: u(x) + int_0^x u(y)dy = 1 + 2x + x^2/2")
    println("Exact solution: u(x)=1+x")
    println("N = ", N)
    println()

    row = solve_simple_secondkind_sparse(; N=N)
    println("status = ", row.status)
    println("rhs_validation_error = ", row.rhs_validation_error)
    println("body_residual_relative = ", row.body_residual_relative)
    println("max_abs_error = ", row.max_abs_error)
    println("raw nnz/bw = ", row.raw_operator_nnz, " / ", row.raw_operator_bandwidth)
    println("body nnz/bw = ", row.body_operator_nnz, " / ", row.body_operator_bandwidth)
    println("system nnz/bw = ", row.final_system_nnz, " / ", row.full_bandwidth)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
