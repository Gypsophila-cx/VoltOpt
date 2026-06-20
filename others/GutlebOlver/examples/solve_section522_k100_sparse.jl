using GutlebOlver

function main()
    N = SECTION522_ACCEPTANCE_N
    kerneldepth = SECTION522_ACCEPTANCE_KERNELDEPTH
    k = SECTION522_K

    println("Section 5.2.2 k=100 strict-band/semi-banded sparse solve")
    println("Equation: u'(x) - int_0^x y*exp(x^2)*u(y)dy = g(x,100), u(0)=0")
    println("Exact solution: u(x)=atan(100*x)")
    println("N = ", N)
    println("kerneldepth = ", kerneldepth)
    println()

    row = solve_section522_k100_sparse(; N=N, kerneldepth=kerneldepth, k=k)
    print_section522_result(row)
    println()
    print_section522_summary(row)
    return row
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
