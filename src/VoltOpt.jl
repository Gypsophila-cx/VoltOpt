module VoltOpt

include("CompVolt/compVolt.jl")
include("chebtechs.jl")
include("RestMat/restmat_cheb.jl")
include("RestMat/restmat_jacobi.jl")
include("VoltMat/voltmat_cheb.jl")
include("VoltMat/voltmat_leg.jl")
include("VoltMat/voltmat_jacobi.jl")

const VoltMat_Jacobi = Voltmat_Jacobi

const voltmat_cheb = VoltMat_Cheb.voltmat_cheb
const voltmat_cheb_square = VoltMat_Cheb.voltmat_cheb_square
const voltmat_square_from_coeffmat_fft_convolution =
    VoltMat_Cheb.voltmat_square_from_coeffmat_fft_convolution

const voltmat_leg = VoltMat_Leg.voltmat_leg
const legendre_interp_coeffs_Khat = VoltMat_Leg.legendre_interp_coeffs_Khat
const adaptive_legendre_interp_coeffs_Khat =
    VoltMat_Leg.adaptive_legendre_interp_coeffs_Khat
const legendre_volterra_symband_from_C =
    VoltMat_Leg.legendre_volterra_symband_from_C
const legendre_values = VoltMat_Leg.legendre_values
const legendre_values! = VoltMat_Leg.legendre_values!
const gausslegendre_rule = VoltMat_Leg.gausslegendre_rule

const voltmat_jacobi = Voltmat_Jacobi.voltmat_jacobi
const voltmat_jacobi_full = Voltmat_Jacobi.voltmat_jacobi_full
const jacobi_interp_coeffs_Khat = Voltmat_Jacobi.jacobi_interp_coeffs_Khat
const adaptive_jacobi_interp_coeffs_Khat =
    Voltmat_Jacobi.adaptive_jacobi_interp_coeffs_Khat
const jacobi_volterra_topdense_from_C =
    Voltmat_Jacobi.jacobi_volterra_topdense_from_C
const jacobi_values = Voltmat_Jacobi.jacobi_values
const jacobi_values! = Voltmat_Jacobi.jacobi_values!
const gaussjacobi_nodes = Voltmat_Jacobi.gaussjacobi_nodes

export vals2cheb,
    cheb2vals_lobatto_fft,
    cheb2vals_2d_fft,
    cheb2_vals2coeffs,
    adaptive_cheb2_coeffmat,
    truncate_coeffs_robust,
    find_chop_index,
    cheb_intmat,
    cheb_multmat,
    restmat_cheb,
    restmat_jacobi,
    multrestrict,
    compVolt,
    compVolt_fast,
    compVolt_matrix,
    compVolt_matrix_ultra,
    clenshaw_curtis_fft_weights,
    clenshaw_curtis_analytical_weights,
    CompVoltGrid,
    compVolt_nonlinear_fast,
    compVolt_nonlinear,
    CompVoltNewtonGMRESCache,
    update_newton_cache!,
    compVolt_linearized_integral!,
    compVolt_newton_jv!,
    compVolt_newton_jv_operator,
    compVolt_nonlinear_residual,
    VoltMat_Cheb,
    VoltMat_Leg,
    Voltmat_Jacobi,
    VoltMat_Jacobi,
    voltmat_cheb,
    voltmat_cheb_square,
    voltmat_square_from_coeffmat_fft_convolution,
    voltmat_leg,
    legendre_interp_coeffs_Khat,
    adaptive_legendre_interp_coeffs_Khat,
    legendre_volterra_symband_from_C,
    legendre_values,
    legendre_values!,
    gausslegendre_rule,
    voltmat_jacobi,
    voltmat_jacobi_full,
    jacobi_interp_coeffs_Khat,
    adaptive_jacobi_interp_coeffs_Khat,
    jacobi_volterra_topdense_from_C,
    jacobi_values,
    jacobi_values!,
    gaussjacobi_nodes

end # module VoltOpt
