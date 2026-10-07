#include "v07_abi.hpp"
#include "v07_cone_io.hpp"
#include "msrp_bundle.hpp"
#if defined(V07_MOCK_NATIVE_HOOK)
extern "C" void snn_qp_v07(
    const double* A_cfg, const double* C_cfg, const double* G_cfg,
    const double* cns_cfg, const double* row_scale_cfg, const double* b_in,
    const double* d_in, const double* x0_in, std::uint32_t* A_ddr,
    std::uint32_t* C_ddr, std::uint32_t* Ct_ddr, std::uint32_t* G_ddr,
    long long* x_raw_out, unsigned long long* telemetry_out,
    volatile const std::uint32_t* mb_in, volatile std::uint32_t* mb_out,
    int command, int launch_mode, int route_mode, int start_mode,
    int shift_stride, int tail_policy, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f, const std::uint32_t* cone_kinds, const std::uint32_t* cone_offsets, const std::uint32_t* cone_lengths, const double* cone_radii, const double* cone_mus, const double* cone_centers, int cone_count);

void v07_mock_native_invoke(
    const msrp_v05::Problem* problem, const double* b_in, const double* d_in,
    const double* x0_in, std::uint32_t* a_ddr, std::uint32_t* c_ddr,
    std::uint32_t* ct_ddr, std::uint32_t* g_ddr, long long* raw_out,
    unsigned long long* telemetry_out, std::uint32_t* mb_in,
    std::uint32_t* mb_out, int /*command*/, const snn_v07::ConeImage* cones) {
    // The mailbox fields have already been published by V07Session.  SERVE
    // makes the native dispatcher consume exactly those fields, while the
    // ONESHOT launch keeps this hook independent of an XRT scheduler.  The
    // kernel's static resident state still persists between calls, so this
    // exercises configure/refresh/solve state transitions and wrap values.
    snn_qp_v07(
        problem->A.data(), problem->C.data(), problem->G.data(),
        problem->c_norms_sq.data(), problem->row_scale.data(), b_in, d_in,
        x0_in, a_ddr, c_ddr, ct_ddr, g_ddr, raw_out, telemetry_out, mb_in,
        mb_out, snn_v07::SERVE, snn_v07::ONESHOT, snn_v07::AUTO,
        snn_v07::HOST_X0, 0, snn_v07::HOLD_TAIL, problem->n, problem->m,
        problem->k0, problem->constraint_tol, problem->iterations,
        problem->projection_cap, problem->has_lower ? 1 : 0, problem->lower,
        problem->has_upper ? 1 : 0, problem->upper, cones->kinds.data(), cones->offsets.data(), cones->lengths.data(),
        cones->radii.data(), cones->mus.data(), cones->centers.data(), cones->count);
}
#endif
