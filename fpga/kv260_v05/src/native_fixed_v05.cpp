#include "bench_support.hpp"
#include "msrp_bundle.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void snn_qp_v05(
    const double* A_in, const double* b_in, const double* C_in,
    const double* d_in, const double* cns_in, const double* row_scale_in,
    const double* G_in, const double* x0_in, long long* x_raw_out,
    unsigned long long* telemetry_out, int n, int m, double k0_f,
    double ctol_f, int n_iters, int projmax, int has_lower, double lower_f,
    int has_upper, double upper_f);

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <binary64-problem.bin> <output.bin>\n",
                     argv[0]);
        return 2;
    }
    const msrp_v05::Problem q = msrp_v05::load_problem(argv[1]);
    if (q.source_is_float32) {
        std::fprintf(stderr, "v0.5 hardware path requires binary64 input\n");
        return 2;
    }
    std::vector<long long> raw(q.n, 0);
    std::array<unsigned long long, msrp_v05::TELEMETRY_WORDS> meta{};

    const std::uint64_t started = msrp_v05::monotonic_raw_ns();
    snn_qp_v05(
        q.A.data(), q.b.data(), q.C.data(), q.d.data(),
        q.c_norms_sq.data(), q.row_scale.data(), q.G.data(), q.x0.data(),
        raw.data(), meta.data(), q.n, q.m, q.k0, q.constraint_tol,
        q.iterations, q.projection_cap, q.has_lower ? 1 : 0, q.lower,
        q.has_upper ? 1 : 0, q.upper);
    const std::uint64_t finished = msrp_v05::monotonic_raw_ns();

    std::vector<std::int64_t> signed_raw(raw.begin(), raw.end());
    std::array<std::uint64_t, msrp_v05::TELEMETRY_WORDS> unsigned_meta{};
    for (std::size_t i = 0; i < unsigned_meta.size(); ++i)
        unsigned_meta[i] = meta[i];
    msrp_v05::write_fixed_output(argv[2], q.n, signed_raw, unsigned_meta);

    std::printf(
        "{\"schema\":\"msrp-v05-native-fixed-v1\",\"n\":%d,\"m\":%d,"
        "\"status\":%llu,\"iterations\":%llu,\"events\":%llu,"
        "\"digest\":\"%016llx\",\"elapsed_seconds\":%.17g}\n",
        q.n, q.m, meta[1], meta[3], meta[5], meta[10],
        static_cast<double>(finished - started) * 1e-9);
    return meta[1] == 0 ? 0 : 3;
}
