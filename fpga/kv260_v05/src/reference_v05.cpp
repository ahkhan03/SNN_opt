#include "canonical_solve.hpp"
#include "msrp_bundle.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <problem.bin> <output.bin>\n", argv[0]);
        return 2;
    }
    const msrp_v05::Problem problem = msrp_v05::load_problem(argv[1]);
    const msrp_v05::SolveResult result =
        msrp_v05::solve_canonical(problem);
    const auto meta = msrp_v05::telemetry(problem, result);
    msrp_v05::write_double_output(argv[2], problem.n, result.x, meta);
    std::printf(
        "{\"schema\":\"msrp-v05-double-reference-v1\","
        "\"source_precision\":\"%s\",\"n\":%d,\"m\":%d,"
        "\"status\":%d,\"iterations\":%d,\"events\":%llu,"
        "\"digest\":\"%016llx\"}\n",
        problem.source_is_float32 ? "float32-bundle" : "binary64-bundle",
        problem.n, problem.m, result.status, result.iterations_executed,
        static_cast<unsigned long long>(result.observer.total),
        static_cast<unsigned long long>(result.observer.digest));
    return result.status == 0 ? 0 : 3;
}
