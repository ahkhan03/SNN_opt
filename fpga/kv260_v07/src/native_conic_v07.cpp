// Native emulator harness for the resident v07 cone path.  The binary input
// format is private to run_cone_parity.sh and lets one resident top exercise
// ball, friction, and mixed anchor fixtures.
#include "v07_abi.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

extern "C" void snn_qp_v07(
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, std::uint32_t*, std::uint32_t*,
    std::uint32_t*, std::uint32_t*, long long*, unsigned long long*,
    volatile const std::uint32_t*, volatile std::uint32_t*, int, int, int, int,
    int, int, int, int, double, double, int, int, int, double, int, double,
    const std::uint32_t*, const std::uint32_t*, const std::uint32_t*,
    const double*, const double*, const double*, int);

template <typename T> bool get(std::ifstream& in, T& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    std::uint32_t magic=0, n=0, m=0, cone_count=0;
    if (!get(in, magic) || magic != UINT32_C(0x56303743) ||
        !get(in,n) || !get(in,m) || !get(in,cone_count) || n > 1024 || m > 1024 ||
        cone_count > 64) return 2;
    double k0=0, ctol=0, lower=0, upper=0;
    std::int32_t iters=0, projmax=0, has_lower=0, has_upper=0;
    if (!get(in,k0) || !get(in,ctol) || !get(in,iters) || !get(in,projmax) ||
        !get(in,has_lower) || !get(in,lower) || !get(in,has_upper) || !get(in,upper)) return 2;
    std::vector<double> A(n*n), C(m*n), G(m*m), cns(m), scale(m), b(n), d(m), x0(n);
    auto read_vec = [&](auto& v) { return v.empty() || static_cast<bool>(in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()*sizeof(v[0])))); };
    if (!read_vec(A) || !read_vec(C) || !read_vec(G) || !read_vec(cns) || !read_vec(scale) ||
        !read_vec(b) || !read_vec(d) || !read_vec(x0)) return 2;
    std::vector<std::uint32_t> kinds(cone_count), offsets(cone_count), lengths(cone_count);
    std::vector<double> radii(cone_count), mus(cone_count), centers(cone_count);
    if (!read_vec(kinds) || !read_vec(offsets) || !read_vec(lengths) || !read_vec(radii) ||
        !read_vec(mus) || !read_vec(centers)) return 2;
    std::vector<std::uint32_t> a_ddr(n*n), c_ddr(m*n), ct_ddr(n*m), g_ddr(m*m);
    std::vector<long long> raw(n);
    std::array<unsigned long long, snn_v07::TELEMETRY_WORDS> telemetry{};
    std::array<std::uint32_t, snn_v07::MAILBOX_WORDS> mb_in{}, mb_out{};
    const double* Cptr = C.empty() ? nullptr : C.data();
    const double* Gptr = G.empty() ? nullptr : G.data();
    const double* cnsptr = cns.empty() ? nullptr : cns.data();
    const double* scaleptr = scale.empty() ? nullptr : scale.data();
    const double* dptr = d.empty() ? nullptr : d.data();
    const std::uint32_t* kindptr = kinds.empty() ? nullptr : kinds.data();
    const std::uint32_t* offptr = offsets.empty() ? nullptr : offsets.data();
    const std::uint32_t* lenptr = lengths.empty() ? nullptr : lengths.data();
    const double* radiusptr = radii.empty() ? nullptr : radii.data();
    const double* muptr = mus.empty() ? nullptr : mus.data();
    const double* centerptr = centers.empty() ? nullptr : centers.data();
    auto call = [&](int command, std::uint32_t seq) {
        mb_in[snn_v07::MAILBOX_SEQUENCE] = seq;
        mb_in[snn_v07::MAILBOX_COMMAND] = static_cast<std::uint32_t>(command);
        snn_qp_v07(A.data(), Cptr, Gptr, cnsptr, scaleptr, b.data(), dptr, x0.data(),
            a_ddr.data(), c_ddr.data(), ct_ddr.data(), g_ddr.data(), raw.data(), telemetry.data(),
            mb_in.data(), mb_out.data(), command, snn_v07::ONESHOT, snn_v07::AUTO,
            snn_v07::HOST_X0, 0, snn_v07::HOLD_TAIL, static_cast<int>(n), static_cast<int>(m),
            k0, ctol, iters, projmax, has_lower, lower, has_upper, upper,
            kindptr, offptr, lenptr, radiusptr, muptr, centerptr, static_cast<int>(cone_count));
    };
    call(snn_v07::CONFIGURE, 1);
    if (mb_out[snn_v07::OUT_ERROR_CODE] != snn_v07::ERR_OK) {
        std::fprintf(stderr, "configure error=%u\n", mb_out[snn_v07::OUT_ERROR_CODE]);
        return 3;
    }
    call(snn_v07::SOLVE, 2);
    std::ofstream out(argv[2]); if (!out) return 2;
    out << "{\"schema\":\"v07-native-resident-cone-v2\",\"cone_count\":" << cone_count
        << ",\"events\":" << telemetry[5] << ",\"status\":" << telemetry[1]
        << ",\"iterations\":" << telemetry[2] << ",\"digest\":\"";
    out << std::hex << telemetry[10] << std::dec << "\",\"raw\":[";
    for (std::size_t i=0; i<raw.size(); ++i) out << (i ? "," : "") << raw[i];
    out << "],\"telemetry\":[";
    for (unsigned i=0;i<telemetry.size();++i) out << (i ? "," : "") << telemetry[i];
    out << "]}\n";
    std::printf("events=%llu status=%llu\n", telemetry[5], telemetry[1]);
    return 0;
}
