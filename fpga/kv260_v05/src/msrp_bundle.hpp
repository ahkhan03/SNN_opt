#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace msrp_v05 {

constexpr std::array<char, 8> FLOAT_MAGIC{
    'M', 'S', 'R', 'P', 'H', 'W', '1', '\0'};
constexpr std::array<char, 8> DOUBLE_MAGIC{
    'M', 'S', 'R', 'P', 'D', 'L', '1', '\0'};
constexpr std::array<char, 8> FIXED_OUTPUT_MAGIC{
    'M', 'S', 'R', 'P', 'F', 'X', '1', '\0'};
constexpr std::array<char, 8> DOUBLE_OUTPUT_MAGIC{
    'M', 'S', 'R', 'P', 'D', 'P', '1', '\0'};
constexpr std::uint32_t BUNDLE_VERSION = 1;
constexpr std::size_t TELEMETRY_WORDS = 16;
constexpr std::uint64_t TELEMETRY_MAGIC = UINT64_C(0x4d53525056303531);
constexpr std::uint64_t NO_CANDIDATE = UINT64_MAX;

struct Problem {
    int n = 0;
    int m = 0;
    int iterations = 0;
    int projection_cap = 0;
    bool has_lower = false;
    bool has_upper = false;
    double k0 = 0.0;
    double constraint_tol = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    bool source_is_float32 = false;
    std::vector<double> A;
    std::vector<double> b;
    std::vector<double> C;
    std::vector<double> d;
    std::vector<double> c_norms_sq;
    std::vector<double> row_scale;
    std::vector<double> G;
    std::vector<double> x0;
};

template <class T>
inline void read_exact(std::FILE* file, T* out, std::size_t count,
                       const char* label) {
    if (count == 0) return;
    if (std::fread(out, sizeof(T), count, file) != count) {
        std::fprintf(stderr, "short read while reading %s\n", label);
        std::exit(2);
    }
}

template <class T>
inline void write_exact(std::FILE* file, const T* values, std::size_t count,
                        const char* label) {
    if (count == 0) return;
    if (std::fwrite(values, sizeof(T), count, file) != count) {
        std::fprintf(stderr, "short write while writing %s\n", label);
        std::exit(2);
    }
}

inline bool equal_magic(const char* observed,
                        const std::array<char, 8>& expected) {
    return std::memcmp(observed, expected.data(), expected.size()) == 0;
}

template <class Source>
inline std::vector<double> read_vector(std::FILE* file, std::size_t count,
                                       const char* label) {
    std::vector<Source> source(count);
    read_exact(file, source.data(), source.size(), label);
    return std::vector<double>(source.begin(), source.end());
}

inline Problem load_problem(const std::string& path,
                            bool require_float32 = false) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }

    char magic[8];
    read_exact(file, magic, 8, "bundle magic");
    const bool is_float = equal_magic(magic, FLOAT_MAGIC);
    const bool is_double = equal_magic(magic, DOUBLE_MAGIC);
    if (!is_float && !is_double) {
        std::fprintf(stderr, "unrecognized problem-bundle magic in %s\n",
                     path.c_str());
        std::exit(2);
    }
    if (require_float32 && !is_float) {
        std::fprintf(stderr, "hardware path requires a float32 bundle\n");
        std::exit(2);
    }

    std::uint32_t header[7];
    read_exact(file, header, 7, "bundle header");
    if (header[0] != BUNDLE_VERSION) {
        std::fprintf(stderr, "unsupported bundle version %u\n", header[0]);
        std::exit(2);
    }

    Problem problem;
    problem.n = static_cast<int>(header[1]);
    problem.m = static_cast<int>(header[2]);
    problem.iterations = static_cast<int>(header[3]);
    problem.projection_cap = static_cast<int>(header[4]);
    problem.has_lower = header[5] != 0;
    problem.has_upper = header[6] != 0;
    problem.source_is_float32 = is_float;
    if (problem.n <= 0 || problem.n > 64 || problem.m <= 0 ||
        problem.m > 64 || problem.iterations <= 0 ||
        problem.projection_cap <= 0) {
        std::fprintf(stderr, "problem dimensions or horizons are out of range\n");
        std::exit(2);
    }

    if (is_float) {
        float scalars[4];
        read_exact(file, scalars, 4, "float32 scalars");
        problem.k0 = scalars[0];
        problem.constraint_tol = scalars[1];
        problem.lower = scalars[2];
        problem.upper = scalars[3];
        problem.A = read_vector<float>(
            file, static_cast<std::size_t>(problem.n) * problem.n, "A");
        problem.b = read_vector<float>(file, problem.n, "b");
        problem.C = read_vector<float>(
            file, static_cast<std::size_t>(problem.m) * problem.n, "C");
        problem.d = read_vector<float>(file, problem.m, "d");
        problem.c_norms_sq =
            read_vector<float>(file, problem.m, "c_norms_sq");
        problem.row_scale =
            read_vector<float>(file, problem.m, "row_scale");
        problem.G = read_vector<float>(
            file, static_cast<std::size_t>(problem.m) * problem.m, "G");
        problem.x0 = read_vector<float>(file, problem.n, "x0");
    } else {
        double scalars[4];
        read_exact(file, scalars, 4, "binary64 scalars");
        problem.k0 = scalars[0];
        problem.constraint_tol = scalars[1];
        problem.lower = scalars[2];
        problem.upper = scalars[3];
        problem.A = read_vector<double>(
            file, static_cast<std::size_t>(problem.n) * problem.n, "A");
        problem.b = read_vector<double>(file, problem.n, "b");
        problem.C = read_vector<double>(
            file, static_cast<std::size_t>(problem.m) * problem.n, "C");
        problem.d = read_vector<double>(file, problem.m, "d");
        problem.c_norms_sq =
            read_vector<double>(file, problem.m, "c_norms_sq");
        problem.row_scale =
            read_vector<double>(file, problem.m, "row_scale");
        problem.G = read_vector<double>(
            file, static_cast<std::size_t>(problem.m) * problem.m, "G");
        problem.x0 = read_vector<double>(file, problem.n, "x0");
    }

    const int trailing = std::fgetc(file);
    if (trailing != EOF) {
        std::fprintf(stderr, "problem bundle contains trailing bytes\n");
        std::exit(2);
    }
    std::fclose(file);
    return problem;
}

inline void write_fixed_output(
    const std::string& path, int n, const std::vector<std::int64_t>& raw,
    const std::array<std::uint64_t, TELEMETRY_WORDS>& telemetry) {
    if (static_cast<int>(raw.size()) != n) {
        std::fprintf(stderr, "fixed-output length mismatch\n");
        std::exit(2);
    }
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    write_exact(file, FIXED_OUTPUT_MAGIC.data(), 8, "fixed-output magic");
    const std::uint32_t header[2] = {BUNDLE_VERSION,
                                     static_cast<std::uint32_t>(n)};
    write_exact(file, header, 2, "fixed-output header");
    write_exact(file, raw.data(), raw.size(), "fixed-output state");
    write_exact(file, telemetry.data(), telemetry.size(),
                "fixed-output telemetry");
    std::fclose(file);
}

inline void write_double_output(
    const std::string& path, int n, const std::vector<double>& x,
    const std::array<std::uint64_t, TELEMETRY_WORDS>& telemetry) {
    if (static_cast<int>(x.size()) != n) {
        std::fprintf(stderr, "double-output length mismatch\n");
        std::exit(2);
    }
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    write_exact(file, DOUBLE_OUTPUT_MAGIC.data(), 8, "double-output magic");
    const std::uint32_t header[2] = {BUNDLE_VERSION,
                                     static_cast<std::uint32_t>(n)};
    write_exact(file, header, 2, "double-output header");
    write_exact(file, x.data(), x.size(), "double-output state");
    write_exact(file, telemetry.data(), telemetry.size(),
                "double-output telemetry");
    std::fclose(file);
}

}  // namespace msrp_v05
