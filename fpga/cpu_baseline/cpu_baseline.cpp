// Lean double-precision CPU baseline for the SNN-QP fixed-horizon solve.
//
// One binary, two modes (--mode canonical|dykstra), same command line and
// output file format as the v0.5 baseline (fpga/kv260_v05/src/cpu_baseline_v05.cpp).
// See README.md for scope, flags and the equal-footing facts.
//
// Exactness rule: every optimisation here changes data layout or loop shape
// only. Each floating-point value is produced by the same operations in the
// same order as the generic library core (src/snn_opt/_native/snn_qp_core.hpp),
// so results are bitwise identical to a sequentially-summed build of it.

#include "../kv260_v05/src/bench_support.hpp"
#include "../kv260_v05/src/msrp_bundle.hpp"
#include "../kv260_v05/src/v05_double_core.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__aarch64__) && !defined(LEAN_PORTABLE_SCAN)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lean {

using msrp_v05::Problem;
using msrp_v05::SolveResult;

constexpr int MAX_DIM = 1024;
constexpr int ROW_GUARD = 1000;       // dykstra: row steps per round
constexpr int ROUND_CAP = 100000;     // dykstra: rounds per outer iteration
constexpr double DYKSTRA_TOL = 1e-12;
constexpr double INF = std::numeric_limits<double>::infinity();

// Numeric kernels exist in exactly one compiled copy, so the single-thread and
// team paths cannot differ in how the compiler contracts multiply-adds.
#define LEAN_NOINLINE __attribute__((noinline))

LEAN_NOINLINE static void gradient_step(double* x, const double* ax,
                                        const double* b, double k0, int n) {
    for (int i = 0; i < n; ++i) x[i] -= k0 * (ax[i] + b[i]);
}

// Optional section profile (-DLEAN_PROFILE): wall-clock seconds per section,
// printed to stderr. Timer calls add overhead of their own; use it to see the
// split, not for headline numbers.
#ifdef LEAN_PROFILE
enum { P_HESS, P_GRAD, P_INIT, P_SELECT, P_ROWEV, P_FACEV, P_DCLIP, P_DDOTS, P_DSTEPS, P_DFINAL, P_COUNT };
static double g_prof[P_COUNT];
static inline double prof_now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
#define PROF_BEGIN(v) const double v = prof_now()
#define PROF_END(v, id) (g_prof[id] += prof_now() - v)
#else
#define PROF_BEGIN(v)
#define PROF_END(v, id)
#endif

enum class Mode { Canonical, Dykstra };

struct Counters {
    std::uint64_t rounds = 0;       // dykstra rounds
    std::uint64_t row_steps = 0;    // dykstra row steps
    std::uint64_t guard_hits = 0;   // dykstra row-guard exits
    std::uint64_t round_caps = 0;   // dykstra round-cap exits
};

// Constant-across-iterations layouts derived once from the bundle.
constexpr int BLK = 16;  // Hessian output block (doubles)

struct Lean {
    const Problem* q;
    int n, m;
    int nblk;                 // ceil(n / BLK)
    std::vector<double> Ab;   // Ab[(blk*n + k)*BLK + j] = A[blk*BLK+j][k], zero padded
    std::vector<double> Ct;   // Ct[k*m+i] = C[i][k]  (column k contiguous)
    double lo, hi;            // +-inf when the bound is absent
};

static Lean prepare(const Problem& q) {
    Lean s;
    s.q = &q;
    s.n = q.n;
    s.m = q.m;
    s.nblk = (q.n + BLK - 1) / BLK;
    s.Ab.assign(static_cast<std::size_t>(s.nblk) * q.n * BLK, 0.0);
    for (int i = 0; i < q.n; ++i)
        for (int k = 0; k < q.n; ++k)
            s.Ab[(static_cast<std::size_t>(i / BLK) * q.n + k) * BLK + i % BLK] =
                q.A[static_cast<std::size_t>(i) * q.n + k];
    s.Ct.resize(static_cast<std::size_t>(q.n) * q.m);
    for (int i = 0; i < q.m; ++i)
        for (int k = 0; k < q.n; ++k)
            s.Ct[static_cast<std::size_t>(k) * q.m + i] =
                q.C[static_cast<std::size_t>(i) * q.n + k];
    s.lo = q.has_lower ? q.lower : -INF;
    s.hi = q.has_upper ? q.upper : INF;
    return s;
}

// ax[b*BLK + j] for blocks b0..b1-1: each output is summed over k = 0..n-1 in
// order (the order of the row dot product), 16 outputs at a time in registers,
// reading one contiguous stream.
LEAN_NOINLINE static void hessian_blocks(const double* __restrict Ab,
                           const double* __restrict x, double* __restrict ax,
                           int n, int b0, int b1) {
    for (int b = b0; b < b1; ++b) {
        double acc[BLK] = {};
        const double* a = Ab + static_cast<std::size_t>(b) * n * BLK;
        for (int k = 0; k < n; ++k) {
            const double xk = x[k];
            __builtin_prefetch(a + 8 * BLK);
            __builtin_prefetch(a + 8 * BLK + 8);
            for (int j = 0; j < BLK; ++j) acc[j] += a[j] * xk;
            a += BLK;
        }
        for (int j = 0; j < BLK; ++j) ax[b * BLK + j] = acc[j];
    }
}

// out[i] = init[i] + sum_k C[i][k] z[k] (init may be null for 0), each row
// summed in k order; rows are interleaved in groups so the dependent adds of
// different rows overlap.
static void row_dots(const double* __restrict C, const double* __restrict z,
                     const double* init, double* __restrict out, int n, int m) {
    int i = 0;
    for (; i + 4 <= m; i += 4) {
        const double* c0 = C + static_cast<std::size_t>(i) * n;
        const double* c1 = c0 + n;
        const double* c2 = c1 + n;
        const double* c3 = c2 + n;
        double s0 = init ? init[i] : 0.0, s1 = init ? init[i + 1] : 0.0;
        double s2 = init ? init[i + 2] : 0.0, s3 = init ? init[i + 3] : 0.0;
        for (int k = 0; k < n; ++k) {
            const double zk = z[k];
            s0 += c0[k] * zk; s1 += c1[k] * zk;
            s2 += c2[k] * zk; s3 += c3[k] * zk;
        }
        out[i] = s0; out[i + 1] = s1; out[i + 2] = s2; out[i + 3] = s3;
    }
    for (; i + 2 <= m; i += 2) {
        const double* c0 = C + static_cast<std::size_t>(i) * n;
        const double* c1 = c0 + n;
        double s0 = init ? init[i] : 0.0, s1 = init ? init[i + 1] : 0.0;
        for (int k = 0; k < n; ++k) {
            const double zk = z[k];
            s0 += c0[k] * zk; s1 += c1[k] * zk;
        }
        out[i] = s0; out[i + 1] = s1;
    }
    if (i < m) {
        const double* c0 = C + static_cast<std::size_t>(i) * n;
        double s0 = init ? init[i] : 0.0;
        for (int k = 0; k < n; ++k) s0 += c0[k] * z[k];
        out[i] = s0;
    }
}

// ---- linear winner-take-all scan ----------------------------------------
// Every event evaluates all m row scores and all 2n facet scores and takes the
// strict first maximum in the order rows, lower facets, upper facets (a tie
// goes to the earlier family, and within a family to the lower index).
//
// The A53 is in-order with long dependent-compare latency, so a sequential
// "if (v > best)" scan costs about 9-11 cycles per candidate. Instead:
//   pass 1: a value-only maximum with independent accumulators (max is exact
//           and order independent), emitting one partial maximum per chunk
//           of 8 elements for each facet family;
//   pass 2: for the winning family only, find the first chunk whose partial
//           equals the maximum, then the first element inside it.
// Equality (not a reduction order) selects the index, so the winner is the
// strict first maximum, and +0/-0 compare equal exactly as in the sequential
// scan. The reported best value is always recomputed at the chosen index.
constexpr int CHUNK = 8;

// Scores of both facet families for x[0..n), partial maxima per chunk in
// pl/pu (two lanes per chunk), overall maxima in ml/mu.
static inline void facet_pass(const double* __restrict x, int n, double lo,
                              double hi, double* __restrict pl,
                              double* __restrict pu, double& ml, double& mu) {
    int i = 0, c = 0;
#if defined(__aarch64__) && !defined(LEAN_PORTABLE_SCAN)
    const float64x2_t vlo = vdupq_n_f64(lo), vhi = vdupq_n_f64(hi);
    float64x2_t aL = vdupq_n_f64(-INF), aU = aL;
    for (; i + CHUNK <= n; i += CHUNK, ++c) {
        const float64x2_t v0 = vld1q_f64(x + i), v1 = vld1q_f64(x + i + 2);
        const float64x2_t v2 = vld1q_f64(x + i + 4), v3 = vld1q_f64(x + i + 6);
        const float64x2_t pL = vmaxnmq_f64(
            vmaxnmq_f64(vsubq_f64(vlo, v0), vsubq_f64(vlo, v1)),
            vmaxnmq_f64(vsubq_f64(vlo, v2), vsubq_f64(vlo, v3)));
        const float64x2_t pU = vmaxnmq_f64(
            vmaxnmq_f64(vsubq_f64(v0, vhi), vsubq_f64(v1, vhi)),
            vmaxnmq_f64(vsubq_f64(v2, vhi), vsubq_f64(v3, vhi)));
        vst1q_f64(pl + 2 * c, pL);
        vst1q_f64(pu + 2 * c, pU);
        aL = vmaxnmq_f64(aL, pL);
        aU = vmaxnmq_f64(aU, pU);
    }
    ml = vmaxnmvq_f64(aL);
    mu = vmaxnmvq_f64(aU);
#else
    ml = mu = -INF;
    for (; i + CHUNK <= n; i += CHUNK, ++c) {
        double l[CHUNK], u[CHUNK];
        for (int j = 0; j < CHUNK; ++j) { l[j] = lo - x[i + j]; u[j] = x[i + j] - hi; }
        const double l0 = __builtin_fmax(__builtin_fmax(l[0], l[1]), __builtin_fmax(l[2], l[3]));
        const double l1 = __builtin_fmax(__builtin_fmax(l[4], l[5]), __builtin_fmax(l[6], l[7]));
        const double u0 = __builtin_fmax(__builtin_fmax(u[0], u[1]), __builtin_fmax(u[2], u[3]));
        const double u1 = __builtin_fmax(__builtin_fmax(u[4], u[5]), __builtin_fmax(u[6], u[7]));
        pl[2 * c] = l0; pl[2 * c + 1] = l1;
        pu[2 * c] = u0; pu[2 * c + 1] = u1;
        ml = __builtin_fmax(ml, __builtin_fmax(l0, l1));
        mu = __builtin_fmax(mu, __builtin_fmax(u0, u1));
    }
#endif
    if (i < n) {  // tail chunk: one real lane, one -inf lane
        double tl = -INF, tu = -INF;
        for (; i < n; ++i) {
            tl = __builtin_fmax(tl, lo - x[i]);
            tu = __builtin_fmax(tu, x[i] - hi);
        }
        pl[2 * c] = tl; pl[2 * c + 1] = -INF;
        pu[2 * c] = tu; pu[2 * c + 1] = -INF;
        ml = __builtin_fmax(ml, tl);
        mu = __builtin_fmax(mu, tu);
    }
}

// First chunk whose partial equals the family maximum.
static inline int first_chunk(const double* p, double top) {
    int c = 0;
    while (p[2 * c] != top && p[2 * c + 1] != top) ++c;
    return c;
}

// Row family: maximum with four accumulators (rows are few, so one pass).
static inline double row_max(const double* r, const double* rs, int m) {
    double a0 = -INF, a1 = -INF, a2 = -INF, a3 = -INF;
    int i = 0;
    for (; i + 4 <= m; i += 4) {
        a0 = __builtin_fmax(a0, r[i] * rs[i]);
        a1 = __builtin_fmax(a1, r[i + 1] * rs[i + 1]);
        a2 = __builtin_fmax(a2, r[i + 2] * rs[i + 2]);
        a3 = __builtin_fmax(a3, r[i + 3] * rs[i + 3]);
    }
    for (; i < m; ++i) a0 = __builtin_fmax(a0, r[i] * rs[i]);
    return __builtin_fmax(__builtin_fmax(a0, a1), __builtin_fmax(a2, a3));
}

struct ScanScratch {
    std::vector<double> pl, pu;
    explicit ScanScratch(int n)
        : pl(2 * static_cast<std::size_t>((n + CHUNK - 1) / CHUNK)),
          pu(pl.size()) {}
};

// ---- canonical projection sweep -----------------------------------------
// Mirrors snn_qp::project_unified for m >= 1 (the bundle contract). Returns
// true when the projection cap is hit with the joint violation still above
// tolerance (the solve then aborts, as in the library).
LEAN_NOINLINE static bool projection_sweep(const Lean& s, double* x, double* r, ScanScratch& f,
                             int outer, msrp_v05::Observer& obs) {
    const Problem& q = *s.q;
    const int n = s.n, m = s.m;
    PROF_BEGIN(t_init);
    row_dots(q.C.data(), x, nullptr, r, n, m);
    for (int i = 0; i < m; ++i) r[i] += q.d[i];
    PROF_END(t_init, P_INIT);

    for (int ordinal = 0; ordinal < q.projection_cap; ++ordinal) {
        PROF_BEGIN(t_sel);
        const double* rs = q.row_scale.data();
        double ml, mu;
        facet_pass(x, n, s.lo, s.hi, f.pl.data(), f.pu.data(), ml, mu);
        double best = row_max(r, rs, m);
        int kind = 0;
        if (ml > best) { best = ml; kind = 1; }
        if (mu > best) { best = mu; kind = 2; }
        if (best <= q.constraint_tol) {
            PROF_END(t_sel, P_SELECT);
            return false;
        }
        int j = 0;
        if (kind == 0) {
            while (r[j] * rs[j] != best) ++j;
            best = r[j] * rs[j];
        } else if (kind == 1) {
            j = first_chunk(f.pl.data(), best) * CHUNK;
            while (s.lo - x[j] != best) ++j;
            best = s.lo - x[j];
        } else {
            j = first_chunk(f.pu.data(), best) * CHUNK;
            while (x[j] - s.hi != best) ++j;
            best = x[j] - s.hi;
        }
        PROF_END(t_sel, P_SELECT);

        PROF_BEGIN(t_ev);
        if (kind == 0) {
            const double k1 = r[j] / q.c_norms_sq[j];
            const double* cj = q.C.data() + static_cast<std::size_t>(j) * n;
            for (int k = 0; k < n; ++k) x[k] -= k1 * cj[k];
            const double* gj = q.G.data() + static_cast<std::size_t>(j) * m;
            for (int i = 0; i < m; ++i) r[i] -= k1 * gj[i];
            PROF_END(t_ev, P_ROWEV);
        } else {
            const double delta = kind == 1 ? best : -best;
            x[j] += delta;
            const double* cc = s.Ct.data() + static_cast<std::size_t>(j) * m;
            for (int i = 0; i < m; ++i) r[i] += delta * cc[i];
            PROF_END(t_ev, P_FACEV);
        }
        obs.record(kind, j, m, n, outer, ordinal);
    }

    // Cap reached: fresh joint violation, exactly joint_max_violation.
    ++obs.cap_rechecks;
    row_dots(q.C.data(), x, nullptr, r, n, m);
    double mv = 0.0;
    for (int i = 0; i < m; ++i) {
        const double g = (r[i] + q.d[i]) * q.row_scale[i];
        if (g > mv) mv = g;
    }
    if (q.has_lower)
        for (int i = 0; i < n; ++i)
            if (q.lower - x[i] > mv) mv = q.lower - x[i];
    if (q.has_upper)
        for (int i = 0; i < n; ++i)
            if (x[i] - q.upper > mv) mv = x[i] - q.upper;
    return mv > q.constraint_tol;
}

// ---- dykstra projection (box member, then exact greedy row sweep) --------
struct DykstraScratch {
    std::vector<double> p, qv, y, z, g;
    explicit DykstraScratch(int n, int m) : p(n), qv(n), y(n), z(n), g(m) {}
};

LEAN_NOINLINE static void dykstra_project(const Lean& s, double* x, DykstraScratch& w,
                            Counters& c) {
    const Problem& q = *s.q;
    const int n = s.n, m = s.m;
    double* __restrict p = w.p.data();
    double* __restrict qv = w.qv.data();
    double* __restrict y = w.y.data();
    double* __restrict z = w.z.data();
    double* g = w.g.data();
    std::memset(p, 0, sizeof(double) * n);
    std::memset(qv, 0, sizeof(double) * n);
    const double lo = s.lo, hi = s.hi;

    for (int round = 0; round < ROUND_CAP; ++round) {
        PROF_BEGIN(t_c);
        for (int k = 0; k < n; ++k) {
            const double sum = x[k] + p[k];
            const double v = sum < lo ? lo : (sum > hi ? hi : sum);
            y[k] = v;
            p[k] = sum - v;
            z[k] = v + qv[k];
        }
        PROF_END(t_c, P_DCLIP);
        PROF_BEGIN(t_d);
        row_dots(q.C.data(), z, q.d.data(), g, n, m);
        PROF_END(t_d, P_DDOTS);
        ++c.rounds;
        PROF_BEGIN(t_s);
        for (int step = 0;; ++step) {
            int jb = 0;
            double gb = -INF;
            for (int j = 0; j < m; ++j)
                if (g[j] > gb) { gb = g[j]; jb = j; }
            if (gb <= 0) break;
            if (step >= ROW_GUARD) { ++c.guard_hits; break; }
            const double st = gb / q.c_norms_sq[jb];
            const double* cj = q.C.data() + static_cast<std::size_t>(jb) * n;
            for (int i = 0; i < n; ++i) z[i] -= st * cj[i];
            const double* gj = q.G.data() + static_cast<std::size_t>(jb) * m;
            for (int j = 0; j < m; ++j) g[j] -= st * gj[j];
            g[jb] = 0.0;
            ++c.row_steps;
        }
        PROF_END(t_s, P_DSTEPS);
        PROF_BEGIN(t_f);
        // max is exact, so four independent maxima are legitimate.
        double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            const double a0 = std::fabs(z[i] - x[i]);
            const double a1 = std::fabs(z[i + 1] - x[i + 1]);
            const double a2 = std::fabs(z[i + 2] - x[i + 2]);
            const double a3 = std::fabs(z[i + 3] - x[i + 3]);
            d0 = a0 > d0 ? a0 : d0; d1 = a1 > d1 ? a1 : d1;
            d2 = a2 > d2 ? a2 : d2; d3 = a3 > d3 ? a3 : d3;
        }
        for (; i < n; ++i) {
            const double a = std::fabs(z[i] - x[i]);
            d0 = a > d0 ? a : d0;
        }
        for (int k = 0; k < n; ++k) {
            qv[k] = y[k] + qv[k] - z[k];
            x[k] = z[k];
        }
        const double d01 = d0 > d1 ? d0 : d1, d23 = d2 > d3 ? d2 : d3;
        PROF_END(t_f, P_DFINAL);
        if ((d01 > d23 ? d01 : d23) < DYKSTRA_TOL) return;
    }
    ++c.round_caps;
}

// ---- solve: one persistent team for the whole horizon --------------------
static SolveResult solve(const Lean& s, Mode mode, int threads, Counters& c) {
    const Problem& q = *s.q;
    const int n = s.n, m = s.m;
    SolveResult out;
    out.x = q.x0;
    out.threads = 1;
    c = Counters{};
    std::vector<double> ax(static_cast<std::size_t>(s.nblk) * BLK), r(m);
    DykstraScratch dyk(n, m);
    ScanScratch facets(n);
    bool stop = false;
    int observed = 1;
    // Hessian output blocks per thread.
    const int chunk = (s.nblk + threads - 1) / threads;

    auto body = [&](int tid) {
        const int b0 = std::min(s.nblk, tid * chunk);
        const int b1 = std::min(s.nblk, b0 + chunk);
        for (int outer = 0; outer < q.iterations; ++outer) {
            PROF_BEGIN(t_h);
            hessian_blocks(s.Ab.data(), out.x.data(), ax.data(), n, b0, b1);
            if (tid == 0) PROF_END(t_h, P_HESS);
#ifdef _OPENMP
#pragma omp barrier
#endif
            if (tid == 0) {
                PROF_BEGIN(t_g);
                gradient_step(out.x.data(), ax.data(), q.b.data(), q.k0, n);
                PROF_END(t_g, P_GRAD);
                if (mode == Mode::Canonical)
                    stop = projection_sweep(s, out.x.data(), r.data(), facets,
                                            outer, out.observer);
                else
                    dykstra_project(s, out.x.data(), dyk, c);
                out.iterations_executed = outer + 1;
                if (stop) out.status = 2;
            }
#ifdef _OPENMP
#pragma omp barrier
#endif
            if (stop) break;
        }
    };

#ifdef _OPENMP
    if (threads > 1) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#pragma omp parallel
        {
#pragma omp single
            observed = omp_get_num_threads();
            body(omp_get_thread_num());
        }
        // A team smaller than requested still covers the columns: body used
        // the requested chunking, so refuse to report a short team as valid.
        if (observed != threads) out.status = 4;
    } else {
        body(0);
    }
#else
    (void)threads;
    body(0);
#endif
    out.threads = observed;
    if (mode == Mode::Dykstra) {
        out.observer.rows = c.row_steps;
        out.observer.total = c.row_steps;
    }
    return out;
}

static int requested_threads() {
    const char* text = std::getenv("OMP_NUM_THREADS");
    return text ? msrp_v05::positive_integer(text, "OMP_NUM_THREADS") : 1;
}

static void counters_json(Mode mode, const SolveResult& r, const Counters& c,
                          char* buf, std::size_t cap) {
    if (mode == Mode::Canonical)
        std::snprintf(
            buf, cap,
            "\"mode\":\"canonical\",\"rows\":%llu,\"lower\":%llu,"
            "\"upper\":%llu,\"total\":%llu,\"cap_rechecks\":%llu",
            (unsigned long long)r.observer.rows,
            (unsigned long long)r.observer.lower,
            (unsigned long long)r.observer.upper,
            (unsigned long long)r.observer.total,
            (unsigned long long)r.observer.cap_rechecks);
    else
        std::snprintf(
            buf, cap,
            "\"mode\":\"dykstra\",\"rounds\":%llu,\"row_steps\":%llu,"
            "\"guard_hits\":%llu,\"round_caps\":%llu",
            (unsigned long long)c.rounds, (unsigned long long)c.row_steps,
            (unsigned long long)c.guard_hits,
            (unsigned long long)c.round_caps);
}

// Same JSON object and schema as msrp_v05::print_latency_json, with the work
// counters appended as extra keys (existing readers ignore unknown keys).
static void print_latency(const Problem& q, int threads, int warmups,
                          const std::vector<double>& secs,
                          const std::array<std::uint64_t, 16>& t, Mode mode,
                          const SolveResult& r, const Counters& c) {
    char extra[512];
    counters_json(mode, r, c, extra, sizeof extra);
    std::printf(
        "{\"schema\":\"msrp-v05-a53-latency-v1\",\"n\":%d,\"m\":%d,"
        "\"threads\":%d,\"warmups\":%d,\"clock\":\"CLOCK_MONOTONIC_RAW\","
        "\"clock_resolution_ns\":%llu,\"timer_probe_count\":1000000,"
        "\"timer_overhead_median_ns\":%llu,\"status\":%llu,"
        "\"iterations\":%llu,\"events\":%llu,\"digest\":\"%016llx\","
        "%s,\"complete_seconds\":[",
        q.n, q.m, threads, warmups,
        (unsigned long long)msrp_v05::monotonic_raw_resolution_ns(),
        (unsigned long long)msrp_v05::timer_overhead_median_ns(),
        (unsigned long long)t[1], (unsigned long long)t[3],
        (unsigned long long)t[5], (unsigned long long)t[10], extra);
    for (std::size_t i = 0; i < secs.size(); ++i)
        std::printf("%s%.17g", i ? "," : "", secs[i]);
    std::printf("],\"kernel_seconds\":[");
    for (std::size_t i = 0; i < secs.size(); ++i)
        std::printf("%s%.17g", i ? "," : "", secs[i]);
    std::printf("]}\n");
}

}  // namespace lean

int main(int argc, char** argv) {
    using namespace lean;
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <binary64-problem.bin> <output.bin> "
                     "[--mode canonical|dykstra] [--warmups N] [--reps N] "
                     "[--energy-reps N]\n",
                     argv[0]);
        return 2;
    }
    Mode mode = Mode::Canonical;
    std::vector<char*> rest(argv, argv + 3);
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "canonical") mode = Mode::Canonical;
            else if (v == "dykstra") mode = Mode::Dykstra;
            else { std::fprintf(stderr, "unknown mode: %s\n", v.c_str()); return 2; }
        } else {
            rest.push_back(argv[i]);
        }
    }
    const msrp_v05::BenchOptions options = msrp_v05::parse_bench_options(
        static_cast<int>(rest.size()), rest.data(), 3);
    const Problem q = msrp_v05::load_problem(argv[1], false, MAX_DIM, MAX_DIM);
    if (q.source_is_float32) {
        std::fprintf(stderr, "CPU baseline requires binary64 input\n");
        return 2;
    }
    const Lean s = prepare(q);
    const int threads = requested_threads();

    SolveResult last;
    Counters c;
    for (int i = 0; i < options.warmups; ++i) last = solve(s, mode, threads, c);

    if (options.energy_repetitions > 0) {
        msrp_v05::wait_for_go();
        const std::uint64_t started = msrp_v05::monotonic_raw_ns();
        for (int i = 0; i < options.energy_repetitions; ++i)
            last = solve(s, mode, threads, c);
        const std::uint64_t finished = msrp_v05::monotonic_raw_ns();
        const auto meta = msrp_v05::telemetry(q, last);
        msrp_v05::write_double_output(argv[2], q.n, last.x, meta);
        std::printf(
            "DONE REPS %d DURATION_SECONDS %.17g STATUS %d EVENTS %llu "
            "DIGEST %016llx",
            options.energy_repetitions,
            static_cast<double>(finished - started) * 1e-9, last.status,
            (unsigned long long)last.observer.total,
            (unsigned long long)last.observer.digest);
        if (mode == Mode::Canonical)
            std::printf(" ROWS %llu LOWER %llu UPPER %llu\n",
                        (unsigned long long)last.observer.rows,
                        (unsigned long long)last.observer.lower,
                        (unsigned long long)last.observer.upper);
        else
            std::printf(" ROUNDS %llu ROW_STEPS %llu\n",
                        (unsigned long long)c.rounds,
                        (unsigned long long)c.row_steps);
        std::fflush(stdout);
        return last.status == 0 ? 0 : 3;
    }

    std::vector<double> latency;
    latency.reserve(options.repetitions);
    for (int i = 0; i < options.repetitions; ++i) {
        const std::uint64_t started = msrp_v05::monotonic_raw_ns();
        last = solve(s, mode, threads, c);
        const std::uint64_t finished = msrp_v05::monotonic_raw_ns();
        latency.push_back(static_cast<double>(finished - started) * 1e-9);
    }
    const auto meta = msrp_v05::telemetry(q, last);
    msrp_v05::write_double_output(argv[2], q.n, last.x, meta);
#ifdef LEAN_PROFILE
    {
        const double reps = options.warmups + options.repetitions;
        const char* names[P_COUNT] = {"hessian", "gradient", "sweep_init",
                                      "select", "row_event", "facet_event",
                                      "dyk_clip", "dyk_dots", "dyk_steps", "dyk_final"};
        for (int i = 0; i < P_COUNT; ++i)
            std::fprintf(stderr, "PROFILE %s_us_per_iter %.2f\n", names[i],
                         g_prof[i] / (reps * q.iterations) * 1e6);
    }
#endif
    print_latency(q, last.threads, options.warmups, latency, meta, mode, last, c);
    return last.status == 0 ? 0 : 3;
}
