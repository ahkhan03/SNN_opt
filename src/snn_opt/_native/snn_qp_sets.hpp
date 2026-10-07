// Bounded host-side numerical helpers for the extended native set descriptors.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace snn_qp {
namespace sets {

constexpr int MAX_DIM = 8;
constexpr int DEFAULT_SWEEP_CAP = 32;
constexpr double DEFAULT_TOL = 8.0 * std::numeric_limits<double>::epsilon();

inline void require_finite(const double* a, int n, const char* what) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]))
            throw std::invalid_argument(std::string(what) + " received a non-finite block");
    }
}

struct SVDResult {
    int rows = 0;
    int cols = 0;
    int k = 0;
    int sweeps = 0;
    double sigma[MAX_DIM]{};
    double u[MAX_DIM][MAX_DIM]{}; // rows x k
    double v[MAX_DIM][MAX_DIM]{}; // cols x k
};

inline double offdiag_columns(const double w[MAX_DIM][MAX_DIM], int rows, int cols) {
    double worst = 0.0;
    for (int i = 0; i < cols; ++i) {
        double a = 0.0;
        for (int r = 0; r < rows; ++r) a += w[r][i] * w[r][i];
        for (int j = i + 1; j < cols; ++j) {
            double b = 0.0, c = 0.0;
            for (int r = 0; r < rows; ++r) {
                b += w[r][j] * w[r][j];
                c += w[r][i] * w[r][j];
            }
            if (a > 0.0 && b > 0.0)
                worst = std::max(worst, std::fabs(c) / (std::sqrt(a) * std::sqrt(b)));
            // A numerically zero column contributes no Gram off-diagonal.
            // Avoid a product of squared norms here: it can underflow long
            // before either column has vanished on rank-deficient blocks.
            else if (c != 0.0)
                worst = std::max(worst, std::fabs(c));
        }
    }
    return worst;
}

inline void jacobi_svd(const double* input, int rows, int cols,
                       int cap, double tol, SVDResult& out) {
    if (rows <= 0 || rows > MAX_DIM || cols <= 0 || cols > MAX_DIM)
        throw std::invalid_argument("spectral ball Jacobi SVD dimensions exceed the native 8x8 cap");
    if (cap < 0 || !std::isfinite(tol) || tol < 0.0)
        throw std::invalid_argument("spectral ball Jacobi SVD received invalid cap or tolerance");
    require_finite(input, rows * cols, "spectral ball Jacobi SVD");

    // Normalize once to keep Gram products and rotations finite for both
    // very large and very small inputs.  Rescale only the singular values.
    double scale = 0.0;
    for (int k = 0; k < rows * cols; ++k)
        scale = std::max(scale, std::fabs(input[k]));
    if (scale == 0.0) scale = 1.0;
    const bool transposed = rows < cols;
    const int wr = transposed ? cols : rows;
    const int wc = transposed ? rows : cols;
    double w[MAX_DIM][MAX_DIM]{};
    double vwork[MAX_DIM][MAX_DIM]{};
    for (int r = 0; r < wr; ++r)
        for (int c = 0; c < wc; ++c)
            w[r][c] = (transposed ? input[c * cols + r] : input[r * cols + c]) / scale;
    for (int i = 0; i < wc; ++i) vwork[i][i] = 1.0;

    int sweeps = 0;
    bool converged = (offdiag_columns(w, wr, wc) <= tol);
    for (; !converged && sweeps < cap; ++sweeps) {
        for (int i = 0; i < wc; ++i) {
            for (int j = i + 1; j < wc; ++j) {
                double a = 0.0, b = 0.0, c = 0.0;
                for (int r = 0; r < wr; ++r) {
                    a += w[r][i] * w[r][i];
                    b += w[r][j] * w[r][j];
                    c += w[r][i] * w[r][j];
                }
                if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
                    throw std::invalid_argument("spectral ball Jacobi SVD produced a non-finite rotation");
                if (c == 0.0) continue;
                const double d = b - a;
                const double r = std::hypot(d, 2.0 * c);
                const double denom = std::fabs(d) + r;
                if (!(denom > 0.0) || !std::isfinite(denom))
                    throw std::invalid_argument("spectral ball Jacobi SVD produced a non-finite rotation");
                const double sign = d < 0.0 ? -1.0 : 1.0;
                const double t = (2.0 * c * sign) / denom;
                const double cs = 1.0 / std::sqrt(1.0 + t * t);
                const double sn = cs * t;
                for (int rr = 0; rr < wr; ++rr) {
                    const double wi = w[rr][i], wj = w[rr][j];
                    w[rr][i] = cs * wi - sn * wj;
                    w[rr][j] = sn * wi + cs * wj;
                }
                for (int rr = 0; rr < wc; ++rr) {
                    const double vi = vwork[rr][i], vj = vwork[rr][j];
                    vwork[rr][i] = cs * vi - sn * vj;
                    vwork[rr][j] = sn * vi + cs * vj;
                }
            }
        }
        converged = (offdiag_columns(w, wr, wc) <= tol);
    }
    if (!converged)
        throw std::invalid_argument("spectral ball Jacobi SVD did not converge within the sweep cap");
    out = SVDResult{};
    out.rows = rows; out.cols = cols; out.k = std::min(rows, cols); out.sweeps = sweeps;
    double norms[MAX_DIM]{};
    int order[MAX_DIM]{};
    for (int q = 0; q < wc; ++q) {
        for (int r = 0; r < wr; ++r) norms[q] = std::hypot(norms[q], w[r][q]);
        order[q] = q;
    }
    for (int i = 0; i < wc; ++i)
        for (int j = i + 1; j < wc; ++j)
            if (norms[order[j]] > norms[order[i]]) std::swap(order[i], order[j]);
    // Reconstruction is basis invariant; stable descending singular values
    // make the test hook and first-argmax top pair deterministic.
    for (int q = 0; q < out.k; ++q) {
        const int src = order[q];
        const double norm = norms[src];
        out.sigma[q] = norm * scale;
        if (!std::isfinite(out.sigma[q]))
            throw std::invalid_argument("spectral ball Jacobi SVD produced a non-finite singular value");
        for (int r = 0; r < wr; ++r) {
            const double unit = norm > 0.0 ? w[r][src] / norm : (r == q ? 1.0 : 0.0);
            if (transposed) out.v[r][q] = unit;
            else out.u[r][q] = unit;
        }
        for (int r = 0; r < wc; ++r) {
            if (transposed) out.u[r][q] = vwork[r][src];
            else out.v[r][q] = vwork[r][src];
        }
    }

}

inline void spectral_top_pair(const double* input, int rows, int cols,
                              double& sigma, double* u, double* v,
                              int cap = DEFAULT_SWEEP_CAP,
                              double tol = DEFAULT_TOL, int* sweeps = nullptr) {
    SVDResult f;
    jacobi_svd(input, rows, cols, cap, tol, f);
    int best = 0;
    for (int q = 1; q < f.k; ++q)
        if (f.sigma[q] > f.sigma[best]) best = q;
    sigma = f.sigma[best];
    for (int r = 0; r < rows; ++r) u[r] = f.u[r][best];
    for (int c = 0; c < cols; ++c) v[c] = f.v[c][best];
    if (sweeps) *sweeps = f.sweeps;
}

inline void spectral_project(const double* input, int rows, int cols,
                            double radius, double* output,
                            int cap = DEFAULT_SWEEP_CAP,
                            double tol = DEFAULT_TOL) {
    SVDResult f;
    jacobi_svd(input, rows, cols, cap, tol, f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            double value = 0.0;
            for (int q = 0; q < f.k; ++q) {
                const double clipped = std::min(f.sigma[q], radius);
                value += f.u[r][q] * clipped * f.v[c][q];
            }
            output[r * cols + c] = value;
        }
    require_finite(output, rows * cols, "spectral ball Jacobi projection");
}

struct EighResult {
    int n = 0;
    int sweeps = 0;
    double values[MAX_DIM]{};
    double vectors[MAX_DIM][MAX_DIM]{};
};

inline double offdiag_symmetric(const double a[MAX_DIM][MAX_DIM], int n,
                                double scale) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) sum += 2.0 * a[i][j] * a[i][j];
    return scale == 0.0 ? 0.0 : std::sqrt(sum) / scale;
}

inline void jacobi_eigh(const double* input, int n, int cap, double tol,
                        EighResult& out) {
    if (n <= 0 || n > MAX_DIM)
        throw std::invalid_argument("PSD Jacobi eigensolver dimension exceeds the native 8x8 cap");
    if (cap < 0 || !std::isfinite(tol) || tol < 0.0)
        throw std::invalid_argument("PSD Jacobi eigensolver received invalid cap or tolerance");
    require_finite(input, n * n, "PSD Jacobi eigensolver");
    double scale = 0.0;
    for (int k = 0; k < n * n; ++k)
        scale = std::max(scale, std::fabs(input[k]));
    if (scale == 0.0) scale = 1.0;
    double a[MAX_DIM][MAX_DIM]{};
    double v[MAX_DIM][MAX_DIM]{};
    double fro = 0.0;
    for (int i = 0; i < n; ++i) {
        v[i][i] = 1.0;
        for (int j = 0; j < n; ++j) {
            if (input[i * n + j] != input[j * n + i])
                throw std::invalid_argument("PSD Jacobi eigensolver requires a symmetric matrix");
            a[i][j] = input[i * n + j] / scale;
            fro += a[i][j] * a[i][j];
        }
    }
    fro = std::sqrt(fro);
    bool converged = offdiag_symmetric(a, n, fro) <= tol;
    int sweeps = 0;
    for (; !converged && sweeps < cap; ++sweeps) {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double c = a[i][j];
                if (c == 0.0) continue;
                const double d = a[j][j] - a[i][i];
                const double rr = std::hypot(d, 2.0 * c);
                const double denom = std::fabs(d) + rr;
                if (!(denom > 0.0) || !std::isfinite(denom))
                    throw std::invalid_argument("PSD Jacobi eigensolver produced a non-finite rotation");
                const double sign = d < 0.0 ? -1.0 : 1.0;
                const double t = 2.0 * c * sign / denom;
                const double cs = 1.0 / std::sqrt(1.0 + t * t);
                const double sn = cs * t;
                const double ai = a[i][i], aj = a[j][j];
                for (int k = 0; k < n; ++k) {
                    if (k == i || k == j) continue;
                    const double ki = a[k][i], kj = a[k][j];
                    a[k][i] = a[i][k] = cs * ki - sn * kj;
                    a[k][j] = a[j][k] = sn * ki + cs * kj;
                }
                a[i][i] = cs * cs * ai - 2.0 * cs * sn * c + sn * sn * aj;
                a[j][j] = sn * sn * ai + 2.0 * cs * sn * c + cs * cs * aj;
                a[i][j] = a[j][i] = 0.0;
                for (int k = 0; k < n; ++k) {
                    const double vi = v[k][i], vj = v[k][j];
                    v[k][i] = cs * vi - sn * vj;
                    v[k][j] = sn * vi + cs * vj;
                }
            }
        }
        converged = offdiag_symmetric(a, n, fro) <= tol;
    }
    if (!converged)
        throw std::invalid_argument("PSD Jacobi eigensolver did not converge within the sweep cap");
    out = EighResult{};
    out.n = n; out.sweeps = sweeps;
    int order[MAX_DIM]{};
    for (int i = 0; i < n; ++i) order[i] = i;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (a[order[j]][order[j]] > a[order[i]][order[i]])
                std::swap(order[i], order[j]);
    for (int i = 0; i < n; ++i) {
        const int src = order[i];
        out.values[i] = a[src][src] * scale;
        if (!std::isfinite(out.values[i]))
            throw std::invalid_argument("PSD Jacobi eigensolver produced a non-finite eigenvalue");
        for (int j = 0; j < n; ++j) out.vectors[j][i] = v[j][src];
    }

}

inline void psd_project(const double* packed, int n, double* output,
                        int cap = DEFAULT_SWEEP_CAP,
                        double tol = DEFAULT_TOL) {
    double matrix[MAX_DIM][MAX_DIM]{};
    const double root2 = std::sqrt(2.0);
    int at = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
            const double x = (i == j) ? packed[at] : packed[at] / root2;
            matrix[i][j] = matrix[j][i] = x;
            ++at;
        }
    double input[MAX_DIM * MAX_DIM]{};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) input[i * n + j] = matrix[i][j];
    EighResult e;
    jacobi_eigh(input, n, cap, tol, e);
    double rebuilt[MAX_DIM][MAX_DIM]{};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                rebuilt[i][j] += e.vectors[i][k] * std::max(0.0, e.values[k]) * e.vectors[j][k];
    at = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
            output[at++] = (i == j) ? rebuilt[i][j] : root2 * rebuilt[i][j];
        }
    const int dim = n * (n + 1) / 2;
    require_finite(output, dim, "PSD Jacobi projection");
}

} // namespace sets
} // namespace snn_qp
