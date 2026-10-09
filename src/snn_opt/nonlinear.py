"""Opt-in nonlinear and conic projection candidates.

The released :mod:`snn_opt.solver` path is deliberately polyhedral.  This
module contains the small immutable descriptors used by the optional Python
extension: differentiable-inequality cutters, exact-set projectors, and the
closed-form projectors used by the conic experiments.  The callbacks all use
the *full* solver state vector.  ``coordinates`` is an optional declaration
which lets a projector operate on a local subset while leaving every other
coordinate untouched.

The lift helpers import ``OptimizationProblem`` lazily.  Keeping that import
lazy avoids a solver <-> helper import cycle while still making the factories
available from the public package namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

# Private identity token used by the native descriptor serializer.  The token
# prevents a kkt_data string alone from silently replacing a custom oracle.
_SNN_NATIVE_SPECTRAL_TOKEN = object()
_SNN_NATIVE_PSD_TOKEN = object()


def _normalise_coordinates(value: Optional[Sequence[int]]) -> Optional[tuple[int, ...]]:
    """Validate and freeze a coordinate declaration.

    The ambient dimension is checked by ``SNNSolver`` because candidate
    objects are intentionally constructible before a problem is available.
    """
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("candidate coordinates must be a one-dimensional sequence")
    if arr.size == 0:
        raise ValueError("candidate coordinates must not be empty")
    # Reject 1.5 and NaN rather than silently truncating through astype(int).
    try:
        as_float = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate coordinates must be finite integers") from exc
    if (not np.all(np.isfinite(as_float))
            or not np.all(as_float == np.floor(as_float))):
        raise ValueError("candidate coordinates must be finite integers")
    coords = tuple(int(i) for i in as_float)
    if len(set(coords)) != len(coords):
        raise ValueError("candidate coordinates must be unique")
    if any(i < 0 for i in coords):
        raise ValueError("candidate coordinates must be non-negative")
    return coords


@dataclass(frozen=True)
class CutterCandidate:
    """A differentiable convex inequality ``value(x) <= 0``.

    ``jacobian`` returns the gradient (a local vector is accepted when
    ``coordinates`` is supplied).  ``normal`` is optional KKT metadata: it
    should return an outward unit normal, or ``None`` away from the active
    boundary.  The solver always normalises the returned value again, so a
    harmless scale error in a custom hook cannot affect the certificate.
    """

    value: Callable[[np.ndarray], float]
    jacobian: Callable[[np.ndarray], np.ndarray]
    name: str = "cutter"
    normal: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None
    coordinates: Optional[Sequence[int]] = None
    kkt_data: Any = None

    def __post_init__(self) -> None:
        if not callable(self.value):
            raise TypeError("CutterCandidate.value must be callable")
        if not callable(self.jacobian):
            raise TypeError("CutterCandidate.jacobian must be callable")
        if self.normal is not None and not callable(self.normal):
            raise TypeError("CutterCandidate.normal must be callable or None")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CutterCandidate.name must be a non-empty string")
        object.__setattr__(self, "coordinates",
                           _normalise_coordinates(self.coordinates))

    @property
    def kind(self) -> str:
        return "cutter"


@dataclass(frozen=True)
class ProjectorCandidate:
    """An exact projector onto a closed convex set.

    ``project(x)`` may return an ambient vector or, when ``coordinates`` is
    declared, only the projected local coordinates.  ``normal`` and
    ``kkt_data`` are optional certificate metadata.  A custom projector with
    no normal remains usable; its final KKT certificate is reported as
    ``not_available`` rather than guessed.
    """

    project: Callable[[np.ndarray], np.ndarray]
    name: str = "set"
    coordinates: Optional[Sequence[int]] = None
    kkt_data: Any = None
    normal: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None

    def __post_init__(self) -> None:
        if not callable(self.project):
            raise TypeError("ProjectorCandidate.project must be callable")
        if self.normal is not None and not callable(self.normal):
            raise TypeError("ProjectorCandidate.normal must be callable or None")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ProjectorCandidate.name must be a non-empty string")
        object.__setattr__(self, "coordinates",
                           _normalise_coordinates(self.coordinates))

    @property
    def kind(self) -> str:
        return "set"


class AffineSubspaceProjector(ProjectorCandidate):
    """Exact projection onto an affine equality or graph subspace.

    ``B`` and ``h`` follow the lift notation: the graph form is
    ``q = B x + h`` for a state ``(x, q)``.  For convenience a state whose
    length equals ``B.shape[1]`` is also interpreted as the ordinary equality
    ``B x = h``.  The graph branch uses the specified precomputed
    ``(I + B.T @ B)^-1`` block, while the equality branch uses the usual
    minimum-distance projection.  Equality normals are exposed with both signs
    because the solver's certificate fit is nonnegative.
    """

    def __init__(self, B: np.ndarray, h: np.ndarray, name: str = "affine_subspace",
                 coordinates: Optional[Sequence[int]] = None):
        B_arr = np.asarray(B, dtype=float)
        h_arr = np.asarray(h, dtype=float).reshape(-1)
        if B_arr.ndim == 1:
            B_arr = B_arr.reshape(1, -1)
        if B_arr.ndim != 2:
            raise ValueError("AffineSubspaceProjector.B must be a 2-D array")
        if B_arr.shape[0] != h_arr.size:
            raise ValueError("AffineSubspaceProjector.B rows must match h length")
        if not np.all(np.isfinite(B_arr)) or not np.all(np.isfinite(h_arr)):
            raise ValueError("AffineSubspaceProjector.B and h must be finite")
        if B_arr.shape[0] == 0 or B_arr.shape[1] == 0:
            raise ValueError("AffineSubspaceProjector.B must be non-empty")
        gram = B_arr @ B_arr.T
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            gram_inv = np.linalg.pinv(gram)
        correction_map = B_arr.T @ gram_inv
        graph_block = np.eye(B_arr.shape[1]) + B_arr.T @ B_arr
        try:
            graph_inv = np.linalg.inv(graph_block)
        except np.linalg.LinAlgError:
            graph_inv = np.linalg.pinv(graph_block)

        def project(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float).reshape(-1)
            n_x = B_arr.shape[1]
            p = B_arr.shape[0]
            if x_arr.size == n_x + p:
                # Graph state y=(x,q), q=B x+h.
                u = x_arr[:n_x]
                q = x_arr[n_x:]
                u_new = u + graph_inv @ B_arr.T @ (q - B_arr @ u - h_arr)
                out = x_arr.copy()
                out[:n_x] = u_new
                out[n_x:] = B_arr @ u_new + h_arr
                return out
            if x_arr.size != n_x:
                raise ValueError(
                    "AffineSubspaceProjector received a state with "
                    f"dimension {x_arr.size}, expected {n_x} (equality) or "
                    f"{n_x + p} (graph)")
            return x_arr - correction_map @ (B_arr @ x_arr - h_arr)

        def normal(x: np.ndarray) -> np.ndarray:
            # Both signs make the equality multiplier unrestricted in a
            # nonnegative fit.  Degenerate rows are omitted by the solver.
            x_size = np.asarray(x).size
            if x_size == B_arr.shape[1] + B_arr.shape[0]:
                graph_rows = np.zeros((B_arr.shape[0], x_size), dtype=float)
                graph_rows[:, :B_arr.shape[1]] = -B_arr
                graph_rows[:, B_arr.shape[1]:] = np.eye(B_arr.shape[0])
                return np.vstack((graph_rows, -graph_rows))
            return np.vstack((B_arr, -B_arr))

        def slack(x: np.ndarray) -> np.ndarray:
            return np.zeros(2 * B_arr.shape[0], dtype=float)

        super().__init__(project=project, name=name, coordinates=coordinates,
                         kkt_data={"slack": slack, "equality": True},
                         normal=normal)
        object.__setattr__(self, "B", B_arr.copy())
        object.__setattr__(self, "h", h_arr.copy())
        object.__setattr__(self, "_projection_correction_map", correction_map)
        object.__setattr__(self, "_graph_inverse", graph_inv)


def halfspace_projector(c: np.ndarray, d: float = 0.0,
                        name: str = "halfspace",
                        coordinates: Optional[Sequence[int]] = None
                        ) -> ProjectorCandidate:
    """Return the exact Euclidean projector onto ``c @ x + d <= 0``.

    This is the set-valued counterpart of :func:`affine_cutter`.  The
    callback accepts an ambient state; with ``coordinates`` it may instead
    operate on the declared local coordinates while leaving all other entries
    unchanged.  Zero-normal rows are treated as the identity when they are
    feasible and rejected at construction time when they are not.
    """
    c_arr = np.asarray(c, dtype=float).reshape(-1)
    d_val = float(np.asarray(d, dtype=float).reshape(-1)[0])
    if c_arr.size == 0 or not np.all(np.isfinite(c_arr)) or not np.isfinite(d_val):
        raise ValueError("halfspace coefficients must be finite and non-empty")
    norm_sq = float(c_arr @ c_arr)
    if norm_sq <= 1e-24 and d_val > 0.0:
        raise ValueError("an infeasible zero-normal halfspace has no projector")
    coords = _normalise_coordinates(coordinates)
    if coords is not None and len(coords) != c_arr.size:
        raise ValueError("halfspace coordinates must match coefficient length")

    def _local(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if coords is None:
            if arr.size != c_arr.size:
                raise ValueError(
                    f"halfspace expects a state of dimension {c_arr.size}")
            return arr, arr.copy()
        if max(coords, default=-1) >= arr.size:
            raise ValueError("halfspace coordinate exceeds state dimension")
        return arr[list(coords)], arr

    def project(x: np.ndarray) -> np.ndarray:
        local, ambient = _local(x)
        residual = float(c_arr @ local + d_val)
        out = ambient.copy()
        if residual > 0.0 and norm_sq > 1e-24:
            update = -(residual / norm_sq) * c_arr
            if coords is None:
                out = local + update
            else:
                out[list(coords)] = local + update
        return out

    def normal(x: np.ndarray) -> Optional[np.ndarray]:
        if norm_sq <= 1e-24:
            return None
        arr = np.asarray(x, dtype=float).reshape(-1)
        out = np.zeros(arr.size, dtype=float)
        unit = c_arr / np.sqrt(norm_sq)
        if coords is None:
            out[:] = unit
        else:
            if max(coords, default=-1) >= arr.size:
                raise ValueError("halfspace coordinate exceeds state dimension")
            out[list(coords)] = unit
        return out

    def slack(x: np.ndarray) -> float:
        local, _ = _local(x)
        return -float(c_arr @ local + d_val) / np.sqrt(norm_sq) if norm_sq > 1e-24 else np.inf

    return ProjectorCandidate(
        project=project, name=name, coordinates=coords, normal=normal,
        kkt_data={"slack": slack, "set": "halfspace", "affine": True,
                  "normal": c_arr.copy(), "offset": d_val})


def affine_cutter(c: np.ndarray, d: float = 0.0, name: str = "affine_cutter",
                  coordinates: Optional[Sequence[int]] = None) -> CutterCandidate:
    """Build a cutter whose affine inequality exactly matches a row.

    This adapter is useful for the affine-identity regression fixture.  It
    keeps the same residual and normalisation as the released row path.
    """
    c_arr = np.asarray(c, dtype=float).reshape(-1)
    d_val = float(np.asarray(d, dtype=float).reshape(-1)[0])
    if c_arr.size == 0 or not np.all(np.isfinite(c_arr)) or not np.isfinite(d_val):
        raise ValueError("affine cutter coefficients must be finite and non-empty")

    def value(x: np.ndarray) -> float:
        return float(np.asarray(c_arr @ np.asarray(x, dtype=float) + d_val).reshape(-1)[0])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return c_arr.copy()

    def normal(x: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(c_arr))
        return c_arr / norm

    return CutterCandidate(value=value, jacobian=jacobian, name=name,
                           normal=normal, coordinates=coordinates,
                           kkt_data={"affine": True, "slack":
                                     lambda x: -value(x) / np.linalg.norm(c_arr)})


def _subset_center(center: Optional[np.ndarray], n_local: int) -> np.ndarray:
    if center is None:
        return np.zeros(n_local, dtype=float)
    arr = np.asarray(center, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, n_local)
    if arr.size != n_local:
        raise ValueError("ball center must be scalar or have one entry per index")
    if not np.all(np.isfinite(arr)):
        raise ValueError("ball center must be finite")
    return arr.copy()


def ball_projector(indices: Sequence[int], radius: float,
                   center: Optional[np.ndarray] = None,
                   name: str = "ball") -> ProjectorCandidate:
    """Return a radial projector on the selected coordinates."""
    coords = _normalise_coordinates(indices)
    assert coords is not None
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("ball radius must be finite and non-negative")
    center_arr = _subset_center(center, len(coords))

    def _local(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if max(coords, default=-1) >= arr.size:
            raise ValueError("ball projector coordinate exceeds state dimension")
        return arr[list(coords)]

    def project(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1).copy()
        local = _local(arr)
        delta = local - center_arr
        norm = float(np.linalg.norm(delta))
        if norm > radius and norm > 0.0:
            arr[list(coords)] = center_arr + (radius / norm) * delta
        elif radius == 0.0:
            arr[list(coords)] = center_arr
        return arr

    def normal(x: np.ndarray) -> Optional[np.ndarray]:
        local = _local(x)
        delta = local - center_arr
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-15:
            return None
        out = np.zeros(np.asarray(x).size, dtype=float)
        out[list(coords)] = delta / norm
        return out

    def slack(x: np.ndarray) -> float:
        return radius - float(np.linalg.norm(_local(x) - center_arr))

    return ProjectorCandidate(project=project, name=name, coordinates=coords,
                              normal=normal, kkt_data={"slack": slack,
                                                        "set": "ball",
                                                        "radius": radius,
                                                        "center": center_arr.copy()})


def _box_bound(value, label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"box {label} bound must be numeric") from exc
    if arr.size == 0:
        raise ValueError(f"box {label} bound must not be empty")
    if np.any(np.isnan(arr)):
        raise ValueError(f"box {label} bound must not be NaN")
    return arr


def box_projector(lower, upper,
                  coordinates: Optional[Sequence[int]] = None,
                  name: str = "box") -> ProjectorCandidate:
    """Return the exact projector onto ``lower <= x <= upper`` (elementwise).

    ``lower`` and ``upper`` are scalars or 1-D arrays, broadcast to the
    coordinate block; ``-inf``/``+inf`` leave that side open.  Without
    ``coordinates`` the box acts on the ambient state.  The projection is a
    clip, so a box is one Dykstra member with one correction vector, not
    ``2n`` halfspaces.  The compiled backend runs it natively (descriptor
    kind 9), alone or inside :func:`joint_dykstra_projector`.
    """
    coords = _normalise_coordinates(coordinates)
    lo = _box_bound(lower, "lower")
    hi = _box_bound(upper, "upper")
    if coords is not None:
        size = len(coords)
    elif lo.size == 1 or hi.size == 1 or lo.size == hi.size:
        size = max(lo.size, hi.size)
    else:
        raise ValueError("box lower and upper bounds have different lengths")
    for arr, label in ((lo, "lower"), (hi, "upper")):
        if arr.size not in (1, size):
            raise ValueError(
                f"box {label} bound must be scalar or have one entry per coordinate")
    lo = np.broadcast_to(lo, (size,)).copy()
    hi = np.broadcast_to(hi, (size,)).copy()
    if np.any(lo == np.inf) or np.any(hi == -np.inf):
        raise ValueError("box bounds must leave a non-empty set (lower < +inf, upper > -inf)")
    if np.any(lo > hi):
        raise ValueError("box lower bound exceeds upper bound")
    # A scalar ambient box applies to any state dimension; an array ambient
    # box fixes the dimension, which is checked at call time.
    scalar_ambient = coords is None and size == 1

    idx_list = None if coords is None else list(coords)

    def _bounds(arr: np.ndarray):
        if coords is not None:
            if max(coords, default=-1) >= arr.size:
                raise ValueError("box projector coordinate exceeds state dimension")
            return lo, hi, idx_list
        if scalar_ambient:
            return lo[0], hi[0], None
        if arr.size != size:
            raise ValueError(f"box expects a state of dimension {size}")
        return lo, hi, None

    def project(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        lo_b, hi_b, idx = _bounds(arr)
        if idx is None:
            return np.clip(arr, lo_b, hi_b)
        out = arr.copy()
        out[idx] = np.clip(arr[idx], lo_b, hi_b)
        return out

    def _local_slacks(x: np.ndarray):
        arr = np.asarray(x, dtype=float).reshape(-1)
        lo_b, hi_b, idx = _bounds(arr)
        local = arr if idx is None else arr[idx]
        return local - lo_b, hi_b - local, idx

    def slack(x: np.ndarray) -> float:
        below, above, _ = _local_slacks(x)
        return float(np.min(np.minimum(below, above)))

    def normal(x: np.ndarray) -> Optional[np.ndarray]:
        below, above, idx = _local_slacks(x)
        tight = np.minimum(below, above)
        k = int(np.argmin(tight))
        if tight[k] > 0.0:
            return None
        out = np.zeros(np.asarray(x).size, dtype=float)
        j = k if idx is None else idx[k]
        out[j] = 1.0 if above[k] <= below[k] else -1.0
        return out

    return ProjectorCandidate(project=project, name=name, coordinates=coords,
                              normal=normal, kkt_data={"slack": slack,
                                                        "normal": normal,
                                                        "set": "box",
                                                        "lower": lo.copy(),
                                                        "upper": hi.copy(),
                                                        "euclidean_project": project})


def _soc_polar_projector(mu: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return the Euclidean projector onto the polar of ``||z|| <= mu*t``.

    In ``(s, w)`` coordinates the polar is
    ``||w|| <= -s / mu``.  Setting ``u = -s`` turns this into the scaled SOC
    ``||w|| <= (1/mu) u``; the sign is restored after that projection.  The
    callback works on the local ``(s, w...)`` ordering used in the certificate.
    """
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("SOC polar slope must be finite and positive")

    def project(y: np.ndarray) -> np.ndarray:
        arr = np.asarray(y, dtype=float).reshape(-1)
        if arr.size < 2:
            raise ValueError("SOC polar vectors need a scalar and a lateral part")
        u = -float(arr[0])
        z = arr[1:]
        zn = float(np.linalg.norm(z))
        slope = 1.0 / mu
        if zn <= slope * u:
            u_new, z_new = u, z.copy()
        elif u + slope * zn <= 0.0:
            u_new, z_new = 0.0, np.zeros_like(z)
        else:
            u_new = (u + slope * zn) / (1.0 + slope * slope)
            z_new = (slope * u_new / zn) * z if zn > 0.0 else np.zeros_like(z)
        return np.concatenate(([-u_new], z_new))

    return project


def soc_projector(t_index: int, z_indices: Sequence[int],
                  name: str = "soc") -> ProjectorCandidate:
    """Return the standard second-order-cone projector.

    Coordinates are ordered as ``(t, z...)`` conceptually, while the ambient
    vector retains its caller-provided indices.
    """
    coords_z = _normalise_coordinates(z_indices)
    assert coords_z is not None
    t_arr = _normalise_coordinates([t_index])
    assert t_arr is not None
    t_idx = t_arr[0]
    if t_idx in coords_z:
        raise ValueError("SOC scalar coordinate must not occur in z_indices")

    def _parts(x: np.ndarray) -> tuple[np.ndarray, float]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if max((t_idx, *coords_z), default=-1) >= arr.size:
            raise ValueError("SOC projector coordinate exceeds state dimension")
        return arr[list(coords_z)], float(arr[t_idx])

    def project(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1).copy()
        z, t = _parts(arr)
        zn = float(np.linalg.norm(z))
        if zn <= t:
            return arr
        if zn <= -t:
            arr[t_idx] = 0.0
            arr[list(coords_z)] = 0.0
            return arr
        alpha = 0.5 * (zn + t)
        arr[t_idx] = alpha
        arr[list(coords_z)] = (alpha / zn) * z if zn > 0.0 else 0.0
        return arr

    def normal(x: np.ndarray) -> Optional[np.ndarray]:
        z, t = _parts(x)
        zn = float(np.linalg.norm(z))
        if zn <= 1e-15:
            # There is no unique radial direction on the scalar axis.  The
            # certificate handles the apex with the complete polar cone.
            return None
        out = np.zeros(np.asarray(x).size, dtype=float)
        out[t_idx] = -1.0
        out[list(coords_z)] = z / zn
        out /= np.linalg.norm(out)
        return out

    def slack(x: np.ndarray) -> float:
        z, t = _parts(x)
        return t - float(np.linalg.norm(z))

    return ProjectorCandidate(project=project, name=name,
                              coordinates=(t_idx, *coords_z), normal=normal,
                              kkt_data={"slack": slack, "set": "soc",
                                        "polar_project": _soc_polar_projector(1.0)})


def scaled_soc_projector(t_index: int, z_indices: Sequence[int], mu: float,
                         name: str = "scaled_soc") -> ProjectorCandidate:
    """Return the exact Euclidean projector for ``||z|| <= mu*t``.

    The implementation is the standard second-order-cone projection after
    replacing the cone slope by ``mu``.  It operates on the declared ambient
    coordinates, leaves all other coordinates untouched, and exposes the
    outward boundary normal ``(-mu, z/||z||)`` for the KKT certificate.
    ``mu`` must be finite and strictly positive.
    """
    coords_z = _normalise_coordinates(z_indices)
    assert coords_z is not None
    t_arr = _normalise_coordinates([t_index])
    assert t_arr is not None
    t_idx = t_arr[0]
    if t_idx in coords_z:
        raise ValueError("scaled SOC scalar coordinate must not occur in z_indices")
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("scaled SOC slope mu must be finite and positive")

    def _parts(x: np.ndarray) -> tuple[np.ndarray, float]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if max((t_idx, *coords_z), default=-1) >= arr.size:
            raise ValueError("scaled SOC projector coordinate exceeds state dimension")
        return arr[list(coords_z)], float(arr[t_idx])

    def project(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1).copy()
        z, t = _parts(arr)
        zn = float(np.linalg.norm(z))
        if zn <= mu * t:
            return arr
        # The radial boundary formula is the projection onto the boundary
        # ray.  When its scalar parameter is non-positive, the closest point
        # is the apex; the equivalent axial test is t + mu*||z|| <= 0.
        if t + mu * zn <= 0.0:
            arr[t_idx] = 0.0
            arr[list(coords_z)] = 0.0
            return arr
        t_new = (t + mu * zn) / (1.0 + mu * mu)
        z_new_norm = mu * t_new
        arr[t_idx] = t_new
        arr[list(coords_z)] = (z_new_norm / zn) * z if zn > 0.0 else 0.0
        return arr

    def normal(x: np.ndarray) -> Optional[np.ndarray]:
        z, t = _parts(x)
        zn = float(np.linalg.norm(z))
        out = np.zeros(np.asarray(x).size, dtype=float)
        if zn <= 1e-15:
            # The apex has a full polar cone, so there is no unique outward
            # ray to return.  The certificate uses the polar projection model.
            return None
        out[t_idx] = -mu
        out[list(coords_z)] = z / zn
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 0.0 else None

    def slack(x: np.ndarray) -> float:
        z, t = _parts(x)
        return float(mu * t - np.linalg.norm(z))

    return ProjectorCandidate(project=project, name=name,
                              coordinates=(t_idx, *coords_z), normal=normal,
                              kkt_data={"slack": slack, "set": "scaled_soc",
                                        "mu": mu,
                                        "polar_project": _soc_polar_projector(mu)})


def _psd_shape(shape: Union[Sequence[int], int]) -> int:
    """Validate and normalize the dimension of a packed symmetric matrix."""
    value = np.asarray(shape)
    if value.ndim == 0:
        n = int(value)
    else:
        flat = value.reshape(-1)
        if flat.size == 1:
            n = int(flat[0])
        elif flat.size == 2 and int(flat[0]) == int(flat[1]):
            # Accept the natural matrix-shape spelling as well as ``n``.
            n = int(flat[0])
        else:
            raise ValueError("PSD cone shape must be a positive integer or (n, n)")
    if n <= 0:
        raise ValueError("PSD cone shape must be a positive integer")
    return n


class _Svec:
    """Frobenius-preserving upper-triangular symmetric packing.

    Off-diagonal entries carry ``sqrt(2)`` so that the Euclidean norm of the
    packed vector equals the matrix Frobenius norm.  Keeping this tiny helper
    local avoids an import from an analysis/panel tree while matching the
    canonical packing used by the Shor-lift experiments.
    """

    def __init__(self, n: int):
        self.n = int(n)
        self._pairs = tuple((i, j) for i in range(n) for j in range(i, n))
        self.dim = len(self._pairs)

    def pack(self, matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (self.n, self.n):
            raise ValueError("PSD projector matrix has the wrong shape")
        out = np.empty(self.dim, dtype=float)
        root2 = np.sqrt(2.0)
        for k, (i, j) in enumerate(self._pairs):
            out[k] = arr[i, j] if i == j else root2 * arr[i, j]
        return out

    def unpack(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=float).reshape(-1)
        if arr.size != self.dim:
            raise ValueError(
                f"PSD projector expects {self.dim} packed entries, got {arr.size}")
        out = np.zeros((self.n, self.n), dtype=float)
        root2 = np.sqrt(2.0)
        for value, (i, j) in zip(arr, self._pairs):
            entry = float(value) if i == j else float(value) / root2
            out[i, j] = entry
            out[j, i] = entry
        return out


def psd_cone_projector(shape: Union[Sequence[int], int],
                       coordinates: Optional[Sequence[int]] = None,
                       name: str = "psd_cone") -> ProjectorCandidate:
    """Return the exact Euclidean projector onto a positive-semidefinite cone.

    The state stores a symmetric matrix in Frobenius-preserving ``svec``
    packing.  Projection unpacks, clips negative eigenvalues, and packs again.
    ``coordinates`` can embed the packed block in a larger ambient state.
    Native backends support ``n <= 8``; larger blocks use ``backend='python'``.
    """
    n = _psd_shape(shape)
    svec = _Svec(n)
    coords = _normalise_coordinates(coordinates)
    if coords is not None and len(coords) != svec.dim:
        raise ValueError("PSD cone coordinates must cover the packed symmetric block")

    def _local(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if coords is None:
            if arr.size != svec.dim:
                raise ValueError(
                    f"PSD cone expects a state of dimension {svec.dim}")
            return arr, arr.copy()
        if max(coords, default=-1) >= arr.size:
            raise ValueError("PSD cone coordinate exceeds state dimension")
        return arr[list(coords)], arr

    def project(x: np.ndarray) -> np.ndarray:
        local, ambient = _local(x)
        matrix = svec.unpack(local)
        eigenvalues, vectors = np.linalg.eigh(matrix)
        clipped = np.clip(eigenvalues, 0.0, None)
        projected_local = svec.pack((vectors * clipped) @ vectors.T)
        if coords is None:
            return projected_local
        out = ambient.copy()
        out[list(coords)] = projected_local
        return out

    # The natural-map certificate uses this exact projector.  A finite normal
    # basis is not valid on repeated-eigenvalue faces, so leave ``normal``
    # unset and let the certificate's gradient-map model handle the cone.
    project._snn_native_set = {"token": _SNN_NATIVE_PSD_TOKEN,
                               "family": "psd_cone", "n": n,
                               "coordinates": coords}
    return ProjectorCandidate(
        project=project, name=name, coordinates=coords,
        kkt_data={"euclidean_project": project, "set": "psd_cone",
                  "shape": (n, n), "packed_dimension": svec.dim})


def psd_projector(shape: Union[Sequence[int], int],
                  coordinates: Optional[Sequence[int]] = None,
                  name: str = "psd_cone") -> ProjectorCandidate:
    """Alias for :func:`psd_cone_projector`."""
    return psd_cone_projector(shape, coordinates=coordinates, name=name)


def _member_projector(member: Union[ProjectorCandidate, Callable[[np.ndarray], np.ndarray]],
                      index: int) -> ProjectorCandidate:
    """Normalize a Dykstra member to a :class:`ProjectorCandidate`."""
    if isinstance(member, ProjectorCandidate):
        return member
    if not callable(member):
        raise TypeError(
            f"Dykstra member {index} must be a ProjectorCandidate or callable")
    return ProjectorCandidate(project=member, name=f"member[{index}]")


class DykstraProjector(ProjectorCandidate):
    """Cold-start Euclidean projector onto an intersection of convex sets.

    ``members`` are exact projector candidates.  Each call starts with zero
    Dykstra corrections, so the result is independent of prior calls.  The
    latest call's diagnostics are attached to ``last_diagnostics`` and to the
    callback as ``_dykstra_diagnostics`` for the solver's event accounting.
    The callback returns the best point reached when the inner cap is hit and
    marks that condition explicitly in the diagnostics. The KKT certificate
    trusts a converged call to the configured positional tolerance; this
    tolerance is not alpha-amplification-bounded by the certificate.
    """

    def __init__(self,
                 members: Sequence[Union[ProjectorCandidate,
                                         Callable[[np.ndarray], np.ndarray]]],
                 *, tolerance: float = 1e-12,
                 tol: Optional[float] = None,
                 max_iterations: int = 10000,
                 max_iter: Optional[int] = None,
                 max_inner_iterations: Optional[int] = None,
                 inner_tolerance: Optional[float] = None,
                 name: str = "dykstra",
                 coordinates: Optional[Sequence[int]] = None):
        try:
            members_list = list(members)
        except TypeError as exc:
            raise TypeError("Dykstra members must be an iterable") from exc
        members_tuple = tuple(_member_projector(v, i)
                              for i, v in enumerate(members_list))
        if not members_tuple:
            raise ValueError("DykstraProjector needs at least one member set")
        if tol is not None:
            tolerance = tol
        if inner_tolerance is not None:
            tolerance = inner_tolerance
        if max_iter is not None:
            max_iterations = max_iter
        if max_inner_iterations is not None:
            max_iterations = max_inner_iterations
        tolerance = float(tolerance)
        max_iterations = int(max_iterations)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Dykstra tolerance must be finite and positive")
        if max_iterations <= 0:
            raise ValueError("Dykstra max_iterations must be positive")
        coords = _normalise_coordinates(coordinates)
        diagnostics = {
            "converged": False,
            "cap_hit": False,
            "iterations": 0,
            "inner_iterations": 0,
            "projection_events": 0,
            "max_member_residual": float("inf"),
            "settled_residual": float("inf"),
            "correction_settled_residual": float("inf"),
            "events": [],
            "member_names": tuple(getattr(v, "name", f"member[{i}]")
                                   for i, v in enumerate(members_tuple)),
        }

        def _embed(raw: np.ndarray, y: np.ndarray, member: ProjectorCandidate,
                   member_index: int) -> np.ndarray:
            arr = np.asarray(raw, dtype=float).reshape(-1)
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            if arr.size == y_arr.size:
                out = arr
            else:
                member_coords = getattr(member, "coordinates", None)
                if member_coords is None or arr.size != len(member_coords):
                    raise ValueError(
                        f"Dykstra member {member_index} returned dimension {arr.size}; "
                        f"expected {y_arr.size}")
                out = y_arr.copy()
                out[list(member_coords)] = arr
            if not np.all(np.isfinite(out)):
                raise ValueError(f"Dykstra member {member_index} returned non-finite data")
            member_coords = getattr(member, "coordinates", None)
            if member_coords is not None:
                outside = np.ones(y_arr.size, dtype=bool)
                outside[list(member_coords)] = False
                if np.any(np.abs(out[outside] - y_arr[outside]) > 1e-10):
                    raise ValueError(
                        f"Dykstra member {member_index} changed coordinates outside its declaration")
            return np.asarray(out, dtype=float)

        def project(x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x, dtype=float).reshape(-1)
            if not np.all(np.isfinite(arr)):
                raise ValueError("Dykstra input must be finite")
            if coords is not None and max(coords, default=-1) >= arr.size:
                raise ValueError("Dykstra coordinate exceeds state dimension")
            # The members operate on the ambient state.  A declared Dykstra
            # coordinate block scopes the returned update, matching the other
            # candidate factories and allowing ambient member coordinates.
            state = arr.copy()
            corrections = [np.zeros_like(state) for _ in members_tuple]
            events = []
            max_residual = float("inf")
            settled_residual = float("inf")
            correction_settled_residual = float("inf")
            converged = False
            cycles = 0
            event_count = 0
            for cycle in range(max_iterations):
                previous = state.copy()
                previous_corrections = [correction.copy() for correction in corrections]
                correction_settled_residual = 0.0
                for i, member in enumerate(members_tuple):
                    shifted = state + corrections[i]
                    try:
                        projected_raw = member.project(shifted)
                    except Exception as exc:
                        raise ValueError(
                            f"Dykstra member {i} ({member.name!r}) project callback failed") from exc
                    projected = _embed(projected_raw, shifted, member, i)
                    correction = shifted - projected
                    correction_settled_residual += float(
                        np.linalg.norm(correction - previous_corrections[i]))
                    corrections[i] = correction
                    state = projected
                    correction_norm = float(np.linalg.norm(correction))
                    if not np.isfinite(correction_norm):
                        raise ValueError(f"Dykstra member {i} returned a non-finite correction")
                    events.append({
                        "member_index": i,
                        "member_name": str(getattr(member, "name", f"member[{i}]")),
                        "delta": (-correction).copy(),
                        "correction_norm": correction_norm,
                    })
                    event_count += 1
                cycles = cycle + 1
                settled_residual = float(np.linalg.norm(state - previous))
                # A separate residual pass is needed because later member
                # projections can move the point off an earlier set.  These
                # probes are not committed Dykstra corrections and therefore
                # are not counted as events.
                probe_residuals = []
                for i, member in enumerate(members_tuple):
                    try:
                        probe_raw = member.project(state)
                    except Exception as exc:
                        raise ValueError(
                            f"Dykstra member {i} ({member.name!r}) probe failed") from exc
                    probe = _embed(probe_raw, state, member, i)
                    probe_residuals.append(float(np.linalg.norm(probe - state)))
                max_residual = max(probe_residuals, default=0.0)
                # Once the first cycle has put the state near the
                # intersection, the stopping scale must describe that set,
                # rather than the potentially remote input.
                if coords is None:
                    state_norm = float(np.linalg.norm(state))
                else:
                    state_norm = float(np.linalg.norm(state[list(coords)]))
                scale = max(1.0, state_norm)
                threshold = tolerance * scale
                if (settled_residual <= threshold
                        and max_residual <= threshold
                        and correction_settled_residual <= threshold):
                    converged = True
                    break
            cap_hit = not converged
            if coords is not None:
                outside = np.ones(arr.size, dtype=bool)
                outside[list(coords)] = False
                if np.any(np.abs(state[outside] - arr[outside]) > 1e-10):
                    raise ValueError("Dykstra projector changed coordinates outside its declaration")
                out = arr.copy()
                out[list(coords)] = state[list(coords)]
            else:
                out = state
            diagnostics.clear()
            diagnostics.update({
                "converged": bool(converged),
                "cap_hit": bool(cap_hit),
                "hit_cap": bool(cap_hit),
                "iterations": int(cycles),
                "inner_iterations": int(cycles),
                "projection_events": int(event_count),
                "member_projection_events": int(event_count),
                "max_member_residual": float(max_residual),
                "settled_residual": float(settled_residual),
                "correction_settled_residual": float(correction_settled_residual),
                "events": events,
                "member_names": tuple(getattr(v, "name", f"member[{i}]")
                                       for i, v in enumerate(members_tuple)),
            })
            return out

        # Function attributes are deliberately used as a tiny diagnostics
        # channel.  The immutable candidate API remains compatible with custom
        # callbacks, while the solver can recover inner event details without
        # changing the public callback return type.
        project._dykstra_diagnostics = diagnostics
        project._is_dykstra_projector = True
        super().__init__(project=project, name=name, coordinates=coords,
                         kkt_data={"euclidean_project": project,
                                   "set": "dykstra",
                                   "member_names": diagnostics["member_names"],
                                   "tolerance": tolerance,
                                   "max_iterations": max_iterations})
        object.__setattr__(self, "members", members_tuple)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "last_diagnostics", diagnostics)

    @classmethod
    def from_rows(cls, C: np.ndarray, d: np.ndarray,
                  members: Sequence[ProjectorCandidate] = (), **kwargs):
        """Build a Dykstra projector from affine rows plus extra members."""
        return joint_dykstra_projector(C, d, members=members, **kwargs)


def dykstra_projector(
        members: Sequence[Union[ProjectorCandidate, Callable[[np.ndarray], np.ndarray]]],
        **kwargs) -> DykstraProjector:
    """Factory alias for :class:`DykstraProjector`."""
    return DykstraProjector(members, **kwargs)


def joint_dykstra_projector(
        C: np.ndarray, d: np.ndarray,
        members: Sequence[Union[ProjectorCandidate,
                                Callable[[np.ndarray], np.ndarray]]] = (),
        *, projectors: Optional[Sequence[ProjectorCandidate]] = None,
        cones: Optional[Sequence[ProjectorCandidate]] = None,
        **kwargs) -> DykstraProjector:
    """Build one Dykstra candidate for rows and built-in cone projectors.

    ``C x + d <= 0`` becomes one exact halfspace member per row.  ``members``
    is the preferred name for additional sets; ``projectors`` and ``cones``
    are accepted as readable aliases and are concatenated in that order.
    """
    from scipy.sparse import issparse
    if issparse(C):
        raise ValueError(
            "joint Dykstra C must be a dense array (scipy sparse C is not "
            "supported for Dykstra members)")
    C_arr = np.asarray(C, dtype=float)
    d_arr = np.asarray(d, dtype=float).reshape(-1)
    if C_arr.ndim == 1:
        C_arr = C_arr.reshape(1, -1)
    if C_arr.ndim != 2:
        raise ValueError("joint Dykstra C must be a 2-D array")
    if C_arr.shape[0] != d_arr.size:
        raise ValueError("joint Dykstra C rows must match d length")
    if C_arr.shape[1] == 0:
        raise ValueError("joint Dykstra C must have at least one column")
    if not np.all(np.isfinite(C_arr)) or not np.all(np.isfinite(d_arr)):
        raise ValueError("joint Dykstra rows must be finite")
    rows = tuple(halfspace_projector(C_arr[i], d_arr[i], name=f"row[{i}]")
                 for i in range(C_arr.shape[0]))
    extras = []
    extras.extend(tuple(members))
    if projectors is not None:
        extras.extend(tuple(projectors))
    if cones is not None:
        extras.extend(tuple(cones))
    return DykstraProjector(rows + tuple(extras), **kwargs)


def dykstra_intersection_projector(*args, **kwargs) -> DykstraProjector:
    """Compatibility alias for :class:`DykstraProjector`."""
    return DykstraProjector(*args, **kwargs)


def joint_projector(C: np.ndarray, d: np.ndarray, *args, **kwargs) -> DykstraProjector:
    """Short alias for :func:`joint_dykstra_projector`."""
    return joint_dykstra_projector(C, d, *args, **kwargs)


def _spectral_shape(shape: Union[Sequence[int], int]) -> tuple[int, int]:
    if np.asarray(shape).ndim == 0:
        n = int(shape)
        shape = (n, n)
    try:
        rows, cols = (int(v) for v in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("spectral ball shape must be a two-entry sequence") from exc
    if rows <= 0 or cols <= 0:
        raise ValueError("spectral ball shape must be positive")
    return rows, cols


def _spectral_candidate_parts(shape: Union[Sequence[int], int], radius: float,
                              coordinates: Optional[Sequence[int]]):
    rows, cols = _spectral_shape(shape)
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("spectral ball radius must be finite and non-negative")
    coords = _normalise_coordinates(coordinates)
    size = rows * cols
    if coords is not None and len(coords) != size:
        raise ValueError("spectral ball coordinates must cover shape.size entries")

    def local(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if coords is None:
            if arr.size != size:
                raise ValueError(
                    f"spectral ball expects a state of dimension {size}")
            return arr
        if max(coords, default=-1) >= arr.size:
            raise ValueError("spectral ball coordinate exceeds state dimension")
        return arr[list(coords)]

    def ambient(local_value: np.ndarray, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if coords is None:
            return np.asarray(local_value, dtype=float).reshape(-1)
        out = arr.copy()
        out[list(coords)] = np.asarray(local_value, dtype=float).reshape(-1)
        return out

    def value(x: np.ndarray) -> float:
        singular = np.linalg.svd(local(x).reshape(rows, cols),
                                  compute_uv=False)
        return float(singular[0] - radius)

    def jacobian(x: np.ndarray) -> np.ndarray:
        matrix = local(x).reshape(rows, cols)
        U, singular, Vt = np.linalg.svd(matrix, full_matrices=False)
        # A rank-one event is retained for the projection sweep.  The exact
        # normal cone is supplied separately by the projector metadata.
        if U.shape[1] == 0:
            local_grad = np.zeros(size, dtype=float)
        else:
            local_grad = np.outer(U[:, 0], Vt[0]).reshape(-1)
        # Return the LOCAL gradient: the solver scatters a local-length vector
        # into zeros (_candidate_vector). Routing it through ambient() would copy
        # the rest of the state into the normal and move every other block.
        return local_grad

    def project(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        matrix = local(arr).reshape(rows, cols)
        U, singular, Vt = np.linalg.svd(matrix, full_matrices=False)
        clipped = np.minimum(singular, radius)
        local_projected = (U * clipped) @ Vt
        return ambient(local_projected.reshape(-1), arr)

    stamp = {"token": _SNN_NATIVE_SPECTRAL_TOKEN,
             "family": "spectral_ball", "shape": (rows, cols),
             "radius": radius, "coordinates": coords}
    value._snn_native_set = dict(stamp, jacobian=jacobian, project=project)
    project._snn_native_set = dict(stamp)
    return rows, cols, radius, coords, value, jacobian, project


def spectral_norm_cutter(shape: Union[Sequence[int], int], radius: float = 1.0,
                         coordinates: Optional[Sequence[int]] = None,
                         name: str = "spectral_ball") -> CutterCandidate:
    """Build the spectral-norm ball cutter and its exact SVD-clip projector.

    The cutter retains the rank-one leading singular-vector event used by the
    projection sweep.  Its ``kkt_data['euclidean_project']`` callback gives
    the certificate the full spectral-ball normal cone, including tied faces.
    ``shape`` is the matrix shape (an integer means a square matrix).
    Native backends support at most 8 rows and 8 columns, at top level only.
    Larger blocks use ``backend='python'``. Python retains the LAPACK oracle.
    """
    rows, cols, radius, coords, value, jacobian, project = \
        _spectral_candidate_parts(shape, radius, coordinates)
    return CutterCandidate(
        value=value, jacobian=jacobian, name=name, coordinates=coords,
        kkt_data={"euclidean_project": project, "set": "spectral_ball",
                  "shape": (rows, cols), "radius": radius,
                  "event_oracle": "svd_top_pair"})


def spectral_ball_cutter(shape: Union[Sequence[int], int], radius: float = 1.0,
                         coordinates: Optional[Sequence[int]] = None,
                         name: str = "spectral_ball") -> CutterCandidate:
    """Alias for :func:`spectral_norm_cutter`."""
    return spectral_norm_cutter(shape, radius, coordinates, name)


def spectral_ball_projector(shape: Union[Sequence[int], int], radius: float = 1.0,
                            coordinates: Optional[Sequence[int]] = None,
                            name: str = "spectral_ball") -> ProjectorCandidate:
    """Build an exact spectral-ball projector, also usable inside Dykstra.

    Native backends support at most 8 rows and 8 columns; larger blocks use
    ``backend='python'``. Python retains the LAPACK SVD implementation.
    """
    rows, cols, radius, coords, _value, _jacobian, project = \
        _spectral_candidate_parts(shape, radius, coordinates)
    return ProjectorCandidate(
        project=project, name=name, coordinates=coords,
        kkt_data={"euclidean_project": project, "set": "spectral_ball",
                  "shape": (rows, cols), "radius": radius})


class LiftedSOCResult(NamedTuple):
    """Return object for :func:`lift_soc_l1` and :func:`lift_soc_l2`.

    It behaves like the documented three-tuple ``(problem, coordinates,
    cone_candidate)`` and also gives named attributes for interactive use.
    """

    problem: Any
    coordinates: Mapping[str, Any]
    cone: ProjectorCandidate

    @property
    def coordinate_map(self) -> Mapping[str, Any]:
        return self.coordinates

    @property
    def cone_candidate(self) -> ProjectorCandidate:
        return self.cone


def _lift_inputs(A, b, K, c, e, f):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    if K.ndim == 1:
        K = K.reshape(1, -1)
    c = np.asarray(c, dtype=float).reshape(-1)
    e = np.zeros(b.size, dtype=float) if e is None else np.asarray(e, dtype=float).reshape(-1)
    f = float(np.asarray(f, dtype=float).reshape(-1)[0])
    if A.shape != (b.size, b.size):
        raise ValueError("A must be square with dimension matching b")
    if K.shape[1] != b.size or K.shape[0] != c.size:
        raise ValueError("K and c dimensions are inconsistent with b")
    if e.size != b.size:
        raise ValueError("e must have the same dimension as b")
    if not all(np.all(np.isfinite(v)) for v in (A, b, K, c, e)) or not np.isfinite(f):
        raise ValueError("lift inputs must be finite")
    if not np.allclose(A, A.T, atol=1e-12, rtol=1e-12):
        raise ValueError("A must be symmetric for the lifted objective")
    return A, b, K, c, e, f


def _lift_soc(A, b, K, c, e, f, *, arm: str, cone: str = "soc") -> LiftedSOCResult:
    from .solver import OptimizationProblem  # lazy import, see module docstring

    if cone.lower() not in ("soc", "second_order", "second-order"):
        raise ValueError("cone must be 'soc'")
    A, b, K, c, e, f = _lift_inputs(A, b, K, c, e, f)
    n, q = b.size, K.shape[0]
    t_variable = bool(np.linalg.norm(e) > 1e-14)
    x_idx = np.arange(n, dtype=int)
    z_idx = np.arange(n, n + q, dtype=int)
    if t_variable:
        t_idx = int(n + q)
        dim = n + q + 1
    else:
        t_idx = None
        dim = n + q

    A_lift = np.zeros((dim, dim), dtype=float)
    A_lift[:n, :n] = A
    b_lift = np.zeros(dim, dtype=float)
    b_lift[:n] = b

    rows = []
    offs = []
    # z - Kx = c, represented with the same row orientation in both arms.
    coupling = np.zeros((q, dim), dtype=float)
    coupling[:, :n] = -K
    coupling[:, n:n + q] = np.eye(q)
    rows.extend([coupling, -coupling])
    offs.extend([-c, c])

    if t_variable:
        trow = np.zeros(dim, dtype=float)
        trow[:n] = -e
        trow[t_idx] = 1.0
        rows.extend([trow.reshape(1, -1), -trow.reshape(1, -1)])
        offs.extend([np.array([-f]), np.array([f])])

    if arm == "l1":
        C = np.vstack(rows) if rows else np.zeros((0, dim), dtype=float)
        d = np.concatenate(offs) if offs else np.zeros(0, dtype=float)
        if t_variable:
            cone_candidate = soc_projector(t_idx, z_idx, name="soc")
        else:
            cone_candidate = ball_projector(z_idx, f, name="ball")
        nonlinear_candidates = (cone_candidate,)
    elif arm == "l2":
        # A single equality projector replaces all opposed rows.  The generic
        # graph projector receives the map from x to the auxiliary q=(z,t)
        # block, so it can use the precomputed (I+B^T B)^-1 formula.
        if t_variable:
            B = np.vstack((K, e.reshape(1, -1)))
            h = np.concatenate((c, [f]))
        else:
            B = K
            h = c
        subspace = AffineSubspaceProjector(B, h, name="affine_subspace")
        C = np.zeros((0, dim), dtype=float)
        d = np.zeros(0, dtype=float)
        if t_variable:
            cone_candidate = soc_projector(t_idx, z_idx, name="soc")
        else:
            cone_candidate = ball_projector(z_idx, f, name="ball")
        nonlinear_candidates = (subspace, cone_candidate)
    else:
        raise ValueError("arm must be 'l1' or 'l2'")

    problem = OptimizationProblem(A_lift, b_lift, C, d,
                                  nonlinear_candidates=nonlinear_candidates)
    coords = {"x": x_idx, "z": z_idx, "t": t_idx,
              "dimension": dim, "t_variable": t_variable}
    return LiftedSOCResult(problem, coords, cone_candidate)


def lift_soc_l1(A, b, K, c, e, f, cone: str = "soc") -> LiftedSOCResult:
    """Build the equality-pair plus exact-cone (L1) lifted problem."""
    return _lift_soc(A, b, K, c, e, f, arm="l1", cone=cone)


def lift_soc_l2(A, b, K, c, e, f, cone: str = "soc") -> LiftedSOCResult:
    """Build the affine-subspace-projector plus exact-cone (L2) lift."""
    return _lift_soc(A, b, K, c, e, f, arm="l2", cone=cone)


__all__ = [
    "CutterCandidate",
    "ProjectorCandidate",
    "AffineSubspaceProjector",
    "DykstraProjector",
    "affine_cutter",
    "halfspace_projector",
    "ball_projector",
    "box_projector",
    "soc_projector",
    "scaled_soc_projector",
    "psd_cone_projector",
    "psd_projector",
    "dykstra_projector",
    "joint_dykstra_projector",
    "dykstra_intersection_projector",
    "joint_projector",
    "spectral_norm_cutter",
    "spectral_ball_cutter",
    "spectral_ball_projector",
    "LiftedSOCResult",
    "lift_soc_l1",
    "lift_soc_l2",
]
