# Spiking Neural Network Dynamics for Real-Time Constrained Optimization

## Abstract

This report presents a computational framework that interprets spiking neural network (SNN) dynamics as gradient-based constrained optimization. We develop a real-time optimization algorithm inspired by leaky integrate-and-fire neuron models, where constraint violations trigger discrete correction events analogous to neural spikes. The method solves quadratic and linear programs through continuous gradient descent punctuated by boundary reflections, enabling efficient real-time implementation on embedded systems. We demonstrate the approach through robotic manipulator control, where the algorithm computes optimal joint velocities subject to end-effector constraints at each control timestep. The simplicity of the computational primitives (matrix-vector operations and projections) makes the method suitable for real-time applications requiring sub-millisecond solve times.

---

## 1. Introduction and Background

### 1.1 Motivation

Real-time control systems often require solving optimization problems at high frequencies, sometimes at kilohertz rates for robotic systems or millisecond rates for autonomous vehicles. Traditional optimization solvers, while mathematically sophisticated, may introduce computational overhead that challenges real-time constraints. Interior point methods require matrix factorizations, active set methods maintain complex data structures, and gradient projection methods may require careful line search procedures.

Meanwhile, biological neural systems solve complex optimization-like problems in real time using networks of simple computational units. Neurons integrate inputs, compare against thresholds, and produce discrete output spikes. This suggests that simple, iterative algorithms based on local computations might suffice for many optimization tasks, particularly when approximate solutions updated at high frequency are preferable to exact solutions computed slowly.

Recent work has established connections between spiking neural network dynamics and convex optimization, showing that certain classes of SNNs implicitly solve quadratic and linear programs. This report develops these insights into a practical algorithmic framework suitable for real-time control applications.

### 1.2 Connection to Spiking Neural Networks

A leaky integrate-and-fire (LIF) neuron model describes voltage dynamics as:

```math
\dot{V}_i = -\lambda V_i + I_i(t)
```

where $V_i$ is the membrane voltage, $\lambda$ is the leak rate, and $I_i(t)$ represents input currents. When $V_i$ reaches a threshold $T_i$, the neuron fires a spike and the voltage resets.

For a network of $N$ neurons with recurrent connectivity, the voltage dynamics become:

```math
\dot{\mathbf{V}} = -\lambda\mathbf{V} + \mathbf{F}\mathbf{c}(t) + \mathbf{\Omega}\mathbf{s}(t) + \mathbf{I}_{bg}
```

where $\mathbf{V} \in \mathbb{R}^N$ are voltages, $\mathbf{F}$ encodes feedforward weights, $\mathbf{c}(t)$ are inputs, $\mathbf{\Omega}$ represents recurrent connectivity, and $\mathbf{s}(t)$ are spike trains modeled as sums of delta functions.

The key insight connecting SNNs to optimization is that voltage thresholds naturally correspond to inequality constraints. The condition $\mathbf{V} \leq \mathbf{T}$ defines a feasible region in state space, and spikes act as projection operators that keep the system within this region while descending an objective function.

---

## 2. Mathematical Formulation

### 2.1 The Constrained Optimization Problem

We consider convex optimization problems with linear inequality constraints:

```math
\begin{aligned}
\min_{\mathbf{y}} \quad & E(\mathbf{y}) = \frac{\lambda}{2}\mathbf{y}^\top\mathbf{y} + \mathbf{b}^\top\mathbf{y} \\
\text{subject to} \quad & \mathbf{C}\mathbf{y} + \mathbf{d} \leq \mathbf{0}
\end{aligned}
```

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the optimization variable
- $\lambda \geq 0$ determines whether we have a quadratic program ($\lambda > 0$) or linear program ($\lambda = 0$)
- $\mathbf{b} \in \mathbb{R}^n$ provides a linear cost term
- $\mathbf{C} \in \mathbb{R}^{m \times n}$ defines the constraint geometry with $m$ constraints
- $\mathbf{d} \in \mathbb{R}^m$ specifies constraint offsets

The constraint $\mathbf{C}\mathbf{y} + \mathbf{d} \leq \mathbf{0}$ defines the feasible region as the intersection of $m$ half-spaces. Each row $\mathbf{C}_i$ of the constraint matrix defines a hyperplane, and the $i$-th constraint requires that $\mathbf{y}$ lies on one side of this hyperplane.

### 2.2 Geometric Interpretation

The $i$-th inequality constraint can be written as:

```math
\mathbf{C}_i^\top \mathbf{y} \leq -d_i
```

This defines a closed half-space in $\mathbb{R}^n$. The boundary of the feasible region consists of points where at least one constraint is active (holds with equality). At the optimal solution $\mathbf{y}^*$, a subset of constraints will be active, defining the optimal face of the feasible polytope.

The gradient of the objective function is:

```math
\nabla E(\mathbf{y}) = \lambda\mathbf{y} + \mathbf{b}
```

For a quadratic program ($\lambda > 0$), this gradient points away from the origin with magnitude proportional to $\Vert\mathbf{y}\Vert$, creating a "pull" toward zero. For a linear program ($\lambda = 0$), the gradient is constant at $\mathbf{b}$.

### 2.3 Gradient Descent with Boundary Projections

The unconstrained gradient descent dynamics are:

```math
\dot{\mathbf{y}} = -k_0 \nabla E(\mathbf{y}) = -k_0(\lambda\mathbf{y} + \mathbf{b})
```

where $k_0 > 0$ is the step size parameter. This would minimize $E(\mathbf{y})$ in the absence of constraints, but may leave the feasible region.

To enforce constraints, we augment the dynamics with projection events. When the $i$-th constraint is violated (i.e., $g_i(\mathbf{y}) = \mathbf{C}_i^\top \mathbf{y} + d_i > 0$), we project back onto the constraint boundary.

#### Fixed Step Projection (Original Method)

The original formulation uses a fixed step size:

```math
\mathbf{y} \leftarrow \mathbf{y} - k_1 \mathbf{C}_i
```

where $k_1 > 0$ controls the projection magnitude. This requires tuning $k_1$ and may need multiple iterations to reach the boundary.

#### Adaptive Projection (Improved Method)

A more efficient approach computes the exact step to reach the constraint boundary. For a violated constraint $g_j(\mathbf{y}) = \mathbf{c}_j^\top \mathbf{y} + d_j > 0$, the exact orthogonal projection onto the constraint hyperplane is:

```math
\mathbf{y} \leftarrow \mathbf{y} - \frac{g_j(\mathbf{y})}{\|\mathbf{c}_j\|^2} \mathbf{c}_j
```

**Derivation:** We seek the point $\mathbf{y}'$ on the hyperplane $\mathbf{c}_j^\top \mathbf{y}' + d_j = 0$ that is closest to $\mathbf{y}$. The projection moves along the normal direction $\mathbf{c}_j$:

```math
\mathbf{y}' = \mathbf{y} - \alpha \mathbf{c}_j
```

Substituting into the constraint equation:

```math
\mathbf{c}_j^\top(\mathbf{y} - \alpha \mathbf{c}_j) + d_j = 0
```

```math
\mathbf{c}_j^\top \mathbf{y} + d_j = \alpha \|\mathbf{c}_j\|^2
```

```math
\alpha = \frac{g_j(\mathbf{y})}{\|\mathbf{c}_j\|^2}
```

This adaptive step eliminates $k_1$ as a hyperparameter and projects exactly onto the boundary in one step per constraint.

**Neuromorphic Interpretation:** The adaptive projection is equally neuromorphic: it corresponds to a neuron that resets its membrane potential exactly to the threshold after firing, rather than decaying by a fixed amount. Both are valid integrate-and-fire models.

### 2.4 Algorithm Dynamics

The complete algorithm alternates between two phases:

**Phase 1: Gradient Descent** (continuous or discrete)

When all constraints are satisfied, follow the gradient. This can be implemented as:
- **Continuous-time (IVP):** Integrate $\dot{\mathbf{y}} = -k_0(\lambda\mathbf{y} + \mathbf{b})$ using ODE solvers with event detection
- **Discrete-time (Euler):** Apply $\mathbf{y} \leftarrow \mathbf{y} - k_0 \nabla E(\mathbf{y})$ at fixed time steps

The Euler method is often more stable for tightly constrained problems, and is equally neuromorphic since neurons accumulate potential in discrete time steps.

**Phase 2: Constraint Projection** (discrete events / spikes)

When any constraint becomes violated, apply corrections. Two methods are available:

**Fixed Step (original):**

```math
\mathbf{y} \leftarrow \mathbf{y} - k_1 \mathbf{C}^\top \mathbb{1}_{\mathbf{g}(\mathbf{y}) > 0}
```

where $\mathbb{1}_{\mathbf{g}(\mathbf{y}) > 0}$ is an indicator vector for violated constraints.

**Adaptive Step (improved):**
Project onto the most violated constraint, where "most violated" is measured by
the *normalized* (geometric) distance so differently scaled rows compete fairly,
with an exact step:

```math
j = \arg\max_i \frac{g_i(\mathbf{y})}{\|\mathbf{c}_i\|}
```

```math
\mathbf{y} \leftarrow \mathbf{y} - \frac{g_j(\mathbf{y})}{\|\mathbf{c}_j\|^2} \mathbf{c}_j
```

Since v0.5.0 box bounds participate in the same sweep as implicit unit-normal
facets, competing on the same normalized distance.

Repeat until all constraints satisfied.

The adaptive method eliminates $k_1$ as a hyperparameter and converges faster by computing exact projections. The algorithm converges to a neighborhood of the optimal solution $\mathbf{y}^*$ with error bounded by $k_0$ and the constraint tolerance.

### 2.5 Convergence Properties

**Theorem 1 (Informal):** For a convex quadratic program with $\lambda > 0$, proper choice of step sizes $k_0$ and $k_1$ ensures that the iterates remain in the feasible region and converge to a bounded neighborhood of the optimal solution.

The proof sketch relies on several observations:

1. The gradient descent phase decreases the objective function: $\frac{d}{dt}E(\mathbf{y}) = -k_0 \Vert\nabla E(\mathbf{y})\Vert^2 \leq 0$

2. Projections maintain feasibility: after applying corrections, $\mathbf{C}\mathbf{y} + \mathbf{d} \leq \mathbf{0}$

3. The objective function is bounded below on the feasible set (by convexity and compactness arguments)

4. The discrete jumps introduce bounded error: $\Vert\mathbf{y} - \mathbf{y}^*\Vert = O(k_1)$

A rigorous convergence analysis would require specifying bounds on $k_0$ relative to the condition number of the Hessian $\lambda \mathbf{I}$ and bounds on $k_1$ relative to the constraint geometry. For practical implementation, empirical tuning of these parameters suffices.

---

## 3. Algorithm Implementation

### 3.1 Pseudocode

#### Euler Integration with Adaptive Projection (Recommended)

```
Algorithm: SNN-Inspired Constrained Optimization (Euler + Adaptive)

Input: A, b, C, d, y_0, k_0, max_iter, tol
Output: y (approximate solution)

Initialize y ← y_0
Precompute ||c_j||² for each constraint j

for iter = 1 to max_iter:
    # Phase 1: Gradient descent step
    y ← y - k_0 * (A*y + b)
    
    # Phase 2: Adaptive projection (spike phase)
    while true:
        g ← C*y + d                    # Constraint values (membrane voltages)
        j ← argmax(g / ‖c‖)            # Most violated by normalized distance
        if g[j]/‖c_j‖ ≤ tol: break     # All satisfied (geometric tolerance)
        
        # Exact projection onto constraint boundary (spike)
        k_adaptive ← g[j] / ||c_j||²
        y ← y - k_adaptive * c_j
    
return y
```

#### IVP Integration with Fixed Projection (Original)

```
Algorithm: SNN-Inspired Constrained Optimization (IVP + Fixed)

Input: A, b, C, d, y_0, k_0, k_1, t_end
Output: y(t) for t ∈ [0, t_end]

Initialize y ← y_0, t ← 0

while t < t_end:
    # Phase 1: Constraint enforcement (discrete)
    while C*y + d has any positive elements:
        violations ← (C*y + d > 0)
        y ← y - k_1 * C^T * violations
    
    # Phase 2: Gradient descent (continuous)
    Integrate dy/dt = -k_0*(A*y + b) until:
        - Next constraint violation, or
        - Time reaches t_end
    
    Update t to current time
    
return y
```

The Euler + Adaptive method is recommended for most applications as it:
1. Eliminates $k_1$ as a hyperparameter
2. Provides more stable convergence for tightly constrained problems
3. Computes exact boundary projections in one step per constraint

### 3.2 Implementation Details

**Constraint Violation Detection:**
The algorithm monitors the constraint function $\mathbf{g}(\mathbf{y}) = \mathbf{C}\mathbf{y} + \mathbf{d}$ during gradient descent. When any component becomes positive, the integration halts and projections are applied. This can be implemented using event detection in ODE solvers.

**Multiple Simultaneous Violations:**
When multiple constraints are violated, the projection step corrects all violations simultaneously:

```math
\mathbf{y} \leftarrow \mathbf{y} - k_1 \mathbf{C}^\top \mathbb{1}_{\mathbf{g}(\mathbf{y}) > 0}
```

In practice, this may require iterating the projection step several times until all constraints are satisfied, particularly near vertices of the feasible polytope where many constraints are nearly active.

**Numerical Stability:**
To prevent numerical issues near constraint boundaries:
- Use a small tolerance $\epsilon$ when checking violations: $g_i > \epsilon$ rather than $g_i > 0$
- Limit the maximum number of projection iterations per timestep
- Monitor the objective function value to detect divergence

**Parameter Selection:**
The step sizes $k_0$ and $k_1$ must be chosen considering:
- Problem conditioning: larger eigenvalues of $A$ may require smaller $k_0$
- Constraint geometry: acute angles between constraints may require careful $k_1$ tuning
- Real-time requirements: smaller steps mean more iterations but better accuracy

---

## 4. Computational Complexity

### 4.1 Per-Iteration Cost

Each iteration of the algorithm requires:

**Gradient evaluation:** $O(n^2)$ for computing $\mathbf{A}\mathbf{y}$, assuming $\mathbf{A}$ is dense

**Constraint evaluation:** $O(mn)$ for computing $\mathbf{C}\mathbf{y} + \mathbf{d}$

**Projection step:** $O(mn)$ for computing $\mathbf{C}^\top \mathbb{1}$

The dominant cost is typically the matrix-vector products, which are $O(n^2 + mn)$ per iteration. For sparse matrices, this reduces to $O(\text{nnz}(A) + \text{nnz}(C))$ where nnz denotes the number of non-zero entries.

### 4.2 Convergence Rate

The number of iterations required depends on:
- Initial distance to the solution: $\Vert\mathbf{y}_0 - \mathbf{y}^*\Vert$
- Problem conditioning: condition number of $\mathbf{A}$
- Step size $k_0$: larger steps mean fewer iterations but risk instability

For well-conditioned problems, convergence is typically linear with rate determined by $`k_0 \lambda_{\min}(A)`$ where $\lambda_{\min}$ is the smallest eigenvalue.

### 4.3 Comparison to Standard Methods

**Interior Point Methods:**
- Complexity: $O(n^3)$ per iteration due to matrix factorizations
- Iterations: Typically $10-50$ iterations to high accuracy
- Advantage: Polynomial-time guarantee for global optimum
- Disadvantage: Heavy per-iteration cost

**Active Set Methods:**
- Complexity: $O(n^3)$ per iteration (solving linear systems)
- Iterations: Varies, can be exponential in worst case
- Advantage: Exploits problem structure, good warm-start performance
- Disadvantage: Complex data structures, bookkeeping overhead

**Gradient Projection Methods:**
- Complexity: $O(n^2 + mn)$ per iteration (similar to our method)
- Iterations: Depends on conditioning and line search
- Advantage: Simple, suitable for large-scale problems
- Disadvantage: Slower convergence than Newton methods

**This Method (SNN-Inspired):**
- Complexity: $O(n^2 + mn)$ per iteration
- Iterations: Typically $10-100$ for practical convergence
- Advantage: Extremely simple implementation, no auxiliary data structures, suitable for embedded systems
- Disadvantage: Approximate solutions, may require tuning

For real-time control applications where approximate solutions computed frequently are preferable to exact solutions computed slowly, the simplicity advantage becomes decisive.

---

## 5. Real-Time Control Applications

### 5.1 Receding Horizon Control Framework

For control problems, we apply the optimization algorithm in a model predictive control (MPC) framework:

1. **Measure current state** $\mathbf{x}(t)$
2. **Solve optimization** for control input $\mathbf{u}^*(t)$
3. **Apply control** for one timestep
4. **Advance dynamics** to $\mathbf{x}(t + \Delta t)$
5. **Repeat** at next timestep

This receding horizon approach requires solving a new optimization problem at each control timestep, making computational efficiency critical.

### 5.2 Discrete-Time Formulation

At each control timestep $t_k$, we solve:

```math
\begin{aligned}
\min_{\mathbf{u}} \quad & \frac{1}{2}\mathbf{u}^\top \mathbf{A} \mathbf{u} + \mathbf{b}^\top \mathbf{u} \\
\text{subject to} \quad & \mathbf{C}(t_k, \mathbf{x}_k) \mathbf{u} + \mathbf{d}(t_k, \mathbf{x}_k) \leq \mathbf{0}
\end{aligned}
```

where the constraint matrices $\mathbf{C}$ and $\mathbf{d}$ may depend on the current time and state. Critically, these are **held constant during each optimization solve**, avoiding the "chasing a moving target" problem.

### 5.3 Warm Starting

A key advantage for control applications is **warm starting**: we initialize each solve with the solution from the previous timestep:

```math
\mathbf{u}_0^{(k)} = \mathbf{u}^{*(k-1)}
```

For smoothly varying problems, this provides an excellent initialization, often requiring only a few iterations to converge. This dramatically reduces computational cost compared to cold-start methods.

### 5.4 Handling Constraint Changes

When constraints change rapidly between timesteps, care must be taken:

**Feasibility maintenance:** The warm-start solution may violate new constraints. The projection phase automatically handles this by correcting violations before gradient descent begins.

**Constraint geometry changes:** If constraint orientations change significantly, the solution may need to move large distances in state space. This may require more iterations or acceptance of larger approximation error.

---

## 6. Case Study: Robotic Manipulator Control

### 6.1 Problem Setup

Consider a robotic manipulator with $n$ joints (degrees of freedom). The configuration is described by joint angles $\boldsymbol{\theta} \in \mathbb{R}^n$, and we control joint velocities $\dot{\boldsymbol{\theta}} = \mathbf{u} \in \mathbb{R}^n$.

**Control Objective:** Track a desired end-effector velocity $\dot{\mathbf{r}}_d(t) \in \mathbb{R}^3$ while minimizing control effort.

**Kinematics:** The relationship between joint velocities and end-effector velocity is:

```math
\dot{\mathbf{r}} = \mathbf{J}(\boldsymbol{\theta}) \mathbf{u}
```

where $\mathbf{J}(\boldsymbol{\theta}) \in \mathbb{R}^{3 \times n}$ is the Jacobian matrix, computed from the manipulator's forward kinematics.

### 6.2 Optimization Formulation

At each timestep, we solve:

```math
\begin{aligned}
\min_{\mathbf{u}} \quad & \frac{1}{2}\mathbf{u}^\top \mathbf{u} \\
\text{subject to} \quad & \|\mathbf{J}(\boldsymbol{\theta}) \mathbf{u} - \dot{\mathbf{r}}_d\| \leq \delta
\end{aligned}
```

where $\delta > 0$ is a tolerance on velocity tracking error. This objective minimizes control effort (encouraging smooth motions) while approximately tracking the desired velocity.

### 6.3 Constraint Reformulation

The constraint $\Vert\mathbf{J}\mathbf{u} - \dot{\mathbf{r}}_d\Vert \leq \delta$ is equivalent to:

```math
-\delta \leq [\mathbf{J}\mathbf{u} - \dot{\mathbf{r}}_d]_i \leq \delta, \quad i = 1,2,3
```

This can be written as linear inequalities:

```math
\begin{aligned}
\mathbf{J}\mathbf{u} - \dot{\mathbf{r}}_d &\leq \delta \mathbf{1} \\
-\mathbf{J}\mathbf{u} + \dot{\mathbf{r}}_d &\leq \delta \mathbf{1}
\end{aligned}
```

where $\mathbf{1} = [1, 1, 1]^\top$. This gives us $m = 6$ linear constraints in the form required by our algorithm:

```math
\mathbf{C} = \begin{bmatrix} \mathbf{J} \\ -\mathbf{J} \end{bmatrix}, \quad \mathbf{d} = \begin{bmatrix} -\dot{\mathbf{r}}_d - \delta\mathbf{1} \\ \dot{\mathbf{r}}_d - \delta\mathbf{1} \end{bmatrix}
```

### 6.4 Algorithm Application

The control loop proceeds as follows:

```
for each control timestep k:
    1. Measure current joint angles θ_k
    2. Compute Jacobian J(θ_k)
    3. Set up optimization:
       - A = I (identity matrix)
       - b = 0
       - C = [J; -J]
       - d = [-ṙ_d - δ; ṙ_d - δ]
       - u_0 = u_{k-1} (warm start)
    
    4. Solve optimization using SNN algorithm
       → obtain u_k*
    
    5. Apply control: θ_{k+1} = θ_k + u_k* Δt
```

### 6.5 Discussion: Velocity vs Position Control

This formulation optimizes in **velocity space** rather than position space. This creates an important characteristic:

**Open-loop position tracking:** Small errors in velocity optimization accumulate as position drift over time. If the optimal velocity is computed with error $\epsilon$, the position error grows as $O(\epsilon \cdot t)$.

**Why velocity space?** We optimize velocities because:
1. The constraint $\mathbf{J}\mathbf{u} = \dot{\mathbf{r}}_d$ is linear in $\mathbf{u}$ (whereas position constraints would be nonlinear through inverse kinematics)
2. Computational simplicity allows very high control rates
3. For short time horizons, accumulated drift is acceptable

**Mitigation strategies:** To reduce position drift:
1. Run the control loop at high frequency (reducing integration time)
2. Add position feedback: $\dot{\mathbf{r}}_d \leftarrow \dot{\mathbf{r}}_d + K_p(\mathbf{r}_d - \mathbf{r})$
3. Accept that this is an instantaneous velocity controller, suitable for trajectory tracking but not long-term position holding

### 6.6 Computational Performance

For a 7-DOF manipulator ($n = 7$):
- Jacobian computation: $O(n) \approx O(10)$ operations (exploiting kinematic structure)
- Optimization solve: $O(n^2 + mn) = O(49 + 42) \approx O(100)$ operations per iteration
- Typical iterations to convergence: 10-50
- Total computation: ~1000-5000 floating point operations

On a modern embedded processor (e.g., ARM Cortex-M7 at 400 MHz), this easily achieves sub-millisecond solve times, enabling kilohertz control rates.

---

## 7. Implementation Considerations

### 7.1 Numerical Precision

The algorithm uses only basic operations (matrix-vector multiplications, additions, comparisons), which are numerically stable for well-conditioned problems. Potential numerical issues:

**Constraint boundary oscillations:** Near constraint boundaries, floating-point errors may cause oscillations between feasible and infeasible. Use a small tolerance $\epsilon = 10^{-6}$ when checking constraints.

**Accumulation of projection errors:** Many rapid projections may cause drift. Monitor $\Vert\mathbf{C}\mathbf{y} + \mathbf{d}\Vert$ to ensure constraint satisfaction.

### 7.2 Real-Time Guarantees

For hard real-time systems, deterministic execution is required:

**Bounded iteration counts:** Set maximum iteration limits for both projection loops and gradient descent steps.

**Fixed-step integration:** Use fixed timestep integrators (e.g., forward Euler, RK4 with fixed steps) rather than adaptive methods.

**Worst-case analysis:** Analyze maximum computation time for worst-case constraint configurations.

### 7.3 Parameter Tuning Guidelines

**Step size $k_0$ (Auto-computed by default):**

The gradient descent step size can be automatically computed from the Hessian's Lipschitz constant:

```math
k_0 = \frac{k_0^{\text{scale}}}{\lambda_{\max}(\mathbf{A})}
```

where $\lambda_{\max}(\mathbf{A})$ is the largest eigenvalue of the Hessian matrix. This guarantees convergence for convex QPs since the step size is bounded by the inverse of the Lipschitz constant of the gradient.

- **Auto mode (recommended):** Set `k0=None` to automatically compute from $\lambda_{\max}(\mathbf{A})$
- **Manual mode:** For $\mathbf{A} = \mathbf{I}$, try $k_0 \in [0.01, 0.1]$
- Use `k0_scale` (default 0.5) to adjust the conservativeness of the auto-computed step

**Neuromorphic interpretation:** Auto $k_0$ is computed once during network initialization (analogous to setting synaptic time constants based on network topology), not per-iteration.

**Projection method:**
- **Adaptive (recommended):** Eliminates $k_1$ as a hyperparameter by computing exact projections. Use this for most problems.
- **Fixed:** Uses constant step $k_1$. May be useful when adaptive projection causes numerical issues (rare).

**Projection magnitude $k_1$ (only for fixed projection):**
- Start with $k_1 \approx k_0$
- Increase if many projection iterations are needed
- Decrease if projections overshoot into the interior

**Constraint tolerance:**
- Default $10^{-6}$ works for most problems
- Decrease for higher precision (may need more iterations)
- Increase for faster convergence with looser constraint satisfaction

**Tolerance $\delta$ (for tracking problems):**
- For control problems, relates to acceptable tracking error
- Larger $\delta$ makes the problem easier (larger feasible region)
- Smaller $\delta$ gives tighter tracking but may be infeasible

### 7.4 Software Implementation

The algorithm is straightforward to implement in any programming language. Key considerations:

**Matrix libraries:** Use efficient BLAS implementations for matrix-vector operations
**Memory allocation:** Pre-allocate all arrays to avoid runtime memory management
**Profiling:** Profile to identify computational bottlenecks
**Testing:** Verify against standard QP solvers on test problems

### 7.5 Box Constraint Handling

Many optimization problems include simple bound constraints (box constraints):

```math
l_i \leq y_i \leq u_i
```

Since v0.5.0 these are handled as **implicit unit-normal facets inside the same
projection sweep** that handles the rows of $\mathbf{C}$. A bound behaves as one
more constraint the population can spike against; because its normal is a single
coordinate axis, the correction is an $O(1)$ single-coordinate update with an
$O(m)$ lateral residual refresh, rather than a general $O(n)$ projection. Bounds
therefore cost less than encoding them as rows, without being handled by a
separate mechanism.

> **Historical note, and a correction to earlier versions of this document.**
> Before v0.5.0 bounds were enforced by a **terminal clip** applied *after* the
> halfspace sweep, with nothing re-projecting behind it, and this section
> recommended that design. **That recommendation was wrong.** Clipping onto the
> box after projecting onto a halfspace is a sequential projection onto two sets,
> and sequential projection onto two sets is not projection onto their
> intersection: this is the classical POCS failure. When a bound and an
> interacting row are simultaneously active, the clip pushes the iterate back out
> of the halfspace, the sweep pushes it back out of the box, and the solve stalls
> at a point that is feasible for neither. The signature is a box violation of
> exactly $0$ next to a small but stubborn row violation, and an objective that
> can come in *below* the true optimum, which is impossible for a feasible point.
>
> The previous text also claimed the two mechanisms "decouple". They do not, and
> the case where they interact most strongly is precisely the SVM dual below,
> where the box $0 \le \alpha_i \le C$ and the equality
> $\mathbf{y}^\top\boldsymbol{\alpha} = 0$ are both active at the solution.

**Neuromorphic interpretation:** a bound corresponds to **neuron saturation**.
Biological neurons have natural firing-rate bounds: a neuron cannot fire at
negative rates (lower bound) and has a maximum rate set by its refractory period
(upper bound). Treating that saturation as one more facet the population spikes
against keeps the interpretation while making the geometry correct.

**Application to SVM:** for the SVM dual with $0 \leq \alpha_i \leq C$ and the
equality $\mathbf{y}^\top \boldsymbol{\alpha} = 0$, set `lower_bound=0` and
`upper_bound=C` and let the unified sweep resolve the bounds and the equality
together. On v0.5.0 or later, check `result.joint_feasible` rather than the
row-only violation: the joint measure is the one that accounts for both.

### 7.6 Extension to Equality Constraints

The formulation handles inequality constraints naturally. Equality constraints $`\mathbf{A}_{eq}\mathbf{y} = \mathbf{b}_{eq}`$ can be incorporated as pairs of inequalities:

```math
\mathbf{A}_{eq}\mathbf{y} \leq \mathbf{b}_{eq}, \quad -\mathbf{A}_{eq}\mathbf{y} \leq -\mathbf{b}_{eq}
```

**Why this works with adaptive projection:** The adaptive projection formula $\mathbf{y} \leftarrow \mathbf{y} - \frac{g_j}{\Vert\mathbf{c}_j\Vert^2}\mathbf{c}_j$ projects exactly onto the constraint boundary. For equality constraints expressed as two opposing inequalities:
- If $\mathbf{a}^\top \mathbf{y} > b$ (positive side): the first inequality is violated, projection moves toward $\mathbf{a}^\top \mathbf{y} = b$
- If $\mathbf{a}^\top \mathbf{y} < b$ (negative side): the second inequality is violated, projection moves toward $\mathbf{a}^\top \mathbf{y} = b$
- Either way, we end up on the equality hyperplane

This approach doubles the number of constraints but works well in practice. For problems with many equality constraints, direct projection onto the equality manifold may be more efficient as a preprocessing step.

### 7.7 Convergence Detection and Early Stopping

For efficiency, the solver implements multi-criteria convergence detection:

**KKT-Cone Certificate (v0.6.0, the authoritative optimality test):**
At a constrained optimum, $-\nabla f(x^\star)$ lies in the cone generated by the
active outward facet normals with nonnegative multipliers. The solver measures
the distance to that condition directly, over *all* unit-normalized facets
$\hat{\mathbf{n}}_i$ with signed slacks $s_i$ (no active-set window), by one
augmented nonnegative least-squares fit:

```math
\hat\mu \in \arg\min_{\mu \ge 0}
\left\| \begin{bmatrix} N^\top \\ |s|^\top / \ell_x \end{bmatrix} \mu
- \begin{bmatrix} -\nabla f(x) \\ 0 \end{bmatrix} \right\|_2,
\qquad \ell_x = \max(1, \|x\|_2)
```

The appended complementarity row makes it expensive for the fit to load a
*slack* facet's normal, which is what removes the need for an active-set
window. The residual
$`r_{\mathrm{kkt}} = \sqrt{\Vert\nabla f + N^\top\hat\mu\Vert_2^2 + (|s|^\top\hat\mu/\ell_x)^2}`$
carries gradient units in both components and is accepted when

```math
r_{\mathrm{kkt}} \le \epsilon_{\mathrm{abs}} + \epsilon_{\mathrm{rel}} \cdot
\max(\|Ax\|_2, \|b\|_2, \|N^\top\hat\mu\|_2)
```

so while the relative term dominates the threshold the decision is
invariant under positive objective rescaling, constraint row order, and row
duplication (the $\epsilon_{\mathrm{abs}}$ floor deliberately takes over at
near-zero gradient scales). It is evaluated host-side on every backend.

*Historical note.* Through v0.5 the optimality test was a per-facet
independent gradient projection,
$`\nabla_{\text{proj}} f = \nabla f - \sum_{j \in \text{active}} \min(0, \nabla f \cdot \mathbf{c}_j / \Vert\mathbf{c}_j\Vert^2) \, \mathbf{c}_j`$
(only components whose removal blocks descent into the facet are subtracted),
compared against an *absolute* tolerance. That quantity is structurally
nonzero at constrained optima whose active normals are correlated (the
independent removals leave an $O(\mu \cos\theta)$ cross-term), and an absolute
threshold cannot fire on problems with a large gradient scale. It survives
only as the `legacy_projected_gradient` compatibility mode and as a
diagnostic.

**Objective Plateau Detection:**
Convergence is also indicated when the objective value stabilizes:

```math
\frac{\max_{i \in W} f(x^{(i)}) - \min_{i \in W} f(x^{(i)})}{\max(|f(x^{(k)})|, 10^{-10})} < \epsilon_{\text{obj}}
```

where $W$ is the trailing window of $w$ iterations: the objective's range over
the window, normalized by the latest value.

**Feasibility Requirement:**
Early stopping only triggers when the solution is feasible (max constraint violation below threshold).

**Patience Counter:**
To avoid premature termination, convergence must be detected for $p$ consecutive checks.

**Neuromorphic Interpretation:**
Convergence detection corresponds to monitoring network equilibrium. When voltage changes stabilize (objective plateau) and the drive on every neuron is balanced by nonnegative synaptic reaction from the active constraint population (the KKT-cone certificate), the network has settled into its energy minimum.

### 7.8 KKT Conditions and Implicit Lagrange Multipliers

The algorithm implicitly satisfies the Karush-Kuhn-Tucker (KKT) conditions for optimality.

**KKT System for QP:**

```math
\begin{aligned}
\mathbf{A}\mathbf{x} + \mathbf{b} + \mathbf{C}^\top \boldsymbol{\lambda} &= \mathbf{0} & \text{(Stationarity)} \\
\mathbf{C}\mathbf{x} + \mathbf{d} &\leq \mathbf{0} & \text{(Primal Feasibility)} \\
\boldsymbol{\lambda} &\geq \mathbf{0} & \text{(Dual Feasibility)} \\
\lambda_i (\mathbf{C}\mathbf{x} + \mathbf{d})_i &= 0 & \text{(Complementary Slackness)}
\end{aligned}
```

**Key Insight: Projection Coefficients ARE Lagrange Multipliers**

When projecting constraint $j$ with violation $g_j = \mathbf{c}_j^\top \mathbf{x} + d_j > 0$:

```math
\mathbf{x}_{\text{new}} = \mathbf{x} - \frac{g_j}{\|\mathbf{c}_j\|^2} \mathbf{c}_j = \mathbf{x} - \lambda_j \mathbf{c}_j
```

The adaptive projection coefficient $\lambda_j = g_j / \Vert\mathbf{c}_j\Vert^2$ **is** the Lagrange multiplier for constraint $j$. At convergence, stationarity requires $\nabla f + \sum_j \lambda_j \mathbf{c}_j = \mathbf{0}$, which is satisfied when gradient descent and projections balance.

**Algorithm ↔ KKT Mapping:**

| Algorithm Step | KKT Condition | How It's Enforced |
|----------------|---------------|-------------------|
| Gradient descent | Stationarity | $\mathbf{x} \leftarrow \mathbf{x} - k_0(\mathbf{A}\mathbf{x} + \mathbf{b})$ drives $\nabla f \to \mathbf{0}$ |
| Adaptive projection | Primal feasibility | Projects onto $\mathbf{C}\mathbf{x} + \mathbf{d} = \mathbf{0}$ |
| $\lambda_j = g_j / \Vert\mathbf{c}_j\Vert^2$ | Dual variable | Computed implicitly during projection |
| Only project when $g_j > 0$ | Complementary slackness | $\lambda_j = 0$ when constraint inactive |
| Implicit bound facets | Box feasibility | Bounds join the same sweep, so $\mathbf{x} \in [\mathbf{l}, \mathbf{u}]$ holds *jointly* with $\mathbf{C}\mathbf{x} + \mathbf{d} \le \mathbf{0}$ (see 7.5) |

**Neuromorphic Interpretation of KKT:**

| KKT Condition | Neuromorphic Analog |
|---------------|---------------------|
| Stationarity | Network equilibrium (no net current flow) |
| Primal feasibility | All neurons within firing bounds |
| Dual feasibility | Inhibitory ($\lambda > 0$) connections only |
| Complementary slackness | Silent neurons for inactive constraints |

The adaptive projection $\lambda_j = g_j / \Vert\mathbf{c}_j\Vert^2$ is analogous to computing the synaptic strength needed to bring a neuron back into its valid firing range.

---

## 8. Beyond Polytopes: Nonlinear and Conic Constraints

Sections 2 to 7 treat a feasible set cut out by halfspaces. Nothing in the
drift-and-spike picture needs that: the drift is gradient descent on $f$,
and a spike is any correction that returns the state to the feasible set.
From v0.7.0 the solver accepts a feasible set

```math
\mathcal{F} = \{x : Cx + d \le 0\} \;\cap\; K_1 \cap \dots \cap K_r,
```

where each $K_q$ is a closed convex set supplied as a *candidate*. In the
neural picture each candidate is one more population of constraint neurons
competing in the same winner-take-all sweep; what changes is the correction
its spike applies.

### 8.1 Two kinds of spike

**Cutters.** For a differentiable convex inequality $g(x) \le 0$, a spike
steps to the supporting halfspace at the current point,

```math
x \leftarrow x - \frac{g(x)}{\|\nabla g(x)\|^2}\,\nabla g(x),
```

which is exactly the row spike of Section 2.3 when $g$ is affine. Convexity
means the linearisation never cuts away feasible points, so repeated cuts
approach $\lbrace g \le 0 \rbrace$ from outside.

**Exact projectors.** When the Euclidean projector $P_K$ has a closed form,
a spike is the reset $x \leftarrow P_K(x)$. The built-in sets are:

* **Ball** $\lbrace \Vert x_I - c\Vert \le r \rbrace$: radial scaling onto the sphere, $x_I \leftarrow c + r\,(x_I - c)/\Vert x_I - c\Vert$, when $\Vert x_I - c\Vert > r$; the identity otherwise.
* **Second-order cone** $\lbrace (t, z) : \Vert z\Vert \le \mu t \rbrace$ (friction cones; $\mu = 1$ is the Lorentz cone).
  A point outside both the cone and its polar maps to

  ```math
  t' = \frac{t + \mu\|z\|}{1 + \mu^2}, \qquad z' = \mu t'\,\frac{z}{\|z\|},
  ```

  and a point in the polar cone ($t + \mu\Vert z\Vert \le 0$) maps to the apex.
* **PSD cone** $\lbrace X \succeq 0 \rbrace$: eigendecompose and clip negative eigenvalues. The state holds
  $\operatorname{svec}(X)$ (off-diagonals scaled by $\sqrt{2}$), so state distance equals Frobenius distance.
* **Spectral-norm ball** $\lbrace \Vert X\Vert_2 \le r \rbrace$: SVD and clip singular values at $r$.
* **Affine subspace** $\lbrace Bx = h \rbrace$: $x \leftarrow x - B^\top (BB^\top)^{-1}(Bx - h)$.

Winners are chosen by distance: a row scores $(c_j^\top x + d_j)/\Vert c_j\Vert$,
a cutter $\max(g,0)/\Vert \nabla g\Vert$, a projector $\Vert P_K(x) - x\Vert$, so every
candidate competes on the same geometric scale.

### 8.2 Why exact projection removes the step-size offset

For convex $f$ and closed convex $\mathcal{F}$, $x^\star$ is optimal if and
only if it is a fixed point of the projected-gradient map

```math
T(x) = P_{\mathcal{F}}\big(x - \alpha \nabla f(x)\big) \quad\text{for any } \alpha > 0 .
```

So if one Euler step followed by the projection sweep implements $T$ with
the **exact** projection onto $\mathcal{F}$, the fixed point is exactly
$x^\star$, whatever $k_0$ is. The greedy sweep of Section 2.3 is exact when
one constraint is active, but at a point where several are active it lands
on a feasible point that is generally not $P_{\mathcal{F}}$ of the input.
That is where the $O(k_0)$ offset reported in the README's *Accuracy and tuning* section comes from.
On the README's Figure 1 problem (50 variables, 30 rows, seven active at the
optimum) the greedy sweep's objective gap sits at $6.7\times10^{-4}$ and does
not move between 4k and 40k iterations. Passing the same rows as one exact
joint projector (Section 8.3) removes the floor: with the default
certificate the run stops at iteration 5051 with a gap of $2.1\times10^{-10}$,
and with `kkt_rel_tol=1e-9` it reaches $3.8\times10^{-10}$ from $x^\star$ at
iteration 9901. The price is an inner Dykstra loop per Euler step
(`benchmarks/05_exact_projection.py` reproduces these numbers).

### 8.3 Intersections: Dykstra's algorithm

Projecting onto each set in turn finds *a* point of $K_1 \cap K_2$
(alternating projections: von Neumann for subspaces, Bregman for general
convex sets), but not the *nearest* one. Dykstra's
algorithm (Boyle and Dykstra, 1986, reference 8) does, by carrying one correction
$p_i$ per set:

```math
\begin{aligned}
& y \leftarrow x, \quad p_i \leftarrow 0 \quad (i = 1,\dots,r) \\
& \text{repeat: for } i = 1,\dots,r: \quad z \leftarrow P_{K_i}(y + p_i),\;\; p_i \leftarrow y + p_i - z,\;\; y \leftarrow z,
\end{aligned}
```

and $y \to P_{K_1 \cap \dots \cap K_r}(x)$. `dykstra_projector` packages this
loop as one projector candidate, restarted from $p_i = 0$ on every call. Use
it whenever several sets can be active together. Left as separate
candidates, the winner-take-all sweep alternates between them and, with the
drift in between, can stall away from the optimum: on the friction-cone
grasp of `examples/example8_friction_cone_grasp.py` the separate form stops
at its iteration cap with a relative KKT defect of $5\times10^{-2}$, while
the Dykstra form certifies in 201 iterations, within $4\times10^{-13}$ of a
Newton-polished reference.

### 8.4 Certification

The KKT certificate of Section 7.7 extends by adding the normal cone of each
candidate to the facet normals: the unit normal for a cutter or a smooth
boundary point, and the full polar cone at nonsmooth points (a cone apex,
tied singular values). That fit stays in gradient units.

When $f$ is $\mu$-strongly convex, every candidate carries an exact
certificate projector (any Dykstra wrapper, the PSD cone, the spectral ball
or cutter) on disjoint coordinates, and rows and bounds are strictly slack,
the certificate uses state units instead. For $0 < \alpha \le 1/L$, $T$ is a
contraction with factor at most $1 - \alpha\mu$, so

```math
\|x - x^\star\| \;\le\; \frac{\|x - T(x)\|}{\alpha\,\mu},
```

which bounds the error directly. A bare ball or second-order cone keeps the
gradient-unit fit; wrapping it in `dykstra_projector` qualifies it. A
Dykstra projector is trusted to its inner tolerance (default $10^{-12}$,
relative to $\max(1, \Vert x\Vert)$); the bound does not account for that
error.

### 8.5 The neural reading

Nothing in the event structure changes. A cone or ball candidate is a
population whose spike resets the state onto a curved wall instead of a
flat one, and the raster still shows which constraints are recruited and
released as the network searches for the active set. Inside a Dykstra
candidate each member projection is recorded as its own event, so the
raster of a jointly projected constraint family stays readable.

---

## 9. Conclusions and Future Directions

### 9.1 Summary

We have developed a computationally efficient algorithm for real-time constrained optimization inspired by spiking neural network dynamics. The method alternates between gradient descent on a quadratic or linear objective and discrete projections to enforce inequality constraints. The simplicity of the computational primitives makes the algorithm suitable for embedded implementation, achieving sub-millisecond solve times for moderate-sized problems.

The receding horizon control framework enables application to dynamic systems, with warm starting from previous solutions providing rapid convergence. We demonstrated the approach on robotic manipulator velocity control, where the method computes optimal joint velocities satisfying end-effector constraints at kilohertz rates.

### 9.2 Advantages

**Computational simplicity:** The polyhedral sweep needs only matrix-vector products, comparisons and rank-one row updates; the conic sets of Section 8 add an eigendecomposition (PSD), an SVD (spectral ball) or a small factorisation (affine subspaces)

**Real-time suitability:** Predictable computational cost, easily implemented on embedded processors

**Warm-start efficiency:** Excellent performance when solving sequences of similar problems

**Interpretability:** Direct connection to physical/neural dynamics aids understanding and debugging

### 9.3 Limitations

**Approximate solutions:** The method finds solutions within a neighborhood of the optimum, with error depending on discretization parameters

**Parameter tuning:** Step sizes require empirical tuning for each problem class

**Convexity requirement:** The convergence analysis assumes convex objectives and convex constraint sets (polyhedral, or conic and other convex sets as in Section 8)

**Velocity-level control limitations:** For the manipulator application, operating in velocity space leads to position drift over long horizons

### 9.4 Future Research Directions

Several extensions merit investigation:

**Per-iteration adaptive $k_0$:** The current implementation computes $k_0$ once from the Hessian eigenvalue. Per-iteration methods like Barzilai-Borwein could further accelerate convergence.

**Equality constraints:** Equalities can be written as pairs of inequalities, or (since v0.7.0) projected directly with an affine-subspace projector; see Section 8.

**Nonconvex extensions:** Exploring whether the projection-based approach extends to nonconvex problems, possibly guaranteeing local optimality.

**Learning-based tuning:** Using machine learning to map problem features to optimal algorithm parameters.

**Hardware acceleration:** Implementation on GPUs or custom neuromorphic hardware for massive parallelization.

**Theoretical analysis:** Rigorous convergence rates and approximation error bounds as functions of problem parameters.

---

## Appendix A: MATLAB Implementation

### Core Solver Function

```matlab
function [t, X] = snn_solver(A, b, C, d, t_end, x0, k0, k1)
    % Solves: min x^TAx/2 + b^Tx, subject to Cx + d <= 0
    % 
    % Inputs:
    %   A: Hessian matrix (n x n)
    %   b: Linear cost vector (n x 1)
    %   C: Constraint matrix (m x n)
    %   d: Constraint offset (m x 1)
    %   t_end: Simulation end time
    %   x0: Initial guess (n x 1)
    %   k0: Gradient descent step size
    %   k1: Projection step size
    %
    % Outputs:
    %   t: Time vector
    %   X: State trajectory (length(t) x n)
    
    t_store = cell(1);
    x_store = cell(1);
    
    t_store{1} = 0;
    x_store{1} = x0.';
    
    tspan = [0, t_end];
    
    % Event detection: stop when constraint violated
    function [value, isterminal, direction] = myEvents(~, x)
        y = C*x + d;
        value = all(y <= 0);  % True while feasible
        isterminal = 1;       % Stop integration
        direction = 0;        % Detect any crossing
    end
    
    % Gradient descent dynamics
    function dotx = myode(~, x)
        fgrad = A*x + b;
        dotx = -k0*fgrad;
    end
    
    options = odeset('Events', @myEvents, 'MaxStep', 0.1);
    idx = 2;
    
    while tspan(1) < tspan(2)
        % Phase 1: Project back into feasible region
        while true
            y = C*x0 + d;
            if any(y > 0)
                % Apply projection for violated constraints
                x0 = x0 - k1*C'*(y > 0);
            else
                break
            end
        end
        
        % Phase 2: Gradient descent until constraint hit
        [t_, X_] = ode45(@myode, tspan, x0, options);
        tspan(1) = t_(end);
        x0 = X_(end, :)';
        
        % Store trajectory segment
        t_store{idx} = t_;
        x_store{idx} = X_;
        idx = idx + 1;
    end
    
    % Concatenate all segments
    t = cat(1, t_store{:});
    X = cat(1, x_store{:});
end
```

### Example Usage

```matlab
% Simple 2D problem: minimize ||x||^2 subject to x1 + 2*x2 <= 1
n = 2;
A = eye(n);
b = zeros(n, 1);
C = [1, 2];
d = -1;
x0 = [2; 2];

% Algorithm parameters
t_end = 100;
k0 = 0.05;
k1 = 0.05;

% Solve
[t, X] = snn_solver(A, b, C, d, t_end, x0, k0, k1);

% Plot trajectory
figure;
plot(X(:,1), X(:,2));
xlabel('x_1'); ylabel('x_2');
title('Optimization Trajectory');
```

---

## References

1. Mancoo, A., Keemink, S. W., & Machens, C. K. (2020). Understanding spiking networks through convex optimization. *Advances in Neural Information Processing Systems*, 33, 1-12.

2. Boyd, S., & Vandenberghe, L. (2004). *Convex optimization*. Cambridge University Press.

3. Nocedal, J., & Wright, S. (2006). *Numerical optimization* (2nd ed.). Springer.

4. Boerlin, M., Machens, C. K., & Denève, S. (2013). Predictive coding of dynamical variables in balanced spiking networks. *PLoS Computational Biology*, 9(11), e1003258.

5. Barrett, D. G., Denève, S., & Machens, C. K. (2013). Firing rate predictions in optimal balanced networks. *Advances in Neural Information Processing Systems*, 26, 1538-1546.

6. Eliasmith, C., & Anderson, C. H. (2004). *Neural engineering: Computation, representation, and dynamics in neurobiological systems*. MIT Press.

7. Lynch, K. M., & Park, F. C. (2017). *Modern robotics: Mechanics, planning, and control*. Cambridge University Press.

8. Boyle, J. P., & Dykstra, R. L. (1986). A method for finding projections onto the intersection of convex sets in Hilbert spaces. In *Advances in Order Restricted Statistical Inference*, Lecture Notes in Statistics 37, 28-47. Springer.