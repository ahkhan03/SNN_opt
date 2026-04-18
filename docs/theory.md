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

$$\dot{V}_i = -\lambda V_i + I_i(t)$$

where $V_i$ is the membrane voltage, $\lambda$ is the leak rate, and $I_i(t)$ represents input currents. When $V_i$ reaches a threshold $T_i$, the neuron fires a spike and the voltage resets.

For a network of $N$ neurons with recurrent connectivity, the voltage dynamics become:

$$\dot{\mathbf{V}} = -\lambda\mathbf{V} + \mathbf{F}\mathbf{c}(t) + \mathbf{\Omega}\mathbf{s}(t) + \mathbf{I}_{bg}$$

where $\mathbf{V} \in \mathbb{R}^N$ are voltages, $\mathbf{F}$ encodes feedforward weights, $\mathbf{c}(t)$ are inputs, $\mathbf{\Omega}$ represents recurrent connectivity, and $\mathbf{s}(t)$ are spike trains modeled as sums of delta functions.

The key insight connecting SNNs to optimization is that voltage thresholds naturally correspond to inequality constraints. The condition $\mathbf{V} \leq \mathbf{T}$ defines a feasible region in state space, and spikes act as projection operators that keep the system within this region while descending an objective function.

---

## 2. Mathematical Formulation

### 2.1 The Constrained Optimization Problem

We consider convex optimization problems with linear inequality constraints:

$$\begin{aligned}
\min_{\mathbf{y}} \quad & E(\mathbf{y}) = \frac{\lambda}{2}\mathbf{y}^\top\mathbf{y} + \mathbf{b}^\top\mathbf{y} \\
\text{subject to} \quad & \mathbf{C}\mathbf{y} + \mathbf{d} \leq \mathbf{0}
\end{aligned}$$

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the optimization variable
- $\lambda \geq 0$ determines whether we have a quadratic program ($\lambda > 0$) or linear program ($\lambda = 0$)
- $\mathbf{b} \in \mathbb{R}^n$ provides a linear cost term
- $\mathbf{C} \in \mathbb{R}^{m \times n}$ defines the constraint geometry with $m$ constraints
- $\mathbf{d} \in \mathbb{R}^m$ specifies constraint offsets

The constraint $\mathbf{C}\mathbf{y} + \mathbf{d} \leq \mathbf{0}$ defines the feasible region as the intersection of $m$ half-spaces. Each row $\mathbf{C}_i$ of the constraint matrix defines a hyperplane, and the $i$-th constraint requires that $\mathbf{y}$ lies on one side of this hyperplane.

### 2.2 Geometric Interpretation

The $i$-th inequality constraint can be written as:

$$\mathbf{C}_i^\top \mathbf{y} \leq -d_i$$

This defines a closed half-space in $\mathbb{R}^n$. The boundary of the feasible region consists of points where at least one constraint is active (holds with equality). At the optimal solution $\mathbf{y}^*$, a subset of constraints will be active, defining the optimal face of the feasible polytope.

The gradient of the objective function is:

$$\nabla E(\mathbf{y}) = \lambda\mathbf{y} + \mathbf{b}$$

For a quadratic program ($\lambda > 0$), this gradient points away from the origin with magnitude proportional to $\|\mathbf{y}\|$, creating a "pull" toward zero. For a linear program ($\lambda = 0$), the gradient is constant at $\mathbf{b}$.

### 2.3 Gradient Descent with Boundary Projections

The unconstrained gradient descent dynamics are:

$$\dot{\mathbf{y}} = -k_0 \nabla E(\mathbf{y}) = -k_0(\lambda\mathbf{y} + \mathbf{b})$$

where $k_0 > 0$ is the step size parameter. This would minimize $E(\mathbf{y})$ in the absence of constraints, but may leave the feasible region.

To enforce constraints, we augment the dynamics with projection events. When the $i$-th constraint is violated (i.e., $g_i(\mathbf{y}) = \mathbf{C}_i^\top \mathbf{y} + d_i > 0$), we project back onto the constraint boundary.

#### Fixed Step Projection (Original Method)

The original formulation uses a fixed step size:

$$\mathbf{y} \leftarrow \mathbf{y} - k_1 \mathbf{C}_i$$

where $k_1 > 0$ controls the projection magnitude. This requires tuning $k_1$ and may need multiple iterations to reach the boundary.

#### Adaptive Projection (Improved Method)

A more efficient approach computes the exact step to reach the constraint boundary. For a violated constraint $g_j(\mathbf{y}) = \mathbf{c}_j^\top \mathbf{y} + d_j > 0$, the exact orthogonal projection onto the constraint hyperplane is:

$$\mathbf{y} \leftarrow \mathbf{y} - \frac{g_j(\mathbf{y})}{\|\mathbf{c}_j\|^2} \mathbf{c}_j$$

**Derivation:** We seek the point $\mathbf{y}'$ on the hyperplane $\mathbf{c}_j^\top \mathbf{y}' + d_j = 0$ that is closest to $\mathbf{y}$. The projection moves along the normal direction $\mathbf{c}_j$:

$$\mathbf{y}' = \mathbf{y} - \alpha \mathbf{c}_j$$

Substituting into the constraint equation:
$$\mathbf{c}_j^\top(\mathbf{y} - \alpha \mathbf{c}_j) + d_j = 0$$
$$\mathbf{c}_j^\top \mathbf{y} + d_j = \alpha \|\mathbf{c}_j\|^2$$
$$\alpha = \frac{g_j(\mathbf{y})}{\|\mathbf{c}_j\|^2}$$

This adaptive step eliminates $k_1$ as a hyperparameter and projects exactly onto the boundary in one step per constraint.

**Neuromorphic Interpretation:** The adaptive projection is equally neuromorphic—it corresponds to a neuron that resets its membrane potential exactly to the threshold after firing, rather than decaying by a fixed amount. Both are valid integrate-and-fire models.

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
$$\mathbf{y} \leftarrow \mathbf{y} - k_1 \mathbf{C}^\top \mathbb{1}_{\mathbf{g}(\mathbf{y}) > 0}$$

where $\mathbb{1}_{\mathbf{g}(\mathbf{y}) > 0}$ is an indicator vector for violated constraints.

**Adaptive Step (improved):**
Project onto the most violated constraint with exact step:
$$j = \arg\max_i g_i(\mathbf{y})$$
$$\mathbf{y} \leftarrow \mathbf{y} - \frac{g_j(\mathbf{y})}{\|\mathbf{c}_j\|^2} \mathbf{c}_j$$

Repeat until all constraints satisfied.

The adaptive method eliminates $k_1$ as a hyperparameter and converges faster by computing exact projections. The algorithm converges to a neighborhood of the optimal solution $\mathbf{y}^*$ with error bounded by $k_0$ and the constraint tolerance.

### 2.5 Convergence Properties

**Theorem 1 (Informal):** For a convex quadratic program with $\lambda > 0$, proper choice of step sizes $k_0$ and $k_1$ ensures that the iterates remain in the feasible region and converge to a bounded neighborhood of the optimal solution.

The proof sketch relies on several observations:

1. The gradient descent phase decreases the objective function: $\frac{d}{dt}E(\mathbf{y}) = -k_0 \|\nabla E(\mathbf{y})\|^2 \leq 0$

2. Projections maintain feasibility: after applying corrections, $\mathbf{C}\mathbf{y} + \mathbf{d} \leq \mathbf{0}$

3. The objective function is bounded below on the feasible set (by convexity and compactness arguments)

4. The discrete jumps introduce bounded error: $\|\mathbf{y} - \mathbf{y}^*\| = O(k_1)$

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
        j ← argmax(g)                  # Most violated constraint
        if g[j] ≤ tol: break           # All satisfied
        
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

$$\mathbf{y} \leftarrow \mathbf{y} - k_1 \mathbf{C}^\top \mathbb{1}_{\mathbf{g}(\mathbf{y}) > 0}$$

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
- Initial distance to the solution: $\|\mathbf{y}_0 - \mathbf{y}^*\|$
- Problem conditioning: condition number of $\mathbf{A}$
- Step size $k_0$: larger steps mean fewer iterations but risk instability

For well-conditioned problems, convergence is typically linear with rate determined by $k_0 \lambda_{\min}(A)$ where $\lambda_{\min}$ is the smallest eigenvalue.

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

$$\begin{aligned}
\min_{\mathbf{u}} \quad & \frac{1}{2}\mathbf{u}^\top \mathbf{A} \mathbf{u} + \mathbf{b}^\top \mathbf{u} \\
\text{subject to} \quad & \mathbf{C}(t_k, \mathbf{x}_k) \mathbf{u} + \mathbf{d}(t_k, \mathbf{x}_k) \leq \mathbf{0}
\end{aligned}$$

where the constraint matrices $\mathbf{C}$ and $\mathbf{d}$ may depend on the current time and state. Critically, these are **held constant during each optimization solve**, avoiding the "chasing a moving target" problem.

### 5.3 Warm Starting

A key advantage for control applications is **warm starting**: we initialize each solve with the solution from the previous timestep:

$$\mathbf{u}_0^{(k)} = \mathbf{u}^{*(k-1)}$$

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

$$\dot{\mathbf{r}} = \mathbf{J}(\boldsymbol{\theta}) \mathbf{u}$$

where $\mathbf{J}(\boldsymbol{\theta}) \in \mathbb{R}^{3 \times n}$ is the Jacobian matrix, computed from the manipulator's forward kinematics.

### 6.2 Optimization Formulation

At each timestep, we solve:

$$\begin{aligned}
\min_{\mathbf{u}} \quad & \frac{1}{2}\mathbf{u}^\top \mathbf{u} \\
\text{subject to} \quad & \|\mathbf{J}(\boldsymbol{\theta}) \mathbf{u} - \dot{\mathbf{r}}_d\| \leq \delta
\end{aligned}$$

where $\delta > 0$ is a tolerance on velocity tracking error. This objective minimizes control effort (encouraging smooth motions) while approximately tracking the desired velocity.

### 6.3 Constraint Reformulation

The constraint $\|\mathbf{J}\mathbf{u} - \dot{\mathbf{r}}_d\| \leq \delta$ is equivalent to:

$$-\delta \leq [\mathbf{J}\mathbf{u} - \dot{\mathbf{r}}_d]_i \leq \delta, \quad i = 1,2,3$$

This can be written as linear inequalities:

$$\begin{aligned}
\mathbf{J}\mathbf{u} - \dot{\mathbf{r}}_d &\leq \delta \mathbf{1} \\
-\mathbf{J}\mathbf{u} + \dot{\mathbf{r}}_d &\leq \delta \mathbf{1}
\end{aligned}$$

where $\mathbf{1} = [1, 1, 1]^\top$. This gives us $m = 6$ linear constraints in the form required by our algorithm:

$$\mathbf{C} = \begin{bmatrix} \mathbf{J} \\ -\mathbf{J} \end{bmatrix}, \quad \mathbf{d} = \begin{bmatrix} -\dot{\mathbf{r}}_d - \delta\mathbf{1} \\ \dot{\mathbf{r}}_d - \delta\mathbf{1} \end{bmatrix}$$

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

**Accumulation of projection errors:** Many rapid projections may cause drift. Monitor $\|\mathbf{C}\mathbf{y} + \mathbf{d}\|$ to ensure constraint satisfaction.

### 7.2 Real-Time Guarantees

For hard real-time systems, deterministic execution is required:

**Bounded iteration counts:** Set maximum iteration limits for both projection loops and gradient descent steps.

**Fixed-step integration:** Use fixed timestep integrators (e.g., forward Euler, RK4 with fixed steps) rather than adaptive methods.

**Worst-case analysis:** Analyze maximum computation time for worst-case constraint configurations.

### 7.3 Parameter Tuning Guidelines

**Step size $k_0$ (Auto-computed by default):**

The gradient descent step size can be automatically computed from the Hessian's Lipschitz constant:

$$k_0 = \frac{k_0^{\text{scale}}}{\lambda_{\max}(\mathbf{A})}$$

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

$$l_i \leq y_i \leq u_i$$

While these can be expressed as linear inequalities, a more efficient and numerically stable approach is **clipping**:

$$\mathbf{y} \leftarrow \text{clip}(\mathbf{y}, \mathbf{l}, \mathbf{u})$$

applied after each gradient descent and projection step.

**Why clipping is better than projection for box constraints:**

1. **Efficiency:** Single operation per variable vs. iterative projection
2. **Stability:** No oscillation between box constraint boundaries
3. **Decoupling:** Box constraints don't interfere with equality constraint projections

**Neuromorphic interpretation:** Clipping corresponds directly to **neuron saturation**—biological neurons have natural firing rate bounds. A neuron cannot fire at negative rates (lower bound) and has a maximum firing rate due to refractory periods (upper bound). This is more neuromorphic than treating bounds as linear inequality constraints!

**Application to SVM:** For the SVM dual problem with $0 \leq \alpha_i \leq C$:
- Set `lower_bound=0` and `upper_bound=C`
- Only the equality constraint $\mathbf{y}^\top \boldsymbol{\alpha} = 0$ needs projection
- This dramatically improves convergence and stability

### 7.6 Extension to Equality Constraints

The formulation handles inequality constraints naturally. Equality constraints $\mathbf{A}_{eq}\mathbf{y} = \mathbf{b}_{eq}$ can be incorporated as pairs of inequalities:

$$\mathbf{A}_{eq}\mathbf{y} \leq \mathbf{b}_{eq}, \quad -\mathbf{A}_{eq}\mathbf{y} \leq -\mathbf{b}_{eq}$$

**Why this works with adaptive projection:** The adaptive projection formula $\mathbf{y} \leftarrow \mathbf{y} - \frac{g_j}{\|\mathbf{c}_j\|^2}\mathbf{c}_j$ projects exactly onto the constraint boundary. For equality constraints expressed as two opposing inequalities:
- If $\mathbf{a}^\top \mathbf{y} > b$ (positive side): the first inequality is violated, projection moves toward $\mathbf{a}^\top \mathbf{y} = b$
- If $\mathbf{a}^\top \mathbf{y} < b$ (negative side): the second inequality is violated, projection moves toward $\mathbf{a}^\top \mathbf{y} = b$
- Either way, we end up on the equality hyperplane

This approach doubles the number of constraints but works well in practice. For problems with many equality constraints, direct projection onto the equality manifold may be more efficient as a preprocessing step.

### 7.7 Convergence Detection and Early Stopping

For efficiency, the solver implements multi-criteria convergence detection:

**Projected Gradient Norm:**
At the constrained optimum, the gradient projected onto feasible descent directions is zero. For each active constraint $j$ (where $g_j \approx 0$), we project out the component of the gradient in the constraint normal direction:

$$\nabla_{\text{proj}} f = \nabla f - \sum_{j \in \text{active}} \left( \frac{\nabla f \cdot \mathbf{c}_j}{\|\mathbf{c}_j\|^2} \right) \mathbf{c}_j$$

When $\|\nabla_{\text{proj}} f\| < \epsilon$, we are near a KKT point.

**Objective Plateau Detection:**
Convergence is also indicated when the objective value stabilizes:

$$\frac{|f(x^{(k)}) - f(x^{(k-w)})|}{|f(x^{(k-w)})|} < \epsilon_{\text{obj}}$$

over a window of $w$ iterations.

**Feasibility Requirement:**
Early stopping only triggers when the solution is feasible (max constraint violation below threshold).

**Patience Counter:**
To avoid premature termination, convergence must be detected for $p$ consecutive checks.

**Neuromorphic Interpretation:**
Convergence detection corresponds to monitoring network equilibrium. When voltage changes stabilize (objective plateau) and there are no pending spikes (projected gradient zero), the network has settled into its energy minimum.

### 7.8 KKT Conditions and Implicit Lagrange Multipliers

The algorithm implicitly satisfies the Karush-Kuhn-Tucker (KKT) conditions for optimality.

**KKT System for QP:**
$$\begin{aligned}
\mathbf{A}\mathbf{x} + \mathbf{b} + \mathbf{C}^\top \boldsymbol{\lambda} &= \mathbf{0} & \text{(Stationarity)} \\
\mathbf{C}\mathbf{x} + \mathbf{d} &\leq \mathbf{0} & \text{(Primal Feasibility)} \\
\boldsymbol{\lambda} &\geq \mathbf{0} & \text{(Dual Feasibility)} \\
\lambda_i (\mathbf{C}\mathbf{x} + \mathbf{d})_i &= 0 & \text{(Complementary Slackness)}
\end{aligned}$$

**Key Insight: Projection Coefficients ARE Lagrange Multipliers**

When projecting constraint $j$ with violation $g_j = \mathbf{c}_j^\top \mathbf{x} + d_j > 0$:

$$\mathbf{x}_{\text{new}} = \mathbf{x} - \frac{g_j}{\|\mathbf{c}_j\|^2} \mathbf{c}_j = \mathbf{x} - \lambda_j \mathbf{c}_j$$

The adaptive projection coefficient $\lambda_j = g_j / \|\mathbf{c}_j\|^2$ **is** the Lagrange multiplier for constraint $j$. At convergence, stationarity requires $\nabla f + \sum_j \lambda_j \mathbf{c}_j = \mathbf{0}$, which is satisfied when gradient descent and projections balance.

**Algorithm ↔ KKT Mapping:**

| Algorithm Step | KKT Condition | How It's Enforced |
|----------------|---------------|-------------------|
| Gradient descent | Stationarity | $\mathbf{x} \leftarrow \mathbf{x} - k_0(\mathbf{A}\mathbf{x} + \mathbf{b})$ drives $\nabla f \to \mathbf{0}$ |
| Adaptive projection | Primal feasibility | Projects onto $\mathbf{C}\mathbf{x} + \mathbf{d} = \mathbf{0}$ |
| $\lambda_j = g_j / \|\mathbf{c}_j\|^2$ | Dual variable | Computed implicitly during projection |
| Only project when $g_j > 0$ | Complementary slackness | $\lambda_j = 0$ when constraint inactive |
| Box clipping | Box feasibility | $\mathbf{x} \in [\mathbf{l}, \mathbf{u}]$ |

**Neuromorphic Interpretation of KKT:**

| KKT Condition | Neuromorphic Analog |
|---------------|---------------------|
| Stationarity | Network equilibrium (no net current flow) |
| Primal feasibility | All neurons within firing bounds |
| Dual feasibility | Inhibitory ($\lambda > 0$) connections only |
| Complementary slackness | Silent neurons for inactive constraints |

The adaptive projection $\lambda_j = g_j / \|\mathbf{c}_j\|^2$ is analogous to computing the synaptic strength needed to bring a neuron back into its valid firing range.

---

## 8. Conclusions and Future Directions

### 8.1 Summary

We have developed a computationally efficient algorithm for real-time constrained optimization inspired by spiking neural network dynamics. The method alternates between gradient descent on a quadratic or linear objective and discrete projections to enforce inequality constraints. The simplicity of the computational primitives makes the algorithm suitable for embedded implementation, achieving sub-millisecond solve times for moderate-sized problems.

The receding horizon control framework enables application to dynamic systems, with warm starting from previous solutions providing rapid convergence. We demonstrated the approach on robotic manipulator velocity control, where the method computes optimal joint velocities satisfying end-effector constraints at kilohertz rates.

### 8.2 Advantages

**Computational simplicity:** Only matrix-vector operations, no matrix factorizations or complex data structures

**Real-time suitability:** Predictable computational cost, easily implemented on embedded processors

**Warm-start efficiency:** Excellent performance when solving sequences of similar problems

**Interpretability:** Direct connection to physical/neural dynamics aids understanding and debugging

### 8.3 Limitations

**Approximate solutions:** The method finds solutions within a neighborhood of the optimum, with error depending on discretization parameters

**Parameter tuning:** Step sizes require empirical tuning for each problem class

**Convexity requirement:** The convergence analysis assumes convex objectives and constraint sets

**Velocity-level control limitations:** For the manipulator application, operating in velocity space leads to position drift over long horizons

### 8.4 Future Research Directions

Several extensions merit investigation:

**Per-iteration adaptive $k_0$:** The current implementation computes $k_0$ once from the Hessian eigenvalue. Per-iteration methods like Barzilai-Borwein could further accelerate convergence.

**Equality constraints:** The current approach converts equality constraints to pairs of inequalities. Direct projection onto equality constraint manifolds may improve efficiency for problems with many equalities.

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