# Mathematical formulation and implementation correspondence

This document describes the **At** (extraction-period) and **By** (cumulative) QUBO encodings used by the BF-DCQO mining-scheduling experiments in this repository. It is based on the four experimental Python scripts in `experiments/fixed/` and `experiments/random/`, rather than on the earlier `00-TwoDimensionalMine.py` demonstration. The scripts build a QUBO, convert it internally into an Ising Hamiltonian and evaluate BF-DCQO with Qiskit Aer. For environment versions and execution instructions, see [`../README.md`](../README.md).

## 1. Mining problem, indexing and economic values

Let `N` denote the number of blocks, `T` the number of production periods and `C_t` the maximum blocks mined in period `t`. Mathematical time indices in this document run from `t=1` to `T`; Python array indices run from `0` to `T-1`. Let `V_i` be the undiscounted economic value of block `i`, and let `d` be the discount rate. The scripts calculate

```math
B_{i,t}=\frac{V_i}{(1+d)^{t-1}},\qquad t=1,\ldots,T,
```

and normalize the whole matrix using

```math
B_{\max}=\max_{i,t}|B_{i,t}|,\qquad \widehat B_{i,t}=\frac{B_{i,t}}{B_{\max}}.
```

If the entire economic matrix is zero, the implementation uses a scale factor of one. Both formulations minimize the negative of normalized discounted revenue plus penalties.

## 2. Geometric precedences

Let `P(i)` be the set of *existing* direct predecessors of block `i`. For 2D coordinates `(x,y)`, **larger `y` denotes a higher layer** in these scripts. Predecessor candidates lie at `(x-1,y+1)`, `(x,y+1)` and `(x+1,y+1)`. Only candidate coordinates included in the block model are retained. Thus a block may have zero, one, two or three predecessors; **the code does not introduce a special penalty forbidding edge blocks with fewer than three predecessors**. Blocks at the highest modeled layer have no predecessors. The 3D branch similarly searches a 3-by-3 neighborhood in the immediately overlying layer (smaller `z`).

Mining a block is permitted only after, or in the same period as, all its existing direct predecessors. Each distinct precedence relationship is penalized once per relevant term as specified below.

## 3. At formulation: extraction-period variables

The variable `x_{i,t}` indicates that block `i` is mined **in period `t`**, not that it has already been mined by that period:

```math
x_{i,t}\in\{0,1\}.
```

The normalized QUBO objective implemented by `build_qubo` in the At scripts is

```math
F_{\mathrm{At}}(x,s) =
-\sum_{i=1}^{N}\sum_{t=1}^{T}\widehat B_{i,t}x_{i,t}
+\beta\sum_{i=1}^{N}\sum_{1\le t<\tau\le T}x_{i,t}x_{i,\tau}
+\gamma\sum_i\sum_{k\in P(i)}\sum_{t=1}^{T}
 x_{i,t}\left(1-\sum_{\tau=1}^{t}x_{k,\tau}\right)
+\alpha\sum_{t=1}^{T}\left(\sum_i x_{i,t}+s_t-C_t\right)^2.
```

The second term discourages extracting the same block in more than one period: every pair of distinct periods is coupled. The third term rewards no unmined successor/predecessor violation when the predecessor has been extracted at or before `t`; its construction uses a linear `+gamma` on the child variable and a quadratic `-gamma` for each admissible preceding period. The capacity penalty counts extraction-period variables within the same period.

The code also enforces a global penalty hierarchy for the At single-extraction term. Let `d_k` denote the number of direct successors of block `k` and `d_max` the maximum such degree. With `b = max |widehat B|`, it chooses

```math
\alpha=c_{\mathrm{capacity}}b,\quad
\gamma=c_{\mathrm{precedence}}b,\quad
\beta=\max\left(c_{\mathrm{validity}}b,\ d_{\max}\gamma+\varepsilon\right),
```

with `epsilon = 1e-6` by default. In the experiment entry points checked here, the three user-supplied factors are `1.5` (`c_capacity`, `c_validity`, `c_precedence`). Accordingly, **beta is not necessarily 1.5** after the degree-based adjustment. This hierarchy is specific to At.

## 4. By formulation: cumulative variables

The variable `y_{i,t}` (called `x` internally in parts of the By Python implementation) equals one if block `i` **has been extracted by the end of period `t`**. The initial state is defined by `y_{i,0}=0` and is not an optimization variable. Mining during period `t` is represented by `y_{i,t}-y_{i,t-1}`.

The implemented By QUBO is

```math
F_{\mathrm{By}}(y,s) =
-\sum_i\sum_{t=1}^{T}\widehat B_{i,t}\left(y_{i,t}-y_{i,t-1}\right)
+\mu\sum_i\sum_{t=1}^{T-1}y_{i,t}(1-y_{i,t+1})
+\gamma\sum_i\sum_{k\in P(i)}\sum_{t=1}^{T}
 y_{i,t}(1-y_{k,t})
+\alpha\sum_{t=1}^{T}\left[\sum_i\left(y_{i,t}-y_{i,t-1}\right)+s_t-C_t\right]^2.
```

The first term is implemented in telescoped, *linear* form:

```math
\sum_i\left[\sum_{t=1}^{T-1}(\widehat B_{i,t+1}-\widehat B_{i,t})y_{i,t}-\widehat B_{i,T}y_{i,T}\right].
```

The second term prevents a mined block from reverting to unmined status. Precedence couples a successor and its predecessor **in the same period only**. Capacity couples neighboring cumulative time layers because extraction is a time difference.

The By scripts use **global penalties without a degree-based hierarchy**:

```math
\alpha=c_{\mathrm{capacity}}b,\qquad
\mu=c_{\mathrm{monotonicity}}b,\qquad
\gamma=c_{\mathrm{precedence}}b.
```

The main entry points of the checked fixed-case and random-instance scripts supply `1.5` for all three factors, although some helper functions declare other *default argument values*. The values passed by `main()` are the ones that determine the documented experiments.

## 5. Capacity slack variables and binary layout

The same `bounded_slack_weights(C_t)` procedure is used in both formulations. It constructs nonnegative integer weights `w_{t,k}` whose sum equals `C_t`. Slack is encoded as

```math
s_t=\sum_k w_{t,k}s_{t,k},\qquad s_{t,k}\in\{0,1\}.
```

Examples from the code: `C=2` gives `[1,1]`, `C=3` gives `[1,2]`, and `C=5` gives `[1,2,2]`. The maximum encoded slack therefore equals the capacity, rather than the unrestricted maximum from an arbitrary binary register.

Within each period the program orders the `N` block variables first, followed by that period's slack bits. The total qubit count is

```math
N_q=NT+\sum_{t=1}^{T}L_t,
```

where `L_t` is the number of slack bits in period `t`. In the fixed case `N=9`, `T=2`, `C_1=C_2=3`; hence there are two slack bits per period and `N_q=9*2+2*2=22`.

## 6. QUBO and Ising representation

Both implementations accumulate a scalar offset and dictionaries of linear and quadratic coefficients:

```math
F(\mathbf q)=c+\sum_u h_u q_u+\sum_{u\lt v}J_{uv}q_uq_v,\qquad q_u\in\{0,1\}.
```

The common helper `add_affine_square` expands each weighted squared penalty analytically. For an affine expression `a_0 + sum_i a_i q_i`, it adds constant `lambda*a_0^2`, linear coefficients `lambda*(a_i^2+2*a_0*a_i)`, and pair coefficients `2*lambda*a_i*a_j`. This uses `q_i^2=q_i` for binary variables.

The BF-DCQO solver converts QUBO variables internally using `q_u=(1-Z_u)/2`. In its `SparsePauliOp` construction, for binary coefficients `h_u` and `J_uv`, the Ising coefficients are

```math
\widetilde J_{uv}=J_{uv}/4,\qquad
\widetilde h_u=-h_u/2-\frac14\sum_{v\ne u}J_{uv},\qquad
E_0=c+\frac12\sum_u h_u+\frac14\sum_{u\lt v}J_{uv}.
```

Thus the problem Hamiltonian is

```math
H_p=E_0 I+\sum_u\widetilde h_u Z_u+\sum_{u\lt v}\widetilde J_{uv}Z_uZ_v.
```

The QUBO should **not** be converted to Ising a second time before passing it to this solver.

## 7. Verified fixed-case configuration and experimental distinctions

The fixed scripts define nine blocks with undiscounted values `[20, -45, 60, -30, 25, 120, 320, 220, 500]`, coordinates `[(1,3),(2,3),(3,3),(4,3),(5,3),(2,2),(3,2),(4,2),(3,1)]`, two periods, capacity `[3,3]` and discount rate `d=0.08`. They run independent BF-DCQO budgets `R=1,...,30`, use Qiskit Aer on CPU, set `eta=0.5`, and request `1,000,000` shots per configured run. The fixed experiments report greedy-refined best-shot analyses over sweep counts `s=0,...,10`.

The random-instance scripts instead aggregate results across 100 instances using **raw terminal** optimal-sampling probabilities without greedy refinement (`s=0`). Their summary plots include median and interquartile range (IQR) and paired At/By comparisons. Raw terminal probabilities should not be confused with greedy-refined results.

## 8. Theory-to-code index

| Topic | At implementation | By implementation |
|---|---|---|
| Discounting and normalization | `discounted_values`, `normalize_values` | Same functions |
| Precedence generation | `build_geometric_precedence` | Same function |
| Variable indexing / slack weights | `build_indices`, `bounded_slack_weights` | Same functions |
| Penalty initialization | `initialize_penalties` (includes successor-degree bound) | `initialize_penalties` (no hierarchy) |
| Objective and constraints | `build_qubo` | `build_qubo` |
| QUBO-to-Ising conversion | `build_spin_glass_hamiltonian` | `build_spin_glass_hamiltonian` |

The referenced scripts are `experiments/fixed/At_22Q_FixedCase.py`, `experiments/fixed/By_22Q_FixedCase.py`, `experiments/random/At_22Q_RandomInstances_s0.py`, and `experiments/random/By_22Q_RandomInstances_s0.py`.

**Scope note:** This document records the expressions and entry-point parameters actually found in the checked scripts. It does not claim that the chosen penalty factors are mathematically sufficient to guarantee feasible global optima in every possible mining instance; those claims require separate proof or validation. Scientific environment versions are maintained in the repository README and requirements file.
