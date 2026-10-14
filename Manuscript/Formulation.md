# QUBO formulations for open-pit mine scheduling

This document describes the **At** (period-specific) and **By** (cumulative) QUBO formulations implemented in the experiment scripts in this repository. It supersedes the earlier single-formulation documentation. Mathematical periods are indexed by $t=1,\ldots,T$; Python arrays use zero-based indices.

## 1. Scheduling problem and economic values

Let $i=1,\ldots,N$ index mine blocks, $t=1,\ldots,T$ index production periods, $V_i$ denote undiscounted economic block value, $d\geq0$ the discount rate, $C_t\geq0$ the integer extraction capacity, and $\mathcal P_i$ the set of direct predecessors of block $i$.

The discounted value is

$$B_{i,t}=\frac{V_i}{(1+d)^{t-1}}.$$

For QUBO construction, the scripts normalize these values using $S=\max_{i,t}|B_{i,t}|$ (using $S=1$ when all values vanish) and $b_{i,t}=B_{i,t}/S$. Minimizing negative normalized revenue is equivalent to maximizing discounted revenue in the feasible region.

### Geometric predecessors

In **2D**, coordinates $(x,y)$ have **larger $y$ for higher layers** in the current scripts. Candidate immediate predecessors are $(x-1,y+1)$, $(x,y+1)$, and $(x+1,y+1)$. In **3D**, candidates are the nine positions $(x+\Delta x,y+\Delta y,z-1)$ with $\Delta x,\Delta y\in\{-1,0,1\}$; the scripts treat the smallest $z$ as the uppermost layer. Only candidates actually present in the input block set are included. Missing neighbors are **not** automatically replaced by a prohibition on mining edge blocks.

## 2. Shared capacity encoding

For each period, introduce binary slack variables $z_{t,k}$ and positive integer weights $w_{t,k}$:

$$s_t=\sum_k w_{t,k}z_{t,k},\qquad 0\leq s_t\leq C_t.$$

The implementation constructs weights incrementally as $1,2,4,\ldots$, truncating the final weight so that their sum is **exactly $C_t$**. For example, $C_t=2$ gives $[1,1]$; $C_t=5$ gives $[1,2,2]$. The slack can encode every integer from zero through $C_t$; representations need not be unique. For $C_t=0$, no slack bits are needed.

The penalty for a period's capacity is $\alpha(\text{extracted}_t+s_t-C_t)^2$. It is zero for a feasible extraction count with suitable slack and positive for excess extraction.

## 3. At formulation: extraction in a specific period

Define $a_{i,t}=1$ if block $i$ is mined **exactly in period $t$**, and zero otherwise. The implemented objective is

$$\begin{aligned}
F_{\mathrm{At}}={}&-\sum_{i,t}b_{i,t}a_{i,t}\\
&+\beta\sum_i\sum_{1\leq t<\tau\leq T}a_{i,t}a_{i,\tau}\\
&+\gamma\sum_i\sum_{j\in\mathcal P_i}\sum_{t=1}^T a_{i,t}\left(1-\sum_{\tau=1}^{t}a_{j,\tau}\right)\\
&+\alpha\sum_{t=1}^T\left(\sum_i a_{i,t}+s_t-C_t\right)^2.
\end{aligned}$$

The second term discourages mining a block more than once. The third term penalizes extracting a block before (or without) its predecessor, provided the one-time-extraction constraint is respected. The QUBO is quadratic: the apparent time sum is a linear combination of binary variables multiplied by $a_{i,t}$.

**Penalty initialization.** With $E=\max_{i,t}|b_{i,t}|$, the implementation sets $\alpha=c_{\mathrm{capacity}}E$, $\gamma=c_{\mathrm{precedence}}E$, and

$$\beta=\max\{c_{\mathrm{validity}}E,\ d_{\max}\gamma+\varepsilon\},$$

where $d_{\max}$ is the maximum number of direct **successors** for which a block acts as predecessor, and $\varepsilon>0$. Thus $\beta>d_{\max}\gamma$. These are the **implemented initialization rules**, not a universal proof that all possible global minimizers are feasible. Check the particular experiment script for its actual factor values.

## 4. By formulation: extracted by the end of a period

Define $y_{i,t}=1$ if block $i$ has been mined **by the end of period $t$**. Set $y_{i,0}=0$ as a fixed initial condition, not a decision variable. The extraction indicator for period $t$ is $y_{i,t}-y_{i,t-1}$. The implemented QUBO is

$$\begin{aligned}
F_{\mathrm{By}}={}&-\sum_{i=1}^N\sum_{t=1}^T b_{i,t}(y_{i,t}-y_{i,t-1})\\
&+\mu\sum_{i=1}^N\sum_{t=1}^{T-1}y_{i,t}(1-y_{i,t+1})\\
&+\gamma\sum_i\sum_{j\in\mathcal P_i}\sum_{t=1}^T y_{i,t}(1-y_{j,t})\\
&+\alpha\sum_{t=1}^T\left(\sum_i(y_{i,t}-y_{i,t-1})+s_t-C_t\right)^2.
\end{aligned}$$

The revenue term telescopes to a linear QUBO contribution:

$$-\sum_i\left[\sum_{t=1}^{T-1}(b_{i,t}-b_{i,t+1})y_{i,t}+b_{i,T}y_{i,T}\right].$$

The monotonicity term penalizes a block reappearing after extraction. The precedence term enforces that a mined block's direct predecessors have been mined by the same period. The capacity term counts **new** extraction events per period, rather than all previously mined blocks.

**Penalty initialization.** The code uses $\alpha=c_{\mathrm{capacity}}E$, $\mu=c_{\mathrm{monotonicity}}E$, and $\gamma=c_{\mathrm{precedence}}E$, with one global coefficient for each constraint family. Unlike At, no degree-based hierarchy is imposed by this initializer.

## 5. QUBO and Ising representation

Both implementations store the energy as

$$F(\mathbf q)=c+\sum_u h_uq_u+\sum_{u<v}J_{uv}q_uq_v,\qquad q_u\in\{0,1\}.$$

Linear and quadratic terms are accumulated in dictionaries and the capacity squares are expanded using $q_u^2=q_u$. Binary variables are ordered period by period, with block decisions followed by that period's slack bits. The QUBO is then converted internally to an Ising Hamiltonian represented by Qiskit's `SparsePauliOp` using $q_u=(1-Z_u)/2$.

## 6. Experimental protocols

### Fixed 22-qubit case

`experiments/fixed/At_22Q_FixedCase.py` and `By_22Q_FixedCase.py` construct the two QUBOs and execute independent BF-DCQO runs for maximum iteration budgets $R$. Each run starts from zero bias and evolves through its own iterations $r=1,\ldots,R$. Raw measurements are used to update the bias. At the end, the best raw bitstring observed **across that run** is greedily refined with $s=0,\ldots,10$ single-bit-flip sweeps. Outputs distinguish raw terminal optimal-sampling probability, within-run raw best/accumulated statistics, and postprocessing results. The greedy-refined best-string metric is **not** the same as the probability of sampling the optimum directly.

### Random 22-qubit instances

`experiments/random/At_22Q_RandomInstances_s0.py` and `By_22Q_RandomInstances_s0.py` evaluate independent runs over 100 random instances, with $R=1,\ldots,30$ and $s=0$ (no greedy refinement). Each $(\text{instance},R)$ run restarts from zero bias. The main probability statistic is the **raw optimal probability at the terminal iteration** of that run. Aggregates include median and interquartile range (IQR), and the plotting script also compares matched instances at $R=30$. The random-instance scripts use Gurobi for classical optimization/reference calculations; running them requires a suitable Gurobi installation and license.

The experiments use Qiskit Aer `statevector` simulation and aggregate counts rather than storing every measured shot. Exact shot counts, seeds, and other settings should be read from the checked-in experiment scripts and reported alongside any published numerical result. Results are stochastic and depend on software environment and configured random seeds.

## 7. Scope and reproducibility

The QUBO is an unconstrained penalized representation: **low QUBO energy does not automatically imply a feasible mine schedule** unless constraint satisfaction is verified. These scripts implement specific heuristic penalty initialization rules; no universal feasibility guarantee is claimed here. Results obtained on classical statevector simulation should not be described as quantum-hardware results.

See [README.md](README.md) for the repository layout, dependencies, and execution order.
