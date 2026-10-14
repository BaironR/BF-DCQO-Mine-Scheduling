from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import ast
import json
import time
from pathlib import Path
from gurobipy import Model, GRB, quicksum
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.circuit.library import PauliEvolutionGate


"""
"At"-formulation QUBO model for mine scheduling.

Current workflow:
1. Compute discounted and normalized economic values.
2. Generate 2D or 3D geometric precedence relationships.
3. Initialize global penalties and enforce the hierarchy bound from the paper.
4. Build bounded slack variables whose maximum is exactly C_t.
5. Construct the "At" QUBO.
6. Build the QUBO dictionary for Solver01.
7. Solver01 internally converts the QUBO to Ising.
8. For each random instance, run independent BF-DCQO experiments with R=1,...,30.
9. Reset the bias to zero at the start of each independent R run.
10. Use s=0, so no greedy refinement is applied.
11. Record RAW statistics at the terminal iteration r=R and export compact results.

In the "At" formulation, x[i,t] = 1 indicates that block i is extracted
exactly in period t.

Independent BF-DCQO runs are executed with Qiskit AerSimulator.
For each maximum R, the solver starts again with zero bias and executes r=1,...,R.
With s=0, greedy refinement is disabled; the main metric is the RAW optimal-solution
probability at the terminal iteration r=R. Results are exported to JSON with a
checkpoint after each (instance, R) pair.
"""

TOL = 1e-10
Linear = Dict[int, float]
Quadratic = Dict[Tuple[int, int], float]
IndexMap = Dict[Tuple[str, int, int], int]
PrecedenceMap = Dict[int, List[int]]


@dataclass
class QUBO:
    constant: float
    linear: Linear
    quadratic: Quadratic
    number_variables: int
    indices: IndexMap
    slack_weights: List[List[int]]

    def energy(self, bits: Sequence[int]) -> float:
        x = np.asarray(bits, dtype=float)
        if x.shape != (self.number_variables,):
            raise ValueError(
                f"Expected {self.number_variables} variables, but received {x.size}."
            )
        value = float(self.constant)
        value += sum(coef * x[i] for i, coef in self.linear.items())
        value += sum(coef * x[i] * x[j] for (i, j), coef in self.quadratic.items())
        return float(value)


@dataclass
class ModelArtifacts:
    discounted_values_raw: np.ndarray
    normalized_values: np.ndarray
    economic_scale: float
    precedence: PrecedenceMap
    alpha: float
    beta: float
    gamma: float
    qubo: QUBO



def add_linear(linear: Linear, i: int, value: float) -> None:
    new_value = linear.get(i, 0.0) + float(value)
    if abs(new_value) <= TOL:
        linear.pop(i, None)
    else:
        linear[i] = new_value


def add_quadratic(quadratic: Quadratic, i: int, j: int, value: float) -> None:
    if i == j:
        raise ValueError("A quadratic term must involve two distinct variables.")
    u, v = sorted((int(i), int(j)))
    new_value = quadratic.get((u, v), 0.0) + float(value)
    if abs(new_value) <= TOL:
        quadratic.pop((u, v), None)
    else:
        quadratic[(u, v)] = new_value


def add_affine_square(
    constant: float,
    linear: Linear,
    quadratic: Quadratic,
    affine_constant: float,
    coefficients: Mapping[int, float],
    penalty: float,
) -> float:
    """Add penalty*(affine_constant + sum_i a_i*x_i)^2 to the QUBO."""
    c = float(affine_constant)
    lam = float(penalty)
    items = sorted(
        (int(i), float(a))
        for i, a in coefficients.items()
        if abs(float(a)) > TOL
    )

    constant += lam * c * c

    # x_i^2 = x_i for binary variables.
    for i, a_i in items:
        add_linear(linear, i, lam * (a_i * a_i + 2.0 * c * a_i))

    for position, (i, a_i) in enumerate(items):
        for j, a_j in items[position + 1:]:
            add_quadratic(quadratic, i, j, 2.0 * lam * a_i * a_j)

    return float(constant)


def discounted_values(
    block_values: Sequence[float],
    periods: int,
    discount_rate: float,
) -> np.ndarray:
    """
    Compute B[t,i] = V[i]/(1+r)^t, for t=0,...,T-1.

    Using one-based mathematical indexing:
        B_{i,tau} = V_i/(1+r)^(tau-1), tau=1,...,T.
    """
    if periods <= 0:
        raise ValueError("The number of periods must be positive.")
    if discount_rate < 0.0:
        raise ValueError("The discount rate must be nonnegative.")

    values = np.asarray(block_values, dtype=float)
    return np.vstack(
        [values / ((1.0 + discount_rate) ** t) for t in range(periods)]
    )


def normalize_values(values_by_period: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize by max |B[t,i]| while preserving signs."""
    matrix = np.asarray(values_by_period, dtype=float)
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if scale <= TOL:
        return matrix.copy(), 1.0
    return matrix / scale, scale


def bounded_slack_weights(capacity: int) -> List[int]:
    """
    Build weights whose maximum representable value is exactly capacity.

    Examples:
        C=2  -> [1, 1]
        C=3  -> [1, 2]
        C=5  -> [1, 2, 2]
        C=20 -> [1, 2, 4, 8, 5]
    """
    if capacity < 0:
        raise ValueError("Capacity cannot be negative.")
    if capacity == 0:
        return []

    weights: List[int] = []
    represented_max = 0
    next_power = 1

    while represented_max < capacity:
        weight = min(next_power, capacity - represented_max)
        weights.append(int(weight))
        represented_max += int(weight)
        next_power *= 2

    return weights


def build_geometric_precedence(
    coordinates: Sequence[Tuple[int, ...]],
) -> PrecedenceMap:
    """
    Generate direct precedence relations from coordinates.

    2D: (x, y), where a larger y denotes a higher layer.

        Each block may have at most three direct predecessors,
        located at:

            (x - 1, y + 1)
            (x,     y + 1)
            (x + 1, y + 1)

        This convention represents a triangular geometry in which
        each lower block is supported by three upper blocks.

    3D: retain the original geometry with a 3x3 neighborhood in the
        immediately higher layer.

    Only positions that actually exist in the model are included.
    Consequently, boundary blocks may have one or
    no predecessors.
    """
    if not coordinates:
        raise ValueError("At least one coordinate is required.")

    dimension = len(coordinates[0])

    if dimension not in (2, 3):
        raise ValueError("Coordinates must be 2D or 3D.")

    if any(len(coord) != dimension for coord in coordinates):
        raise ValueError(
            "All coordinates must have the same dimensionality."
        )

    position_to_block = {
        tuple(coordinate): block
        for block, coordinate in enumerate(coordinates)
    }

    precedence: PrecedenceMap = {}

    if dimension == 2:
        maximum_y = max(coordinate[1] for coordinate in coordinates)

        for block, coordinate in enumerate(coordinates):
            x, y = coordinate

            # Blocks in the top layer have no predecessors.
            if y == maximum_y:
                continue

            # Only three upper supporting blocks.
            candidates = [
                (x - 1, y + 1),
                (x,     y + 1),
                (x + 1, y + 1),
            ]

            predecessors = [
                position_to_block[position]
                for position in candidates
                if position in position_to_block
            ]

            if predecessors:
                precedence[block] = predecessors

    else:
        minimum_depth = min(coordinate[-1] for coordinate in coordinates)

        for block, coordinate in enumerate(coordinates):
            x, y, z = coordinate

            if z == minimum_depth:
                continue

            candidates = [
                (x + dx, y + dy, z - 1)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            ]

            predecessors = [
                position_to_block[position]
                for position in candidates
                if position in position_to_block
            ]

            if predecessors:
                precedence[block] = predecessors

    # Check specific to two-dimensional instances.
    if dimension == 2:
        invalid_blocks = {
            block: predecessors
            for block, predecessors in precedence.items()
            if len(predecessors) > 3
        }

        if invalid_blocks:
            raise RuntimeError(
                "Some blocks were assigned more than three predecessors: "
                f"{invalid_blocks}"
            )

    return precedence


def build_indices(
    number_blocks: int,
    periods: int,
    capacities: Sequence[int],
) -> Tuple[IndexMap, List[List[int]], int]:
    if number_blocks <= 0:
        raise ValueError("The number of blocks must be positive.")
    if periods <= 0:
        raise ValueError("The number of periods must be positive.")
    if len(capacities) != periods:
        raise ValueError("A capacity must be specified for each period.")

    indices: IndexMap = {}
    slack_weights: List[List[int]] = []
    q = 0

    for t in range(periods):
        for i in range(number_blocks):
            indices[("x", i, t)] = q
            q += 1

        weights_t = bounded_slack_weights(int(capacities[t]))
        slack_weights.append(weights_t)
        for k in range(len(weights_t)):
            indices[("s", t, k)] = q
            q += 1

    return indices, slack_weights, q



def successor_degrees(
    precedence: Mapping[int, Sequence[int]],
    number_blocks: int,
) -> Tuple[Dict[int, int], int]:
    """
    Compute d_k for each block k.

    d_k is the number of successor blocks for which block k is a direct predecessor.

    The sufficient local condition for the At formulation is:

        A_val^(at) > d_k * A_prec^(at)

    To use global penalty coefficients, adopt the conservative
    condition corresponding to the maximum degree:

        A_val^(at) > d_max * A_prec^(at),

    where d_max = max_k d_k.
    """
    degrees = {k: 0 for k in range(number_blocks)}

    for _, predecessors in precedence.items():
        for predecessor in predecessors:
            k = int(predecessor)
            if k not in degrees:
                raise ValueError(
                    f"Predecessor {k} is outside the valid block index range."
                )
            degrees[k] += 1

    d_max = max(degrees.values(), default=0)
    return degrees, int(d_max)


def initialize_penalties(
    normalized_values: np.ndarray,
    precedence: Mapping[int, Sequence[int]],
    c_capacity: float = 1.5,
    c_validity: float = 2.0,
    c_precedence: float = 1.5,
    hierarchy_epsilon: float = 1e-6,
) -> Tuple[float, float, float]:
    """
    Initialize three global penalty coefficients for the At formulation:

        alpha = A_cap
        beta  = A_val^(at)
        gamma = A_prec^(at)

    The economic values have already been normalized by
    max |B[t,i]|. This economic scale is used as a common reference.

    If d_k is the number of direct successors for which
    block k acts as a predecessor, then:
        A_val^(at) > d_k * A_prec^(at).

    Enforce:
        beta > d_max * gamma,

    con d_max = max_k d_k.
    """
    B = np.asarray(normalized_values, dtype=float)

    if B.size == 0:
        raise ValueError("The economic matrix cannot be empty.")

    if min(c_capacity, c_validity, c_precedence) <= 0:
        raise ValueError("Penalty factors must be positive.")

    if hierarchy_epsilon <= 0:
        raise ValueError("hierarchy_epsilon must be positive.")

    periods, number_blocks = B.shape
    economic_reference = float(np.max(np.abs(B)))

    # Base global penalty coefficients.
    alpha = float(c_capacity * economic_reference)
    beta_base = float(c_validity * economic_reference)
    gamma = float(c_precedence * economic_reference)

    # d_k and d_max as defined in Sec. 3.4.
    _, d_max = successor_degrees(
        precedence=precedence,
        number_blocks=number_blocks,
    )

    # beta > d_max * gamma.
    hierarchy_bound = float(d_max * gamma)

    beta = max(
        beta_base,
        hierarchy_bound + float(hierarchy_epsilon),
    )

    return alpha, float(beta), gamma


def build_qubo(
    normalized_values: np.ndarray,
    precedence: Mapping[int, Sequence[int]],
    capacities: Sequence[int],
    alpha: float,
    beta: float,
    gamma: float,
) -> QUBO:
    """
    Build the "At" QUBO formulation:

        F_at = F_obj + F_val + F_prec + F_cap

    using a single coefficient per constraint type:

        alpha = A_cap
        beta  = A_val
        gamma = A_prec^(at)
    """
    B = np.asarray(normalized_values, dtype=float)
    periods, number_blocks = B.shape

    capacities_array = np.asarray(capacities, dtype=int)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)

    if capacities_array.shape != (periods,):
        raise ValueError("capacities must have one entry per period.")
    if np.any(capacities_array < 0):
        raise ValueError("Capacities cannot be negative.")
    if min(alpha, beta, gamma) < 0:
        raise ValueError("Penalty coefficients must be nonnegative.")

    indices, slack_weights, number_variables = build_indices(
        number_blocks=number_blocks,
        periods=periods,
        capacities=capacities_array,
    )

    constant = 0.0
    linear: Linear = {}
    quadratic: Quadratic = {}

    # 1. Economic objective.
    for i in range(number_blocks):
        for t in range(periods):
            qx = indices[("x", i, t)]
            add_linear(linear, qx, -B[t, i])

    # 2. Single extraction with global beta.
    for i in range(number_blocks):
        for t in range(periods - 1):
            for tau in range(t + 1, periods):
                q_t = indices[("x", i, t)]
                q_tau = indices[("x", i, tau)]
                add_quadratic(quadratic, q_t, q_tau, beta)

    # 3. Precedence with global gamma.
    for i, predecessors in precedence.items():
        for predecessor in predecessors:
            for t in range(periods):
                child_q = indices[("x", int(i), t)]
                add_linear(linear, child_q, gamma)

                for tau in range(t + 1):
                    predecessor_q = indices[("x", int(predecessor), tau)]
                    add_quadratic(
                        quadratic,
                        child_q,
                        predecessor_q,
                        -gamma,
                    )

    # 4. Capacity with global alpha.
    for t in range(periods):
        coefficients: Dict[int, float] = {}

        for i in range(number_blocks):
            current = indices[("x", i, t)]
            coefficients[current] = 1.0

        for k, weight in enumerate(slack_weights[t]):
            slack_q = indices[("s", t, k)]
            coefficients[slack_q] = float(weight)

        constant = add_affine_square(
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            affine_constant=-float(capacities_array[t]),
            coefficients=coefficients,
            penalty=alpha,
        )

    return QUBO(
        constant=constant,
        linear=linear,
        quadratic=quadratic,
        number_variables=number_variables,
        indices=indices,
        slack_weights=slack_weights,
    )


def build_qubo_problem(qubo: QUBO) -> Dict[str, float]:
    """
    Build the binary QUBO dictionary expected by Solver01.

    Solver01 internally performs the QUBO -> Ising conversion, so
    no prior Ising conversion is performed in the At formulation.
    """
    problem: Dict[str, float] = {
        "()": float(qubo.constant)
    }

    for i in sorted(qubo.linear):
        problem[f"({i},)"] = float(qubo.linear[i])

    for i, j in sorted(qubo.quadratic):
        problem[f"({i}, {j})"] = float(
            qubo.quadratic[(i, j)]
        )

    return problem


def decode_at_solution(
    bits: Sequence[int],
    qubo: QUBO,
    number_blocks: int,
    periods: int,
) -> Dict[str, object]:
    """
    Decode a solution of the At formulation.

    In At:
        x[i,t] = 1
    means block i is extracted exactly in period t.
    """
    bits_array = np.asarray(bits, dtype=int)

    if bits_array.shape != (qubo.number_variables,):
        raise ValueError(
            f"The bitstring has {bits_array.size} variables, "
            f"but the QUBO contains {qubo.number_variables}."
        )

    extracted = np.zeros(
        (number_blocks, periods),
        dtype=int,
    )

    for i in range(number_blocks):
        for t in range(periods):
            extracted[i, t] = bits_array[
                qubo.indices[("x", i, t)]
            ]

    extracted_by_period = {
        t + 1: [
            i + 1
            for i in range(number_blocks)
            if extracted[i, t] == 1
        ]
        for t in range(periods)
    }

    extraction_period_by_block = {
        i + 1: [
            t + 1
            for t in range(periods)
            if extracted[i, t] == 1
        ]
        for i in range(number_blocks)
    }

    slack_values: List[int] = []

    for t in range(periods):
        slack_t = 0
        for k, weight in enumerate(
            qubo.slack_weights[t]
        ):
            slack_bit = bits_array[
                qubo.indices[("s", t, k)]
            ]
            slack_t += int(weight) * int(slack_bit)

        slack_values.append(slack_t)

    return {
        "extracted_state": extracted,
        "extracted_by_period": extracted_by_period,
        "extraction_period_by_block": extraction_period_by_block,
        "slack_values": slack_values,
    }


def calculate_economic_value(
    decoded_solution: Mapping[str, object],
    discounted_values_raw: np.ndarray,
) -> Dict[str, object]:
    """
    Calculate the actual economic value of an At solution.

        NPV = sum_t sum_i B[t,i] * x[i,t]

    Use the original discounted values B_raw, not the normalized values.
    """
    B_raw = np.asarray(
        discounted_values_raw,
        dtype=float,
    )

    extracted = np.asarray(
        decoded_solution["extracted_state"],
        dtype=int,
    )

    number_blocks, periods = extracted.shape
    expected_shape = (periods, number_blocks)

    if B_raw.shape != expected_shape:
        raise ValueError(
            "Incompatible dimensions for computing economic value: "
            f"B_raw has shape {B_raw.shape}, "
            f"but expected {expected_shape}."
        )

    contributions = extracted * B_raw.T

    value_by_period = contributions.sum(axis=0)
    value_by_block = contributions.sum(axis=1)
    total_value = float(value_by_period.sum())

    return {
        "total_value": total_value,
        "value_by_period": value_by_period,
        "value_by_block": value_by_block,
        "contributions": contributions,
    }

# ==========================================================================
# Solver01_IBM_Simulator: BF-DCQO with Qiskit
# ==========================================================================

# --------------------------------------------------------------------------
# Helpers: Pauli labels and bit <-> index conversions (Qiskit ordering)
# --------------------------------------------------------------------------
def _label(paulis, n):
    """Build an n-character Pauli label from a dict {qubit: 'X'|'Y'|'Z'}.
    Position n-1-q corresponds to qubit q (rightmost char = qubit 0)."""
    chars = ["I"] * n
    for q, p in paulis.items():
        chars[n - 1 - q] = p
    return "".join(chars)


def _index_to_bits(idx, n):
    """Bitstring x (numpy int array) of statevector index idx: x[q] = bit q."""
    return np.array([(idx >> q) & 1 for q in range(n)], dtype=int)


def _bits_to_index(bits):
    """Inverse of _index_to_bits (qubit q at bit q)."""
    return int(sum(int(b) << q for q, b in enumerate(bits)))


def _parse_key(key):
    """Parse a QUBO dictionary key such as "()", "(3,)" or "(1, 4)" into a
    tuple of ints using ``ast.literal_eval``."""
    term = ast.literal_eval(key)
    if not isinstance(term, tuple):
        term = (term,)
    return tuple(int(i) for i in term)


# --------------------------------------------------------------------------
# 1. QUBO --> spin-glass (Ising) Hamiltonian as a SparsePauliOp
# --------------------------------------------------------------------------
def build_spin_glass_hamiltonian(X):
    """Build the spin-glass (Ising) problem Hamiltonian H_p from a QUBO dict.

    Uses the mapping x_j = (1 - z_j)/2 with z_j -> sigma_j^z (docs/BF-DCQO.md,
    Sec. 1), giving

        H_p = sum_j h_j Z_j + sum_{j<k} J_{jk} Z_j Z_k + E_0 * I ,
        J_{jk} = q_{jk}/4,
        h_j    = -q_j/2 - (1/4) sum_{k != j} q_{jk},
        E_0    = offset + sum_j q_j/2 + sum_{j<k} q_{jk}/4 .

    Returns
    -------
    Hp : qiskit.quantum_info.SparsePauliOp
        Diagonal Ising Hamiltonian including the constant offset.
    n : int
        Number of qubits / binary variables.
    """
    linear, quad, offset = {}, {}, 0.0

    for key, val in X.items():
        term = _parse_key(key)
        if len(term) == 0:
            offset += float(val)
        elif len(term) == 1:
            linear[term[0]] = linear.get(term[0], 0.0) + float(val)
        elif len(term) == 2:
            j, k = sorted(term)
            quad[(j, k)] = quad.get((j, k), 0.0) + float(val)
        else:
            raise ValueError(f"Only up to quadratic terms are supported, got key {key!r}")

    used = set(linear)
    for (j, k) in quad:
        used.update((j, k))
    if not used:
        raise ValueError("QUBO has no variables (only a constant offset).")
    n = max(used) + 1

    h = np.zeros(n)
    J = {}
    E0 = offset
    for j, q_j in linear.items():
        h[j] += -q_j / 2.0
        E0 += q_j / 2.0
    for (j, k), q_jk in quad.items():
        J[(j, k)] = J.get((j, k), 0.0) + q_jk / 4.0
        h[j] += -q_jk / 4.0
        h[k] += -q_jk / 4.0
        E0 += q_jk / 4.0

    terms = []
    for j in range(n):
        if h[j] != 0.0:
            terms.append((_label({j: "Z"}, n), h[j]))
    for (j, k), J_jk in J.items():
        if J_jk != 0.0:
            terms.append((_label({j: "Z", k: "Z"}, n), J_jk))
    terms.append((_label({}, n), E0))                  # constant offset

    Hp = SparsePauliOp.from_list(terms).simplify()
    return Hp, n


# --------------------------------------------------------------------------
# Helpers for the counterdiabatic generator (SparsePauliOp algebra)
# --------------------------------------------------------------------------
def _pauli_dict(op):
    """{label: coeff} of a simplified SparsePauliOp."""
    op = op.simplify()
    return {lbl: c for lbl, c in zip(op.paulis.to_labels(), op.coeffs)}


def _trace_prod(A, B):
    """Normalized Tr(A B) / 2**n = sum_P a_P b_P (Pauli orthogonality)."""
    da, db = _pauli_dict(A), _pauli_dict(B)
    return sum(da[p] * db[p] for p in da if p in db)


def _frob(A, B):
    """Normalized Tr(A! B) / 2**n = sum_P conj(a_P) b_P."""
    da, db = _pauli_dict(A), _pauli_dict(B)
    return sum(np.conj(da[p]) * db[p] for p in da if p in db)


def _commutator(A, B):
    return (A.dot(B) - B.dot(A)).simplify()


def _quadratic_integral(a, b, c):
    """int_0^1 dl / (a l**2 + b l + c) in closed form
    (docs/BF-DCQO_Implementation.md, Sec. 11)."""
    Delta = 4.0 * a * c - b * b
    tol = 1e-12
    if Delta > tol:
        s = np.sqrt(Delta)
        return (2.0 / s) * (np.arctan((2.0 * a + b) / s) - np.arctan(b / s))
    elif Delta < -tol:
        xi = np.sqrt(-Delta)
        num = (2.0 * a + b - xi) * (b + xi)
        den = (2.0 * a + b + xi) * (b - xi)
        return (1.0 / xi) * np.log(abs(num / den))
    else:
        return 2.0 * (1.0 / b - 1.0 / (2.0 * a + b))


def _prepare_cd_static(Hp):
    n = Hp.num_qubits

    Hi0 = SparsePauliOp.from_list(
        [
            (_label({q: "X"}, n), -1.0)
            for q in range(n)
        ]
    ).simplify()

    O1 = _commutator(Hi0, Hp)
    X_base = _commutator(Hi0, O1)
    Yop = _commutator(Hp, O1)

    beta1 = _trace_prod(O1, O1).real
    beta4 = _frob(Yop, Yop).real

    z_commutators = []

    for q in range(n):
        Zq = SparsePauliOp.from_list(
            [
                (
                    _label({q: "Z"}, n),
                    1.0,
                )
            ]
        )

        z_commutators.append(
            _commutator(Zq, O1)
        )

    A_op = (-0.5j * O1).simplify()
    A_op = SparsePauliOp(
        A_op.paulis,
        np.real(A_op.coeffs),
    ).simplify()

    return {
        "A_op": A_op,
        "X_base": X_base,
        "Yop": Yop,
        "beta1": float(beta1),
        "beta4": float(beta4),
        "z_commutators": z_commutators,
    }


def _theta_from_bias(
    bias,
    cd_static,
):
    Xop = cd_static["X_base"]

    for q, bq in enumerate(
        np.asarray(bias, dtype=float)
    ):
        if abs(float(bq)) <= 1e-15:
            continue

        Xop = (
            Xop
            - float(bq)
            * cd_static["z_commutators"][q]
        ).simplify()

    Yop = cd_static["Yop"]

    beta1 = float(cd_static["beta1"])
    beta2 = _frob(Xop, Xop).real
    beta3 = 2.0 * _frob(Xop, Yop).real
    beta4 = float(cd_static["beta4"])

    a = beta2 - beta3 + beta4
    b = beta3 - 2.0 * beta2
    c = beta2

    if (
        abs(beta1) < 1e-15
        or (
            abs(a) < 1e-15
            and abs(b) < 1e-15
            and abs(c) < 1e-15
        )
    ):
        return 0.0

    return float(
        beta1
        * _quadratic_integral(
            a,
            b,
            c,
        )
    )


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------
def build_circuit(
    Hp,
    bias,
    cd_static,
    measure=True,
):
    n = Hp.num_qubits

    A_op = cd_static["A_op"]
    Theta = _theta_from_bias(
        bias,
        cd_static,
    )

    qc = QuantumCircuit(n)

    for q in range(n):
        b = float(bias[q])
        R = np.sqrt(1.0 + b * b)
        a0 = np.sqrt((R + b) / (2.0 * R))
        a1 = np.sqrt((R - b) / (2.0 * R))

        qc.ry(
            2.0 * np.arctan2(a1, a0),
            q,
        )

    if len(A_op) > 0 and abs(Theta) > 1e-15:
        qc.append(
            PauliEvolutionGate(
                A_op,
                time=-2.0 * Theta,
            ),
            range(n),
        )

    if measure:
        qc.measure_all()

    return qc



# --------------------------------------------------------------------------
# Energy evaluation without dense matrices
# --------------------------------------------------------------------------
def _parse_qubo_coefficients(X):
    offset = 0.0
    linear = {}
    quadratic = {}

    for key, value in X.items():
        term = _parse_key(key)
        if len(term) == 0:
            offset += float(value)
        elif len(term) == 1:
            i = term[0]
            linear[i] = linear.get(i, 0.0) + float(value)
        elif len(term) == 2:
            i, j = sorted(term)
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + float(value)
        else:
            raise ValueError("Only QUBO terms up to degree 2 are supported.")

    return float(offset), linear, quadratic


def _qubo_energy_from_bits(bits, parsed_qubo):
    offset, linear, quadratic = parsed_qubo
    x = np.asarray(bits, dtype=int)

    value = offset
    value += sum(coef * x[i] for i, coef in linear.items())
    value += sum(
        coef * x[i] * x[j]
        for (i, j), coef in quadratic.items()
    )
    return float(value)


def _qubo_energy_from_index(idx, n, parsed_qubo):
    return _qubo_energy_from_bits(
        _index_to_bits(int(idx), n),
        parsed_qubo,
    )


# --------------------------------------------------------------------------
# Incremental greedy method for the sweep experiment
# --------------------------------------------------------------------------
def _prepare_fast_qubo(
    parsed_qubo,
    n,
):
    offset, linear_dict, quadratic_dict = parsed_qubo

    linear = np.zeros(
        n,
        dtype=float,
    )

    for j, value in linear_dict.items():
        linear[int(j)] += float(value)

    neighbors = [
        []
        for _ in range(n)
    ]

    edges = []

    for (j, k), value in quadratic_dict.items():
        j = int(j)
        k = int(k)
        value = float(value)

        neighbors[j].append(
            (k, value)
        )

        neighbors[k].append(
            (j, value)
        )

        edges.append(
            (j, k, value)
        )

    return {
        "offset": float(offset),
        "linear": linear,
        "neighbors": neighbors,
        "edges": edges,
    }


def _fast_qubo_energy_from_bits(
    bits,
    fast_qubo,
):
    x = np.asarray(
        bits,
        dtype=np.int8,
    )

    value = float(
        fast_qubo["offset"]
    )

    value += float(
        np.dot(
            fast_qubo["linear"],
            x,
        )
    )

    for j, k, coefficient in fast_qubo["edges"]:
        if x[j] and x[k]:
            value += coefficient

    return float(value)


def greedy_fixed_sweeps_fast(
    bitstring,
    sweeps,
    fast_qubo,
):
    bits = np.asarray(
        bitstring,
        dtype=np.int8,
    ).copy()

    current_energy = _fast_qubo_energy_from_bits(
        bits,
        fast_qubo,
    )

    local_fields = np.asarray(
        fast_qubo["linear"],
        dtype=float,
    ).copy()

    for j, k, coefficient in fast_qubo["edges"]:
        local_fields[j] += coefficient * bits[k]
        local_fields[k] += coefficient * bits[j]

    for _ in range(
        max(0, int(sweeps))
    ):
        improved = False

        for j in range(bits.size):
            old_bit = int(bits[j])

            delta_energy = (
                (1 - 2 * old_bit)
                * local_fields[j]
            )

            if delta_energy < -1e-12:
                new_bit = 1 - old_bit
                bits[j] = new_bit
                current_energy += float(delta_energy)

                bit_change = new_bit - old_bit

                for k, coefficient in fast_qubo["neighbors"][j]:
                    local_fields[k] += coefficient * bit_change

                improved = True

        if not improved:
            break

    return bits, float(current_energy)


# --------------------------------------------------------------------------
# 2. BF-DCQO in the impulse regime (Qiskit Aer direct sampling)
# --------------------------------------------------------------------------

def _counts_to_sampled_indices(
    counts,
    number_qubits,
):
    sampled = np.empty(
        sum(
            int(count)
            for count in counts.values()
        ),
        dtype=np.int64,
    )

    position = 0

    for bitstring, count in counts.items():
        clean_bitstring = str(
            bitstring
        ).replace(
            " ",
            "",
        )

        clean_bitstring = clean_bitstring.zfill(
            int(
                number_qubits
            )
        )

        index = int(
            clean_bitstring,
            2,
        )

        count = int(
            count
        )

        sampled[
            position:
            position + count
        ] = index

        position += count

    return sampled


def bf_dcqo(
    Hp,
    parsed_qubo,
    r,
    shots=2048,
    elite_fraction=0.1,
    lambda_b=1.0,
    eta=0.5,
    seed=None,
    return_history=False,
    optimal_energy=None,
    optimal_tolerance=1e-9,
    fixed_sweep=2,
    aer_device="CPU",
):
    n = Hp.num_qubits

    rng = np.random.default_rng(
        seed
    )

    fixed_sweep = int(
        fixed_sweep
    )

    if fixed_sweep < 0:
        raise ValueError(
            "fixed_sweep must be nonnegative."
        )

    fast_qubo = _prepare_fast_qubo(
        parsed_qubo,
        n,
    )

    cd_static = _prepare_cd_static(
        Hp
    )

    backend = AerSimulator(
        method="statevector",
        device=str(
            aer_device
        ),
    )

    bias = np.zeros(
        n,
        dtype=float,
    )

    best_bits = None
    best_energy = np.inf
    history = []

    metrics = {
        "fixed_sweep": fixed_sweep,
        "aer_device": str(
            aer_device
        ),
        "mean_energy_by_iteration": [],
        "std_energy_by_iteration": [],
        "min_energy_by_iteration": [],
        "max_energy_by_iteration": [],
        "mean_energy_after_fixed_sweep": [],
        "min_energy_after_fixed_sweep": [],
        "optimal_count_fixed_sweep": [],
        "optimal_probability_fixed_sweep": [],
        "raw_optimal_count_by_iteration": [],
        "raw_optimal_probability_by_iteration": [],
        "unique_states_by_iteration": [],
        "time_build_seconds": [],
        "time_transpile_seconds": [],
        "time_aer_seconds": [],
        "time_classical_seconds": [],
        "time_total_iteration_seconds": [],
        "total_shots": 0,
        "optimal_energy_reference": (
            None
            if optimal_energy is None
            else float(
                optimal_energy
            )
        ),
    }

    for iteration in range(
        max(
            1,
            int(r),
        )
    ):
        iteration_start = time.perf_counter()

        # --------------------------------------------------------------
        # 1. Circuit construction/decomposition.
        # --------------------------------------------------------------
        t0 = time.perf_counter()

        qc = build_circuit(
            Hp=Hp,
            bias=bias,
            cd_static=cd_static,
            measure=True,
        )

        qc_decomposed = qc.decompose(
            reps=10
        )

        build_seconds = (
            time.perf_counter()
            - t0
        )

        # --------------------------------------------------------------
        # 2. Transpilation for Aer.
        # --------------------------------------------------------------
        t0 = time.perf_counter()

        qc_aer = transpile(
            qc_decomposed,
            backend=backend,
            optimization_level=0,
        )

        transpile_seconds = (
            time.perf_counter()
            - t0
        )

        # --------------------------------------------------------------
        # 3. Direct sampling with Aer.
        # --------------------------------------------------------------
        t0 = time.perf_counter()

        simulator_seed = (
            None
            if seed is None
            else int(seed)
            + int(iteration)
        )

        result = backend.run(
            qc_aer,
            shots=int(
                shots
            ),
            seed_simulator=simulator_seed,
        ).result()

        counts = result.get_counts(
            qc_aer
        )

        aer_seconds = (
            time.perf_counter()
            - t0
        )

        # --------------------------------------------------------------
        # 4. Reconstruction and postprocessing.
        # --------------------------------------------------------------
        classical_start = time.perf_counter()

        sampled = _counts_to_sampled_indices(
            counts=counts,
            number_qubits=n,
        )

        if sampled.size != int(shots):
            raise RuntimeError(
                f"Aer returned {sampled.size} shots, "
                f"but {shots} were expected."
            )

        rng.shuffle(
            sampled
        )

        unique_states, inverse = np.unique(
            sampled,
            return_inverse=True,
        )

        metrics[
            "unique_states_by_iteration"
        ].append(
            int(
                unique_states.size
            )
        )

        unique_raw_energies = np.empty(
            unique_states.size,
            dtype=float,
        )

        unique_greedy_energies = np.empty(
            unique_states.size,
            dtype=float,
        )

        for position, idx in enumerate(
            unique_states
        ):
            bits = _index_to_bits(
                int(idx),
                n,
            )

            unique_raw_energies[
                position
            ] = (
                _fast_qubo_energy_from_bits(
                    bits,
                    fast_qubo,
                )
            )

            if fixed_sweep == 0:
                # s=0: no greedy postprocessing.
                # The energy after the sweep is exactly the RAW energy.
                greedy_energy = unique_raw_energies[position]
            else:
                _, greedy_energy = (
                    greedy_fixed_sweeps_fast(
                        bitstring=bits,
                        sweeps=fixed_sweep,
                        fast_qubo=fast_qubo,
                    )
                )

            unique_greedy_energies[
                position
            ] = (
                greedy_energy
            )

        sample_energies = (
            unique_raw_energies[
                inverse
            ]
        )

        greedy_energies = (
            unique_greedy_energies[
                inverse
            ]
        )

        mean_energy = float(
            np.mean(
                sample_energies
            )
        )

        std_energy = float(
            np.std(
                sample_energies
            )
        )

        min_energy = float(
            np.min(
                sample_energies
            )
        )

        max_energy = float(
            np.max(
                sample_energies
            )
        )

        mean_greedy_energy = float(
            np.mean(
                greedy_energies
            )
        )

        min_greedy_energy = float(
            np.min(
                greedy_energies
            )
        )

        metrics[
            "mean_energy_by_iteration"
        ].append(
            mean_energy
        )

        metrics[
            "std_energy_by_iteration"
        ].append(
            std_energy
        )

        metrics[
            "min_energy_by_iteration"
        ].append(
            min_energy
        )

        metrics[
            "max_energy_by_iteration"
        ].append(
            max_energy
        )

        metrics[
            "mean_energy_after_fixed_sweep"
        ].append(
            mean_greedy_energy
        )

        metrics[
            "min_energy_after_fixed_sweep"
        ].append(
            min_greedy_energy
        )

        if optimal_energy is not None:
            raw_optimal_count = int(
                np.count_nonzero(
                    np.isclose(
                        sample_energies,
                        float(
                            optimal_energy
                        ),
                        atol=float(
                            optimal_tolerance
                        ),
                        rtol=0.0,
                    )
                )
            )

            greedy_optimal_count = int(
                np.count_nonzero(
                    np.isclose(
                        greedy_energies,
                        float(
                            optimal_energy
                        ),
                        atol=float(
                            optimal_tolerance
                        ),
                        rtol=0.0,
                    )
                )
            )
        else:
            raw_optimal_count = 0
            greedy_optimal_count = 0

        raw_optimal_probability = (
            raw_optimal_count
            / float(
                shots
            )
        )

        greedy_optimal_probability = (
            greedy_optimal_count
            / float(
                shots
            )
        )

        metrics[
            "raw_optimal_count_by_iteration"
        ].append(
            raw_optimal_count
        )

        metrics[
            "raw_optimal_probability_by_iteration"
        ].append(
            float(
                raw_optimal_probability
            )
        )

        metrics[
            "optimal_count_fixed_sweep"
        ].append(
            greedy_optimal_count
        )

        metrics[
            "optimal_probability_fixed_sweep"
        ].append(
            float(
                greedy_optimal_probability
            )
        )

        metrics[
            "total_shots"
        ] += int(
            shots
        )

        local_best = int(
            np.argmin(
                sample_energies
            )
        )

        if (
            sample_energies[
                local_best
            ]
            < best_energy
        ):
            best_energy = float(
                sample_energies[
                    local_best
                ]
            )

            best_bits = _index_to_bits(
                int(
                    sampled[
                        local_best
                    ]
                ),
                n,
            )

        history.append(
            best_energy
        )

        # --------------------------------------------------------------
        # 5. BF-DCQO feedback based on RAW energies.
        # --------------------------------------------------------------
        n_elite = max(
            1,
            int(
                elite_fraction
                * len(
                    sampled
                )
            ),
        )

        elite_idx = sampled[
            np.argsort(
                sample_energies
            )[:n_elite]
        ]

        elite_bits = np.asarray(
            [
                _index_to_bits(
                    int(i),
                    n,
                )
                for i in elite_idx
            ],
            dtype=np.int8,
        )

        m = np.mean(
            1
            - 2 * elite_bits,
            axis=0,
        )

        bias = (
            (1.0 - eta)
            * bias
            + eta
            * lambda_b
            * m
        )

        classical_seconds = (
            time.perf_counter()
            - classical_start
        )

        total_iteration_seconds = (
            time.perf_counter()
            - iteration_start
        )

        metrics[
            "time_build_seconds"
        ].append(
            float(
                build_seconds
            )
        )

        metrics[
            "time_transpile_seconds"
        ].append(
            float(
                transpile_seconds
            )
        )

        metrics[
            "time_aer_seconds"
        ].append(
            float(
                aer_seconds
            )
        )

        metrics[
            "time_classical_seconds"
        ].append(
            float(
                classical_seconds
            )
        )

        metrics[
            "time_total_iteration_seconds"
        ].append(
            float(
                total_iteration_seconds
            )
        )

        print(
            f"BF-DCQO iteration "
            f"{iteration + 1}/{max(1, int(r))}: "
            f"best RAW={best_energy:.12f}, "
            f"Popt(s={fixed_sweep})="
            f"{100.0 * greedy_optimal_probability:.4f}%, "
            f"unique={unique_states.size}/{int(shots)}, "
            f"build={build_seconds:.2f}s, "
            f"transpile={transpile_seconds:.2f}s, "
            f"Aer={aer_seconds:.2f}s, "
            f"classical={classical_seconds:.2f}s, "
            f"total={total_iteration_seconds:.2f}s"
        )

        del sampled
        del sample_energies
        del greedy_energies

    if return_history:
        return (
            best_bits,
            history,
            metrics,
        )

    return best_bits



def solve_qubo_with_ibm_simulator(
    qubo: QUBO,
    r: int = 30,
    shots: int = 2048,
    elite_fraction: float = 0.1,
    lambda_b: float = 1.0,
    eta: float = 0.5,
    fixed_sweep: int = 2,
    seed: int | None = 0,
    optimal_energy: float | None = None,
    aer_device: str = "CPU",
) -> Tuple[np.ndarray, float, List[float], Dict[str, object]]:
    qubo_problem = build_qubo_problem(
        qubo
    )

    parsed_qubo = _parse_qubo_coefficients(
        qubo_problem
    )

    Hp, n = build_spin_glass_hamiltonian(
        qubo_problem
    )

    if n != qubo.number_variables:
        raise RuntimeError(
            f"The solver detected {n} qubits, "
            f"but the QUBO contains "
            f"{qubo.number_variables} variables."
        )

    bf_bits, history, metrics = bf_dcqo(
        Hp=Hp,
        parsed_qubo=parsed_qubo,
        r=r,
        shots=shots,
        elite_fraction=elite_fraction,
        lambda_b=lambda_b,
        eta=eta,
        seed=seed,
        return_history=True,
        optimal_energy=optimal_energy,
        fixed_sweep=fixed_sweep,
        aer_device=aer_device,
    )

    best_bits = np.asarray(
        bf_bits,
        dtype=int,
    )

    best_energy = qubo.energy(
        best_bits
    )

    return (
        best_bits,
        float(best_energy),
        history,
        metrics,
    )


# ==========================================================================
# RANDOM 22-QUBIT EXPERIMENT + GUROBI
# ==========================================================================

# Same ranges as By to enable paired comparisons.
BLOCK_VALUE_RANGES_BY_Y = {
    3: (-50, 100),
    2: (75, 350),
    1: (300, 550),
}


def generate_random_block_values(
    coordinates: Sequence[Tuple[int, int]],
    seed: int,
    ranges_by_y: Mapping[int, Tuple[int, int]] = BLOCK_VALUE_RANGES_BY_Y,
) -> List[int]:
    """Generate reproducible random economic values by depth."""
    rng = np.random.default_rng(int(seed))
    values: List[int] = []

    for coordinate in coordinates:
        _, y = coordinate

        if y not in ranges_by_y:
            raise KeyError(
                f"No economic value range is configured for y={y}."
            )

        low, high = ranges_by_y[y]

        if low > high:
            raise ValueError(
                f"Invalid range for y={y}: ({low}, {high})."
            )

        values.append(
            int(
                rng.integers(
                    int(low),
                    int(high) + 1,
                )
            )
        )

    return values


def solve_classical_optimum_gurobi(
    block_values: Sequence[float],
    coordinates: Sequence[Tuple[int, int]],
    capacities: Sequence[int],
    periods: int,
    discount_rate: float,
) -> Dict[str, object]:
    """
    Solve the mining instance exactly using Gurobi.

    Cumulative variables y[i,t] are used to obtain the classical optimum,
    even though the QUBO subsequently solved uses the At formulation.
    """
    number_blocks = len(block_values)
    precedence = build_geometric_precedence(coordinates)
    B_raw = discounted_values(
        block_values,
        periods,
        discount_rate,
    )

    model = Model("random_mining_22q_at")
    model.Params.OutputFlag = 0

    y = {
        (i, t): model.addVar(
            vtype=GRB.BINARY,
            name=f"y_{i}_{t}",
        )
        for i in range(number_blocks)
        for t in range(periods)
    }

    model.setObjective(
        quicksum(
            float(B_raw[t, i])
            * (
                y[i, t]
                - (
                    y[i, t - 1]
                    if t > 0
                    else 0
                )
            )
            for i in range(number_blocks)
            for t in range(periods)
        ),
        GRB.MAXIMIZE,
    )

    # Monotonicity.
    for i in range(number_blocks):
        for t in range(periods - 1):
            model.addConstr(
                y[i, t] <= y[i, t + 1]
            )

    # Cumulative precedence relations.
    for i, predecessors in precedence.items():
        for predecessor in predecessors:
            for t in range(periods):
                model.addConstr(
                    y[i, t]
                    <= y[int(predecessor), t]
                )

    # Exact extraction capacity.
    for t in range(periods):
        model.addConstr(
            quicksum(
                y[i, t]
                - (
                    y[i, t - 1]
                    if t > 0
                    else 0
                )
                for i in range(number_blocks)
            )
            <= int(capacities[t])
        )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Gurobi did not find an optimal solution. Status={model.Status}"
        )

    accumulated = np.zeros(
        (number_blocks, periods),
        dtype=int,
    )

    for i in range(number_blocks):
        for t in range(periods):
            accumulated[i, t] = int(
                round(y[i, t].X)
            )

    extracted = np.zeros_like(accumulated)
    extracted[:, 0] = accumulated[:, 0]

    for t in range(1, periods):
        extracted[:, t] = (
            accumulated[:, t]
            - accumulated[:, t - 1]
        )

    extracted_by_period = {
        str(t + 1): [
            i + 1
            for i in range(number_blocks)
            if extracted[i, t] == 1
        ]
        for t in range(periods)
    }

    return {
        "optimal_npv": float(model.ObjVal),
        "extracted_by_period": extracted_by_period,
    }


def build_compact_run_result(
    run_index: int,
    block_seed: int,
    block_values: Sequence[int],
    optimal_npv: float,
    economic_scale: float,
    optimal_qubo_energy: float,
    gurobi_result: Mapping[str, object],
    best_energy: float,
    metrics: Mapping[str, object],
    fixed_sweep: int,
    solver_r: int,
    alpha: float,
    beta: float,
    gamma: float,
    d_max: int,
    beta_bound: float,
    successor_count: Mapping[int, int],
) -> Dict[str, object]:
    return {
        "run": int(run_index),
        "block_seed": int(block_seed),
        "block_values": [
            int(v)
            for v in block_values
        ],
        "optimal_npv": float(optimal_npv),
        "economic_scale": float(economic_scale),
        "optimal_qubo_energy": float(
            optimal_qubo_energy
        ),
        "gurobi_extracted_by_period": (
            gurobi_result[
                "extracted_by_period"
            ]
        ),
        "penalties": {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "d_max": int(d_max),
            "beta_bound": float(beta_bound),
            "hierarchy_satisfied": bool(
                beta > beta_bound
            ),
        },
        "successor_degrees": {
            str(k): int(v)
            for k, v in successor_count.items()
        },
        "best_sampled_qubo_energy": float(
            best_energy
        ),
        "best_energy_below_feasible_optimum": bool(
            best_energy
            < optimal_qubo_energy
            - 1e-8
        ),
        "r_values": list(
            range(
                1,
                int(solver_r) + 1,
            )
        ),
        "fixed_sweep": int(fixed_sweep),
        "optimal_probability_fixed_sweep": [
            float(v)
            for v in metrics[
                "optimal_probability_fixed_sweep"
            ]
        ],
        "optimal_count_fixed_sweep": [
            int(v)
            for v in metrics[
                "optimal_count_fixed_sweep"
            ]
        ],
        "raw_optimal_probability_by_iteration": [
            float(v)
            for v in metrics[
                "raw_optimal_probability_by_iteration"
            ]
        ],
        "mean_energy_by_iteration": [
            float(v)
            for v in metrics[
                "mean_energy_by_iteration"
            ]
        ],
        "std_energy_by_iteration": [
            float(v)
            for v in metrics[
                "std_energy_by_iteration"
            ]
        ],
        "min_energy_by_iteration": [
            float(v)
            for v in metrics[
                "min_energy_by_iteration"
            ]
        ],
        "max_energy_by_iteration": [
            float(v)
            for v in metrics[
                "max_energy_by_iteration"
            ]
        ],
        "mean_energy_after_fixed_sweep": [
            float(v)
            for v in metrics[
                "mean_energy_after_fixed_sweep"
            ]
        ],
        "min_energy_after_fixed_sweep": [
            float(v)
            for v in metrics[
                "min_energy_after_fixed_sweep"
            ]
        ],
        "unique_states_by_iteration": [
            int(v)
            for v in metrics[
                "unique_states_by_iteration"
            ]
        ],
        "timing": {
            "build_seconds": [
                float(v)
                for v in metrics[
                    "time_build_seconds"
                ]
            ],
            "transpile_seconds": [
                float(v)
                for v in metrics[
                    "time_transpile_seconds"
                ]
            ],
            "aer_seconds": [
                float(v)
                for v in metrics[
                    "time_aer_seconds"
                ]
            ],
            "classical_seconds": [
                float(v)
                for v in metrics[
                    "time_classical_seconds"
                ]
            ],
            "total_iteration_seconds": [
                float(v)
                for v in metrics[
                    "time_total_iteration_seconds"
                ]
            ],
        },
    }


def save_experiment_json(
    filename: Path,
    configuration: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
) -> None:
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_runs = sorted(
        runs,
        key=lambda item: int(
            item["run"]
        ),
    )

    output: Dict[str, object] = {
        "configuration": dict(
            configuration
        ),
        "completed_runs": len(
            ordered_runs
        ),
        "runs": ordered_runs,
    }

    if ordered_runs:
        probability_curves = np.asarray(
            [
                run[
                    "optimal_probability_fixed_sweep"
                ]
                for run in ordered_runs
            ],
            dtype=float,
        )

        output[
            "aggregate_statistics"
        ] = {
            "number_instances": len(
                ordered_runs
            ),
            "r_values": ordered_runs[
                0
            ]["r_values"],
            "fixed_sweep": ordered_runs[
                0
            ]["fixed_sweep"],
            "mean_optimal_probability_by_r": (
                np.mean(
                    probability_curves,
                    axis=0,
                )
                .astype(float)
                .tolist()
            ),
            "central_97_percent_band": {
                "lower_percentile": 1.5,
                "upper_percentile": 98.5,
                "lower_optimal_probability_by_r": (
                    np.percentile(
                        probability_curves,
                        1.5,
                        axis=0,
                    )
                    .astype(float)
                    .tolist()
                ),
                "upper_optimal_probability_by_r": (
                    np.percentile(
                        probability_curves,
                        98.5,
                        axis=0,
                    )
                    .astype(float)
                    .tolist()
                ),
            },
        }

    with filename.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_existing_runs(
    filename: Path,
) -> List[Dict[str, object]]:
    """Resume the experiment from an existing JSON file."""
    if not filename.exists():
        return []

    with filename.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return list(
        data.get(
            "runs",
            [],
        )
    )



def _independent_solver_seed(
    base_seed: int,
    run_index: int,
    R: int,
) -> int:
    """Reproducible and distinct seed for each (instance, R) pair."""
    return int(base_seed) + 100_000 * (int(run_index) - 1) + 1_000 * int(R)


def _load_checkpoint(filename: Path) -> Dict[str, object]:
    """Load an independent-R experiment checkpoint, if one exists."""
    if not filename.exists():
        return {
            "configuration": {},
            "instances": [],
            "aggregate_statistics": {},
        }

    with filename.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "instances" not in data:
        raise RuntimeError(
            "The existing output file does not match the expected "
            "independent-R format. Use a new output filename or move "
            "the previous file."
        )

    return data


def _aggregate_independent_R(
    instances: Sequence[Mapping[str, object]],
    R_values: Sequence[int],
) -> Dict[str, object]:
    """
    Aggregate instances by R budget using the median and IQR.

    The main metric is RAW P_opt at the terminal iteration r=R.
    """
    aggregate: Dict[str, object] = {
        "R_values": [int(R) for R in R_values],
        "number_instances_total": int(len(instances)),
        "number_instances_by_R": [],
        "median_raw_optimal_probability_by_R": [],
        "q1_raw_optimal_probability_by_R": [],
        "q3_raw_optimal_probability_by_R": [],
        "median_terminal_mean_energy_by_R": [],
        "q1_terminal_mean_energy_by_R": [],
        "q3_terminal_mean_energy_by_R": [],
    }

    for R in R_values:
        probabilities: List[float] = []
        terminal_means: List[float] = []
        key = str(int(R))

        for instance in instances:
            R_results = instance.get("R_results", {})
            if key not in R_results:
                continue

            result_R = R_results[key]
            probabilities.append(
                float(result_R["raw_optimal_probability"])
            )
            terminal_means.append(
                float(result_R["terminal_mean_energy"])
            )

        aggregate["number_instances_by_R"].append(
            int(len(probabilities))
        )

        if probabilities:
            probs = np.asarray(probabilities, dtype=float)
            means = np.asarray(terminal_means, dtype=float)

            aggregate["median_raw_optimal_probability_by_R"].append(
                float(np.median(probs))
            )
            aggregate["q1_raw_optimal_probability_by_R"].append(
                float(np.percentile(probs, 25.0))
            )
            aggregate["q3_raw_optimal_probability_by_R"].append(
                float(np.percentile(probs, 75.0))
            )

            aggregate["median_terminal_mean_energy_by_R"].append(
                float(np.median(means))
            )
            aggregate["q1_terminal_mean_energy_by_R"].append(
                float(np.percentile(means, 25.0))
            )
            aggregate["q3_terminal_mean_energy_by_R"].append(
                float(np.percentile(means, 75.0))
            )
        else:
            aggregate["median_raw_optimal_probability_by_R"].append(None)
            aggregate["q1_raw_optimal_probability_by_R"].append(None)
            aggregate["q3_raw_optimal_probability_by_R"].append(None)
            aggregate["median_terminal_mean_energy_by_R"].append(None)
            aggregate["q1_terminal_mean_energy_by_R"].append(None)
            aggregate["q3_terminal_mean_energy_by_R"].append(None)

    return aggregate


def _save_independent_R_checkpoint(
    filename: Path,
    configuration: Mapping[str, object],
    instances: Sequence[Mapping[str, object]],
    R_values: Sequence[int],
) -> None:
    """Atomically save progress after each (instance, R) pair."""
    filename.parent.mkdir(parents=True, exist_ok=True)

    ordered_instances = sorted(
        instances,
        key=lambda item: int(item["run"]),
    )

    completed_pairs = sum(
        len(instance.get("R_results", {}))
        for instance in ordered_instances
    )

    fully_completed_instances = sum(
        len(instance.get("R_results", {})) == len(R_values)
        for instance in ordered_instances
    )

    output = {
        "configuration": dict(configuration),
        "completed_instance_R_pairs": int(completed_pairs),
        "fully_completed_instances": int(fully_completed_instances),
        "instances": ordered_instances,
        "aggregate_statistics": _aggregate_independent_R(
            instances=ordered_instances,
            R_values=R_values,
        ),
    }

    temporary = filename.with_suffix(filename.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(filename)


def main() -> None:
    # ------------------------------------------------------------------
    # 22-qubit At case: same geometry as By so that the
    # 100 instances can be compared in pairs.
    # ------------------------------------------------------------------
    coordinates = [
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (2, 2),
        (3, 2),
        (4, 2),
        (3, 1),
    ]

    capacities = [3, 3]
    periods = 2
    discount_rate = 0.08

    # 9*2 At variables + 4 slack bits = 22 qubits.

    # ------------------------------------------------------------------
    # Final random-instance experiment.
    # ------------------------------------------------------------------
    number_random_instances = 100
    base_block_seed = 1000

    # Each R denotes an independent BF-DCQO run starting with
    # bias=0 and internally executing r=1,...,R.
    R_values = list(range(1, 31))

    # Same number of shots as the corrected fixed case.
    solver_shots = 1_000_000
    solver_elite_fraction = 0.1
    solver_lambda_b = 1.0
    solver_eta = 0.5

    # Each (instance, R) pair uses a distinct yet reproducible solver seed.
    solver_seed_base = 0

    # s=0: no greedy refinement. All reported probabilities are RAW.
    fixed_sweep = 0

    # Switch to "GPU" if qiskit-aer was installed with CUDA support.
    aer_device = "CPU"

    # At penalty factors used in the original random-instance experiment.
    c_capacity = 1.5
    c_validity = 1.5
    c_precedence = 1.5
    hierarchy_epsilon = 1e-6

    results_filename = (
        Path(__file__).resolve().parent / "../../data/At_22Q_random_100_independent_R_s0.json"
    )

    total_bf_iterations_per_instance = int(sum(R_values))
    total_bf_iterations = int(
        number_random_instances * total_bf_iterations_per_instance
    )
    total_requested_shots = int(
        total_bf_iterations * solver_shots
    )

    configuration = {
        "formulation": "At",
        "case_name": "22_qubits_random_independent_R",
        "number_qubits": 22,
        "number_blocks": len(coordinates),
        "coordinates": [list(c) for c in coordinates],
        "capacities": capacities,
        "periods": periods,
        "discount_rate": discount_rate,
        "number_random_instances_requested": number_random_instances,
        "base_block_seed": base_block_seed,
        "block_value_ranges_by_y": {
            str(y): list(bounds)
            for y, bounds in BLOCK_VALUE_RANGES_BY_Y.items()
        },
        "penalty_factors": {
            "c_capacity": c_capacity,
            "c_validity": c_validity,
            "c_precedence": c_precedence,
            "hierarchy_epsilon": hierarchy_epsilon,
        },
        "R_values": R_values,
        "solver_shots_per_iteration": solver_shots,
        "solver_elite_fraction": solver_elite_fraction,
        "solver_lambda_b": solver_lambda_b,
        "solver_eta": solver_eta,
        "solver_seed_base": solver_seed_base,
        "seed_policy": (
            "base + 100000*(instance-1) + 1000*R; "
            "bf_dcqo adds the internal iteration index"
        ),
        "aer_method": "statevector",
        "aer_device": aer_device,
        "fixed_sweep": fixed_sweep,
        "greedy_applied": False,
        "terminal_metric_definition": (
            "For each independent maximum R, statistics are taken only "
            "from its final internal BF-DCQO iteration r=R."
        ),
        "aggregate_definition": "median_and_IQR_across_random_instances",
        "total_bf_iterations_per_instance": total_bf_iterations_per_instance,
        "total_bf_iterations_requested": total_bf_iterations,
        "total_quantum_shots_requested": total_requested_shots,
        "checkpoint_policy": "save_after_every_instance_R_pair",
    }

    checkpoint = _load_checkpoint(results_filename)
    instances: List[Dict[str, object]] = list(
        checkpoint.get("instances", [])
    )

    instances_by_run = {
        int(instance["run"]): instance
        for instance in instances
    }

    print("=" * 78)
    print("RANDOM At EXPERIMENT - INDEPENDENT R - s=0")
    print("Random instances:", number_random_instances)
    print("R:", R_values[0], "a", R_values[-1])
    print("Shots per iteration:", f"{solver_shots:,}")
    print(
        "BF-DCQO iterations per instance:",
        f"{total_bf_iterations_per_instance:,}",
    )
    print(
        "Total BF-DCQO iterations:",
        f"{total_bf_iterations:,}",
    )
    print(
        "Total requested shots:",
        f"{total_requested_shots:,}",
    )
    print("Greedy: NO (s=0)")
    print("Checkpoint:", results_filename)
    print("=" * 78)

    for run_index in range(1, number_random_instances + 1):
        block_seed = base_block_seed + run_index - 1

        block_values = generate_random_block_values(
            coordinates=coordinates,
            seed=block_seed,
        )

        # --------------------------------------------------------------
        # If the instance exists in the checkpoint, reuse its data.
        # Otherwise, solve the classical optimum first and build its data.
        # --------------------------------------------------------------
        if run_index in instances_by_run:
            instance_result = instances_by_run[run_index]
            R_results = instance_result.setdefault("R_results", {})
        else:
            gurobi_result = solve_classical_optimum_gurobi(
                block_values=block_values,
                coordinates=coordinates,
                capacities=capacities,
                periods=periods,
                discount_rate=discount_rate,
            )

            optimal_npv = float(
                gurobi_result["optimal_npv"]
            )

            B_raw = discounted_values(
                block_values,
                periods,
                discount_rate,
            )
            B_normalized, economic_scale = normalize_values(B_raw)
            precedence = build_geometric_precedence(coordinates)

            alpha, beta, gamma = initialize_penalties(
                normalized_values=B_normalized,
                precedence=precedence,
                c_capacity=c_capacity,
                c_validity=c_validity,
                c_precedence=c_precedence,
                hierarchy_epsilon=hierarchy_epsilon,
            )

            successor_count, d_max = successor_degrees(
                precedence=precedence,
                number_blocks=len(block_values),
            )
            beta_bound = float(d_max * gamma)

            if not beta > beta_bound:
                raise RuntimeError(
                    "The condition beta > d_max * gamma is not satisfied."
                )

            qubo = build_qubo(
                normalized_values=B_normalized,
                precedence=precedence,
                capacities=capacities,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            if qubo.number_variables != 22:
                raise RuntimeError(
                    f"The case no longer contains 22 qubits: "
                    f"{qubo.number_variables}"
                )

            optimal_qubo_energy = (
                -optimal_npv / float(economic_scale)
            )

            instance_result = {
                "run": int(run_index),
                "block_seed": int(block_seed),
                "block_values": [int(v) for v in block_values],
                "optimal_npv": float(optimal_npv),
                "economic_scale": float(economic_scale),
                "optimal_qubo_energy": float(optimal_qubo_energy),
                "gurobi_extracted_by_period": (
                    gurobi_result["extracted_by_period"]
                ),
                "penalties": {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "gamma": float(gamma),
                    "d_max": int(d_max),
                    "beta_bound": float(beta_bound),
                    "hierarchy_satisfied": bool(beta > beta_bound),
                },
                "successor_degrees": {
                    str(k): int(v)
                    for k, v in successor_count.items()
                },
                "R_results": {},
            }

            instances.append(instance_result)
            instances_by_run[run_index] = instance_result
            R_results = instance_result["R_results"]

            # Immediately save the new instance definition.
            _save_independent_R_checkpoint(
                filename=results_filename,
                configuration=configuration,
                instances=instances,
                R_values=R_values,
            )

        # --------------------------------------------------------------
        # Rebuild the instance QUBO whether it is new or
        # was recovered from the checkpoint.
        # --------------------------------------------------------------
        B_raw = discounted_values(
            block_values,
            periods,
            discount_rate,
        )
        B_normalized, economic_scale = normalize_values(B_raw)
        precedence = build_geometric_precedence(coordinates)

        alpha, beta, gamma = initialize_penalties(
            normalized_values=B_normalized,
            precedence=precedence,
            c_capacity=c_capacity,
            c_validity=c_validity,
            c_precedence=c_precedence,
            hierarchy_epsilon=hierarchy_epsilon,
        )

        successor_count, d_max = successor_degrees(
            precedence=precedence,
            number_blocks=len(block_values),
        )
        beta_bound = float(d_max * gamma)

        if not beta > beta_bound:
            raise RuntimeError(
                "The condition beta > d_max * gamma is not satisfied."
            )

        qubo = build_qubo(
            normalized_values=B_normalized,
            precedence=precedence,
            capacities=capacities,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        if qubo.number_variables != 22:
            raise RuntimeError(
                f"The case no longer contains 22 qubits: "
                f"{qubo.number_variables}"
            )

        optimal_qubo_energy = float(
            instance_result["optimal_qubo_energy"]
        )

        print("\n" + "=" * 78)
        print(
            f"INSTANCE {run_index}/{number_random_instances} "
            f"| block seed={block_seed}"
        )
        print("Values:", block_values)
        print("Optimal NPV:", instance_result["optimal_npv"])
        print("Optimal QUBO energy:", optimal_qubo_energy)
        print("beta > d_max*gamma:", beta > beta_bound)

        for R in R_values:
            R_key = str(int(R))

            if R_key in R_results:
                print(
                    f"  R={R:2d}: already exists in the checkpoint; skipping."
                )
                continue

            # ----------------------------------------------------------
            # INDEPENDENT RUN FOR THIS R.
            # bf_dcqo() initializes bias=np.zeros(...) internally.
            # Therefore, R=R+1 does NOT continue the previous run.
            # ----------------------------------------------------------
            independent_seed = _independent_solver_seed(
                base_seed=solver_seed_base,
                run_index=run_index,
                R=R,
            )

            print(
                f"\n  --- R={R:2d} independent "
                f"| solver seed={independent_seed} ---"
            )

            run_start = time.perf_counter()

            best_bits, best_energy, history, metrics = (
                solve_qubo_with_ibm_simulator(
                    qubo=qubo,
                    r=R,
                    shots=solver_shots,
                    elite_fraction=solver_elite_fraction,
                    lambda_b=solver_lambda_b,
                    eta=solver_eta,
                    fixed_sweep=0,
                    seed=independent_seed,
                    optimal_energy=optimal_qubo_energy,
                    aer_device=aer_device,
                )
            )

            elapsed = time.perf_counter() - run_start

            # ----------------------------------------------------------
            # Save ONLY the RAW statistics of the terminal iteration
            # r=R. Iterations r<R are used only to build
            # the bias within this independent run.
            # ----------------------------------------------------------
            terminal_index = -1

            terminal_probability = float(
                metrics[
                    "raw_optimal_probability_by_iteration"
                ][terminal_index]
            )
            terminal_count = int(
                metrics[
                    "raw_optimal_count_by_iteration"
                ][terminal_index]
            )
            terminal_mean = float(
                metrics[
                    "mean_energy_by_iteration"
                ][terminal_index]
            )
            terminal_std = float(
                metrics[
                    "std_energy_by_iteration"
                ][terminal_index]
            )
            terminal_min = float(
                metrics[
                    "min_energy_by_iteration"
                ][terminal_index]
            )
            terminal_max = float(
                metrics[
                    "max_energy_by_iteration"
                ][terminal_index]
            )
            terminal_unique = int(
                metrics[
                    "unique_states_by_iteration"
                ][terminal_index]
            )

            R_results[R_key] = {
                "R": int(R),
                "internal_r_values": list(range(1, int(R) + 1)),
                "solver_seed": int(independent_seed),
                "shots_terminal_iteration": int(solver_shots),
                "fixed_sweep": 0,
                "raw_optimal_count": terminal_count,
                "raw_optimal_probability": terminal_probability,
                "terminal_mean_energy": terminal_mean,
                "terminal_std_energy": terminal_std,
                "terminal_min_energy": terminal_min,
                "terminal_max_energy": terminal_max,
                "terminal_unique_states": terminal_unique,
                "best_raw_energy_across_run": float(best_energy),
                "best_raw_history": [float(v) for v in history],
                "timing": {
                    "terminal_build_seconds": float(
                        metrics["time_build_seconds"][terminal_index]
                    ),
                    "terminal_transpile_seconds": float(
                        metrics["time_transpile_seconds"][terminal_index]
                    ),
                    "terminal_aer_seconds": float(
                        metrics["time_aer_seconds"][terminal_index]
                    ),
                    "terminal_classical_seconds": float(
                        metrics["time_classical_seconds"][terminal_index]
                    ),
                    "terminal_total_iteration_seconds": float(
                        metrics[
                            "time_total_iteration_seconds"
                        ][terminal_index]
                    ),
                    "full_independent_run_seconds": float(elapsed),
                },
            }

            # Checkpoint after EACH instance-R pair.
            _save_independent_R_checkpoint(
                filename=results_filename,
                configuration=configuration,
                instances=instances,
                R_values=R_values,
            )

            print(
                f"  R={R:2d} completed | "
                f"Popt RAW(r=R)={100.0 * terminal_probability:.6f}% | "
                f"Emean={terminal_mean:.8f} | "
                f"Emin={terminal_min:.8f} | "
                f"time={elapsed:.2f}s"
            )

            del metrics
            del best_bits

    _save_independent_R_checkpoint(
        filename=results_filename,
        configuration=configuration,
        instances=instances,
        R_values=R_values,
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPLETED")
    print("JSON:", results_filename)
    print("Instances:", number_random_instances)
    print("s=0: no greedy refinement")
    print("Aggregation: median + IQR")
    print("=" * 78)


if __name__ == "__main__":
    main()
