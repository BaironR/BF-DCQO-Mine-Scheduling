from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
from collections import Counter

import numpy as np
import ast
import json
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.circuit.library import PauliEvolutionGate


"""QUBO model for open-pit mine scheduling (the At formulation).

Experimental workflow:
1. Compute discounted and normalized block values.
2. Generate 2D or 3D geometric precedence constraints.
3. Initialize global penalties and bounded capacity slack variables.
4. Build the binary QUBO and convert it internally to an Ising Hamiltonian.
5. Perform independent BF-DCQO runs for R=1,...,R_max, resetting bias to zero.
6. Use raw samples for elite selection, magnetization, and feedback.
7. Apply greedy refinement only to the best raw shot from each full run,
   evaluating s=0,...,10 sweeps.
8. Record raw optimal sampling probabilities, greedy energies, optimality
   gaps, and optimality indicators.

Qiskit AerSimulator runs the statevector simulation. To reduce memory
usage, aggregated counts are stored rather than individual shots.
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
                f"Se esperaban {self.number_variables} variables, pero se recibieron {x.size}."
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
        raise ValueError("A quadratic term must involve distinct variables.")
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
    """Add penalty * (affine_constant + sum_i a_i*x_i)^2 to the QUBO."""
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
    """Compute B[t,i] = V[i]/(1+r)^t for t=0,...,T-1.
        
            In one-based notation: B_{i,tau}=V_i/(1+r)^(tau-1), tau=1,...,T."""
    if periods <= 0:
        raise ValueError("The number of periods must be positive.")
    if discount_rate < 0.0:
        raise ValueError("The discount rate must be nonnegative.")

    values = np.asarray(block_values, dtype=float)
    return np.vstack(
        [values / ((1.0 + discount_rate) ** t) for t in range(periods)]
    )


def normalize_values(values_by_period: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize by max |B[t,i]| while preserving the signs."""
    matrix = np.asarray(values_by_period, dtype=float)
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if scale <= TOL:
        return matrix.copy(), 1.0
    return matrix / scale, scale


def bounded_slack_weights(capacity: int) -> List[int]:
    """Construct slack weights whose maximum representable value equals capacity.
        
            Examples: C=2 -> [1,1]; C=3 -> [1,2]; C=5 -> [1,2,2];
            C=20 -> [1,2,4,8,5]."""
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
    """Generate immediate geometric precedence relationships.
        
            In 2D, higher y means a higher layer. Each block may have up to
            three direct predecessors at (x-1,y+1), (x,y+1), (x+1,y+1).
            In 3D, use a 3x3 window in the layer directly above.
            Only existing coordinates are included, so boundary blocks may
            have fewer predecessors or none."""
    if not coordinates:
        raise ValueError("At least one coordinate is required.")

    dimension = len(coordinates[0])

    if dimension not in (2, 3):
        raise ValueError("Coordinates must be 2D or 3D.")

    if any(len(coord) != dimension for coord in coordinates):
        raise ValueError(
            "All coordinates must have the same dimension."
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

            # Only three supporting blocks above.
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

    # Specific validation for two-dimensional instances.
    if dimension == 2:
        invalid_blocks = {
            block: predecessors
            for block, predecessors in precedence.items()
            if len(predecessors) > 3
        }

        if invalid_blocks:
            raise RuntimeError(
                "Blocks with more than three predecessors were generated: "
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
        raise ValueError("One capacity value is required per period.")

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
    """Compute d_k: the number of direct successors of block k.
        
            The sufficient local At condition is A_val^(at) > d_k*A_prec^(at).
            For global penalties, impose the conservative bound
            A_val^(at) > d_max*A_prec^(at), where d_max = max_k d_k."""
    degrees = {k: 0 for k in range(number_blocks)}

    for _, predecessors in precedence.items():
        for predecessor in predecessors:
            k = int(predecessor)
            if k not in degrees:
                raise ValueError(
                    f"El predecesor {k} está fuera del rango de bloques."
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
    """Initialize global constraint penalties using normalized economic values.
        
            Each penalty uses the global economic scale max |B[t,i]|.
            At uses alpha (capacity), beta (single extraction), gamma (precedence),
            with beta > d_max*gamma. By uses alpha (capacity), mu (monotonicity),
            and gamma (precedence)."""
    B = np.asarray(normalized_values, dtype=float)

    if B.size == 0:
        raise ValueError("The economic matrix cannot be empty.")

    if min(c_capacity, c_validity, c_precedence) <= 0:
        raise ValueError("Penalty factors must be positive.")

    if hierarchy_epsilon <= 0:
        raise ValueError("hierarchy_epsilon must be positive.")

    periods, number_blocks = B.shape
    economic_reference = float(np.max(np.abs(B)))

    # Base global penalties.
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
    """Construct the At QUBO with one global coefficient for each
            constraint type: economic objective, validity, precedence, and capacity."""
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
    """Build the binary QUBO dictionary expected by Solver01.
        
            The solver converts the binary QUBO into an Ising Hamiltonian internally;
            the model must not be converted to Ising beforehand."""
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
    """Decode an At bitstring, where x[i,t]=1 means block i is
            extracted exactly in period t."""
    bits_array = np.asarray(bits, dtype=int)

    if bits_array.shape != (qubo.number_variables,):
        raise ValueError(
            f"El bitstring tiene {bits_array.size} variables, "
            f"pero el QUBO contiene {qubo.number_variables}."
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
    """Compute the unnormalized economic value (NPV) of a solution
            from the original discounted values, excluding QUBO penalties."""
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
            f"B_raw tiene forma {B_raw.shape}, "
            f"pero se esperaba {expected_shape}."
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


def _prepare_cd(Hp, bias):
    """Return (A_op, Theta) for the biased iteration.

    A_op is the Hermitian generator with U_imp = exp(-i * (-2 Theta) * A_op)
    (docs/BF-DCQO_Implementation.md, Secs. 4-11).  The bias is diagonal, so it
    drops out of O1 = [H_i, H_p] but enters Theta through X = [H_i, O1]."""
    n = Hp.num_qubits

    # Biased initial Hamiltonian  H_i = -sum X_j - sum b_j Z_j
    terms = [(_label({q: "X"}, n), -1.0) for q in range(n)]
    terms += [(_label({q: "Z"}, n), -bias[q]) for q in range(n) if bias[q] != 0.0]
    Hi = SparsePauliOp.from_list(terms).simplify()

    O1 = _commutator(Hi, Hp)                           # anti-Hermitian
    Xop = _commutator(Hi, O1)
    Yop = _commutator(Hp, O1)

    beta1 = _trace_prod(O1, O1).real                   # = -||O1||_F^2 <= 0
    beta2 = _frob(Xop, Xop).real                       # ||X||_F^2
    beta3 = 2.0 * _frob(Xop, Yop).real                 # Tr(X!Y)+Tr(Y!X)
    beta4 = _frob(Yop, Yop).real                       # ||Y||_F^2

    a = beta2 - beta3 + beta4
    b = beta3 - 2.0 * beta2
    c = beta2
    if abs(beta1) < 1e-15 or (abs(a) < 1e-15 and abs(b) < 1e-15 and abs(c) < 1e-15):
        Theta = 0.0
    else:
        Theta = beta1 * _quadratic_integral(a, b, c)

    # A = O1 / (2i) = -0.5j * O1  (Hermitian, so drop residual imaginary noise)
    A_op = (-0.5j * O1).simplify()
    A_op = SparsePauliOp(A_op.paulis, np.real(A_op.coeffs)).simplify()
    return A_op, Theta


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------
def build_circuit(Hp, bias, measure=True):
    """Build the impulse-regime BF-DCQO circuit for a given bias vector.

    A single Trotter step is used: the temporal integral of alpha_1 is already
    contained in Theta (docs/BF-DCQO_Implementation.md, Secs. 10-11)."""
    n = Hp.num_qubits
    A_op, Theta = _prepare_cd(Hp, bias)

    qc = QuantumCircuit(n)

    # State preparation: local ground state of -X_q - b_q Z_q via Ry.
    # Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>, matched to the
    # amplitudes a0 = sqrt((R+b)/2R), a1 = sqrt((R-b)/2R).
    for q in range(n):
        b = bias[q]
        R = np.sqrt(1.0 + b * b)
        a0 = np.sqrt((R + b) / (2.0 * R))
        a1 = np.sqrt((R - b) / (2.0 * R))
        qc.ry(2.0 * np.arctan2(a1, a0), q)

    # Counterdiabatic evolution U_imp = exp(-i (-2 Theta) A_op), realized with a
    # single Trotter step (the default PauliEvolutionGate synthesis).
    if len(A_op) > 0 and abs(Theta) > 1e-15:
        qc.append(PauliEvolutionGate(A_op, time=-2.0 * Theta), range(n))

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
# Incremental greedy refinement for the sweep experiment
# --------------------------------------------------------------------------
def greedy_sweep_history(
    bitstring,
    max_sweeps,
    parsed_qubo,
):
    """Apply greedy refinement incrementally and return the energy after
            each sweep, starting with the unrefined energy at index zero.
            After reaching a local minimum, remaining energies stay constant.
            Also return the final bitstring."""
    bits = np.array(bitstring, dtype=int).copy()
    current_energy = _qubo_energy_from_bits(
        bits,
        parsed_qubo,
    )

    energies = [float(current_energy)]

    for _ in range(max(0, int(max_sweeps))):
        improved = False

        for j in range(bits.size):
            candidate = bits.copy()
            candidate[j] ^= 1

            candidate_energy = _qubo_energy_from_bits(
                candidate,
                parsed_qubo,
            )

            if candidate_energy < current_energy:
                bits = candidate
                current_energy = float(candidate_energy)
                improved = True

        energies.append(float(current_energy))

        if not improved:
            remaining = int(max_sweeps) + 1 - len(energies)
            if remaining > 0:
                energies.extend(
                    [float(current_energy)] * remaining
                )
            break

    if len(energies) < int(max_sweeps) + 1:
        energies.extend(
            [float(current_energy)]
            * (int(max_sweeps) + 1 - len(energies))
        )

    return bits, energies



def _counts_to_weighted_states(
    counts,
    number_qubits,
    parsed_qubo,
):
    """Convert Qiskit counts into unique state indices, multiplicities,
            and energies without expanding one entry per shot. This supports
            large sample counts with reduced memory use."""
    indices: List[int] = []
    weights: List[int] = []

    for bitstring, count in counts.items():
        clean_bitstring = str(bitstring).replace(" ", "")
        clean_bitstring = clean_bitstring.zfill(int(number_qubits))
        indices.append(int(clean_bitstring, 2))
        weights.append(int(count))

    state_indices = np.asarray(indices, dtype=np.int64)
    state_weights = np.asarray(weights, dtype=np.int64)

    if state_indices.size == 0:
        raise RuntimeError("The simulator returned no results.")

    energies = np.asarray(
        [
            _qubo_energy_from_index(
                int(idx),
                int(number_qubits),
                parsed_qubo,
            )
            for idx in state_indices
        ],
        dtype=float,
    )

    return state_indices, state_weights, energies


def _weighted_energy_statistics(
    energies: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute count-weighted mean, standard deviation, minimum, and maximum."""
    energies = np.asarray(energies, dtype=float)
    weights = np.asarray(weights, dtype=np.int64)

    total = int(np.sum(weights))
    if total <= 0:
        raise ValueError("The sum of counts must be positive.")

    mean = float(
        np.sum(energies * weights, dtype=np.float64)
        / float(total)
    )

    variance = float(
        np.sum(
            weights * (energies - mean) ** 2,
            dtype=np.float64,
        )
        / float(total)
    )

    return (
        mean,
        float(np.sqrt(max(variance, 0.0))),
        float(np.min(energies)),
        float(np.max(energies)),
    )


def _weighted_optimal_count(
    energies: np.ndarray,
    weights: np.ndarray,
    optimal_energy: float | None,
    optimal_tolerance: float,
) -> int:
    """Count optimal shots directly from aggregated counts."""
    if optimal_energy is None:
        return 0

    mask = np.isclose(
        np.asarray(energies, dtype=float),
        float(optimal_energy),
        atol=float(optimal_tolerance),
        rtol=0.0,
    )

    return int(
        np.sum(
            np.asarray(weights, dtype=np.int64)[mask],
            dtype=np.int64,
        )
    )


def _energy_frequency_dict(
    energies: np.ndarray,
    weights: np.ndarray,
    decimals: int = 12,
) -> Dict[str, int]:
    """Summarize the energy distribution without storing individual shots.
            Accumulate equal rounded energies with their multiplicities."""
    counter: Counter = Counter()

    for energy, count in zip(
        np.asarray(energies, dtype=float),
        np.asarray(weights, dtype=np.int64),
    ):
        counter[round(float(energy), int(decimals))] += int(count)

    return {
        f"{energy:.{int(decimals)}f}": int(count)
        for energy, count in sorted(
            counter.items(),
            key=lambda item: item[0],
        )
    }


def _elite_magnetization_from_counts(
    state_indices: np.ndarray,
    state_weights: np.ndarray,
    energies: np.ndarray,
    number_qubits: int,
    elite_fraction: float,
) -> Tuple[np.ndarray, int]:
    """Compute elite-set magnetization without expanding all samples.
            Select the lowest-energy shots up to floor(elite_fraction*N_s),
            with at least one shot. If the boundary falls within one state
            count, take only the required multiplicity."""
    if not (0.0 < float(elite_fraction) <= 1.0):
        raise ValueError("elite_fraction must be in the interval (0, 1].")

    state_indices = np.asarray(state_indices, dtype=np.int64)
    state_weights = np.asarray(state_weights, dtype=np.int64)
    energies = np.asarray(energies, dtype=float)

    total_shots = int(np.sum(state_weights))
    n_elite = max(
        1,
        int(float(elite_fraction) * total_shots),
    )

    # Primary sort: energy. Secondary sort: index for reproducibility.
    order = np.lexsort(
        (
            state_indices,
            energies,
        )
    )

    remaining = int(n_elite)
    magnetization_sum = np.zeros(
        int(number_qubits),
        dtype=float,
    )

    for position in order:
        if remaining <= 0:
            break

        available = int(state_weights[position])
        take = min(available, remaining)

        bits = _index_to_bits(
            int(state_indices[position]),
            int(number_qubits),
        )

        magnetization_sum += (
            float(take)
            * (1.0 - 2.0 * bits)
        )

        remaining -= take

    if remaining != 0:
        raise RuntimeError(
            "Could not construct the elite set from the available counts."
        )

    magnetization = (
        magnetization_sum
        / float(n_elite)
    )

    return magnetization, int(n_elite)


# --------------------------------------------------------------------------
# 2. BF-DCQO in the impulse regime (one run with budget R)
# --------------------------------------------------------------------------
def bf_dcqo(
    Hp,
    parsed_qubo,
    r,
    shots=1_000_000,
    elite_fraction=0.1,
    lambda_b=1.0,
    eta=0.5,
    seed=None,
    return_history=False,
    optimal_energy=None,
    optimal_tolerance=1e-9,
    aer_device="CPU",
):
    """Run one BF-DCQO experiment with maximum iteration budget R=r.
            Start with zero bias and perform r internal iterations. Use only
            raw shots for energy evaluation, elite selection, magnetization,
            and bias updates. Greedy refinement is applied after the run,
            to its best raw shot. Process aggregated counts to save memory."""
    n = Hp.num_qubits

    backend = AerSimulator(
        method="statevector",
        device=str(aer_device),
    )

    bias = np.zeros(n)
    best_bits = None
    best_energy = np.inf
    history: List[float] = []

    metrics = {
        "mean_energy_by_iteration": [],
        "std_energy_by_iteration": [],
        "min_energy_by_iteration": [],
        "max_energy_by_iteration": [],
        "optimal_shots_by_iteration": [],
        "optimal_probability_by_iteration": [],
        "elite_size_by_iteration": [],
        "total_optimal_shots": 0,
        "total_shots": 0,
        "optimal_energy_reference": (
            None
            if optimal_energy is None
            else float(optimal_energy)
        ),
        "terminal_energy_frequencies": {},
        "terminal_optimal_shots": 0,
        "terminal_optimal_probability": 0.0,
    }

    number_iterations = max(1, int(r))

    for iteration in range(number_iterations):
        qc = build_circuit(
            Hp,
            bias,
            measure=True,
        )

        qc_decomposed = qc.decompose(
            reps=10
        )

        qc_aer = transpile(
            qc_decomposed,
            backend=backend,
            optimization_level=0,
        )

        simulator_seed = (
            None
            if seed is None
            else int(seed) + int(iteration)
        )

        result = backend.run(
            qc_aer,
            shots=int(shots),
            seed_simulator=simulator_seed,
        ).result()

        counts = result.get_counts(qc_aer)

        (
            state_indices,
            state_weights,
            state_energies,
        ) = _counts_to_weighted_states(
            counts=counts,
            number_qubits=n,
            parsed_qubo=parsed_qubo,
        )

        returned_shots = int(
            np.sum(
                state_weights,
                dtype=np.int64,
            )
        )

        if returned_shots != int(shots):
            raise RuntimeError(
                f"Aer devolvió {returned_shots} shots, "
                f"pero se esperaban {shots}."
            )

        (
            mean_energy,
            std_energy,
            min_energy,
            max_energy,
        ) = _weighted_energy_statistics(
            energies=state_energies,
            weights=state_weights,
        )

        metrics["mean_energy_by_iteration"].append(
            mean_energy
        )
        metrics["std_energy_by_iteration"].append(
            std_energy
        )
        metrics["min_energy_by_iteration"].append(
            min_energy
        )
        metrics["max_energy_by_iteration"].append(
            max_energy
        )

        optimal_shots = _weighted_optimal_count(
            energies=state_energies,
            weights=state_weights,
            optimal_energy=optimal_energy,
            optimal_tolerance=optimal_tolerance,
        )

        optimal_probability = (
            float(optimal_shots) / float(shots)
            if optimal_energy is not None
            else 0.0
        )

        metrics["optimal_shots_by_iteration"].append(
            int(optimal_shots)
        )
        metrics["optimal_probability_by_iteration"].append(
            float(optimal_probability)
        )
        metrics["total_optimal_shots"] += int(
            optimal_shots
        )
        metrics["total_shots"] += int(
            shots
        )

        # Best raw shot found across the ENTIRE run so far.
        local_best_position = int(
            np.argmin(state_energies)
        )

        local_best_energy = float(
            state_energies[local_best_position]
        )

        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_bits = _index_to_bits(
                int(
                    state_indices[
                        local_best_position
                    ]
                ),
                n,
            )

        history.append(
            float(best_energy)
        )

        # BF-DCQO feedback using RAW shots only.
        (
            magnetization,
            n_elite,
        ) = _elite_magnetization_from_counts(
            state_indices=state_indices,
            state_weights=state_weights,
            energies=state_energies,
            number_qubits=n,
            elite_fraction=elite_fraction,
        )

        metrics["elite_size_by_iteration"].append(
            int(n_elite)
        )

        # Preserve the smoothed update from the original implementation.
        bias = (
            (1.0 - eta) * bias
            + eta * lambda_b * magnetization
        )

        # Retain the complete distribution only for the TERMINAL iteration
        # of this R run. This avoids exporting hundreds of millions of shots.
        if iteration == number_iterations - 1:
            metrics[
                "terminal_energy_frequencies"
            ] = _energy_frequency_dict(
                energies=state_energies,
                weights=state_weights,
            )

            metrics[
                "terminal_optimal_shots"
            ] = int(optimal_shots)

            metrics[
                "terminal_optimal_probability"
            ] = float(optimal_probability)

        line = (
            f"Run R={number_iterations:2d} | "
            f"internal iteration r={iteration + 1:2d}/{number_iterations}: "
            f"mejor energía acumulada = {best_energy:.12f}, "
            f"media = {mean_energy:.12f}, "
            f"std = {std_energy:.12f}"
        )

        if optimal_energy is not None:
            line += (
                f", Popt_raw = "
                f"{100.0 * optimal_probability:.6f}%"
            )

        print(line)

    metrics["global_raw_optimal_probability"] = (
        float(metrics["total_optimal_shots"])
        / float(metrics["total_shots"])
        if (
            optimal_energy is not None
            and int(metrics["total_shots"]) > 0
        )
        else 0.0
    )

    if best_bits is None:
        raise RuntimeError(
            "BF-DCQO finished without producing a best solution."
        )

    if return_history:
        return (
            np.asarray(best_bits, dtype=int),
            history,
            metrics,
        )

    return np.asarray(best_bits, dtype=int)


# --------------------------------------------------------------------------
# 3. Greedy descent refinement
# --------------------------------------------------------------------------
def greedy_descent(bitstring, s, parsed_qubo):
    """Apply greedy refinement by directly evaluating QUBO energies."""
    bits = np.array(bitstring, dtype=int).copy()
    current_energy = _qubo_energy_from_bits(
        bits,
        parsed_qubo,
    )

    for _ in range(max(0, int(s))):
        improved = False

        for j in range(bits.size):
            candidate = bits.copy()
            candidate[j] ^= 1

            candidate_energy = _qubo_energy_from_bits(
                candidate,
                parsed_qubo,
            )

            if candidate_energy < current_energy:
                bits = candidate
                current_energy = candidate_energy
                improved = True

        if not improved:
            break

    return bits


def _merge_frequency_dicts(
    destination: Counter,
    source: Mapping[str, int],
) -> None:
    """Accumulate an {energy: count} mapping into a Counter."""
    for energy, count in source.items():
        destination[str(energy)] += int(count)


def run_independent_bf_dcqo_experiments(
    qubo: QUBO,
    max_R: int = 30,
    shots: int = 1_000_000,
    elite_fraction: float = 0.1,
    lambda_b: float = 1.0,
    eta: float = 0.5,
    sweep_values: Sequence[int] = tuple(range(0, 11)),
    seed: int | None = 0,
    optimal_energy: float | None = None,
    optimal_tolerance: float = 1e-9,
    aer_device: str = "CPU",
) -> Dict[str, object]:
    """Execute independent BF-DCQO runs for R=1,...,max_R.
            Each run resets the bias, history, best solution, and random seed.
            After each run, refine only its best raw sample for each requested
            sweep count. Track raw optimal probabilities and refined energies,
            optimality gaps, and optimality indicators for every (R,s)."""
    if int(max_R) <= 0:
        raise ValueError("max_R must be positive.")

    sweep_values = sorted(
        {
            int(s)
            for s in sweep_values
        }
    )

    if not sweep_values:
        raise ValueError(
            "sweep_values must contain at least one value."
        )

    if sweep_values[0] != 0:
        raise ValueError(
            "sweep_values must include 0."
        )

    if any(s < 0 for s in sweep_values):
        raise ValueError(
            "Sweeps must be nonnegative."
        )

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
            f"El solver detectó {n} qubits, "
            f"pero el QUBO contiene {qubo.number_variables} variables."
        )

    runs: List[Dict[str, object]] = []

    raw_terminal_probability_by_R: List[float] = []
    raw_global_probability_by_R: List[float] = []
    best_raw_energy_by_R: List[float] = []

    greedy_energy_matrix: List[List[float]] = []
    greedy_gap_matrix: List[List[float]] = []
    greedy_optimal_indicator_matrix: List[List[int]] = []

    aggregate_terminal_frequencies: Counter = Counter()

    overall_best_raw_energy = np.inf
    overall_best_raw_bits = None

    overall_best_refined_energy = np.inf
    overall_best_refined_bits = None
    overall_best_refined_R = None
    overall_best_refined_s = None

    max_sweeps = max(
        sweep_values
    )

    for R in range(1, int(max_R) + 1):
        print(
            "\n"
            + "=" * 78
        )
        print(
            f"INICIO CORRIDA INDEPENDIENTE R={R}"
        )
        print(
            "=" * 78
        )

        # Use distinct seeds so R=1, R=2, ... are
        # independent and reproducible runs. Within each run,
        # bf_dcqo adds the internal iteration index to this seed.
        run_seed = (
            None
            if seed is None
            else int(seed) + 100_000 * int(R)
        )

        (
            best_raw_bits,
            history,
            metrics,
        ) = bf_dcqo(
            Hp=Hp,
            parsed_qubo=parsed_qubo,
            r=R,
            shots=shots,
            elite_fraction=elite_fraction,
            lambda_b=lambda_b,
            eta=eta,
            seed=run_seed,
            return_history=True,
            optimal_energy=optimal_energy,
            optimal_tolerance=optimal_tolerance,
            aer_device=aer_device,
        )

        best_raw_energy = float(
            _qubo_energy_from_bits(
                best_raw_bits,
                parsed_qubo,
            )
        )

        # Apply greedy ONLY to the best raw shot from the entire run.
        (
            best_refined_bits_at_max_s,
            full_sweep_energies,
        ) = greedy_sweep_history(
            bitstring=best_raw_bits,
            max_sweeps=max_sweeps,
            parsed_qubo=parsed_qubo,
        )

        selected_sweep_energies = [
            float(
                full_sweep_energies[s]
            )
            for s in sweep_values
        ]

        if optimal_energy is not None:
            selected_gaps = [
                float(
                    energy
                    - float(optimal_energy)
                )
                for energy in selected_sweep_energies
            ]

            selected_optimal_indicators = [
                int(
                    np.isclose(
                        energy,
                        float(optimal_energy),
                        atol=float(optimal_tolerance),
                        rtol=0.0,
                    )
                )
                for energy in selected_sweep_energies
            ]
        else:
            selected_gaps = [
                float("nan")
                for _ in selected_sweep_energies
            ]
            selected_optimal_indicators = [
                0
                for _ in selected_sweep_energies
            ]

        terminal_probability = float(
            metrics[
                "terminal_optimal_probability"
            ]
        )

        global_probability = float(
            metrics[
                "global_raw_optimal_probability"
            ]
        )

        raw_terminal_probability_by_R.append(
            terminal_probability
        )

        raw_global_probability_by_R.append(
            global_probability
        )

        best_raw_energy_by_R.append(
            best_raw_energy
        )

        greedy_energy_matrix.append(
            selected_sweep_energies
        )

        greedy_gap_matrix.append(
            selected_gaps
        )

        greedy_optimal_indicator_matrix.append(
            selected_optimal_indicators
        )

        _merge_frequency_dicts(
            aggregate_terminal_frequencies,
            metrics[
                "terminal_energy_frequencies"
            ],
        )

        if best_raw_energy < overall_best_raw_energy:
            overall_best_raw_energy = best_raw_energy
            overall_best_raw_bits = np.asarray(
                best_raw_bits,
                dtype=int,
            ).copy()

        # Greedy only decreases energy. Record the first sweep that
        # reaches the minimum energy on this trajectory.
        run_best_position = int(
            np.argmin(
                np.asarray(
                    selected_sweep_energies,
                    dtype=float,
                )
            )
        )
        run_best_refined_energy = float(
            selected_sweep_energies[
                run_best_position
            ]
        )
        run_best_refined_s = int(
            sweep_values[
                run_best_position
            ]
        )

        if run_best_refined_energy < overall_best_refined_energy:
            overall_best_refined_energy = (
                run_best_refined_energy
            )
            overall_best_refined_bits = np.asarray(
                best_refined_bits_at_max_s,
                dtype=int,
            ).copy()
            overall_best_refined_R = int(R)
            overall_best_refined_s = int(
                run_best_refined_s
            )

        run_record = {
            "R": int(R),
            "seed": (
                None
                if run_seed is None
                else int(run_seed)
            ),
            "internal_iterations": int(R),
            "shots_per_iteration": int(shots),
            "total_shots_in_run": int(
                R * int(shots)
            ),
            "best_raw_energy": float(
                best_raw_energy
            ),
            "best_raw_bitstring": "".join(
                str(int(bit))
                for bit in best_raw_bits
            ),
            "best_raw_energy_history": [
                float(v)
                for v in history
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
            "raw_optimal_shots_by_iteration": [
                int(v)
                for v in metrics[
                    "optimal_shots_by_iteration"
                ]
            ],
            "raw_optimal_probability_by_iteration": [
                float(v)
                for v in metrics[
                    "optimal_probability_by_iteration"
                ]
            ],
            "terminal_raw_optimal_shots": int(
                metrics[
                    "terminal_optimal_shots"
                ]
            ),
            "terminal_raw_optimal_probability": float(
                terminal_probability
            ),
            "global_raw_optimal_shots": int(
                metrics[
                    "total_optimal_shots"
                ]
            ),
            "global_raw_optimal_probability": float(
                global_probability
            ),
            "elite_size_by_iteration": [
                int(v)
                for v in metrics[
                    "elite_size_by_iteration"
                ]
            ],
            "greedy_sweep_values": [
                int(s)
                for s in sweep_values
            ],
            "best_shot_greedy_energies": [
                float(v)
                for v in selected_sweep_energies
            ],
            "best_shot_greedy_gaps": [
                float(v)
                for v in selected_gaps
            ],
            "best_shot_greedy_optimal_indicator": [
                int(v)
                for v in selected_optimal_indicators
            ],
            "best_refined_energy": float(
                run_best_refined_energy
            ),
            "first_sweep_reaching_best_refined_energy": int(
                run_best_refined_s
            ),
            "best_bitstring_after_max_sweep": "".join(
                str(int(bit))
                for bit in best_refined_bits_at_max_s
            ),
            "terminal_energy_frequencies": dict(
                metrics[
                    "terminal_energy_frequencies"
                ]
            ),
        }

        runs.append(
            run_record
        )

        print(
            f"FIN R={R}: "
            f"E_best_raw={best_raw_energy:.12f}, "
            f"Popt_raw_terminal="
            f"{100.0 * terminal_probability:.6f}%, "
            f"E_best_greedy(s={sweep_values[-1]})="
            f"{run_best_refined_energy:.12f}"
        )

    if overall_best_raw_bits is None:
        raise RuntimeError(
            "No raw solution was obtained."
        )

    if overall_best_refined_bits is None:
        raise RuntimeError(
            "No refined solution was obtained."
        )

    total_quantum_shots = int(
        int(shots)
        * int(max_R)
        * (int(max_R) + 1)
        // 2
    )

    return {
        "formulation": "At",
        "max_R": int(max_R),
        "R_values": list(
            range(
                1,
                int(max_R) + 1,
            )
        ),
        "shots_per_iteration": int(shots),
        "total_quantum_iterations": int(
            int(max_R)
            * (int(max_R) + 1)
            // 2
        ),
        "total_quantum_shots": int(
            total_quantum_shots
        ),
        "elite_fraction": float(
            elite_fraction
        ),
        "lambda_b": float(
            lambda_b
        ),
        "eta": float(
            eta
        ),
        "optimal_energy": (
            None
            if optimal_energy is None
            else float(optimal_energy)
        ),
        "optimal_tolerance": float(
            optimal_tolerance
        ),
        "sweep_values": [
            int(s)
            for s in sweep_values
        ],
        "raw_terminal_optimal_probability_by_R": [
            float(v)
            for v in raw_terminal_probability_by_R
        ],
        "raw_global_optimal_probability_by_R": [
            float(v)
            for v in raw_global_probability_by_R
        ],
        "best_raw_energy_by_R": [
            float(v)
            for v in best_raw_energy_by_R
        ],
        "best_shot_greedy_energy_matrix": [
            [float(v) for v in row]
            for row in greedy_energy_matrix
        ],
        "best_shot_greedy_gap_matrix": [
            [float(v) for v in row]
            for row in greedy_gap_matrix
        ],
        "best_shot_greedy_optimal_indicator_matrix": [
            [int(v) for v in row]
            for row in greedy_optimal_indicator_matrix
        ],
        "aggregated_terminal_energy_frequencies": {
            str(energy): int(count)
            for energy, count in sorted(
                aggregate_terminal_frequencies.items(),
                key=lambda item: float(item[0]),
            )
        },
        "overall_best_raw_energy": float(
            overall_best_raw_energy
        ),
        "overall_best_raw_bitstring": "".join(
            str(int(bit))
            for bit in overall_best_raw_bits
        ),
        "overall_best_refined_energy": float(
            overall_best_refined_energy
        ),
        "overall_best_refined_bitstring": "".join(
            str(int(bit))
            for bit in overall_best_refined_bits
        ),
        "overall_best_refined_R": int(
            overall_best_refined_R
        ),
        "overall_best_refined_s": int(
            overall_best_refined_s
        ),
        "runs": runs,
    }


def export_independent_results(
    filename: str,
    results: Mapping[str, object],
) -> None:
    """Export compact results from independent runs to JSON."""
    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )


def main() -> None:
    block_values = [
        20,
        -45,
        60,
        -30,
        25,
        120,
        320,
        220,
        500,
    ]

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

    # ==================================================================
    # Experimental configuration
    # ==================================================================
    # Run 30 independent experiments:
    #   R=1, R=2, ..., R=30.
    # Each run resets the bias to zero.
    solver_max_R = 30

    # Requested N_s for the new experiments.
    # To use a different maximum later, change only this line.
    n_shots = 1_000_000

    solver_elite_fraction = 0.1
    solver_lambda_b = 1.0
    solver_eta = 0.5

    # Apply greedy ONLY to the best raw shot of each R run.
    solver_sweeps = list(
        range(0, 11)
    )

    # Reproducible base seed. Each R has a different seed.
    solver_seed = 0

    aer_device = "CPU"

    optimal_qubo_energy = -1.0266666666666666
    optimal_tolerance = 1e-9

    # 1. Economic values.
    B_raw = discounted_values(
        block_values=block_values,
        periods=periods,
        discount_rate=discount_rate,
    )

    B_normalized, economic_scale = normalize_values(
        B_raw
    )

    # 2. Geometric precedences.
    precedence = build_geometric_precedence(
        coordinates
    )

    # 3. Global penalties for the At formulation.
    alpha, beta, gamma = initialize_penalties(
        normalized_values=B_normalized,
        precedence=precedence,
        c_capacity=1.5,
        c_validity=1.5,
        c_precedence=1.5,
    )

    successor_count, d_max = successor_degrees(
        precedence=precedence,
        number_blocks=len(
            block_values
        ),
    )

    beta_bound = d_max * gamma

    # 4. QUBO At.
    qubo = build_qubo(
        normalized_values=B_normalized,
        precedence=precedence,
        capacities=capacities,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    print(
        "Discounted values B[t,i]:"
    )
    print(
        np.round(
            B_raw,
            6,
        )
    )

    print(
        "\nNormalized values:"
    )
    print(
        np.round(
            B_normalized,
            6,
        )
    )

    print(
        "\nEconomic scale B_max:",
        round(
            economic_scale,
            6,
        ),
    )

    print(
        "Precedences:",
        precedence,
    )

    print(
        "\nInitial global penalties:"
    )
    print(
        "Alpha (capacity):",
        round(
            alpha,
            6,
        ),
    )
    print(
        "Gamma (precedence):",
        round(
            gamma,
            6,
        ),
    )
    print(
        "d_k by block:",
        successor_count,
    )
    print(
        "d_max:",
        d_max,
    )
    print(
        "Paper bound for Beta: Beta >",
        round(
            beta_bound,
            6,
        ),
    )
    print(
        "Beta (single extraction):",
        round(
            beta,
            6,
        ),
    )
    print(
        "Hierarchy Beta > d_max * Gamma:",
        bool(
            beta > beta_bound
        ),
    )

    print(
        "\nSlack weights:",
        qubo.slack_weights,
    )

    print(
        "Total number of variables/qubits:",
        qubo.number_variables,
    )

    total_quantum_iterations = (
        solver_max_R
        * (solver_max_R + 1)
        // 2
    )

    total_requested_shots = (
        total_quantum_iterations
        * n_shots
    )

    print(
        "\n=== EXPERIMENTAL CONFIGURATION ==="
    )
    print(
        f"Independent runs: R=1..{solver_max_R}"
    )
    print(
        f"Iteraciones BF-DCQO totales: {total_quantum_iterations}"
    )
    print(
        f"N_s per iteration: {n_shots:,}"
    )
    print(
        f"Shots totales solicitados: {total_requested_shots:,}"
    )
    print(
        "Greedy: only on the best raw shot of each run."
    )
    print(
        "Evaluated greedy sweeps:",
        solver_sweeps,
    )

    results = run_independent_bf_dcqo_experiments(
        qubo=qubo,
        max_R=solver_max_R,
        shots=n_shots,
        elite_fraction=solver_elite_fraction,
        lambda_b=solver_lambda_b,
        eta=solver_eta,
        sweep_values=solver_sweeps,
        seed=solver_seed,
        optimal_energy=optimal_qubo_energy,
        optimal_tolerance=optimal_tolerance,
        aer_device=aer_device,
    )

    overall_best_raw_bits = np.asarray(
        [
            int(bit)
            for bit in results[
                "overall_best_raw_bitstring"
            ]
        ],
        dtype=int,
    )

    overall_best_refined_bits = np.asarray(
        [
            int(bit)
            for bit in results[
                "overall_best_refined_bitstring"
            ]
        ],
        dtype=int,
    )

    decoded_raw = decode_at_solution(
        bits=overall_best_raw_bits,
        qubo=qubo,
        number_blocks=len(
            block_values
        ),
        periods=periods,
    )

    decoded_refined = decode_at_solution(
        bits=overall_best_refined_bits,
        qubo=qubo,
        number_blocks=len(
            block_values
        ),
        periods=periods,
    )

    economic_raw = calculate_economic_value(
        decoded_solution=decoded_raw,
        discounted_values_raw=B_raw,
    )

    economic_refined = calculate_economic_value(
        decoded_solution=decoded_refined,
        discounted_values_raw=B_raw,
    )

    results["overall_best_raw_economic_value"] = float(
        economic_raw[
            "total_value"
        ]
    )

    results[
        "overall_best_refined_economic_value"
    ] = float(
        economic_refined[
            "total_value"
        ]
    )

    results_filename = Path(__file__).resolve().parents[2] / "data" / "At_results_independent_R.json"

    export_independent_results(
        filename=results_filename,
        results=results,
    )

    print(
        "\n=== OVERALL RESULT ==="
    )
    print(
        "Best raw energy:",
        results[
            "overall_best_raw_energy"
        ],
    )
    print(
        "Best raw bitstring:",
        results[
            "overall_best_raw_bitstring"
        ],
    )
    print(
        "NPV raw:",
        f"${economic_raw['total_value']:,.6f} USD",
    )

    print(
        "\nBest energy after greedy:",
        results[
            "overall_best_refined_energy"
        ],
    )
    print(
        "Best bitstring after greedy:",
        results[
            "overall_best_refined_bitstring"
        ],
    )
    print(
        "Obtained at:",
        f"R={results['overall_best_refined_R']}, "
        f"s={results['overall_best_refined_s']}",
    )
    print(
        "Refined NPV:",
        f"${economic_refined['total_value']:,.6f} USD",
    )

    print(
        "\nResults exported to:",
        results_filename,
    )

    print(
        "\nData available for plotting:"
    )
    print(
        "1. raw_terminal_optimal_probability_by_R"
    )
    print(
        "2. raw_global_optimal_probability_by_R"
    )
    print(
        "3. best_shot_greedy_energy_matrix"
    )
    print(
        "4. best_shot_greedy_gap_matrix"
    )
    print(
        "5. best_shot_greedy_optimal_indicator_matrix"
    )


if __name__ == "__main__":
    main()
