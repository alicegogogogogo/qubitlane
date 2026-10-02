"""Statevector simulation for the supported OpenQASM 2.0 subset.

Bit order convention: the statevector index is little-endian. Qubit `q[0]` is
the least significant bit of the basis-state index, so `q[0]=1, q[1]=0` is
index `1`. Basis labels printed by the public API are **big-endian** (most
significant qubit first), which is the conventional OpenQASM readout order:
index `1` of a two-qubit register is printed as `01`.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from .errors import ValidationError
from .qasm import Circuit, Operation

TOLERANCE = 1e-9
ROUND_DIGITS = 12

MAX_NOISE_QUBITS = 8

_SQRT_HALF = math.sqrt(0.5)


def _rotation(imaginary_sine: bool, angle: float) -> complex:
    """`-i sin(angle)` for `rx`, `sin(angle)` for `ry`."""

    sine = math.sin(angle)
    return complex(0.0, -sine) if imaginary_sine else complex(sine, 0.0)


def initial_state(qubits: int) -> list[complex]:
    state = [0j] * (1 << qubits)
    state[0] = 1 + 0j
    return state


def apply_operation(state: list[complex], qubits: int, operation: Operation) -> None:
    """Apply one gate to `state` in place. `qubits` is the register width."""

    if operation.name == "measure":
        return
    if operation.name in ("cx", "cz"):
        _apply_controlled(state, operation.targets[0], operation.targets[1], operation.name)
        return
    if operation.name == "cu3":
        _apply_cu3(state, operation.targets[0], operation.targets[1], operation.angles)
        return
    _apply_single_qubit(state, operation, qubits)


def _require_angles(name: str, angles: tuple[float, ...] | None) -> tuple[float, ...]:
    if angles is None:
        raise ValidationError(
            f"gate {name} has an unbound parameter; "
            "simulate the circuit with parameter bindings"
        )
    return angles


def _u3_entries(angles: tuple[float, ...]) -> tuple[complex, complex, complex, complex]:
    """The four entries `(a, b, c, d)` of the OpenQASM U(theta, phi, lambda).

        U = | a  b | = |  cos(t/2)           -e^{i lambda} sin(t/2) |
            | c  d |   |  e^{i phi} sin(t/2)   e^{i(phi+lambda)} cos(t/2) |
    """

    theta, phi, lam = angles
    half = theta / 2
    cosine = math.cos(half)
    sine = math.sin(half)
    a = complex(cosine, 0.0)
    b = -cmath.exp(1j * lam) * sine
    c = cmath.exp(1j * phi) * sine
    d = cmath.exp(1j * (phi + lam)) * cosine
    return a, b, c, d


def _apply_single_qubit(state: list[complex], operation: Operation, qubits: int) -> None:
    target = operation.targets[0]
    if target >= qubits:  # pragma: no cover - the parser rejects out-of-range qubits
        raise ValidationError(f"qubit {target} is out of range for a {qubits}-qubit circuit")
    mask = 1 << target
    angle = operation.angle
    name = operation.name
    if name in ("rx", "ry", "rz") and angle is None:
        raise ValidationError(
            f"gate {name} has an unbound parameter; "
            "simulate the circuit with parameter bindings"
        )
    if name == "u3":
        u3 = _u3_entries(_require_angles("u3", operation.angles))
    else:
        u3 = None
    for index in range(len(state)):
        if index & mask:
            continue
        low = state[index]
        high = state[index | mask]
        if name == "h":
            state[index] = (low + high) * _SQRT_HALF
            state[index | mask] = (low - high) * _SQRT_HALF
        elif name == "x":
            state[index], state[index | mask] = high, low
        elif name == "y":
            state[index] = -1j * high
            state[index | mask] = 1j * low
        elif name == "z":
            state[index | mask] = -high
        elif name == "s":
            state[index | mask] = 1j * high
        elif name == "t":
            state[index | mask] = cmath.exp(1j * math.pi / 4) * high
        elif name == "rx":
            cosine = math.cos(angle / 2)
            sine = _rotation(True, angle / 2)
            state[index] = cosine * low + sine * high
            state[index | mask] = sine * low + cosine * high
        elif name == "ry":
            cosine = math.cos(angle / 2)
            sine = _rotation(False, angle / 2)
            state[index] = cosine * low - sine * high
            state[index | mask] = sine * low + cosine * high
        elif name == "rz":
            state[index] = cmath.exp(-1j * angle / 2) * low
            state[index | mask] = cmath.exp(1j * angle / 2) * high
        elif name == "u3":
            a, b, c, d = u3
            state[index] = a * low + b * high
            state[index | mask] = c * low + d * high
        else:  # pragma: no cover - the parser rejects unknown gates first
            raise ValidationError(f"unsupported gate {name!r}")


def _apply_cu3(
    state: list[complex],
    control: int,
    target: int,
    angles: tuple[float, ...] | None,
) -> None:
    """Apply U(theta, phi, lambda) to `target` where `control` is 1."""

    a, b, c, d = _u3_entries(_require_angles("cu3", angles))
    control_mask = 1 << control
    target_mask = 1 << target
    for index in range(len(state)):
        if not index & control_mask or index & target_mask:
            continue
        low = state[index]
        high = state[index | target_mask]
        state[index] = a * low + b * high
        state[index | target_mask] = c * low + d * high


def _apply_controlled(state: list[complex], control: int, target: int, name: str) -> None:
    control_mask = 1 << control
    target_mask = 1 << target
    for index in range(len(state)):
        if not index & control_mask or index & target_mask:
            continue
        if name == "cx":
            state[index], state[index | target_mask] = (
                state[index | target_mask],
                state[index],
            )
        else:  # cz
            state[index | target_mask] = -state[index | target_mask]


def simulate(circuit: Circuit) -> list[complex]:
    state = initial_state(circuit.qubits)
    for operation in circuit.operations:
        apply_operation(state, circuit.qubits, operation)
    return state


# -- density-matrix simulation with depolarizing noise ----------------------
#
# The density matrix is flattened into a vector of length `4^qubits` with
# `rho[row * 2^qubits + column]`: the column bits occupy the low half and the
# row bits the high half, so the vector behaves like a statevector of
# `2 * qubits` qubits. Conjugating by a gate then means applying `U` to the
# row (high) qubits and `conj(U)` to the column (low) qubits, because
# `vec(U rho U-dagger) = (U (x) conj(U)) vec(rho)`.

_PAULIS: tuple[tuple[tuple[complex, ...], ...], ...] = (
    ((0, 1), (1, 0)),  # X
    ((0, -1j), (1j, 0)),  # Y
    ((1, 0), (0, -1)),  # Z
)


def _gate_matrix(operation: Operation) -> tuple[tuple[complex, ...], ...]:
    """The unitary of one gate, matching `_apply_single_qubit`/`_apply_controlled`."""

    name = operation.name
    if name in ("rx", "ry", "rz") and operation.angle is None:
        raise ValidationError(
            f"gate {name} has an unbound parameter; "
            "simulate the circuit with parameter bindings"
        )
    if name in ("u3", "cu3") and operation.angles is None:
        raise ValidationError(
            f"gate {name} has an unbound parameter; "
            "simulate the circuit with parameter bindings"
        )
    if name == "h":
        return ((_SQRT_HALF, _SQRT_HALF), (_SQRT_HALF, -_SQRT_HALF))
    if name == "x":
        return ((0, 1), (1, 0))
    if name == "y":
        return ((0, -1j), (1j, 0))
    if name == "z":
        return ((1, 0), (0, -1))
    if name == "s":
        return ((1, 0), (0, 1j))
    if name == "t":
        return ((1, 0), (0, cmath.exp(1j * math.pi / 4)))
    if name == "rx":
        cosine = math.cos(operation.angle / 2)
        sine = _rotation(True, operation.angle / 2)
        return ((cosine, sine), (sine, cosine))
    if name == "ry":
        cosine = math.cos(operation.angle / 2)
        sine = _rotation(False, operation.angle / 2)
        return ((cosine, -sine), (sine, cosine))
    if name == "rz":
        return (
            (cmath.exp(-1j * operation.angle / 2), 0),
            (0, cmath.exp(1j * operation.angle / 2)),
        )
    if name == "u3":
        a, b, c, d = _u3_entries(operation.angles)
        return ((a, b), (c, d))
    if name == "cx":
        return ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))
    if name == "cz":
        return ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, -1))
    if name == "cu3":
        a, b, c, d = _u3_entries(operation.angles)
        return (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, a, b),
            (0, 0, c, d),
        )
    raise ValidationError(f"unsupported gate {name!r}")  # pragma: no cover


def _conjugate(matrix: tuple[tuple[complex, ...], ...]) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(entry.conjugate() for entry in row) for row in matrix)


def _apply_matrix(
    state: list[complex], targets: tuple[int, ...], matrix: tuple[tuple[complex, ...], ...]
) -> None:
    """Apply `matrix` to `targets` of a statevector, in place.

    Bit `len(targets) - 1 - j` of a matrix row/column index corresponds to
    `targets[j]`, so a two-qubit matrix is indexed by `(control, target)`.
    """

    width = len(targets)
    size = 1 << width
    offsets: list[int] = []
    for sub in range(size):
        offset = 0
        for position, target in enumerate(targets):
            if sub & (1 << (width - 1 - position)):
                offset |= 1 << target
        offsets.append(offset)
    covered = 0
    for target in targets:
        covered |= 1 << target
    for index in range(len(state)):
        if index & covered:
            continue
        gathered = [state[index | offset] for offset in offsets]
        for row in range(size):
            entries = matrix[row]
            total = 0j
            for column in range(size):
                coefficient = entries[column]
                if coefficient:
                    total += coefficient * gathered[column]
            state[index | offsets[row]] = total


def _depolarize(rho: list[complex], qubits: int, target: int, probability: float) -> None:
    """Mix `rho` as `(1 - p) rho + p/3 (X rho X + Y rho Y + Z rho Z)` on `target`."""

    original = list(rho)
    keep = 1.0 - probability
    for index in range(len(rho)):
        rho[index] *= keep
    share = probability / 3.0
    row_target = (qubits + target,)
    column_target = (target,)
    for pauli in _PAULIS:
        transformed = list(original)
        _apply_matrix(transformed, row_target, pauli)
        _apply_matrix(transformed, column_target, _conjugate(pauli))
        for index in range(len(rho)):
            rho[index] += share * transformed[index]


def simulate_density(circuit: Circuit, probability: float) -> list[complex]:
    """Exact mixed-state evolution under gate-by-gate depolarizing noise.

    Every gate is conjugated onto the density matrix in statement order, and
    each of the gate's target qubits then passes through the depolarizing
    channel once. `measure` statements are skipped and never trigger noise.
    """

    qubits = circuit.qubits
    rho = [0j] * (1 << (2 * qubits))
    rho[0] = 1 + 0j
    for operation in circuit.operations:
        if operation.name == "measure":
            continue
        matrix = _gate_matrix(operation)
        row_targets = tuple(qubits + target for target in operation.targets)
        _apply_matrix(rho, row_targets, matrix)
        _apply_matrix(rho, operation.targets, _conjugate(matrix))
        for target in operation.targets:
            _depolarize(rho, qubits, target, probability)
    return rho


def density_probabilities(rho: list[complex], qubits: int) -> list[float]:
    """The diagonal of a flattened density matrix: one probability per basis state."""

    width = 1 << qubits
    return [rho[index * width + index].real for index in range(width)]


def assert_normalized_weights(weights: list[float]) -> float:
    """Return the probability-sum defect, raising when it exceeds the tolerance."""

    defect = abs(sum(weights) - 1.0)
    if defect >= TOLERANCE:
        raise ValidationError(
            f"probabilities are not normalised: |sum minus 1| = {defect:.3e}"
        )
    return defect


def norm_squared(state: list[complex]) -> float:
    return sum(abs(amplitude) ** 2 for amplitude in state)


def assert_normalized(state: list[complex]) -> float:
    """Return the normalisation defect, raising when it exceeds the tolerance."""

    defect = abs(norm_squared(state) - 1.0)
    if defect >= TOLERANCE:
        raise ValidationError(
            f"statevector is not normalised: |sum of |alpha|^2 minus 1| = {defect:.3e}"
        )
    return defect


def basis_label(index: int, qubits: int) -> str:
    """Big-endian bit label: qubit `qubits - 1` is the leftmost character."""

    return format(index, f"0{qubits}b")


def clean(value: float) -> float:
    rounded = round(value, ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def clean_complex(value: complex) -> dict[str, float]:
    real = clean(value.real)
    imaginary = clean(value.imag)
    if imaginary == 0:
        imaginary = 0.0
    return {"real": real, "imag": imaginary}


def probabilities(state: list[complex]) -> list[float]:
    return [abs(amplitude) ** 2 for amplitude in state]


def nonzero_amplitudes(circuit: Circuit, state: list[complex]) -> list[dict[str, Any]]:
    """Amplitude entries whose probability is above the reporting threshold."""

    entries: list[dict[str, Any]] = []
    for index, amplitude in enumerate(state):
        probability = abs(amplitude) ** 2
        if probability < TOLERANCE:
            continue
        entries.append(
            {
                "index": index,
                "basis": basis_label(index, circuit.qubits),
                "amplitude": clean_complex(amplitude),
                "probability": clean(probability),
            }
        )
    return entries


def distribution(circuit: Circuit, state: list[complex]) -> dict[str, float]:
    """Every basis state with its probability, keyed by big-endian label."""

    return weight_distribution(circuit, probabilities(state))


def weight_distribution(circuit: Circuit, weights: list[float]) -> dict[str, float]:
    """Every basis state with its probability, keyed by big-endian label."""

    return {
        basis_label(index, circuit.qubits): clean(weight)
        for index, weight in enumerate(weights)
    }


class Sampler:
    """Deterministic measurement sampling driven by an explicit seed.

    The generator is a splitmix64 stream: it depends only on the seed and the
    number of draws, so the same seed always yields the same counts.
    """

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.seed = seed & self.MASK
        self.state = self.seed
        self.draws = 0

    def _next(self) -> float:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self.MASK
        value ^= value >> 31
        self.draws += 1
        return (value >> 11) / float(1 << 53)

    def sample(self, weights: list[float], shots: int, qubits: int) -> dict[str, int]:
        """Draw `shots` basis labels with probability proportional to `weights`."""

        cumulative: list[float] = []
        running = 0.0
        for weight in weights:
            running += weight
            cumulative.append(running)
        total = running
        counts: dict[str, int] = {}
        for _ in range(shots):
            threshold = self._next() * total
            index = _locate(cumulative, threshold)
            label = basis_label(index, qubits)
            counts[label] = counts.get(label, 0) + 1
        return counts


def _locate(cumulative: list[float], threshold: float) -> int:
    low = 0
    high = len(cumulative) - 1
    while low < high:
        middle = (low + high) // 2
        if cumulative[middle] <= threshold:
            low = middle + 1
        else:
            high = middle
    return low
