"""Exact density-matrix evolution with a gate-level depolarizing channel.

The density matrix is stored as a flat row-major list of complex numbers:
`rho[row * dim + col]`, where `dim = 2^qubits` and the basis index follows the
same little-endian convention as the statevector simulator. After every gate,
each target qubit of that gate undergoes the channel

    rho -> (1 - p) * rho + p / 3 * (X rho X + Y rho Y + Z rho Z)

applied once per participating qubit, in statement order. `measure` statements
do not trigger the channel.
"""

from __future__ import annotations

import cmath
import math

from .errors import ValidationError
from .qasm import Circuit, Operation
from .simulator import TOLERANCE

MAX_NOISE_QUBITS = 8

_SQRT_HALF = math.sqrt(0.5)

_PAULI_X = ((0 + 0j, 1 + 0j), (1 + 0j, 0 + 0j))
_PAULI_Y = ((0 + 0j, -1j), (1j, 0 + 0j))
_PAULI_Z = ((1 + 0j, 0 + 0j), (0 + 0j, -1 + 0j))

_CX = (
    (1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j),
    (0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j),
    (0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j),
    (0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j),
)
_CZ = (
    (1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j),
    (0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j),
    (0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j),
    (0 + 0j, 0 + 0j, 0 + 0j, -1 + 0j),
)


def _gate_matrix(operation: Operation) -> tuple[tuple[complex, complex], ...]:
    """The 2x2 unitary of a single-qubit gate, matching `simulator` exactly."""

    name = operation.name
    if name == "h":
        return (
            (_SQRT_HALF, _SQRT_HALF),
            (_SQRT_HALF, -_SQRT_HALF),
        )
    if name == "x":
        return _PAULI_X
    if name == "y":
        return _PAULI_Y
    if name == "z":
        return _PAULI_Z
    if name == "s":
        return ((1 + 0j, 0 + 0j), (0 + 0j, 1j))
    if name == "t":
        return ((1 + 0j, 0 + 0j), (0 + 0j, cmath.exp(1j * math.pi / 4)))
    angle = operation.angle
    if name == "rx":
        cosine = math.cos(angle / 2)
        sine = complex(0.0, -math.sin(angle / 2))
        return ((cosine, sine), (sine, cosine))
    if name == "ry":
        cosine = math.cos(angle / 2)
        sine = math.sin(angle / 2)
        return ((cosine, -sine), (sine, cosine))
    if name == "rz":
        return (
            (cmath.exp(-1j * angle / 2), 0 + 0j),
            (0 + 0j, cmath.exp(1j * angle / 2)),
        )
    raise ValidationError(f"unsupported gate {name!r}")  # pragma: no cover


def _apply_single(
    rho: list[complex], dim: int, target: int, matrix: tuple[tuple[complex, complex], ...]
) -> None:
    """In place: `rho -> (I (x) U (x) I) rho (I (x) U+ (x) I)` for a 2x2 `U`."""

    (a, b), (c, d) = matrix
    ca, cb, cc, cd = a.conjugate(), b.conjugate(), c.conjugate(), d.conjugate()
    mask = 1 << target
    bases = [index for index in range(dim) if not index & mask]
    for i in bases:
        row0 = i * dim
        row1 = (i | mask) * dim
        for j in bases:
            j1 = j | mask
            r00 = rho[row0 + j]
            r01 = rho[row0 + j1]
            r10 = rho[row1 + j]
            r11 = rho[row1 + j1]
            left0 = r00 * ca + r01 * cb
            left1 = r10 * ca + r11 * cb
            right0 = r00 * cc + r01 * cd
            right1 = r10 * cc + r11 * cd
            rho[row0 + j] = a * left0 + b * left1
            rho[row0 + j1] = a * right0 + b * right1
            rho[row1 + j] = c * left0 + d * left1
            rho[row1 + j1] = c * right0 + d * right1


def _apply_two(
    rho: list[complex],
    dim: int,
    targets: tuple[int, ...],
    matrix: tuple[tuple[complex, ...], ...],
) -> None:
    """In place: conjugate `rho` by a 4x4 gate acting on `targets`.

    Matrix index bit 1 is `targets[0]` (the control), bit 0 is `targets[1]`.
    """

    masks = (1 << targets[0], 1 << targets[1])
    offsets = (0, masks[1], masks[0], masks[0] | masks[1])
    conj = tuple(tuple(matrix[r][c].conjugate() for c in range(4)) for r in range(4))
    bases = [index for index in range(dim) if not index & offsets[3]]
    for i in bases:
        rows = tuple((i | offset) * dim for offset in offsets)
        for j in bases:
            cols = tuple(j | offset for offset in offsets)
            block = [[rho[rows[r] + cols[c]] for c in range(4)] for r in range(4)]
            tmp = [
                [sum(matrix[r][k] * block[k][c] for k in range(4)) for c in range(4)]
                for r in range(4)
            ]
            out = [
                [sum(tmp[r][k] * conj[c][k] for k in range(4)) for c in range(4)]
                for r in range(4)
            ]
            for r in range(4):
                for c in range(4):
                    rho[rows[r] + cols[c]] = out[r][c]


def _depolarize(rho: list[complex], dim: int, target: int, probability: float) -> None:
    """In place: `rho -> (1 - p) rho + p / 3 (X rho X + Y rho Y + Z rho Z)`."""

    if probability == 0.0:
        return
    share = probability / 3.0
    keep = 1.0 - probability
    result = [keep * value for value in rho]
    for pauli in (_PAULI_X, _PAULI_Y, _PAULI_Z):
        work = rho[:]  # `rho` still holds the pre-channel state here
        _apply_single(work, dim, target, pauli)
        for index, value in enumerate(work):
            result[index] += share * value
    rho[:] = result


def simulate_density(circuit: Circuit, probability: float) -> list[complex]:
    """Evolve the density matrix, applying the channel after every gate."""

    dim = 1 << circuit.qubits
    rho = [0j] * (dim * dim)
    rho[0] = 1 + 0j
    for operation in circuit.operations:
        if operation.name == "measure":
            continue
        if operation.name in ("cx", "cz"):
            matrix = _CX if operation.name == "cx" else _CZ
            _apply_two(rho, dim, operation.targets, matrix)
        else:
            _apply_single(rho, dim, operation.targets[0], _gate_matrix(operation))
        for target in operation.targets:
            _depolarize(rho, dim, target, probability)
    return rho


def assert_trace(rho: list[complex], dim: int) -> float:
    """Return the trace defect, raising when it exceeds the tolerance."""

    trace = sum(rho[index * dim + index] for index in range(dim))
    defect = abs(trace - 1.0)
    if defect >= TOLERANCE:
        raise ValidationError(
            f"density matrix is not normalised: |trace minus 1| = {defect:.3e}"
        )
    return defect


def density_probabilities(rho: list[complex], dim: int) -> list[float]:
    """The diagonal of `rho` as real probabilities, clamped at zero."""

    return [max(rho[index * dim + index].real, 0.0) for index in range(dim)]
