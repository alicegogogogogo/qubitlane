"""A tiny OpenQASM 2.0 subset parser.

Only the grammar below is accepted. Everything else is a `validation_error`
that names the 1-based source line.

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg name[n];
    creg name[n];
    h|x|y|z|s|t q[i];
    rx|ry|rz(theta) q[i];
    cx|cz q[control], q[target];
    measure q[i] -> c[j];
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import ParseError, ValidationError

MAX_QUBITS = 16
MAX_CLBITS = 64

SINGLE_QUBIT_GATES = ("h", "x", "y", "z", "s", "t")
PARAMETER_GATES = ("rx", "ry", "rz")
TWO_QUBIT_GATES = ("cx", "cz")
GATES = SINGLE_QUBIT_GATES + PARAMETER_GATES + TWO_QUBIT_GATES

QUBIT = r"q\[(\d+)\]"
CLBIT = r"c\[(\d+)\]"

_HEADER = re.compile(r"^OPENQASM\s+2\.0$")
_INCLUDE = re.compile(r'^include\s+"([^"]*)"$')
_QREG = re.compile(r"^qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[(\d+)\]$")
_CREG = re.compile(r"^creg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[(\d+)\]$")
_PARAMETER_GATE = re.compile(
    rf"^(rx|ry|rz)\s*\(\s*([^()]*)\s*\)\s+{QUBIT}$"
)
_TWO_QUBIT_GATE = re.compile(rf"^(cx|cz)\s+{QUBIT}\s*,\s*{QUBIT}$")
_SINGLE_QUBIT_GATE = re.compile(rf"^(h|x|y|z|s|t)\s+{QUBIT}$")
_MEASURE = re.compile(rf"^measure\s+{QUBIT}\s*->\s*{CLBIT}$")

_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_PI_MULTIPLE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))\s*\*\s*pi$")
_PI_DIVISOR = re.compile(r"^pi\s*/\s*([+-]?(?:\d+\.?\d*|\.\d+))$")
_PI = re.compile(r"^[+-]?pi$")


@dataclass(frozen=True)
class Operation:
    """One executable statement of the circuit."""

    kind: str  # "gate" or "measure"
    name: str  # gate name, or "measure"
    targets: tuple[int, ...]
    angle: float | None = None
    clbit: int | None = None
    line: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "targets": list(self.targets)}
        if self.angle is not None:
            payload["angle"] = self.angle
        if self.clbit is not None:
            payload["clbit"] = self.clbit
        return payload


@dataclass(frozen=True)
class Circuit:
    """A parsed circuit: register sizes plus the ordered operation list."""

    id: str
    qubits: int
    clbits: int
    operations: tuple[Operation, ...]
    qasm: str
    source_lines: int
    measurements: tuple[tuple[int, int], ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.id,
            "qubits": self.qubits,
            "clbits": self.clbits,
            "bit_order": "little_endian",
            "measurements": [
                {"qubit": qubit, "clbit": clbit} for qubit, clbit in self.measurements
            ],
            "operations": [operation.as_dict() for operation in self.operations],
            "source_lines": self.source_lines,
            "qasm": self.qasm,
        }


def identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 100:
        raise ValidationError(f"{field} must be a non-empty string of at most 100 characters")
    if any(character.isspace() for character in value):
        raise ValidationError(f"{field} must not contain whitespace")
    return value


def parse_number(text: str, line: int) -> float:
    """Parse a gate angle, allowing `pi`, `pi/2`, `-pi/4`, and `2*pi` forms."""

    candidate = text.strip()
    if not candidate:
        raise ParseError(f"line {line}: angle is missing")
    sign = 1.0
    if candidate.startswith("+"):
        candidate = candidate[1:].strip()
    elif candidate.startswith("-"):
        candidate = candidate[1:].strip()
        sign = -1.0
    if candidate[:1] in ("+", "-"):
        raise ParseError(f"line {line}: angle {text.strip()!r} has a repeated sign")
    if _PI.match(candidate):
        return sign * math.pi
    multiple = _PI_MULTIPLE.match(candidate)
    if multiple:
        return sign * float(multiple.group(1)) * math.pi
    divisor = _PI_DIVISOR.match(candidate)
    if divisor:
        value = float(divisor.group(1))
        if value == 0:
            raise ParseError(f"line {line}: division by zero in angle expression")
        return sign * math.pi / value
    if not _NUMBER.match(candidate):
        raise ParseError(f"line {line}: angle {text.strip()!r} is not a numeric expression")
    value = float(candidate)
    if not math.isfinite(value):
        raise ParseError(f"line {line}: angle must be finite")
    return sign * value


def statements(qasm: str) -> list[tuple[int, str]]:
    """Split source into `(line number, statement)` pairs, stripping comments."""

    if not isinstance(qasm, str) or not qasm.strip():
        raise ValidationError("qasm must be a non-empty string")
    collected: list[tuple[int, str]] = []
    for number, raw in enumerate(qasm.splitlines(), start=1):
        content = raw.split("//", 1)[0]
        for piece in content.split(";"):
            statement = piece.strip()
            if statement:
                collected.append((number, statement))
    return collected


def parse_circuit(qasm: str, circuit_id: str) -> Circuit:
    """Parse OpenQASM 2.0 text into a `Circuit`, or raise a descriptive error."""

    entries = statements(qasm)
    if not entries:
        raise ValidationError("qasm contains no statements")

    first_line, first = entries[0]
    if not _HEADER.match(first):
        raise ParseError(f"line {first_line}: the first statement must be 'OPENQASM 2.0'")

    index = 1
    if index < len(entries):
        line, text = entries[index]
        include = _INCLUDE.match(text)
        if include:
            if include.group(1) != "qelib1.inc":
                raise ParseError(f'line {line}: only include "qelib1.inc" is supported')
            index += 1

    qubits: int | None = None
    clbits: int | None = None
    for line, text in entries[index:]:
        quantum = _QREG.match(text)
        classical = _CREG.match(text)
        if not quantum and not classical:
            break
        name, size_text = (quantum or classical).group(1), (quantum or classical).group(2)
        size = int(size_text)
        if quantum:
            if name != "q":
                raise ParseError(f"line {line}: the quantum register must be named q")
            if qubits is not None:
                raise ParseError(f"line {line}: qreg q is declared more than once")
            if size < 1:
                raise ParseError(f"line {line}: register size must be at least 1")
            if size > MAX_QUBITS:
                raise ParseError(f"line {line}: at most {MAX_QUBITS} qubits are supported")
            qubits = size
        else:
            if name != "c":
                raise ParseError(f"line {line}: the classical register must be named c")
            if clbits is not None:
                raise ParseError(f"line {line}: creg c is declared more than once")
            if size < 1:
                raise ParseError(f"line {line}: register size must be at least 1")
            if size > MAX_CLBITS:
                raise ParseError(f"line {line}: at most {MAX_CLBITS} classical bits are supported")
            clbits = size
        index += 1

    if qubits is None:
        raise ValidationError("qasm must declare a quantum register, for example 'qreg q[2]'")
    clbits = clbits or 0

    operations: list[Operation] = []
    measurements: list[tuple[int, int]] = []
    for line, text in entries[index:]:
        operation = _parse_operation(text, line, qubits, clbits)
        operations.append(operation)
        if operation.kind == "measure" and operation.clbit is not None:
            measurements.append((operation.targets[0], operation.clbit))

    if not operations:
        raise ValidationError("qasm must contain at least one gate or measure statement")
    return Circuit(
        id=circuit_id,
        qubits=qubits,
        clbits=clbits,
        operations=tuple(operations),
        qasm=qasm,
        source_lines=len(qasm.splitlines()),
        measurements=tuple(measurements),
    )


def _parse_operation(text: str, line: int, qubits: int, clbits: int) -> Operation:
    measure = _MEASURE.match(text)
    if measure:
        if clbits == 0:
            raise ParseError(f"line {line}: measure requires a classical register")
        qubit = _checked_bit(int(measure.group(1)), qubits, "qubit", line)
        clbit = _checked_bit(int(measure.group(2)), clbits, "classical bit", line)
        return Operation("measure", "measure", (qubit,), clbit=clbit, line=line)

    parameter = _PARAMETER_GATE.match(text)
    if parameter:
        angle = parse_number(parameter.group(2), line)
        target = _checked_bit(int(parameter.group(3)), qubits, "qubit", line)
        return Operation("gate", parameter.group(1), (target,), angle=angle, line=line)

    two_qubit = _TWO_QUBIT_GATE.match(text)
    if two_qubit:
        control = _checked_bit(int(two_qubit.group(2)), qubits, "qubit", line)
        target = _checked_bit(int(two_qubit.group(3)), qubits, "qubit", line)
        if control == target:
            raise ParseError(
                f"line {line}: {two_qubit.group(1)} requires two distinct qubits"
            )
        return Operation("gate", two_qubit.group(1), (control, target), line=line)

    single = _SINGLE_QUBIT_GATE.match(text)
    if single:
        target = _checked_bit(int(single.group(2)), qubits, "qubit", line)
        return Operation("gate", single.group(1), (target,), line=line)

    name = text.split("(", 1)[0].split(" ", 1)[0].strip()
    if name in GATES:
        raise ParseError(f"line {line}: malformed {name} statement {text!r}")
    raise ParseError(f"line {line}: unsupported statement or gate {text!r}")


def _checked_bit(value: int, size: int, label: str, line: int) -> int:
    if value >= size:
        raise ParseError(
            f"line {line}: {label} index {value} is out of range for a register of size {size}"
        )
    return value
