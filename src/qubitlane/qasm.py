"""A tiny OpenQASM 2.0 subset parser.

Only the grammar below is accepted. Everything else is a `validation_error`
that names the 1-based source line.

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg name[n];
    creg name[n];
    h|x|y|z|s|t q[i];
    rx|ry|rz(theta) q[i];
    u3(theta, phi, lambda) q[i];
    cu3(theta, phi, lambda) q[control], q[target];
    cx|cz q[control], q[target];
    measure q[i] -> c[j];

Gate angles may also be parameter expressions: identifiers such as `theta`
mixed with numbers, `pi`, and the `+ - * /` operators, for example
`rx(theta/2 + pi)` — such circuits are executed after binding every
parameter to a concrete value.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import ParamValueInvalidError, ParseError, ValidationError

MAX_QUBITS = 16
MAX_CLBITS = 64

SINGLE_QUBIT_GATES = ("h", "x", "y", "z", "s", "t")
PARAMETER_GATES = ("rx", "ry", "rz")
U3_GATES = ("u3", "cu3")
TWO_QUBIT_GATES = ("cx", "cz")
GATES = SINGLE_QUBIT_GATES + PARAMETER_GATES + U3_GATES + TWO_QUBIT_GATES

QUBIT = r"q\[(\d+)\]"
CLBIT = r"c\[(\d+)\]"

_HEADER = re.compile(r"^OPENQASM\s+2\.0$")
_INCLUDE = re.compile(r'^include\s+"([^"]*)"$')
_QREG = re.compile(r"^qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[(\d+)\]$")
_CREG = re.compile(r"^creg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[(\d+)\]$")
_PARAMETER_GATE = re.compile(
    rf"^(rx|ry|rz)\s*\(\s*([^()]*)\s*\)\s+{QUBIT}$"
)
_U3_GATE = re.compile(rf"^u3\s*\(\s*([^()]*?)\s*\)\s+{QUBIT}$")
_CU3_GATE = re.compile(rf"^cu3\s*\(\s*([^()]*?)\s*\)\s+{QUBIT}\s*,\s*{QUBIT}$")
_TWO_QUBIT_GATE = re.compile(rf"^(cx|cz)\s+{QUBIT}\s*,\s*{QUBIT}$")
_SINGLE_QUBIT_GATE = re.compile(rf"^(h|x|y|z|s|t)\s+{QUBIT}$")
_MEASURE = re.compile(rf"^measure\s+{QUBIT}\s*->\s*{CLBIT}$")

_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_PI_MULTIPLE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))\s*\*\s*pi$")
_PI_DIVISOR = re.compile(r"^pi\s*/\s*([+-]?(?:\d+\.?\d*|\.\d+))$")
_PI = re.compile(r"^[+-]?pi$")


@dataclass(frozen=True)
class AngleExpression:
    """A gate angle that may mention named parameters.

    `node` is a tiny AST of tuples: `("const", value)`, `("name", identifier)`,
    `("add"|"sub"|"mul"|"div", left, right)`, or `("neg", operand)`.
    """

    text: str
    names: tuple[str, ...]
    node: tuple

    @property
    def is_constant(self) -> bool:
        return not self.names

    def evaluate(self, bindings: dict[str, float]) -> float:
        try:
            value = _evaluate(self.node, bindings)
        except (ArithmeticError, OverflowError):
            value = math.inf
        if not math.isfinite(value):
            raise ParamValueInvalidError(
                f"angle expression {self.text!r} does not evaluate to a finite number"
            )
        return value


def _evaluate(node: tuple, bindings: dict[str, float]) -> float:
    kind = node[0]
    if kind == "const":
        return node[1]
    if kind == "name":
        return float(bindings[node[1]])
    if kind == "neg":
        return -_evaluate(node[1], bindings)
    left = _evaluate(node[1], bindings)
    right = _evaluate(node[2], bindings)
    if kind == "add":
        return left + right
    if kind == "sub":
        return left - right
    if kind == "mul":
        return left * right
    return left / right


@dataclass(frozen=True)
class Operation:
    """One executable statement of the circuit.

    Single-angle gates (`rx`/`ry`/`rz`) carry `angle`/`expression`; the three
    angles of `u3`/`cu3` carry `angles`/`expressions`, in declaration order.
    """

    kind: str  # "gate" or "measure"
    name: str  # gate name, or "measure"
    targets: tuple[int, ...]
    angle: float | None = None
    clbit: int | None = None
    line: int = 0
    expression: AngleExpression | None = None
    angles: tuple[float, ...] | None = None
    expressions: tuple[AngleExpression, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "targets": list(self.targets)}
        if self.expressions is not None:
            payload["angles"] = [
                expression.evaluate({}) if expression.is_constant else expression.text
                for expression in self.expressions
            ]
        elif self.angles is not None:
            payload["angles"] = list(self.angles)
        elif self.angle is not None:
            payload["angle"] = self.angle
        if self.expression is not None:
            payload["expression"] = self.expression.text
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
    parameters: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        document = {
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
        if self.parameters:
            document["parameters"] = list(self.parameters)
        return document

    def bind(self, values: dict[str, float]) -> "Circuit":
        """Resolve every parameter expression against `values`.

        The result is a circuit whose operations carry only concrete angles,
        ready for the simulator. Unbound names raise `KeyError` from the
        expression; non-finite results raise `PARAM_VALUE_INVALID`.
        """

        operations = tuple(
            self._bind_operation(operation, values) for operation in self.operations
        )
        return Circuit(
            id=self.id,
            qubits=self.qubits,
            clbits=self.clbits,
            operations=operations,
            qasm=self.qasm,
            source_lines=self.source_lines,
            measurements=self.measurements,
        )

    @staticmethod
    def _bind_operation(operation: Operation, values: dict[str, float]) -> Operation:
        if operation.expressions is not None:
            return Operation(
                operation.kind,
                operation.name,
                operation.targets,
                angles=tuple(
                    expression.evaluate(values) for expression in operation.expressions
                ),
                clbit=operation.clbit,
                line=operation.line,
            )
        if operation.expression is not None:
            return Operation(
                operation.kind,
                operation.name,
                operation.targets,
                angle=operation.expression.evaluate(values),
                clbit=operation.clbit,
                line=operation.line,
            )
        return operation


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


_ANGLE_TOKEN = re.compile(
    r"\s*(?:(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>[+\-*/]))"
)


def parse_angle(text: str, line: int) -> AngleExpression:
    """Parse a gate angle into an `AngleExpression`.

    Accepts everything `parse_number` accepts plus parameter identifiers and
    flat `+ - * /` arithmetic, for example `theta`, `2*theta`, or
    `theta/2 + pi`. A single leading sign is allowed; parentheses are not
    part of the grammar.
    """

    candidate = text.strip()
    if not candidate:
        raise ParseError(f"line {line}: angle is missing")
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(candidate):
        match = _ANGLE_TOKEN.match(candidate, position)
        if not match or match.end() == position:
            raise ParseError(
                f"line {line}: angle {candidate!r} is not a numeric expression"
            )
        position = match.end()
        if match.lastgroup == "number":
            tokens.append(("number", match.group("number")))
        elif match.lastgroup == "name":
            tokens.append(("name", match.group("name")))
        else:
            tokens.append(("op", match.group("op")))

    index = 0

    def peek() -> tuple[str, str] | None:
        return tokens[index] if index < len(tokens) else None

    def advance() -> tuple[str, str]:
        nonlocal index
        token = tokens[index]
        index += 1
        return token

    def invalid() -> ParseError:
        return ParseError(f"line {line}: angle {candidate!r} is not a numeric expression")

    def factor() -> tuple:
        token = peek()
        if token is None:
            raise invalid()
        kind, value = advance()
        if kind == "number":
            number = float(value)
            if not math.isfinite(number):
                raise ParseError(f"line {line}: angle must be finite")
            return ("const", number)
        if kind == "name":
            if value == "pi":
                return ("const", math.pi)
            return ("name", value)
        raise invalid()

    def term() -> tuple:
        node = factor()
        while peek() is not None and peek() in (("op", "*"), ("op", "/")):
            operator = advance()[1]
            right = factor()
            node = _fold(operator, node, right, line, candidate)
        return node

    def expression() -> tuple:
        sign = ("op", "+")
        if peek() is not None and peek() in (("op", "+"), ("op", "-")):
            sign = advance()
            if peek() is not None and peek() in (("op", "+"), ("op", "-")):
                raise ParseError(
                    f"line {line}: angle {candidate!r} has a repeated sign"
                )
        node = term()
        if sign == ("op", "-"):
            node = _fold("neg", node, None, line, candidate)
        while peek() is not None and peek() in (("op", "+"), ("op", "-")):
            operator = advance()[1]
            right = term()
            node = _fold(operator, node, right, line, candidate)
        return node

    node = expression()
    if peek() is not None:
        raise invalid()

    names: list[str] = []

    def collect(current: tuple) -> None:
        if current[0] == "name" and current[1] not in names:
            names.append(current[1])
        for child in current[1:]:
            if isinstance(child, tuple):
                collect(child)

    collect(node)
    return AngleExpression(candidate, tuple(names), node)


def _fold(operator: str, left: tuple, right: tuple | None, line: int, text: str) -> tuple:
    """Constant-fold one operator node, rejecting non-finite constants."""

    operands = (left,) if right is None else (left, right)
    if all(operand[0] == "const" for operand in operands):
        if operator == "neg":
            return ("const", -left[1])
        a, b = left[1], right[1]
        if operator == "+":
            value = a + b
        elif operator == "-":
            value = a - b
        elif operator == "*":
            value = a * b
        else:
            if b == 0:
                raise ParseError(f"line {line}: division by zero in angle expression")
            value = a / b
        if not math.isfinite(value):
            raise ParseError(f"line {line}: angle must be finite")
        return ("const", value)
    if operator == "neg":
        return ("neg", left)
    return ({"+": "add", "-": "sub", "*": "mul", "/": "div"}[operator], left, right)


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
    parameters: list[str] = []
    for line, text in entries[index:]:
        operation = _parse_operation(text, line, qubits, clbits)
        operations.append(operation)
        if operation.kind == "measure" and operation.clbit is not None:
            measurements.append((operation.targets[0], operation.clbit))
        angle_expressions: tuple[AngleExpression, ...] = ()
        if operation.expressions is not None:
            angle_expressions = operation.expressions
        elif operation.expression is not None:
            angle_expressions = (operation.expression,)
        for angle_expression in angle_expressions:
            for name in angle_expression.names:
                if name not in parameters:
                    parameters.append(name)

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
        parameters=tuple(parameters),
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
        expression = parse_angle(parameter.group(2), line)
        target = _checked_bit(int(parameter.group(3)), qubits, "qubit", line)
        if expression.is_constant:
            return Operation(
                "gate",
                parameter.group(1),
                (target,),
                angle=expression.evaluate({}),
                line=line,
            )
        return Operation(
            "gate", parameter.group(1), (target,), expression=expression, line=line
        )

    u3 = _U3_GATE.match(text)
    if u3:
        expressions = _parse_three_angles(u3.group(1), line, "u3")
        target = _checked_bit(int(u3.group(2)), qubits, "qubit", line)
        return _u3_operation("u3", (target,), expressions, line)

    cu3 = _CU3_GATE.match(text)
    if cu3:
        expressions = _parse_three_angles(cu3.group(1), line, "cu3")
        control = _checked_bit(int(cu3.group(2)), qubits, "qubit", line)
        target = _checked_bit(int(cu3.group(3)), qubits, "qubit", line)
        if control == target:
            raise ParseError(f"line {line}: cu3 requires two distinct qubits")
        return _u3_operation("cu3", (control, target), expressions, line)

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


def _split_angles(body: str, line: int, gate: str) -> list[str]:
    """Split a gate's angle list on top-level commas (no nested parentheses)."""

    parts = [part.strip() for part in body.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ParseError(f"line {line}: {gate} requires exactly three angles")
    return parts


def _parse_three_angles(body: str, line: int, gate: str) -> tuple[AngleExpression, ...]:
    """Parse the `theta, phi, lambda` angle list of a `u3`/`cu3` statement."""

    return tuple(parse_angle(part, line) for part in _split_angles(body, line, gate))


def _u3_operation(
    name: str,
    targets: tuple[int, ...],
    expressions: tuple[AngleExpression, ...],
    line: int,
) -> Operation:
    """Build a `u3`/`cu3` operation, folding an all-constant angle list."""

    if all(expression.is_constant for expression in expressions):
        return Operation(
            "gate",
            name,
            targets,
            angles=tuple(expression.evaluate({}) for expression in expressions),
            line=line,
        )
    return Operation("gate", name, targets, expressions=expressions, line=line)


def _checked_bit(value: int, size: int, label: str, line: int) -> int:
    if value >= size:
        raise ParseError(
            f"line {line}: {label} index {value} is out of range for a register of size {size}"
        )
    return value
