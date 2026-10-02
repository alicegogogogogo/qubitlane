"""Request models for the QubitLane public contract."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .errors import (
    ParamArrayEmptyError,
    ParamArrayLengthMismatchError,
    ParamValueInvalidError,
    ShotsInvalidError,
    ValidationError,
)
from .qasm import identifier

MAX_QASM_BYTES = 64 * 1024
MAX_SHOTS = 100_000
MAX_SEED = (1 << 63) - 1
MAX_BATCH_TOTAL_SHOTS = MAX_SHOTS

DEFAULT_SHOTS = 1024
DEFAULT_SEED = 0

PARAMETER_FIELDS = ("parameters", "parameter_bindings", "bindings")


def _require_object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValidationError(f"{label} body must be a JSON object")
    return raw


def _reject_unknown(raw: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValidationError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _shots(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("shots must be an integer")
    if value < 1 or value > MAX_SHOTS:
        raise ValidationError(f"shots must be between 1 and {MAX_SHOTS}")
    return value


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("seed must be an integer")
    if value < 0 or value > MAX_SEED:
        raise ValidationError(f"seed must be between 0 and {MAX_SEED}")
    return value


def _batch_shots(value: Any) -> int:
    """Shots for a parameter sweep: any positive integer is syntactically valid.

    The total work is bounded separately by the batch limit, so a rejected
    value here is exactly one that is not a positive integer.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ShotsInvalidError("shots must be an integer")
    if value < 1:
        raise ShotsInvalidError("shots must be a positive integer")
    return value


def _parameters(value: Any) -> dict[str, list[float]]:
    """Validate a `name -> non-empty numeric array` binding map."""

    if not isinstance(value, dict):
        raise ValidationError("parameters must be a JSON object")
    if not value:
        raise ValidationError("parameters must bind at least one parameter")
    bindings: dict[str, list[float]] = {}
    length: int | None = None
    for name, array in value.items():
        if not isinstance(array, list):
            raise ValidationError(f"parameter {name!r} must be an array of numbers")
        if not array:
            raise ParamArrayEmptyError(f"parameter {name!r} must not be an empty array")
        values: list[float] = []
        for element in array:
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                raise ParamValueInvalidError(
                    f"parameter {name!r} must contain only numbers"
                )
            element = float(element)
            if not math.isfinite(element):
                raise ParamValueInvalidError(
                    f"parameter {name!r} must contain only finite numbers"
                )
            values.append(element)
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ParamArrayLengthMismatchError(
                "all parameter arrays must have the same length"
            )
        bindings[name] = values
    return bindings


@dataclass(frozen=True)
class NoiseSpec:
    """A validated `noise` object: depolarizing channel with probability `p`."""

    type: str
    probability: float

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "probability": self.probability}


def _noise(value: Any) -> NoiseSpec:
    if not isinstance(value, dict):
        raise ValidationError("noise must be a JSON object")
    _reject_unknown(value, {"type", "probability"}, "noise")
    if "type" not in value:
        raise ValidationError("noise must contain a type field")
    if "probability" not in value:
        raise ValidationError("noise must contain a probability field")
    if value["type"] != "depolarizing":
        raise ValidationError('noise type must be "depolarizing"')
    probability = value["probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValidationError("noise probability must be a number")
    probability = float(probability)
    if not math.isfinite(probability):
        raise ValidationError("noise probability must be a finite number")
    if probability < 0.0 or probability > 1.0:
        raise ValidationError("noise probability must be between 0 and 1")
    return NoiseSpec("depolarizing", probability)


def content_id(qasm: str) -> str:
    """Deterministic circuit identifier derived from the source text."""

    digest = hashlib.sha256(qasm.encode("utf-8")).hexdigest()[:16]
    return f"c-{digest}"


@dataclass(frozen=True)
class CircuitRequest:
    id: str
    qasm: str

    @classmethod
    def parse(cls, raw: Any) -> "CircuitRequest":
        body = _require_object(raw, "circuit")
        _reject_unknown(body, {"id", "qasm"}, "circuit request")
        if "qasm" not in body:
            raise ValidationError("circuit request must contain a qasm string")
        qasm = body["qasm"]
        if not isinstance(qasm, str) or not qasm.strip():
            raise ValidationError("qasm must be a non-empty string")
        if len(qasm.encode("utf-8")) > MAX_QASM_BYTES:
            raise ValidationError(f"qasm must be at most {MAX_QASM_BYTES} bytes")
        if "id" in body and body["id"] is not None:
            circuit_id = identifier(body["id"], "circuit id")
        else:
            circuit_id = content_id(qasm)
        return cls(id=circuit_id, qasm=qasm)


@dataclass(frozen=True)
class SimulationRequest:
    shots: int
    seed: int
    noise: NoiseSpec | None = None
    parameters: dict[str, list[float]] | None = None

    @classmethod
    def parse(cls, raw: Any) -> "SimulationRequest":
        if raw is None:
            return cls(DEFAULT_SHOTS, DEFAULT_SEED)
        body = _require_object(raw, "simulation")
        _reject_unknown(
            body, {"shots", "seed", "noise", *PARAMETER_FIELDS}, "simulation request"
        )
        aliases = [field for field in PARAMETER_FIELDS if field in body]
        if len(aliases) > 1:
            raise ValidationError("parameter bindings must be provided only once")
        is_batch = bool(aliases)
        if "shots" in body:
            shots = _batch_shots(body["shots"]) if is_batch else _shots(body["shots"])
        else:
            shots = DEFAULT_SHOTS
        seed = _seed(body["seed"]) if "seed" in body else DEFAULT_SEED
        noise = _noise(body["noise"]) if "noise" in body else None
        parameters = _parameters(body[aliases[0]]) if is_batch else None
        return cls(shots, seed, noise, parameters)
