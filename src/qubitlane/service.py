from __future__ import annotations

import hashlib
from typing import Any, Callable

from .errors import ConflictError, NotFoundError, ValidationError
from .model import CircuitRequest, SimulationRequest
from .qasm import Circuit, identifier, parse_circuit
from .simulator import (
    Sampler,
    assert_normalized,
    distribution,
    nonzero_amplitudes,
    probabilities,
    simulate,
)
from .store import Store


def _require_key(key: str | None) -> str:
    if not key or not key.strip():
        raise ValidationError("Idempotency-Key header is required")
    if len(key) > 200:
        raise ValidationError("Idempotency-Key must be at most 200 characters")
    return key


class QubitLane:
    """Circuits are immutable; every simulation is an append-only job."""

    def __init__(self, database: str = ":memory:"):
        self.store = Store(database)

    # -- circuits ---------------------------------------------------------

    def create_circuit(self, raw: Any, key: str | None) -> dict[str, Any]:
        _require_key(key)
        request = CircuitRequest.parse(raw)
        circuit = parse_circuit(request.qasm, request.id)
        document = circuit.as_dict()

        def create() -> dict[str, Any]:
            row = self.store.connection.execute(
                "SELECT document FROM circuits WHERE id = ?", (circuit.id,)
            ).fetchone()
            if row:
                stored = self.store.decode(row["document"])
                if stored["qasm"] == circuit.qasm:
                    return stored
                raise ConflictError(f"circuit {circuit.id} already exists with different qasm")
            self.store.connection.execute(
                "INSERT INTO circuits(id, document) VALUES (?, ?)",
                (circuit.id, self.store.encode(document)),
            )
            return document

        return self._idempotent(key, f"create-circuit:{circuit.id}", create)

    def get_circuit(self, circuit_id: str) -> dict[str, Any]:
        row = self.store.connection.execute(
            "SELECT document FROM circuits WHERE id = ?", (circuit_id,)
        ).fetchone()
        if not row:
            raise NotFoundError(f"circuit {circuit_id} was not found")
        return self.store.decode(row["document"])

    def _load(self, circuit_id: str) -> Circuit:
        document = self.get_circuit(circuit_id)
        return parse_circuit(document["qasm"], document["id"])

    # -- simulation -------------------------------------------------------

    def simulate(self, circuit_id: str, raw: Any, key: str | None) -> dict[str, Any]:
        _require_key(key)
        identifier(circuit_id, "circuit id")
        request = SimulationRequest.parse(raw)
        self.get_circuit(circuit_id)

        job_id = self._job_id(circuit_id, request.shots, request.seed)

        def run() -> dict[str, Any]:
            existing = self.store.connection.execute(
                "SELECT document FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if existing:
                return self.store.decode(existing["document"])
            job = self._execute(circuit_id, job_id, request)
            self.store.connection.execute(
                "INSERT INTO jobs(id, circuit_id, document) VALUES (?, ?, ?)",
                (job_id, circuit_id, self.store.encode(job)),
            )
            return job

        return self._idempotent(key, f"simulate:{job_id}", run)

    def get_statevector(self, circuit_id: str) -> dict[str, Any]:
        identifier(circuit_id, "circuit id")
        circuit = self._load(circuit_id)
        state = simulate(circuit)
        defect = assert_normalized(state)
        return {
            "circuit_id": circuit.id,
            "qubits": circuit.qubits,
            "bit_order": "little_endian",
            "normalization_error": defect,
            "amplitudes": nonzero_amplitudes(circuit, state),
        }

    def get_job(self, job_id: str) -> dict[str, Any]:
        identifier(job_id, "job id")
        row = self.store.connection.execute(
            "SELECT document FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if not row:
            raise NotFoundError(f"job {job_id} was not found")
        return self.store.decode(row["document"])

    # -- internals --------------------------------------------------------

    def _execute(self, circuit_id: str, job_id: str, request: SimulationRequest) -> dict[str, Any]:
        circuit = self._load(circuit_id)
        state = simulate(circuit)
        defect = assert_normalized(state)
        weights = probabilities(state)
        sampler = Sampler(request.seed)
        raw_counts = sampler.sample(weights, request.shots, circuit.qubits)
        counts = {self._measured_key(circuit, label): value for label, value in raw_counts.items()}
        ordered = {key: counts[key] for key in sorted(counts)}
        return {
            "id": job_id,
            "circuit_id": circuit.id,
            "state": "completed",
            "shots": request.shots,
            "seed": request.seed,
            "qubits": circuit.qubits,
            "bit_order": "little_endian",
            "normalization_error": defect,
            "probabilities": distribution(circuit, state),
            "counts": ordered,
            "measured_bits": self._measured_bit_count(circuit),
            "created_at": self.store.now(),
        }

    @staticmethod
    def _measured_bit_count(circuit: Circuit) -> int:
        if not circuit.measurements:
            return circuit.qubits
        return max(clbit for _, clbit in circuit.measurements) + 1

    @staticmethod
    def _job_id(circuit_id: str, shots: int, seed: int) -> str:
        raw = f"{circuit_id}|{shots}|{seed}".encode("utf-8")
        return f"j-{hashlib.sha256(raw).hexdigest()[:16]}"

    @staticmethod
    def _measured_key(circuit: Circuit, label: str) -> str:
        """Project a big-endian basis label onto the measured classical bits.

        `label` is big-endian, so character `i` holds qubit `qubits - 1 - i`.
        The result is big-endian as well: character `i` holds classical bit
        `width - 1 - i`. A classical bit that no `measure` statement wrote
        stays `0`.
        """

        if not circuit.measurements:
            return label
        width = QubitLane._measured_bit_count(circuit)
        bits = ["0"] * width
        for qubit, clbit in circuit.measurements:
            bits[clbit] = label[circuit.qubits - 1 - qubit]
        return "".join(reversed(bits))

    def _idempotent(
        self, key: str | None, operation: str, action: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        checked = _require_key(key)
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT operation, response FROM idempotency WHERE key = ?", (checked,)
            ).fetchone()
            if existing:
                if existing["operation"] != operation:
                    raise ConflictError("idempotency key was already used for another operation")
                return self.store.decode(existing["response"])
            response = action()
            connection.execute(
                "INSERT INTO idempotency(key, operation, response) VALUES (?, ?, ?)",
                (checked, operation, self.store.encode(response)),
            )
            return response
