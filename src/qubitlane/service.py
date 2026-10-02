from __future__ import annotations

import hashlib
from typing import Any, Callable

from .errors import (
    BatchLimitExceededError,
    ConflictError,
    NotFoundError,
    ParamUndefinedError,
    ValidationError,
)
from .model import MAX_BATCH_TOTAL_SHOTS, CircuitRequest, SimulationRequest
from .qasm import Circuit, identifier, parse_circuit
from .simulator import (
    MAX_NOISE_QUBITS,
    Sampler,
    assert_normalized,
    assert_normalized_weights,
    density_probabilities,
    distribution,
    nonzero_amplitudes,
    probabilities,
    simulate,
    simulate_density,
    weight_distribution,
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
        document = self.get_circuit(circuit_id)
        if request.noise is not None and document["qubits"] > MAX_NOISE_QUBITS:
            raise ValidationError(
                f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits"
            )

        if request.parameters is not None:
            return self._simulate_batch(circuit_id, request, key)

        job_id = self._job_id(circuit_id, request)

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

    def _simulate_batch(
        self, circuit_id: str, request: SimulationRequest, key: str | None
    ) -> dict[str, Any]:
        circuit = self._load(circuit_id)
        self._resolve_bindings(circuit, request.parameters)
        scenario_count = len(next(iter(request.parameters.values())))
        if scenario_count * request.shots > MAX_BATCH_TOTAL_SHOTS:
            raise BatchLimitExceededError(
                f"{scenario_count} scenarios at {request.shots} shots each exceed "
                f"the limit of {MAX_BATCH_TOTAL_SHOTS} total shots"
            )

        job_id = self._job_id(circuit_id, request)

        def run() -> dict[str, Any]:
            existing = self.store.connection.execute(
                "SELECT document FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if existing:
                return self.store.decode(existing["document"])
            job = self._execute_batch(circuit, job_id, request)
            self.store.connection.execute(
                "INSERT INTO jobs(id, circuit_id, document) VALUES (?, ?, ?)",
                (job_id, circuit_id, self.store.encode(job)),
            )
            return job

        return self._idempotent(key, f"simulate:{job_id}", run)

    @staticmethod
    def _resolve_bindings(circuit: Circuit, parameters: dict[str, list[float]]) -> None:
        for name in parameters:
            if name not in circuit.parameters:
                raise ParamUndefinedError(
                    f"parameter {name!r} does not appear in the circuit's "
                    "parameter expressions"
                )
        for name in circuit.parameters:
            if name not in parameters:
                raise ParamUndefinedError(
                    f"parameter {name!r} is not bound by the request"
                )

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
        if request.noise is not None:
            rho = simulate_density(circuit, request.noise.probability)
            weights = density_probabilities(rho, circuit.qubits)
            defect = assert_normalized_weights(weights)
            probabilities_document = weight_distribution(circuit, weights)
        else:
            state = simulate(circuit)
            defect = assert_normalized(state)
            weights = probabilities(state)
            probabilities_document = distribution(circuit, state)
        sampler = Sampler(request.seed)
        raw_counts = sampler.sample(weights, request.shots, circuit.qubits)
        counts = {self._measured_key(circuit, label): value for label, value in raw_counts.items()}
        ordered = {key: counts[key] for key in sorted(counts)}
        job = {
            "id": job_id,
            "circuit_id": circuit.id,
            "state": "completed",
            "shots": request.shots,
            "seed": request.seed,
            "qubits": circuit.qubits,
            "bit_order": "little_endian",
            "normalization_error": defect,
            "probabilities": probabilities_document,
            "counts": ordered,
            "measured_bits": self._measured_bit_count(circuit),
            "created_at": self.store.now(),
        }
        if request.noise is not None:
            job["noise"] = request.noise.as_dict()
        return job

    def _execute_batch(
        self, circuit: Circuit, job_id: str, request: SimulationRequest
    ) -> dict[str, Any]:
        names = list(request.parameters)
        arrays = request.parameters
        scenario_count = len(arrays[names[0]])
        scenarios: list[dict[str, Any]] = []
        for index in range(scenario_count):
            values = {name: arrays[name][index] for name in names}
            bound = circuit.bind(values)
            if request.noise is not None:
                rho = simulate_density(bound, request.noise.probability)
                weights = density_probabilities(rho, bound.qubits)
                assert_normalized_weights(weights)
            else:
                state = simulate(bound)
                assert_normalized(state)
                weights = probabilities(state)
            sampler = Sampler(request.seed + index)
            raw_counts = sampler.sample(weights, request.shots, bound.qubits)
            counts = {
                self._measured_key(bound, label): value
                for label, value in raw_counts.items()
            }
            ordered = {key: counts[key] for key in sorted(counts)}
            scenarios.append({"index": index, "values": values, "counts": ordered})
        job = {
            "id": job_id,
            "circuit_id": circuit.id,
            "state": "completed",
            "shots": request.shots,
            "seed": request.seed,
            "qubits": circuit.qubits,
            "bit_order": "little_endian",
            "measured_bits": self._measured_bit_count(circuit),
            "parameters": names,
            "scenario_count": scenario_count,
            "scenarios": scenarios,
            "created_at": self.store.now(),
        }
        if request.noise is not None:
            job["noise"] = request.noise.as_dict()
        return job

    @staticmethod
    def _measured_bit_count(circuit: Circuit) -> int:
        if not circuit.measurements:
            return circuit.qubits
        return max(clbit for _, clbit in circuit.measurements) + 1

    @staticmethod
    def _job_id(circuit_id: str, request: SimulationRequest) -> str:
        raw = f"{circuit_id}|{request.shots}|{request.seed}"
        if request.noise is not None:
            raw += f"|{request.noise.type}|{request.noise.probability!r}"
        if request.parameters is not None:
            bindings = ",".join(
                f"{name}={'/'.join(repr(value) for value in values)}"
                for name, values in request.parameters.items()
            )
            raw += f"|params:{bindings}"
        return f"j-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"

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
