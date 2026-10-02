from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from .errors import (
    BatchLimitExceededError,
    ConflictError,
    NotFoundError,
    ParameterUndefinedError,
    ValidationError,
)
from .model import MAX_BATCH_TOTAL_SHOTS, CircuitRequest, NoiseSpec, SimulationRequest
from .qasm import Circuit, bind_parameters, identifier, parameter_names, parse_circuit
from .simulator import (
    MAX_NOISE_QUBITS,
    Sampler,
    assert_normalized,
    assert_normalized_weights,
    density_probabilities,
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
        self._check_parameters(document, request)

        job_id = self._job_id(circuit_id, request)

        def run() -> dict[str, Any]:
            existing = self.store.connection.execute(
                "SELECT document FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if existing:
                return self.store.decode(existing["document"])
            if request.parameters is not None:
                job = self._execute_batch(circuit_id, job_id, request)
            else:
                job = self._execute(circuit_id, job_id, request)
            self.store.connection.execute(
                "INSERT INTO jobs(id, circuit_id, document) VALUES (?, ?, ?)",
                (job_id, circuit_id, self.store.encode(job)),
            )
            return job

        return self._idempotent(key, f"simulate:{job_id}", run)

    @staticmethod
    def _check_parameters(document: dict[str, Any], request: SimulationRequest) -> None:
        """Validate the request's bindings against the circuit's parameters.

        Every binding name must resolve to a parameter expression in the
        circuit, every circuit parameter must be bound, and the total draw
        count `scenarios * shots` must stay within the service limit. Any
        failure raises before a job is created.
        """

        defined = tuple(
            dict.fromkeys(
                operation["parameter"]
                for operation in document["operations"]
                if "parameter" in operation
            )
        )
        bindings = request.parameters
        if bindings is None:
            if defined:
                raise ParameterUndefinedError(
                    f"circuit parameter {defined[0]!r} has no binding in the request"
                )
            return
        for name in bindings.names:
            if name not in defined:
                raise ParameterUndefinedError(
                    f"parameter {name!r} is not used by the circuit"
                )
        for name in defined:
            if name not in bindings.names:
                raise ParameterUndefinedError(
                    f"circuit parameter {name!r} has no binding in the request"
                )
        total = bindings.scenarios * request.shots
        if total > MAX_BATCH_TOTAL_SHOTS:
            raise BatchLimitExceededError(
                f"batch of {bindings.scenarios} scenarios at {request.shots} shots each "
                f"exceeds the limit of {MAX_BATCH_TOTAL_SHOTS} total shots"
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
        weights, defect = self._weights(circuit, request.noise)
        probabilities_document = weight_distribution(circuit, weights)
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
        self, circuit_id: str, job_id: str, request: SimulationRequest
    ) -> dict[str, Any]:
        """Run one scenario per parameter position, in scenario order.

        Each scenario binds its values, executes the circuit, and draws
        `shots` samples from a sampler seeded by `(seed, index)`, so the
        stored result is identical no matter how scenarios are scheduled.
        """

        circuit = self._load(circuit_id)
        bindings = request.parameters
        scenarios: list[dict[str, Any]] = []
        for index in range(bindings.scenarios):
            values = {
                name: bindings.values[position][index]
                for position, name in enumerate(bindings.names)
            }
            bound = bind_parameters(circuit, values)
            weights, _ = self._weights(bound, request.noise)
            sampler = Sampler(request.seed + index)
            raw_counts = sampler.sample(weights, request.shots, bound.qubits)
            counts = {
                self._measured_key(bound, label): value
                for label, value in raw_counts.items()
            }
            scenarios.append(
                {
                    "index": index,
                    "values": values,
                    "counts": {key: counts[key] for key in sorted(counts)},
                }
            )
        job = {
            "id": job_id,
            "circuit_id": circuit.id,
            "state": "completed",
            "shots": request.shots,
            "seed": request.seed,
            "qubits": circuit.qubits,
            "bit_order": "little_endian",
            "scenario_count": bindings.scenarios,
            "parameters": list(bindings.names),
            "scenarios": scenarios,
            "measured_bits": self._measured_bit_count(circuit),
            "created_at": self.store.now(),
        }
        if request.noise is not None:
            job["noise"] = request.noise.as_dict()
        return job

    @staticmethod
    def _weights(circuit: Circuit, noise: NoiseSpec | None) -> tuple[list[float], float]:
        """The final basis-state weights and the normalisation defect."""

        if noise is not None:
            rho = simulate_density(circuit, noise.probability)
            weights = density_probabilities(rho, circuit.qubits)
            return weights, assert_normalized_weights(weights)
        state = simulate(circuit)
        return probabilities(state), assert_normalized(state)

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
            canonical = json.dumps(
                {
                    name: list(request.parameters.values[position])
                    for position, name in enumerate(request.parameters.names)
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            raw += f"|parameters:{canonical}"
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
