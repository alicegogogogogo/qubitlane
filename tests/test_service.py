import math
import tempfile
import unittest
from pathlib import Path

from qubitlane.errors import (
    BatchLimitExceededError,
    ConflictError,
    NotFoundError,
    ParamArrayEmptyError,
    ParamArrayLengthMismatchError,
    ParamUndefinedError,
    ParamValueInvalidError,
    ParseError,
    ShotsInvalidError,
    ValidationError,
)
from qubitlane.qasm import parse_circuit
from qubitlane.service import QubitLane
from qubitlane.simulator import (
    TOLERANCE,
    basis_label,
    initial_state,
    norm_squared,
    simulate,
)
BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


def circuit_document(qasm: str, circuit_id: str = "c-test") -> dict:
    return {"id": circuit_id, "qasm": qasm}


class QasmParsingTests(unittest.TestCase):
    def test_parses_registers_gates_and_measurements(self):
        circuit = parse_circuit(BELL, "bell")
        self.assertEqual(2, circuit.qubits)
        self.assertEqual(2, circuit.clbits)
        self.assertEqual(("h", "cx", "measure", "measure"), tuple(op.name for op in circuit.operations))
        self.assertEqual(((0, 0), (1, 1)), circuit.measurements)

    def test_unknown_gate_is_rejected_with_line_number(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nccx q[0], q[0], q[0];\n"
        with self.assertRaisesRegex(ParseError, r"line 3: unsupported statement or gate"):
            parse_circuit(qasm, "bad")

    def test_out_of_range_qubit_is_rejected_with_line_number(self):
        qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[2];\n"
        with self.assertRaisesRegex(ParseError, r"line 3: qubit index 2 is out of range"):
            parse_circuit(qasm, "bad")

    def test_measurement_needs_a_classical_register(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nmeasure q[0] -> c[0];\n"
        with self.assertRaisesRegex(ParseError, r"line 3: measure requires a classical register"):
            parse_circuit(qasm, "bad")

    def test_classical_bit_out_of_range(self):
        qasm = 'OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[1];\n'
        with self.assertRaisesRegex(ParseError, r"line 4: classical bit index 1 is out of range"):
            parse_circuit(qasm, "bad")

    def test_header_and_register_shape_are_enforced(self):
        with self.assertRaisesRegex(ParseError, "first statement"):
            parse_circuit("qreg q[1];\nh q[0];", "bad")
        with self.assertRaisesRegex(ValidationError, "quantum register"):
            parse_circuit('OPENQASM 2.0;\ninclude "qelib1.inc";\ncreg c[1];\n', "bad")
        with self.assertRaisesRegex(ParseError, "at most 16 qubits"):
            parse_circuit("OPENQASM 2.0;\nqreg q[17];\nh q[0];\n", "bad")
        with self.assertRaisesRegex(ParseError, "named q"):
            parse_circuit("OPENQASM 2.0;\nqreg a[1];\nh a[0];\n", "bad")
        with self.assertRaisesRegex(ParseError, "only include"):
            parse_circuit('OPENQASM 2.0;\ninclude "other.inc";\nqreg q[1];\nh q[0];\n', "bad")

    def test_angle_expressions(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nrx(pi/2) q[0];\nry(-2*pi) q[0];\nrz(0.5) q[0];\n"
        circuit = parse_circuit(qasm, "angles")
        rx, ry, rz = circuit.operations
        self.assertAlmostEqual(math.pi / 2, rx.angle, places=12)
        self.assertAlmostEqual(-2 * math.pi, ry.angle, places=12)
        self.assertAlmostEqual(0.5, rz.angle, places=12)

    def test_malformed_angles_are_rejected(self):
        with self.assertRaisesRegex(ParseError, "division by zero"):
            parse_circuit("OPENQASM 2.0;\nqreg q[1];\nrx(pi/0) q[0];\n", "bad")
        with self.assertRaisesRegex(ParseError, "repeated sign"):
            parse_circuit("OPENQASM 2.0;\nqreg q[1];\nrx(--1) q[0];\n", "bad")
        with self.assertRaisesRegex(ParseError, "numeric expression"):
            parse_circuit("OPENQASM 2.0;\nqreg q[1];\nrx(pi+) q[0];\n", "bad")

    def test_unknown_field_and_missing_qasm_are_rejected(self):
        service = QubitLane(":memory:")
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            service.create_circuit({"id": "x", "qasm": BELL, "shots": 1}, "k1")
        with self.assertRaisesRegex(ValidationError, "must contain a qasm"):
            service.create_circuit({"id": "x"}, "k2")


class SimulationTests(unittest.TestCase):
    def test_bell_state_vector_is_entangled_and_normalised(self):
        circuit = parse_circuit(BELL, "bell")
        state = simulate(circuit)
        self.assertAlmostEqual(1.0, norm_squared(state), places=12)
        self.assertAlmostEqual(1 / math.sqrt(2), abs(state[0]), places=12)
        self.assertAlmostEqual(1 / math.sqrt(2), abs(state[3]), places=12)
        self.assertLess(abs(state[1]), TOLERANCE)
        self.assertLess(abs(state[2]), TOLERANCE)
        self.assertEqual("00", basis_label(0, 2))
        self.assertEqual("11", basis_label(3, 2))
        self.assertEqual("01", basis_label(1, 2))

    def test_gate_matrices_are_unitary(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[2];\n"
            "h q[0];\nt q[0];\ns q[1];\nz q[1];\ny q[0];\nx q[1];\n"
            "rx(0.7) q[0];\nry(1.1) q[1];\nrz(-0.3) q[0];\ncx q[0], q[1];\ncz q[1], q[0];\n"
        )
        state = simulate(parse_circuit(qasm, "unit"))
        self.assertLess(abs(norm_squared(state) - 1.0), TOLERANCE)

    def test_initial_state_is_zero_and_normalised(self):
        state = initial_state(3)
        self.assertEqual(8, len(state))
        self.assertEqual(1 + 0j, state[0])
        self.assertAlmostEqual(1.0, norm_squared(state), places=12)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = QubitLane(str(Path(self.directory.name) / "qubitlane.db"))
        self.counter = 0

    def tearDown(self):
        self.directory.cleanup()

    def key(self, prefix: str = "k") -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}"

    def test_circuit_round_trip(self):
        created = self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        self.assertEqual("bell", created["id"])
        self.assertEqual(2, created["qubits"])
        self.assertEqual("little_endian", created["bit_order"])
        self.assertEqual(created, self.service.get_circuit("bell"))
        missing = self.service.create_circuit({"qasm": BELL}, self.key())
        self.assertTrue(missing["id"].startswith("c-"))

    def test_same_seed_repeats_and_different_seed_differs(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        first = self.service.simulate(
            "bell", {"shots": 400, "seed": 7}, self.key()
        )
        repeat = self.service.simulate("bell", {"shots": 400, "seed": 7}, self.key())
        self.assertEqual(first, repeat)
        self.assertEqual(400, sum(repeat["counts"].values()))
        self.assertEqual({"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}, repeat["probabilities"])
        other = self.service.simulate("bell", {"shots": 400, "seed": 8}, self.key())
        self.assertNotEqual(first["counts"], other["counts"])

    def test_job_is_retrievable_and_ids_are_deterministic(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        job = self.service.simulate("bell", {"shots": 100, "seed": 3}, self.key())
        self.assertEqual("completed", job["state"])
        self.assertEqual(job, self.service.get_job(job["id"]))
        repeated = self.service.simulate("bell", {"shots": 100, "seed": 3}, self.key())
        self.assertEqual(job["id"], repeated["id"])

    def test_defaults_are_applied_when_body_is_empty(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        job = self.service.simulate("bell", None, self.key())
        self.assertEqual(1024, job["shots"])
        self.assertEqual(0, job["seed"])
        self.assertEqual(1024, sum(job["counts"].values()))

    def test_statevector_endpoint_reports_basis_and_normalisation(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        payload = self.service.get_statevector("bell")
        self.assertEqual(2, payload["qubits"])
        self.assertLess(payload["normalization_error"], TOLERANCE)
        self.assertEqual(["00", "11"], [entry["basis"] for entry in payload["amplitudes"]])
        for entry in payload["amplitudes"]:
            self.assertAlmostEqual(0.5, entry["probability"], places=12)
        self.assertAlmostEqual(
            1 / math.sqrt(2), payload["amplitudes"][0]["amplitude"]["real"], places=12
        )

    def test_probability_distribution_covers_every_basis_state(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nrx(0.4) q[0];\n"
        self.service.create_circuit(circuit_document(qasm, "one"), self.key())
        job = self.service.simulate("one", {"shots": 50, "seed": 1}, self.key())
        self.assertEqual({"0", "1"}, set(job["probabilities"]))
        self.assertAlmostEqual(1.0, sum(job["probabilities"].values()), places=9)
        self.assertAlmostEqual(math.sin(0.2) ** 2, job["probabilities"]["1"], places=12)

    def test_measurements_project_onto_classical_bits_little_endian(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\n"
            "x q[1];\nmeasure q[1] -> c[0];\nmeasure q[0] -> c[1];\n"
        )
        self.service.create_circuit(circuit_document(qasm, "swap"), self.key())
        job = self.service.simulate("swap", {"shots": 8, "seed": 5}, self.key())
        self.assertEqual(2, job["measured_bits"])
        self.assertEqual({"01": 8}, job["counts"])
        self.assertEqual({"00": 0.0, "01": 0.0, "10": 1.0, "11": 0.0}, job["probabilities"])

    def test_idempotency_key_replays_the_first_result(self):
        first = self.service.create_circuit(circuit_document(BELL, "bell"), "shared")
        second = self.service.create_circuit(circuit_document(BELL, "bell"), "shared")
        self.assertEqual(first, second)
        with self.assertRaises(ConflictError):
            self.service.create_circuit(circuit_document(BELL, "other"), "shared")

    def test_idempotency_key_is_required(self):
        with self.assertRaisesRegex(ValidationError, "Idempotency-Key"):
            self.service.create_circuit(circuit_document(BELL, "bell"), None)
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        with self.assertRaisesRegex(ValidationError, "Idempotency-Key"):
            self.service.simulate("bell", {"shots": 10}, None)

    def test_repeated_simulation_with_same_key_is_stable(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        first = self.service.simulate("bell", {"shots": 64, "seed": 2}, "run")
        repeated = self.service.simulate("bell", {"shots": 64, "seed": 2}, "run")
        self.assertEqual(first, repeated)

    def test_reused_id_with_different_qasm_conflicts(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        with self.assertRaisesRegex(ConflictError, "different qasm"):
            self.service.create_circuit(
                circuit_document("OPENQASM 2.0;\nqreg q[1];\nx q[0];\n", "bell"),
                self.key(),
            )

    def test_unknown_circuit_and_job_are_not_found(self):
        with self.assertRaises(NotFoundError):
            self.service.get_circuit("missing")
        with self.assertRaises(NotFoundError):
            self.service.get_statevector("missing")
        with self.assertRaises(NotFoundError):
            self.service.get_job("missing")
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        with self.assertRaises(NotFoundError):
            self.service.simulate("missing", {"shots": 1}, self.key())

    def test_simulation_request_validation(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            self.service.simulate("bell", {"shots": 10, "qubits": 3}, self.key())
        with self.assertRaisesRegex(ValidationError, "shots must be"):
            self.service.simulate("bell", {"shots": 0}, self.key())
        with self.assertRaisesRegex(ValidationError, "shots must be an integer"):
            self.service.simulate("bell", {"shots": True}, self.key())
        with self.assertRaisesRegex(ValidationError, "shot"):
            self.service.simulate("bell", {"shots": 100_001}, self.key())
        with self.assertRaisesRegex(ValidationError, "seed must be"):
            self.service.simulate("bell", {"shots": 1, "seed": -1}, self.key())

    def test_persisted_circuit_survives_a_new_service_instance(self):
        path = str(Path(self.directory.name) / "shared.db")
        first = QubitLane(path)
        first.create_circuit(circuit_document(BELL, "bell"), self.key())
        job = first.simulate("bell", {"shots": 32, "seed": 11}, self.key())
        second = QubitLane(path)
        self.assertEqual("bell", second.get_circuit("bell")["id"])
        self.assertEqual(job, second.get_job(job["id"]))

    def test_unsupported_gate_through_the_service_is_a_validation_error(self):
        qasm = "OPENQASM 2.0;\nqreg q[2];\nswap q[0], q[1];\n"
        with self.assertRaisesRegex(ParseError, "unsupported"):
            self.service.create_circuit(circuit_document(qasm, "bad"), self.key())


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = QubitLane(str(Path(self.directory.name) / "qubitlane.db"))
        self.counter = 0

    def tearDown(self):
        self.directory.cleanup()

    def key(self, prefix: str = "k") -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}"

    def noise(self, probability: float = 0.01) -> dict:
        return {"type": "depolarizing", "probability": probability}

    def test_noisy_job_reports_noise_and_is_deterministic(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        body = {"shots": 200, "seed": 9, "noise": self.noise(0.05)}
        first = self.service.simulate("bell", body, self.key())
        self.assertEqual(self.noise(0.05), first["noise"])
        self.assertEqual(200, sum(first["counts"].values()))
        self.assertEqual({"00", "01", "10", "11"}, set(first["probabilities"]))
        self.assertAlmostEqual(1.0, sum(first["probabilities"].values()), places=9)
        self.assertLess(first["normalization_error"], TOLERANCE)
        repeated = self.service.simulate("bell", body, self.key())
        self.assertEqual(first, repeated)
        self.assertEqual(first, self.service.get_job(first["id"]))

    def test_noise_changes_the_job_id_and_counts(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        quiet = self.service.simulate("bell", {"shots": 200, "seed": 9}, self.key())
        noisy = self.service.simulate(
            "bell", {"shots": 200, "seed": 9, "noise": self.noise(0.4)}, self.key()
        )
        self.assertNotEqual(quiet["id"], noisy["id"])
        self.assertNotIn("noise", quiet)
        self.assertNotEqual(quiet["probabilities"], noisy["probabilities"])

    def test_numerically_equal_noise_shares_the_job_id(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        first = self.service.simulate(
            "bell", {"shots": 10, "seed": 1, "noise": self.noise(0.5)}, self.key()
        )
        second = self.service.simulate(
            "bell", {"shots": 10, "seed": 1, "noise": self.noise(0.50)}, self.key()
        )
        self.assertEqual(first["id"], second["id"])
        third = self.service.simulate(
            "bell", {"shots": 10, "seed": 1, "noise": self.noise(0.25)}, self.key()
        )
        self.assertNotEqual(first["id"], third["id"])

    def test_zero_probability_noise_matches_the_quiet_distribution(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        quiet = self.service.simulate("bell", {"shots": 50, "seed": 4}, self.key())
        noisy = self.service.simulate(
            "bell", {"shots": 50, "seed": 4, "noise": self.noise(0.0)}, self.key()
        )
        self.assertEqual(quiet["probabilities"], noisy["probabilities"])
        self.assertEqual(quiet["counts"], noisy["counts"])

    def test_depolarizing_channel_mixes_towards_uniform(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nx q[0];\n"
        self.service.create_circuit(circuit_document(qasm, "one"), self.key())
        job = self.service.simulate(
            "one", {"shots": 10, "seed": 1, "noise": self.noise(0.3)}, self.key()
        )
        # (1 - p) rho + p/3 (X rho X + Y rho Y + Z rho Z) leaves P(1) = 1 - 2p/3
        self.assertAlmostEqual(0.8, job["probabilities"]["1"], places=12)
        self.assertAlmostEqual(0.2, job["probabilities"]["0"], places=12)

    def test_two_qubit_gate_applies_the_channel_to_each_target(self):
        qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n"
        self.service.create_circuit(circuit_document(qasm, "bell2"), self.key())
        job = self.service.simulate(
            "bell2", {"shots": 10, "seed": 1, "noise": self.noise(0.25)}, self.key()
        )
        expected = {"00": 13 / 36, "01": 5 / 36, "10": 5 / 36, "11": 13 / 36}
        for label, probability in expected.items():
            self.assertAlmostEqual(probability, job["probabilities"][label], places=12)

    def test_measure_does_not_trigger_the_channel(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\n"
            "x q[0];\nmeasure q[0] -> c[0];\n"
        )
        self.service.create_circuit(circuit_document(qasm, "m"), self.key())
        job = self.service.simulate(
            "m", {"shots": 10, "seed": 1, "noise": self.noise(1.0)}, self.key()
        )
        # p = 1 after the x gate leaves P(1) = 1/3; the measure adds no noise
        self.assertAlmostEqual(1 / 3, job["probabilities"]["1"], places=12)
        self.assertEqual(1, job["measured_bits"])

    def test_noise_validation_errors(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        with self.assertRaisesRegex(ValidationError, "noise must be a JSON object"):
            self.service.simulate("bell", {"noise": "depolarizing"}, self.key())
        with self.assertRaisesRegex(ValidationError, "noise must be a JSON object"):
            self.service.simulate("bell", {"noise": None}, self.key())
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            self.service.simulate(
                "bell", {"noise": {**self.noise(), "extra": 1}}, self.key()
            )
        with self.assertRaisesRegex(ValidationError, "must contain a type"):
            self.service.simulate("bell", {"noise": {"probability": 0.1}}, self.key())
        with self.assertRaisesRegex(ValidationError, "must contain a probability"):
            self.service.simulate("bell", {"noise": {"type": "depolarizing"}}, self.key())
        with self.assertRaisesRegex(ValidationError, 'type must be "depolarizing"'):
            self.service.simulate(
                "bell", {"noise": {"type": "bitflip", "probability": 0.1}}, self.key()
            )
        with self.assertRaisesRegex(ValidationError, "probability must be a number"):
            self.service.simulate(
                "bell", {"noise": {"type": "depolarizing", "probability": True}}, self.key()
            )
        with self.assertRaisesRegex(ValidationError, "probability must be a number"):
            self.service.simulate(
                "bell", {"noise": {"type": "depolarizing", "probability": "0.1"}}, self.key()
            )
        with self.assertRaisesRegex(ValidationError, "probability must be a finite number"):
            self.service.simulate(
                "bell",
                {"noise": {"type": "depolarizing", "probability": float("nan")}},
                self.key(),
            )
        with self.assertRaisesRegex(ValidationError, "probability must be between 0 and 1"):
            self.service.simulate(
                "bell", {"noise": {"type": "depolarizing", "probability": 1.5}}, self.key()
            )
        with self.assertRaisesRegex(ValidationError, "probability must be between 0 and 1"):
            self.service.simulate(
                "bell", {"noise": {"type": "depolarizing", "probability": -0.1}}, self.key()
            )

    def test_boundary_probabilities_are_accepted(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        for probability in (0, 1):
            job = self.service.simulate(
                "bell",
                {"shots": 5, "seed": 1, "noise": self.noise(probability)},
                self.key(),
            )
            self.assertEqual(float(probability), job["noise"]["probability"])

    def test_noise_is_limited_to_eight_qubits(self):
        qasm = "OPENQASM 2.0;\nqreg q[9];\nh q[0];\n"
        self.service.create_circuit(circuit_document(qasm, "wide"), self.key())
        with self.assertRaisesRegex(ValidationError, "at most 8 qubits"):
            self.service.simulate("wide", {"noise": self.noise()}, self.key())
        # the failed request created no job, and the quiet 16-qubit bound is intact
        quiet = self.service.simulate("wide", {"shots": 1}, self.key())
        self.assertNotIn("noise", quiet)
        self.assertEqual(9, quiet["qubits"])

    def test_noisy_simulation_of_unknown_circuit_is_not_found(self):
        with self.assertRaises(NotFoundError):
            self.service.simulate("missing", {"noise": self.noise()}, self.key())


PARAMETER_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
rx(theta) q[0];
measure q[0] -> c[0];
"""


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = QubitLane(str(Path(self.directory.name) / "qubitlane.db"))
        self.counter = 0

    def tearDown(self):
        self.directory.cleanup()

    def key(self, prefix: str = "k") -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}"

    def create_parameter_circuit(self, qasm: str = PARAMETER_QASM, circuit_id: str = "sweep"):
        return self.service.create_circuit({"id": circuit_id, "qasm": qasm}, self.key())

    def job_count(self) -> int:
        row = self.service.store.connection.execute(
            "SELECT COUNT(*) AS n FROM jobs"
        ).fetchone()
        return row["n"]

    def test_parameterized_circuit_document(self):
        document = self.create_parameter_circuit()
        self.assertEqual(["theta"], document["parameters"])
        (operation,) = [
            operation for operation in document["operations"] if operation["name"] == "rx"
        ]
        self.assertEqual("theta", operation["expression"])
        self.assertNotIn("angle", operation)

    def test_angle_expression_forms(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[1];\n"
            "rx(theta/2 + pi) q[0];\nry(2*phi) q[0];\nrz(-phi) q[0];\n"
        )
        circuit = parse_circuit(qasm, "expr")
        self.assertEqual(("theta", "phi"), circuit.parameters)
        bound = circuit.bind({"theta": math.pi, "phi": 0.25})
        rx, ry, rz = bound.operations
        self.assertAlmostEqual(1.5 * math.pi, rx.angle, places=12)
        self.assertAlmostEqual(0.5, ry.angle, places=12)
        self.assertAlmostEqual(-0.25, rz.angle, places=12)

    def test_batch_scenarios_are_indexed_in_order(self):
        self.create_parameter_circuit()
        job = self.service.simulate(
            "sweep",
            {"shots": 100, "seed": 5, "parameters": {"theta": [0.0, math.pi, 2 * math.pi]}},
            self.key(),
        )
        self.assertEqual("completed", job["state"])
        self.assertEqual("sweep", job["circuit_id"])
        self.assertEqual(100, job["shots"])
        self.assertEqual(5, job["seed"])
        self.assertEqual(["theta"], job["parameters"])
        self.assertEqual(3, job["scenario_count"])
        self.assertEqual([0, 1, 2], [scenario["index"] for scenario in job["scenarios"]])
        self.assertEqual(
            [{"theta": 0.0}, {"theta": math.pi}, {"theta": 2 * math.pi}],
            [scenario["values"] for scenario in job["scenarios"]],
        )
        for scenario in job["scenarios"]:
            self.assertEqual(100, sum(scenario["counts"].values()))
        # rx(0) keeps |0>, rx(pi) flips to |1>, rx(2*pi) returns to |0>
        self.assertEqual({"0": 100}, job["scenarios"][0]["counts"])
        self.assertEqual({"1": 100}, job["scenarios"][1]["counts"])
        self.assertEqual({"0": 100}, job["scenarios"][2]["counts"])

    def test_batch_is_deterministic_and_rereads_without_rerunning(self):
        self.create_parameter_circuit()
        body = {"shots": 200, "seed": 3, "parameters": {"theta": [0.1, 0.2, 0.3, 0.4]}}
        first = self.service.simulate("sweep", body, self.key())
        repeated = self.service.simulate("sweep", body, self.key())
        self.assertEqual(first, repeated)
        self.assertEqual(first, self.service.get_job(first["id"]))
        other_seed = self.service.simulate("sweep", {**body, "seed": 4}, self.key())
        self.assertNotEqual(first["id"], other_seed["id"])
        self.assertEqual(
            [scenario["values"] for scenario in first["scenarios"]],
            [scenario["values"] for scenario in other_seed["scenarios"]],
        )

    def test_single_scenario_matches_the_plain_job(self):
        self.create_parameter_circuit()
        fixed = PARAMETER_QASM.replace("rx(theta)", "rx(0.7)")
        self.service.create_circuit({"id": "fixed", "qasm": fixed}, self.key())
        plain = self.service.simulate("fixed", {"shots": 300, "seed": 11}, self.key())
        batch = self.service.simulate(
            "sweep",
            {"shots": 300, "seed": 11, "parameters": {"theta": [0.7]}},
            self.key(),
        )
        self.assertEqual(1, batch["scenario_count"])
        self.assertEqual(plain["counts"], batch["scenarios"][0]["counts"])
        self.assertNotEqual(plain["id"], batch["id"])

    def test_parameter_names_must_resolve(self):
        self.create_parameter_circuit()
        with self.assertRaises(ParamUndefinedError) as caught:
            self.service.simulate(
                "sweep", {"parameters": {"omega": [0.1]}}, self.key()
            )
        self.assertEqual("PARAM_UNDEFINED", caught.exception.code)
        with self.assertRaises(ParamUndefinedError):
            self.service.simulate(
                "sweep",
                {"parameters": {"theta": [0.1], "omega": [0.1]}},
                self.key(),
            )
        self.assertEqual(0, self.job_count())

    def test_missing_binding_is_param_undefined(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nrx(theta) q[0];\nry(phi) q[0];\n"
        self.create_parameter_circuit(qasm, "two")
        with self.assertRaises(ParamUndefinedError):
            self.service.simulate("two", {"parameters": {"theta": [0.1]}}, self.key())
        self.assertEqual(0, self.job_count())

    def test_plain_circuit_rejects_bindings(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        with self.assertRaises(ParamUndefinedError):
            self.service.simulate(
                "bell", {"parameters": {"theta": [0.1]}}, self.key()
            )
        self.assertEqual(0, self.job_count())

    def test_parameter_array_validation(self):
        self.create_parameter_circuit()
        with self.assertRaises(ParamArrayEmptyError) as caught:
            self.service.simulate("sweep", {"parameters": {"theta": []}}, self.key())
        self.assertEqual("PARAM_ARRAY_EMPTY", caught.exception.code)
        with self.assertRaisesRegex(ValidationError, "array of numbers"):
            self.service.simulate("sweep", {"parameters": {"theta": 0.1}}, self.key())
        with self.assertRaisesRegex(ValidationError, "JSON object"):
            self.service.simulate("sweep", {"parameters": [0.1]}, self.key())
        self.assertEqual(0, self.job_count())

    def test_multi_parameter_length_mismatch(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nrx(theta) q[0];\nry(phi) q[0];\n"
        self.create_parameter_circuit(qasm, "two")
        with self.assertRaises(ParamArrayLengthMismatchError):
            self.service.simulate(
                "two",
                {"parameters": {"theta": [0.1, 0.2], "phi": [0.3]}},
                self.key(),
            )
        self.assertEqual(0, self.job_count())

    def test_parameter_values_must_be_finite_numbers(self):
        self.create_parameter_circuit()
        for bad in (float("nan"), float("inf"), float("-inf"), "0.1", True, None):
            with self.assertRaises(ParamValueInvalidError) as caught:
                self.service.simulate(
                    "sweep", {"parameters": {"theta": [bad]}}, self.key()
                )
            self.assertEqual("PARAM_VALUE_INVALID", caught.exception.code)
        self.assertEqual(0, self.job_count())

    def test_batch_shots_must_be_a_positive_integer(self):
        self.create_parameter_circuit()
        for bad in (0, -3, 1.5, True, "10"):
            with self.assertRaises(ShotsInvalidError) as caught:
                self.service.simulate(
                    "sweep",
                    {"shots": bad, "parameters": {"theta": [0.1]}},
                    self.key(),
                )
            self.assertEqual("SHOTS_INVALID", caught.exception.code)
        self.assertEqual(0, self.job_count())

    def test_batch_limit_is_enforced_before_any_job_exists(self):
        self.create_parameter_circuit()
        with self.assertRaises(BatchLimitExceededError) as caught:
            self.service.simulate(
                "sweep",
                {"shots": 60_000, "parameters": {"theta": [0.1, 0.2]}},
                self.key(),
            )
        self.assertEqual("BATCH_LIMIT_EXCEEDED", caught.exception.code)
        with self.assertRaises(BatchLimitExceededError):
            self.service.simulate(
                "sweep",
                {"shots": 100_001, "parameters": {"theta": [0.1]}},
                self.key(),
            )
        self.assertEqual(0, self.job_count())
        # exactly at the limit is accepted
        job = self.service.simulate(
            "sweep",
            {"shots": 50_000, "parameters": {"theta": [0.1, 0.2]}},
            self.key(),
        )
        self.assertEqual(2, job["scenario_count"])

    def test_batch_errors_do_not_consume_the_idempotency_key(self):
        self.create_parameter_circuit()
        with self.assertRaises(ParamArrayEmptyError):
            self.service.simulate("sweep", {"parameters": {"theta": []}}, "reuse")
        job = self.service.simulate(
            "sweep", {"parameters": {"theta": [0.1]}}, "reuse"
        )
        self.assertEqual(1, job["scenario_count"])

    def test_plain_simulation_is_unchanged_without_bindings(self):
        self.service.create_circuit(circuit_document(BELL, "bell"), self.key())
        job = self.service.simulate("bell", {"shots": 10, "seed": 1}, self.key())
        self.assertIn("probabilities", job)
        self.assertIn("counts", job)
        self.assertNotIn("scenarios", job)
        self.assertNotIn("parameters", job)

    def test_unbound_parameterized_circuit_is_a_validation_error(self):
        self.create_parameter_circuit()
        with self.assertRaisesRegex(ValidationError, "unbound parameter"):
            self.service.simulate("sweep", {"shots": 10}, self.key())
        with self.assertRaisesRegex(ValidationError, "unbound parameter"):
            self.service.get_statevector("sweep")

    def test_batch_with_noise(self):
        self.create_parameter_circuit()
        job = self.service.simulate(
            "sweep",
            {
                "shots": 50,
                "seed": 2,
                "noise": {"type": "depolarizing", "probability": 0.1},
                "parameters": {"theta": [0.0, math.pi]},
            },
            self.key(),
        )
        self.assertEqual({"type": "depolarizing", "probability": 0.1}, job["noise"])
        self.assertEqual(2, job["scenario_count"])
        for scenario in job["scenarios"]:
            self.assertEqual(50, sum(scenario["counts"].values()))

    def test_batch_job_survives_a_new_service_instance(self):
        path = str(Path(self.directory.name) / "shared.db")
        first = QubitLane(path)
        first.create_circuit({"id": "sweep", "qasm": PARAMETER_QASM}, self.key())
        job = first.simulate(
            "sweep", {"shots": 20, "parameters": {"theta": [0.0, 1.0]}}, self.key()
        )
        second = QubitLane(path)
        self.assertEqual(job, second.get_job(job["id"]))


class U3Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = QubitLane(str(Path(self.directory.name) / "qubitlane.db"))
        self.counter = 0

    def tearDown(self):
        self.directory.cleanup()

    def key(self, prefix: str = "k") -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}"

    def test_u3_document_constant_and_parameter_angles(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[2];\n"
            "u3(pi/2, 0.1, -pi) q[0];\n"
            "cu3(theta, phi, lam) q[0], q[1];\n"
        )
        document = self.service.create_circuit({"id": "u", "qasm": qasm}, self.key())
        u3, cu3 = document["operations"]
        self.assertEqual("u3", u3["name"])
        self.assertEqual([0], u3["targets"])
        self.assertAlmostEqual(math.pi / 2, u3["angles"][0], places=12)
        self.assertEqual(0.1, u3["angles"][1])
        self.assertAlmostEqual(-math.pi, u3["angles"][2], places=12)
        self.assertNotIn("angle", u3)
        self.assertEqual("cu3", cu3["name"])
        self.assertEqual([0, 1], cu3["targets"])
        self.assertEqual(["theta", "phi", "lam"], cu3["angles"])
        self.assertEqual(["theta", "phi", "lam"], document["parameters"])

    def test_mixed_constant_and_parameter_angles(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nu3(theta, pi, 0.5) q[0];\n"
        document = self.service.create_circuit({"id": "m", "qasm": qasm}, self.key())
        self.assertEqual(["theta"], document["parameters"])
        self.assertEqual(["theta", math.pi, 0.5], document["operations"][0]["angles"])
        circuit = parse_circuit(qasm, "m")
        bound = circuit.bind({"theta": 0.7})
        angles = bound.operations[0].angles
        self.assertAlmostEqual(0.7, angles[0], places=12)
        self.assertAlmostEqual(math.pi, angles[1], places=12)
        self.assertEqual(0.5, angles[2])

    def test_parameters_use_first_appearance_order(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[2];\n"
            "u3(b, a, pi) q[0];\ncu3(a, c, b) q[0], q[1];\n"
        )
        circuit = parse_circuit(qasm, "o")
        self.assertEqual(("b", "a", "c"), circuit.parameters)

    def test_u3_matches_known_gates_up_to_global_phase(self):
        cases = (
            ("u3(pi, 0, pi) q[0]", "x q[0]"),
            ("u3(pi/2, 0, pi) q[0]", "h q[0]"),
            ("u3(0.9, 0, 0) q[0]", "ry(0.9) q[0]"),
            ("u3(0.9, -pi/2, pi/2) q[0]", "rx(0.9) q[0]"),
        )
        for u3_statement, other in cases:
            a = simulate(parse_circuit(f"OPENQASM 2.0;\nqreg q[1];\n{u3_statement};\n", "a"))
            b = simulate(parse_circuit(f"OPENQASM 2.0;\nqreg q[1];\n{other};\n", "b"))
            self.assertAlmostEqual(abs(a[0]), abs(b[0]), places=12)
            self.assertAlmostEqual(abs(a[1]), abs(b[1]), places=12)

    def test_cu3_is_controlled(self):
        # control 0: target untouched
        q0 = "OPENQASM 2.0;\nqreg q[2];\ncu3(0.4, 0.5, 0.6) q[1], q[0];\n"
        state0 = simulate(parse_circuit(q0, "c0"))
        self.assertAlmostEqual(1.0, abs(state0[0]), places=12)

        # control 1: equals the bare u3 on the target inside the 11 sector
        q1 = "OPENQASM 2.0;\nqreg q[2];\nx q[1];\ncu3(0.4, 0.5, 0.6) q[1], q[0];\n"
        state1 = simulate(parse_circuit(q1, "c1"))
        q2 = "OPENQASM 2.0;\nqreg q[1];\nu3(0.4, 0.5, 0.6) q[0];\n"
        bare = simulate(parse_circuit(q2, "u"))
        self.assertAlmostEqual(bare[0], state1[2], places=12)
        self.assertAlmostEqual(bare[1], state1[3], places=12)

    def test_u3_and_cu3_parse_errors_carry_line_numbers(self):
        cases = (
            "OPENQASM 2.0;\nqreg q[2];\nu3(pi, 0) q[0];\n",
            "OPENQASM 2.0;\nqreg q[2];\nu3(pi, 0, 0, 0) q[0];\n",
            "OPENQASM 2.0;\nqreg q[2];\nu3(pi, , 0) q[0];\n",
            "OPENQASM 2.0;\nqreg q[2];\nu3(pi, 0, x?) q[0];\n",
            "OPENQASM 2.0;\nqreg q[2];\ncu3(pi, 0, pi) q[0], q[0];\n",
            "OPENQASM 2.0;\nqreg q[2];\nu3(pi, 0, pi) q[5];\n",
            "OPENQASM 2.0;\nqreg q[2];\ncu3(pi, 0, pi) q[0], q[5];\n",
        )
        for qasm in cases:
            with self.assertRaisesRegex(ParseError, r"line 3:"):
                parse_circuit(qasm, "bad")

    def test_unbound_u3_is_a_validation_error(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\nu3(theta, phi, lam) q[0];\n"
        self.service.create_circuit({"id": "u", "qasm": qasm}, self.key())
        with self.assertRaisesRegex(ValidationError, "unbound parameter"):
            self.service.simulate("u", {"shots": 10}, self.key())
        with self.assertRaisesRegex(ValidationError, "unbound parameter"):
            self.service.get_statevector("u")

    def test_u3_batch_and_noise(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[2];\n"
            "h q[0];\ncu3(theta, 0, pi) q[0], q[1];\n"
        )
        self.service.create_circuit({"id": "cu", "qasm": qasm}, self.key())
        quiet = self.service.simulate(
            "cu",
            {"shots": 100, "seed": 4, "parameters": {"theta": [0.0, math.pi]}},
            self.key(),
        )
        self.assertEqual(2, quiet["scenario_count"])
        # cu3(pi,0,pi) flips the target when the control is 1: scenario 1 entangles
        self.assertGreater(quiet["scenarios"][1]["counts"].get("11", 0), 0)
        noisy = self.service.simulate(
            "cu",
            {
                "shots": 50,
                "seed": 2,
                "noise": {"type": "depolarizing", "probability": 0.1},
                "parameters": {"theta": [0.0, math.pi]},
            },
            self.key(),
        )
        self.assertEqual(0.1, noisy["noise"]["probability"])
        for scenario in noisy["scenarios"]:
            self.assertEqual(50, sum(scenario["counts"].values()))

    def test_u3_noise_matches_statevector_at_zero_probability(self):
        qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncu3(0.4, 0.5, 0.6) q[0], q[1];\n"
        self.service.create_circuit({"id": "n", "qasm": qasm}, self.key())
        quiet = self.service.simulate("n", {"shots": 40, "seed": 5}, self.key())
        noisy = self.service.simulate(
            "n",
            {"shots": 40, "seed": 5, "noise": {"type": "depolarizing", "probability": 0.0}},
            self.key(),
        )
        self.assertEqual(quiet["probabilities"], noisy["probabilities"])
        self.assertEqual(quiet["counts"], noisy["counts"])


if __name__ == "__main__":
    unittest.main()