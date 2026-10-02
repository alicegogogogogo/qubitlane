import math
import tempfile
import unittest
from pathlib import Path

from qubitlane.errors import ConflictError, NotFoundError, ParseError, ValidationError
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


class BatchTests(unittest.TestCase):
    PARAMETERIZED = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[1];\n"
        "creg c[1];\n"
        "rx(theta) q[0];\n"
        "measure q[0] -> c[0];\n"
    )

    TWO_PARAMETER = (
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "rx(theta) q[0];\n"
        "rz(phi) q[0];\n"
    )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = QubitLane(str(Path(self.directory.name) / "qubitlane.db"))
        self.counter = 0

    def tearDown(self):
        self.directory.cleanup()

    def key(self, prefix: str = "k") -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}"

    def create(self, qasm: str, circuit_id: str) -> None:
        self.service.create_circuit(circuit_document(qasm, circuit_id), self.key())

    def error_code(self, callback, *arguments) -> str:
        with self.assertRaises(ValidationError) as context:
            callback(*arguments)
        return context.exception.code

    def test_parameterized_gate_parses_to_a_parameter_operation(self):
        circuit = parse_circuit(self.PARAMETERIZED, "p")
        rx = circuit.operations[0]
        self.assertEqual("theta", rx.parameter)
        self.assertIsNone(rx.angle)
        document = self.service.create_circuit(
            circuit_document(self.PARAMETERIZED, "p"), self.key()
        )
        self.assertEqual("theta", document["operations"][0]["parameter"])

    def test_batch_returns_scenarios_in_index_order(self):
        self.create(self.PARAMETERIZED, "p")
        job = self.service.simulate(
            "p",
            {"shots": 100, "seed": 3, "parameters": {"theta": [0.0, math.pi]}},
            self.key(),
        )
        self.assertEqual("completed", job["state"])
        self.assertEqual(2, job["scenario_count"])
        self.assertEqual(["theta"], job["parameters"])
        self.assertEqual([0, 1], [scenario["index"] for scenario in job["scenarios"]])
        first, second = job["scenarios"]
        self.assertEqual({"theta": 0.0}, first["values"])
        self.assertEqual({"theta": math.pi}, second["values"])
        # theta = 0 keeps |0>; theta = pi flips to |1>
        self.assertEqual({"0": 100}, first["counts"])
        self.assertEqual({"1": 100}, second["counts"])
        # shots are per scenario, not shared across the batch
        for scenario in job["scenarios"]:
            self.assertEqual(100, sum(scenario["counts"].values()))

    def test_multiple_parameters_bind_positionally(self):
        self.create(self.TWO_PARAMETER, "p2")
        job = self.service.simulate(
            "p2",
            {
                "shots": 50,
                "seed": 1,
                "parameters": {"theta": [0.0, math.pi], "phi": [0.0, 0.0]},
            },
            self.key(),
        )
        self.assertEqual(["theta", "phi"], job["parameters"])
        self.assertEqual(
            [{"theta": 0.0, "phi": 0.0}, {"theta": math.pi, "phi": 0.0}],
            [scenario["values"] for scenario in job["scenarios"]],
        )
        self.assertEqual({"0": 50}, job["scenarios"][0]["counts"])
        self.assertEqual({"1": 50}, job["scenarios"][1]["counts"])

    def test_batch_is_deterministic_per_seed_and_stable_across_seeds(self):
        self.create(self.PARAMETERIZED, "p")
        body = {"shots": 400, "seed": 7, "parameters": {"theta": [0.3, 1.1, 2.4]}}
        first = self.service.simulate("p", body, self.key())
        repeat = self.service.simulate("p", dict(body), self.key())
        self.assertEqual(first, repeat)
        self.assertEqual(first, self.service.get_job(first["id"]))
        other = self.service.simulate("p", {**body, "seed": 8}, self.key())
        self.assertNotEqual(first["id"], other["id"])
        self.assertNotEqual(
            [s["counts"] for s in first["scenarios"]],
            [s["counts"] for s in other["scenarios"]],
        )
        # a different seed never changes the scenario order
        self.assertEqual(
            [s["values"] for s in first["scenarios"]],
            [s["values"] for s in other["scenarios"]],
        )

    def test_batch_job_id_depends_on_the_bindings(self):
        self.create(self.PARAMETERIZED, "p")
        base = {"shots": 10, "seed": 1}
        first = self.service.simulate(
            "p", {**base, "parameters": {"theta": [0.1]}}, self.key()
        )
        same = self.service.simulate(
            "p", {**base, "parameters": {"theta": [0.1]}}, self.key()
        )
        changed = self.service.simulate(
            "p", {**base, "parameters": {"theta": [0.2]}}, self.key()
        )
        other_shots = self.service.simulate(
            "p", {"shots": 11, "seed": 1, "parameters": {"theta": [0.1]}}, self.key()
        )
        self.assertEqual(first["id"], same["id"])
        self.assertNotEqual(first["id"], changed["id"])
        self.assertNotEqual(first["id"], other_shots["id"])

    def test_batch_result_is_reread_without_reexecuting(self):
        self.create(self.PARAMETERIZED, "p")
        body = {"shots": 64, "seed": 5, "parameters": {"theta": [0.2, 0.9]}}
        job = self.service.simulate("p", body, self.key())
        for _ in range(3):
            self.assertEqual(job, self.service.get_job(job["id"]))
        replayed = self.service.simulate("p", body, self.key())
        self.assertEqual(job, replayed)

    def test_unbound_circuit_parameter_is_param_undefined(self):
        self.create(self.PARAMETERIZED, "p")
        code = self.error_code(self.service.simulate, "p", {"shots": 10}, self.key())
        self.assertEqual("PARAM_UNDEFINED", code)

    def test_unknown_binding_name_is_param_undefined(self):
        self.create(self.PARAMETERIZED, "p")
        code = self.error_code(
            self.service.simulate,
            "p",
            {"shots": 10, "parameters": {"theta": [0.1], "gamma": [0.2]}},
            self.key(),
        )
        self.assertEqual("PARAM_UNDEFINED", code)

    def test_binding_on_a_plain_circuit_is_param_undefined(self):
        self.create(BELL, "bell")
        code = self.error_code(
            self.service.simulate,
            "bell",
            {"shots": 10, "parameters": {"theta": [0.1]}},
            self.key(),
        )
        self.assertEqual("PARAM_UNDEFINED", code)

    def test_partially_bound_circuit_is_param_undefined(self):
        self.create(self.TWO_PARAMETER, "p2")
        code = self.error_code(
            self.service.simulate,
            "p2",
            {"shots": 10, "parameters": {"theta": [0.1]}},
            self.key(),
        )
        self.assertEqual("PARAM_UNDEFINED", code)

    def test_array_length_mismatch(self):
        self.create(self.TWO_PARAMETER, "p2")
        code = self.error_code(
            self.service.simulate,
            "p2",
            {"shots": 10, "parameters": {"theta": [0.1, 0.2], "phi": [0.1]}},
            self.key(),
        )
        self.assertEqual("PARAM_ARRAY_LENGTH_MISMATCH", code)

    def test_empty_array_is_rejected(self):
        self.create(self.PARAMETERIZED, "p")
        code = self.error_code(
            self.service.simulate, "p", {"shots": 10, "parameters": {"theta": []}}, self.key()
        )
        self.assertEqual("PARAM_ARRAY_EMPTY", code)

    def test_non_finite_and_non_numeric_values_are_rejected(self):
        self.create(self.PARAMETERIZED, "p")
        for value in (float("nan"), float("inf"), -float("inf"), "0.5", True):
            code = self.error_code(
                self.service.simulate,
                "p",
                {"shots": 10, "parameters": {"theta": [value]}},
                self.key(),
            )
            self.assertEqual("PARAM_VALUE_INVALID", code)

    def test_invalid_shots_in_a_batch_is_shots_invalid(self):
        self.create(self.PARAMETERIZED, "p")
        for shots in (0, -3, 1.5, True, "10"):
            code = self.error_code(
                self.service.simulate,
                "p",
                {"shots": shots, "parameters": {"theta": [0.1]}},
                self.key(),
            )
            self.assertEqual("SHOTS_INVALID", code)

    def test_batch_limit_is_enforced_without_creating_a_job(self):
        self.create(self.PARAMETERIZED, "p")
        code = self.error_code(
            self.service.simulate,
            "p",
            {"shots": 501, "parameters": {"theta": [0.1] * 200}},
            self.key(),
        )
        self.assertEqual("BATCH_LIMIT_EXCEEDED", code)
        # the boundary itself is accepted: 200 * 500 == 100_000
        job = self.service.simulate(
            "p", {"shots": 500, "parameters": {"theta": [0.1] * 200}}, self.key()
        )
        self.assertEqual(200, job["scenario_count"])

    def test_failed_batch_requests_create_no_job(self):
        self.create(self.PARAMETERIZED, "p")
        before = self.service.store.connection.execute(
            "SELECT COUNT(*) AS n FROM jobs"
        ).fetchone()["n"]
        for body in (
            {"shots": 10, "parameters": {"gamma": [0.1]}},
            {"shots": 10, "parameters": {"theta": []}},
            {"shots": 10, "parameters": {"theta": [float("nan")]}},
            {"shots": 0, "parameters": {"theta": [0.1]}},
            {"shots": 100_001, "parameters": {"theta": [0.1]}},
        ):
            with self.assertRaises(ValidationError):
                self.service.simulate("p", body, self.key())
        after = self.service.store.connection.execute(
            "SELECT COUNT(*) AS n FROM jobs"
        ).fetchone()["n"]
        self.assertEqual(before, after)

    def test_plain_requests_keep_their_existing_error_codes(self):
        self.create(BELL, "bell")
        code = self.error_code(self.service.simulate, "bell", {"shots": 0}, self.key())
        self.assertEqual("validation_error", code)

    def test_batch_supports_noise(self):
        self.create(self.PARAMETERIZED, "p")
        job = self.service.simulate(
            "p",
            {
                "shots": 20,
                "seed": 2,
                "noise": {"type": "depolarizing", "probability": 0.1},
                "parameters": {"theta": [0.0, math.pi]},
            },
            self.key(),
        )
        self.assertEqual({"type": "depolarizing", "probability": 0.1}, job["noise"])
        self.assertEqual(2, job["scenario_count"])
        for scenario in job["scenarios"]:
            self.assertEqual(20, sum(scenario["counts"].values()))

    def test_batch_survives_a_new_service_instance(self):
        path = str(Path(self.directory.name) / "shared.db")
        first = QubitLane(path)
        first.create_circuit(circuit_document(self.PARAMETERIZED, "p"), self.key())
        job = first.simulate(
            "p", {"shots": 32, "seed": 4, "parameters": {"theta": [0.4, 0.8]}}, self.key()
        )
        second = QubitLane(path)
        self.assertEqual(job, second.get_job(job["id"]))


if __name__ == "__main__":
    unittest.main()
