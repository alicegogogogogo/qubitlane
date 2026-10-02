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

    def create(self, qasm: str, circuit_id: str) -> None:
        self.service.create_circuit(circuit_document(qasm, circuit_id), self.key())

    def test_noisy_job_reports_noise_and_stays_normalised(self):
        self.create(BELL, "bell")
        job = self.service.simulate(
            "bell",
            {"shots": 200, "seed": 4, "noise": {"type": "depolarizing", "probability": 0.1}},
            self.key(),
        )
        self.assertEqual({"type": "depolarizing", "probability": 0.1}, job["noise"])
        self.assertEqual({"00", "01", "10", "11"}, set(job["probabilities"]))
        self.assertAlmostEqual(1.0, sum(job["probabilities"].values()), places=9)
        self.assertLess(job["normalization_error"], TOLERANCE)
        self.assertEqual(200, sum(job["counts"].values()))
        repeat = self.service.simulate(
            "bell",
            {"shots": 200, "seed": 4, "noise": {"type": "depolarizing", "probability": 0.1}},
            self.key(),
        )
        self.assertEqual(job, repeat)

    def test_zero_probability_matches_the_noiseless_result(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[2];\n"
            "h q[0];\nt q[1];\nrx(0.7) q[0];\nry(1.1) q[1];\nrz(-0.3) q[0];\n"
            "cx q[0], q[1];\ncz q[1], q[0];\n"
        )
        self.create(qasm, "gates")
        plain = self.service.simulate("gates", {"shots": 10, "seed": 1}, self.key())
        noisy = self.service.simulate(
            "gates",
            {"shots": 10, "seed": 1, "noise": {"type": "depolarizing", "probability": 0}},
            self.key(),
        )
        self.assertEqual(plain["probabilities"], noisy["probabilities"])
        self.assertNotEqual(plain["id"], noisy["id"])

    def test_measure_does_not_trigger_the_channel(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"
        self.create(qasm, "one")
        job = self.service.simulate(
            "one",
            {"shots": 30, "seed": 2, "noise": {"type": "depolarizing", "probability": 1}},
            self.key(),
        )
        self.assertAlmostEqual(2 / 3, job["probabilities"]["0"], places=12)
        self.assertAlmostEqual(1 / 3, job["probabilities"]["1"], places=12)

    def test_two_qubit_gate_applies_the_channel_to_each_target_once(self):
        qasm = "OPENQASM 2.0;\nqreg q[2];\nx q[0];\ncx q[0], q[1];\n"
        self.create(qasm, "pair")
        job = self.service.simulate(
            "pair",
            {"shots": 30, "seed": 2, "noise": {"type": "depolarizing", "probability": 1}},
            self.key(),
        )
        expected = {"00": 2 / 9, "01": 2 / 9, "10": 2 / 9, "11": 1 / 3}
        for label, probability in expected.items():
            self.assertAlmostEqual(probability, job["probabilities"][label], places=12)

    def test_noise_is_part_of_the_job_id(self):
        self.create(BELL, "bell")
        plain = self.service.simulate("bell", {"shots": 10, "seed": 1}, self.key())
        quiet = self.service.simulate(
            "bell",
            {"shots": 10, "seed": 1, "noise": {"type": "depolarizing", "probability": 0.01}},
            self.key(),
        )
        same = self.service.simulate(
            "bell",
            {"shots": 10, "seed": 1, "noise": {"type": "depolarizing", "probability": 0.010}},
            self.key(),
        )
        louder = self.service.simulate(
            "bell",
            {"shots": 10, "seed": 1, "noise": {"type": "depolarizing", "probability": 0.02}},
            self.key(),
        )
        self.assertNotEqual(plain["id"], quiet["id"])
        self.assertEqual(quiet["id"], same["id"])
        self.assertEqual(quiet, same)
        self.assertNotEqual(quiet["id"], louder["id"])
        self.assertEqual(quiet, self.service.get_job(quiet["id"]))

    def test_idempotent_replay_returns_the_first_noisy_job(self):
        self.create(BELL, "bell")
        body = {"shots": 50, "seed": 9, "noise": {"type": "depolarizing", "probability": 0.5}}
        first = self.service.simulate("bell", body, "noisy-key")
        second = self.service.simulate("bell", body, "noisy-key")
        self.assertEqual(first, second)

    def test_measured_bits_projection_still_applies_with_noise(self):
        qasm = (
            "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\n"
            "x q[1];\nmeasure q[1] -> c[0];\n"
        )
        self.create(qasm, "proj")
        job = self.service.simulate(
            "proj",
            {"shots": 16, "seed": 3, "noise": {"type": "depolarizing", "probability": 0}},
            self.key(),
        )
        self.assertEqual(1, job["measured_bits"])
        self.assertEqual({"1": 16}, job["counts"])

    def test_noise_validation_errors_create_no_job(self):
        self.create(BELL, "bell")
        bad_bodies = [
            ({"noise": "depolarizing"}, "JSON object"),
            ({"noise": None}, "JSON object"),
            ({"noise": {"type": "depolarizing", "probability": 0.1, "p": 1}}, "unknown fields"),
            ({"noise": {"probability": 0.1}}, "must contain a type"),
            ({"noise": {"type": "amplitude", "probability": 0.1}}, "type must be"),
            ({"noise": {"type": "depolarizing"}}, "must contain a probability"),
            ({"noise": {"type": "depolarizing", "probability": True}}, "probability must be a number"),
            ({"noise": {"type": "depolarizing", "probability": "0.1"}}, "probability must be a number"),
            ({"noise": {"type": "depolarizing", "probability": -0.1}}, "between 0 and 1"),
            ({"noise": {"type": "depolarizing", "probability": 1.1}}, "between 0 and 1"),
            ({"noise": {"type": "depolarizing", "probability": float("nan")}}, "between 0 and 1"),
        ]
        for body, message in bad_bodies:
            with self.subTest(body=body):
                with self.assertRaisesRegex(ValidationError, message):
                    self.service.simulate("bell", body, self.key())
        jobs = self.service.store.connection.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
        self.assertEqual(0, jobs["n"])

    def test_probability_accepts_the_closed_interval(self):
        self.create(BELL, "bell")
        for probability in (0, 1):
            job = self.service.simulate(
                "bell",
                {"shots": 5, "seed": 1, "noise": {"type": "depolarizing", "probability": probability}},
                self.key(),
            )
            self.assertEqual(float(probability), job["noise"]["probability"])

    def test_noise_is_limited_to_eight_qubits(self):
        qasm = "OPENQASM 2.0;\nqreg q[9];\nh q[0];\n"
        self.create(qasm, "wide")
        with self.assertRaisesRegex(ValidationError, "at most 8 qubits"):
            self.service.simulate(
                "wide",
                {"shots": 1, "noise": {"type": "depolarizing", "probability": 0.1}},
                self.key(),
            )
        plain = self.service.simulate("wide", {"shots": 4, "seed": 1}, self.key())
        self.assertEqual(9, plain["qubits"])
        self.assertNotIn("noise", plain)

    def test_noiseless_surface_is_unchanged(self):
        self.create(BELL, "bell")
        job = self.service.simulate("bell", {"shots": 32, "seed": 6}, self.key())
        self.assertNotIn("noise", job)
        self.assertEqual(job, self.service.get_job(job["id"]))
        statevector = self.service.get_statevector("bell")
        self.assertEqual(["00", "11"], [entry["basis"] for entry in statevector["amplitudes"]])


if __name__ == "__main__":
    unittest.main()
