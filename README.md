# QubitLane

QubitLane is a small backend that stores OpenQASM 2.0 circuits, simulates them
with an exact statevector, and reports measurement sampling for a requested
number of `shots` under a deterministic seed.

The initial release intentionally supports a compact public contract:

- only a documented OpenQASM 2.0 subset is parsed (`qreg`, `creg`, `h`, `x`,
  `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `u3`, `cx`, `cz`, `cu3`, `measure`);
- the statevector index is **little-endian**: `q[0]` is the least significant
  bit, while every probability key and count key printed by the API is
  **big-endian** (most significant qubit first, which is the usual OpenQASM
  readout order);
- circuits are immutable once created, and every simulation is an append-only
  job;
- sampling uses an explicit `seed`, so the same circuit, `shots`, and `seed`
  always produce byte-identical counts;
- both state-changing POSTs require an `Idempotency-Key` header.

## Requirements

- Python 3.11 or newer
- no third-party runtime dependencies

## Run the service

```bash
PYTHONPATH=src python -m qubitlane.server --host 127.0.0.1 --port 8080 --database qubitlane.db
```

The process prints `QubitLane listening on http://127.0.0.1:8080` after it has
bound the port.

## HTTP API

All request and response bodies are JSON. Unknown fields are rejected with
`validation_error`. Both `POST /circuits` and `POST /circuits/{id}/simulate`
require an `Idempotency-Key` header and answer 400 without one; replaying a key
returns the first stored response instead of re-simulating. The `GET`
endpoints are pure reads.

### Health

```http
GET /health
```

Returns `{"status":"ok"}`.

### Create a circuit

```http
POST /circuits
Idempotency-Key: demo-circuit-1
Content-Type: application/json

{
  "id": "bell",
  "qasm": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
}
```

`id` is optional. Without it the identifier is derived from the source text as
`c-<first 16 hex characters of sha256(qasm)>`, so identical QASM always yields
the same id. With the example above the derived id is `c-f420d748fb8675b2`.
Returns HTTP 201 with the stored circuit:

```json
{"id":"bell","name":"bell","qubits":2,"clbits":2,"bit_order":"little_endian",
 "measurements":[{"qubit":0,"clbit":0},{"qubit":1,"clbit":1}],
 "operations":[{"name":"h","targets":[0]},{"name":"cx","targets":[0,1]},
               {"name":"measure","targets":[0],"clbit":0},
               {"name":"measure","targets":[1],"clbit":1}],
 "source_lines":8,"qasm":"OPENQASM 2.0;\n..."}
```

Creating the same id again with identical QASM returns the stored circuit
(HTTP 201). Creating the same id with different QASM is a `conflict` (409).

### Language subset

`OPENQASM 2.0;` must be the first statement, and the optional
`include "qelib1.inc";` is the only accepted include. The quantum register must
be named `q`, the classical register `c`, and each may be declared once before
any gate. Sizes are limited to 16 qubits and 64 classical bits.

| statement | arity |
| --- | --- |
| `h q[i]`, `x q[i]`, `y q[i]`, `z q[i]`, `s q[i]`, `t q[i]` | one qubit |
| `rx(angle) q[i]`, `ry(angle) q[i]`, `rz(angle) q[i]` | one qubit |
| `u3(theta, phi, lambda) q[i]` | one qubit, three angles |
| `cx q[control], q[target]`, `cz q[control], q[target]` | two distinct qubits |
| `cu3(theta, phi, lambda) q[control], q[target]` | two distinct qubits, three angles |
| `measure q[i] -> c[j]` | requires `creg` |

`angle` accepts a plain number, `pi`, `-pi/2`, `2*pi`, and `pi/4` forms.
Each `u3`/`cu3` angle accepts the same number, `pi`, parameter name, and flat
`+ - * /` expression rules. `u3` implements the OpenQASM
`U(theta, phi, lambda)` unitary (up to a global phase):

```
U = |  cos(theta/2)               -e^{i lambda} sin(theta/2)       |
    |  e^{i phi} sin(theta/2)     e^{i(phi+lambda)} cos(theta/2)   |
```

`cu3` leaves `target` unchanged when `control` is `0` and applies that same
`U` when `control` is `1`; the control and target must name different qubits.
`u3(pi,0,pi)` is `X`, `u3(pi/2,0,pi)` is `H`, `u3(theta,0,0)` is `ry`, and
`u3(theta,-pi/2,pi/2)` is `rx`.

In the stored circuit, a `u3`/`cu3` operation carries `angles`: three numbers
in declaration order when every angle is constant, or three expression texts
when at least one angle names a parameter (a constant angle inside such a list
is still emitted as a number). The circuit's `parameters` lists every named
parameter in first-appearance order.

Statements are separated by `;` and may span lines; `//` starts a comment.
Anything else — `swap`, `ccx`, barriers, custom `gate` declarations, unknown
registers, or an index outside its register — is a `validation_error` whose
message names the 1-based source line, for example
`line 3: qubit index 5 is out of range for a register of size 2`.

### Simulate a circuit

```http
POST /circuits/bell/simulate
Idempotency-Key: demo-simulate-1
Content-Type: application/json

{"shots": 512, "seed": 7}
```

The body is optional. `shots` defaults to `1024` and must be an integer between
1 and 100000; `seed` defaults to `0` and must be an integer between 0 and
`2^63 - 1`. Returns HTTP 201 with the job:

```json
{"id":"j-43fb21b28a2fd972","circuit_id":"bell","state":"completed","shots":512,
 "seed":7,"qubits":2,"bit_order":"little_endian",
 "normalization_error":2.220446049250313e-16,
 "probabilities":{"00":0.5,"01":0.0,"10":0.0,"11":0.5},
 "counts":{"00":260,"11":252},"measured_bits":2,
 "created_at":"2026-10-02T05:52:43.179639Z"}
```

- `probabilities` contains **every** basis state keyed by its big-endian label,
  so the keys always cover `2^qubits` outcomes.
- `counts` contains only the labels that were drawn at least once; the counts
  sum to `shots`.
- `measured_bits` is the projection width of the count keys: `1 +` the highest
  measured classical bit, or `qubits` when the circuit has no `measure`
  statement. Each count key is the big-endian projection of the basis label
  onto the classical bits, where classical bit `j` holds the qubit measured
  into `c[j]` and an unmeasured classical bit stays `0`. For the Bell circuit
  the state `|11>` is reported as `11`, and a circuit that measures
  `q[1] -> c[0]` while `q[1]` is `|1>` reports `01`.
- The job id is derived deterministically from the circuit id, `shots`, and
  `seed` (`j-` plus 16 hex characters), so repeating a request after losing the
  response yields the same job instead of a second sampling run.

#### Optional depolarizing noise

```http
POST /circuits/bell/simulate
Idempotency-Key: demo-simulate-2
Content-Type: application/json

{"shots": 512, "seed": 7, "noise": {"type": "depolarizing", "probability": 0.01}}
```

An optional `noise` object switches the simulation from a pure statevector to
an exact mixed-state (density matrix) evolution. `noise` accepts exactly two
fields: `type`, which must be `"depolarizing"`, and `probability`, a finite
JSON number in the closed interval `[0, 1]` (a boolean is not a number). After
every gate, each of the gate's target qubits passes through the channel
`(1 - p) rho + p/3 (X rho X + Y rho Y + Z rho Z)` once, in statement order;
`measure` statements never trigger the channel. Noisy simulation supports at
most 8 qubits (the quiet limit stays 16).

A noisy job echoes the noise back as `"noise":{"type":"depolarizing",
"probability":...}` and is otherwise identical in shape: `probabilities` still
covers all `2^qubits` big-endian basis states with a sum within `1e-9` of 1,
and `counts` are sampled deterministically from the final probabilities with
the same classical-bit projection. The noise configuration is part of the job
id, so numerically equal configurations share one id while any change in
`probability` yields a different job. Omitting `noise` leaves the job id, the
response, and the counts exactly as before.

### Read the statevector

```http
GET /circuits/bell/statevector
```

```json
{"circuit_id":"bell","qubits":2,"bit_order":"little_endian",
 "normalization_error":2.220446049250313e-16,
 "amplitudes":[{"index":0,"basis":"00","amplitude":{"real":0.707106781187,"imag":0.0},"probability":0.5},
               {"index":3,"basis":"11","amplitude":{"real":0.707106781187,"imag":0.0},"probability":0.5}]}
```

`amplitudes` lists only the basis states whose probability is at least
`1e-9`; every entry carries its `index` (little-endian), its big-endian
`basis` label, the amplitude split into `real`/`imag`, and its probability.
Amplitudes and probabilities are rounded to 12 decimal places.

### Read a job

```http
GET /jobs/j-43fb21b28a2fd972
```

Returns the same document that `simulate` returned.

## Data model

A circuit is created once and never mutated; a job records one completed
sampling run over a circuit. The stored records are:

- `circuit`: `id`, `qubits`, `clbits`, `bit_order`, `measurements`,
  `operations`, `source_lines`, `qasm`;
- `job`: `id`, `circuit_id`, `state` (always `completed`), `shots`, `seed`,
  `qubits`, `bit_order`, `normalization_error`, `probabilities`, `counts`,
  `measured_bits`, `created_at`, plus `noise` when the simulation requested
  depolarizing noise.

## Invariants

- The statevector is normalised: `|sum |alpha|^2 - 1| < 1e-9` after every gate.
  `normalization_error` reports the measured defect, which is a small number
  such as `2.220446049250313e-16` and is always below the tolerance.
- Gate application preserves the `2^qubits` dimension of the statevector.
- Identical `(circuit, shots, seed)` always produce identical `counts`, and
  `|counts| == shots`.
- Out-of-range qubit or classical bit indices, unknown gates, and unknown
  request fields are rejected as `validation_error` rather than ignored.

## Errors

Errors use this shape:

```json
{"error":{"code":"validation_error","message":"line 3: qubit index 5 is out of range for a register of size 2"}}
```

Validation errors return 400, missing circuits or jobs return 404, and reusing
an existing circuit id with different QASM returns 409.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
