OPENQASM 2.0;
include "qelib1.inc";

// Quantinuum QCCD Example 2: [[4,2,2]] Error-Detecting Code with Ancilla Reuse
// Data qubits: q[0..3] | Stabilizer ancillas: q[4] (XXXX), q[5] (ZZZZ)
// All-to-all QCCD connectivity executes the non-local K_{2,4} stabilizer checks
// with zero SWAP gates, mid-circuit ancilla reset, and two-round syndrome tracking.

qreg q[6];
creg syn_r1[2];
creg syn_r2[2];
creg out[4];

// Encode logical |10_L> = (|1100> + |0011>) / sqrt(2)
h q[0];
cx q[0], q[1];
cx q[0], q[2];
cx q[0], q[3];
x q[0];
x q[1];
barrier q[0], q[1], q[2], q[3];

// Round 1: Non-local weight-4 XXXX and ZZZZ stabilizer extraction
h q[4];
cx q[4], q[0];
cx q[4], q[1];
cx q[4], q[2];
cx q[4], q[3];
h q[4];

cx q[0], q[5];
cx q[1], q[5];
cx q[2], q[5];
cx q[3], q[5];

measure q[4] -> syn_r1[0];
measure q[5] -> syn_r1[1];
reset q[4];
reset q[5];

// Round 2: Re-verify stabilizers using mid-circuit reset ancillas
h q[4];
cx q[4], q[0];
cx q[4], q[1];
cx q[4], q[2];
cx q[4], q[3];
h q[4];

cx q[0], q[5];
cx q[1], q[5];
cx q[2], q[5];
cx q[3], q[5];

measure q[4] -> syn_r2[0];
measure q[5] -> syn_r2[1];
reset q[4];
reset q[5];

// Final code-space readout
measure q[0] -> out[0];
measure q[1] -> out[1];
measure q[2] -> out[2];
measure q[3] -> out[3];
