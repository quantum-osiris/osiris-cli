OPENQASM 2.0;
include "qelib1.inc";

// Quantinuum QCCD Example 1: 3-Qubit Bit-Flip QEC with Mid-Circuit Reset & Feedforward
// Data qubits: q[0], q[1], q[2] | Reusable syndrome ancillas: q[3], q[4]
// Demonstrates:
//   1. Logical |1_L> = |111> encoding + injected single-qubit X fault on q[0]
//   2. Mid-circuit projective syndrome measurement + ancilla reset
//   3. Real-time classical feedforward recovery (if(syn == k) x q[i])
//   4. Second-round syndrome verification reusing reset ancillas q[3], q[4]

qreg q[5];
creg syn[2];
creg syn_r2[2];
creg data[3];

// Prepare |1_L> = |111>
x q[0];
cx q[0], q[1];
cx q[0], q[2];
barrier q[0], q[1], q[2];

// Inject single-qubit bit-flip fault on q[0] (|111> -> |011>)
x q[0];
barrier q[0], q[1], q[2];

// Round 1 syndrome extraction: Z0*Z1 -> syn[0], Z1*Z2 -> syn[1]
cx q[0], q[3];
cx q[1], q[3];
cx q[1], q[4];
cx q[2], q[4];
measure q[3] -> syn[0];
measure q[4] -> syn[1];
reset q[3];
reset q[4];

// Real-time classical feedforward recovery
if(syn == 1) x q[0];
if(syn == 2) x q[2];
if(syn == 3) x q[1];

// Round 2 verification reusing reset ancillas q[3], q[4]
cx q[0], q[3];
cx q[1], q[3];
cx q[1], q[4];
cx q[2], q[4];
measure q[3] -> syn_r2[0];
measure q[4] -> syn_r2[1];
reset q[3];
reset q[4];

// Final logical data readout (ideal post-recovery state: data == 111, syn_r2 == 00)
measure q[0] -> data[0];
measure q[1] -> data[1];
measure q[2] -> data[2];
