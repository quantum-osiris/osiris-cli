OPENQASM 2.0;
include "qelib1.inc";

// Quantinuum QCCD Example 3: [[7,1,3]] Steane CSS Syndrome Decoding & Ancilla Reuse
// Data qubits: q[0..6] | Reusable syndrome ancillas: q[7..9]
// Demonstrates why all-to-all QCCD ion transport outperforms heavy-hex on non-planar QEC:
//   - Fano-plane weight-4 stabilizers couple non-adjacent qubits with 0 SWAPs on H2/Helios
//   - Full 7-syndrome real-time feedforward bit-flip correction
//   - Mid-circuit reset reuses q[7..9] for X-stabilizer verification (10 qubits vs 13)

qreg q[10];
creg z_syn[3];
creg x_syn[3];
creg data[7];

// Encode Steane |0_L> across Fano plane
h q[0];
h q[1];
h q[3];
cx q[0], q[2];
cx q[1], q[2];
cx q[0], q[4];
cx q[3], q[4];
cx q[1], q[5];
cx q[3], q[5];
cx q[0], q[6];
cx q[1], q[6];
cx q[3], q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];

// Inject single-qubit X fault on q[6] (Hamming index 7 -> z_syn == 7)
x q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];

// Extract weight-4 Z syndromes into q[7..9] (Hamming parity check matrix H)
// Bit 0 (weight 1): q[0], q[2], q[4], q[6]
cx q[0], q[7];
cx q[2], q[7];
cx q[4], q[7];
cx q[6], q[7];

// Bit 1 (weight 2): q[1], q[2], q[5], q[6]
cx q[1], q[8];
cx q[2], q[8];
cx q[5], q[8];
cx q[6], q[8];

// Bit 2 (weight 4): q[3], q[4], q[5], q[6]
cx q[3], q[9];
cx q[4], q[9];
cx q[5], q[9];
cx q[6], q[9];

measure q[7] -> z_syn[0];
measure q[8] -> z_syn[1];
measure q[9] -> z_syn[2];
reset q[7];
reset q[8];
reset q[9];

// Classical feedforward Hamming decoder (z_syn in 1..7 identifies flipped qubit)
if(z_syn == 1) x q[0];
if(z_syn == 2) x q[1];
if(z_syn == 3) x q[2];
if(z_syn == 4) x q[3];
if(z_syn == 5) x q[4];
if(z_syn == 6) x q[5];
if(z_syn == 7) x q[6];

// Reuse reset ancillas q[7..9] for weight-4 X-stabilizer verification
h q[7];
h q[8];
h q[9];

cx q[7], q[0];
cx q[7], q[2];
cx q[7], q[4];
cx q[7], q[6];

cx q[8], q[1];
cx q[8], q[2];
cx q[8], q[5];
cx q[8], q[6];

cx q[9], q[3];
cx q[9], q[4];
cx q[9], q[5];
cx q[9], q[6];

h q[7];
h q[8];
h q[9];

measure q[7] -> x_syn[0];
measure q[8] -> x_syn[1];
measure q[9] -> x_syn[2];
reset q[7];
reset q[8];
reset q[9];

// Final code-space data readout
measure q[0] -> data[0];
measure q[1] -> data[1];
measure q[2] -> data[2];
measure q[3] -> data[3];
measure q[4] -> data[4];
measure q[5] -> data[5];
measure q[6] -> data[6];
