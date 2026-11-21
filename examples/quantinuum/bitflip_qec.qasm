// 3-qubit bit-flip repetition QEC cycle with mid-circuit syndrome extraction,
// classical feedforward recovery, and ancilla reset for Quantinuum QCCD systems.
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] data;
qubit[2] anc;
bit[2] syn;
bit[3] out;

// Encode logical |+> across 3 data qubits
h data[0];
cx data[0], data[1];
cx data[0], data[2];

// Parity checks: Z0 Z1 -> anc[0], Z1 Z2 -> anc[1]
cx data[0], anc[0];
cx data[1], anc[0];
cx data[1], anc[1];
cx data[2], anc[1];

// Mid-circuit syndrome measurement and ancilla reset for qubit reuse
syn[0] = measure anc[0];
syn[1] = measure anc[1];
reset anc[0];
reset anc[1];

// Real-time classical feedforward bit-flip recovery
if (syn == 1) {
    x data[0];
}
if (syn == 3) {
    x data[1];
}
if (syn == 2) {
    x data[2];
}

// Final data-register readout
out[0] = measure data[0];
out[1] = measure data[1];
out[2] = measure data[2];
