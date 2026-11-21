# Quantinuum Nexus & QCCD Examples

Runnable OpenQASM 2.0 quantum error-correction and syndrome-extraction circuits designed for Quantinuum H2 (`H2-1`, `H2-2`) and `Helios` QCCD trapped-ion architectures.

Each circuit exercises capabilities where QCCD ion transport has a structural hardware advantage over fixed nearest-neighbor planar lattices:
1. **All-to-all two-qubit connectivity** (`0` SWAP routing overhead on non-planar stabilizer graphs).
2. **Mid-circuit projective measurement and optical-pumping `reset`** (ancilla reuse across syndrome rounds).
3. **Real-time classical feedforward (`if (creg == val)`)** within coherence time.

---

## Included Circuits

| File | Qubits | Code / Protocol | Key QCCD Features Exercised |
| --- | --- | --- | --- |
| [`bitflip_qec.qasm`](bitflip_qec.qasm) | 5 (`3` data + `2` ancilla) | `[[3,1,1]]` Bit-Flip QEC | Injected fault on `q[0]`, mid-circuit syndrome readout, ancilla `reset`, `if(syn == k)` recovery, Round-2 verification |
| [`code422_syndrome_detect.qasm`](code422_syndrome_detect.qasm) | 6 (`4` data + `2` ancilla) | `[[4,2,2]]` Error Detection | Non-local $K_{2,4}$ weight-4 $X^{\otimes 4}$ and $Z^{\otimes 4}$ checks (`0` SWAPs on QCCD), two-round ancilla reuse |
| [`steane_7q_x_syndrome.qasm`](steane_7q_x_syndrome.qasm) | 10 (`7` data + `3` ancilla) | `[[7,1,3]]` Steane CSS Code | Fano-plane weight-4 Z & X syndromes, full 7-syndrome Hamming feedforward decoder, 3 ancillas reused for 6 checks |

---

## Commands

### 1. Static `H2-1SC` Gatekeeper + Local QCCD Trajectory Emulation
```bash
osiris verify examples/quantinuum/bitflip_qec.qasm \
  --target H2-1SC \
  --emulate qccd \
  --shots 1024 \
  --record-ledger
```

### 2. Compare All-to-All QCCD (`H2-1`, `Helios`) vs. Heavy-Hex (`ibm_fez`)
```bash
osiris bench examples/quantinuum/steane_7q_x_syndrome.qasm \
  --compare quantinuum_h2_1,quantinuum_helios,ibm_fez \
  --format markdown
```

### 3. Governed Tier 3 Execution Pre-Flight
```bash
# Mint an operator-signed Tier 3 token bound to exact circuit SHA-256, target, shots, and HQC cap
export OSIRIS_TIER3_HMAC_SECRET="local-operator-signing-key"
export OSIRIS_TIER3_TOKEN="$(osiris grant-tier3 examples/quantinuum/bitflip_qec.qasm --target H2-1 --shots 500 --budget-hqc 25 --quiet)"

# Validate pre-flight + HQC budget + cryptographic token
osiris run examples/quantinuum/bitflip_qec.qasm \
  --target H2-1 \
  --shots 500 \
  --budget-hqc 25 \
  --approval-token "$OSIRIS_TIER3_TOKEN"
```

### 4. Verify Cryptographic Receipt Chain Offline
```bash
osiris ledger verify
```
