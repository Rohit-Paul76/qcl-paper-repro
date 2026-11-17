# qcl-paper-repro

A learning reproduction of Figures 3 and 4 of Mitarai, Negoro, Kitagawa and Fujii, "Quantum circuit learning" (Physical Review A 98, 032309, 2018).

A fixed 6-qubit circuit with 6 layers is trained with BFGS on a simulator (PennyLane `default.qubit`). The data is encoded into the qubits, a fixed random entangler and trained single-qubit rotations alternate for each layer, and the output is the expectation of Z on the first qubit.

## Files

| File | What it does |
|---|---|
| `data.py` | the inputs and targets for the two experiments |
| `entangler.py` | the fixed random entangling unitary |
| `circuit.py` | encoding, layers and readout |
| `train.py` | losses and the BFGS loop |
| `fig3.py` | fits x^2, e^x, sin(pi x) and |x| with one circuit, retraining the angles and a scale `a` |
| `fig4.py` | classifies a blob inside a ring |
| `figures/` | the plots the two scripts produced |

## Run

```
pip install pennylane scipy matplotlib numpy
python fig3.py
python fig4.py
```

## What my runs gave

- Figure 3: 500 BFGS iterations were used, ending at a loss of about 1.4e-2 (it stopped on the iteration limit, not on convergence).
- Figure 4: training accuracy 0.94 after 135 iterations.

These are single runs with my own random entangler, so the plots differ in detail from the paper's.
