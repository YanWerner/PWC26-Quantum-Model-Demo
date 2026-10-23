# IQGAN on the Wisconsin Breast Cancer dataset

A Jupyter notebook that trains a small quantum generative model (IQGAN-style: a quantum generator with a fixed swap-test discriminator) on tabular medical data, and shows what the trained generator actually produces, both exactly and with simulated shot noise.

**Notebook:** `IQGAN_breast_cancer.ipynb`

## What the notebook does

1. **Data preparation**: loads the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn (569 patients, 30 features), scales every feature to [0, 1], keeps only the **malignant** patients (label 0, 212 patients), reduces them to `k = 4` principal components with PCA and scales the components to [0, 1] again.
2. **Circuit**: builds a 9-qubit circuit with PennyLane:
   - 1 ancilla qubit
   - 4 data qubits: a real patient is loaded by angle encoding (`2·asin(√x)`, `RY` rotations)
   - 4 generator qubits: one `RX` and one `RY` rotation per qubit, 8 trainable parameters
   - a **swap test** between data and generator qubits acts as the (fixed, non-trainable) discriminator
3. **Circuit drawing** of the full IQGAN circuit.
4. **Training**: the generator loss is the probability of measuring 1 on the ancilla; its complement, the probability of measuring 0, is the fidelity between generated and real patient. Adam optimizer with a cosine learning-rate schedule; the best generator weights are kept in memory.
5. **Loss plot** of the generator loss during training.
6. **A generated patient**: the trained generator's output mapped back through the inverse PCA and inverse scaling to the 30 clinical features, shown next to the mean and standard deviation of the real malignant patients.


## Installation

Python 3 with the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

or, from the first cell of the notebook (same folder):

```python
%pip install -r requirements.txt
```

PennyLane **0.45 or newer** is required (the shot-noise chapter uses `qml.set_shots`). The dataset ships with scikit-learn, so no download is needed.

## Running

Open `IQGAN_breast_cancer.ipynb` in Jupyter, JupyterLab or VS Code and run all cells from top to bottom. With the default settings it takes well under a minute on a laptop CPU.

## Settings you can change

All in the data-preparation cell:

| Variable | Default | Meaning |
|---|---|---|
| `k` | 4 | number of PCA components = data qubits = generator qubits (circuit size is `2k + 1` qubits) |
| `n_epochs` | 10 | training epochs |
| `batch_size` | 32 | patients per batch |
| `learning_rate` | 0.001 | Adam learning rate |
| `seed` | 2408 | random seed |

With 212 malignant patients, one epoch is only 7 batches, so the default 10 epochs are 70 optimizer steps. Increase `n_epochs` for a better-converged generator.

## Notes

- Generated features are mapped back through the inverse PCA. Because only `k` components are kept, some reconstructed clinical values can fall outside the physically possible range (e.g. slightly negative measurements), especially when the generator is not fully trained.
- The x-axis label of the loss plot says "logged every 40 batches". With this dataset the loss is logged on the last batch of each epoch, so effectively once per epoch.
