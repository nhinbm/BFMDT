# BFMDT

Bi-directional Fusing Monotonic Decision Trees.

Python implementation of the algorithm from:
> "Fusing Monotonic Decision Tree Based on Related Family"
> IEEE Transactions on Knowledge and Data Engineering, 2024

## Setup

```bash
# Clone
git clone git@github.com:nhinbm/BFMDT.git
cd BFMDT

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements/base.txt

# For development
pip install -r requirements/dev.txt

# For running experiments
pip install -r requirements/experiment.txt
```

## Usage

```python
from bfmdt import BFMDTClassifier

clf = BFMDTClassifier(sigma="auto", delta=0.01, max_reducts=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

## Run Experiments

```bash
# Run on a single dataset
python experiments/run_benchmarks.py --dataset wine

# Run on all 18 datasets
python experiments/run_benchmarks.py --dataset all

# Custom parameters
python experiments/run_benchmarks.py --dataset wine --n_folds 5 --output_dir results/
```

## Run Tests

```bash
pytest tests/
```

## Docs

- [PIPELINE.md](docs/PIPELINE.md) -- Algorithm pipeline
- [INTERFACES.md](docs/INTERFACES.md) -- Module interfaces
- [STRUCTURE.md](docs/STRUCTURE.md) -- Project structure and datasets
