PY ?= python

.PHONY: install test all pd weights allocation ablation
install:
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest
pd:
	$(PY) scripts/01_train_pd.py
weights:
	$(PY) scripts/02_weight_ablation.py
allocation:
	$(PY) scripts/03_allocation_experiment.py --jobs 4
ablation:
	$(PY) scripts/04_component_ablation.py
all: test pd weights allocation ablation
