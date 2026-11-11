PY ?= python

.PHONY: install install-full test all pd weights allocation ablation scaling visuals run dashboard
install:
	$(PY) -m pip install -e ".[dev]"
install-full:
	$(PY) -m pip install -e ".[dev,full]"
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
scaling:
	$(PY) scripts/06_scaling_and_noise.py
visuals:
	$(PY) scripts/05_generate_visuals.py
run:
	$(PY) scripts/run_qfar.py --cohort 0 --qiskit --zne
dashboard:
	$(PY) scripts/07_dashboard.py
all: test pd weights allocation ablation scaling visuals
