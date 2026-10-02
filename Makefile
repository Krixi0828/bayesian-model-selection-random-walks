PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup smoke paper e0 e1 e2 e3 e4 e5 e6 e7 bundle clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

smoke:
	$(PY) run_all.py --profile smoke --jobs 2

paper:
	$(PY) run_all.py --profile paper --jobs 4

$(foreach e,e0 e1 e2 e3 e4 e5 e6 e7,$(eval $(e): ; $(PY) run_all.py --profile paper --experiments $(e) --jobs 4))

bundle:
	bash scripts/make_results_bundle.sh paper

clean:
	rm -rf results/smoke results/paper results_bundle_*.zip
