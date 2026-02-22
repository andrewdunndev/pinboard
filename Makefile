VENV    := .venv
PYTHON  := $(VENV)/bin/python3
PIP     := $(VENV)/bin/pip
ACTIVATE := . $(VENV)/bin/activate

.PHONY: help venv layout composite render test clean clean-all

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

venv: $(VENV)/bin/activate ## Create virtual environment and install dependencies

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

layout: venv ## Generate layout.json only
	$(PYTHON) pipeline.py --layout-only

composite: venv ## Generate layout + composited image
	$(PYTHON) pipeline.py --composite-only

render: venv ## Full pipeline: layout + composite + video (~15 min)
	$(PYTHON) pipeline.py

test: venv ## Quick 120s test clip, reusing existing composite (~2 min)
	$(PYTHON) pipeline.py --test --video-only

clean: ## Remove generated output files
	rm -rf output/

clean-all: clean ## Remove output and virtual environment
	rm -rf $(VENV)
