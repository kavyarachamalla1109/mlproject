PYTHON ?= python
CONFIG ?= configs/train_config.yaml

.PHONY: install test train clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

train:
	$(PYTHON) -m src.main --config $(CONFIG)

clean:
	rm -rf .pytest_cache
	rm -f artifacts/*.json artifacts/*.md artifacts/*.csv artifacts/*.pkl
	rm -f data/processed/*
	touch artifacts/.gitkeep data/processed/.gitkeep
