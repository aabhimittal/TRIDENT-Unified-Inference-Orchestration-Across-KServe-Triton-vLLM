.PHONY: install test lint run demo-backends demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check trident tests

run:
	trident --config config/trident.example.yaml

demo-backends:
	python examples/mock_backends.py

# In a second terminal, after `make demo-backends`:
demo:
	trident --config examples/demo.yaml --port 8080
