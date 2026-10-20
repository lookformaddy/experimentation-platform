.PHONY: test demo serve install
install:
	pip install -r requirements.txt && pip install -e .
test:
	PYTHONPATH=src python -m pytest -q
demo:
	PYTHONPATH=src python examples/run_demo.py
serve:
	uvicorn experimentation_platform.serving:app --host 127.0.0.1 --port 8000
