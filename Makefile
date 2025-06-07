.PHONY: test deploy_modal run dev

install:
	pip install uv && \
	uv venv && \
	source .venv/bin/activate && \
	uv sync && \
	modal setup

deploy_modal:
	modal deploy src/modal_app.py

test_modal:
	uv run test/test_modal.py

run:
	uv run src/app.py

dev:
	gradio src/app.py