.PHONY: test deploy run dev install hf

install:
	pip install -U uv && \
	uv venv && \
	source .venv/bin/activate && \
	uv sync && \
	modal setup

deploy:
	modal deploy src/modal_app.py

test_modal:
	uv run test/test_modal.py

run:deploy
	uv run src/app.py

dev:
	gradio src/app.py

hf:
	chmod 777 hf.sh
	./hf.sh