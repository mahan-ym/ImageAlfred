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

requirements:
	uv pip compile --no-annotate pyproject.toml --no-deps --no-strip-extras --no-header \
	| sed -E 's/([a-zA-Z0-9_-]+(\[[a-zA-Z0-9_,-]+\])?)[=><~!].*/\1/g' \
	> requirements.txt