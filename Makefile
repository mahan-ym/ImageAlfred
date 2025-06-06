.PHONY: test deploy_modal run dev

deploy_modal:
	modal deploy src/modal_app.py

test:
	uv run test/test_modal.py

run:
	uv run src/app.py

dev:
	gradio src/app.py