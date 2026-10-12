.PHONY: install test serve run backends ecosystem docker-up docker-down

install:
	uv sync --locked

test:
	uv run python -m unittest discover -s tests -v

serve:
	./start.sh serve

run:
	./start.sh run "A concise cinematic introduction." --output outputs/demo.mp4

backends:
	uv run videogenerator backends

ecosystem:
	uv run videogenerator ecosystem

docker-up:
	docker compose up --build

docker-down:
	docker compose down
