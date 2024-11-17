.PHONY: install
install:
	@poetry config virtualenvs.in-project true
	@poetry install

.PHONY: run 
run:
	@poetry run python einops_game/main.py