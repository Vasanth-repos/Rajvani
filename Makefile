.PHONY: check setup pipeline clean

check:
	@bash scripts/run_checks.sh

setup:
	@bash scripts/setup_env.sh

pipeline:
	@bash scripts/run_full_pipeline.sh --dialect mwr

clean:
	@rm -rf logs/* checkpoints/*
