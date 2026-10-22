.PHONY: install lint format test sample-data train serve docker

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts

format:
	ruff check --fix src tests scripts && ruff format src tests scripts

test:
	pytest --cov=codeformer --cov-report=term-missing

sample-data:
	python scripts/make_sample_data.py --out data/sample --n-train 4000 --n-val 400 --n-realtime 300

train:
	codeformer train --config configs/default.yaml --train data/sample/train.csv \
		--val data/sample/val.csv --out artifacts/model.pt

serve:
	CODEFORMER_CHECKPOINT=artifacts/model.pt uvicorn --factory codeformer.api.app:create_app --port 8000

docker:
	docker build -t codeformer:latest .
