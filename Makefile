# ---------------------------
# Added so Python can import src/ and ml_components/
# ---------------------------
export PYTHONPATH := $(PWD)
#                          ↑
# (PWD) = the project root ("immo-ML-project"), so Python now sees:
#   immo-ML-project/src/
#   immo-ML-project/ml_components/
#   immo-ML-project/tools/
# and all imports work everywhere (Makefile, CLI, scripts)

.PHONY: version-models train test clean

# Version all pipelines
version-models:
	python tools/resave_models.py

# Example for training everything before versioning
train:
	python src/models/trainer.py

# Unit tests
test:
	pytest -q

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
