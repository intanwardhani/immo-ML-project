#!/bin/bash

set -e

TRAINING_MODELS_DIR="./models"
DEPLOYMENT_REPO_DIR="../immo-ML-deployment/models"

echo "🔄 Syncing latest models to deployment repo..."

for model in "$TRAINING_MODELS_DIR"/*_latest.pkl; do
    echo "📤 Copying $(basename "$model")"
    cp "$model" "$DEPLOYMENT_REPO_DIR/"
done

echo "🎉 Sync complete!"