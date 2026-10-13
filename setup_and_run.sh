#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PY="${PYTHON:-python3}"
MODE="${1:-run-us}"

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    echo "ERROR: Python 3.10+ is required (3.12 recommended)."
    "$PY" --version
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo ">> Creating virtual environment in $VENV"
    "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo ">> Installing dependencies"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

if [ ! -f .env ]; then
    cp .env.example .env
    echo ">> Created .env from .env.example - add your API key(s) before using a hosted backend."
fi

mkdir -p logs outputs temp

echo ">> Verifying the environment"
python main.py --list-model-profiles > /dev/null && echo "   Config loads OK."

case "$MODE" in
    --setup-only)
        echo
        echo "Setup complete. Activate the venv with:  source $VENV/bin/activate"
        ;;
    --serve)
        echo ">> Starting the REST API on http://localhost:8000"
        python api_server.py
        ;;
    --bg)
        echo ">> Running the Bulgarian recipes demo"
        python main.py \
            --input "data/BG_2026.04.22.recipe.en.json" \
            --context "Use the provided JSON file of Bulgarian recipes and generate 15 new different unique Bulgarian breakfast cuisine recipes in JSON file format which follows the input JSON data schema. Avoid the duplicate recipes." \
            --model-profile "Qwen/Qwen3-8B"
        ;;
    *)
        echo ">> Running the US recipes demo"
        python main.py \
            --input "data/all_US_recipes_combined.json" \
            --context "Use the provided JSON file of US recipes and generate 15 new different unique Italian breakfast cuisine recipes in JSON file format which follows the input JSON data schema. Avoid the duplicate recipes." \
            --model-profile "Qwen/Qwen3-8B"
        ;;
esac
