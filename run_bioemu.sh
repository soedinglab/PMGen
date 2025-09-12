#!/bin/bash

# Name of the environment to activate
ENV_NAME="bioemu"

# Detect available environment manager
if command -v conda &> /dev/null; then
    TOOL="conda"
    eval "$($TOOL shell.bash hook)"
elif command -v mamba &> /dev/null; then
    TOOL="mamba"
    eval "$($TOOL shell.bash hook)"
elif command -v micromamba &> /dev/null; then
    TOOL="micromamba"
    eval "$($TOOL shell.bash hook)"
else
    echo "❌ Error: conda, mamba, or micromamba not found in PATH."
    exit 1
fi

echo "✅ Using $TOOL to activate environment '$ENV_NAME'..."
$TOOL activate "$ENV_NAME"

# Now forward all arguments to the Python script
python run_bioemu.py "$@"