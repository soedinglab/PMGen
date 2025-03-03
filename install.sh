#!/bin/bash

set -e  # Exit immediately if a command fails

# Define an error handler function
error_handler() {
    echo "========================================="
    echo "⚠ An error occurred during the installation process."
    echo "If it is a GPU related issue, please adjust your CUDA and CUDNN version in .yml files."
    echo "Please open an issue on our GitHub. We try to fix it in 24 hours!"
    echo "GitHub: https://github.com/AmirAsgary/ParseFold-MHC"
    echo "========================================="
    exit 1
}

# Trap any error and call the error handler
trap 'error_handler' ERR

CURRENT_DIR=$(pwd)
TMP_DIR="$CURRENT_DIR/tmp"
ENV_NAME="parsefold_mhc"
ENV_FILE="parsefold_mhc.yml"
AFFINE_ZIP_URL="https://owncloud.gwdg.de/index.php/s/kXp5POS99SseFtG/download"
AFFINE_ZIP_NAME="AFfine.zip"
AFFINE_FOLDER="AFfine"
PANDORA_MODIF_PATH="$CURRENT_DIR/PANDORA/PANDORA/Pandora/Modelling_functions.py"
PANDORA_PMHC_PATH="$CURRENT_DIR/PANDORA/PANDORA/PMHC/PMHC.py"

echo "========================================="
echo " Starting Installation: ParseFold-MHC"
echo " Thank you for using our tool!"
echo "========================================="

# Step 0: Display References
echo "### References for the tools used in this pipeline ###"
echo "1. PANDORA - GitHub: https://github.com/X-lab-3D/PANDORA"
echo "   Paper: https://www.frontiersin.org/articles/10.3389/fimmu.2022.878762/full"
echo "2. AFfine - GitHub: https://github.com/phbradley/alphafold_finetune"
echo "   Paper: https://www.pnas.org/doi/abs/10.1073/pnas.2216697120"
echo "3. Modeller - Website: https://salilab.org/modeller/"
echo "   Paper: A. Fiser, R.K. Do, & A. Sali. Modeling of loops in protein structures, Protein Science 9. 1753-1773, 2000."
echo "4. AlphaFold - GitHub: https://github.com/google-deepmind/alphafold"
echo "   Paper: https://www.nature.com/articles/s41586-021-03819-2"
echo "5. ProteinMPNN - Github https://github.com/dauparas/ProteinMPNN"
echo "   Paper: https://www.science.org/doi/10.1126/science.add2187"
echo "########################################################"

# Step 1: Ask for Modeller License Key
echo -n "Please enter your Modeller license key: "
read -r KEY_MODELLER
export KEY_MODELLER="$KEY_MODELLER"
echo "✔ Modeller license key has been set."

# Step 2: Check if mamba exists; otherwise, use conda
if command -v mamba &>/dev/null; then
    echo "✔ Mamba found and using it"
    CONDA_CMD="mamba"
    ACTIVATE_CMD="mamba activate"
else
    CONDA_CMD="conda"
    ACTIVATE_CMD="conda activate"
    echo "⚠ Mamba not found! Falling back to Conda."
fi

# Step 3: Ensure Conda is initialized
if [ -z "$CONDA_PREFIX" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Step 4: Create and activate Conda/Mamba environment
echo "✔ Creating and activating environment: $ENV_NAME..."

if ! conda env list | grep -q "$ENV_NAME"; then
    if [ -f "$ENV_FILE" ]; then
        echo "✔ Found $ENV_FILE. Creating environment with dependencies..."
        $CONDA_CMD env create -f "$ENV_FILE"
    else
        echo "⚠ No $ENV_FILE found."
    fi
else
    echo "✔ Environment $ENV_NAME already exists."
fi

# Activate the environment
$ACTIVATE_CMD "$ENV_NAME"

# Step 5: Clone and Install PANDORA
echo "✔ Cloning and installing PANDORA..."
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

if [ ! -d "PANDORA" ]; then
    git clone https://github.com/X-lab-3D/PANDORA.git
fi

cd PANDORA
pip install -e .

# Step 6: Ensure dependencies are correctly set after installing PANDORA
echo "✔ Reinstalling dependencies to match $ENV_FILE..."
$CONDA_CMD env update -f "$CURRENT_DIR/$ENV_FILE"

# Step 7: Fetch PANDORA Data
echo "✔ Fetching PANDORA data..."
pandora-fetch

# Step 8: Download and Extract AFfine Data
echo "✔ Downloading AFfine data..."
cd "$CURRENT_DIR"
mkdir -p data
cd data

if [ ! -d "$AFFINE_FOLDER" ]; then
    if [ ! -f "$AFFINE_ZIP_NAME" ]; then
        wget -O "$AFFINE_ZIP_NAME" "$AFFINE_ZIP_URL"
    fi
    unzip -o "$AFFINE_ZIP_NAME"
    echo "✔ AFfine data extracted."
else
    echo "✔ AFfine folder already exists. Skipping download."
fi

# Step 10: Dowlnload modified files for PANDORA
echo "Change modified scripts in PANDORA"
rm "$PANDORA_MODIF_PATH"
rm "$PANDORA_PMHC_PATH"
mv "data/modified_files/Modelling_functions.py" "$PANDORA_MODIF_PATH"
mv "data/modified_files/PMHC.py" "$PANDORA_PMHC_PATH"

# Step 11: Install ProteinMPNN
echo "ProteinMPNN installation, we need to set it up on a different env."
git clone https://github.com/dauparas/ProteinMPNN.git
echo "✔ ProteinMPNN setup is done. "
# Step 12: Cleanup and Completion
cd "$CURRENT_DIR"
echo "✔ Installation completed successfully!"
echo "Please check and modify 'user_setting.py' file to customize for your usage"
echo "If you want to use NetMHCpan predictions for Anchor, please make sure to install it and provide its path to 'user_setting.py'"
echo "To use Parsefold, run: 'conda activate parsefold_mhc' "
echo "========================================="
echo "✅ ParseFold-MHC is ready to use!"
