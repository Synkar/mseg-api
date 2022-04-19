#!/bin/sh

# Extracts the BDD100K semantic segmentation dataset.

# By using this script, you agree to the terms
# and conditions of the BDD dataset.

# ------------------- Set up Environment Variables + Directory ----------

# Destination directory for BDD
BDD_DST_DIR=$1 

echo "BDD will be downloaded to "$BDD_DST_DIR
mkdir -p $BDD_DST_DIR

# --------------------- Extraction --------------------------------------
echo "Extracting BDD100K dataset..."
# should be about 1.3 GB, includes images and labels
mkdir -p $BDD_DST_DIR
cd $BDD_DST_DIR
wget http://dl.yf.io/bdd100k/legacy/bdd100k_seg_2018.zip
unzip bdd100k_seg_2018.zip
mv bdd100k_seg_2018 bdd100k_seg

cd ..
echo "BDD100K dataset extracted."

# No need to remap, since BDD distributed in the `semseg` format.