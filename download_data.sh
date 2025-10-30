#!/bin/bash
# Download sample data from PI-CAI challenge

echo "Downloading sample PI-CAI data..."

# create data directory
mkdir -p data
cd data

# download a single fold (smallest dataset)
echo "Downloading fold 0 (this may take a while)..."
curl -L "https://zenodo.org/record/6624726/files/picai_public_images_fold0.zip?download=1" \
     -o picai_public_images_fold0.zip

echo "Extracting..."
unzip -q picai_public_images_fold0.zip

echo "Organizing files..."
# The structure should be:
# data/picai_public_images_fold0/<patient_id>/<patient_id>_<modality>.mha

echo "Download complete!"
echo "Sample files are in: data/picai_public_images_fold0/"
echo ""
echo "You can use any patient folder for testing."
echo "Each patient folder contains: *_t2w.mha, *_adc.mha, *_hbv.mha"