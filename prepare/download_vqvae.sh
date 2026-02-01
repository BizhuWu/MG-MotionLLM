mkdir -p checkpoints
cd checkpoints

echo "The pretrained_vqvae will be stored in the './checkpoints' folder"
echo "Downloading"
gdown "https://drive.google.com/uc?id=1tOw9wiu6jkzBy-bLe2iy47KAjE50DgjP"

echo "Extracting"
unzip pretrained_vqvae.zip

echo "Cleaning"
rm pretrained_vqvae.zip

echo "Downloading done!"
