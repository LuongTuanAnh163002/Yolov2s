#!/bin/bash
# Soccer player dataset http://cocodataset.org
# Download command: bash ./scripts/get_fruit.sh

# Download/unzip dataset
d='./' # unzip directory
file_id='1btZfd9hFpY7J_UGDMHkUtia-2VggcLRP' # ID của file trên Google Drive
url="https://drive.google.com/uc?export=download&id=$file_id"
filename='fruit_dataset.zip'

gdown $file_id

echo 'Unzipping' $filename '...'
unzip -q $filename -d $d

echo 'Removing' $filename '...'
rm $filename

echo 'Download complete!'
 
