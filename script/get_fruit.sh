#!/bin/bash
# Soccer player dataset http://cocodataset.org
# Download command: bash ./scripts/get_fruit.sh

# Download/unzip dataset
d='./' # unzip directory
url=https://drive.google.com/file/d/1btZfd9hFpY7J_UGDMHkUtia-2VggcLRP/view?usp=sharing/
f='fruit_dataset.zip' # or 'coco2017labels.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
wait # finish background tasks
 
