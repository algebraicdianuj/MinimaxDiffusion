

rm -rf image_woof_dataset
wget -O - https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz | tar xz
mkdir image_woof_dataset
cp -r imagewoof2/train/* image_woof_dataset
rm -r imagewoof2

