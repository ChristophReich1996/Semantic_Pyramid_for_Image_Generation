git clone https://github.com/vadimkantorov/caffemodel2pytorch.git
cd caffemodel2pytorch
wget http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel
python -m caffemodel2pytorch vgg16_places365.caffemodel -o '../vgg_places365.h5'
cd ../
python caffe2pytorchvgg16.py
rm -rf caffemodel2pytorch
