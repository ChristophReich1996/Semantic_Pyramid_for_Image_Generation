# Source https://github.com/vadimkantorov/caffemodel2pytorch
import collections, torch, torchvision, numpy, h5py, os
import torch.nn as nn

model = torchvision.models.vgg16()
model.classifier[6] = nn.Linear(in_features=4096, out_features=365, bias=True)
model.features = torch.nn.Sequential(collections.OrderedDict(zip(
    ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
     'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
     'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
     'pool5'], model.features)))
model.classifier = torch.nn.Sequential(
    collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8a'], model.classifier)))

state_dict = h5py.File('vgg_places365.h5', 'r')
model.load_state_dict(
    {l: torch.from_numpy(numpy.array(v)).view_as(p) for k, v in state_dict.items() for l, p in model.named_parameters()
     if k in l})
if not os.path.exists('pre_trained_models'):
    os.mkdir('pre_trained_models')
torch.save(model, os.path.join('pre_trained_models', 'vgg_places_365.pt'))
