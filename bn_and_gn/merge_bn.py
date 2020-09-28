import torch
import os
from collections import OrderedDict
import cv2
import numpy as np
import torchvision.transforms as transforms


def merge(params, name, layer):
    # global variables
    global weights, bias
    global bn_param

    if layer == 'Convolution':
        # save weights and bias when meet conv layer
        if 'weight' in name:
            weights = params.data
            bias = torch.zeros(weights.size()[0])
        elif 'bias' in name:
            bias = params.data
        bn_param = {}

    elif layer == 'BatchNorm':
        # save bn params
        bn_param[name.split('.')[-1]] = params.data

        # running_var is the last bn param in pytorch
        if 'running_var' in name:
            # let us merge bn ~
            tmp = bn_param['weight'] / torch.sqrt(bn_param['running_var'] + 1e-5)
            weights = tmp.view(tmp.size()[0], 1, 1, 1) * weights
            bias = tmp*(bias - bn_param['running_mean']) + bn_param['bias']

            return weights, bias

    return None, None
    
    
model = Generalized_CNN(cfg)

test_weights = get_weights(osp.join(pet_root, cfg.CKPT), cfg.TEST.WEIGHTS)
print(test_weights)   # path

print('Finding trained model weights...')
try:
    print('Loading weights from %s ...')
    trained_weights = torch.load(test_weights)
    print('Weights load success')
except:
    raise ValueError('No trained model found or loading error occurs')


print('Going through pytorch net weights...')
new_weights = OrderedDict()
inner_product_flag = False
for name, params in trained_weights['model'].items():
    # if len(params.size()) == 4:
    if 'conv1.weight' in name or 'conv2.weight' in name or 'conv.weight' in name:
        _, _ = merge_bn(params, name, 'Convolution')
        prev_layer = name
        inner_product_flag = False
    elif len(params.size()) == 1 and not inner_product_flag:
        w, b = merge_bn(params, name, 'BatchNorm')
        if w is not None:
            new_weights[prev_layer] = w
            new_weights[prev_layer.replace('weight', 'bias')] = b
    else:
        if 'bn1.num_batches_tracked' not in name and 'bn2.num_batches_tracked' not in name and 'bn.num_batches_tracked' not in name:
            # print(name)
            new_weights[name] = params
            inner_product_flag = True

# remove  'Module.'  in  new_weights.key()
for i in list(new_weights.keys()):
    a = i.split('.', 1)
    new_weights[a[1]] = new_weights.pop(i)
        

torch.save(new_weights, '/home/user/Program/Share/wh/TRT/model_merge.pth')
new_model_path = '/home/user/Program/Share/wh/TRT/model_merge.pth'
print(new_model_path)

load_weights(model, new_model_path)

model.eval()
