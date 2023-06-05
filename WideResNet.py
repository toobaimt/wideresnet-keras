import torch
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import re
import requests


def WRN():  # , params):
    '''Bottleneck WRN-50-2 model definition
    '''

    def tr(v):
        if v.ndim == 4:
            return v.transpose(2, 3, 1, 0)
        elif v.ndim == 2:
            return v.transpose()
        return v

    # Download torch weights
    url = 'https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth'
    torch_wts = 'wide-resnet-50-2-export-5ae25d50.pth'
    data = requests.get(url)
    with open(torch_wts, 'wb') as file:
        file.write(data.content)

    params = {k: v.numpy() for k, v in torch.load(torch_wts).items()}

    # Keras versions
    def conv2d(x, params, name, stride=1, padding=0):
        x = layers.ZeroPadding2D(padding=padding)(x)
        if '%s.bias' % name in params:
            z = layers.Conv2D(params['%s.weight' % name].shape[0], params['%s.weight' % name].shape[-1], strides=stride,
                              padding='valid')(x)
            return z
        else:
            z = layers.Conv2D(params['%s.weight' % name].shape[0], params['%s.weight' % name].shape[-1], strides=stride,
                              padding='valid', use_bias=False)(x)
            return z

    def group(input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0')
            o = layers.ReLU()(o)
            o = conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, padding=1)
            o = layers.ReLU()(o)
            o = conv2d(o, params, b_base + '2')
            if i == 0:
                o += conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = layers.ReLU()(o)
        return o

    # determine network size by parameters
    blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
                   for k in params.keys()]) for j in range(4)]

    ''' Keras model '''
    inputs = layers.Input([224, 224, 3])
    o = conv2d(inputs, params, 'conv0', 2, 3)
    o = layers.ReLU()(o)
    o = layers.ZeroPadding2D(padding=1)(o)
    o = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(o)
    o_g0 = group(o, params, 'group0', 1, blocks[0])
    o_g1 = group(o_g0, params, 'group1', 2, blocks[1])
    o_g2 = group(o_g1, params, 'group2', 2, blocks[2])
    o_g3 = group(o_g2, params, 'group3', 2, blocks[3])
    o = layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(o_g3)
    o = layers.Reshape(target_shape=[2048])(o)
    o = layers.Dense(1000)(o)

    model = models.Model(inputs, o)

    # Loading weights
    params_sorted = {k: v for k, v in sorted(params.items())}

    params_sorted.pop('fc.weight')
    params_sorted.pop('fc.bias')

    vals = list(params_sorted.values())

    idx = 0
    vals_idx = 0
    while idx < len(model.layers) - 1:
        while len(model.layers[idx].trainable_weights) == 0 and idx < len(model.layers) - 1:
            idx += 1
        if idx == len(model.layers) - 1:
            break
        model.layers[idx].set_weights([np.transpose(vals[vals_idx + 1], (2, 3, 1, 0)), vals[vals_idx]])
        vals_idx += 2
        idx += 1

    model.layers[-1].set_weights([np.transpose(params['fc.weight'], (1, 0)), params['fc.bias']])
    return model


if __name__ == '__main__':
    def normalize_WRN(im):
        im /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return (im - mean) / std


    test_data_preprocessor = ImageDataGenerator(
        preprocessing_function=normalize_WRN,
        samplewise_center=False,
    )
    val_ds = test_data_preprocessor.flow_from_directory(
        'val_data',
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        # keep_aspect_ratio=True
    )

    model = WRN()

    acc = 0
    for idx, (x, y) in enumerate(val_ds):
        print(idx)
        if idx >= len(val_ds):
            break

        logits = model.predict(x, steps=1)
        acc += int(np.array_equal(np.argmax(y), np.argmax(logits)))

    print('Accuracy = ', acc / len(val_ds))
    # achieves 96.34 acc vs 98 of original on val set
    # probably due to slightly different preprocessing (resize+crop vs only crop here)
