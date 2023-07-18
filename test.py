import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
if opt.task != 'MMIS' and opt.dataset_mode != 'photo2art':
    model.eval()


# test
print('Number of images: ', len(dataloader))
for i, data_i in enumerate(tqdm(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['cpath']
    print("Generated model ouput shape: ", generated.shape)
    for b in range(generated.shape[0]):
        # print(i, 'process image... %s' % img_path[b])
        if opt.show_input:
            if opt.task == 'SIS':
                visuals = OrderedDict([('input_label', data_i['label'][b]),
                                       ('real_image', data_i['image'][b]),
                                       ('synthesized_image', generated[b])])
            else:
                visuals = OrderedDict([('content', data_i['label'][b]),
                                       ('style', data_i['image'][b]),
                                       ('synthesized_image', generated[b])])
        else:
            visuals = OrderedDict([('synthesized_image', generated[b])])
