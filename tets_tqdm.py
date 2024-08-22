import functools
import itertools
import os.path
import threading

import einops
import jax
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import webdataset as wds
import glob
from timm.models.vision_transformer import VisionTransformer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from webdataset.shardlists import expand_urls
import torchvision.transforms.v2 as T

from model_vit.convert_pytorch_to_flax import convert_pytorch_to_flax_vit
from model_vit.modeling import ViT

#
# while threading.active_count() > 1:
#     print(threading.active_count())
# '/home/jtitor/PycharmProjects/jax-dit/shard_path2/shards-{00000..00008}.tar',
""""""

urls = ['/home/jtitor/PycharmProjects/jax-dit/shard_path2/shards-{00000..00008}.tar',
        '/home/jtitor/PycharmProjects/jax-dit/shard_path2/shards-{00000..00008}.tar']
files = []
for url in urls:
    files.extend(expand_urls(url))

files = [f'/home/jtitor/test_cp/{x}' for x in os.listdir('/home/jtitor/test_cp')]
# print(files)

# files='/home/jtitor/PycharmProjects/jax-dit/shard_path2/shards-{00000..00008}.tar'

# files = '/root/ADV-ViT/shard_path2/shards-{00000..00001}.tar'
valid_transforms = T.Compose([
    T.Resize(int(224 / 0.875), interpolation=3),
    T.CenterCrop(224),
    T.PILToTensor(),
])

dataset = wds.DataPipeline(
    wds.SimpleShardList(files, seed=1),
    # itertools.cycle,
    # wds.detshuffle(),
    # wds.slice(jax.process_index(), None, jax.process_count()),
    # wds.split_by_worker,
    wds.tarfile_to_samples(),
    # wds.detshuffle(),
    wds.decode("pil", handler=wds.ignore_and_continue),
    wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
    # partial(repeat_samples, repeats=args.augment_repeats),
    wds.map_tuple(valid_transforms, torch.tensor),
)

dataloader = DataLoader(dataset, batch_size=64)


def get_model_vit():
    model = timm.create_model('vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True)

    params = convert_pytorch_to_flax_vit(model.state_dict(), model.blocks[0].attn.num_heads)

    vit_model = ViT(
        patch_size=14,
        layers=24,
        dim=1024,
        heads=16,
        # pooling="gap",
        # posemb="sincos2d"
    )

    return vit_model, params

@jax.jit
def infer(x):
    return vit_model.apply({'params': params['model']}, x, det=True, )


vit_model, params = get_model_vit()
for i, data in enumerate(dataloader):
    x, y = data
    x = x.to(torch.float32) / 255

    x_jax = einops.rearrange(x, 'b c h w ->b h w c ')

    y = y

    x_jax = x_jax.detach().numpy()

    logit_jax = infer(x_jax)

    model_predict_label = np.array(logit_jax).argmax(axis=1)
    # print(model_predict_label,y)
    # print(model_predict_label.dtype,y.dtype)
    y = np.array(y)

    print(np.sum(model_predict_label == y) / x.shape[0])

# model=model.cuda()
# with torch.no_grad():
#     for i, data in enumerate(dataloader):
#         x, y = data
#         # print(data['cls'])
#         # print(data)
#         # plt.imshow(data[0])
#         # plt.show()
#         x = x.cuda().to(torch.float32) / 255
#         y = y.cuda()
#         logits = model(x)
#
#         model_predict_label = logits.argmax(dim=1)
#
#         print((model_predict_label == y).sum() / x.shape[0])
