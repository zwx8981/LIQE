import torch
import numpy as np
import clip
from utils import _preprocess2
import random
from itertools import product
from PIL import Image, ImageFile
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

dists = ['jpeg2000 compression', 'jpeg compression', 'white noise', 'gaussian blur', 'fastfading', 'fnoise', 'contrast', 'lens', 'motion', 'diffusion', 'shifting',
         'color quantization', 'oversaturation', 'desaturation', 'white with color', 'impulse', 'multiplicative',
         'white noise with denoise', 'brighten', 'darken', 'shifting the mean', 'jitter', 'noneccentricity patch',
         'pixelate', 'quantization', 'color blocking', 'sharpness', 'realistic blur', 'realistic noise',
         'underexposure', 'overexposure', 'realistic contrast change', 'other realistic']

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

type2label = {'jpeg2000 compression':0, 'jpeg compression':1, 'white noise':2, 'gaussian blur':3, 'fastfading':4, 'fnoise':5, 'contrast':6, 'lens':7, 'motion':8,
              'diffusion':9, 'shifting':10, 'color quantization':11, 'oversaturation':12, 'desaturation':13,
              'white with color':14, 'impulse':15, 'multiplicative':16, 'white noise with denoise':17, 'brighten':18,
              'darken':19, 'shifting the mean':20, 'jitter':21, 'noneccentricity patch':22, 'pixelate':23,
              'quantization':24, 'color blocking':25, 'sharpness':26, 'realistic blur':27, 'realistic noise':28,
              'underexposure':29, 'overexposure':30, 'realistic contrast change':31, 'other realistic':32}

dist_map = {'jpeg2000 compression':'jpeg2000 compression', 'jpeg compression':'jpeg compression',
                   'white noise':'noise', 'gaussian blur':'blur', 'fastfading': 'jpeg2000 compression', 'fnoise':'noise',
                   'contrast':'contrast', 'lens':'blur', 'motion':'blur', 'diffusion':'color', 'shifting':'blur',
                   'color quantization':'quantization', 'oversaturation':'color', 'desaturation':'color',
                   'white with color':'noise', 'impulse':'noise', 'multiplicative':'noise',
                   'white noise with denoise':'noise', 'brighten':'overexposure', 'darken':'underexposure', 'shifting the mean':'other',
                   'jitter':'spatial', 'noneccentricity patch':'spatial', 'pixelate':'spatial', 'quantization':'quantization',
                   'color blocking':'spatial', 'sharpness':'contrast', 'realistic blur':'blur', 'realistic noise':'noise',
                   'underexposure':'underexposure', 'overexposure':'overexposure', 'realistic contrast change':'contrast', 'other realistic':'other'}

map2label = {'jpeg2000 compression':0, 'jpeg compression':1, 'noise':2, 'blur':3, 'color':4,
             'contrast':5, 'overexposure':6, 'underexposure':7, 'spatial':8, 'quantization':9, 'other':10}

dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

scene2label = {'animal':0, 'cityscape':1, 'human':2, 'indoor':3, 'landscape':4, 'night':5, 'plant':6, 'still_life':7,
               'others':8}

preprocess2 = _preprocess2()

def do_batch(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, logits_per_text = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_text = logits_per_text.view(-1, batch_size, num_patch)

    logits_per_image = logits_per_image.mean(1)
    logits_per_text = logits_per_text.mean(2)

    logits_per_image = F.softmax(logits_per_image, dim=1)

    return logits_per_image, logits_per_text


seed = 20200626
num_patch = 15

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
ckpt = './checkpoints/LIQE.pt'
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint)

joint_texts = torch.cat([clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                         in product(qualitys, scenes, dists_map)]).to(device)


img1 = 'data/6898804586.jpg'
img2 = 'data/I02_01_03.png'

print('###Image loading###')

I1 = Image.open(img1)
I2 = Image.open(img2)

print('###Preprocessing###')

I1 = preprocess2(I1)
I1 = I1.unsqueeze(0)
n_channels = 3
kernel_h = 224
kernel_w = 224
if (I1.size(2) >= 1024) | (I1.size(3) >= 1024):
    step = 48
else:
    step = 32
I1_patches = I1.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                  n_channels,
                                                                                                  kernel_h,
                                                                                                  kernel_w)
sel_step = I1_patches.size(0) // num_patch
sel = torch.zeros(num_patch)
for i in range(num_patch):
    sel[i] = sel_step * i
sel = sel.long()
I1_patches = I1_patches[sel, ...]

I2 = preprocess2(I2)
I2 = I2.unsqueeze(0)
n_channels = 3
kernel_h = 224
kernel_w = 224
if (I2.size(2) >= 1024) | (I2.size(3) >= 1024):
    step = 48
else:
    step = 32
I2_patches = I2.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                  n_channels,
                                                                                                  kernel_h,
                                                                                                  kernel_w)
sel_step = I2_patches.size(0) // num_patch
sel = torch.zeros(num_patch)
for i in range(num_patch):
    sel[i] = sel_step * i
sel = sel.long()
I2_patches = I2_patches[sel, ...]

I1_patches = I1_patches.to(device)
I2_patches = I2_patches.to(device)

print('###Model Forward###')

with torch.no_grad():
    logits_per_image1, _ = do_batch(I1_patches.unsqueeze(0), joint_texts)
    logits_per_image2, _ = do_batch(I2_patches.unsqueeze(0), joint_texts)

print('###Marginalization###')

logits_per_image1 = logits_per_image1.view(-1, len(qualitys), len(scenes), len(dists_map))
logits_quality1 = logits_per_image1.sum(3).sum(2)
similarity_scene1 = logits_per_image1.sum(3).sum(1)
similarity_distortion1 = logits_per_image1.sum(1).sum(1)

quality_prediction1 = 1 * logits_quality1[:, 0] + 2 * logits_quality1[:, 1] + 3 * logits_quality1[:, 2] + \
                4 * logits_quality1[:, 3] + 5 * logits_quality1[:, 4]

distortion_index1 = similarity_distortion1.argmax(dim=1)
scene_index1 = similarity_scene1.argmax(dim=1)

print('Image #1 is a photo of {} with {} artifacts, which has a perceptual quality of {} as quantified by LIQE'.
      format(scenes[scene_index1], dists_map[distortion_index1], quality_prediction1.item()))

logits_per_image2 = logits_per_image2.view(-1, len(qualitys), len(scenes), len(dists_map))
logits_quality2 = logits_per_image2.sum(3).sum(2)
similarity_scene2 = logits_per_image2.sum(3).sum(1)
similarity_distortion2 = logits_per_image2.sum(1).sum(1)

quality_prediction2 = 1 * logits_quality2[:, 0] + 2 * logits_quality2[:, 1] + 3 * logits_quality2[:, 2] + \
                4 * logits_quality2[:, 3] + 5 * logits_quality2[:, 4]


distortion_index2 = similarity_distortion2.argmax(dim=1)
scene_index2 = similarity_scene2.argmax(dim=1)

print('Image #2 is a photo of {} with {} artifacts, which has a perceptual quality of {} quantified by LIQE'.
      format(scenes[scene_index2], dists_map[distortion_index2], quality_prediction2.item()))
