import torch
import numpy as np
import clip
import random
from itertools import product
from PIL import Image, ImageFile
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Normalize

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


class LIQE(nn.Module):
    def __init__(self, ckpt, device):
        super(LIQE, self).__init__()
        self.model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        checkpoint = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(checkpoint)
        joint_texts = torch.cat(
            [clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
             in product(qualitys, scenes, dists_map)]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(joint_texts)
            self.text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        self.step = 32
        self.num_patch = 15
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        x = self.normalize(x)
        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, 224, 224)

        sel_step = x.size(0) // self.num_patch
        sel = torch.zeros(self.num_patch)
        for i in range(self.num_patch):
            sel[i] = sel_step * i
        sel = sel.long()
        x = x[sel, ...]
        #num_patch = x.size(1)
        #x = x.view(-1, x.size(2), x.size(3), x.size(4))

        image_features = self.model.encode_image(x)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ self.text_features.t()

        logits_per_image = logits_per_image.view(batch_size, self.num_patch, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)

        logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
        logits_quality = logits_per_image.sum(3).sum(2)

        similarity_scene = logits_per_image.sum(3).sum(1)
        similarity_distortion = logits_per_image.sum(1).sum(1)
        distortion_index = similarity_distortion.argmax(dim=1)
        scene_index = similarity_scene.argmax(dim=1)

        scene = scenes[scene_index]
        distortion = dists_map[distortion_index]

        quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

        return quality, scene, distortion

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = './LIQE.pt'
    liqe = LIQE(ckpt, device)

    x = torch.randn(1,3,512,512).to(device)
    q, s, d = liqe(x)