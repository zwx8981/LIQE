import torch
import numpy as np
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from LIQE import LIQE
from torchvision.transforms import ToTensor

seed = 20200626
num_patch = 15

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
ckpt = './checkpoints/LIQE.pt'
model = LIQE(ckpt, device)

img1 = 'data/6898804586.jpg'
img2 = 'data/I02_01_03.png'

print('###Image loading###')

I1 = Image.open(img1)
I2 = Image.open(img2)

I1 = ToTensor()(I1).unsqueeze(0)
I2 = ToTensor()(I2).unsqueeze(0)

print('###Preprocessing###')
with torch.no_grad():
    q1, s1, d1 = model(I1)
    q2, s2, d2 = model(I2)

print('Image #1 is a photo of {} with {} artifacts, which has a perceptual quality of {} as quantified by LIQE'.
      format(s1, d1, q1.item()))

print('Image #2 is a photo of {} with {} artifacts, which has a perceptual quality of {} as quantified by LIQE'.
      format(s2, d2, q2.item()))