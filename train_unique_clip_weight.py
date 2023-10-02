import torch
import torch.nn as nn
import numpy as np
#from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import clip
import random
import time
from MNL_Loss import Fidelity_Loss, loss_m4, Multi_Fidelity_Loss, Fidelity_Loss_distortion
import scipy.stats
from utils import set_dataset, _preprocess2, _preprocess3, convert_models_to_fp32
import torch.nn.functional as F
from itertools import product
import os
import pickle
from weight_methods import WeightMethods

##############################textual template####################################
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
##############################textual template####################################

##############################general setup####################################
live_set = '../IQA_Database/databaserelease2/'
csiq_set = '../IQA_Database/CSIQ/'
bid_set = '../IQA_Database/BID/'
clive_set = '../IQA_Database/ChallengeDB_release/'
koniq10k_set = '../IQA_Database/koniq-10k/'
kadid10k_set = '../IQA_Database/kadid10k/'

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtl = 0 # 0:all 1:q+s 2:q+d

initial_lr = 5e-6
num_epoch = 80
bs = 32

train_patch = 3

loss_img2 = Fidelity_Loss_distortion()
loss_scene = Multi_Fidelity_Loss()

scene_texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in scenes]).to(device)
dist_texts = torch.cat([clip.tokenize(f"a photo with {c} artifacts") for c in dists]).to(device)
distmap_texts = torch.cat([clip.tokenize(f"a photo with {c} artifacts") for c in dists_map]).to(device)
quality_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys]).to(device)

joint_texts = torch.cat([clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                         in product(qualitys, scenes, dists_map)]).to(device)

if mtl == 1:
    joint_texts = torch.cat([clip.tokenize(f"a photo of a {c}, which is of {q} quality") for q, c
                             in product(qualitys, scenes)]).to(device)
elif mtl == 2:
    joint_texts = torch.cat([clip.tokenize(f"a photo with {d} artifacts, which is of {q} quality") for q, d
                             in product(qualitys, dists_map)]).to(device)

##############################general setup####################################

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()


def calc_loss(quality_pred, quality_gt, num_sample_per_task, distortion_pred, distortion_gt, scene_pred, scene_gt):

    quality_loss = loss_m4(quality_pred, num_sample_per_task, quality_gt.detach())
    distortion_loss = loss_img2(distortion_pred, distortion_gt.detach())
    scene_loss = loss_scene(scene_pred, scene_gt.detach())

    return [quality_loss, distortion_loss, scene_loss]

def calc_loss2(quality_pred, quality_gt, num_sample_per_task, scene_pred, scene_gt):

    quality_loss = loss_m4(quality_pred, num_sample_per_task, quality_gt.detach())
    scene_loss = loss_scene(scene_pred, scene_gt.detach())

    return [quality_loss, scene_loss]

def calc_loss3(quality_pred, quality_gt, num_sample_per_task, distortion_pred, distortion_gt):

    quality_loss = loss_m4(quality_pred, num_sample_per_task, quality_gt.detach())
    distortion_loss = loss_img2(distortion_pred, distortion_gt.detach())

    return [quality_loss, distortion_loss]

opt = 0
def freeze_model(opt):
    model.logit_scale.requires_grad = False
    if opt == 0: #do nothing
        return
    elif opt == 1: # freeze text encoder
        for p in model.token_embedding.parameters():
            p.requires_grad = False
        for p in model.transformer.parameters():
            p.requires_grad = False
        model.positional_embedding.requires_grad = False
        model.text_projection.requires_grad = False
        for p in model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in model.parameters():
            p.requires_grad =False


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

def train(model, best_result, best_epoch, srcc_dict, scene_dict, type_dict):
    start_time = time.time()
    beta = 0.9
    running_loss = 0 if epoch == 0 else train_loss[-1]
    running_duration = 0.0
    num_steps_per_epoch = 200
    local_counter = epoch * num_steps_per_epoch + 1
    model.eval()
    loaders = []
    for loader in train_loaders:
        loaders.append(iter(loader))

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    for step in range(num_steps_per_epoch):
        #total_loss = 0
        all_batch = []
        scene_gt_batch = []
        dist_gt_batch = []
        gmos_batch = []
        num_sample_per_task = []

        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration:
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            x, gmos, dist, scene1, scene2, scene3, valid = sample_batched['I'], sample_batched['mos'], sample_batched[
                'dist_type'], sample_batched['scene_content1'], sample_batched['scene_content2'], \
                                                    sample_batched['scene_content3'], sample_batched['valid']
            x = x.to(device)
            gmos = gmos.to(device)
            gmos_batch.append(gmos)
            num_sample_per_task.append(x.size(0))

            scene_gt = np.zeros((len(scene1), len(scenes)), dtype=float)
            dist_gt = np.zeros((len(dist), len(dists_map)), dtype=float)

            for i in range(len(scene1)):
                if valid[i] == 1:
                    scene_gt[i, scene2label[scene1[i]]] = 1.0
                elif valid[i] == 2:
                    scene_gt[i, scene2label[scene1[i]]] = 1.0
                    scene_gt[i, scene2label[scene2[i]]] = 1.0
                elif valid[i] == 3:
                    scene_gt[i, scene2label[scene1[i]]] = 1.0
                    scene_gt[i, scene2label[scene2[i]]] = 1.0
                    scene_gt[i, scene2label[scene3[i]]] = 1.0
                dist_gt[i, map2label[dist_map[dist[i]]]] = 1.0
            scene_gt = torch.from_numpy(scene_gt).to(device)
            dist_gt = torch.from_numpy(dist_gt).to(device)

            # preserve all samples into a batch, will be used for optimization of scene and distortion type later
            all_batch.append(x)
            scene_gt_batch.append(scene_gt)
            dist_gt_batch.append(dist_gt)

        all_batch = torch.cat(all_batch, dim=0)
        scene_gt_batch = torch.cat(scene_gt_batch, dim=0)
        dist_gt_batch = torch.cat(dist_gt_batch, dim=0)
        gmos_batch = torch.cat(gmos_batch, dim=0)

        optimizer.zero_grad()
        logits_per_image, _ = do_batch(all_batch, joint_texts)

        if mtl == 0:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
        elif mtl == 1:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes))
        elif mtl == 2:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(dists_map))

        #quality logits:
        if mtl == 0:
            logits_quality = logits_per_image.sum(3).sum(2)
            logits_scene = logits_per_image.sum(3).sum(1)
            logits_distortion = logits_per_image.sum(1).sum(1)

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

            total_loss = loss_m4(logits_quality, num_sample_per_task, gmos_batch.detach()).mean() + \
                         loss_img2(logits_distortion, dist_gt_batch.detach()).mean() + \
                         loss_scene(logits_scene, scene_gt_batch.detach()).mean()

            all_loss = calc_loss(logits_quality, gmos_batch.detach(), num_sample_per_task,
                                 logits_distortion, dist_gt_batch.detach(),
                                 logits_scene, scene_gt_batch.detach())
        elif mtl == 1:
            logits_quality = logits_per_image.sum(2)
            logits_scene = logits_per_image.sum(1)

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

            total_loss = loss_m4(logits_quality, num_sample_per_task, gmos_batch.detach()).mean() + \
                         loss_scene(logits_scene, scene_gt_batch.detach()).mean()

            all_loss = calc_loss2(logits_quality, gmos_batch.detach(), num_sample_per_task,
                                   logits_scene, scene_gt_batch.detach())

        elif mtl == 2:
            logits_quality = logits_per_image.sum(2)
            logits_distortion = logits_per_image.sum(1)

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

            total_loss = loss_m4(logits_quality, num_sample_per_task, gmos_batch.detach()).mean() + \
                         loss_img2(logits_distortion, dist_gt_batch.detach()).mean()

            all_loss = calc_loss2(logits_quality, gmos_batch.detach(), num_sample_per_task,
                                   logits_distortion, dist_gt_batch.detach())

        shared_parameters = None
        last_shared_layer = None

        if not torch.isnan(total_loss):
            # weight losses and backward
            total_loss = weighting_method.backwards(
                all_loss,
                epoch=epoch,
                logsigmas=None,
                shared_parameters=shared_parameters,
                last_shared_params=last_shared_layer,
                returns=True
            )
        else:
            total_loss.backward()
            continue

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        # statistics
        running_loss = beta * running_loss + (1 - beta) * total_loss.data.item()
        loss_corrected = running_loss / (1 - beta ** local_counter)

        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** local_counter)
        examples_per_sec = x.size(0) / duration_corrected
        format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (epoch, step + 1, num_steps_per_epoch, loss_corrected,
                            examples_per_sec, duration_corrected))

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)
    quality_result = {'val':{}, 'test':{}}
    scene_result = {'val':{}, 'test':{}}
    distortion_result = {'val':{}, 'test':{}}
    all_result = {'val':{}, 'test':{}}
    if (epoch >= 0):
        scene_acc1, dist_acc1, srcc1 = eval(live_val_loader, phase='val', dataset='live')
        scene_acc11, dist_acc11, srcc11 = eval(live_test_loader, phase='test', dataset='live')

        scene_acc2, dist_acc2, srcc2 = eval(csiq_val_loader, phase='val', dataset='csiq')
        scene_acc22, dist_acc22, srcc22 = eval(csiq_test_loader, phase='test', dataset='csiq')

        scene_acc3, dist_acc3, srcc3 = eval(bid_val_loader, phase='val', dataset='bid')
        scene_acc33, dist_acc33, srcc33 = eval(bid_test_loader, phase='test', dataset='bid')

        scene_acc4, dist_acc4, srcc4 = eval(clive_val_loader, phase='val', dataset='clive')
        scene_acc44, dist_acc44, srcc44 = eval(clive_test_loader, phase='test', dataset='clive')

        scene_acc5, dist_acc5, srcc5 = eval(koniq10k_val_loader, phase='val', dataset='koniq10k')
        scene_acc55, dist_acc55, srcc55 = eval(koniq10k_test_loader, phase='test', dataset='koniq10k')

        scene_acc6, dist_acc6, srcc6 = eval(kadid10k_val_loader, phase='val', dataset='kadid10k')
        scene_acc66, dist_acc66, srcc66 = eval(kadid10k_test_loader, phase='test', dataset='kadid10k')

        quality_result['val'] = {'live':srcc1, 'csiq':srcc2, 'bid':srcc3, 'clive':srcc4, 'koniq10k':srcc5,
                                 'kadid10k':srcc6}
        quality_result['test'] = {'live': srcc11, 'csiq': srcc22, 'bid': srcc33, 'clive': srcc44, 'koniq10k': srcc55,
                                 'kadid10k': srcc66}
        scene_result['val'] = {'live':scene_acc1, 'csiq':scene_acc2, 'bid':scene_acc3, 'clive':scene_acc4, 'koniq10k':scene_acc5,
                                 'kadid10k':scene_acc6}
        scene_result['test'] = {'live':scene_acc11, 'csiq':scene_acc22, 'bid':scene_acc33, 'clive':scene_acc44, 'koniq10k':scene_acc55,
                                 'kadid10k':scene_acc66}
        distortion_result['val'] = {'live':dist_acc1, 'csiq':dist_acc2, 'bid':dist_acc3, 'clive':dist_acc4, 'koniq10k':dist_acc5,
                                 'kadid10k':dist_acc6}
        distortion_result['test'] = {'live':dist_acc11, 'csiq':dist_acc22, 'bid':dist_acc33, 'clive':dist_acc44, 'koniq10k':dist_acc55,
                                 'kadid10k':dist_acc6}

        all_result['val'] = {'quality':quality_result['val'], 'scene':scene_result['val'],
                             'distortion':distortion_result['val']}
        all_result['test'] = {'quality': quality_result['test'], 'scene': scene_result['test'],
                             'distortion': distortion_result['test']}


        srcc_avg = (srcc1 + srcc2 + srcc3 + srcc4 + srcc5 + srcc6) / 6

        scene_avg = (scene_acc1 + scene_acc2 + scene_acc3 + scene_acc4 + scene_acc5 + scene_acc6) / 6

        dist_avg = (dist_acc1 + dist_acc2 + dist_acc3 + dist_acc4 + dist_acc5 + dist_acc6) / 6


        if mtl == 0:
            current_avg = (srcc_avg + scene_avg + dist_avg) / 3
        elif mtl == 1:
            current_avg = (srcc_avg + scene_avg) / 2
        elif mtl == 2:
            current_avg = (srcc_avg + dist_avg) / 2

        if current_avg > best_result['avg']:
            print('**********New overall best!**********')
            best_epoch['avg'] = epoch
            best_result['avg'] = current_avg
            srcc_dict['live'] = srcc11
            srcc_dict['csiq'] = srcc22
            srcc_dict['bid'] = srcc33
            srcc_dict['clive'] = srcc44
            srcc_dict['koniq10k'] = srcc55
            srcc_dict['kadid10k'] = srcc66

            scene_dict['live'] = scene_acc11
            scene_dict['csiq'] = scene_acc22
            scene_dict['bid'] = scene_acc33
            scene_dict['clive'] = scene_acc44
            scene_dict['koniq10k'] = scene_acc55
            scene_dict['kadid10k'] = scene_acc66

            type_dict['live'] = dist_acc11
            type_dict['csiq'] = dist_acc22
            type_dict['bid'] = dist_acc33
            type_dict['clive'] = dist_acc44
            type_dict['koniq10k'] = dist_acc55
            type_dict['kadid10k'] = dist_acc66

            ckpt_name = os.path.join('checkpoints', str(session+1), 'avg_best_ckpt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results':all_result
            }, ckpt_name)  # just change to your preferred folder/filename

        if srcc_avg > best_result['quality']:
            print('**********New quality best!**********')
            best_epoch['quality'] = epoch
            best_result['quality'] = srcc_avg
            srcc_dict1['live'] = srcc11
            srcc_dict1['csiq'] = srcc22
            srcc_dict1['bid'] = srcc33
            srcc_dict1['clive'] = srcc44
            srcc_dict1['koniq10k'] = srcc55
            srcc_dict1['kadid10k'] = srcc66

            scene_dict1['live'] = scene_acc11
            scene_dict1['csiq'] = scene_acc22
            scene_dict1['bid'] = scene_acc33
            scene_dict1['clive'] = scene_acc44
            scene_dict1['koniq10k'] = scene_acc55
            scene_dict1['kadid10k'] = scene_acc66

            type_dict1['live'] = dist_acc11
            type_dict1['csiq'] = dist_acc22
            type_dict1['bid'] = dist_acc33
            type_dict1['clive'] = dist_acc44
            type_dict1['koniq10k'] = dist_acc55
            type_dict1['kadid10k'] = dist_acc66

            ckpt_name = os.path.join('checkpoints', str(session + 1), 'quality_best_ckpt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results': all_result
            }, ckpt_name)  # just change to your preferred folder/filename

        if scene_avg > best_result['scene']:
            print('**********New scene best!**********')
            best_epoch['scene'] = epoch
            best_result['scene'] = scene_avg
            srcc_dict2['live'] = srcc11
            srcc_dict2['csiq'] = srcc22
            srcc_dict2['bid'] = srcc33
            srcc_dict2['clive'] = srcc44
            srcc_dict2['koniq10k'] = srcc55
            srcc_dict2['kadid10k'] = srcc66

            scene_dict2['live'] = scene_acc11
            scene_dict2['csiq'] = scene_acc22
            scene_dict2['bid'] = scene_acc33
            scene_dict2['clive'] = scene_acc44
            scene_dict2['koniq10k'] = scene_acc55
            scene_dict2['kadid10k'] = scene_acc66

            type_dict2['live'] = dist_acc11
            type_dict2['csiq'] = dist_acc22
            type_dict2['bid'] = dist_acc33
            type_dict2['clive'] = dist_acc44
            type_dict2['koniq10k'] = dist_acc55
            type_dict2['kadid10k'] = dist_acc66

            ckpt_name = os.path.join('checkpoints', str(session + 1), 'scene_best_ckpt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results': all_result
            }, ckpt_name)  # just change to your preferred folder/filename

        if dist_avg > best_result['distortion']:
            print('**********New distortion best!**********')
            best_epoch['distortion'] = epoch
            best_result['distortion'] = dist_avg
            srcc_dict3['live'] = srcc11
            srcc_dict3['csiq'] = srcc22
            srcc_dict3['bid'] = srcc33
            srcc_dict3['clive'] = srcc44
            srcc_dict3['koniq10k'] = srcc55
            srcc_dict3['kadid10k'] = srcc66

            scene_dict3['live'] = scene_acc11
            scene_dict3['csiq'] = scene_acc22
            scene_dict3['bid'] = scene_acc33
            scene_dict3['clive'] = scene_acc44
            scene_dict3['koniq10k'] = scene_acc55
            scene_dict3['kadid10k'] = scene_acc66

            type_dict3['live'] = dist_acc11
            type_dict3['csiq'] = dist_acc22
            type_dict3['bid'] = dist_acc33
            type_dict3['clive'] = dist_acc44
            type_dict3['koniq10k'] = dist_acc55
            type_dict3['kadid10k'] = dist_acc66

            ckpt_name = os.path.join('checkpoints', str(session + 1), 'distortion_best_ckpt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results': all_result
            }, ckpt_name)  # just change to your preferred folder/filename

    return best_result, best_epoch, srcc_dict, scene_dict, type_dict, all_result


def eval(loader, phase, dataset):
    model.eval()
    correct_scene = 0.0
    correct_dist = 0.0
    q_mos = []
    q_hat = []
    num_scene = 0
    num_dist = 0
    for step, sample_batched in enumerate(loader, 0):

        x, gmos, dist, scene1, scene2, scene3, valid = sample_batched['I'], sample_batched['mos'], sample_batched[
            'dist_type'], sample_batched['scene_content1'], sample_batched['scene_content2'], \
                                                       sample_batched['scene_content3'], sample_batched['valid']

        x = x.to(device)
        #q_mos.append(gmos.data.numpy())
        q_mos = q_mos + gmos.cpu().tolist()

        #x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        # Calculate features
        with torch.no_grad():
            logits_per_image, _ = do_batch(x, joint_texts)

        if mtl == 0:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
        elif mtl == 1:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes))
        elif mtl == 2:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(dists_map))


        if mtl == 0:
            logits_quality = logits_per_image.sum(3).sum(2)
            similarity_scene = logits_per_image.sum(3).sum(1)
            similarity_distortion = logits_per_image.sum(1).sum(1)
        elif mtl == 1:
            logits_quality = logits_per_image.sum(2)
            similarity_scene = logits_per_image.sum(1)
        elif mtl == 2:
            logits_quality = logits_per_image.sum(2)
            #logits_scene = logits_per_image
            similarity_distortion = logits_per_image.sum(1)

        quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

        q_hat = q_hat + quality_preds.cpu().tolist()

        if mtl != 1:
            indice2 = similarity_distortion.argmax(dim=1)
            for i in range(len(dist)):
                if dist_map[dist[i]] == dists_map[indice2[i]]:  # dist_map: mapping dict #dists_map: type list
                    correct_dist += 1
                num_dist += 1
        if mtl != 2:
            for i in range(len(valid)):
                if valid[i] == 1:
                    indice = similarity_scene.argmax(dim=1)
                    # indice = indice.squeeze()
                    if scene1[i] == scenes[indice[i]]:
                        correct_scene += 1
                    num_scene += 1
                elif valid[i] == 2:
                    _, indices = similarity_scene.topk(k=2, dim=1)
                    # indices = indices.squeeze()
                    if (scene1[i] == scenes[indices[i, 0]]) | (scene1[i] == scenes[indices[i, 1]]):
                        correct_scene += 1
                    if (scene2[i] == scenes[indices[i, 0]]) | (scene2[i] == scenes[indices[i, 1]]):
                        correct_scene += 1
                    num_scene += 2
                elif valid[i] == 3:
                    _, indices = similarity_scene.topk(k=3, dim=1)
                    indices = indices.squeeze()
                    if (scene1[i] == scenes[indices[i, 0]]) | (scene1[i] == scenes[indices[i, 1]]) | (
                            scene1[i] == scenes[indices[i, 2]]):
                        correct_scene += 1
                    if (scene2[i] == scenes[indices[i, 0]]) | (scene2[i] == scenes[indices[i, 1]]) | (
                            scene2[i] == scenes[indices[i, 2]]):
                        correct_scene += 1
                    if (scene3[i] == scenes[indices[i, 0]]) | (scene3[i] == scenes[indices[i, 1]]) | (
                            scene3[i] == scenes[indices[i, 2]]):
                        correct_scene += 1
                    num_scene += 3

    if mtl == 0:
        scene_acc = correct_scene / num_scene
        dist_acc = correct_dist / num_dist
    elif mtl == 1:
        scene_acc = correct_scene / num_scene
        dist_acc = 0
    elif mtl == 2:
        scene_acc = 0
        dist_acc = correct_dist / num_dist

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]

    #print_text = dataset + ':' + phase + ': ' + 'scene accuracy:{}, distortion accuracy:{}, srcc:{}'.format(scene_acc, dist_acc, srcc)
    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)

    return scene_acc, dist_acc, srcc

num_workers = 8
for session in range(0,10):
    if mtl == 0:
        weighting_method = WeightMethods(
            method='dwa',
            n_tasks=3,
            alpha=1.5,
            temp=2.0,
            n_train_batch=200,
            n_epochs=num_epoch,
            main_task=0,
            device=device
        )
    else:
        weighting_method = WeightMethods(
            method='dwa',
            n_tasks=2,
            alpha=1.5,
            temp=2.0,
            n_train_batch=200,
            n_epochs=num_epoch,
            main_task=0,
            device=device
        )

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    freeze_model(opt)

    best_result = {'avg': 0.0, 'quality': 0.0, 'scene': 0.0, 'distortion': 0.0}
    best_epoch = {'avg': 0, 'quality': 0, 'scene': 0, 'distortion': 0}

    # avg
    srcc_dict = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    scene_dict = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    type_dict = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}

    # quality
    srcc_dict1 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    scene_dict1 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    type_dict1 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}

    # scene
    srcc_dict2 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    scene_dict2 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    type_dict2 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}

    # distortion
    srcc_dict3 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    scene_dict3 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}
    type_dict3 = {'live': 0.0, 'csiq': 0.0, 'bid': 0.0, 'clive': 0.0, 'koniq10k': 0.0, 'kadid10k': 0.0}

    live_train_csv = os.path.join('../IQA_Database/databaserelease2/splits2', str(session+1), 'live_train_clip.txt')
    live_val_csv = os.path.join('../IQA_Database/databaserelease2/splits2', str(session+1), 'live_val_clip.txt')
    live_test_csv = os.path.join('../IQA_Database/databaserelease2/splits2', str(session+1), 'live_test_clip.txt')

    csiq_train_csv = os.path.join('../IQA_Database/CSIQ/splits2', str(session+1), 'csiq_train_clip.txt')
    csiq_val_csv = os.path.join('../IQA_Database/CSIQ/splits2', str(session+1), 'csiq_val_clip.txt')
    csiq_test_csv = os.path.join('../IQA_Database/CSIQ/splits2', str(session+1), 'csiq_test_clip.txt')

    bid_train_csv = os.path.join('../IQA_Database/BID/splits2', str(session+1), 'bid_train_clip.txt')
    bid_val_csv = os.path.join('../IQA_Database/BID/splits2', str(session+1), 'bid_val_clip.txt')
    bid_test_csv = os.path.join('../IQA_Database/BID/splits2', str(session+1), 'bid_test_clip.txt')

    clive_train_csv = os.path.join('../IQA_Database/ChallengeDB_release/splits2', str(session+1), 'clive_train_clip.txt')
    clive_val_csv = os.path.join('../IQA_Database/ChallengeDB_release/splits2', str(session+1), 'clive_val_clip.txt')
    clive_test_csv = os.path.join('../IQA_Database/ChallengeDB_release/splits2', str(session+1), 'clive_test_clip.txt')

    koniq10k_train_csv = os.path.join('../IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_train_clip.txt')
    koniq10k_val_csv = os.path.join('../IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_val_clip.txt')
    koniq10k_test_csv = os.path.join('../IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_test_clip.txt')

    kadid10k_train_csv = os.path.join('../IQA_Database/kadid10k/splits2', str(session+1), 'kadid10k_train_clip.txt')
    kadid10k_val_csv = os.path.join('../IQA_Database/kadid10k/splits2', str(session+1), 'kadid10k_val_clip.txt')
    kadid10k_test_csv = os.path.join('../IQA_Database/kadid10k/splits2', str(session+1), 'kadid10k_test_clip.txt')

    live_train_loader = set_dataset(live_train_csv, 4, live_set, num_workers, preprocess3, train_patch, False)
    live_val_loader = set_dataset(live_val_csv, 16, live_set, num_workers, preprocess2, 15, True)
    live_test_loader = set_dataset(live_test_csv, 16, live_set, num_workers, preprocess2, 15, True)

    csiq_train_loader = set_dataset(csiq_train_csv, 4, csiq_set, num_workers, preprocess3, train_patch, False)
    csiq_val_loader = set_dataset(csiq_val_csv, 16, csiq_set, num_workers, preprocess2, 15, True)
    csiq_test_loader = set_dataset(csiq_test_csv, 16, csiq_set, num_workers, preprocess2, 15, True)

    bid_train_loader = set_dataset(bid_train_csv, 4, bid_set, num_workers, preprocess3, train_patch, False)
    bid_val_loader = set_dataset(bid_val_csv, 16, bid_set, num_workers, preprocess2, 15, True)
    bid_test_loader = set_dataset(bid_test_csv, 16, bid_set, num_workers, preprocess2, 15, True)

    clive_train_loader = set_dataset(clive_train_csv, 4, clive_set, num_workers, preprocess3, train_patch, False)
    clive_val_loader = set_dataset(clive_val_csv, 16, clive_set, num_workers, preprocess2, 15, True)
    clive_test_loader = set_dataset(clive_test_csv, 16, clive_set, num_workers, preprocess2, 15, True)

    koniq10k_train_loader = set_dataset(koniq10k_train_csv, 16, koniq10k_set, num_workers, preprocess3, train_patch, False)
    koniq10k_val_loader = set_dataset(koniq10k_val_csv, 16, koniq10k_set, num_workers, preprocess2, 15, True)
    koniq10k_test_loader = set_dataset(koniq10k_test_csv, 16, koniq10k_set, num_workers, preprocess2, 15, True)

    kadid10k_train_loader = set_dataset(kadid10k_train_csv, 16, kadid10k_set, num_workers, preprocess3, train_patch, False)
    kadid10k_val_loader = set_dataset(kadid10k_val_csv, 16, kadid10k_set, num_workers, preprocess2, 15, True)
    kadid10k_test_loader = set_dataset(kadid10k_test_csv, 16, kadid10k_set, num_workers, preprocess2, 15, True)

    train_loaders = [live_train_loader, csiq_train_loader, bid_train_loader, clive_train_loader,
                     koniq10k_train_loader, kadid10k_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch, srcc_dict, scene_dict, type_dict, all_result = train(model, best_result, best_epoch, srcc_dict,
                                                                               scene_dict, type_dict)
        scheduler.step()

        result_pkl[str(epoch)] = all_result

        print(weighting_method.method.lambda_weight[:, epoch])

        print('...............current average best...............')
        print('best average epoch:{}'.format(best_epoch['avg']))
        print('best average result:{}'.format(best_result['avg']))
        for dataset in srcc_dict.keys():
            print_text = dataset + ':' + 'scene:{}, distortion:{}, srcc:{}'.format(
                scene_dict[dataset], type_dict[dataset], srcc_dict[dataset])
            print(print_text)

        print('...............current quality best...............')
        print('best quality epoch:{}'.format(best_epoch['quality']))
        print('best quality result:{}'.format(best_result['quality']))
        for dataset in srcc_dict1.keys():
            print_text = dataset + ':' + 'scene:{}, distortion:{}, srcc:{}'.format(
                scene_dict1[dataset], type_dict1[dataset], srcc_dict1[dataset])
            print(print_text)

        print('...............current scene best...............')
        print('best scene epoch:{}'.format(best_epoch['scene']))
        print('best scene result:{}'.format(best_result['scene']))
        for dataset in srcc_dict1.keys():
            print_text = dataset + ':' + 'scene:{}, distortion:{}, srcc:{}'.format(
                scene_dict2[dataset], type_dict2[dataset], srcc_dict2[dataset])
            print(print_text)

        print('...............current distortion best...............')
        print('best distortion epoch:{}'.format(best_epoch['distortion']))
        print('best distortion result:{}'.format(best_result['distortion']))
        for dataset in srcc_dict1.keys():
            print_text = dataset + ':' + 'scene:{}, distortion:{}, srcc:{}'.format(
                scene_dict3[dataset], type_dict3[dataset], srcc_dict3[dataset])
            print(print_text)

    pkl_name = os.path.join('checkpoints', str(session+1), 'all_results.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(result_pkl, f)

    lambdas = weighting_method.method.lambda_weight
    pkl_name = os.path.join('checkpoints', str(session+1), 'lambdas.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(lambdas, f)








