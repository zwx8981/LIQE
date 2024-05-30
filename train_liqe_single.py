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
from utils import (set_dataset_qonly, _preprocess2, _preprocess3, convert_models_to_fp32, set_dataset_spaq,
                   set_dataset_ava, _preprocess2a, _preprocess3a, set_dataset_aigc)
import torch.nn.functional as F
from itertools import product
import os
import pickle
from weight_methods import WeightMethods

##############################textual template####################################
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

##############################general setup####################################
koniq10k_set = '../IQA_Database/koniq-10k/1024x768'
seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 5e-6
num_epoch = 6
bs = 64

train_patch = 3

loss_img2 = Fidelity_Loss_distortion()
loss_scene = Multi_Fidelity_Loss()

joint_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys]).to(device)

##############################general setup####################################

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

preprocess2a = _preprocess2a()
preprocess3a = _preprocess3a()


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

def train(model, best_result, best_epoch, srcc_dict):
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
        gmos_batch = []
        num_sample_per_task = []

        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration:
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            x, gmos = sample_batched['I'], sample_batched['mos']

            x = x.to(device)
            gmos = gmos.to(device)
            gmos_batch.append(gmos)
            num_sample_per_task.append(x.size(0))

            # preserve all samples into a batch, will be used for optimization of scene and distortion type later
            all_batch.append(x)

        all_batch = torch.cat(all_batch, dim=0)
        gmos_batch = torch.cat(gmos_batch, dim=0)

        optimizer.zero_grad()
        logits_per_image, _ = do_batch(all_batch, joint_texts)

        logits_per_image = logits_per_image.view(-1, len(qualitys))
        logits_quality = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                         4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]

        total_loss = loss_m4(logits_quality, num_sample_per_task, gmos_batch.detach()).mean()

        total_loss = total_loss

        total_loss.backward()

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

    all_result = {'val':{}, 'test':{}}
    if (epoch >= 0):

        srcc1 = eval(koniq10k_val_loader, phase='val', dataset='koniq10k')
        srcc11 = eval(koniq10k_test_loader, phase='test', dataset='koniq10k')

        srcc_avg = srcc1

        current_avg = srcc_avg

        if current_avg > best_result['avg']:
            print('**********New overall best!**********')
            best_epoch['avg'] = epoch
            best_result['avg'] = current_avg
            srcc_dict['koniq10k'] = srcc11

            ckpt_name = os.path.join('checkpoints', str(session+1), 'liqe_qonly.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results':all_result
            }, ckpt_name)  # just change to your preferred folder/filename

    return best_result, best_epoch, srcc_dict, all_result


def eval(loader, phase, dataset):
    model.eval()
    q_mos = []
    q_hat = []
    for step, sample_batched in enumerate(loader, 0):

        x, gmos = sample_batched['I'], sample_batched['mos']

        x = x.to(device)
        #q_mos.append(gmos.data.numpy())
        q_mos = q_mos + gmos.cpu().tolist()

        # Calculate features
        with torch.no_grad():
            logits_per_image, _ = do_batch(x, joint_texts)

        logits_per_image = logits_per_image.view(-1, len(qualitys))

        logits_quality = logits_per_image


        quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

        q_hat = q_hat + quality_preds.cpu().tolist()

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)
    return srcc

num_workers = 8
for session in range(0,1):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    freeze_model(opt)

    best_result = 0
    best_epoch = 0

    # avg
    srcc_dict = {'koniq10k':0.0}

    koniq10k_train_csv = os.path.join('../IQA_Database/koniq-10k/meta_info_KonIQ10kDataset.csv')
    koniq10k_val_csv = os.path.join('../IQA_Database/koniq-10k/meta_info_KonIQ10kDataset.csv')
    koniq10k_test_csv = os.path.join('../IQA_Database/koniq-10k/meta_info_KonIQ10kDataset.csv')

    koniq10k_train_loader = set_dataset_qonly(koniq10k_train_csv, 32, koniq10k_set, num_workers, preprocess3,
                                              train_patch, False, set=0)
    koniq10k_val_loader = set_dataset_qonly(koniq10k_val_csv, 32, koniq10k_set, num_workers, preprocess2,
                                            15, True, set=1)
    koniq10k_test_loader = set_dataset_qonly(koniq10k_test_csv, 32, koniq10k_set, num_workers, preprocess2,
                                             15, True, set=2)

    train_loaders = [koniq10k_train_loader]

    #train_loaders = [koniq10k_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch, srcc_dict, all_result = train(model, best_result, best_epoch, srcc_dict)
        scheduler.step()

        result_pkl[str(epoch)] = all_result

        print('...............current average best...............')
        print('best average epoch:{}'.format(best_epoch['avg']))
        print('best average result:{}'.format(best_result['avg']))
        for dataset in srcc_dict.keys():
            print_text = dataset + ':' + 'srcc:{}'.format(srcc_dict[dataset])
            print(print_text)

    pkl_name = os.path.join('checkpoints', str(session+1), 'all_results.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(result_pkl, f)
