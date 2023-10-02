import torch
import numpy as np
#from torch.utils.data import DataLoader
import clip
import random
import scipy.stats
from utils import set_dataset, _preprocess2
import torch.nn.functional as F
from itertools import product
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

preprocess2 = _preprocess2()

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

live_set = '../IQA_Database/databaserelease2/'
csiq_set = '../IQA_Database/CSIQ/'
bid_set = '../IQA_Database/BID/'
clive_set = '../IQA_Database/ChallengeDB_release/'
koniq10k_set = '../IQA_Database/koniq-10k/'
kadid10k_set = '../IQA_Database/kadid10k/'

mtl = 0 # 0:all 1:q+s 2:q+d

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

scene_texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in scenes]).to(device)
dist_texts = torch.cat([clip.tokenize(f"a photo with {c} artifacts") for c in dists]).to(device)
distmap_texts = torch.cat([clip.tokenize(f"a photo with {c} artifacts") for c in dists_map]).to(device)
quality_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys]).to(device)


joint_texts = torch.cat([clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                         in product(qualitys, scenes, dists_map)]).to(device)

#joint_texts = distmap_texts
if mtl == 1:
    joint_texts = torch.cat([clip.tokenize(f"a photo of a {c}, which is of {q} quality") for q, c
                             in product(qualitys, scenes)]).to(device)
elif mtl == 2:
    joint_texts = torch.cat([clip.tokenize(f"a photo with {d} artifacts, which is of {q} quality") for q, d
                             in product(qualitys, dists_map)]).to(device)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

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

        # for p in model.text_projection.parameters():
        #     p.requires_grad = False
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

    #return scene_acc, dist_acc, srcc
    return scene_acc, dist_acc, q_mos, q_hat


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), np.std(y_pred)]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return SRCC, KRCC, PLCC, RMSE

num_workers = 8
all_scene = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
all_dist = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
all_srcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
all_plcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
all_rmse = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}
all_krcc = {'live': [], 'csiq': [], 'bid': [], 'clive': [], 'koniq10k': [], 'kadid10k': []}

for session in range(0,10):
    print('session {}'.format(session+1))
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    ckpt = os.path.join('../CLIP/checkpoints_final/checkpoints', str(session+1), 'quality_best_ckpt.pt')
    #ckpt = os.path.join('checkpoints', str(session + 1), 'quality_best_ckpt.pt')
    #ckpt = os.path.join('checkpoints', str(session + 1), 'scene_best_ckpt.pt')
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

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

    live_test_csv = os.path.join('../IQA_Database/databaserelease2/splits2', str(session+1), 'live_test_clip.txt')
    csiq_test_csv = os.path.join('../IQA_Database/CSIQ/splits2', str(session+1), 'csiq_test_clip.txt')
    bid_test_csv = os.path.join('../IQA_Database/BID/splits2', str(session+1), 'bid_test_clip.txt')
    clive_test_csv = os.path.join('../IQA_Database/ChallengeDB_release/splits2', str(session+1), 'clive_test_clip.txt')
    koniq10k_test_csv = os.path.join('../IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_test_clip.txt')
    kadid10k_test_csv = os.path.join('../IQA_Database/kadid10k/splits2', str(session+1), 'kadid10k_test_clip.txt')

    live_test_loader = set_dataset(live_test_csv, 16, live_set, num_workers, preprocess2, 15, True)
    csiq_test_loader = set_dataset(csiq_test_csv, 16, csiq_set, num_workers, preprocess2, 15, True)
    bid_test_loader = set_dataset(bid_test_csv, 16, bid_set, num_workers, preprocess2, 15, True)
    clive_test_loader = set_dataset(clive_test_csv, 16, clive_set, num_workers, preprocess2, 15, True)
    koniq10k_test_loader = set_dataset(koniq10k_test_csv, 16, koniq10k_set, num_workers, preprocess2, 15, True)
    kadid10k_test_loader = set_dataset(kadid10k_test_csv, 16, kadid10k_set, num_workers, preprocess2, 15, True)

    scene_acc1, dist_acc1, q_mos1, q_hat1 = eval(live_test_loader, 'test', 'live')
    scene_acc2, dist_acc2, q_mos2, q_hat2 = eval(csiq_test_loader, 'test', 'csiq')
    scene_acc3, dist_acc3, q_mos3, q_hat3 = eval(bid_test_loader, 'test', 'bid')
    scene_acc4, dist_acc4, q_mos4, q_hat4 = eval(clive_test_loader, 'test', 'clive')
    scene_acc5, dist_acc5, q_mos5, q_hat5 = eval(koniq10k_test_loader, 'test', 'koniq10k')
    scene_acc6, dist_acc6, q_mos6, q_hat6 = eval(kadid10k_test_loader, 'test', 'kadid10k')

    all_scene['live'].append(scene_acc1)
    all_scene['csiq'].append(scene_acc2)
    all_scene['bid'].append(scene_acc3)
    all_scene['clive'].append(scene_acc4)
    all_scene['koniq10k'].append(scene_acc5)
    all_scene['kadid10k'].append(scene_acc6)

    all_dist['live'].append(dist_acc1)
    all_dist['csiq'].append(dist_acc2)
    all_dist['bid'].append(dist_acc3)
    all_dist['clive'].append(dist_acc4)
    all_dist['koniq10k'].append(dist_acc5)
    all_dist['kadid10k'].append(dist_acc6)

    srcc1, krcc1, plcc1, rmse1 = compute_metrics(q_hat1, q_mos1)
    srcc2, krcc2, plcc2, rmse2 = compute_metrics(q_hat2, q_mos2)
    srcc3, krcc3, plcc3, rmse3 = compute_metrics(q_hat3, q_mos3)
    srcc4, krcc4, plcc4, rmse4 = compute_metrics(q_hat4, q_mos4)
    srcc5, krcc5, plcc5, rmse5 = compute_metrics(q_hat5, q_mos5)
    srcc6, krcc6, plcc6, rmse6 = compute_metrics(q_hat6, q_mos6)

    all_srcc['live'].append(srcc1)
    all_srcc['csiq'].append(srcc2)
    all_srcc['bid'].append(srcc3)
    all_srcc['clive'].append(srcc4)
    all_srcc['koniq10k'].append(srcc5)
    all_srcc['kadid10k'].append(srcc6)

    all_krcc['live'].append(krcc1)
    all_krcc['csiq'].append(krcc2)
    all_krcc['bid'].append(krcc3)
    all_krcc['clive'].append(krcc4)
    all_krcc['koniq10k'].append(krcc5)
    all_krcc['kadid10k'].append(krcc6)

    all_plcc['live'].append(plcc1)
    all_plcc['csiq'].append(plcc2)
    all_plcc['bid'].append(plcc3)
    all_plcc['clive'].append(plcc4)
    all_plcc['koniq10k'].append(plcc5)
    all_plcc['kadid10k'].append(plcc6)

    all_rmse['live'].append(rmse1)
    all_rmse['csiq'].append(rmse2)
    all_rmse['bid'].append(rmse3)
    all_rmse['clive'].append(rmse4)
    all_rmse['koniq10k'].append(rmse5)
    all_rmse['kadid10k'].append(rmse6)


def final_avg(all_srcc, all_krcc, all_plcc, all_rmse, all_scene, all_dist):
    median_srcc = np.mean(np.array(all_srcc))
    median_krcc = np.mean(np.array(all_krcc))
    median_plcc = np.mean(np.array(all_plcc))
    median_rmse = np.mean(np.array(all_rmse))
    median_scene = np.mean(np.array(all_scene))
    median_dist = np.mean(np.array(all_dist))

    std_srcc = np.std(np.array(all_srcc))
    std_krcc = np.std(np.array(all_krcc))
    std_plcc = np.std(np.array(all_plcc))
    std_rmse = np.std(np.array(all_rmse))
    std_scene = np.std(np.array(all_scene))
    std_dist = np.std(np.array(all_dist))

    return [median_srcc, median_krcc, median_plcc, median_rmse, median_scene, median_dist, std_srcc, std_krcc,
            std_plcc, std_rmse, std_scene, std_dist]

#live_results
live_results = final_avg(all_srcc['live'], all_krcc['live'], all_plcc['live'], all_rmse['live'], all_scene['live'], all_dist['live'])
csiq_results = final_avg(all_srcc['csiq'], all_krcc['csiq'], all_plcc['csiq'], all_rmse['csiq'], all_scene['csiq'], all_dist['csiq'])
bid_results = final_avg(all_srcc['bid'], all_krcc['bid'], all_plcc['bid'], all_rmse['bid'], all_scene['bid'], all_dist['bid'])
clive_results = final_avg(all_srcc['clive'], all_krcc['clive'], all_plcc['clive'], all_rmse['clive'], all_scene['clive'], all_dist['clive'])
koniq10k_results = final_avg(all_srcc['koniq10k'], all_krcc['koniq10k'], all_plcc['koniq10k'], all_rmse['koniq10k'], all_scene['koniq10k'], all_dist['koniq10k'])
kadid10k_results = final_avg(all_srcc['kadid10k'], all_krcc['kadid10k'], all_plcc['kadid10k'], all_rmse['kadid10k'], all_scene['kadid10k'], all_dist['kadid10k'])

print('live')
print(live_results)
print('csiq')
print(csiq_results)
print('bid')
print(bid_results)
print('clive')
print(clive_results)
print('koniq10k')
print(koniq10k_results)
print('kadid10k')
print(kadid10k_results)

