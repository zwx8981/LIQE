# Language-Image Quality Evaluator (LIQE)

The official repo of [**Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective**](https://arxiv.org/pdf/2303.14968.pdf) (CVPR2023)

# Abstract

We aim at advancing blind image quality assessment (BIQA), which predicts the human perception of image quality without any reference information. We develop a general and automated multitask learning scheme for BIQA to exploit  auxiliary knowledge from other tasks, in a way that the model parameter sharing and the loss weighting are determined automatically. Specifically, we first describe all candidate label combinations (from multiple tasks) using a textual template, and compute the joint probability from the cosine similarities of the visual-textual embeddings. Predictions of each task can be inferred from the joint distribution, and optimized by carefully designed loss functions. Through comprehensive experiments on learning three tasks - BIQA, scene classification, and distortion type identification, we verify that the proposed BIQA method 1) benefits from the  scene classification and distortion type identification tasks and outperforms the state-of-the-art on multiple IQA datasets, 2) is  more robust in the group maximum differentiation competition, and 3) realigns the quality annotations from different IQA datasets more effectively.

![image](https://github.com/zwx8981/LIQE/blob/main/clip_biqa.png)

# Requirement

torch 1.8+

torchvision

Python 3

pip install ftfy regex tqdm

pip install git+https://github.com/openai/CLIP.git

# Training on 10 splits
```bash
python train_unique_clip_weight.py
```

# Evaluation on test-sets
```bash
python BIQA_benchmark.py
```

# Demo: detailed pipeline
```bash
python demo.py
```


# Demo2: import LIQE as a standalone module and perform inference
```bash
python demo2.py
```

# Pre-trained weights

Google Drive: 

https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view?usp=sharing

百度网盘： 

链接: https://pan.baidu.com/s/1KHjj7T8y2H_eKE6w7HnWJA 提取码: 2b8v 

## New! Zero-shot (cross-database) performance on the AIGC dataset ([*AGIQA-3K*](https://arxiv.org/pdf/2306.04717.pdf)) (perception subset)

| BIQA Model      | SRCC     | Paper     |
| ---------- | :-----------:  | :-----------: |
| DBCNN     | 0.6454     | [DBCNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8576582) |
| HyperIQA     | 0.6291     | [HyperIQA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) |
| TReS     | 0.6460     | [TReS](https://openaccess.thecvf.com/content/WACV2022/papers/Golestaneh_No-Reference_Image_Quality_Assessment_via_Transformers_Relative_Ranking_and_Self-Consistency_WACV_2022_paper.pdf) |
| UNIQUE     | 0.6659     | [UNIQUE](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9369977) |
| MUSIQ     | 0.6294     | [MUSIQ](https://openaccess.thecvf.com/content/ICCV2021/papers/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.pdf) |
| PaQ-2-PiQ     | 0.5023     | [PaQ2PiQ](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ying_From_Patches_to_Pictures_PaQ-2-PiQ_Mapping_the_Perceptual_Space_of_CVPR_2020_paper.pdf) |
| CLIPIQA     | 0.6580     | [CLIPIQA](https://ojs.aaai.org/index.php/AAAI/article/view/25353) |
| CLIPIQA+     | 0.6831     | [CLIPIQA+](https://ojs.aaai.org/index.php/AAAI/article/view/25353) |
| MANIQA     | 0.6950     | [MANIQA](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.pdf) |
| LIQE (Ours)     | **0.7212**     | [LIQE](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_CVPR_2023_paper.pdf) |

# Citation
```bash
@inproceedings{zhang2023liqe,  
  title={Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective},  
  author={Zhang, Weixia and Zhai, Guangtao and Wei, Ying and Yang, Xiaokang and Ma, Kede},  
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},  
  pages={14071--14081},
  year={2023}
}
```

