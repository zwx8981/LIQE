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


# Demo2: import LIQE as an independent module and perform inference
```bash
python demo2.py
```

# Pre-trained weights

Google Drive: 

https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view?usp=sharing

百度网盘： 

链接: https://pan.baidu.com/s/1KHjj7T8y2H_eKE6w7HnWJA 提取码: 2b8v 

# New! Zero-shot (cross-database) performance on the AIGC dataset ([*AGIQA-3K*](https://arxiv.org/pdf/2306.04717.pdf)

| BIQA Model      | SRCC     | Paper     |
| ---------- | :-----------:  | :-----------: |
| DBCNN     | 0.6454     | [](https://ieeexplore.ieee.org/abstract/document/8576582) |

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

