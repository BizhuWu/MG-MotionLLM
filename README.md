<div align="center">
<h1>(CVPR 2025) MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities</h1>

[Bizhu Wu](https://scholar.google.com/citations?user=u7nZ3bgAAAAJ&hl=en) · [Jinheng Xie](https://scholar.google.com/citations?user=smbRMokAAAAJ&hl=en) · [Keming Shen]() · [Zhe Kong](https://scholar.google.com/citations?user=4X3yLwsAAAAJ&hl=en)

[Jianfeng Ren*](https://scholar.google.com/citations?user=ZZ928OgAAAAJ&hl=en) · [Ruibin Bai](https://scholar.google.com/citations?user=oP6AThIAAAAJ&hl=en) · [Rong Qu](https://scholar.google.com/citations?user=ErszCRMAAAAJ&hl=en) ·   [Linlin Shen*](https://scholar.google.com/citations?user=AZ_y9HgAAAAJ&hl=en)

<sup>*</sup>Corresponding Authors

[![arXiv](https://img.shields.io/badge/arXiv-MGMotionLLM-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2504.02478)

</div>



## Table of Content
* [1. Paper Description](#1-paper-description)
* [2. Installation](#2-installation)
* [3. Pretrained Models](#3-pretrained-models)
* [4. Datasets](#4-datasets)
* [5. Evaluation](#5-evaluation)
* [6. Train Your Own Model](#6-train-your-own-model)
* [7. Visualization](#7-visualization)
* [8. Acknowledgement](#8-acknowledgement)
* [9. Bibtex](#9-bibtex)



## 1. Paper Description
**MG-MotionLLM** can address diverse motion-relevant tasks at multiple granularities by giving different instructions in a unified manner. 
- **coarse-grained**: e.g. text-to-motion and motion captioning (upper block) 
- **fine-grained**: e.g. motion-to-detailed text and motion localization (bottom block).

<div align="center">    
  <img src="./assets/teaser.png" alt="teaser" align=center, height=500 />
</div>


To achieve this, we propose multi-granularity training scheme with novel auxiliary tasks captures motion-related features at different levels, improving understanding across a wide range of tasks. Specifically, we pretrain the model with a total of **28** distinct motion-relevant tasks, including **12** existing classical **coarse-grained** tasks and **16** newly proposed **fine-grained** ones. Here, we display examples of prompt templates for a part of tasks used during training.

<div align="center">    
  <img src="./assets/tasks_template.png" alt="tasks_template" align=center />
</div>



## 2. Installation

### 2.1. Environment
```
conda env create -f environment.yml
conda activate mg-motionllm
```

### 2.2. Dependencies
For text-to-motion evaluation
```
bash prepare/download_evaluators.sh
bash prepare/download_glove.sh
```



## 3. Pretrained Models
For pretrained **VQ-VAE** models
```
bash prepare/download_vqvae.sh
```

Once downloaded, you should have a folder like this:
```
MG-MotionLLM/checkpoints
├── pretrained_vqvae
│   ├── kit.pth
│   └── t2m.pth
```


For pretrained **MG-MotionLLM** models, you have two ways to download:
1. manually download from HuggingFace:

| Model | Link |
|---------|---------|
| GSPretrained-small | [GSPretrained-small](https://huggingface.co/wbz0505/GSPretrained-small) |
| t2m-ft-from-GSPretrained-small | [t2m-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/t2m-ft-from-GSPretrained-small) |
| m2t-ft-from-GSPretrained-small | [m2t-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/m2t-ft-from-GSPretrained-small) |
| tdt2m-ft-from-GSPretrained-small | [tdt2m-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/tdt2m-ft-from-GSPretrained-small) |
| m2dt-ft-from-GSPretrained-small | [m2dt-ft-from-GSPretrained-small](https://huggingface.co/wbz0505/m2dt-ft-from-GSPretrained-small) |
| GSPretrained-base | [GSPretrained-base](https://huggingface.co/wbz0505/GSPretrained-base) |
| t2m-ft-from-GSPretrained-base | [t2m-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/t2m-ft-from-GSPretrained-base) |
| m2t-ft-from-GSPretrained-base | [m2t-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/m2t-ft-from-GSPretrained-base) |
| tdt2m-ft-from-GSPretrained-base | [tdt2m-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/tdt2m-ft-from-GSPretrained-base) |
| m2dt-ft-from-GSPretrained-base | [m2dt-ft-from-GSPretrained-base](https://huggingface.co/wbz0505/m2dt-ft-from-GSPretrained-base) |
| GSPretrained-large | [GSPretrained-large](https://huggingface.co/wbz0505/GSPretrained-large) |
| t2m-ft-from-GSPretrained-large | [t2m-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/t2m-ft-from-GSPretrained-large) |
| m2t-ft-from-GSPretrained-large | [m2t-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/m2t-ft-from-GSPretrained-large) |
| tdt2m-ft-from-GSPretrained-large | [tdt2m-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/tdt2m-ft-from-GSPretrained-large) |
| m2dt-ft-from-GSPretrained-large | [m2dt-ft-from-GSPretrained-large](https://huggingface.co/wbz0505/m2dt-ft-from-GSPretrained-large) |

2. use code to download them. For example,
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'wbz0505/t2m-ft-from-GSPretrained-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```



## 4. Datasets
We are using two 3D human motion-language dataset: HumanML3D and FineMotion.

1. Please follow [HumanML3D](https://github.com/EricGuo5513/HumanML3D) to download and prepare HumanML3D dataset and put them under the directory `dataset` like:
```
./dataset/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy    # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── Std.npy     # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```


2. Please follow [FineMotion](https://github.com/BizhuWu/FineMotion) to download detailed body movement descriptions for motions of HumanML3D, *i.e.*, `BPMSD_auto.zip` and `BPMSD_human.zip`.
You should create an empty directory named `finemotion_texts` under the directory `HumanML3D`, 
and put the `BPMSD_auto.zip` and `BPMSD_human.zip` into this newly created directory and unzip it to obtain the json files `BPMSD_auto.json` and `BPMSD_human.json`.
Now, your `dataset` directory should look like: 
```
MG-MotionLLM/dataset/HumanML3D/
├── new_joint_vecs/
├── finemotion_texts/     # here
│   ├── BPMSD_auto.zip
│   ├── BPMSD_auto.json
│   ├── BPMSD_human.zip
│   ├── BPMSD_human.json
├── texts/
├── Mean.npy
├── Std.npy
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```



3. To tokenize the motion data used for training MG-MotionLLM, please follow the instructions below
```python
# Encode the motions to tokens by pretrianed VQ-VAE and save the token sequence results under `./dataset/HumanML3D/VQVAE/`
# For pretrained VQ-VAE, you can use the model provided.
CUDA_VISIBLE_DEVICES=0 python3 scripts/tokenized_motion.py

# The following script is used to generate motion tokens that strictly aligned with detailed text,
# and save the token sequence results under `./dataset/HumanML3D/VQVAE_start0/`
CUDA_VISIBLE_DEVICES=0 python3 scripts/tokenized_motion_start0.py
```



## 5. Evaluation

To evaluate our models on the **Text-to-Motion** task, 
please use the following command:
```python
# from our final t2m model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_t2m.py --model_name ./t2m-ft-from-GSPretrained-base/checkpoint-300000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_t2m.py --model_name ./GSPretrained_base/checkpoint-300000 
```


To evaluate our models on the **Motion-to-Text** task, 
please use the following command:
```python
# from our final m2t model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_m2t.py --model_name ./m2t-ft-from-GSPretrained-base/checkpoint-100000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_m2t.py --model_name ./GSPretrained_base/checkpoint-300000 
```
Similarity, we follow MotionGPT to use [nlg-metricverse](https://github.com/disi-unibo-nlp/nlg-metricverse) 
to implement linguistic metrics in motion translation task.


To evaluate our models on the **(Text, Detailed Text)-to-Motion** task, 
please use the following command:
```python
# from our final tdt2m model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_tdt2m.py --model_name ./tdt2m-ft-from-GSPretrained-base/checkpoint-300000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_tdt2m.py --model_name ./GSPretrained_base/checkpoint-300000 
```

To evaluate our models on the **Motion-to-Detailed Text** task, 
please use the following command:
```python
# from our final tdt2m model (Granularity-Synergy Pre-training + Task-Specific Instruction Tuning)
CUDA_VISIBLE_DEVICES=0 python3 eval_m2dt.py --model_name ./m2dt-ft-from-GSPretrained-base/checkpoint-300000
# or
# from our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 eval_m2dt.py --model_name ./GSPretrained_base/checkpoint-300000 
```
Similarity, we follow MotionGPT to use [nlg-metricverse](https://github.com/disi-unibo-nlp/nlg-metricverse) 
to implement linguistic metrics in motion translation task.



## 6. Train Your Own Model

To pretrain our Granularity-Synergy Pre-trained model, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_pretraining.py --output_dir ./GSPretrained_base
```


To train a model on the **Text-to-Motion** task, 
please use the following command:
```python
# from the T5 series (motion-unaware language models)
CUDA_VISIBLE_DEVICES=0 python3 main_t2m.py --model_name google-t5/t5-base --output_dir ./t2m-ft-from-t5-base
# or
# fine-tune our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 main_t2m.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./t2m-ft-from-GSPretrained-base
```


To train a model on the **Motion-to-Text** task, 
please use the following command:
```python
# from the T5 series (motion-unaware language models)
CUDA_VISIBLE_DEVICES=0 python3 main_m2t.py --model_name google-t5/t5-base --output_dir ./m2t-ft-from-t5-base
# or
# fine-tune our Granularity-Synergy Pre-trained model
CUDA_VISIBLE_DEVICES=0 python3 main_m2t.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./m2t-ft-from-GSPretrained-base --max_steps 100000

```


To train a model on the **(Text, Detailed Text)-to-Motion** task, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_tdt2m.py --model_name google-t5/t5-base --output_dir ./tdt2m-ft-from-t5-base
```
or to fine-tune our Granularity-Synergy Pre-trained model, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_tdt2m.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./tdt2m-ft-from-GSPretrained-base
```


To train a model on the **Motion-to-Detailed Text** task, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_m2dt.py --model_name google-t5/t5-base --output_dir ./m2dt-ft-from-t5-base
```
or to fine-tune our Granularity-Synergy Pre-trained model, 
please use the following command:
```python
CUDA_VISIBLE_DEVICES=0 python3 main_m2dt.py --model_name ./GSPretrained_base/checkpoint-300000 --output_dir ./m2dt-ft-from-GSPretrained-base
```





## 7. Visualization
We display some novel applications of our MG-MotionLLM.
- **text-driven fine-grained motion editing**: Temporal Editing (left), Spatial Editing (middle), and Spatial-Temporal Editing (right).

<div align="center">    
  <img src="./assets/editing.png" alt="edit" align=center />
</div>

- **fine-grained captioning** of both whole (up) and partial (bottom) motion sequences, and **motion localization via fine-grained textual description** (middle).

<div align="center">    
  <img src="./assets/novel_apps.png" alt="novel_apps" align=center />
</div>




## 8. Acknowledgement
We appreciate helps from the following public code like 
* [MotionGPT](https://github.com/qiqiApink/MotionGPT)
* [MotionGPT](https://github.com/OpenMotionLab/MotionGPT)
* [TM2T](https://github.com/EricGuo5513/TM2T)
* [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
* [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)



## 9. Bibtex
If you use our code in your research, kindly cite our work:

```bibtex
@InProceedings{Wu_2025_CVPR,
    author    = {Wu, Bizhu and Xie, Jinheng and Shen, Keming and Kong, Zhe and Ren, Jianfeng and Bai, Ruibin and Qu, Rong and Shen, Linlin},
    title     = {MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27849-27858}
}

@article{wu2025mg,
  title={MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities},
  author={Wu, Bizhu and Xie, Jinheng and Shen, Keming and Kong, Zhe and Ren, Jianfeng and Bai, Ruibin and Qu, Rong and Shen, Linlin},
  journal={arXiv preprint arXiv:2504.02478},
  year={2025}
}
```
