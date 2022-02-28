# TPLinker

**TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking**

This repository contains all the code of the official implementation for the paper: **[TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://www.aclweb.org/anthology/2020.coling-main.138.pdf).** The paper has been accepted to appear at **COLING 2020**. \[[slides](https://drive.google.com/file/d/1UAIVkuUgs122k02Ijln-AtaX2mHz70N-/view?usp=sharing)\] \[[poster](https://drive.google.com/file/d/1iwFfXZDjwEz1kBK8z1To_YBWhfyswzYU/view?usp=sharing)\]

TPLinker是一个联合抽取模型，解决了**关系重叠**和**嵌套实体**的问题，不受**曝光偏差的影响，并在纽约时报上取得了SOTA性能（TPLinker。**91.9**, TPlinkerPlus: **92.6（+3.0）**）和WebNLG（TPLinker: **91.9**, TPlinkerPlus: **92.3 (+0.5)**).  请注意，TPLinkerPlus的细节将在扩展论文中发表，该论文仍在进行中。
**注：在提出新问题之前，请参考Q&A和已关闭的问题，以找到您的问题。


- [Model](#model)
- [Results](#results)
- [Usage](#usage)
  * [Prerequisites](#prerequisites)
  * [Data](#data)
    + [download data](#download-data)
    + [build data](#build-data)
  * [Pretrained Model and Word Embeddings](#pretrained-model-and-word-embeddings)
  * [Train](#train)
    + [super parameters](#super-parameters)
  * [Evaluation](#evaluation)
- [Citation](#citation)
- [Q&A](#frequently-asked-questions)

## Update
* 2020.11.01: 修复了BuildData.ipynb和build_data_config.yaml中的错误并添加了注释；TPLinkerPlus现在可以支持实体分类，数据格式见[build data](#build-data)；更新了[datasets](#download-data)（为TPLinkerPlus添加`entity_list`）。
* 2020.12.04: `build_data_config.yaml`中原来的默认参数都是针对中文数据集的。这可能会对再现结果产生误导。我改回了用于英文数据集的参数。**注意，对于英文数据集，你必须将 "忽略子词 "设置为 "true"，否则会影响性能，无法达到论文中报告的分数。
* 2020.12.09: 我们公布了快速测试的模型状态。见[超级参数](#super-parameters)。
* 2021.03.22: 在README中增加问答部分。


## Model
<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/97800160-0d4c0c00-1c6e-11eb-960a-0574a6e1f6e9.png" alt="framework" width="768"/>
</p>

## Results
<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/95207775-d0c9f380-081a-11eb-95e9-976c58acab84.png" alt="data_statistics" width="768"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/95995846-9557a680-0e64-11eb-96ca-da77b2f88dbf.png" alt="main_res_plus" width="768"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/95995856-97216a00-0e64-11eb-8b81-2910e48eb3b2.png" alt="split_res_plus" width="768"/>
</p>

## Usage
### Prerequisites
我们的实验是在Python 3.6和Pytorch == 1.6.0上进行的。
主要的要求是：
* tqdm
* glove-python-binary==0.2.0
* transformers==3.0.2
* wandb # for logging the results
* yaml

In the root directory, run
```bash
pip install -e .
```
### Data
#### 下载数据
按照[CasRel](https://github.com/weizhepei/CasRel/tree/master/data)获取并预处理NYT*和WebNLG*（注意：由CasRel命名为NYT和WebNLG）。
以NYT*为例，将train_triples.json和dev_triples.json重命名为train_data.json和valid_data.json，并将其移至`ori_data/nyt_star`，将所有test*.json置于`ori_data/nyt_star/test_data`下。WebNLG*的过程也是如此。

Get raw NYT from [CopyRE](https://github.com/xiangrongzeng/copy_re),  rename raw_train.json and raw_valid.json to train_data.json and valid_data.json and move them to `ori_data/nyt`, rename raw_test.json to test_data.json and put it under `ori_data/nyt/test_data`.

Get WebNLG from [ETL-Span](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data), rename train.json and dev.json to train_data.json and valid_data.json and move them to `ori_data/webnlg`, rename test.json to test_data.json and put it under `ori_data/webnlg/test_data`.

如果你麻烦自己准备数据，你可以下载我们预处理过的[数据集](https://drive.google.com/file/d/1RxBVMSTgBxhGyhaPEWPdtdX1aOmrUPBZ/view?usp=sharing)。

#### build data
Build data by `preprocess/BuildData.ipynb`.
设置配置 in `preprocess/build_data_config.yaml`.
在配置文件中，设置`exp_name`对应于目录名称，设置`ori_data_format`对应于数据的源项目名称。
例如，要建立NYT*，设置`exp_name`为`nyt_star`，设置`ori_data_format`为`casrel`。更多细节见`build_data_config.yaml`。
如果你想在其他数据集上运行，将它们转化为TPLinker的正常格式，然后将`exp_name`设置为`<你的文件夹名称>`，并将`ori_data_format`设置为`tplinker`。

```python
[{
"id": <text_id>,
"text": <text>,
"relation_list": [{
    "subject": <subject>,
    "subj_char_span": <character level span of the subject>, # e.g [3, 10] 这个是可选的。如果没有这个键，在构建数据时，在 "build_data_config.yaml "中把 "add_char_span "设置为true。
    "object": <object>,
    "obj_char_span": <character level span of the object>, # optional
    "predicate": <predicate>,
 }],
"entity_list": [{ # 这个是可选的，只适用于TPLinkerPlus。如果没有这个键，BuildData.ipynb将根据关系列表自动生成一个实体列表。
    "text": <entity>,
    "type": <entity_type>,
    "char_span": <character level span of the object>, # This key relys on subj_char_span and obj_char_span in relation_list, if not given, set "add_char_span" to true in "build_data_config.yaml".
 }],
}]
```

### 预训练模型和词嵌入
下载  [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `../pretrained_models`. Pretrain word embeddings by `preprocess/Pretrain_Word_Embedding.ipynb` and put models under `../pretrained_emb`.

如果你麻烦自己去训练词嵌入，可以直接使用[我们的](https://drive.google.com/file/d/1IQu_tdqEdExqyaeXJ4QSjUbWPQ3kxZKE/view?usp=sharing)。

### Train
设置配置 `tplinker/config.py` 按如下方式:
```python
common["exp_name"] = nyt_star # webnlg_star, nyt, webnlg
common["device_num"] = 0 # 1, 2, 3 ...
common["encoder"] = "BERT" # BiLSTM
train_config["hyper_parameters"]["batch_size"] = 24 # 6 for webnlg and webnlg_star
train_config["hyper_parameters"]["match_pattern"] = "only_head_text" # "only_head_text" for webnlg_star and nyt_star; "whole_text" for webnlg and nyt.

# 如果使用的是BiLSTM,那么需要加载预训练word embedding
bilstm_config["pretrained_word_embedding_path"] = ""../pretrained_word_emb/glove_300_nyt.emb""

# Leave the rest as default
```

开始训练
```
cd tplinker
python train.py
```

#### 超参数
**TPLinker**
```
# NYT*
T_mult: 1
batch_size: 24
dist_emb_size: -1
ent_add_dist: false
epochs: 100
inner_enc_type: lstm
log_interval: 10
loss_weight_recover_steps: 12000
lr: 0.00005
match_pattern: only_head_text
max_seq_len: 100
rel_add_dist: false
rewarm_epoch_num: 2
scheduler: CAWR
seed: 2333
shaking_type: cat
sliding_len: 20

# NYT
match_pattern: whole_text
...(the rest is the same as above)

# WebNLG*
batch_size: 6
loss_weight_recover_steps: 6000
match_pattern: only_head_text
...

# WebNLG
batch_size: 6
loss_weight_recover_steps: 6000
match_pattern: whole_text
...
```

We also provide model states for fast tests. You can download them [here](https://drive.google.com/drive/folders/1GCWNXQN-L09oSG9ZFYi979wk2dTS9h-3?usp=sharing)! 
You can get results as follows:
```
Before you run, make sure:
1. transformers==3.0.2
2. "max_test_seq_len" is set to 512
3. "match_pattern" should be the same as in the training:  `whole_text` for NYT/WebNLG, `only_head_text` for NYT*/WebNLG*.

# The test_triples and the test_data are the complete test datasets. The tuple means (precision, recall, f1).
# In the codes, I did not set the seed in a right way, so you might get different scores (higher or lower). But they should be very close to the results in the paper. 

# NYT*
{'2og80ub4': {'test_triples': (0.9118290017002562,
                               0.9257706535141687,
                               0.9187469407233642),
              'test_triples_1': (0.8641055045871312,
                                 0.929099876695409,
                                 0.8954248365513463),
              'test_triples_2': (0.9435444280804642,
                                 0.9196172248803387,
                                 0.9314271867684752),
              'test_triples_3': (0.9550056242968554,
                                 0.9070512820511851,
                                 0.9304109588540408),
              'test_triples_4': (0.9635099913118189,
                                 0.9527491408933889,
                                 0.9580993520017547),
              'test_triples_5': (0.9177877428997133,
                                 0.9082840236685046,
                                 0.9130111523662223),
              'test_triples_epo': (0.9497520661156711,
                                   0.932489451476763,
                                   0.9410415983777494),
              'test_triples_normal': (0.8659532526048757,
                                      0.9281617869000626,
                                      0.895979020929055),
              'test_triples_seo': (0.9476190476190225,
                                   0.9258206254845974,
                                   0.9365930186452366)}}
                                   
# NYT
{'y84trnyf': {'test_data': (0.916494217894085,
                            0.9272167487684615,
                            0.9218243035924758)}}

# WebNLG*
{'2i4808qi': {'test_triples': (0.921855146124465,
                               0.91777356103726,
                               0.919809825623476),
              'test_triples_1': (0.8759398496237308,
                                 0.8759398496237308,
                                 0.8759398495737308),
              'test_triples_2': (0.9075144508667897,
                                 0.9235294117644343,
                                 0.9154518949934687),
              'test_triples_3': (0.9509043927646122,
                                 0.9460154241642813,
                                 0.9484536081971787),
              'test_triples_4': (0.9297752808986153,
                                 0.9271708683470793,
                                 0.928471248196584),
              'test_triples_5': (0.9360730593603032,
                                 0.8951965065498275,
                                 0.9151785713781879),
              'test_triples_epo': (0.9764705882341453,
                                   0.9431818181807463,
                                   0.959537572203241),
              'test_triples_normal': (0.8780487804874479,
                                      0.8780487804874479,
                                      0.8780487804374479),
              'test_triples_seo': (0.9299698795180023,
                                   0.9250936329587321,
                                   0.9275253473025405)}}
                                   
# WebNLG
{'1g7ehpsl': {'test': (0.8862619808306142,
                       0.8630989421281354,
                       0.8745271121819839)}}
```

**TPLinkerPlus**
```
# NYT*/NYT
# The best F1: 0.931/0.934 (on validation set), 0.926/0.926 (on test set)
T_mult: 1
batch_size: 24
epochs: 250
log_interval: 10
lr: 0.00001
max_seq_len: 100
rewarm_epoch_num: 2
scheduler: CAWR
seed: 2333
shaking_type: cln
sliding_len: 20
tok_pair_sample_rate: 1

# WebNLG*/WebNLG
# The best F1: 0.934/0.889 (on validation set), 0.923/0.882 (on test set)
T_mult: 1 
batch_size: 6 
epochs: 250
log_interval: 10
lr: 0.00001
max_seq_len: 100
rewarm_epoch_num: 2
scheduler: CAWR
seed: 2333
shaking_type: cln
sliding_len: 20
tok_pair_sample_rate: 1
```

### Evaluation
Set configuration in `tplinker/config.py` as follows:
```python

eval_config["model_state_dict_dir"] = "./wandb" # if use wandb, set "./wandb"; if you use default logger, set "./default_log_dir" 
eval_config["run_ids"] = ["46qer3r9", ] # If you use default logger, run id is shown in the output and recorded in the log (see train_config["log_path"]); If you use wandb, it is logged on the platform, check the details of the running projects.
eval_config["last_k_model"] = 1 # only use the last k models in to output results
# Leave the rest as the same as the training
```
Start evaluation by running `tplinker/Evaluation.ipynb`

# Citation
```
@inproceedings{wang-etal-2020-tplinker,
    title = "{TPL}inker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking",
    author = "Wang, Yucheng and Yu, Bowen and Zhang, Yueyang and Liu, Tingwen and Zhu, Hongsong and Sun, Limin",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.138",
    pages = "1572--1582"
}
```

# Frequently Asked Questions
1. 为什么你把所有的实体都变成了 "DEFAULT "类型？TPLinker不能识别实体的类型？
因为对于关系抽取任务来说，没有必要识别实体的类型，因为一个预定义的关系通常有固定的主语和宾语类型。例如，（"地方"，"包含"，"地方"）和（"国家"，"首都"，"城市"）。如果你需要这个特征，你可以重新定义EH-to-ET序列的输出标签，或者使用TPLinkerPlus，它已经有了这个特征。如果你使用TPLinkerPlus，只需在entity_list中设置一个特定的实体类型而不是 "DEFAULT"。

2. <dataset_name>_star和<dataset_name>之间有什么区别？
为了进行公平的比较，我们使用以前的工作中的预处理数据。NYT来自[CopyRE](https://github.com/xiangrongzeng/copy_re)（原始版本）；WebNLG来自[ETL-span](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)；NYT*和WebNLG*来自[CasRel](https://github.com/weizhepei/CasRel/tree/master/data) 。关于这些数据集的详细描述，请参考我们论文的数据描述部分。
对NYT★和WebNLG★使用了部分匹配，如果主语实体和宾语实体的关系和头部都是正确的，就认为提取的三元祖是正确的；对NYT和WebNLG使用了精确匹配，需要对主语和宾语的整个跨度进行匹配
   
3. 以前的工作声称，WebNLG有246个关系。为什么你们使用关系较少的WebNLG和WebNLG*？
我们直接使用之前SoTA模型预处理过的数据集。我们从他们的Github资源库中获取。我们重新计算了数据集中的关系，发现WebNLG和WebNLG*的真实关系数比他们在论文中声称的要少。事实上，他们使用的是一个子集（6000多个样本）WebNLG，而不是原始的WebNLG（10000多个样本），但使用的是原始的统计数据。如果你重新统计他们数据集中的关系，你也会发现这个问题。
   
4. 我的训练过程比你所说的慢得多（24小时）。你能给出任何建议吗？
请使用我在README中提供的超参数来重现结果。如果你想改变它们，请使训练的max_seq_length小于或等于100。根据我的经验，将max_seq_length增加到大于100并没有带来明显的改善，反而对训练速度有很大的影响。使用较小的batch_size也会加快训练速度，但如果你使用太小的batch_size，可能会损害性能。

5. 我看到你把长文拆成了短文。你可能会因为拆分而错过一些glod实体和关系。你是如何处理这个问题的？
我们使用一个滑动窗口来分割样本，这将包含大部分的glod实体和关系。我们在训练和推理中使用不同的最大长度（max_seq_length）。前者被设置为100，后者被设置为512。训练时损失一两个实体或关系是可以的，这不会对训练产生太大影响。


6. 如何将这个模型用于中文的数据集？
请参考问题#15。

7. 我的f1分数在很长时间内都是0。你有什么办法吗？
请参考问题#24和#25。
