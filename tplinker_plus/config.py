import string
import random

common = {
    "exp_name": "duie", # ace05_lu,  duie的实体数量26，关系数量48
    "rel2id": "rel2id.json",
    "ent2id": "ent2id.json",
    "device_num": 0,
#     "encoder": "BiLSTM",
    "encoder": "BERT", 
    "hyper_parameters": {
        "shaking_type": "cln_plus",
        "inner_enc_type": "lstm",
        # match_pattern: only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span, event_extraction
        "match_pattern": "whole_text", 
    },
}
common["run_name"] = "{}+{}+{}".format("TP2", common["hyper_parameters"]["shaking_type"], common["encoder"]) + ""

run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
train_config = {
    "train_data": "train_data.json",  #训练集
    "valid_data": "valid_data.json",  #验证集
    "rel2id": "rel2id.json",  # 关系到id的映射
    # "logger": "wandb", # if wandb, comment the following four lines

#     # if logger is set as default, uncomment the following four lines
    "logger": "default",
    "run_id": run_id,
    "log_path": "./default_log_dir/default.log",
    "path_to_save_model": "./default_log_dir/{}".format(run_id),

    # 保存模型，只有当f1 score大于当前值f1_2_save的时候
    "f1_2_save": 0,
    # 是否根据配置文件，从头开始训练， 和model_state_dict_path相斥
    "fr_scratch": True,
    # 一些需要写到日志中的字符
    "note": "start from scratch",
    # 如果不是scratch开始训练, 需要设定一个model_state_dict
    "model_state_dict_path": "",
    "hyper_parameters": {
        "batch_size": 16,
        "epochs": 100,
        "seed": 2333,
        "log_interval": 10,
        "max_seq_len": 128,
        "sliding_len": 20,
        "scheduler": "CAWR", # Step
        "ghm": False, # 如果你想使用GHM来调整梯度的权重，请设置为True，这将加速训练过程，并可能改善结果。(注意，当前版本的GHM是不稳定的，可能会损害结果）。
        "tok_pair_sample_rate": 1, # (0, 1] 你想对多少百分比的token paris进行抽样训练，如果设置为小于1，这将降低训练速度。
    },
}

eval_config = {
    "model_state_dict_dir": "./wandb", # if use wandb, set "./wandb", or set "./default_log_dir" if you use default logger
    "run_ids": ["1a70p109", ],
    "last_k_model": 1,
    "test_data": "*test*.json", # "*test*.json"
    
    # results
    "save_res": False,
    "save_res_dir": "../results",
    
    # score: set true only if test set is tagged
    "score": True,
    
    "hyper_parameters": {
        "batch_size": 16,
        "force_split": False,
        "max_seq_len": 512,
        "sliding_len": 50,
    },
}

bert_config = {
    "data_home": "data4bert",
    "bert_path": "pretrained_models/chinese-bert-wwm-ext", # bert-base-cased， hfl/chinese-bert-wwm-ext
    "hyper_parameters": {
        "lr": 5e-5,
    },
}
bilstm_config = {
    "data_home": "../data4bilstm",
    "token2idx": "token2idx.json",
    "pretrained_word_embedding_path": "../../pretrained_emb/glove_300_nyt.emb",
    "hyper_parameters": {
         "lr": 1e-3,
         "enc_hidden_size": 300,
         "dec_hidden_size": 600,
         "emb_dropout": 0.1,
         "rnn_dropout": 0.1,
         "word_embedding_dim": 300,
    },
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------dicts above is all you need to set---------------------------------------------------
if common["encoder"] == "BERT":
    hyper_params = {**common["hyper_parameters"], **bert_config["hyper_parameters"]}
    common = {**common, **bert_config}
    common["hyper_parameters"] = hyper_params
elif common["encoder"] == "BiLSTM":
    hyper_params = {**common["hyper_parameters"], **bilstm_config["hyper_parameters"]}
    common = {**common, **bilstm_config}
    common["hyper_parameters"] = hyper_params
    
hyper_params = {**common["hyper_parameters"], **train_config["hyper_parameters"]}
train_config = {**train_config, **common}
train_config["hyper_parameters"] = hyper_params
if train_config["hyper_parameters"]["scheduler"] == "CAWR":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **cawr_scheduler}
elif train_config["hyper_parameters"]["scheduler"] == "Step":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **step_scheduler}
    
hyper_params = {**common["hyper_parameters"], **eval_config["hyper_parameters"]}
eval_config = {**eval_config, **common}
eval_config["hyper_parameters"] = hyper_params