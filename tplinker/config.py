import string
import random

common = {
    "exp_name": "nyt",
    "rel2id": "rel2id.json",
    "device_num": 0,
#     "encoder": "BiLSTM",
    "encoder": "BERT", 
    "hyper_parameters": {
        "shaking_type": "cat", # cat, cat_plus, cln, cln_plus; 实验表明，cat/cat_plus与BiLSTM的效果更好, 而cln/cln_plus与BERT的工作效果更好。论文中的结果是由 "cat "产生的。因此，如果你想重现这些结果，"cat "就足够了，不管是对BERT还是BiLSTM。
        "inner_enc_type": "lstm", # 只有当cat_plus或cln_plus被设置时才有效。这是如何对每个token对之间的内部token进行编码的方法。如果你只想重现结果，就不要管它了。
        "dist_emb_size": -1, # -1表示不使用距离嵌入；其他数字：需要大于输入的max_seq_len。如果你只想复制论文中的结果，则设置-1。
        "ent_add_dist": False, # 如果你想为每个token对添加距离嵌入，则设置为true。(用于实体解码器)
        "rel_add_dist": False, # 与上述相同（用于关系解码器）
        "match_pattern": "whole_text", # only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span
    },
}
common["run_name"] = "{}+{}+{}".format("TP1", common["hyper_parameters"]["shaking_type"], common["encoder"]) + ""

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
        "batch_size": 6,
        "epochs": 100,
        "seed": 2333,
        "log_interval": 10,
        "max_seq_len": 100,  #最大序列长度
        "sliding_len": 20,
        "loss_weight_recover_steps": 6000, # 为了加快训练速度, EH-to-ET 的损失比重是比其它在开始的时候要高， 但是在训练了loss_weight_recover_steps个step之后，损失的比重变正常
        "scheduler": "CAWR", # 学习率计划
    },
}

eval_config = {
    "model_state_dict_dir": "./wandb", # if use wandb, set "./wandb", or set "./default_log_dir" if you use default logger
    "run_ids": ["10suiyrf", ],
    "last_k_model": 1,
    "test_data": "*test*.json", # "*test*.json"
    
    # where to save results
    "save_res": False,
    "save_res_dir": "../results",
    
    # score: set true only if test set is annotated with ground truth
    "score": True,
    
    "hyper_parameters": {
        "batch_size": 32,
        "force_split": False,
        "max_test_seq_len": 512,
        "sliding_len": 50,
    },
}

bert_config = {
    "data_home": "data4bert",
    "bert_path": "pretrained_models/bert-base-cased",
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