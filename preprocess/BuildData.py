#!/usr/bin/env python
# coding: utf-8


import json
import os
from tqdm import tqdm
import re
from transformers import BertTokenizerFast, AutoTokenizer
import copy
import torch
from common.utils import Preprocessor
import yaml
import logging
from pprint import pprint

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
config = yaml.load(open("preprocess/build_data_config.yaml", "r"), Loader = yaml.FullLoader)

exp_name = config["exp_name"]
data_in_dir = os.path.join(config["data_in_dir"], exp_name)
data_out_dir = os.path.join(config["data_out_dir"], exp_name)
if not os.path.exists(data_out_dir):
    os.makedirs(data_out_dir)


#加载数据集

if exp_name == "duie":
    file_name2data = {}
    for path, folds, files in os.walk(data_in_dir):
        for file_name in files:
            if 'train' in file_name or 'valid' in file_name or 'test' in file_name:
                file_path = os.path.join(path, file_name)
                file_name = re.match("(.*?)\.json", file_name).group(1)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = []
                    for line in f:
                        line_dict = json.loads(line)
                        data.append(line_dict)
                # 迷你数据集
                # file_name2data[file_name] = data[:200]
                file_name2data[file_name] = data
    assert file_name2data, f"没有获取到文件，请检数据的目录是否正确"
else:
    file_name2data = {}
    for path, folds, files in os.walk(data_in_dir):
        for file_name in files:
            file_path = os.path.join(path, file_name)
            file_name = re.match("(.*?)\.json", file_name).group(1)
            file_name2data[file_name] = json.load(open(file_path, "r", encoding = "utf-8"))
    assert file_name2data, f"没有获取到文件，请检数据的目录是否正确"

#处理数据


# @specific
if config["encoder"] == "BERT":
    if os.path.exists(config["bert_path"]):
        tokenizer = AutoTokenizer.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"]
elif config["encoder"] == "BiLSTM":
    tokenize = lambda text: text.split(" ")
    def get_tok2char_span_map(text):
        tokens = tokenize(text)
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1 # +1: whitespace
        return tok2char_span



preprocessor = Preprocessor(tokenize_func = tokenize, 
                            get_tok2char_span_map_func = get_tok2char_span_map)


# ## Transform

ori_format = config["ori_data_format"]
if ori_format != "tplinker": # if tplinker, skip transforming
    for file_name, data in file_name2data.items():
        if "train" in file_name:
            data_type = "train"
        if "valid" in file_name:
            data_type = "valid"
        if "test" in file_name:
            data_type = "test"
        data = preprocessor.transform_data(data, ori_format = ori_format, dataset_type = data_type, add_id = True)
        file_name2data[file_name] = data


# ## Clean and Add Spans


# check token level span
def check_tok_span(data):
    def extr_ent(text, tok_span, tok2char_span):
        char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
        char_span = (char_span_list[0][0], char_span_list[-1][1])
        decoded_ent = text[char_span[0]:char_span[1]]
        return decoded_ent

    span_error_memory = set()
    for sample in tqdm(data, desc = "检查token的spans是否合法"):
        text = sample["text"]
        tok2char_span = get_tok2char_span_map(text)
        for ent in sample["entity_list"]:
            tok_span = ent["tok_span"]
            if extr_ent(text, tok_span, tok2char_span) != ent["text"]:
                span_error_memory.add("extr ent: {}---gold ent: {}".format(extr_ent(text, tok_span, tok2char_span), ent["text"]))
                
        for rel in sample["relation_list"]:
            subj_tok_span, obj_tok_span = rel["subj_tok_span"], rel["obj_tok_span"]
            if extr_ent(text, subj_tok_span, tok2char_span) != rel["subject"]:
                span_error_memory.add("extr: {}---gold: {}".format(extr_ent(text, subj_tok_span, tok2char_span), rel["subject"]))
            if extr_ent(text, obj_tok_span, tok2char_span) != rel["object"]:
                span_error_memory.add("extr: {}---gold: {}".format(extr_ent(text, obj_tok_span, tok2char_span), rel["object"]))
                
    return span_error_memory


# clean, add char span, tok span
# collect relations
# check tok spans
rel_set = set()
ent_set = set()
error_statistics = {}
for file_name, data in file_name2data.items():
    print(f"\n开始处理数据集: {file_name}")
    assert len(data) > 0
    if "relation_list" in data[0]: # train or valid data
        # rm redundant whitespaces
        # separate by whitespaces, 清理数据
        data = preprocessor.clean_data_wo_span(data, separate = config["separate_char_by_white"], remove_white_space=config["remove_white_space"])
        error_statistics[file_name] = {}
#         if file_name != "train_data":
#             set_trace()
        # add char span
        if config["add_char_span"]:
            data, miss_sample_list = preprocessor.add_char_span(data, config["ignore_subword"])
            error_statistics[file_name]["miss_samples"] = len(miss_sample_list)
            
#         # clean
#         data, bad_samples_w_char_span_error = preprocessor.clean_data_w_span(data)
#         error_statistics[file_name]["char_span_error"] = len(bad_samples_w_char_span_error)
                            
        # collect relation types and entity types
        for sample in tqdm(data, desc = "构建关系类型和实体类型的集合"):
            if "entity_list" not in sample: # if "entity_list" not in sample, generate entity list with default type
                ent_list = []
                for rel in sample["relation_list"]:
                    ent_list.append({
                        "text": rel["subject"],
                        "type": "DEFAULT",
                        "char_span": rel["subj_char_span"],
                    })
                    ent_list.append({
                        "text": rel["object"],
                        "type": "DEFAULT",
                        "char_span": rel["obj_char_span"],
                    })
                sample["entity_list"] = ent_list
            
            for ent in sample["entity_list"]:
                ent_set.add(ent["type"])
                
            for rel in sample["relation_list"]:
                rel_set.add(rel["predicate"])
               
        # add tok span
        data = preprocessor.add_tok_span(data)

        # check tok span
        if config["check_tok_span"]:
            span_error_memory = check_tok_span(data)
            if len(span_error_memory) > 0:
                print(span_error_memory)
            error_statistics[file_name]["tok_span_error"] = len(span_error_memory)
            
        file_name2data[file_name] = data
print(f"处理过程中的错误信息统计结果如下：")
pprint(error_statistics)


# # Output to Disk

rel_set = sorted(rel_set)
rel2id = {rel:ind for ind, rel in enumerate(rel_set)}

ent_set = sorted(ent_set)
ent2id = {ent:ind for ind, ent in enumerate(ent_set)}

data_statistics = {
    "relation_type_num": len(rel2id),
    "entity_type_num": len(ent2id),
}

for file_name, data in file_name2data.items():
    data_path = os.path.join(data_out_dir, "{}.json".format(file_name))
    json.dump(data, open(data_path, "w", encoding = "utf-8"), ensure_ascii = False)
    logging.info("{} is output to {}".format(file_name, data_path))
    data_statistics[file_name] = len(data)

rel2id_path = os.path.join(data_out_dir, "rel2id.json")
json.dump(rel2id, open(rel2id_path, "w", encoding = "utf-8"), ensure_ascii = False)
logging.info("rel2id is output to {}".format(rel2id_path))

ent2id_path = os.path.join(data_out_dir, "ent2id.json")
json.dump(ent2id, open(ent2id_path, "w", encoding = "utf-8"), ensure_ascii = False)
logging.info("ent2id is output to {}".format(ent2id_path))


data_statistics_path = os.path.join(data_out_dir, "data_statistics.txt")
json.dump(data_statistics, open(data_statistics_path, "w", encoding = "utf-8"), ensure_ascii = False, indent = 4)
logging.info("数据的统计状态保存到  {}".format(data_statistics_path))

print(f"数据集的统计信息结果如下：")
pprint(data_statistics)


# # Genrate WordDict



if config["encoder"] in {"BiLSTM", }:
    all_data = []
    for data in list(file_name2data.values()):
        all_data.extend(data)
        
    token2num = {}
    for sample in tqdm(all_data, desc = "Tokenizing"):
        text = sample['text']
        for tok in tokenize(text):
            token2num[tok] = token2num.get(tok, 0) + 1
    
    token2num = dict(sorted(token2num.items(), key = lambda x: x[1], reverse = True))
    max_token_num = 50000
    token_set = set()
    for tok, num in tqdm(token2num.items(), desc = "Filter uncommon words"):
        if num < 3: # filter words with a frequency of less than 3
            continue
        token_set.add(tok)
        if len(token_set) == max_token_num:
            break
        
    token2idx = {tok:idx + 2 for idx, tok in enumerate(sorted(token_set))}
    token2idx["<PAD>"] = 0
    token2idx["<UNK>"] = 1
#     idx2token = {idx:tok for tok, idx in token2idx.items()}
    
    dict_path = os.path.join(data_out_dir, "token2idx.json")
    json.dump(token2idx, open(dict_path, "w", encoding = "utf-8"), ensure_ascii = False, indent = 4)
    logging.info("token2idx is output to {}, total token num: {}".format(dict_path, len(token2idx))) 

