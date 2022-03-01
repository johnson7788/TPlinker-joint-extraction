import re
from tqdm import tqdm
import copy
import json
import os

class DefaultLogger:
    def __init__(self, log_path, project, run_name, run_id, hyperparameter):
        self.log_path = log_path
        log_dir = "/".join(self.log_path.split("/")[:-1])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.run_id = run_id
        self.log(f"日志保存路径: {log_path}")
        self.log("============================================================================")
        self.log("项目: {}, run_name: {}, run_id: {}\n".format(project, run_name, run_id))
        hyperparameters_format = "--------------超参数------------------- \n{}\n-----------------------------------------"
        self.log(hyperparameters_format.format(json.dumps(hyperparameter, indent = 4)))

    def log(self, text):
        text = "run_id: {}, {}".format(self.run_id, text)
        print(text)
        open(self.log_path, "a", encoding = "utf-8").write("{}\n".format(text))
        
class Preprocessor:
    '''
    1. 将数据集转换为正常格式，以适应我们的代码
    2. 在所有关系中添加token span 头到所有实体， 将被用于标注阶段
    '''
    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func
    
    def transform_data(self, data, ori_format, dataset_type, add_id = True):
        '''
        This function can only deal with three original format used in the previous works. 
        If you want to feed new dataset to the model, just define your own function to transform data.
        data: original data
        ori_format: "casrel", "joint_re", "raw_nyt"
        dataset_type: "train", "valid", "test"; only for generate id for the data
        '''
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc = "Transforming data format"):
            if ori_format == "casrel":
                text = sample["text"]
                rel_list = sample["triple_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "etl_span":
                text = " ".join(sample["tokens"])
                rel_list = sample["spo_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "duie_format":
                text = sample["text"]
                rel_list = sample.get("spo_list", [])
                subj_key, pred_key, obj_key = "subject", "predicate", "object"
            elif ori_format == "raw_nyt":
                text = sample["sentText"]
                rel_list = sample["relationMentions"]
                subj_key, pred_key, obj_key = "em1Text", "label", "em2Text"

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            normal_rel_list = []
            for rel in rel_list:
                if isinstance(rel[obj_key], dict):
                    # duie数据集
                    object = rel[obj_key]['@value']
                else:
                    object = rel[obj_key]
                subject = rel[subj_key]
                if not object or not subject:
                    # 跳过空数据
                    continue
                normal_rel = {
                    "subject": subject,
                    "predicate": rel[pred_key],
                    "object": object,
                }
                normal_rel_list.append(normal_rel)
            normal_sample["relation_list"] = normal_rel_list
            # 如果'object_type'和'subject_type'存在，那么实体的类别也加进去
            if rel_list and 'object_type' in rel_list[0]:
                # 只有duie的数据集
                entity_list = []
                #实体的类别信息
                for rel in rel_list:
                    obj_text = rel['object']['@value']
                    obj_type = rel['object_type']['@value']
                    sub_text = rel['subject']
                    sub_type = rel['subject_type']
                    if not obj_text or not sub_text:
                        # 跳过空数据
                        continue
                    entity_list.append({"text": obj_text, "type": obj_type})
                    entity_list.append({"text": sub_text, "type": sub_type})
                normal_sample["entity_list"] = entity_list
            normal_sample_list.append(normal_sample)
            
        return self._clean_sp_char(normal_sample_list)
    
    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len = 50, encoder = "BERT", data_type = "train"):
        """
        处理数据，长的句子变成短的句子
        :param sample_list: 数据集, 一条数据的示例
        00000 = {dict: 4} {'text': 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .', 'id': 'train_0', 'relation_list': [{'subject': 'Annandale-on-Hudson', 'object': 'College', 'subj_char_span': [68, 87], 'obj_char_span'
 'text' = {str} 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .'
 'id' = {str} 'train_0'
 'relation_list' = {list: 1} [{'subject': 'Annandale-on-Hudson', 'object': 'College', 'subj_char_span': [68, 87], 'obj_char_span': [58, 65], 'predicate': '/location/location/contains', 'subj_tok_span': [17, 24], 'obj_tok_span': [15, 16]}]
 'entity_list' = {list: 2} [{'text': 'Annandale-on-Hudson', 'type': 'DEFAULT', 'char_span': [68, 87], 'tok_span': [17, 24]}, {'text': 'College', 'type': 'DEFAULT', 'char_span': [58, 65], 'tok_span': [15, 16]}]
        :type sample_list: list
        :param max_seq_len: 100  拆分后的最大序列长度
        :type max_seq_len:
        :param sliding_len: 滑动的字符的大小， eg: 20
        :type sliding_len:
        :param encoder: eg: 'BERT'
        :type encoder:str
        :param data_type: eg: 'train'
        :type data_type:str
        :return:
        :rtype:
        """
        new_sample_list = []
        for sample in tqdm(sample_list, desc = "开始拆分成小短句"):
            text_id = sample["id"]   #eg: 'train_0'
            text = sample["text"]    # eg: 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .'
            tokens = self._tokenize(text)   # 变成token, eg: ['Massachusetts', 'AS', '##TO', '##N', 'MA', '##G', '##NA', 'Great', 'Barr', '##ington', ';', 'also', 'at', 'Bar', '##d', 'College', ',', 'Anna', '##nda', '##le', '-', 'on', '-', 'Hudson', ',', 'N', '.', 'Y', '.', ',', 'July', '1', '-', 'Aug', '.']
            tok2char_span = self._get_tok2char_span_map(text)  # 获取每个token的长度 # eg: [(0, 13), (14, 16), (16, 18), (18, 19), (20, 22), (22, 23), (23, 25), (26, 31), (32, 36), (36, 42), (43, 44), (45, 49), (50, 52), (53, 56), (56, 57), (58, 65), (66, 67), (68, 72), (72, 75), (75, 77), (77, 78), (78, 80), (80, 81), (81, 87), (88, 89), (90, 91), (91, 92), (92, 93), (93, 94), (95, 96), (97, 101), (102, 103), (103, 104), (104, 107), (108, 109)]

            # token级别滑动
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT": # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len
                # 获取部分长度对应的token的长度 [(0, 13), (14, 16), (16, 18), (18, 19), (20, 22), (22, 23), (23, 25), (26, 31), (32, 36), (36, 42), (43, 44), (45, 49), (50, 52), (53, 56), (56, 57), (58, 65), (66, 67), (68, 72), (72, 75), (75, 77), (77, 78), (78, 80), (80, 81), (81, 87), (88, 89), (90, 91), (91, 92), (92, 93), (93, 94), (95, 96), (97, 101), (102, 103), (103, 104), (104, 107), (108, 109)]
                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]   # eg: [0, 109]， 总共的字符的长度
                sub_text = text[char_level_span[0]:char_level_span[1]]  # 原始文本进行对应裁剪, 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .'

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "tok_offset": start_ind,
                    "char_offset": char_level_span[0],
                    }
                if data_type == "test": # test set
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else: # 训练集合验证集的数据，要把spo信息加入
                    # spo， 对应关系
                    sub_rel_list = []
                    for rel in sample["relation_list"]:
                        subj_tok_span = rel["subj_tok_span"]   #eg: [17, 24]
                        obj_tok_span = rel["obj_tok_span"]   #eg: [15, 16]
                        # 如果主语和宾语都在这个子句中，就把这spo添加到新的样本中。
                        if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                            and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind:
                            # 拷贝这条关系数据，然后修改对应的起始和结束位置
                            new_rel = copy.deepcopy(rel)
                            new_rel["subj_tok_span"] = [subj_tok_span[0] - start_ind, subj_tok_span[1] - start_ind] # start_ind: tok level offset
                            new_rel["obj_tok_span"] = [obj_tok_span[0] - start_ind, obj_tok_span[1] - start_ind]
                            new_rel["subj_char_span"][0] -= char_level_span[0] # char level offset
                            new_rel["subj_char_span"][1] -= char_level_span[0]
                            new_rel["obj_char_span"][0] -= char_level_span[0]
                            new_rel["obj_char_span"][1] -= char_level_span[0]
                            sub_rel_list.append(new_rel)
                    
                    # entity，对应实体
                    sub_ent_list = []
                    for ent in sample["entity_list"]:
                        tok_span = ent["tok_span"]
                        # 如果这个实体在这个子句中，那么把实体加入这个样本
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind: 
                            new_ent = copy.deepcopy(ent)
                            new_ent["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]
                            
                            new_ent["char_span"][0] -= char_level_span[0]
                            new_ent["char_span"][1] -= char_level_span[0]

                            sub_ent_list.append(new_ent)
                    
                    # 事件
                    if "event_list" in sample:
                        sub_event_list = []
                        for event in sample["event_list"]:
                            trigger_tok_span = event["trigger_tok_span"]
                            if trigger_tok_span[1] > end_ind or trigger_tok_span[0] < start_ind:
                                continue
                            new_event = copy.deepcopy(event)
                            new_arg_list = []
                            for arg in new_event["argument_list"]:
                                if arg["tok_span"][0] >= start_ind and arg["tok_span"][1] <= end_ind:
                                    new_arg_list.append(arg)
                            new_event["argument_list"] = new_arg_list
                            sub_event_list.append(new_event)
                        new_sample["event_list"] = sub_event_list # maybe empty
                        
                    new_sample["entity_list"] = sub_ent_list # maybe empty
                    new_sample["relation_list"] = sub_rel_list # maybe empty
                    split_sample_list.append(new_sample)
                
                # all segments covered, no need to continue
                if end_ind > len(tokens):
                    break
                    
            new_sample_list.extend(split_sample_list)
        return new_sample_list
    
    def _clean_sp_char(self, dataset):
        def clean_text(text):
            text = re.sub("�", "", text)
#             text = re.sub("([A-Za-z]+)", r" \1 ", text)
#             text = re.sub("(\d+)", r" \1 ", text)
#             text = re.sub("\s+", " ", text).strip()
            return text 
        for sample in tqdm(dataset, desc = "Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return dataset
        
    def clean_data_wo_span(self, ori_data, separate = False, remove_white_space=False, data_type = "train"):
        '''
        删除多余空格
        and add whitespaces around tokens to keep special characters from them
        '''
        def clean_text(text):
            text = re.sub("\s+", " ", text).strip()
            if separate:
                text = re.sub("([^A-Za-z0-9])", r" \1 ", text)
                text = re.sub("\s+", " ", text).strip()
            if remove_white_space:
                text = re.sub('[ ]+', '', text)
            return text

        for sample in tqdm(ori_data, desc = "clean data"):
            sample["text"] = clean_text(sample["text"])
            if data_type == "test":
                continue
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return ori_data

    def clean_data_w_span(self, ori_data):
        '''
        stripe whitespaces and change spans
        add a stake to bad samples(char span error) and remove them from the clean data
        '''
        bad_samples, clean_data = [], []
        def strip_white(entity, entity_char_span):
            p = 0
            while entity[p] == " ":
                entity_char_span[0] += 1
                p += 1

            p = len(entity) - 1
            while entity[p] == " ":
                entity_char_span[1] -= 1
                p -= 1
            return entity.strip(), entity_char_span
            
        for sample in tqdm(ori_data, desc = "clean data w char spans"):
            text = sample["text"]

            bad = False
            for rel in sample["relation_list"]:
                # rm whitespaces
                rel["subject"], rel["subj_char_span"] = strip_white(rel["subject"], rel["subj_char_span"])
                rel["object"], rel["obj_char_span"] = strip_white(rel["object"], rel["obj_char_span"])

                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                if rel["subject"] not in text or rel["subject"] != text[subj_char_span[0]:subj_char_span[1]] or \
                    rel["object"] not in text or rel["object"] != text[obj_char_span[0]:obj_char_span[1]]:
                    rel["stake"] = 0
                    bad = True
                    
            if bad:
                bad_samples.append(copy.deepcopy(sample))

            new_rel_list = [rel for rel in sample["relation_list"] if "stake" not in rel]
            if len(new_rel_list) > 0:
                sample["relation_list"] = new_rel_list
                clean_data.append(sample)
        return clean_data, bad_samples
    
    def _get_char2tok_span(self, text):
        '''
        把char的span转换到token的span
        '''
        # eg: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22)]
        tok2char_span = self._get_tok2char_span_map(text)
        char_num = None  # 总的字符数量
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1] for _ in range(char_num)] # [-1, -1] is whitespace
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1
        # if [-1, -1] in char2tok_span:
        #     print(f"注意：发现-1，-1在字符到token的位置映射中，请检查: {text}, 空格和特殊字符都可能造成【-1，-1】的索引， 因为tokenizer对空格和特殊字符和中文的某些标点符号，返回的是不计数的")
        return char2tok_span

    def _get_ent2char_spans(self, text, entities, ignore_subword_match = True):
        '''
        如果ignore_subword_match为true，则查找周围有空格的实体。e.g. "entity" -> " entity "
        '''
        entities = sorted(entities, key = lambda x: len(x), reverse = True)
        text_cp = " {} ".format(text) if ignore_subword_match else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword_match else ent
            for m in re.finditer(re.escape(target_ent), text_cp):
                if not ignore_subword_match and re.match("\d+", target_ent): # avoid matching a inner number of a number
                    if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or (m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                        continue
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
                spans.append(span)
                # 如果没有找到，那么span为[-1,-1]
#             if len(spans) == 0:
#                 set_trace()
            ent2char_spans[ent] = spans
        return ent2char_spans
    
    def add_char_span(self, dataset, ignore_subword_match = True):
        """

        :param dataset:
        :type dataset:
        :param ignore_subword_match:
        :type ignore_subword_match:
        :return:
        :rtype:
        """
        miss_sample_list = []
        for sample in tqdm(dataset, desc = "给数据集添加字符的spans"):
            # 获取所有的实体保存到entities，包括头实体或尾实体
            entities = [rel["subject"] for rel in sample["relation_list"]]
            entities.extend([rel["object"] for rel in sample["relation_list"]])
            if "entity_list" in sample:
                entities.extend([ent["text"] for ent in sample["entity_list"]])
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities, ignore_subword_match = ignore_subword_match)
            
            new_relation_list = []
            for rel in sample["relation_list"]:
                subj_char_spans = ent2char_spans[rel["subject"]]
                obj_char_spans = ent2char_spans[rel["object"]]
                for subj_sp in subj_char_spans:
                    for obj_sp in obj_char_spans:
                        new_relation_list.append({
                            "subject": rel["subject"],
                            "object": rel["object"],
                            "subj_char_span": subj_sp,
                            "obj_char_span": obj_sp,
                            "predicate": rel["predicate"],
                        })
            
            if len(sample["relation_list"]) > len(new_relation_list):
                miss_sample_list.append(sample)
            sample["relation_list"] = new_relation_list
            
            if "entity_list" in sample:
                new_ent_list = []
                for ent in sample["entity_list"]:
                    for char_sp in ent2char_spans[ent["text"]]:
                        new_ent_list.append({
                            "text": ent["text"],
                            "type": ent["type"],
                            "char_span": char_sp,
                        })
                sample["entity_list"] = new_ent_list
        return dataset, miss_sample_list
           
    def add_tok_span(self, dataset):
        '''
        添加token的span, 需要把字符对应的span位置转换成token对应的span的位置
        '''      
        def char_span2tok_span(char_span, char2tok_span):
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span
        
        for sample in tqdm(dataset, desc = "添加token的spans"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for rel in sample["relation_list"]:
                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                rel["subj_tok_span"] = char_span2tok_span(subj_char_span, char2tok_span)
                rel["obj_tok_span"] = char_span2tok_span(obj_char_span, char2tok_span)
            for ent in sample["entity_list"]:
                char_span = ent["char_span"]
                ent["tok_span"] = char_span2tok_span(char_span, char2tok_span)
            if "event_list" in sample:
                for event in sample["event_list"]:
                    event["trigger_tok_span"] = char_span2tok_span(event["trigger_char_span"], char2tok_span)
                    for arg in event["argument_list"]:
                        arg["tok_span"] = char_span2tok_span(arg["char_span"], char2tok_span)
        return dataset