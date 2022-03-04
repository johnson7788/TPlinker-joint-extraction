#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/3/4 11:46 上午
# @File  : compare_result.py
# @Author: johnson
# @Desc  : 比较下原始测试集和真实的预测结果之间的差异
import json

def do_compare(src_file= "data4bert/shenji/test_data.json", predict_result="results/shenji/dHO8Egzw/test_data_res_13.json"):
    """

    :param src_file:
    :type src_file:
    :param predict_result:
    :type predict_result:
    :return:
    :rtype:
    """
    with open(src_file) as f:
        src_data = json.load(f)
    with open(predict_result) as f:
        predict_data = []
        for line in f:
            line_data = json.loads(line)
            predict_data.append(line_data)
    src_data_dict = {i["id"]:i for i in src_data}
    predict_data_dict = {i["id"]:i for i in predict_data}
    for id, predict_value in predict_data_dict.items():
        print("-"* 50)
        src_value = src_data_dict[id]
        print(f"预测的样本的id: {id}")
        print(f"要预测的内容是:")
        print(predict_value['text'])
        print(f"原始的实体是: ")
        for ent in src_value['entity_list']:
            print(f"{ent['text']}\t    {ent['type']}\t")
        print(f"预测的实体是: ")
        for ent in predict_value['entity_list']:
            print(f"{ent['text']}\t    {ent['type']}\t")

        print("-"*20)
        print(f"原始的关系是: ")
        for rel in src_value['relation_list']:
            print(f"{rel['subject']}\t    {rel['object']}\t     {rel['predicate']}")
        print(f"预测的关系是: ")
        for rel in predict_value['relation_list']:
            print(f"{rel['subject']}\t    {rel['object']}\t     {rel['predicate']}")





if __name__ == '__main__':
    do_compare()