# Author: yxsu
import sys
import torch
import argparse
sys.path.append('..')
from bert import BERTLM
from data import SEP, MASK
from main import myModel
import numpy as np

import time
mstime = lambda: int(round(time.time() * 1000))    

def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path, map_location='cpu')
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    return bert_args, model_args, bert_vocab, model_parameters

def init_empty_bert_model(bert_args, bert_vocab, gpu_id):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, bert_args.approx)
    return bert_model

def init_sequence_tagging_model(empty_bert_model, args, bert_args, gpu_id, bert_vocab, model_parameters):
    number_class = args.number_class
    embedding_size = bert_args.embed_dim
    batch_size = args.batch_size
    dropout = args.dropout
    device = gpu_id
    vocab = bert_vocab
    loss_type = args.loss_type
    seq_tagging_model = myModel(empty_bert_model, number_class, embedding_size, batch_size, dropout, 
        device, vocab, loss_type)
    seq_tagging_model.load_state_dict(model_parameters)
    return seq_tagging_model

def get_tag_mask_matrix(batch_text_list):
    tag_matrix = []
    mask_matrix = []
    batch_size = len(batch_text_list)
    max_len = 0
    for instance in batch_text_list:
        max_len = max(len(instance), max_len)
    max_len += 1 # 1 for [CLS]
    for i in range(batch_size):
        one_text_list = batch_text_list[i]
        one_tag = list(np.zeros(max_len).astype(int))
        tag_matrix.append(one_tag)
        one_mask = [1]
        one_valid_len = len(batch_text_list[i])
        for j in range(one_valid_len):
            one_mask.append(1)
        len_diff = max_len - len(one_mask)
        for _ in range(len_diff):
            one_mask.append(0)
        mask_matrix.append(one_mask)
        assert len(one_mask) == len(one_tag)
    return np.array(tag_matrix), np.array(mask_matrix)

def join_str(in_list):
    out_str = ''
    for token in in_list:
        out_str += str(token) + ''
    return out_str.strip()

def predict_one_text_split(text_split_list, seq_tagging_model, label_dict):
    # text_split_list is a list of tokens ['word1', 'word2', ...]
    text_list = [text_split_list]
    tag_matrix, mask_matrix = get_tag_mask_matrix(text_list)


    decode_result, _, _, _ = seq_tagging_model(text_list, mask_matrix, tag_matrix, fine_tune = False)
    valid_text_len = len(text_split_list)
    
    valid_decode_result = decode_result[0][1: valid_text_len + 1]

    tag_result = []
    for token in valid_decode_result:
        tag_result.append(label_dict[int(token)])
    return tag_result
    #return valid_decode_result

def get_text_split_list(text, max_len):
    result_list = []
    text_list = [w for w in text] + [SEP]
    valid_len = len(text_list)
    split_num = (len(text_list) // max_len) + 1
    if split_num == 1:
        result_list = [text_list]
    else:
        b_idx = 0
        e_idx = 1
        for i in range(max_len):
            b_idx = i * max_len
            e_idx = (i + 1) * max_len
            result_list.append(text_list[b_idx:e_idx])
        if e_idx < valid_len:
            result_list.append(text_list[e_idx:])
        else:
            pass
    return result_list

def predict_one_text(text, max_len, seq_tagging_model, label_dict):
    text_split_list = get_text_split_list(text, max_len)
    all_text_result = []
    all_decode_result = []
    for one_text_list in text_split_list:
        one_decode_result = predict_one_text_split(one_text_list, seq_tagging_model, label_dict)
        all_text_result.extend(one_text_list)
        all_decode_result.extend(one_decode_result)
    result_text = join_str(all_text_result)
    tag_predict_result = join_str(all_decode_result)
    return tag_predict_result

def get_label_dict(label_path):
    label_dict = {}
    with open(label_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split()
            label_id = int(content_list[1])
            label = content_list[0]
            label_dict[label_id] = label
    return label_dict

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--out_path',type=str)
    parser.add_argument('--gpu_id',type=int, default=0)
    parser.add_argument('--max_len',type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    ckpt_path = args.ckpt_path
    test_data = args.test_data
    out_path = args.out_path
    gpu_id = args.gpu_id
    max_len = args.max_len
   
    print("loading..")
    bert_args, model_args, bert_vocab, model_parameters = extract_parameters(ckpt_path)
    
    label_dict = {}
    for lid, label in enumerate(bert_vocab._idx2token):
        label_dict[lid] = label
    
    model_args.number_class = len(label_dict)

    empty_bert_model = init_empty_bert_model(bert_args, bert_vocab, gpu_id)
    seq_tagging_model = init_sequence_tagging_model(empty_bert_model, model_args, 
        bert_args, gpu_id, bert_vocab, model_parameters)
    seq_tagging_model.cuda(gpu_id)

    print("eval...")
    seq_tagging_model.eval()
    with torch.no_grad():
        with open(out_path, 'w', encoding = 'utf8') as o:
            with open(test_data, 'r', encoding = 'utf8') as i:
                start = mstime()
                lines = i.readlines()
                for l in lines:
                    content_list = l.strip().split('\t')
                    text = content_list[0]
                    gold = content_list[1]
                    res = predict_one_text(text, max_len, seq_tagging_model, label_dict)
                    res = res.replace(SEP, '').strip()
                    o.writelines(text + "\t" + res + "\t" + gold + "\n")

                print(mstime()-start)
