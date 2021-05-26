import numpy as np
import math
import sys
import os

def get_tag_fmeasure(golden_lists, predict_lists, label_type="char"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    golden_num = 0.
    predict_num = 0.
    right_num = 0.
    all_tag = 0.
    for idx in range(0,sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_num += 1
        all_tag += len(golden_list)
        golden_num += len(golden_list)
        predict_num += len(predict_list)
        golden_full.extend(golden_list)
        predict_full.extend(predict_list)

    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_num+0.0)/all_tag
    print ("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    '''
    print (precision, recall, f_measure)
    print("confusion_matrix:")
    cm = confusion_matrix(golden_full, predict_full)
    print(cm)
    
    print("recall:")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    
    print("precision:")
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    print(cm_normalized)
    '''
    return accuracy, precision, recall, f_measure



## input as sentence level labels
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print ("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    print (precision, recall, f_measure)
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)
            
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-' 
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag 
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix



def readSentence(input_file):
    with open(input_file, 'r', encoding = 'utf8') as i:
        in_lines = i.readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences,labels


def readTwoLabelSentence(input_file, pred_col=-1):
    with open(input_file, 'r', encoding = 'utf8') as i:
        in_lines = i.readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip().split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])
            
    return sentences,golden_labels,predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print ("Get f measure from file:", golden_file, predict_file)
    print ("Label format:",label_type)
    golden_sent,golden_labels = readSentence(golden_file)
    predict_sent,predict_labels = readSentence(predict_file)
    acc, P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("Acc:%s, P:%s R:%s, F:%s"%(acc, P,R,F))



def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    sent,golden_labels,predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    if label_type=="char":
        A,P,R,F = get_tag_fmeasure(golden_labels, predict_labels, label_type)
    else:
        A,P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    #print ("P:%s, R:%s, F:%s"%(P,R,F))
    return P, R, F

def combine_result(gold_path, pred_path, out_path):
    with open(out_path, 'w', encoding = 'utf8') as o:
        with open(gold_path, 'r', encoding = 'utf8') as g:
            gold_lines = g.readlines()
        with open(pred_path, 'r', encoding = 'utf8') as p:
            pred_lines = p.readlines()
        assert len(gold_lines) == len(pred_lines)
        data_num = len(gold_lines)
        for i in range(data_num):
            gold_l = gold_lines[i]
            pred_l = pred_lines[i]
            gold_content_list = gold_l.strip('\n').split('\t')
            text = gold_content_list[0]
            gold_label_str = gold_content_list[1]
            
            pred_l = pred_lines[i]
            pred_content_list = pred_l.strip('\n').split('\t')
            pred_label_str = pred_content_list[1]
            gold_label_list = gold_label_str.split()
            pred_label_list = pred_label_str.split()
            assert len(gold_label_list) == len(pred_label_list)
            
            text_list = text.split()
            instance_len = len(text_list)
            for j in range(instance_len):
                out_str = text_list[j] + ' ' + gold_label_list[j] + ' ' + pred_label_list[j]
                o.writelines(out_str + '\n')
            o.writelines('\n')

if __name__ == '__main__':
    combine_result(sys.argv[1], sys.argv[2], 'tmp')
    P, R, F = fmeasure_from_singlefile('tmp',"BMES")
    print (P, R, F)
