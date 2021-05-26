import sys
import torch
from torch import nn
import torch.nn.functional as F
import random
#### Load pretrained bert model
from bert import BERTLM
from google_bert import BasicTokenizer
from data import Vocab, CLS, SEP, MASK
import numpy as np
from data_loader import DataLoader
from crf_layer import DynamicCRF
import os
from funcs import *


def init_bert_model(args, device, bert_vocab):
    bert_ckpt= torch.load(args.bert_path)
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
        bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    bert_model = bert_model.cuda(device)
    if args.freeze == 1:
        for p in bert_model.parameters():
            p.requires_grad=False
    return bert_model, bert_vocab, bert_args

def ListsToTensor(xs, vocab):
    batch_size = len(xs)
    lens = [ len(x)+2 for x in xs]
    mx_len = max(lens)
    ys = []
    for i, x in enumerate(xs):
        y =  vocab.token2idx([CLS]+x) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    data = torch.LongTensor(ys).t_().contiguous()
    return data

def batchify(data, vocab):
    return ListsToTensor(data, vocab)

class myModel(nn.Module):
    def __init__(self, bert_model, num_class, embedding_size, batch_size, dropout, device, vocab, loss_type='FC_FT_CRF'):
        super(myModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.vocab = vocab
        self.fc = nn.Linear(self.embedding_size, self.num_class)
        self.CRF_layer = DynamicCRF(num_class)
        self.loss_type = loss_type
        self.bert_vocab = vocab
    
    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost) 

    def fc_nll_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
        if gamma is None:
            gamma = 2
        p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1))
        g = (1-torch.clamp(p, min=0.01, max=0.99))**gamma
        #g = (1 - p) ** gamma 
        cost = -g * torch.log(p+1e-8)
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost), g.view(y.shape)

    def forward(self, text_data, in_mask_matrix, in_tag_matrix, fine_tune=False, gamma=None):
        current_batch_size = len(text_data)
        max_len = 0
        for instance in text_data:
            max_len = max(len(instance), max_len)
        seq_len = max_len + 1 # 1 for [CLS]]

        # in_mask_matrix.size() == [batch_size, seq_len]
        # in_tag_matrix.size() == [batch_size, seq_len]
        mask_matrix = torch.tensor(in_mask_matrix, dtype=torch.uint8).t_().contiguous().cuda(self.device)
        tag_matrix = torch.LongTensor(in_tag_matrix).t_().contiguous().cuda(self.device) # size = [seq_len, batch_size]
        assert mask_matrix.size() == tag_matrix.size()
        assert mask_matrix.size() == torch.Size([seq_len, current_batch_size])

        # input text_data.size() = [batch_size, seq_len]
        data = batchify(text_data, self.vocab) # data.size() == [seq_len, batch_size]
        data = data.cuda(self.device)
         
        sequence_representation = self.bert_model.work(data)[0].cuda(self.device) # [seq_len, batch_size, embedding_size]
        # dropout
        sequence_representation = F.dropout(sequence_representation, p=self.dropout, training=self.training)
        sequence_representation = sequence_representation.view(current_batch_size * seq_len, self.embedding_size)
        sequence_emissions = self.fc(sequence_representation)
        sequence_emissions = sequence_emissions.view(seq_len, current_batch_size, self.num_class)
        
        # bert finetune loss
        probs = torch.softmax(sequence_emissions, -1)
        if "FC" in self.loss_type:
            loss_ft_fc, g = self.fc_nll_loss(probs, tag_matrix, mask_matrix, gamma=gamma)
        else:
            loss_ft = self.nll_loss(probs, tag_matrix, mask_matrix)

        sequence_emissions = sequence_emissions.transpose(0, 1)
        tag_matrix = tag_matrix.transpose(0, 1) 
        mask_matrix = mask_matrix.transpose(0, 1)

        if "FC" in self.loss_type:
            #loss_crf_fc = -self.CRF_layer(sequence_emissions, tag_matrix, mask = mask_matrix, reduction='token_mean', g=g.transpose(0, 1), gamma=gamma)
            loss_crf_fc = -self.CRF_layer(sequence_emissions, tag_matrix, mask = mask_matrix, reduction='token_mean', g=None, gamma=gamma)
        else:
            loss_crf = -self.CRF_layer(sequence_emissions, tag_matrix, mask = mask_matrix, reduction='token_mean')

        decode_result = self.CRF_layer.decode(sequence_emissions, mask = mask_matrix)
        self.decode_scores, self.decode_result = decode_result
        self.decode_result = self.decode_result.tolist()
        
        if self.loss_type == 'CRF':
            loss = loss_crf
            return self.decode_result, loss, loss_crf.item(), 0.0
        elif self.loss_type == 'FT_CRF':
            loss = loss_ft + loss_crf
            return self.decode_result, loss, loss_crf.item(), loss_ft.item()
        elif self.loss_type == 'FC_FT_CRF':
            loss = loss_ft_fc + loss_crf_fc
            return self.decode_result, loss, loss_crf_fc.item(), loss_ft_fc.item()
        elif self.loss_type == 'FC_CRF':
            loss = loss_crf_fc
            return self.decode_result, loss, loss_crf_fc.item(), 0.0
        else:
            print("error")
            return self.decode_result, 0, 0, 0

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--train_data',type=str)
    parser.add_argument('--dev_data',type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--label_data',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--dropout',type=float)
    parser.add_argument('--freeze',type=int)
    parser.add_argument('--number_class', type = int)
    parser.add_argument('--number_epoch', type = int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--bert_vocab', type=str)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--prediction_max_len', type=int)
    parser.add_argument('--dev_eval_path', type=str)
    parser.add_argument('--final_eval_path', type=str)
    parser.add_argument('--l2_lambda', type=float)
    parser.add_argument('--training_max_len', type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()

    # --- create model save path --- #
    directory = args.model_save_path
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 

    # myModel construction
    print ('Initializing model...')
    bert_model, bert_vocab, bert_args = init_bert_model(args, args.gpu_id, args.bert_vocab)
    
    id_label_dict = {}#get_id_label_dict(bert_vocab)
    for lid, label in enumerate(bert_vocab._idx2token):
        id_label_dict[lid] = label
    
    batch_size = args.batch_size
    number_class = len(id_label_dict) #args.number_class
    print(number_class)
    embedding_size = bert_args.embed_dim
    fine_tune = args.fine_tune
    loss_type = args.loss_type
    l2_lambda = args.l2_lambda
    model = myModel(bert_model, number_class, embedding_size, batch_size, args.dropout, args.gpu_id, bert_vocab, loss_type)
    model = model.cuda(args.gpu_id)
    print ('Model construction finished.')

    # Data Preparation
    train_path, dev_path, test_path = args.train_data, args.dev_data, args.test_data
    #label_path = args.label_data
    train_max_len = args.training_max_len
    nerdata = DataLoader(train_path, dev_path, test_path, bert_vocab, train_max_len)
    print ('data is ready')


    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    #--- training part ---#
    num_epochs = args.number_epoch
    training_data_num, dev_data_num, test_data_num = nerdata.train_num, nerdata.dev_num, nerdata.test_num
    train_step_num = int(training_data_num / batch_size) + 1
    dev_step_num = dev_data_num
    test_step_num = test_data_num # batch_size = 1 来进行predict
    max_dev_acc = 0.0
    max_dev_f1 = 0.0

    train_f1_list, train_precision_list, train_recall_list = [], [], []
    dev_f1_list, dev_precision_list, dev_recall_list = [], [], []

    prediction_max_len = args.prediction_max_len # 用来分块截取prediction的
    dev_eval_path = args.dev_eval_path
    final_eval_path = args.final_eval_path

    acc_bs = 0.
    for epoch in range(num_epochs):
        loss_accumulated = 0.
        loss_crf_accumulated = 0.
        loss_ft_accumulated = 0.

        model.train()
        print ('-------------------------------------------')
        if epoch % 5 == 0:
            print ('%d epochs have run' % epoch)
        else:
            pass
        total_train_pred = list()
        total_train_true = list()
        batches_processed = 0
        for train_step in range(train_step_num):
            batches_processed += 1
            acc_bs += 1
            optimizer.zero_grad()

            train_batch_text_list, train_batch_tag_list = nerdata.get_next_batch(batch_size, mode = 'train')
            # tag target matrix
            train_tag_matrix = process_batch_tag(train_batch_tag_list, nerdata.label_dict)
            # tag mask matrix
            train_mask_matrix = make_mask(train_batch_tag_list)
            # forward computation
            train_batch_result, train_loss, loss_crf, loss_ft = \
            model(train_batch_text_list, train_mask_matrix, train_tag_matrix, fine_tune, args.gamma)

            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            train_loss = train_loss + l2_lambda * l2_reg
            
            # update
            loss_accumulated += train_loss.item()
            loss_crf_accumulated  += loss_crf
            loss_ft_accumulated  += loss_ft
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            valid_train_batch_result = get_valid_predictions(train_batch_result, train_batch_tag_list, nerdata.label_dict)
            for i in range(batch_size):
                assert len(list(valid_train_batch_result[i])) == len(list(train_batch_tag_list[i]))
                total_train_pred.extend(list(valid_train_batch_result[i]))
                total_train_true.extend(list(train_batch_tag_list[i]))

            if acc_bs % args.print_every == 0:
                print ("gBatch %d, lBatch %d, loss %.5f, loss_crf %.5f, loss_ft %.5f" % \
                        (acc_bs, batches_processed, loss_accumulated / batches_processed,\
                         loss_crf_accumulated / batches_processed, loss_ft_accumulated / batches_processed))
        
            if acc_bs % args.save_every == 0:
                model.eval()
                gold_tag_list = []
                pred_tag_list = []
                with torch.no_grad():
                    with open(dev_eval_path, 'w', encoding = 'utf8') as o:
                        for dev_step in range(dev_step_num):
                            dev_batch_text_list, dev_batch_tag_list = nerdata.get_next_batch(batch_size = 1, mode = 'dev')
                            dev_tag_matrix = process_batch_tag(dev_batch_tag_list, nerdata.label_dict)
                            dev_mask_matrix = make_mask(dev_batch_tag_list)
                            dev_batch_result, _, _, _ = \
                            model(dev_batch_text_list, dev_mask_matrix, dev_tag_matrix, fine_tune = False)

                            dev_text = ''
                            for token in dev_batch_text_list[0]:
                                dev_text += token + ' '
                            dev_text = dev_text.strip()

                            valid_dev_text_len = len(dev_batch_text_list[0])
                            dev_tag_str = ''
                            pred_tags = []
                            for tag in dev_batch_result[0][1:valid_dev_text_len + 1]:
                                dev_tag_str += id_label_dict[int(tag)] + ' '
                                pred_tags.append(int(tag))
                            dev_tag_str = dev_tag_str.strip()
                            out_line = dev_text + '\t' + dev_tag_str
                            o.writelines(out_line + '\n')
                            gold_tag_list.append(dev_batch_tag_list[0])
                            pred_tag_list.append(pred_tags)
                    assert len(gold_tag_list) == len(pred_tag_list) 
                    pp, rr, ff = 0., 0., 0.
                    for glist, plist in zip(gold_tag_list, pred_tag_list):
                        acc = 0.
                        for gi, gtag in enumerate(glist):
                            if gtag == plist[gi]:
                                acc += 1
                        pi = acc / (len(plist)+1e-8)
                        ri = acc / (len(glist)+1e-8)
                        fi = 2 * pi * ri / (pi + ri + 1e-8)
                        pp += pi
                        rr += ri
                        ff += fi

                    one_dev_f1 = ff / len(gold_tag_list)
                    one_dev_precision = pp / len(gold_tag_list)
                    one_dev_recall = rr / len(gold_tag_list)
                    
                    dev_f1_list.append(one_dev_f1)
                    dev_precision_list.append(one_dev_precision)
                    dev_recall_list.append(one_dev_recall)

                    print ('At epoch %d, official dev f1 : %f, precision : %f, recall : %f' % \
                            (epoch, one_dev_f1, one_dev_precision, one_dev_recall))
                    torch.save({'args':args, 'model':model.state_dict(), 
                            'bert_args': bert_args, 
                            'bert_vocab':model.bert_vocab
                            }, directory + '/epoch_%d_dev_f1_%.3f'%(epoch + 1, one_dev_f1))
                    max_dev_f1 = one_dev_f1
                    
                model.train() # !!!!!!


    max_dev_f1_idx = np.argmax(dev_f1_list)
    max_dev_f1 = dev_f1_list[max_dev_f1_idx]
    max_dev_precision = dev_precision_list[max_dev_f1_idx]
    max_dev_recall = dev_recall_list[max_dev_f1_idx]

    print ('-----------------------------------------------------')
    print ('At this run, the maximum dev f1:%f, dev precision:%f, dev recall:%f' % \
        (max_dev_f1, max_dev_precision, max_dev_recall))
    print ('-----------------------------------------------------')
