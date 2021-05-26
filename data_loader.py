import numpy as np

class DataLoader:
    def __init__(self, train_path, dev_path, test_path, label_dict, train_max_len):
        self.train_path, self.dev_path, self.test_path = train_path, dev_path, test_path
        self.label_dict = label_dict
        self.train_max_len = train_max_len
        self.train_text_list, self.train_tag_list = self.process_file(train_path)
        self.dev_text_list, self.dev_tag_list = self.process_file(dev_path)
        self.test_text_list, self.test_tag_list = self.process_file(test_path)
        
        
        self.train_num = len(self.train_text_list)
        self.dev_num = len(self.dev_text_list)
        self.test_num = len(self.test_text_list)
        print ('training number is %d, dev number is %d, test_num is %d' % \
               (self.train_num, self.dev_num, self.test_num))
        
        self.train_idx_list = [i for i in range(self.train_num)]
        np.random.shuffle(self.train_idx_list)
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.test_idx_list = [j for j in range(self.test_num)]
        
        self.train_current_idx = 0
        self.dev_current_idx = 0
        self.test_current_idx = 0

        max_train_seq_len = 0
        for item in self.train_text_list:
            max_train_seq_len = max(len(item), max_train_seq_len)
        max_dev_seq_len = 0
        for item in self.dev_text_list:
            max_dev_seq_len = max(len(item), max_dev_seq_len)
        max_test_seq_len = 0
        for item in self.test_text_list:
            max_test_seq_len = max(len(item), max_test_seq_len)
        print ('Maximum train sequence length: %d, dev sequence length: %d, test sequence length: %d' % \
            (max_train_seq_len, max_dev_seq_len, max_test_seq_len))
        
    def get_next_batch(self, batch_size, mode):
        batch_text_list, batch_tag_list = [], []
        if mode == 'train':
            if self.train_current_idx + batch_size < self.train_num:
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    batch_text_list.append(self.train_text_list[self.train_idx_list[curr_idx]])
                    batch_tag_list.append(self.train_tag_list[self.train_idx_list[curr_idx]])
                self.train_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    if curr_idx > self.train_num - 1:
                        self.shuffle_train_idx()
                        curr_idx = 0
                        self.train_current_idx = 0
                    else:
                        pass
                    batch_text_list.append(self.train_text_list[self.train_idx_list[curr_idx]])
                    batch_tag_list.append(self.train_tag_list[self.train_idx_list[curr_idx]])
                self.train_current_idx = 0
        elif mode == 'dev':
            if self.dev_current_idx + batch_size < self.dev_num:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i
                    batch_text_list.append(self.dev_text_list[curr_idx])
                    batch_tag_list.append(self.dev_tag_list[curr_idx])
                self.dev_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i
                    if curr_idx > self.dev_num - 1: # 对dev_current_idx重新赋值
                        curr_idx = 0
                        self.dev_current_idx = 0
                    else:
                        pass
                    batch_text_list.append(self.dev_text_list[curr_idx])
                    batch_tag_list.append(self.dev_tag_list[curr_idx])
                self.dev_current_idx = 0
        elif mode == 'test':
            if self.test_current_idx + batch_size < self.test_num:
                for i in range(batch_size):
                    curr_idx = self.test_current_idx + i
                    batch_text_list.append(self.test_text_list[curr_idx])
                    batch_tag_list.append(self.test_tag_list[curr_idx])
                self.test_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.test_current_idx + i
                    if curr_idx > self.test_num - 1: # 对test_current_idx重新赋值
                        curr_idx = 0
                        self.test_current_idx = 0
                    else:
                        pass
                    batch_text_list.append(self.test_text_list[curr_idx])
                    batch_tag_list.append(self.test_tag_list[curr_idx])
                self.test_current_idx = 0
        else:
            raise Exception('Wrong batch mode!!!')

        return batch_text_list, batch_tag_list
        
    def shuffle_train_idx(self):
        np.random.shuffle(self.train_idx_list)
        
    def process_file(self, in_path):
        all_text, all_tag = [], []
        with open(in_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_text, one_tag = self.process_one_line(l)
                if len(one_text) > self.train_max_len: # 限制训练过程中序列的最大长度
                    continue
                else:
                    pass
                all_text.append(one_text)
                all_tag.append(one_tag)
        return all_text, all_tag
    
    def process_one_line(self, line):
        content_list = line.strip().split('\t')
        assert len(content_list) == 3
        text_list = [w for w in content_list[0].strip()] #+ ['<-SEP->']
        tag_name_list = [w for w in content_list[1].strip()] + ['<-SEP->'] 
        if len(tag_name_list) > len(text_list):
            text_list += ['<-MASK->'] * (len(tag_name_list) - len(text_list))
            text_list += ['<-SEP->']
            tag_name_list += ['<-SEP->']
        elif len(tag_name_list) < len(text_list):
            tag_name_list += ['<-SEP->'] + ['<-PAD->'] * (len(text_list) - len(tag_name_list))
            text_list += ['<-SEP->']
        else:
            tag_name_list += ['<-SEP->']
            text_list += ['<-SEP->']
        assert len(text_list) == len(tag_name_list)
        tag_list = list()
        for token in tag_name_list:
            tag_list.append(self.label_dict.token2idx(token))
        return text_list, tag_list
