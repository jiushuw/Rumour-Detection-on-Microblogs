import torch
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable
import pickle as pickle
import copy
from skimage import io, transform
import torchvision
import transformers
from torchvision import transforms
from transformers import BertTokenizer
from transformers import BertModel
from process_weibo import *


class Transform_Numpy_Tensor(Dataset):
    def __init__(self, flag, dataset, tokenizer):
        self.text = dataset['original_post']
        self.image = list(dataset['image'])
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        self.tokenizer_bert = tokenizer
        print(flag,
              'item count',
              'text: %d, image: %d, label: %d, event_label: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label))
              )

    def __len__(self):
        return len(self.label)

    def pre_processing_BERT(self, sent):

        # 创建空列表储存输出
        input_ids = []
        attention_mask = []

        encoded_sent = self.tokenizer_bert.encode_plus(
            text=sent,  # 预处理
            add_special_tokens=True,  # `[CLS]`&`[SEP]`
            max_length=400,  # 截断/填充的最大长度
            padding='max_length',  # 句子填充最大长度
            # return_tensors='pt',          # 返回tensor
            return_attention_mask=True,  # 返回attention mask
            truncation=True
        )

        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')

        # 转换tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return input_ids, attention_mask

    def __getitem__(self, idx):
        input_ids, attention_mask = self.pre_processing_BERT(self.text[idx])
        return (input_ids, self.image[idx], attention_mask), self.label[idx], self.event_label[idx]


def Transform_Tensor_Variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def Transform_Tensor_Numpy(x):
    return x.data.cpu().numpy()


def load_data(args):
    # 分割数据集和embed
    train, validate, test = split_embed(args.text_only, args.prefix)

    word_vector_path = args.prefix + 'word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
    args.vocab_size = len(vocab)
    args.max_len = max_len

    word_embedding, mask = word2vec(validate['post_text'], word_idx_map, args)
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    word_embedding, mask = word2vec(test['post_text'], word_idx_map, args)
    test['post_text'] = word_embedding
    test['mask'] = mask

    word_embedding, mask = word2vec(train['post_text'], word_idx_map, args)
    train['post_text'] = word_embedding
    train['mask'] = mask

    print("-" * 50, "End of data loading", "-" * 50)
    return train, validate, test, W
