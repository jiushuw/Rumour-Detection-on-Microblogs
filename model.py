import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as functional

from importlib_metadata import re
import numpy as np
import transformers
from torchvision import models, transforms
from transformers import BertModel

device = torch.device("cuda")
import sys

sys.path.append('src/')

# 导入失败时重启下ssh
from gradreverse import *


# 文本Bert基本模型
class TextEncoder(nn.Module):

    def __init__(self, text_fc2_out=64, text_fc1_out=2742, dropout_p=0.4, fine_tune_module=False):
        super(TextEncoder, self).__init__()

        self.fine_tune_module = fine_tune_module

        # 实例化
        self.bert = BertModel.from_pretrained(
            'bert-base-chinese',
            #                     output_attentions = True,
            return_dict=True)

        self.text_enc_fc1 = torch.nn.Linear(768, text_fc1_out)

        self.text_enc_fc2 = torch.nn.Linear(text_fc1_out, 128)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()

    def forward(self, input_ids, attention_mask):
        """
        输入Bert和分类器，计算logis

        @argu    input_ids (torch.Tensor): input (batch_size,max_length)

        @argu    attention_mask (torch.Tensor): attention mask information (batch_size, max_length)

        @return    logits (torch.Tensor): output (batch_size, num_labels)
        """

        # 输入BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(out['pooler_output'].shape)
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc1(out['pooler_output']))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc2(x))
        )

        return x

    def fine_tune(self):
        """
        固定参数
        """
        for p in self.bert.parameters():
            p.requires_grad = self.fine_tune_module


class EANN(nn.Module):
    def __init__(self, args, W):
        super(EANN, self).__init__()

        self.args = args
        self.event_num = args.event_num
        self.hidden_size = args.hidden_dim
        self.text_encoder = TextEncoder(64, 2742, 0.4, False)
        """Vision_Part"""

        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        resnet50.fc = nn.Linear(2048, self.hidden_size)
        # vgg_19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        for param in resnet50.parameters():
            param.requires_grad = False

        self.vision_model = resnet50

        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        """Classifer_Part"""
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('classifer_fc1', nn.Linear(2 * self.hidden_size, 2))
        self.class_classifier.add_module('classifer_softmax', nn.Softmax(dim=1))

        """Discriminator_Part"""
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('discriminator_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('discriminator_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('discriminator_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('discriminator_softmax', nn.Softmax(dim=1))

    def forward(self, text, image, mask):
        """Image"""
        image = self.vision_model(image)

        """Text"""
        text = self.text_encoder(text, mask)

        """Concat"""
        text_image = torch.cat((text, image), 1)

        """Classifer"""
        # class_output = self.class_classifier(text_image)
        class_output = self.class_classifier(text)
        """Discriminator"""
        reverse_feature = grad_reverse(text, self.args)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, text
        # return class_output, domain_output, text_image