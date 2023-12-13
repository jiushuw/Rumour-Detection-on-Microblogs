import pickle as pickle
from random import *
import numpy as np
from torchvision import transforms
import os
import sys
import pandas as pd
from PIL import Image
from types import *
import jieba
import os.path

from utils import *

import sys

if sys.version_info[0] >= 3:
    unicode = str


def read_image(prefix):
    image_dict = {}
    file_list = [prefix + 'nonrumor_images/',
                 prefix + 'rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([  # 设置图像变换参数
            transforms.Resize(256),  # 变换大小
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),  # 转换为tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 用平均值和标准偏差归一化张量图像
        ])

        for filename in os.listdir(path):
            try:
                image = Image.open(path + filename).convert('RGB')
                image_tensor = data_transforms(image)
                image_dict[filename.split('/')[-1].split(".")[0].lower()] = image_tensor  #
                # 构造字典，选取'/'后、"."前的完整图片名称为key，tensor为value
            except:
                print(filename)  #  输出异常图片名
    print("已载入图片总数量:" + str(len(image_dict)), "张")
    return image_dict


# return: paired_dict
def read_match(flag, image_dict, text_only, prefix):
    # 返回每行的文本内容 & 包含事件的完整内容
    def read_post(flag):
        stop_words = stopwordslist(prefix)  # 构建停用词表
        pre_path = prefix + "tweets/"
        file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt",
                     pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]

        if flag == "trainset":
            id = pickle.load(open(prefix + "train_id.pickle", 'rb'))
        elif flag == "valset":
            id = pickle.load(open(prefix + "validate_id.pickle", 'rb'))
        elif flag == "testset":
            id = pickle.load(open(prefix + "test_id.pickle", 'rb'))

        cleaned_content = []
        all_data = []
        column = ['post_id', 'image_id', 'cleaned_post', 'seged_post', 'label', 'event_label']
        map_id = {}

        for file_idx, file in enumerate(file_list):
            file = open(file, 'rb')

            if (file_idx + 1) % 2 == 1:  # 按照顺序指定每个file的统一label
                label = 0
            else:
                label = 1

            for idx, line in enumerate(file.readlines()):
                if (idx + 1) % 3 == 1:  # 第一行取twitter_id，对应'post_id'
                    line_data = []
                    twitter_id = line.split('|'.encode('UTF-8'))[0]
                    line_data.append(twitter_id)

                if (idx + 1) % 3 == 2:  # 第二行取所有值，对应'image_id'
                    line_data.append(line.lower())

                if (idx + 1) % 3 == 0:  # 第三行文本清理;分词；限制最短长度；
                    cleaned_text = clean_str_sst(str(line, encoding="utf-8"))

                    # 确定句长
                    seg_list = jieba.cut_for_search(cleaned_text)
                    new_seg_list = []
                    for word in seg_list:
                        if word not in stop_words:
                            new_seg_list.append(word)
                    seged_line = " ".join(new_seg_list)

                    if len(seged_line) > 10 and line_data[0].decode(
                            "UTF-8") in id:  # line_data[0]为twitter_id，以twitter_id为索引
                        cleaned_content.append(cleaned_text)
                        line_data.append(cleaned_text)  # 对应'original_post'
                        line_data.append(seged_line)  # 对应'post_text'
                        line_data.append(label)  # 对应'label'

                        event = int(id[line_data[0].decode("UTF-8")])  # pickle文件内已标注好每个twitter_id对应的事件序号

                        # 构造事件映射字典（有序化）
                        if event not in map_id:
                            map_id[event] = len(map_id)
                            event = map_id[event]
                        else:
                            event = map_id[event]

                        line_data.append(event)  # 对应'event_label'

                        all_data.append(line_data)

            file.close()

        data_df = pd.DataFrame(np.array(all_data), columns=column)

        return cleaned_content, data_df

    content, post = read_post(flag)
    print(flag + "大小：" + str(len(content)))
    print(flag + "维度：" + str(post.shape))

    # 配对
    def paired(text_only=False):
        tensored_image = []
        cleaned_text = []
        seged_text = []
        ordered_event = []
        label = []
        post_id = []
        image_list = []
        image_name = ""

        for idx, post_idx in enumerate(post['post_id']):
            # 从文本内容中提出图片名字再去image_dict中查找,一旦找到一张图就break出循环
            for image_id in post.iloc[idx]['image_id'].split('|'):
                image_name = image_id.split("/")[-1].split(".")[0]
                if image_name in image_dict:
                    break

            if text_only or image_name in image_dict:
                if not text_only:
                    image_list.append(image_name)
                    tensored_image.append(image_dict[image_name])  # 规范化后的image

                cleaned_text.append(post.iloc[idx]['cleaned_post'])
                seged_text.append(post.iloc[idx]['seged_post'])
                ordered_event.append(post.iloc[idx]['event_label'])
                post_id.append(post.iloc[idx]['post_id'])
                label.append(post.iloc[idx]['label'])

        label = np.array(label, dtype=np.int_)
        ordered_event = np.array(ordered_event, dtype=np.int_)

        print(flag + "集标签数量：" + str(len(label)),
              "虚假新闻数量：" + str(sum(label)),
              "真实新闻数量：" + str(len(label) - sum(label)))

        data = {"post_text": np.array(seged_text),
                "original_post": np.array(cleaned_text),
                "image": tensored_image,
                "social_feature": [],
                "label": np.array(label),
                "event_label": ordered_event,
                "post_id": np.array(post_id),
                "image_id": image_list}

        return data

    paired_dict = paired(text_only)
    print("配对数据大小：" + str(len(paired_dict["post_text"])), "维度：" + str(len(paired_dict)))

    return paired_dict


def split_embed(text_only, prefix):
    if text_only:
        print("-" * 50, "载入文字", "-" * 50)
        image_dict = []
    else:
        print("-" * 50, "载入图片", "-" * 50)
        image_dict = read_image(prefix)  # 生成image_dict

    print("-" * 50, "图文配对", "-" * 50)
    train_data = read_match("trainset", image_dict, text_only, prefix)
    valiate_data = read_match("valset", image_dict, text_only, prefix)
    test_data = read_match("testset", image_dict, text_only, prefix)

    print("-" * 50, "统计数据量", "-" * 50)
    vocab, all_text = sum_post(train_data, valiate_data, test_data)
    max_l = len(max(all_text, key=len))
    print("推文总量: " + str(len(all_text)), "单词总量: " + str(len(vocab)), "句子最大长度: " + str(max_l))

    word_embedding_path = prefix + "w2v.pickle"
    w2v = pickle.load(open(word_embedding_path, 'rb+'), encoding='ISO-8859-1')

    add_unknown_words(w2v, vocab)

    W, word_idx_map = get_W(w2v)
    W2 = {}
    embedding_file = open(prefix + "word_embedding.pickle", "wb")
    # W：词向量, W2：空, word_idx_map：词序词典, vocab：词频词典, max_l：句子最大长度
    pickle.dump([W, W2, word_idx_map, vocab, max_l], embedding_file)
    embedding_file.close()

    return train_data, valiate_data, test_data
