import json
import os
import pandas as pd
import re
import pickle
import numpy as np
import jieba
import tensorflow as tf
from tensorflow import keras
from online_learning import uni_train


def read_file(path, file_name, dropemp=False):
    file_path = os.path.join(path, file_name)
    op, ip = [], []
    a = pd.DataFrame(columns=["inp", "outp"])

    with open(file_path, encoding='utf-8') as file:
        for index, sent in enumerate(file):
            sent = re.sub("(])", "", sent)
            sent = re.sub("[[]", "", sent)
            if index % 2 == 1:
                op.append(sent.strip())
            else:
                ip.append(sent.strip())
        a["inp"] = ip
        a["outp"] = op

    if dropemp:
        emp = []
        for ind in a.itertuples():
            if ind[1] == "" or ind[2] == "":
                emp.append(ind[0])
        a = a.drop(index=emp).reset_index().drop(columns="index")

    return a


# 返回一个pandasDataFrame文件

def split_words(data):
    for row in data.itertuples():
        in_cut = jieba.cut(row[1])
        out_cut = jieba.cut(row[2])
        in_li = [i for i in in_cut]
        out_li = [i for i in out_cut]
        in_li = ["<START>"] + in_li + ["<END>"]
        out_li = ["<START>"] + out_li + ["<END>"]
        data["inp"][row[0]] = in_li
        data["outp"][row[0]] = out_li


# 分词

def part(lis, left, right):
    temp = lis[left]
    #     print(temp)
    while left < right:
        while left < right and lis[right][1] >= temp[1]:
            right -= 1
        #         print(f"{lis[right]}--->{lis[left]}")
        lis[left] = lis[right]
        while left < right and lis[left][1] <= temp[1]:
            left += 1
        #         print(f"{lis[left]}--->{lis[right]}")
        lis[right] = lis[left]

    mid = left
    lis[mid] = temp
    return mid


def sort_ind_li(lis, left, right):
    if left < right:
        mid = part(lis, left, right)
        sort_ind_li(lis, left, mid - 1)
        sort_ind_li(lis, mid + 1, right)


# 采用快排来处理元组列表，可用于大数据集你妈的写死我了草泥马的5555555

def sort_dataframe(data, tup_li, limit=100):
    new_data = pd.DataFrame(columns=["inp", "outp"])
    sort_inp, sort_outp = [], []
    for i in range(len(data["inp"])):
        if tup_li[i][1] > limit:
            break
        #         new_data["inp"][i] = data["inp"][tup_li[i][0]]
        #         new_data["outp"][i] = data["outp"][tup_li[i][0]]
        sort_inp.append(data["inp"][tup_li[i][0]])
        sort_outp.append(data["outp"][tup_li[i][0]])

    new_data["inp"] = sort_inp
    new_data["outp"] = sort_outp

    return new_data


# 将句子按长度排序

def indexing(a, sort=True, limit=100):
    tok = keras.preprocessing.text.Tokenizer(filters='')

    for sent in a["inp"]:
        tok.fit_on_texts(sent)
    for sent in a["outp"]:
        tok.fit_on_texts(sent)
    print(f"共{len(tok.word_index)}个词！")
    word_size = len(tok.word_index)
    word_index = tok.word_index
    print(word_index)

    a["inp"] = tok.texts_to_sequences(a["inp"])
    a["outp"] = tok.texts_to_sequences(a["outp"])
    for index, sent in enumerate(a["inp"]):
        a["inp"][index] = np.array(sent)
    for index, sent in enumerate(a["outp"]):
        a["outp"][index] = np.array(sent)

    if sort:
        lentup_li = zip([len(i) for i in a["inp"]], [len(j) for j in a["outp"]])
        len_li = [max(tuples[0], tuples[1]) for tuples in lentup_li]
        len_li_indexed = list(zip([i for i in range(len(len_li))], len_li))

        sort_ind_li(len_li_indexed, 0, len(len_li) - 1)
        #         print(len_li_indexed)
        a = sort_dataframe(a, len_li_indexed, limit)

    return a, word_size + 1, word_index, tok


# 将文本索引化

def padding(data, limit):
    data["inp"] = list(keras.preprocessing.sequence.pad_sequences(data["inp"],
                                                                  maxlen=limit,
                                                                  padding="post"))
    data["outp"] = list(keras.preprocessing.sequence.pad_sequences(data["outp"],
                                                                   maxlen=limit,
                                                                   padding="post"))
    return data


# 填充文本

def splindex_sent(sent, word_index):
    cut = jieba.cut(sent)
    splited_sent = ["<start>"] + [i for i in cut] + ["<end>"]

    for index, word in enumerate(splited_sent):
        if word in word_index.keys():
            splited_sent[index] = [word_index[word]]
        else:
            splited_sent[index] = [0]
    return np.array(splited_sent).flatten()


def splindex_sent_foruni(sent, word_index, word_size):
    word_size -= 1
    # print("目前共有",word_size,"个词")
    # 把用于神经元数量的word_size归为实际的word数量
    cut = jieba.cut(sent)
    splited_sent = ["<start>"] + [i for i in cut] + ["<end>"]

    for word in splited_sent:
        if word not in word_index.keys():
            word_size += 1
            word_index[str(word)] = word_size
            # print(f"第{word_size}个词：{word}或者说{str(word)}已经被添加到了字典中")

    # print("word_index的长度：{}，word_size的值：{}".format(len(word_index), word_size))

    for index, word in enumerate(splited_sent):
        if word in word_index.keys():
            splited_sent[index] = [word_index[word]]
        else:
            splited_sent[index] = [0]
    # print("转换后的句子：", splited_sent)

    path = "./data"
    dic_name = "words_dict.json"
    dic_path = os.path.join(path, dic_name)
    with open(dic_path, 'w', encoding='utf-8') as dic_file:
        json.dump(word_index, dic_file, ensure_ascii=False)

    return np.array(splited_sent).flatten(), word_size + 1, word_index


def evaluate(input_sentence, items, temp=0.55):
    word_size, encoder, decoder, limit, tok ,word_index= items[0], items[1], items[2], items[3], items[4],items[6]
    units = encoder.encoding_units
    attention_matrix = np.zeros((limit, limit))
    input_sentence = splindex_sent(input_sentence,word_index)
    # print("在evaluate中句子被转换为了：",input_sentence)

    inputs = keras.preprocessing.sequence.pad_sequences([input_sentence],
                                                        maxlen=limit,
                                                        padding="post")
    inputs = tf.convert_to_tensor(inputs)
    # print("在evaluate中句子被转换为了tf.tensor：", inputs)
    #     print(inputs)
    results = ''
    encoding_hidden = tf.zeros((1, units))

    encoding_outputs, encoding_hidden = encoder(inputs, encoding_hidden)

    decoding_hidden = encoding_hidden
    # print("tok.texts_to_sequences(['START'])：",tok.texts_to_sequences(["<START>"]))
    decoding_input = tf.convert_to_tensor([[1]])
    #     print("decoding_input = ",decoding_input)

    word_index_key = list(word_index.keys())
    word_index_value = list(word_index.values())

    for step in range(limit):
        pred, decoding_hidden, attention_weights = decoder(
            decoding_input, decoding_hidden, encoding_outputs)

        # 之所以要reshape是因为输出的pred还含有一个带有时间步长1的维度
        pred = tf.reshape(pred, [1, word_size])
        res_log = tf.math.log(pred) / temp

        pred_id = tf.random.categorical(res_log, num_samples=1).numpy()[0][0]

        #         pred_id = tf.argmax(pred[0]).numpy()
        # print(pred_id)

        if pred_id == word_index['<end>']:
            return results, input_sentence

        try:
            down_pointer = word_index_value.index(pred_id)
            the_word = word_index_key[down_pointer]
        except:
            the_word = ""
        results += the_word

        #         print("it's :",str(*tok.sequences_to_texts([[pred_id]])))

        decoding_input = tf.expand_dims([pred_id], 0)

    return results, input_sentence


def answer(input_senten, items, temp=1):
    print("you said: ", input_senten)
    results, input_senten = evaluate(input_senten, items, temp=temp)
    print("answered: ", results)
    return results, input_senten


def txt_save(inp, outp):
    file_name = "saved_conversation.txt"
    conver_path = "./my_data"
    txt_path = os.path.join(conver_path, file_name)
    with open(txt_path, 'a') as conversation:
        conversation.write(inp + "\n")
        conversation.write(outp + "\n")
        print("保存好了")


def session(items, temp=1):
    print("\n输入 '#1' 以停止\n输入 '#2' 保存前两句对话\n输入 '#3' 进行即时学习(基于CPU) '#3high'和'#3low'切换强弱学习模式\n")
    str1 = ""
    result = ""
    input_senten = ""
    lr = False

    while (True):
        str1 = input()

        if str1 == "#3high":
            lr = True
        if str1 == "#3low":
            lr = False

        if str1 == "#1":
            return
        elif str1 == "#2":
            txt_save(input_senten, result)
            continue

        if str1 == "#3":
            outp_sent = input("那我应该回答什么捏?\n")
            encoder1, decoder1, loss, new_word_size, word_index = uni_train(input_senten, outp_sent, items,lr)
            print("本次学习差距为：",loss.numpy())
            items[1] = encoder1
            items[2] = decoder1
            items[6] = word_index
            items[0] = new_word_size
            # print(loss)
            continue

        input_senten = str1
        result, ____ = answer(str1, items, temp)
