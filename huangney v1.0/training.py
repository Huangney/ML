import os
import pandas as pd
import time
import pickle
from tensorflow import keras, compat
from functions import read_file, split_words, indexing, padding, session
from model_framework import tf_datas, Encoder, Decoder, train
import json


def self_training(if_gpu=True):
    if if_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("已启用GPU！")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("已启用CPU！")

    pd.set_option('display.max_rows', None)

    limit = 24
    epochs = 15
    batch_size = 8
    embedding_dim = 300
    units = 1024

    print("自定义模式已启动")
    print("请告诉我几个参数：")
    print("最大句子长度（单位：词）：")
    limit = int(input())
    print("训练集分批数量：")
    epochs = int(input())
    print("每一批的大小，不要大于你的总批数")
    batch_size = int(input())
    print("每个词的特征数量：")
    embedding_dim = int(input())
    print("每个网络的神经元个数：")
    units = int(input())
    path = r".\my_data"
    f_name = "dialog.txt"
    json_name = "params.json"
    json_path = os.path.join(path, json_name)

    jsondic = {"limit": limit,
               "epochs": epochs,
               "batch_size": batch_size,
               "embedding_dim": embedding_dim,
               "units": units}

    with open(json_path, 'w') as f:
        json.dump(jsondic, f, sort_keys=True, ensure_ascii=False)
        print("参数已保存!")

    print("确认将您的dialog.txt文件置于旁边的my_data文件下")

    try:
        a = read_file(path, f_name, dropemp=True)
        print("已读取")
    except:
        print("没有找到该文件！")
        time.sleep(5)
        return
    # a是一个pandas Data Frame文件，内置“inp”“outp”两列

    split_words(a)
    # 分词

    a, word_size, word_index, tok = indexing(a, sort=True, limit=limit)
    # 索引化

    word_dict_name = "words_dict.json"
    tok_name = "tok.pkl"
    tok_path = os.path.join(path, tok_name)
    dict_path = os.path.join(path, word_dict_name)
    try:
        with open(dict_path, 'w',encoding='utf-8') as dict_file:
            json.dump(word_index, dict_file, ensure_ascii=False)
    except Exception as e:
        print(e)
        print("字典保存失败")
    try:
        with open(tok_path, 'wb') as tk_file:
            pickle.dump(tok, tk_file, protocol=pickle.HIGHEST_PROTOCOL)
        # 储存toknizer
    except:
        print("tok保存失败")

    try:
        padding(a, limit)
    except:
        print("填充失败")
        return
    # 统一长度填充
    try:
        a_data = tf_datas(a, batch_size=batch_size, epochs=epochs)
    except:
        print("数据集转换失败")
        return
    # 转换为tf.data

    try:
        encoder = Encoder(word_size, embedding_dim, units, batch_size)
        decoder = Decoder(word_size, embedding_dim, units, batch_size)
    except:
        print("模型创建失败")
        return
    # 创造编码器解码器类

    # %%
    try:
        lenth = len(a["inp"])
        optimizer = keras.optimizers.Adam()
        train(a_data, encoder, decoder, epochs, lenth, optimizer)
    except:
        print("模型训练失败")
        return True
    items = [word_size, encoder, decoder, limit, tok]

    # %%

    model_path = "./my_models"
    enmodel_name = "encoder_weights"
    demodel_name = "decoder_weights"
    en_path = os.path.join(model_path, enmodel_name)
    de_path = os.path.join(model_path, demodel_name)
    encoder.save_weights(en_path)
    decoder.save_weights(de_path)

    # %%

    print("训练已完成！模型已保存至my_models！")
    time.sleep(3)

    return True
