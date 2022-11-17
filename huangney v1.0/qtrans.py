import os
import tensorflow as tf
import pandas as pd
import re
import warnings
from time import sleep

def q_trans():
    fil_name = "test"
    path = r"./my_data"

    fil_name = input("请输入txt文件名称：")

    file_name = fil_name + ".txt"

    file_path = os.path.join(path, file_name)
    print("正在读取：" + file_path)

    try:
        F = open(file_path, encoding="utf-8")
    except:
        print("没读到捏\n")
        sleep(2)
        return
    words = [[], []]
    flag = 0  # 1: 1号说话,11:1号的空格 2:2号说话,12:2号的空格
    persons = []
    last_person = 0
    temp = ""
    for i in range(8):
        F.readline()

    for sent in F:
        if len(persons) < 2:
            time_name = re.findall("\d\d:\d\d:\d\d .+$",sent)
            try:
                name = re.findall(" .+",time_name[0])
                if not name[0] in persons and not "系统消息" in name[0]:
                    persons += name
            except:
                pass

        else:
            break

    for sent in F:
        if flag == 1:
            temp += sent.strip("\n")
            last_person = 1
            flag = 0

        if flag == 2:
            temp += sent.strip("\n")
            last_person = 2
            flag = 0

        if persons[0] in sent:
            flag = 1
        elif persons[1] in sent:
            flag = 2

        if flag != 0 and flag != last_person and last_person != 0:
            words[last_person - 1].append(temp)
            temp = ""
    words[last_person - 1].append(temp)

    F.close()

    # # whole_pas = re.sub("[，、 。：《》？“”‘’—…！； （）\u3000]","",wholepass)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(2):
            for index, sent in enumerate(words[i]):
                    words[i][index] = re.sub("\[图片]+|\[表情]+|[请使用最新版手机QQ查看。]", "", sent)



    print("3 inputs = ",words[0][:3])
    print("3 outputs = ",words[1][:3])

    out_name = fil_name + "_已处理" + ".txt"
    out_path = os.path.join(path,out_name)
    Note = open(out_path,mode='w',encoding = "utf-8")
    for inp in zip(words[0],words[1]):
        for a in inp:
            Note.write(a + "\n")
    # Note.write(words[0])
    Note.close()
    print("\n转换完成\n")
    sleep(2)