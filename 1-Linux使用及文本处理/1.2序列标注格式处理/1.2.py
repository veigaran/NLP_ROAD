#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2022-03-17 20:48
@Author:Veigar
@File: 1.2.py
@Github:https://github.com/veigaran
"""
import os
import re


class Sequence:
    def __init__(self, folder, out_path):
        self.folder = folder
        self.out_path = out_path
        self.txt_path_list = []

    def get_all_txt_path(self):
        # path表示路径
        # 返回path下所有文件构成的一个list列表
        filelist = os.listdir(self.folder)
        name_list = []
        # 遍历输出每一个文件的名字和类型
        for item in filelist:
            name_list.append(item)
        return name_list

    def process(self):
        res = []
        txt_list = self.get_all_txt_path()
        for txt in txt_list:
            path = os.path.join(self.folder, txt)
            with open(path, 'r', encoding='utf') as f:
                text = f.read().replace('\n', '')
                res.extend(self.word_process(text))
        with open(self.out_path, 'w', encoding='utf') as fw:
            for info in res:
                fw.write(info)

    @staticmethod
    def word_process(string):
        word_list = re.findall(r"<ICH-TERM>.*?<ICH-TERM>|[^<ICH-TERM><ICH-TERM>]+", string)  # 提取<ICH-TERM>内的内容
        temp = []
        for word in word_list:
            if word != '':
                if word == '。' or word == '！' or word == '？':
                    temp.append(word + "\tO" + "\n\n")
                else:
                    if word[0] != '<':
                        for w in word:
                            if w == '。' or w == '！' or w == '？':
                                temp.append(w + "\tO" + "\n\n")
                            else:
                                temp.append(w + "\tO" + "\n")
                    else:
                        temp.append(word[10] + "\tB" + '-TERM' + "\n")
                        for w in word[11:len(word) - 11]:
                            temp.append(w + "\tI" + '-TERM' + "\n")
                        temp.append(word[len(word) - 11] + "\tE" + '-TERM' + "\n")
        return temp


if __name__ == '__main__':
    model = Sequence('data', '1.txt')
    model.process()
