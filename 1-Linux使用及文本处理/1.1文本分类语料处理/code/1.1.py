#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2022-03-17 20:29
@Author:Veigar
@File: 1.1.py
@Github:https://github.com/veigaran
"""
import os


class textprocess:
    def __init__(self, folder,out_path):
        self.folder = folder
        self.out_path = out_path
        self.txt_path_list = []

    def get_all_txt_path(self, folder):
        # path表示路径
        # 返回path下所有文件构成的一个list列表
        filelist = os.listdir(folder)
        name_list = []
        # 遍历输出每一个文件的名字和类型
        for item in filelist:
            name_list.append(item)
        return name_list

    def single_folder(self, folder_name):
        folder_path = os.path.join(self.folder, folder_name)
        name_list = self.get_all_txt_path(folder_path)
        for txt in name_list:
            txt_path = os.path.join(folder_path, txt)
            self.txt_path_list.append([folder_name, txt_path])

    def write2csv(self):
        result = []
        for path_info in self.txt_path_list:
            with open(path_info[1], 'r', encoding='utf') as f:
                sentence_list = f.readlines()  # f.readlines()表示按行读取并存取在一个列表内
            for line in sentence_list:
                result.append(path_info[0] + "\t" + line)

        with open(self.out_path, 'w', encoding='utf') as fw:
            for info in result:
                fw.write(info)


if __name__ == '__main__':
    model = textprocess("./data",'1.csv')
    dynasty_name = model.get_all_txt_path("./data")
    print(dynasty_name)
    # for name in dynasty_name:
    #     model.single_folder(name)
    # model.write2csv()
