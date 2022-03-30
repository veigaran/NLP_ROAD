#!/usr/bin/env python  
# -*- coding:utf-8 _*-  

class Cluster:
    def __init__(self):
        self.corpus = 'fenci.csv'
        self.cluster_docs = 'cluster_docs.txt'  # 聚类后的类别
        self.cluster_keywords = 'cluster_keywords.txt'
        self.num_clusters = 5  # 聚类的簇数
        self.labels = None  # 真实类别

    def main(self):
        pass

if __name__ == '__main__':
    handler = Cluster()
    handler.main()
