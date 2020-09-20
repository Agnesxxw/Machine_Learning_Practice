'''
Step1: 
    为待聚类的点随机寻找聚类中心
Step2:
    计算每个点到聚类中心的距离，将各个店归类到里该点最近的聚类中去
Step3:
    计算每个聚类中所有点的坐标平均值，并将其作为新的聚类中心，反复执行前两步，知道不再进行大范围移动或者巨雷次数达到要求为止
'''
# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot


class K_Means(object):
    #k是分组数；tolerance中心点误差；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):  
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
    
    def fit(self, data):  # 一开始随机选择k个点作为初始点
        self.centers_ = {}  # 质点
        for i in range(self.k_):
            self.centers_[i] = data[i]
        # 迭代开始
        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)
            # print('分组：', self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis = 0)               
            # 中心点是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break
    
    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index
   
    
if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(k=2)
    k_means.fit(x)
    print(k_means.centers_)
    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

    predict = [[2, 1], [6, 9], [3, 8], [10, 11], [4, 7]]
    for feature in predict:
        cat = k_means.predict(predict)
        pyplot.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

    pyplot.show()
        
            
            
        