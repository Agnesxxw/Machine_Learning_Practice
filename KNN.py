'''
Step1: 处理数据：从CSV中读取数据，并把它们分割成训练数据集和测试数据集。

Step2: 相似度：计算两个数据实例之间的距离。

Step3: 临近：确定最相近的N个实例。

Step4: 结果：从这些实力中生成预测结果。

Step5: 准确度：总结预测的准确度。

Step6: 主程序：把这些串起来
'''
# 数据集： iris.data
# -*- coding:utf-8 -*-

import csv
import random
# 数据集由对3个不同品种的鸢尾花的150组观察数据组成；对于这些花有4个测量维度：萼片长度、萼片宽度、花瓣长度、花瓣宽度，所有的数值都以厘米为单位。需要预测的属性是品种，品种的可能值有：清风藤、云芝、锦葵。
# 有一个标准数据集，在这个数据集中品种是已知的。可以把这个数据集切分成训练数据集和测试数据集，然后用测试结果来评估算法的准确程度。在这个问题上，好的分类算法应该有大于90%的正确率，通常都会达到96%甚至更高。

# Step1
with open('iris.data', 'rb') as csvfile:
    lines = csv.reader(csvfile)
    next(lines, None) # 此处的next语句为不读取标题key值
    for row in lines:
        print (', '.join(row))
# 数据集切分成用来做预测的训练数据集和用来评估准确度的测试数据集
def loadDataSet(fileName, split, trainSet = [], testSet = []):
    with open(fileName, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        for x in range(len(dataSet) - 1):
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
                if random.random() < split:
                    trainSet.append(dataSet[x])
                else:
                    testSet.append(dataSet[x])
trainSet = []
testSet = []
loadDataSet('iris.data', 0.66, trainSet, testSet)
print('trainSet:', trainSet)
print('testSet:', testSet)
# Step2
