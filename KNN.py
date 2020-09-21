'''
Step1: 处理数据：从CSV中读取数据，并把它们分割成训练数据集和测试数据集。

Step2: 相似度：计算两个数据实例之间的距离。

Step3: 临近：确定最相近的N个实例。

Step4: 结果：从这些实例中生成预测结果。

Step5: 准确度：总结预测的准确度。

Step6: 主程序：把这些串起来
'''
# 数据集： iris.data
# -*- coding:utf-8 -*-

import csv
import random
import math
import operator
# 数据集由对3个不同品种的鸢尾花的150组观察数据组成；对于这些花有4个测量维度：萼片长度、萼片宽度、花瓣长度、花瓣宽度，所有的数值都以厘米为单位。需要预测的属性是品种，品种的可能值有：清风藤、云芝、锦葵。
# 有一个标准数据集，在这个数据集中品种是已知的。可以把这个数据集切分成训练数据集和测试数据集，然后用测试结果来评估算法的准确程度。在这个问题上，好的分类算法应该有大于90%的正确率，通常都会达到96%甚至更高。

# Step1
# 数据集切分成用来做预测的训练数据集和用来评估准确度的测试数据集
# 训练数据集数据量/测试数据集数据量的比值取67/33是一个常用的惯例
def loadDataSet(fileName, split, trainSet = [], testSet = []):
    with open(fileName, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        for x in range(len(dataSet) - 1):
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
                if random.random() < split:
                    trainSet.append(dataSet[x])
                else:
                    testSet.append(dataSet[x])

# Step2
def euclideanDistance(instance1, instance2, length): # length 用于指定参与计算的维度
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

# Step3
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Step4
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1] # 属性位于数组的最后一位
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# Step5
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

if __name__ == '__main__':
    trainSet = []
    testSet = []
    split = 0.67
    loadDataSet('iris.data', split, trainSet, testSet)
    print('trainSet:' + repr(len(trainSet)))
    print('testSet:' + repr(len(testSet)))
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predictied = ' + repr(result), 'actual = ' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('accuracy: ' + repr(accuracy) + '%')
    
    