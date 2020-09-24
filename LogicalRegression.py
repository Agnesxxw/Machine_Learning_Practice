'''
初始化W和b
指定learning rate和迭代次数
每次迭代，根据当前W和b计算对应的梯度（J对W，b的偏导数），然后更新W和b
迭代结束，学得W和b，带入模型进行预测，分别测试在训练集合测试集上的准确率，从而评价模型
'''s

import numpy as np 

# Step1: 载入训练和测试数据
def loadData(fileName):
    f = open(fileName)  
    feature = []  
    label = []  
    for row in f.readlines():
        f_tmp = []  
        l_tmp = []  
        number = row.strip().split("\t")  # 得到每行特征和标签
        f_tmp.append(1)  
        for i in range(len(number) - 1):
            f_tmp.append(float(number[i]))
        l_tmp.append(float(number[-1]))
        feature.append(f_tmp)
        label.append(l_tmp)
    f.close()  
    return np.mat(feature), np.mat(label)


# Step2: sigmoid func
def sigmoid(x):
    return 1.0 / 1 + np.exp(-x)

# Step3: Loss Function
def lossFunc(predict, label):
    m = np.shape(predict)[0]
    errSum = 0.0
    for i in range(m):
        if predict[i, 0] > 0 and (1 - predict[i, 0]) > 0:
            errSum -= (label[i, 0] * np.log(predict[i, 0]) + (1 - label[i, 0]) * np.log(1 - predict[i, 0]))
        else:
            errSum -= 0.0
    return errSum / m

# Step4: gradiant descent
def gradDesc(feature, label, maxIteration, alpha):
    m = np.shape(feature[1]) # feature的个数
    w = np.mat(np.ones(m, 1))
    i = 0 
    while i <= maxIteration:  # 当指标小于最大迭代次数时
        i += 1
        sig = sigmoid(feature * w)  # 调用sigmoid函数计算sigmoid的值
        error = label - sig
        w = w + alpha * feature.T * error  # 权重修正
        if i % 100 == 0:
            print("迭代", str(i), "时的错误率为：", str(lossFunc(sig, label)))
    return w

# Step5: Save weights
def saveWeights(weight, fileName):
    m = np.shape(weight)[0]
    f = open(fileName, 'w')
    weight_list = []
    for i in range(m):
        weight_list.append(str(weight[i, 0]))
    f.write('\t'.join(weight_list))
    f.close()
 
 # ------------------------------------------------------------------
    
def loadFile(fileName, num):
    '''
    加载测试集
    '''
    f = open(fileName)
    feature = []
    for row in f.readlines():
        f_tmp = []
        number = row.strip().split("\t")
        if len(number) != num - 1:  # 排除测试集中不符合要求的数据
            continue
        f_tmp.append(1)  # 设置偏置项
        for i in number:
            f_tmp.append(float(i))
        feature.append(f_tmp)
    f.close()
    return np.mat(feature)
 
def loadWeights(weights):
    '''
    加载权重值
    '''
    f = open(weights)
    w = []
    for row in f.readlines():
        number = row.strip().split("\t")
        w_tmp = []
        for i in number:
            w_tmp.append(float(i))
        w.append(w_tmp)
    f.close()
    return np.mat(w)
 
def predict(feature, w):
    '''
    对测试数据进行预测
    '''
    sig = sigmoid(feature * w.T)
    n = np.shape(sig)[0]
    for i in range(n):
        if sig[i, 0] < 0.5:
            sig[i, 0] = 0.0
        else:
            sig[i, 0] = 1.0
    return sig
 
def saveResult(fileName, result):
    '''
    保存预测结果
    '''
    m = np.shape(result)[0]
    res = []
    for i in range(m):
        res.append(str(result[i, 0]))
    f = open(fileName, "w")
    f.write("\t".join(res))
    f.close()
 
if __name__ == "__main__":
    path = "./data/"  # 数据集的存放路径
    w = loadWeights("weights")
    n = np.shape(w)[1]
    lr_test_data = loadFile(path + "test.txt", n)
    sig = predict(lr_test_data, w)
    saveResult("resultData", sig)