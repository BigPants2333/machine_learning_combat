# -*- coding: utf-8 -*-

import xlrd   #读取excel文件
import re     #正则表达式
import jieba  #结巴中文分词
import math
import random
import numpy as np
import os
file='chinesespam.xlsx'
wb=xlrd.open_workbook(filename=file)
ws=wb.sheet_by_name('chinesespam.')
dataset=[]
dataset1=[]
dataset2=[]
temp=[]
vocabList=[]
Percision = 0.0
Percision_All = 0.0
Recall = 0.0
Recall_All = 0.0
General = 0.0
General_All = 0.0

for i in range(1,ws.nrows):
     temp.append(ws.cell(i,0).value)
     if  'ham'==temp[i-1]:
         dataset1.append(0)
     else: 
         dataset1.append(1)
     
for r in range(1,ws.nrows):
     dataset2.append(ws.cell(r,1).value)
     
def textHandle(bigString):
    #分词，有的在网上找到的例子中创建了停止词列表将那些无关语义表达的词如“一天”删去
    list1 = jieba.lcut(bigString)
    newList = [re.sub(r'\W*','',s) for s in list1] 
    #将不是字母，数字，下划线，汉字的字符删去
    return[tok.lower() for tok in newList if len(tok) > 0]
for i in range(len(dataset2)):   #将分好的词存入到列表中
    dataset.append(textHandle(dataset2[i]))
    
#测试用 print(dataset) 
def createVocabList(dataSet):
    vocabSet=set([])  #创建一个空集
    for document in dataSet:
        	vocabSet=vocabSet|set(document) #取两个集合的并集
    return list(vocabSet)     #以list的方式返回结果
vocabList=createVocabList(dataset)   
#该变量是已分好词的列表     
#print(createVocabList(dataset)) 测试用
def setOfWords2Vec(vocabList,inputSet):
    returnVec=np.zeros(len(vocabList)) #生成零向量的array
    for word in inputSet:
            if word in vocabList:
                    returnVec[vocabList.index(word)]=1 #单词出现则记为1
            else: print('the word:%s is not in my Vocabulary!'% word)
    return returnVec #返回0.1向量
#测试用 for i in range(149):
#测试用    print(setOfWords2Vec(vocabList,dataset[i]))
#测试用 8127  print(len(vocabList))
#现在要做 减少8000维度，经过一番想法，还是改用词集模型
def trainNB(trainDataSet,trainLabels):
    numTrains = len(trainDataSet)  #训练数据组数
    numWords = len(trainDataSet[0]) #每组训练的大小
    pClass1 = sum(trainLabels)/float(numTrains) #垃圾邮件出现的概率
    p0Num = np.ones(numWords) #正常邮件分词出现频率
    p1Num = np.ones(numWords) #垃圾邮件分词出现频率
    p0SumWords = 2.0    #正常邮件中分词总数
    p1SumWords = 2.0    #垃圾邮件中分词总数
    for i in range(numTrains):
        if trainLabels[i]==1:
            p1Num += trainDataSet[i]    #统计垃圾邮件各分词
        else:
            p0Num += trainDataSet[i]    #统计正常邮件各分词
    p0SumWords = sum(p0Num)
    p1SumWords = sum(p1Num)
    p0Vect = p0Num/p0SumWords #正常邮件中各分词出现概率
    p1Vect = p1Num/p1SumWords #垃圾邮件中各分词出现概率
    return pClass1,p0Vect,p1Vect
    

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    temp0 = vec2Classify*p0Vec
    temp1 = vec2Classify*p1Vec
    temp00 = []
    temp11 = []
    for x in temp0:
        if x>0:
            temp00.append(math.log(x))
        else:
            temp00.append(0)
    for x in temp1:
        if x>0:
            temp11.append(math.log(x))
        else:
            temp11.append(0)
    p1=sum(temp11)+math.log(pClass1)
    p0=sum(temp00)+math.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    emailLabel = dataset1  #总的邮件的标签
    emailList=[]    #总的邮件的0，1向量
    for i in range(150):
        emailList.append(setOfWords2Vec(vocabList,dataset[i]))
    #我们有150个数据，其中100个正常邮件，50个垃圾邮件
    #采用三折交叉验证，将其分成三份，每份33个正常邮件和16个垃圾邮件
    #两份作为训练集，一份作为样本集
    theScaleOfHam=[i for i in range(100)]
    theScaleOfSpam=[i for i in range(100,150)]
    trainHamIndex =[]  #训练样本-正常邮件
    trainHamIndex = random.sample(theScaleOfHam,66)
    for i in trainHamIndex:
        if i in theScaleOfHam:
            theScaleOfHam.remove(i)
    sampleHamIndex = [] #样本-正常邮件
    sampleHamIndex = random.sample(theScaleOfHam,33)
    trainSpamIndex = [] #训练样本-垃圾邮件
    trainSpamIndex = random.sample(theScaleOfSpam,32)
    for i in trainSpamIndex:
        if i in theScaleOfSpam:
            theScaleOfSpam.remove(i)
    sampleSpamIndex = []  #样本-垃圾邮件
    sampleSpamIndex = random.sample(theScaleOfSpam,16)
    trainDataSet = [] #训练样本
    trainLabels = []  #训练样本标签
    sampleDataSet = []  #样本
    sampleLabels = []   #样本标签
    for index in trainHamIndex:
        trainDataSet.append(emailList[index])
        trainLabels.append(emailLabel[index])
    for index in trainSpamIndex:
        trainDataSet.append(emailList[index])
        trainLabels.append(emailLabel[index])
    for index in sampleHamIndex:
        sampleDataSet.append(emailList[index])
        sampleLabels.append(emailLabel[index])
    for index in sampleSpamIndex:
        sampleDataSet.append(emailList[index])
        sampleLabels.append(emailLabel[index])
    pClass1,p0Vect,p1Vect = trainNB(trainDataSet,trainLabels)
    errorCountHamToSpam = 0.0  #将正常邮件标记为垃圾邮件的数量
    errorCountSpamToHam = 0.0  #将垃圾邮件标记为正常邮件的数量
    for index in  range(33):
        if classifyNB(sampleDataSet[index],p0Vect,p1Vect,pClass1) != sampleLabels[index]:
            errorCountHamToSpam += 1
    for index in range(33,49):
        if classifyNB(sampleDataSet[index],p0Vect,p1Vect,pClass1) != sampleLabels[index]:
            errorCountSpamToHam += 1
    #根据文献《贝叶斯算法在校园留言版垃圾过滤中的应用研究》中所说的准确率，召回率，综合指标，我们有如下计算
    #准确率=提取的正确信息数/提取的信息数=1-提取错误信息数/提取的信息数
    #召回率=提取出的正确信息数/样本中的信息数
    #综合指标F=2*(准确率*召回率)/(准确率+召回率)
    global Percision,Recall,General
    Percision = 1 - (errorCountSpamToHam/(errorCountHamToSpam+16-errorCountSpamToHam))     
    print("准确率为：",Percision)
    Recall = 1 - (float(errorCountSpamToHam)/16)
    print("召回率为：",Recall)
    General = 2*Percision*Recall/(Percision+Recall)
    print("综合值为：",General,"\n")
    #测试用print("errorCountHamToSpam",errorCountHamToSpam)
    #测试用print("errorCountSpamToHam",errorCountSpamToHam)

if __name__=='__main__':
    i = int(input("请输入要循环的次数:"))
    for x in range(i):
        testingNB()
        Percision_All += Percision
        Recall_All += Recall
        General_All += General
    print("平均准确率为：",Percision_All/i)
    print("平均召回率为：",Recall_All/i)
    print("平均综合值为：",General_All/i)
    os.system("pause")
    
