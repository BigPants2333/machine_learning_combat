import csv
import time
import os
import numpy as np
import matplotlib.pyplot as plt

def load_data_set(file_name,n):
    """
    :param
    file_name:文件名
    n:对浮点数的特征值进行离散化的离散程度

    :return
    train_mat:离散化的训练数据集
    train_classes:训练数据集所属的分类(male/female)
    test_mat:离散化的测试数据集
    test_classes:测试数据集所属的分类(male/female)
    label_name:特征的名称
    """
    data_mat=[]
    with open(file_name)as file_obj:
        voice_reader=csv.DictReader(file_obj)
        list_class=[]
        #文件头
        label_name=list(voice_reader.fieldnames)
        #print(label_name)  #打印特征名称
        num=len(label_name)-1
        #print(num)  #打印特征个数

        for line in voice_reader.reader:
            data_mat.append(line[:num])
            gender=1 if line[-1]=='male' else 0
            list_class.append(gender)
        #print(list_class)  #打印性别分类(male->1,female->0)

        #求每一个特征的平均值
        data_mat=np.array(data_mat).astype(float)
        #print(data_mat)  #打印原始数据集
        count_vector=np.count_nonzero(data_mat,axis=0)
        sum_vector=np.sum(data_mat,axis=0)
        mean_vector=sum_vector/count_vector

        #数据缺失的地方用平均值填充
        for row in range(len(data_mat)):
            for col in range(num):
                if data_mat[row][col]==0.0:
                    data_mat[row][col]=mean_vector[col]
        #print(data_mat)  #打印处理缺失数据后的数据集

        #将数据连续型的特征值离散化处理
        min_vetor=data_mat.min(axis=0)
        max_vetor=data_mat.max(axis=0)
        diff_vector=max_vetor-min_vetor
        diff_vector/=(n-1)

        new_data_set=[]
        for i in range(len(data_mat)):
            line=np.array((data_mat[i]-min_vetor)/diff_vector).astype(int)
            new_data_set.append(line)

        #随机划分数据集为训练集和测试集
        test_set=list(range(len(new_data_set)))
        train_set=[]
        #从3168个数据集中抽取70%(2218个)的数据做为训练集,剩余为测试集
        for i in range(2218):
            random_index=int(np.random.uniform(0,len(test_set)))
            train_set.append(test_set[random_index])
            del test_set[random_index]
        
        #print(train_set)  #打印训练集序号
        #print(test_set)  #打印测试集序号

        #训练数据集
        train_mat=[]
        train_classes=[]
        for index in train_set:
            train_mat.append(new_data_set[index])
            train_classes.append(list_class[index])
        #print(train_mat)  #打印训练集
        #print(train_classes)  #打印训练集分类

        #测试数据集
        test_mat=[]
        test_classes=[]
        for index in test_set:
            test_mat.append(new_data_set[index])
            test_classes.append(list_class[index])
        #print(test_mat)  #打印测试集
        #print(test_classes)  #打印测试集分类

    return train_mat,train_classes,test_mat,test_classes,label_name

def naive_bayes(train_matrix,list_classes,n):
    """
    :param
    train_matrix:训练样本矩阵
    list_classes:训练样本分类向量
    n:对浮点数的特征值进行离散化的离散程度

    :return
    p_1_class:任一样本分类为1(male)的概率
    p_1_feature:分类为1(male)情况下的各特征的概率
    p_feature:所有分类下各特征的概率
    """
    #训练样本个数,特征个数
    num_train_data=len(train_matrix)
    num_feature=len(train_matrix[0])
    #分类为1的样本占比
    p_1_class=sum(list_classes)/float(num_train_data)
    #print(num_train_data)  #打印训练样本个数
    #print(num_feature)  #打印特征个数
    #print(p_1_class)  #打印分类为1的概率

    list_classes_1=[]
    train_data_1=[]

    for i in list(range(num_train_data)):
        if list_classes[i]==1:
            list_classes_1.append(i)
            train_data_1.append(train_matrix[i])
    
    #分类为1情况下的各特征的概率
    train_data_1=np.matrix(train_data_1)
    p_1_feature={}
    for i in list(range(num_feature)):
        feature_values=np.array(train_data_1[:,i]).flatten()
        #避免某些特征值概率为0而影响总体概率,每个特征值最少个数为1
        feature_values=feature_values.tolist()+list(range(n))
        p={}
        count=len(feature_values)
        for value in set(feature_values):
            #多个概率相乘结果较小,取对数减小四舍五入的误差
            p[value]=np.log(feature_values.count(value)/float(count))
        p_1_feature[i]=p
    #print(p_1_feature)  #打印分类为1情况下的各特征的概率取对数后的结果

    #所有分类下的各特征的概率
    p_feature={}
    train_matrix=np.matrix(train_matrix)
    for i in list(range(num_feature)):
        feature_values=np.array(train_matrix[:,i]).flatten()
        feature_values=feature_values.tolist()+list(range(n))
        p={}
        count=len(feature_values)
        for value in set(feature_values):
            #多个概率相乘结果较小,取对数减小四舍五入的误差
            p[value]=np.log(feature_values.count(value)/float(count))
        p_feature[i]=p
    #print(p_feature)  #打印所有分类下的各特征的概率取对数后的结果

    return p_1_class,p_1_feature,p_feature

def classify_bayes(test_vector,p_1_class,p_1_feature,p_feature):
    """
    :param
    test_vector:需要分类的测试向量
    p_1_class:任一样本分类为1(male)的概率
    p_1_feature:分类为1(male)情况下的各特征的概率
    p_feature:所有分类下各特征的概率

    :return 一个数字(1->male,0->female)
    """
    #计算每个分类的概率(概率相乘取对数=概率各自对数相加)
    sum=0.0
    for i in list(range(len(test_vector))):
        sum+=p_1_feature[i][test_vector[i]]
        sum-=p_feature[i][test_vector[i]]
    p1=sum+np.log(p_1_class)
    p1=np.exp(p1)
    p0=1-p1
    if p1>p0:
        return 1
    else:
        return 0

def find_n_bayes(file_name,n):
    """
    :param
    file_name:文件名
    n:对浮点数的特征值进行离散化的离散程度
    
    :return
    result_rate:一个包含各种比率的list
                元素依次为：男性正确率,男性错误率,女性正确率,女性错误率,总正确率,总错误率
    """
    #建立训练集与测试集
    train_mat,train_classes,test_mat,test_classes,label_name=load_data_set(file_name,n)
    #得到贝叶斯参数
    p_1_class,p_1_feature,p_feature=naive_bayes(train_mat,train_classes,n)
    
    result_info=[0.0]*14
    for i in list(range(len(test_mat))):
        test_vector=test_mat[i]
        result=classify_bayes(test_vector,p_1_class,p_1_feature,p_feature)
        if test_classes[i]==1:
            result_info[0]+=1  #男性人数
            if result==test_classes[i]:
                result_info[1]+=1  #正确人数
            else:
                result_info[2]+=1  #错误人数
        else:
            result_info[5]+=1  #女性人数
            if result==test_classes[i]:
                result_info[6]+=1  #正确人数
            else:
                result_info[7]+=1  #错误人数
        result_info[10]+=1
    
    result_info[3]=result_info[1]/result_info[0]  #男性正确率
    result_info[4]=result_info[2]/result_info[0]  #男性错误率
    result_info[8]=result_info[6]/result_info[5]  #女性正确率
    result_info[9]=result_info[7]/result_info[5]  #女性错误率
    
    result_info[11]=(result_info[1]+result_info[6])/result_info[10]  #总正确率
    result_info[12]=(result_info[2]+result_info[7])/result_info[10]  #总错误率

    rate_index=[3,4,8,9,11,12]

    """
    #打印测试结果
    info_str=['男性人数','男性识别正确个数','男性识别错误个数','男性正确率','男性错误率',
              '女性人数','女性识别正确个数','女性识别错误个数','女性正确率','女性错误率',
              '总人数','正确率','错误率']
    #打印数据
    for i in rate_index:
        print('%s:%.2f%%'%(info_str[i],result_info[i]*100.0))
    """

    result_rate=[]
    for i in range(6):
        result_rate.append(result_info[rate_index[i]])
    return result_rate

def test_bayes(file_name,n):
    """
    :param
    file_name:文件名
    n:对浮点数的特征值进行离散化的离散程度

    :return
    evaluation:评估指标
    p_1_class:任一样本分类为1(male)的概率
    """
    #建立训练集与测试集
    train_mat,train_classes,test_mat,test_classes,label_name=load_data_set(file_name,n)
    #得到贝叶斯参数
    p_1_class,p_1_feature,p_feature=naive_bayes(train_mat,train_classes,n)

    count_info=[0.0]*9
    for i in list(range(len(test_mat))):
        test_vector=test_mat[i]
        result=classify_bayes(test_vector,p_1_class,p_1_feature,p_feature)
        if test_classes[i]==1:
            count_info[0]+=1  #男性人数
            if result==test_classes[i]:
                count_info[1]+=1  #正确人数
            else:
                count_info[2]+=1  #错误人数
        else:
            count_info[3]+=1  #女性人数
            if result==test_classes[i]:
                count_info[4]+=1  #正确人数
            else:
                count_info[5]+=1  #错误人数
        count_info[6]+=1
    count_info[7]=count_info[1]+count_info[4]  #总正确人数
    count_info[8]=count_info[2]+count_info[5]  #总错误人数

    #评估指标计算
    evaluation=[]
    male_eva=[]  #男性评估指标
    male_eva.append(count_info[1]/count_info[0])  #男性正确率
    male_eva.append(count_info[2]/count_info[0])  #男性错误率
    male_eva.append(count_info[1]/(count_info[1]+count_info[5]))  #男性召回率
    male_eva.append(2*male_eva[0]*male_eva[2]/(male_eva[0]+male_eva[2]))  #男性F1值
    female_eva=[]  #女性评估指标
    female_eva.append(count_info[4]/count_info[3])  #女性正确率
    female_eva.append(count_info[5]/count_info[3])  #女性错误率
    female_eva.append(count_info[4]/(count_info[4]+count_info[2]))  #女性召回率
    female_eva.append(2*female_eva[0]*female_eva[2]/(female_eva[0]+female_eva[2]))  #女性F1值
    total_eva=[]  #总体评估指标
    total_eva.append(count_info[7]/count_info[6])  #总正确率
    total_eva.append(count_info[8]/count_info[6])  #总错误率

    evaluation.append(male_eva)
    evaluation.append(female_eva)
    evaluation.append(total_eva)
    
    return evaluation,p_1_class


if __name__ == '__main__':
    #数据集本地路径 'C:\\Users\\t1762\\Documents\\pythonwork\\naivevoice\\voice.csv'
    file_name='voice.csv'  #调试时数据集路径
    n=11
    evaluation,p_1_class=test_bayes(file_name,n)
    print('离散程度n=%d,训练集中男性占比%.2f%%'%(n,p_1_class*100.0))
    print('男性正确率:%.2f%%,男性错误率:%.2f%%'%(evaluation[0][0]*100.0,evaluation[0][1]*100.0))
    print('女性正确率:%.2f%%,女性错误率:%.2f%%'%(evaluation[1][0]*100.0,evaluation[1][1]*100.0))
    print('总正确率:%.2f%%,总错误率:%.2f%%'%(evaluation[2][0]*100.0,evaluation[2][1]*100.0))
    os.system('pause')
    