import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def sk_Gaussian(file_name):
    """
    :param
    file_name:文件名
    :return
    info_Gaussian:男性正确率、女性正确率、高斯贝叶斯准确率
    """
    #数据集读取和预处理
    data=pd.read_csv(file_name)
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    y=LabelEncoder().fit_transform(y)
    imp=SimpleImputer(missing_values=0,strategy='mean')
    x=imp.fit_transform(x)
    
    #按照7:3的比例划分训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    
    #高斯朴素贝叶斯预测
    gnb=GaussianNB()
    gnb.fit(x_train,y_train)
    y_predict=gnb.predict(x_test)

    #计算准确率指标
    precision_male=precision_score(y_test,y_predict,pos_label=1)  #男性准确率
    precision_female=precision_score(y_test,y_predict,pos_label=0)  #女性准确率
    precision_Gaussian=gnb.score(x_test,y_test)  #高斯贝叶斯准确率
    
    #打印准确率
    #print('男性准确率:%.2f%%'%(round(precision_male*100.0,2)))
    #print('女性准确率:%.2f%%'%(round(precision_female*100.0,2)))
    #print('高斯贝叶斯准确率:%.2f%%'%(round(precision_Gaussian*100.0,2)))

    #打印预测结果评估
    #gender_labels=['female','male']  #标签
    #print(classification_report(y_test,y_predict,target_names=gender_labels))  #打印分类报告
    
    #将准确率信息打包返回
    info_Gaussian=[]
    info_Gaussian.append(precision_male)
    info_Gaussian.append(precision_female)
    info_Gaussian.append(precision_Gaussian)

    return info_Gaussian

def sk_once(file_name):
    """
    :param
    file_name:文件名
    :return
    info_Gaussian:高斯贝叶斯性能评估
    info_Multinomial:多项式贝叶斯性能评估
    """
    #数据集读取和预处理
    data=pd.read_csv(file_name)
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    y=LabelEncoder().fit_transform(y)
    imp=SimpleImputer(missing_values=0,strategy='mean')
    x=imp.fit_transform(x)
    
    #按照7:3的比例划分训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

    #高斯朴素贝叶斯预测
    gnb=GaussianNB()
    gnb.fit(x_train,y_train)
    y_predict=gnb.predict(x_test)
    
    #高斯贝叶斯评估
    male_Gaussian=[]
    female_Gaussian=[]
    info_Gaussian=[]
    #男性训练评估
    male_Gaussian.append(precision_score(y_test,y_predict,pos_label=1))
    male_Gaussian.append(recall_score(y_test,y_predict,pos_label=1))
    male_Gaussian.append(f1_score(y_test,y_predict,pos_label=1))
    #女性训练评估
    female_Gaussian.append(precision_score(y_test,y_predict,pos_label=0))
    female_Gaussian.append(recall_score(y_test,y_predict,pos_label=0))
    female_Gaussian.append(f1_score(y_test,y_predict,pos_label=0))
    #高斯贝叶斯训练评估值打包
    info_Gaussian.append(male_Gaussian)
    info_Gaussian.append(female_Gaussian)

    #多项式朴素贝叶斯预测
    mnb=MultinomialNB()
    mnb.fit(x_train,y_train)
    y_predict=mnb.predict(x_test)

    #多项式贝叶斯评估
    male_Multinomial=[]
    female_Multinomial=[]
    info_Multinomial=[]
    #男性训练评估
    male_Multinomial.append(precision_score(y_test,y_predict,pos_label=1))
    male_Multinomial.append(recall_score(y_test,y_predict,pos_label=1))
    male_Multinomial.append(f1_score(y_test,y_predict,pos_label=1))
    #女性训练评估
    female_Multinomial.append(precision_score(y_test,y_predict,pos_label=0))
    female_Multinomial.append(recall_score(y_test,y_predict,pos_label=0))
    female_Multinomial.append(f1_score(y_test,y_predict,pos_label=0))
    #多项式贝叶斯训练评估值打包
    info_Multinomial.append(male_Multinomial)
    info_Multinomial.append(female_Multinomial)

    return info_Gaussian,info_Multinomial

if __name__ == '__main__':
    file_name='voice.csv'  #调试时数据集路径
    info_Gaussian,info_Multinomial=sk_once(file_name)
    print(info_Gaussian)
    print(info_Multinomial)
    