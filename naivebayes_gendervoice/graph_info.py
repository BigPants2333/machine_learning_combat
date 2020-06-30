import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import voice_bayes as vb
from sklearn_NB import sk_Gaussian
from sklearn_NB import sk_once

def graph_Gaussian(file_name):
    """
    :param
    file_name:文件名
    """
    repeat_times=[1,5,10,50,100,500,1000]  #重复次数(趋势)
    #repeat_times=np.linspace(100,1000,10,endpoint=True)  #重复次数(平滑)
    male=[]  #男性正确率
    female=[]  #女性正确率
    total=[]  #总正确率

    #重复训练
    total_time_start=time.time()
    for i in range(len(repeat_times)):
        tmp_male=0.0
        tmp_female=0.0
        tmp_Gaussian=0.0
        cal_num=0.0
        k=repeat_times[i]
        print('开始计算,当前重复次数k=%d'%(k))
        time_start=time.time()
        while cal_num<float(k):
            info_Gaussian=sk_Gaussian(file_name)
            tmp_male+=info_Gaussian[0]
            tmp_female+=info_Gaussian[1]
            tmp_Gaussian+=info_Gaussian[2]
            
            cal_num+=1.0
            print('当前进度：%0.2f%%'%(cal_num*100.0/float(k)),end='\r')
            
        male.append(tmp_male/float(k))
        female.append(tmp_female/float(k))
        total.append(tmp_Gaussian/float(k))
        time_end=time.time()
        print('\n计算耗时',time_end-time_start,'s')
    total_time_end=time.time()
    print('计算总耗时',total_time_end-total_time_start,'s')
    
    #正确率数据转化为保留两位的百分数
    for i in range(len(repeat_times)):
        male[i]=round(male[i]*100,2)
        female[i]=round(female[i]*100,2)
        total[i]=round(total[i]*100,2)
    
    #正确率绘图
    plt.plot(repeat_times,male,label='Male',marker='*')
    plt.plot(repeat_times,female,label='Female',marker='^')
    plt.plot(repeat_times,total,label='Total',marker='.')
        
    plt.xlabel('number of repetitions')
    plt.ylabel('precision(%)')
    plt.title('The Precision of Gaussian Naive Bayes')
    
    #显示数据点的值
    for a,b in zip(repeat_times,male):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    for a,b in zip(repeat_times,female):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
    for a,b in zip(repeat_times,total):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    
    plt.legend()
    plt.show()

def graph_n_tendency(file_name,k):
    """
    :param
    file_name:文件名
    k:重复次数
    """
    n=list(range(2,2000))
    cal_num=0.0
    amount_num=float(k*len(n))
    temp=[float(k)]*6
    result_rate_all=[]

    print('开始计算,当前重复次数k=%d'%(k))
    time_start=time.time()
    for i in range(len(n)):
        avg_result=[0.0]*6
        loop_num=0.0
        while loop_num<float(k):
            loop_num+=1
            avg_result=np.sum([avg_result,vb.find_n_bayes(file_name,n[i])],axis=0)
            cal_num+=1
            print('当前进度：%0.2f%%'%(cal_num*100.0/amount_num),end='\r')
            #print(avg_result)  #打印重复k次后的各项总和
        avg_result=[a/b for a,b in zip(avg_result,temp)]
        #print(avg_result)  #打印平均值
        result_rate_all.append(avg_result)
    #print(result_rate_all)  #打印所有n值下的数据
    time_end=time.time()
    print('\n计算耗时',time_end-time_start,'s')
    
    male_accuracy=[]
    male_error=[]
    female_accuracy=[]
    female_error=[]
    total_accuracy=[]
    total_error=[]

    for i in range(len(result_rate_all)):
        male_accuracy.append(round(result_rate_all[i][0]*100,2))
        male_error.append(round(result_rate_all[i][1]*100,2))
        female_accuracy.append(round(result_rate_all[i][2]*100,2))
        female_error.append(round(result_rate_all[i][3]*100,2))
        total_accuracy.append(round(result_rate_all[i][4]*100,2))
        total_error.append(round(result_rate_all[i][5]*100,2))

    #正确率绘图
    plt.plot(n,male_accuracy,label='Male')
    plt.plot(n,female_accuracy,label='Female')
    plt.plot(n,total_accuracy,label='Total')
        
    plt.xlabel('the value of n')
    plt.ylabel('accuracy(%)')
    plt.title('The Accuracy with Different n\nnumber of repetitions:%d'%(k))
    
    plt.legend()
    plt.show()
    

def graph_n(file_name,k):
    """
    :param
    file_name:文件名
    k:重复次数
    """
    n=list(range(2,16))
    cal_num=0.0
    amount_num=float(k*len(n))
    temp=[float(k)]*6
    result_rate_all=[]

    print('开始计算,当前重复次数k=%d'%(k))
    time_start=time.time()
    for i in range(len(n)):
        avg_result=[0.0]*6
        loop_num=0.0
        while loop_num<float(k):
            loop_num+=1
            avg_result=np.sum([avg_result,vb.find_n_bayes(file_name,n[i])],axis=0)
            cal_num+=1
            print('当前进度：%0.2f%%'%(cal_num*100.0/amount_num),end='\r')
            #print(avg_result)  #打印重复k次后的各项总和
        avg_result=[a/b for a,b in zip(avg_result,temp)]
        #print(avg_result)  #打印平均值
        result_rate_all.append(avg_result)
    #print(result_rate_all)  #打印所有n值下的数据
    time_end=time.time()
    print('\n计算耗时',time_end-time_start,'s')
    
    male_accuracy=[]
    male_error=[]
    female_accuracy=[]
    female_error=[]
    total_accuracy=[]
    total_error=[]

    for i in range(len(result_rate_all)):
        male_accuracy.append(round(result_rate_all[i][0]*100,2))
        male_error.append(round(result_rate_all[i][1]*100,2))
        female_accuracy.append(round(result_rate_all[i][2]*100,2))
        female_error.append(round(result_rate_all[i][3]*100,2))
        total_accuracy.append(round(result_rate_all[i][4]*100,2))
        total_error.append(round(result_rate_all[i][5]*100,2))

    #正确率绘图
    plt.plot(n,male_accuracy,label='Male',marker='*')
    plt.plot(n,female_accuracy,label='Female',marker='^')
    plt.plot(n,total_accuracy,label='Total',marker='.')
        
    plt.xlabel('the value of n')
    plt.ylabel('accuracy(%)')
    plt.title('The Accuracy with Different n\nnumber of repetitions:%d'%(k))
    
    #显示数据点的值
    for a,b in zip(n,male_accuracy):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    for a,b in zip(n,female_accuracy):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
    for a,b in zip(n,total_accuracy):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    
    plt.legend()
    plt.show()
    
    #错误率绘图
    plt.plot(n,male_error,label='Male',marker="*")
    plt.plot(n,female_error,label='Female',marker="^")
    plt.plot(n,total_error,label='Total',marker='.')
        
    plt.xlabel('the value of n')
    plt.ylabel('error rate(%)')
    plt.title('The Error Rate with Different n\nnumber of repetitions:%d'%(k))
    
    #显示数据点的值
    for a,b in zip(n,male_error):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
    for a,b in zip(n,female_error):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    for a,b in zip(n,total_error):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    
    plt.legend()
    plt.show()

def graph_data(file_name):
    """
    :param
    file_name:文件名
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
        
        #统计无数据缺失的特征
        data_mat=np.array(data_mat).astype(float)
        count_vector=np.count_nonzero(data_mat,axis=0)
        
        #求每一个特征的平均值
        sum_vector=np.sum(data_mat,axis=0)
        mean_vector=sum_vector/count_vector

        #得到特征数据缺失的情况
        missing_data=[len(list_class)]*num
        missing_data=[a-b for a,b in zip(missing_data,count_vector)]
        
        #绘制柱状图
        label_name.pop()
        plt.bar(label_name,missing_data)
        plt.xticks(rotation=45)
        plt.xlabel('features')
        plt.ylabel('number of missing')
        plt.title('Missing Data for Each Feature')
        for a,b in zip(label_name,missing_data):
            plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
        plt.show()
        
        missing_index=[]
        for i in range(len(missing_data)):
            if missing_data[i]==0:
                continue
            else:
                missing_index.append(i)
        
        gender_count=[]
        for i in range(len(missing_index)):
            gender_num=[0.0,0.0]
            for j in range(len(list_class)):
                if data_mat[j][missing_index[i]]==0:
                    if list_class[j]==1:
                        gender_num[0]+=1
                    else:
                        gender_num[1]+=1
            gender_count.append(gender_num)
        #print(gender_count)  #打印统计情况
        
        #绘制饼状图
        name_str=['male','female']
        cols=['c','r']
        for i in range(len(missing_index)):
            plt.pie(gender_count[i],labels=name_str,colors=cols,startangle=90,
                shadow=True,explode=(0,0.1),autopct='%1.1f%%')
            plt.title('The Distribution of Missing Data\nfeature:%s\ntotal num:%d'
                    %(label_name[missing_index[i]],missing_data[missing_index[i]]))
            plt.show()
        
        miss_male=[]
        miss_female=[]
        for i in range(len(missing_index)):
            tmp_male=[]
            tmp_female=[]
            for j in range(len(list_class)):
                if list_class[j]==1:
                    tmp_male.append(data_mat[j][missing_index[i]])
                else:
                    tmp_female.append(data_mat[j][missing_index[i]])
            #tmp_male.sort()
            miss_male.append(tmp_male)
            #tmp_female.sort()
            miss_female.append(tmp_female)
        
        #绘制散点图
        human_num=range(int(len(list_class)/2))
        for i in range(len(missing_index)):
            plt.scatter(human_num,miss_male[i],label='Male',s=10)
            plt.scatter(human_num,miss_female[i],label='Female',s=10)
            plt.hlines(mean_vector[missing_index[i]],0,int(len(list_class)/2)-1,label='average')
            plt.title('The Data Distribution of %s'%(label_name[missing_index[i]]))
            plt.legend()
            plt.show()
        
def graph_discretization_once(file_name,n):
    """
    :param
    file_name:文件名
    n:离散程度
    """
    evaluation,p_1_class=vb.test_bayes(file_name,n)

    p_1_class=round(p_1_class*100.0,2)
    for i in range(len(evaluation)):
        for j in range(len(evaluation[i])):
            evaluation[i][j]=round(evaluation[i][j]*100.0,2)
    
    #绘制性能指标柱状图
    str_info=['precision','error rate','recall','F1 score']
    label_info=['male','female']
    col_info=['b','g']
    for i in range(len(evaluation)-1):
        plt.bar(str_info,evaluation[i],color=col_info[i])
        plt.xlabel('evaluation indicator')
        plt.ylabel('value(%)')
        plt.title('The test result of %s voice\nmale in training is%.2f%%'%(label_info[i],p_1_class))
        for a,b in zip(str_info,evaluation[i]):
            plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
        plt.show()
    plt.bar(str_info[0:2],evaluation[2],color='y')
    plt.xlabel('evaluation indicator')
    plt.ylabel('value(%)')
    plt.title('The test result of total issues')
    for a,b in zip(str_info[0:2],evaluation[2]):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
    plt.show()

def graph_discretization(file_name,n):
    """
    :param
    file_name:文件名
    n:离散程度
    """
    #repeat_times=[1,5,10,50,100,500,1000]  #重复次数(趋势)
    repeat_times=np.linspace(100,1000,10,endpoint=True)  #重复次数(平滑)
    male=[]  #男性正确率
    female=[]  #女性正确率
    total=[]  #总正确率

    #重复训练
    total_time_start=time.time()
    for i in range(len(repeat_times)):
        tmp_male=0.0
        tmp_female=0.0
        tmp_discretization=0.0
        cal_num=0.0
        k=repeat_times[i]
        print('开始计算,当前重复次数k=%d'%(k))
        time_start=time.time()
        while cal_num<float(k):
            evaluation,p_1_class=vb.test_bayes(file_name,n)
            tmp_male+=evaluation[0][0]
            tmp_female+=evaluation[1][0]
            tmp_discretization+=evaluation[2][0]
            cal_num+=1.0
            print('当前进度：%0.2f%%'%(cal_num*100.0/float(k)),end='\r')
            
        male.append(tmp_male/float(k))
        female.append(tmp_female/float(k))
        total.append(tmp_discretization/float(k))
        time_end=time.time()
        print('\n计算耗时',time_end-time_start,'s')
    total_time_end=time.time()
    print('计算总耗时',total_time_end-total_time_start,'s')
    
    #正确率数据转化为保留两位的百分数
    for i in range(len(repeat_times)):
        male[i]=round(male[i]*100,2)
        female[i]=round(female[i]*100,2)
        total[i]=round(total[i]*100,2)
    
    #正确率绘图
    plt.plot(repeat_times,male,label='Male',marker='*')
    plt.plot(repeat_times,female,label='Female',marker='^')
    plt.plot(repeat_times,total,label='Total',marker='.')
        
    plt.xlabel('number of repetitions')
    plt.ylabel('precision(%)')
    plt.title('The Precision of Multinomial Naive Bayes after Discretization')
    
    #显示数据点的值
    for a,b in zip(repeat_times,male):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    for a,b in zip(repeat_times,female):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
    for a,b in zip(repeat_times,total):
        plt.text(a,b,b,ha='center',va='top',fontsize=10)
    
    plt.legend()
    plt.show()
    

def graph_contrast(file_name,n,k):
    """
    :param
    file_name:文件名
    n:离散程度
    k:重复次数
    """
    info_Gaussian=info_Multinomial=info_discretization=[[0.0]*3]*2
    tmp=[[float(k)]*3]*2
    cal_num=0.0

    print('开始计算,当前重复次数k=%d'%(k))
    time_start=time.time()

    #开始重复训练
    while cal_num<float(k):
        #调用sklearn库中高斯贝叶斯和多项式贝叶斯的性能评估
        Gaussian,Multinomial=sk_once(file_name)
        #将数据离散化后使用多项式贝叶斯的性能评估
        evaluation,p_1_class=vb.test_bayes(file_name,n)
        #将离散化后使用多项式贝叶斯的性能评估中所需项提取出来
        discretization=[]
        tmp_male=[]
        tmp_female=[]
        tmp_male.append(evaluation[0][0]),tmp_male.append(evaluation[0][2]),tmp_male.append(evaluation[0][3])
        tmp_female.append(evaluation[1][0]),tmp_female.append(evaluation[1][2]),tmp_female.append(evaluation[1][3])
        discretization.append(tmp_male),discretization.append(tmp_female)
        #求和
        info_Gaussian=np.sum([info_Gaussian,Gaussian],axis=0)
        info_Multinomial=np.sum([info_Multinomial,Multinomial],axis=0)
        info_discretization=np.sum([info_discretization,discretization],axis=0)
        cal_num+=1
        print('当前进度：%0.2f%%'%(cal_num*100.0/float(k)),end='\r')
    #求均值
    info_Gaussian=[a/b for a,b in zip(info_Gaussian,tmp)]
    info_Multinomial=[a/b for a,b in zip(info_Multinomial,tmp)]
    info_discretization=[a/b for a,b in zip(info_discretization,tmp)]
    #将性能评估以百分数形式保存
    for i in range(len(info_Gaussian)):
        for j in range(len(info_Gaussian[i])):
            info_Gaussian[i][j]=round(info_Gaussian[i][j]*100.0,2)
            info_Multinomial[i][j]=round(info_Multinomial[i][j]*100.0,2)
            info_discretization[i][j]=round(info_discretization[i][j]*100.0,2)
    #重复训练结束,打印耗时
    time_end=time.time()
    print('\n计算耗时',time_end-time_start,'s')

    #开始绘图
    indicator_info=['precision','recall','F1 score']  #横轴信息
    gender_info=['Male','Female']
    for i in range(len(gender_info)):
        #使用数据绘图
        plt.plot(indicator_info,info_Gaussian[i],label='Gaussian',marker="*")
        plt.plot(indicator_info,info_Multinomial[i],label='Multinomial',marker="^")
        plt.plot(indicator_info,info_discretization[i],label='Discretization',marker=".")

        #x、y轴意义及图表标题
        plt.xlabel('evaluation indicator')
        plt.ylabel('value(%)')
        plt.title('The Mean Bayesian Performance Evaluation\ngender:%s\nrepeated times:%d'
                    %(gender_info[i],k))

        #显示数据点的值
        for a,b in zip(indicator_info,info_Gaussian[i]):
            plt.text(a,b,b,ha='center',va='top',fontsize=10)
        for a,b in zip(indicator_info,info_Multinomial[i]):
            plt.text(a,b,b,ha='center',va='top',fontsize=10)
        for a,b in zip(indicator_info,info_discretization[i]):
            plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
        
        #显示图像
        plt.legend()
        plt.show()

if __name__ == '__main__':
    repeat_times=[1,10,100,1000]
    file_name='voice.csv'  #调试时数据集路径
    
    #绘制调用sklearn库中高斯贝叶斯的正确率
    graph_Gaussian(file_name)
    
    #绘制数据缺失情况
    graph_data(file_name)
    
    #绘制n对正确率影响的趋势
    graph_n_tendency(file_name,1)
    
    #绘制不同的n值对结果的影响
    for i in repeat_times:
        graph_n(file_name,i)
    
    #绘制离散化(n=11)后情况多项式贝叶斯的正确率
    graph_discretization(file_name,11)
    
    #绘制高斯贝叶斯、多项式贝叶斯、离散化(n=11)后多项式贝叶斯的训练性能对比
    graph_contrast(file_name,11,200)
