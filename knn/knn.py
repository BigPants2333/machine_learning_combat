from numpy import *
import operator

def knn_classify(test_data, train_dataset, train_label, k):
	train_dataset_amount = train_dataset.shape[0]#行数，也即训练样本的的个数，shape[1]为列数
	#print('train_label:',train_label)
	#print('train_dataset:',train_dataset)
	#将输入test_data变成了和train_dataset行列数一样的矩阵
	test_rep_mat =  tile(test_data, (train_dataset_amount,1))#tile(mat,(x,y)) Array类 mat 沿着行重复x次，列重复y次
	diff_mat = test_rep_mat - train_dataset
	#print('diff_mat:',diff_mat)
	#求平方，为后面求距离准备
	sq_diff_mat = diff_mat**2  
	#print('sq_diff_mat:',sq_diff_mat)
	#将平方后的数据相加，sum(axis=1)是将一个矩阵的每一行向量内的数据相加，得到一个list，list的元素个数和行数一样;sum(axis=0)表示按照列向量相加
	sq_dist = sq_diff_mat.sum(axis=1)
	#print('sq_dist:',sq_dist)
	#开平方，得到欧式距离
	distance = sq_dist**0.5
	#print('distance:',distance)
	
	#argsort 将元素从小到大排列，得到这个数组元素在distance中的index(索引)，dist_index元素内容是distance的索引
	dist_index = distance.argsort()	 
	#print('dist_index:',dist_index)
	
	class_count={}		  
	for i in range(k):
		label = train_label[dist_index[i]]
		#如果属于某个类，在该类的基础上加1，相当于增加其权重，如果不是某个类则新建字典的一个key并且等于1
		class_count[label] = class_count.get(label,0) + 1
	#降序排列
	class_count_list = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
	print('排序后的分类结果：',class_count_list)
	return class_count_list[0][0]


############################################################
if __name__ == '__main__':
	print('\n\n')
	train_data_set=array([[2.2,1.4],\
		[2.4,2.3],\
		[1.1,3.4],\
		[8.3,7.3],\
		[9.2,8.3],\
		[10.2,11.1],\
		[11.2,9.3]])
	train_label = ['A','A','A','B','B','B','B']
	test_data = [4.6,3.4]
	print('分类结果为：',knn_classify(test_data,train_data_set,train_label,3))
	