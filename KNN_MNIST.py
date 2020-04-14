from createmat import *
from knn import *

'''
解析输入的字符，如果符合2个数字的输入，则开始测试。如果是其他格式，则返回类型为string，主程序中判断是否继续
'''	
def get_cmd_pars(cmd_str):	
	cmd_medum=[]
	pars_ret=[]
	type_ret  = 'digit'
	cmd_list = cmd_str.split(sep=' ')#切割输入的字符串
	for cl in cmd_list:#这里将空串清除
		if cl != '':
			cmd_medum.append(cl)
	for cr in cmd_medum:#这里判断所有输入的参数是否是纯数字
		if not cr.isdigit():
			type_ret = 'string'
		else:
			pars_ret.append(int(cr))
	if len(pars_ret)<2:#判断输入的参数是否大于2个数字
		type_ret = 'string'
		
	return type_ret,pars_ret
		
#################################################
if __name__ == '__main__':
	
	train_image_file = '..\\pythonwork\\data\\mnist\\train-images.idx3-ubyte'
	train_label_file = '..\\pythonwork\\data\\mnist\\train-labels.idx1-ubyte'
	test_image_file = '..\\pythonwork\\data\\mnist\\t10k-images.idx3-ubyte'
	test_label_file = '..\\pythonwork\\data\\mnist\\t10k-labels.idx1-ubyte'
	
	#选择所有图片作为训练样本。	
	train_image_mat, train_label_list  = read_image_label_all_vector(train_image_file,train_label_file)
#	test_image_mat, test_label_list  = read_image_label_all_vector(test_image_file,test_label_file)
	#选择部分数据作为训练集，第3个参数为偏移起始位置，第4个参数是训练样本数
#	train_image_mat, train_label_list  = read_image_label_vector(train_image_file,train_label_file,0,5000)

	while True:
		#-----------交互式输入控制开始-----------------
		#如果输入的样本数量为0，判断是否退出，如果不为0，继续开始分类。
		cmd = input('输入测试样本偏移和数量(比如 100 50):')
		type_ret,par_ret = get_cmd_pars(cmd)#解析输入的字符串
		if type_ret == 'digit':#如果全部为数字
			offset = par_ret[0]
			amount = par_ret[1]
			if amount == 0: 
				continue
		else: #如果不是数字，提示是否退出程序
			cmd = input('格式不正确，输入Y(y)确定要退出:')
			if cmd == 'y' or cmd == 'Y':#输入了y则表示要退出程序
				break
			continue#没有输入y表示继续循环
		#-----------交互式输入控制结束-----------------
		
		#根据前面的输入偏移和数量，开始读出测试样本
		test_image_mat, test_label_list  = read_image_label_vector(test_image_file,test_label_file,offset,amount)
			
		#开始分类
		err_count = 0.0#记录错误数量
		for i in range(len(test_image_mat)):
			print('当前进度：%2.2f%%'%(100.0*i/len(test_image_mat)))
			#利用knn算法进行分类
			class_result = knn_classify(test_image_mat[i], train_image_mat, train_label_list, 5)#计算分类结果
			print( "第 %d 张图片, 分类器结果: %d, 实际值: %d" % (i,class_result, test_label_list[i]),end=' ')
			#判断分类结果是发和标签一致
			if (class_result != test_label_list[i]): 
				print(' 分类错误！',end = ' ')
				err_count += 1.0
			#打印错误率
			print('当前错误率：%2.2f%%' % (100.0*err_count/(i+0.01)))
			
		print( "\n总错误数: %d" % err_count)
		print( "总错误率: %2.2f%%" % (100.0*err_count/len(test_image_mat)))
