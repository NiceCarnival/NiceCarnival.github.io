

泰坦尼克号乘客数据查验

import pandas as pd
#利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#观察前几行数据，可以发现，数据种类各异，数值型，类别型，甚至还有缺失数据
titanic.head()




#使用pandas，数据都转入pandas独有的dataframe格式（二位数据表格），直接使用info（），查看数据的统计特性
titanic.info()


使用决策树模型预测泰坦尼克号乘客的生还情况

#特征的选择。根据对这场事故的了解，sex，age，pclass这些特征都有可能是决定幸免的关键因素
x = titanic[['pclass','age','sex']]
y = titanic['survived']
#对当前选择的特征进行探查
x.info()




#补充age的数据，使用平均数或中位数都是对模型偏离造成最小影响的策略
x['age'].fillna(x['age'].mean(), inplace = True)
#对补完的数据重新探查
x.info()




#数据分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=33)
#使用scikit-learn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
#转换特征后，发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
x_train = vec.fit_transform(x_train.to_dict(orient = 'record'))
print(vec.feature_names_)




#对测试数据的特征进行转换
x_test = vec.transform(x_test.to_dict(orient = 'record'))
#导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
#使用分割到的训练数据进行模型学习
dtc.fit(x_train, y_train)



#用训练好的决策树模型对测试特征数据进行预测
y_predict = dtc.predict(x_test)
#决策树模型对泰坦尼克号乘客是否生还的预测性能
from sklearn.metrics import classification_report
#输出预测准确性
print(dtc.score(x_test,y_test))



#输出更加详细的分类性能vived
print(classification_report(y_predict, y_test, target_names = ['died', 'survived']))
