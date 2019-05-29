import pandas as pd

#利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#人工选取pclass,age,sex作为判别乘客是否生还的的特征
x = titanic[['pclass','age','sex']]
y = titanic['survived']#补充age的数据，使用平均数都是对模型偏离造成最小影响的策略
x['age'].fillna(x['age'].mean(), inplace = True)

#对原始数据进行分割，25%的乘客数据用于测试
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=33)

#对类别型特征进行转化，成为特征向量
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient = 'record'))
x_test = vec.transform(x_test.to_dict(orient = 'record'))

#使用单一决策树进行模型训练及预测分析
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)

#使用随机森林分类器进行集成模型的训练及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc_y_pred = rfc.predict(x_test)

#使用梯度提升决策树进行集成模型的训练及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
gbc_y_pred = gbc.predict(x_test)


集成模型对泰坦尼克号乘客是否生还预测性能



from sklearn.metrics import classification_report
#输出单一决策树在测试集上的分类准确性，以及更详细的准确率，召回率，F1指标
print('the accuracy of decision tree is',dtc.score(x_test,y_test))
print(classification_report(dtc_y_pred,y_test))


#输出梯度提升在测试集上的分类准确性，以及更详细的准确率，召回率，F1指标
print('the accuracy of decision tree boosting is',gbc.score(x_test,y_test))
print(classification_report(gbc_y_pred,y_test))
