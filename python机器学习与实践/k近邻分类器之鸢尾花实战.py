读取iris数据集细节资料

#导入iris数据加载器
from sklearn.datasets import load_iris

#使用加载器读取数据并存入变量iris
iris = load_iris()
#查验数据规模
iris.data.shape
#查看数据说明。
print(iris.DESCR)


对数据集进行分割


#导入train_test_split用于数据分割
from sklearn.model_selection import train_test_split
#使用train_test_split，利用随机种子random_state采样25%的数据作为测试集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 33)

使用k近邻分类器对鸢尾花数据进行类别预测

#导入数据标准化模块
from sklearn.preprocessing import StandardScaler
#导入k近邻分类器
from sklearn.neighbors import KNeighborsClassifier
#对训练和测试的特征数据进行标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

对k近邻分类器在鸢尾花数据上的预测性能进行评估

#使用k近邻分类器对测试数据进行类别预测，预测结果储存在变量y_predict
knc = KNeighborsClassifier()
knc.fit(x_train,y_train)
y_predict = knc.predict(x_test)
#使用模型自带的评估函数进行准确性测评
print('The Accuracy of K-Nearest Neighbor Classifier is', knc.score(x_test, y_test))
#使用classification_report模块对预测结果做更详细的分析
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = iris.target_names))