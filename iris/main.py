#imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#boring setup
irisnames = {
  'Iris-setosa': 0,
  'Iris-versicolor': 1,
  'Iris-virginica': 2
}
columns= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df=pd.read_csv('iris/iris.data', names=columns)
data = df.values
X = data[:,0:4]
Y = data[:,4]

#learning how to use seaborn
print(df.head())
print(df.describe())
sns.pairplot(df, hue='species')
plt.show()
plt.close()

sns.relplot(df, x='sepal_length', y='sepal_width', hue='species')
plt.show()
plt.close()

#graphing averages
averages=np.empty((3,4))
for i in np.unique(Y):
  for j in range(4):
    averages[irisnames[i], j] = np.mean(X[Y == i, j])
width=0.25
plt.bar(np.arange(4), averages[0], width, label='Iris-setosa')
plt.bar(np.arange(4)+width, averages[1], width, label='Iris-versicolor')
plt.bar(np.arange(4)+2*width, averages[2], width, label='Iris-virginica')
plt.xticks(np.arange(4)+width, df.columns[:-1])
plt.ylabel('Average Value')
plt.title('Average Sepal and Petal Measurements by Species')
plt.legend()
plt.show()
plt.close()

#the ai part
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, Y_train)
predictions = svn.predict(X_test)

#see how bad this is
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))