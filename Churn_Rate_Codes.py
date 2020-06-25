from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data=pd.read_csv(r"C:\\Users\\Lenovo\\Desktop\\Data_Mining\\train.csv")

#Looking at the missing values
print(data.head())
print(data.describe().T)

#Drop these columns because there are too many missing values.
data.drop(["credit_score", "rewards_earned"], axis=1, inplace=True)

#We dropped all of the missing rows
data.dropna(inplace=True)
dataeren=data["user"]
#Dropping cat variables from the data set.
data2=data.drop(["zodiac_sign","payment_type", "housing","user","churn"],axis=1)

#Visualizing the numerical variables
for i, col in enumerate(data2.columns):
    plt.figure(i)
    sn.distplot(data2[col])

#Correlation between variables

fig, ax = plt.subplots(figsize=(10, 10)) 
mask = np.zeros_like(data2.corr())
mask[np.triu_indices_from(mask)] = 1
print(sn.heatmap(data2.corr(), mask= mask, ax= ax))

#Dropping uncessary or highly correlated featues
data.drop(["app_web_user"],axis=1, inplace=True)
data.drop(["user"],axis=1, inplace=True)

#Encoding the data
data=pd.get_dummies(data)
data.drop(["housing_na", "payment_type_na", "zodiac_sign_na"],axis=1, inplace=True)

#Pre-Modeling
x=data.drop(["churn"],axis=1)
y=data ["churn"]

#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state=40)

#Scaling the data
st=StandardScaler()

x_train_scaled= st.fit_transform(x_train)
x_train_scaled=pd.DataFrame(x_train_scaled)
x_train_scaled.columns=x_train.columns

x_test_scaled= st.transform(x_test)
x_test_scaled=pd.DataFrame(x_test_scaled)
x_test_scaled.columns=x_test.columns

#Modeling, fitting and predicting values
model=LogisticRegression(random_state=0)
model.fit(x_train_scaled, y_train)
y_pred=(model.predict(x_test_scaled))


from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score
#Metrics to measure how good the model is
cm=confusion_matrix (y_test, y_pred)
ac=accuracy_score (y_test, y_pred)
ps=precision_score (y_test, y_pred)
rs=recall_score (y_test, y_pred)
f1=f1_score(y_test, y_pred)

print(cm)
print(ac)
print(ps)
print(rs)
print(f1)

from sklearn.model_selection import cross_val_score

#Cross validation
acc=cross_val_score(estimator= model, X=x_train_scaled, y=y_train, cv=10 )

#print(acc.mean())
#print(model)
#print(pd.DataFrame(model.coef_.T, columns=["Katsayı"], index=x_train.columns))


from sklearn.feature_selection import RFE

#Feature selection with RFE
fs=RFE(model, 20).fit(x_train_scaled, y_train)

print(x_train_scaled.columns[fs.support_])

#After selecting 20 features, re-modeling, fitting and predicting values
model=LogisticRegression(random_state=0)
model.fit(x_train_scaled[x_train_scaled.columns[fs.support_]], y_train)
y_pred=(model.predict(x_test_scaled[x_test_scaled.columns[fs.support_]]))

#Metrics to measure how good the model is
cm=confusion_matrix (y_test, y_pred)
ac=accuracy_score (y_test, y_pred)
ps=precision_score (y_test, y_pred)
rs=recall_score (y_test, y_pred)
f1=f1_score(y_test, y_pred)

print(cm)
print(ac)
print(ps)
print(rs)
print(f1)

eren=pd.DataFrame()
#To see y_test and y_pred in the same data frame so that it is more concrete.
eren["Churn"]=y_test
eren["Y_Pred"]=y_pred
print(eren)


















