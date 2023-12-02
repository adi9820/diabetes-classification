#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("diabetes.csv")
data


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data.describe()


# In[6]:


# correleation plot of independent variable
plt.figure(figsize = (12,10))
sns.heatmap(data.corr(), annot = True, fmt = ".2f", cmap = "YlGnBu")
plt.title("correleation heatmap")
plt.show()


# In[9]:


#plotting density graph of prgnancies and target variable
plt.figure(figsize = (12,10))
kde = sns.kdeplot(data["Pregnancies"][data["Outcome"]==1], color="Red", fill = True)
kde = sns.kdeplot(data["Pregnancies"][data["Outcome"]==0], color="Blue", fill = True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("proportion of outcome")
kde.legend(["positive", "negative"])


# In[10]:


#plotting density graphs
plt.figure(figsize = (12,10))
sns.violinplot(data = data, x = "Outcome", y = "Glucose", split = True, linewidth = 2, inner = "quart")


# In[12]:


#plotting density graphs
plt.figure(figsize = (12,10))
kde = sns.kdeplot(data["Glucose"][data["Outcome"]==1], color="Red", fill = True)
kde = sns.kdeplot(data["Glucose"][data["Outcome"]==0], color="Blue", fill = True)
kde.set_xlabel("Glucose")
kde.set_ylabel("Outcome")
kde.legend(["positive", "negative"])


# In[13]:


plt.figure(figsize = (12,10))
sns.violinplot(data = data, x = "Outcome", y = "BloodPressure", split = True, linewidth = 2, inner = "quart")


# In[14]:


plt.figure(figsize = (12,10))
sns.violinplot(data = data, x = "Outcome", y = "SkinThickness", split = True, linewidth = 2, inner = "quart")


# In[15]:


plt.figure(figsize = (12,10))
sns.violinplot(data = data, x = "Outcome", y = "Insulin", split = True, linewidth = 2, inner = "quart")


# In[16]:


plt.figure(figsize = (12,10))
sns.violinplot(data = data, x = "Outcome", y = "BMI", split = True, linewidth = 2, inner = "quart")


# In[47]:


# replacing 0's with mean/median values
data["Glucose"] = data["Glucose"].replace(0, data["Glucose"].median()) # Glucose
data["BloodPressure"] = data["BloodPressure"].replace(0, data["BloodPressure"].median()) # BloodPressure
data["BMI"] = data["BMI"].replace(0, data["BMI"].mean()) # BMI
data["SkinThickness"] = data["SkinThickness"].replace(0, data["SkinThickness"].median()) # SkinThickness
data["Insulin"] = data["Insulin"].replace(0, data["Insulin"].median()) # Insulin


# In[48]:


data


# In[49]:


# splitting dependent and independent data
x = data.drop(["Outcome"], axis=1)
y = data["Outcome"]


# In[50]:


# splitting data in training and testing dataset
from sklearn.model_selection import train_test_split


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)


# In[52]:


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier


# In[53]:


training_accuracy = []
test_accuracy = []
for n_neighbors in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(x_train, y_train)
    
    training_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))


# In[54]:


plt.plot(range(1,20), training_accuracy, label="training accuracy")
plt.plot(range(1,20), test_accuracy, label="test accuracy")
plt.xlabel("n_neighbours")
plt.ylabel("accuracy")
plt.legend()


# In[55]:


knn = KNeighborsClassifier(n_neighbors = 18)
knn.fit(x_train, y_train)
print("training accuracy : ", knn.score(x_train, y_train))
print("test accuracy : ", knn.score(x_test, y_test))


# In[57]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier


# In[58]:


training_accuracy = []
test_accuracy = []
for max_depth in range(1,9):
    dt = DecisionTreeClassifier(random_state = 0, max_depth = max_depth)
    dt.fit(x_train, y_train)
    
    training_accuracy.append(dt.score(x_train, y_train))
    test_accuracy.append(dt.score(x_test, y_test))


# In[36]:


plt.plot(range(1,9), training_accuracy, label="training accuracy")
plt.plot(range(1,9), test_accuracy, label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()


# In[59]:


dt = DecisionTreeClassifier(random_state = 0, max_depth = 4)
dt.fit(x_train, y_train)
print("training accuracy : ", dt.score(x_train, y_train))
print("test accuracy : ", dt.score(x_test, y_test))


# In[74]:


# Neural Network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state = 42)
mlp.fit(x_train, y_train)
print("training accuracy : ", mlp.score(x_train, y_train))
print("test accuracy : ", mlp.score(x_test, y_test))


# In[75]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)


# In[76]:


mlp1 = MLPClassifier(random_state = 42)
mlp1.fit(x_train_scaled, y_train)
print("training accuracy : ", mlp1.score(x_train_scaled, y_train))
print("test accuracy : ", mlp1.score(x_test_scaled, y_test))


# In[77]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[78]:


y_mlp1pred=mlp1.predict(x_test_scaled)


# In[79]:


cm = confusion_matrix(y_test, y_mlp1pred)
cm


# In[80]:


print(classification_report(y_test, y_mlp1pred))


# In[81]:


sns.heatmap(cm, annot = True)


# In[ ]:




