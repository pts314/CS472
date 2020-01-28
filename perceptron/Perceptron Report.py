#!/usr/bin/env python
# coding: utf-8

# Peter Sawyer
# 
# 
# Import stuff:

# In[1]:


from perceptron import PerceptronClassifier
import matplotlib.pyplot as plt
from arff import Arff
import numpy as np


# Because most of the code has been fairly repetative, the code will be run from a standard template

# In[2]:


def runMahCode(arff, shuffle=True, determ=0, training=False, lr=.1, quiet=False):
    mat = Arff(arff,label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1:]
    PClass = PerceptronClassifier(lr=lr,shuffle=shuffle,deterministic=determ)
    Accuracy = 0.0
    if (training):
        X_train, y_train, X_test, y_test = PerceptronClassifier.split_training(data,labels)
        PClass.fit(X_train,y_train)
        Accuracy = PClass.score(X_test,y_test)
    else:
        PClass.fit(data,labels)
        Accuracy = PClass.score(data,labels)
    if not quiet:
        print("Accuracy = [{:.5f}]".format(Accuracy))
        print("Final Weights =",PClass.get_weights())
    else:
        return Accuracy


# # 1. Implement the code

# In[3]:


runMahCode("../data/perceptron/debug/linsep2nonorigin.arff",False, 10)


# Note that I increased the precision from the standard 2. With rounding, the debug works as expected.

# In[4]:


runMahCode("../data/perceptron/evaluation/data_banknote_authentication.arff",
           shuffle=False,determ=10)


# On the eval, we got an accuracy of .987, which is pretty good

# My stopping algorithm keeps track of the score over time, stops after no improvement for a default of 10 epochs, and restoring the best result of those last 10.

# # 2. Create two text files

# In[5]:


cat "linSep.csv"


# In[6]:


cat "nonLinSep.csv"


# In[7]:


linSep = np.genfromtxt('linSep.csv', delimiter=',',skip_header=1)
nonLin = np.genfromtxt("nonLinSep.csv", delimiter=",",skip_header=1)


# In[8]:


plt.scatter(linSep[:,0],y=linSep[:,1],c=linSep[:,2])
plt.title("Linearly Separable")
plt.ylabel("Y: Net Promoter Score")
plt.xlabel("X: pH of Coffee")
plt.show();


plot = plt.scatter(nonLin[:,0],y=nonLin[:,1],c=nonLin[:,2])
plt.title("Non-Linearly Separable")
plt.ylabel("Y: Net Promoter Score")
plt.xlabel("X: pH of Coffee")
plt.show();
# add the plot later


# # 3. Train on both sets

# Linearly Separable:

# In[9]:


data = linSep[:,:-1]
labels = linSep[:,-1:]
for lr in np.linspace(.01, 1, 10):
    PClass = PerceptronClassifier(lr=lr,shuffle=True,deterministic=0)
    PClass.fit(data,labels)
    Accuracy = PClass.score(data,labels)
    print(Accuracy)


# Because everything is centered very close to zero, it doesn't take very long for the learning rate to differentiate the two sets. 
# 
# We also see that the learning rate doesn't change much.

# In[10]:


data = nonLin[:,:-1]
labels = nonLin[:,-1:]
for lr in np.linspace(.01, 1, 10):
    PClass = PerceptronClassifier(lr=lr,shuffle=True,deterministic=0)
    PClass.fit(data,labels)
    Accuracy = PClass.score(data,labels)
    print(Accuracy)


# Again, the learning rate had almost no impact

# In[11]:


data = nonLin[:,:-1]
labels = nonLin[:,-1:]

PClass = PerceptronClassifier(lr=.1,shuffle=True,deterministic=0)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print(Accuracy)
w = PClass.get_weights()


# In[12]:


# Non-Linearly Separable
x = np.linspace(-1, 1, 100)
w1,w2,b = w
plt.title("Non-Linearly Seperable")
plt.ylabel("Y: Net Promoter Score")
plt.xlabel("X: pH of Coffee")
plt.scatter(nonLin[:,0],y=nonLin[:,1],c=nonLin[:,2])
plt.plot(x, (-w1/w2)*x + (-b / w2))
plt.ylim([-1,1])
plt.show()


# It's pretty bad. I'm not quite sure why the weights are so off. Given more time, I might try to refine the algorithm, or start with random weights to see if that changes anything

# In[13]:


data = linSep[:,:-1]
labels = linSep[:,-1:]

PClass = PerceptronClassifier(lr=.1,shuffle=True,deterministic=0)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print(Accuracy)
w = PClass.get_weights()
print(w)


# In[14]:


# Linearly Separable
x = np.linspace(-1, 1, 100)
w1,w2,b = w
plt.title("Linearly Seperable")
plt.ylabel("Y: Net Promoter Score")
plt.xlabel("X: pH of Coffee")
plt.scatter(linSep[:,0],y=linSep[:,1],c=linSep[:,2])
plt.plot(x, (-w1/w2)*x + (-b / w2))
plt.ylim([-1,1])
plt.show()


# That's more like it. Probably because it's a super easy data set, but still not bad.

# # 4. Learn Voting
# First I want to know the baseline if I just try to fit on the entire dataset

# In[15]:


runMahCode("standardVoting.arff")


# Now trying it slightly random five times:

# In[16]:


mat = Arff("standardVoting.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1:]
test = []
train = []
iters = []
all_scores = []
print("Iterations | Training | Testing")
for i in range(5):
    PClass = PerceptronClassifier(lr=.1,shuffle=True)
    Accuracy = 0.0
    X_train, y_train, X_test, y_test = PerceptronClassifier.split_training(data,labels)
    trash, iterr, scores = PClass.fit(X_train,y_train, quiet=True)
    all_scores.append(scores)
    training = PClass.score(X_train,y_train)
    testing = PClass.score(X_test,y_test)
    print("    {:}     |  {:.4f}  | {:.4f}".format(iterr, training, testing))
    iters.append(iterr)
    test.append(testing)
    train.append(training)
print(f"avg iter: {np.mean(iters)}")
print(f"avg training: {np.mean(train)}")
print(f"avg test: {np.mean(test)}")


# In[17]:


averages = []
minLen = min(iters)
#print(min(all_scores[:], [key=len]))
for i in range(minLen):
    averages.append(np.mean(scores[:][i]))
averages = np.array(averages)
plt.plot(np.abs(1-averages))
plt.title("Avg Misclassification across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Avg Misclassification Rate")
plt.show()


# In[18]:


runMahCode("standardVoting.arff")


# From this, we see that the physician fee freeze is the strongest identifing feature, followed by imigration, adoption of the budget resolution, mx missle, and then export-addministration act in south africa. We see that Synfuels cutback doesn't matter. 

# # 5. scikit-learn

# In[19]:


from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split


# In[20]:


ptron = Perceptron(tol=1e-3, random_state=0)
mat = Arff("standardVoting.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)


# In[21]:


ptron.fit(X_train,y_train)


# In[22]:


ptron.score(X_test,y_test)


# We see that our naive perceptron does fairly well compaired to the sklearn version. 

# # 6. Iris Data Set

# In[23]:


from sklearn.datasets import load_iris


# In[24]:


X,y = load_iris(True)


# In[25]:


labels = ['setosa', 'versicolor', 'virginica']


# In[26]:


plt.scatter(X[:,2],X[:,3],c=y )
plt.title("example iris parameters")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[27]:


def giveOneValue(y, value):
    return 1 if (y==value) else 0


# In[28]:


give1 = np.vectorize(giveOneValue)


# In[29]:


ptrons = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
for i in range(3):
    y_new = give1(y_train,i)
    new_ptron = ptron.fit(X_train,y_new)
    ptrons.append(new_ptron)
#     print(y_new-y_train)


# In[30]:


ptrons


# In[31]:


for i in range(3):
    print(f"For {labels[i]}, the accuracy is {ptrons[i].score(X_test,give1(y_test,i))}")


# From the example graph, we could see that the virginica was generally well split from the other two types, while the other two sets are more mixed, and thus harder to separate with a linear model.

# In[ ]:




