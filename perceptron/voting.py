from perceptron import PerceptronClassifier
from arff import Arff
import numpy as np


mat = Arff("standardVoting.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1:]
PClass = PerceptronClassifier(lr=0.1)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print("Accuray = [{:.5f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights())
