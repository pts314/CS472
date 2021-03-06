from perceptron import PerceptronClassifier
from arff import Arff
import numpy as np


mat = Arff("../data/perceptron/evaluation/data_banknote_authentication.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1:]
PClass = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print("Accuray = [{:.5f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights())
