import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

# Get a portion of the MNIST dataset and reduce the number of features
def preprocess_data(num_features=16, size_train=2000, test=0.25):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], -1)) / 255.0
    test_images = test_images.reshape((test_images.shape[0], -1)) / 255.0

    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    pca = PCA(n_components=num_features)
    train_images = pca.fit_transform(train_images)
    test_images = pca.transform(test_images)

    train_images = train_images * 2 * np.pi
    test_images = test_images * 2 * np.pi

    size_test = size_train * test
    return train_images[:size_train], train_labels[:size_train], test_images[:size_test], test_labels[:size_test]

def feature_map(num_qubits=8):
    qc = QuantumCircuit(num_qubits)
    x = ParameterVector("x", length=2 * num_qubits)
    for i in range(num_qubits):
        qc.ry(x[2*i],i)
        qc.rz(x[2*i+1], i)
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
        qc.p(2*(-np.pi+x[2*(i+1)+1])*(-np.pi+x[2*i+1]),i+1)
        qc.cx(i, i+1)
    return qc

encoding = feature_map()
img = encoding.decompose().draw('mpl', scale=1)
plt.show()
