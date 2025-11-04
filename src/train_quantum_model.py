import random

import numpy as np
from PIL import ImageOps, Image
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.utils.loss_functions import L2Loss, CrossEntropyLoss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.utils import to_categorical
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_aer.primitives import EstimatorV2 as Estimator
import matplotlib.pyplot as plt


# Get a portion of the MNIST dataset and reduce the number of features
def preprocess_data(input_features, output, size_train=2000, test=0.25):
    size_test = int(size_train * test)
    n_train = np.random.randint(0, 59999-size_train)
    n_test = np.random.randint(0, 9999-size_test)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], -1))
    test_images = test_images.reshape((test_images.shape[0], -1))

    def reduce_image(img, threshold=10):
        img = img.reshape((28, 28))
        img = Image.fromarray(img.astype(np.uint8))
        coords = np.argwhere(np.asarray(img) > threshold)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img.crop((x0, y0, x1, y1))
        img = img.convert('L')
        img = img.resize((4, 5))
        return np.asarray(img).flatten()/255.

    """scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    pca = PCA(n_components=input_features)
    train_images = pca.fit_transform(train_images)
    test_images = pca.transform(test_images)"""

    train_images = train_images[n_train:n_train+size_train]
    test_images = test_images[n_test:n_test+size_test]
    train_labels = train_labels[n_train:n_train+size_train]
    test_labels = test_labels[n_test:n_test+size_test]

    train_images = np.array([reduce_image(img, 10) for img in train_images])
    test_images = np.array([reduce_image(img, 10) for img in test_images])

    train_images = train_images * 2 * np.pi
    test_images = test_images * 2 * np.pi

    """# Map a number from 0 to 9 to its binary representation of 4 bits, and then each 0 to 1 and each 1 to -1
    train_labels = 1 - 2*np.array([list(map(int, format(label, '04b'))) for label in train_labels])
    test_labels = 1 - 2*np.array([list(map(int, format(label, '04b'))) for label in test_labels])"""

    if output =='one_hot':
        train_labels = to_categorical(train_labels, num_classes=10)
        test_labels = to_categorical(test_labels, num_classes=10)

    return (train_images, train_labels), (test_images, test_labels)

def denseZZ_feature_map(num_features, param_name):
    num_qubits = num_features // 2
    qc = QuantumCircuit(num_qubits, name='Encoding')
    x = ParameterVector(param_name, length=num_features)
    for i in range(num_qubits):
        qc.h(i)
        qc.ry(x[2 * i], i)
        qc.rz(x[2 * i + 1], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(x[2 * i + 3] + x[2 * i + 1], i + 1)
        qc.cx(i, i + 1)
    if num_qubits > 0:
        qc.cx(num_qubits - 1, 0)
        qc.p(x[1] + x[2 * num_qubits - 1], 0)
        qc.cx(num_qubits - 1, 0)
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def pool_circuit(origin, target, param_name):
    if len(origin) != len(target):
        raise ValueError('origin and target must be of the same length')
    num_qubits = len(origin) + len(target)
    params = ParameterVector(param_name, length=num_qubits // 2 * 3)
    qc = QuantumCircuit(num_qubits, name='Pooling Layer')
    for i, (o, t) in enumerate(zip(origin, target)):
        qc.rz(-np.pi / 2, t)
        qc.cx(t, o)
        qc.rz(params[3 * i], o)
        qc.ry(params[3 * i + 1], t)
        qc.cx(o, t)
        qc.ry(params[3 * i + 2], t)
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def convolutional_circuit(num_qubits, param_name):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_name, length=(num_qubits // 2 - 1) * 15 + 3)
    param_pos = 0

    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    for i in range(0, num_qubits-2, 4):
        qc.compose(conv_circuit(params[param_pos : (param_pos + 3)]), [i, i + 1], inplace=True)
        param_pos += 3
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 2], inplace=True)
        param_pos += 3
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 3], inplace=True)
        param_pos += 3
    qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [num_qubits-2, num_qubits-1], inplace=True)
    param_pos += 3

    for i in range(2, num_qubits, 4):
        qc.compose(conv_circuit(params[param_pos : (param_pos + 3)]), [i, i + 1], inplace=True)
        param_pos += 3
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 2], inplace=True)
        param_pos += 3
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 3], inplace=True)
        param_pos += 3

    for i in range(1, num_qubits-1, 4):
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 1], inplace=True)
        param_pos += 3
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 2], inplace=True)
        param_pos += 3

    for i in range(3, num_qubits, 4):
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 1], inplace=True)
        param_pos += 3
        qc.compose(conv_circuit(params[param_pos: (param_pos + 3)]), [i, i + 2], inplace=True)
        param_pos += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# Function that will execute at the end of each iteration
def callback_func(weights, loss):
    losses.append(loss)

def interpreter(x):
    bits = format(x, '010b')
    return bits[::-1].find('1') if '1' in bits else 0

# Feature map to encode classical data into qubits
"""feature_map = denseZZ_feature_map(16, 'x')"""
feature_map = denseZZ_feature_map(20, 'x')

# VQC ansatz composed of convolutional with pooling layers
"""ansatz = QuantumCircuit(8)
ansatz.compose(convolutional_circuit(8, 'c1'), range(8), inplace=True)
ansatz.compose(pool_circuit(range(4), range(4, 8), 'p1'), range(8), inplace=True)
ansatz.compose(convolutional_circuit(4, 'c2'), range(4, 8), inplace=True)"""

ansatz = QuantumCircuit(10)
ansatz.compose(convolutional_circuit(10,'c1'), range(10), inplace=True)
ansatz.compose(denseZZ_feature_map(20, 'zz1'), range(10), inplace=True)
ansatz.compose(convolutional_circuit(10, 'c2'), range(10), inplace=True)

# Full circuit
"""QCNN = QuantumCircuit(8)
QCNN.compose(feature_map, range(8), inplace=True)
QCNN.compose(ansatz, range(8), inplace=True)"""
QCNN = QuantumCircuit(10)
QCNN.compose(feature_map, range(10), inplace=True)
QCNN.compose(ansatz, range(10), inplace=True)

"""# Observable
observables = []
for k in range(4):
    obs = SparsePauliOp.from_list([('I'*k + 'Z' + 'I'*(7-k), 1)])
    observables.append(obs)"""

#img_full = QCNN.draw('mpl', scale=0.6, fold=30)
#img_fm = feature_map(16, 'X').decompose().draw('mpl', scale=0.6)
#img_conv = convolutional_circuit(8, 'C').decompose().draw('mpl')
#img_pool = pool_circuit(range(1), range(1, 2), 'P').decompose().draw('mpl')
#plt.show()

"""qnn = EstimatorQNN(
    circuit=QCNN.decompose(),
    estimator=Estimator(),
    observables=observables,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    gradient=ParamShiftEstimatorGradient(Estimator()),
    num_virtual_qubits=8
)"""

losses = []
def test_vqc():
    vqc = VQC(
        num_qubits=10,
        feature_map=feature_map.decompose(),
        ansatz=ansatz.decompose(),
        optimizer=COBYLA(maxiter=50),
        callback=callback_func,
        sampler=Sampler()
    )
    (train_x, train_y), (test_x, test_y) = preprocess_data(20, 4, size_train=60)
    vqc.fit(train_x, train_y)
    print(vqc.score(test_x, test_y))
#test_vqc()

def test_optimizer():
    qnn = SamplerQNN(
        circuit=QCNN.decompose(),
        num_virtual_qubits=10,
        sampler=Sampler(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=interpreter,
        output_shape=10
    )
    mse = L2Loss()
    cse = CrossEntropyLoss()

    def cost_func_domain(weights, loss_func):
        predictions = qnn.forward(train_x, weights)
        cost = np.mean(loss_func(predictions, train_y))
        callback_func(weights, cost)
        return cost

    (train_x, train_y), (test_x, test_y) = preprocess_data(20, 4, size_train=60)
    optimizer = COBYLA(maxiter=30)
    initial_point = algorithm_globals.random.random(qnn.num_weights)
    opt_result = optimizer.minimize(lambda w: cost_func_domain(w, cse), initial_point)
    print(np.sum(np.argmax(qnn.forward(test_x, opt_result.x), axis=1) == np.argmax(test_y, axis=1)))
#test_optimizer()

def test_qnn():
    qnn = SamplerQNN(
        circuit=QCNN.decompose(),
        num_virtual_qubits=10,
        sampler=Sampler(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=interpreter,
        output_shape=10
    )
    (train_x, train_y), (test_x, test_y) = preprocess_data(20, 4, size_train=60)
    classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=60), callback=callback_func)
    classifier.fit(train_x, train_y)
    print(classifier.score(test_x, test_y))
test_qnn()

def view_data(qnn, train_x, train_y):
    print("Train X shape:", np.shape(train_x))
    print("Train Y shape:", np.shape(train_y))
    print("QNN output shape:", qnn.output_shape)
    sample_output = qnn.forward(train_x, np.random.random(len(qnn.weight_params)))
    print("QNN sample output:", sample_output)
    print("Sample output shape:", np.shape(sample_output))

plt.title("Loss against iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot(range(len(losses)), losses)
plt.show()