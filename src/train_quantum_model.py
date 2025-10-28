import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
import matplotlib.pyplot as plt


# Get a portion of the MNIST dataset and reduce the number of features
def preprocess_data(input_features, output_qubits, size_train=2000, test=0.25):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], -1)) / 255.0
    test_images = test_images.reshape((test_images.shape[0], -1)) / 255.0

    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    pca = PCA(n_components=input_features)
    train_images = pca.fit_transform(train_images)
    test_images = pca.transform(test_images)

    train_images = train_images * 2 * np.pi
    test_images = test_images * 2 * np.pi

    train_labels = np.array([[int(b) for b in np.binary_repr(l, width=output_qubits)] for l in train_labels])
    test_labels = np.array([[int(b) for b in np.binary_repr(l, width=output_qubits)] for l in test_labels])
    train_labels = 1 - 2 * train_labels
    test_labels = 1 - 2 * test_labels

    size_test = int(size_train * test)
    return (train_images[:size_train], train_labels[:size_train]), (test_images[:size_test], test_labels[:size_test])

def denseZZ_feature_map(num_features, param_name):
    num_qubits = num_features // 2
    qc = QuantumCircuit(num_qubits, name='Encoding')
    x = ParameterVector(param_name, length=num_features)
    for i in range(num_qubits):
        qc.ry(x[2 * i], i)
        qc.rz(x[2 * i + 1], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.p(2 * (-np.pi + x[2 * i + 3]) * (-np.pi + x[2 * i + 1]), i + 1)
        qc.cx(i, i + 1)
    if num_qubits > 0:
        qc.cx(num_qubits - 1, 0)
        qc.p(2 * (-np.pi + x[1]) * (-np.pi + x[2 * num_qubits - 1]), 0)
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

def convolutional_circuit(num_qubits, parameter_name):
    qc = efficient_su2(num_qubits, reps=2, parameter_prefix=parameter_name, name='Convolutional Layer')
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# Function that will execute at the end of each iteration
def callback_func(weights, loss):
    losses.append(loss)

# Estimator for the measurements
estimator = Estimator()

# Feature map to encode classical data into qubits
feature_map = denseZZ_feature_map(16, 'x')

# VQC ansatz composed of convolutional with pooling layers
ansatz = QuantumCircuit(8)
ansatz.compose(convolutional_circuit(8, 'c1'), range(8), inplace=True)
ansatz.compose(pool_circuit(range(4), range(4, 8), 'p1'), range(8), inplace=True)
ansatz.compose(convolutional_circuit(4, 'c2'), range(4, 8), inplace=True)

# Full circuit
QCNN = QuantumCircuit(8)
QCNN.compose(feature_map, range(8), inplace=True)
QCNN.compose(ansatz, range(8), inplace=True)

#img_full = QCNN.draw('mpl', scale=0.6, fold=30)
#img_fm = feature_map(16, 'X').decompose().draw('mpl', scale=0.6)
#img_conv = convolutional_circuit(8, 'C').decompose().draw('mpl')
#img_pool = pool_circuit(range(1), range(1, 2), 'P').decompose().draw('mpl')
#plt.show()

# Observabes for the measurement, since there are 10 classes, we need 4 bits
observable = []
for j in range(4):
    obs = SparsePauliOp.from_list([('I' * j + 'Z' + 'I' * (7 - j), 1)])
    observable.append(obs)

qnn = EstimatorQNN(
    circuit=QCNN.decompose(),
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
    gradient=ParamShiftEstimatorGradient(estimator=estimator),
    num_virtual_qubits=8
)

losses = []
classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=200), callback=callback_func)
(train_x, train_y), (test_x, test_y) = preprocess_data(16,4, size_train=10)
