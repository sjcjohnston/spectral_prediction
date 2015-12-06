from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
from spectraNN import NetFactory
import os

# net2= NeuralNet(
#     layers=[  # three layers: one hidden layer
#         ('input', layers.InputLayer),
#         ('hidden', layers.DenseLayer),
#         ('output', layers.DenseLayer),
#         ],
#     # layer parameters:
#     input_shape=(None, 3),  # 96x96 input pixels per batch
#     hidden_num_units=2,  # number of units in hidden layer
#     output_nonlinearity=None,  # output layer uses identity function
#     output_num_units=1,  # 30 target values

#     # optimization method:
#     update=nesterov_momentum,
#     update_learning_rate=0.01,
#     update_momentum=0.9,

#     regression=True,  # flag to indicate we're dealing with regression problem
#     max_epochs=1,  # we want to train this many epochs
#     verbose=1,
#     )


# x = np.array([[1,2,3],[3,2,1],[3,2,1],[3,2,1],[3,2,1],[3,2,1],[3,2,1],[3,2,1]])

# y = np.array([1,2,3,4,5,6,7,8])

# net1.fit(x,y)

if __name__ == "__main__":
    print "Loading Mininets:\n"
    d = os.getcwd()
    nn = NetFactory.load_mini_nets(d)
    print "NN:\t", nn
    print "Loaded {} MiniNets.\n".format(len(nn))