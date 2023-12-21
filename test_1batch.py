from utils_data import *
from utils_1batch import *
from utils_minibatch import *
import numpy as np


np.random.seed(0)
# ________________ MAKE DATA ________________ #
N = 2000
noise = 0.25
# load and visualize data
X, Y = load_data(N, noise)

# visualize the data
path_to_save_plot = os.path.join("input", "viz")
plot_data(X, Y, path_to_save_plot)

# split X, Y to train and test datasets
data = make_data(X, Y, test_size=0.2)

# save data
path_to_save_data = os.path.join("input", "data")
save_data(data, path_to_save_data)


# number of records
m = X.shape[1]
# number of features
n_x = 2
# the number of nodes in the hidden layer
n_hs = np.array([1, 2, 3, 4, 5, 10, 50])
# choice of activation funtion
activations = np.array(["tanh", "sigmoid", "relu"])
# learning rate
learning_rates = np.array([1.2, 0.6, 0.1, 0.01, 0.001])
# number of iterations
number_iterations = np.array([100, 1000, 10000, 100000])


def run_test_1batch():
    # run the test
    test_nodes_1batch(
        file_name,
        data,
        n_hs,
        number_iterations=number_iterations,
        learning_rates=learning_rates,
        activations=activations,
        batch_type="one batch",
    )


if __name__ == "__main__":
    file_name = os.path.join("output", "data", "accuracy", "1batch")
    run_test_1batch()
