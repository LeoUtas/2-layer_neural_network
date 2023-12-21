import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# ________________ load data ________________ #
def load_data(N, noise):
    np.random.seed(0)

    N = N
    noise = noise
    X, Y = datasets.make_moons(n_samples=N, noise=noise)

    return X, Y


# ________________ separate the data into train, dev and test datasets ________________ #
def make_data(X, Y, test_size=0.2):
    # split into training and temporary sets
    X_train, X_dev, Y_train, Y_dev = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    # reshape X to match neural network's expected shape
    X_train = X_train.T
    X_dev = X_dev.T
    # reshape Y to match neural network's expected shape
    Y_train = Y_train.reshape(1, Y_train.shape[0])
    Y_dev = Y_dev.reshape(1, Y_dev.shape[0])

    # Create a dictionary to store the data
    data = {"X_train": X_train, "X_dev": X_dev, "Y_train": Y_train, "Y_dev": Y_dev}

    return data


# ________________ save data ________________ #
def save_data(data, path_to_save_data):
    directory = os.path.dirname(path_to_save_data)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key, value in data.items():
        np.save(os.path.join(f"{path_to_save_data}", f"{key}.npy"), value)


# ________________ load data ________________ #
def load_saveddata(path_to_saveddata):
    keys = ["X_train", "X_dev", "Y_train", "Y_dev"]
    data = {}
    for key in keys:
        full_path_to_saveddata = os.path.join(path_to_saveddata, f"{key}.npy")
        data[key] = np.load(full_path_to_saveddata)

    return data


# ________________ plot the loaded data ________________ #
def plot_data(X, Y, path_to_save_plot):
    # Separate the points based on categories
    category_0 = X[Y == 0]
    category_1 = X[Y == 1]

    # Plot the points, with different markers for each category
    plt.scatter(
        category_0[:, 0],
        category_0[:, 1],
        s=20,
        c="blue",
        marker="x",
        label="Category 0",
    )
    plt.scatter(
        category_1[:, 0],
        category_1[:, 1],
        s=10,
        c="red",
        marker="o",
        label="Category 1",
    )

    plt.legend()
    plt.savefig(os.path.join(path_to_save_plot, "viz.png"))
