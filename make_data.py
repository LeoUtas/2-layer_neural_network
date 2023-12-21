from utils_data import *


# ________________ MAKE DATA ________________ #
np.random.seed(0)

N = 2000
noise = 0.25


def run_make_data(N, noise):
    # load data from sklearn
    X, Y = load_data(N, noise)

    # save loaded data to input
    data = make_data(X, Y, test_size=0.2)
    save_data(data, path="input/")


if __name__ == "__main__":
    run_make_data()
