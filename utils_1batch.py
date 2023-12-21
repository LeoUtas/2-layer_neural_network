import numpy as np
import matplotlib.pyplot as plt
import csv, os, time, pickle


# ________________ initialize parameters ________________ #
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # to check if the parameters are in correct shapes
    assert W1.shape == (n_h, n_x)
    assert b1.shape == (n_h, 1)
    assert W2.shape == (n_y, n_h)
    assert b2.shape == (n_y, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


# ________________ sigmoid function ________________ #
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# ________________ relu function ________________ #
def relu(x):
    return np.maximum(0, x)


# ________________ derivative of relu function ________________ #
def relu_derivative(x):
    x_copy = x.copy()
    x_copy[x_copy <= 0] = 0
    x_copy[x_copy > 0] = 1
    return x_copy


# ________________ compute forward propagation ________________ #
def forward_propagation(X, parameters, activation):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1

    if activation == "tanh":
        A1 = np.tanh(Z1)
    elif activation == "sigmoid":
        A1 = sigmoid(Z1)
    elif activation == "relu":
        A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # check if the shape is correct as expected
    assert A2.shape == (1, X.shape[1])

    # store some values for the back_propagation usage later
    temp_cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }

    return A2, temp_cache


# ________________ compute the cost ________________ #
def compute_cost(A2, Y):
    # get the number of examples
    m = Y.shape[1]

    # compute the loss function
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    # sum of loss funtions = the cost function
    cost = -np.sum(logprobs) / m

    cost = float(np.squeeze(cost))

    return cost


# ________________ compute back propagation ________________ #
def backward_propagation(parameters, temp_cache, X, Y, activation):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = temp_cache["A1"]
    A2 = temp_cache["A2"]

    # compute the backward_propagation
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    if activation == "tanh":
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # derivative of tanh
    elif activation == "sigmoid":
        dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))  # derivative of sigmoid
    elif activation == "relu":
        dZ1 = np.dot(W2.T, dZ2) * relu_derivative(A1)  # derivative of ReLU

    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return gradients


# ________________ update the parameters ________________ #
def update_parameters(parameters, grads, learning_rate):
    # retrieve the parameters from the input
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # retrieve the gradient from the input
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    # update the parameters after comparing
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }

    return parameters


# ________________ build the NN model of one hidden layer using 1 batch ________________ #
def nn_1layer_1batch(
    X, Y, n_h, learning_rate, activation, number_iterations, print_cost=False
):
    # set up
    np.random.seed(0)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # initialize cost array
    costs = np.zeros(number_iterations)

    # Loop through forward and backward propagations
    for i in range(0, number_iterations):
        # apply forward_propagation
        A2, temp_cache = forward_propagation(X, parameters, activation)

        # compute the cost
        cost = compute_cost(A2, Y)

        # save the cost
        costs[i] = cost

        # apply backward_propagation
        grads = backward_propagation(parameters, temp_cache, X, Y, activation)

        # gradient descent parameter updats
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)

        # print the cost after every 1000 loops
        if print_cost and i % 1000 == 0:
            print("Cost after interation %i: %f" % (i, cost))

    return parameters, costs


# ________________ make predictions using the NN ________________ #
def predict(X, parameters, activation):
    X = X.T
    A2, temp_cache = forward_propagation(X, parameters, activation)
    predictions = (A2 > 0.5).astype(int)

    return predictions


# ________________ make plot of the decision boundary using the NN ________________ #
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


# ________________ compute the accuracy of the NN ________________ #
def compute_accuracy(Y, Y_hat):
    accuracy = float(
        (np.dot(Y, Y_hat.T) + np.dot(1 - Y, 1 - Y_hat.T)) / float(Y.size) * 100
    )

    accuracy = round(accuracy, 2)

    return accuracy


# ________________ record the test outputs ________________ #
def record_data(
    file_name,
    n_h,
    activation,
    batch_type,
    learning_rate,
    number_iterations,
    accuracy_data,
    train_time,
):
    directory = os.path.dirname(file_name)
    print(f"Attempting to create directory: {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Specify the filename for the .csv file
    file_name = file_name + ".csv"

    # Determine whether to write the header based on whether the file already exists
    write_header = not os.path.exists(file_name)

    # Write the accuracy data to the .csv file
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header row only if the file did not exist before
        if write_header:
            writer.writerow(
                [
                    "Number of nodes",
                    "Activation",
                    "Batch type",
                    "Learning rate",
                    "Iterations",
                    "Train",
                    "Dev",
                    # "Test",
                    "Train time",
                ]
            )

        # Write each row of the accuracy table to the CSV file along with the learning rate and "one batch" value
        writer.writerow(
            [
                n_h,
                activation,
                batch_type,
                learning_rate,
                number_iterations,
                accuracy_data[0],
                accuracy_data[1],
                # accuracy_data[2],
                train_time,
            ]
        )


# ________________ record the parameter outputs ________________ #
def save_parameters(parameters, filename):
    # Ensure the directory exists; if not, create it
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as f:
        pickle.dump(parameters, f)


# ________________ test the model using different number of iterations ________________ #
def test_iteration_1batch(
    file_name,
    data,
    n_h,
    number_iterations,
    learning_rate,
    activation,
    batch_type="one batch",
):
    for i in range(len(number_iterations)):
        # Train the model on the training set
        X_train = data["X_train"]
        Y_train = data["Y_train"]

        # Record the start time
        start_time = time.time()

        parameters, _ = nn_1layer_1batch(
            X_train,
            Y_train,
            n_h,
            activation=activation,
            learning_rate=learning_rate,
            number_iterations=number_iterations[i],
            print_cost=False,
        )

        # Record the end time
        end_time = time.time()

        # Calculate the execution time
        train_time = end_time - start_time

        Y_hat_train = predict(X_train.T, parameters, activation)
        accuracy_train = compute_accuracy(Y_train, Y_hat_train)

        # Validate on the dev set
        X_dev = data["X_dev"]
        Y_dev = data["Y_dev"]
        Y_hat_dev = predict(X_dev.T, parameters, activation)
        accuracy_dev = compute_accuracy(Y_dev, Y_hat_dev)

        print(
            f"Accuracy on training: {accuracy_train} % vs dev: {accuracy_dev} % train time: {train_time}"
        )

        accuracy_data = [accuracy_train, accuracy_dev]

        record_data(
            file_name,
            n_h,
            activation,
            batch_type,
            learning_rate,
            number_iterations[i],
            accuracy_data,
            train_time,
        )

        # Save parameters
        save_parameters_filename = os.path.join(
            "output",
            "data",
            "parameters",
            f"parameters_{n_h}_{activation}_{batch_type}_{learning_rate}_{number_iterations[i]}.pkl",
        )
        save_parameters(parameters, save_parameters_filename)


# ________________ test the model using different learning rates ________________ #
def test_learningrate_1batch(
    file_name,
    data,
    n_h,
    number_iterations,
    learning_rates,
    activation,
    batch_type="one batch",
):
    for j in range(len(learning_rates)):
        test_iteration_1batch(
            file_name,
            data,
            n_h,
            number_iterations,
            learning_rates[j],
            activation,
            batch_type="one batch",
        )


# ________________ test the model using different activation functions ________________ #
def test_activation_1batch(
    file_name,
    data,
    n_h,
    number_iterations,
    learning_rates,
    activations,
    batch_type="one batch",
):
    for k in range(len(activations)):
        test_learningrate_1batch(
            file_name,
            data,
            n_h,
            number_iterations,
            learning_rates,
            activations[k],
            batch_type="one batch",
        )


# ________________ test the model using different number of nodes ________________ #
def test_nodes_1batch(
    file_name,
    data,
    n_hs,
    number_iterations,
    learning_rates,
    activations,
    batch_type="one batch",
):
    for l in range(len(n_hs)):
        test_activation_1batch(
            file_name,
            data,
            n_hs[l],
            number_iterations,
            learning_rates,
            activations,
            batch_type="one batch",
        )
