import numpy as np
from utils_1batch import *


# ________________ build the NN model of one hidden layer using mini batches ________________ #
def nn_1layer_minibatch(
    X,
    Y,
    n_h,
    learning_rate,
    activation,
    minibatch_size=64,
    number_iterations=1000,
    print_cost=False,
):
    # Set up
    np.random.seed(0)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    m = X.shape[1]  # Number of training examples

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Initialize cost array
    costs = np.zeros(number_iterations)

    # Loop through forward and backward propagations
    for i in range(0, number_iterations):
        # Shuffle and partition into minibatches
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((n_y, m))
        minibatches = [
            (
                shuffled_X[:, k : k + minibatch_size],
                shuffled_Y[:, k : k + minibatch_size],
            )
            for k in range(0, m, minibatch_size)
        ]

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            # Apply forward propagation
            A2, temp_cache = forward_propagation(minibatch_X, parameters, activation)

            # Compute the cost
            cost = compute_cost(A2, minibatch_Y)

            # Apply backward propagation
            grads = backward_propagation(
                parameters, temp_cache, minibatch_X, minibatch_Y, activation
            )

            # Gradient descent parameter updates
            parameters = update_parameters(
                parameters, grads, learning_rate=learning_rate
            )

        # Save the cost
        costs[i] = cost

        # Print the cost after every 1000 loops
        if print_cost and i % 1000 == 0:
            print("Cost after interation %i: %f" % (i, cost))

    return parameters, costs


# ________________ test the model using different number of iterations ________________ #
def test_iteration_minibatch(
    file_name,
    data,
    n_h,
    learning_rate,
    activation,
    number_iterations,
    minibatch_size=64,
    batch_type="mini batches",
):
    for i in range(len(number_iterations)):
        # Train the model on the training set
        X_train = data["X_train"]
        Y_train = data["Y_train"]

        # Record the start time
        start_time = time.time()

        parameters, _ = nn_1layer_minibatch(
            X_train,
            Y_train,
            n_h,
            learning_rate=learning_rate,
            activation=activation,
            number_iterations=number_iterations[i],
            minibatch_size=minibatch_size,
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

        # Test on the test set
        # X_test = data["X_test"]
        # Y_test = data["Y_test"]
        # Y_hat_test = predict(X_test.T, parameters, activation)
        # accuracy_test = compute_accuracy(Y_test, Y_hat_test)

        print(f"Accuracy on training: {accuracy_train} % vs dev: {accuracy_dev} % ")

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


# ________________ test the model using different learning rates ________________ #
def test_learningrate_minibatch(
    file_name,
    data,
    n_h,
    learning_rates,
    activation,
    number_iterations,
    minibatch_size=64,
    batch_type="mini batches",
):
    for j in range(len(learning_rates)):
        test_iteration_minibatch(
            file_name,
            data,
            n_h,
            learning_rates[j],
            activation,
            number_iterations,
            minibatch_size=64,
            batch_type="mini batches",
        )


# ________________ test the model using different activation functions ________________ #
def test_activation_minibatch(
    file_name,
    data,
    n_h,
    learning_rates,
    activations,
    number_iterations,
    minibatch_size=64,
    batch_type="mini batches",
):
    for k in range(len(activations)):
        test_learningrate_minibatch(
            file_name,
            data,
            n_h,
            learning_rates,
            activations[k],
            number_iterations,
            minibatch_size=64,
            batch_type="mini batches",
        )


# ________________ test the model using different number of nodes ________________ #
def test_nodes_minibatch(
    file_name,
    data,
    n_hs,
    learning_rates,
    activations,
    number_iterations,
    minibatch_size=64,
    batch_type="mini batches",
):
    for l in range(len(n_hs)):
        test_activation_minibatch(
            file_name,
            data,
            n_hs[l],
            learning_rates,
            activations,
            number_iterations,
            minibatch_size=64,
            batch_type="mini batches",
        )
