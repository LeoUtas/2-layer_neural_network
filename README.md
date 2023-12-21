<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li><a href="#a-2-layer-architecture?">A 2-layer architecture?</a></li>
    <li><a href="#data-simulation">Data simulation</a></li>
    <li><a href="#a-general-form-of-neural-network">A general form of neural network</a></li>
    <li><a href="#how-a-neural-network-make-predictions">How a neural network make predictions</a></li>
    <li><a href="#how-a-neural-network-learn">How a neural network learn</a></li>
    <li><a href="#test-different-selection-of-model-configuration">Test different selection of model configuration</a></li>
    <li><a href="#examining-results">Examining results</a></li>
  </ol>
</details>

# 2-layer Neural Network

### Introduction

This article provides the development of a 2-hidden-layer neural network (NN) only using NumPy. This project is a practical introduction to the fundamentals of deep learning and neural network architecture. The main focus will be on the step-by-step construction of the network, aiming to provide a clear and straightforward understanding of its underlying mechanics (i.e., the mathematics behind NNs).

### A 2-layer neural network?

There is no secret behind the selection of 2 layers. In this project, we will experiment with different choices for hyperparameters for the NN; therefore, a 2-layer architecture is simple enough to make the test feasible.

### Data simulation

First of all, we simulate some data using datasets from sklearn.

```python
from utils_data import *

N = 2000
noise = 0.25
# load and visualize data
X, Y = load_data(N, noise)

# visualize the data
path_to_save_plot = os.path.join("input", "viz")
plot_data(X, Y, path_to_save_plot)
```

<p align="center">
  <a href="">
    <img src="/input/viz/viz.png" width="620" alt=""/>
  </a>
</p>

Our dataset consists of two categories, represented by red and blue dots. If you like to think of a real-world problem, the blue could represent males, and the red could represent females in a sample.
The objective is to develop a model that accurately distinguishes between the red and blue groups. The challenge here is that the data isn't linearly separable; in other words, it's likely difficult to draw a straight line that cleanly divides the two groups. This limitation means that linear models (e.g., logistic regression) are unlikely to be effective. This scenario highlights one of the key strengths of neural networks: their ability to handle data effectively that isn't linearly separable.

### A general form of neural network

A neural network comprises layers of interconnected nodes (or "neurons"). These layers include:

-   Input Layer: This is where the network receives its input data.
-   Hidden Layers: These layers, which can be one or multiple, perform computations on the input data. Each neuron in these layers applies 2 mathematical functions to the data.
-   Output Layer: This layer produces the final output of the network, such as a classification (e.g., identifying whether a data point belongs to red or blue groups) or a continuous value (e.g., predicting house prices).

</br>

<p align="center">
  <a href="">
    <img src="/input/viz/architecture.png" width="560" alt=""/>
  </a>
</p>

Source: Andrew Ng

</br>

### How a neural network make predictions

In this example of 2-layer NN, data (e.g., $x^{(i)}$) flows from the input layer, undergoes computing in the hidden layers, and the output layer generates the outcome (i.e., $\hat{y}^{(i)}$). . Mathematically, the NN generates a probability that determines the outcome prediction of $\hat{y}^{(i)}$ belongs to group 0 or 1. The computation can be written as follows:

</br>

$$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1]}\tag{1}$$

$$a^{[1] (i)} = g(z^{[1] (i)})\tag{2}$$

$$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2]}\tag{3}$$

$$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$

$$y^{(i)}_{\text{prediction}} = \begin{cases} 1 & \text{if } a^{[2](i)} > 0.5 \\ 0 & \text{otherwise} \end{cases} \tag{5}$$

This project will examine three options for the activation function g(): <a href="https://en.wikipedia.org/wiki/Sigmoid_function#:~:text=A%20sigmoid%20function%20is%20a,refer%20to%20the%20same%20object.">sigmoid</a>, <a href="https://reference.wolfram.com/language/ref/Tanh.html#:~:text=Tanh%20is%20the%20hyperbolic%20tangent,and%20hyperbolic%20cosine%20functions%20via%20.">tanh</a> and <a href="https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning">relu</a>.

In python code

```python
# define helper functions in utils_1batch.py

# ________________ sigmoid function ________________ #
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
# ________________ relu function ________________ #
def relu(x):
    return np.maximum(0, x)
# ________________ initialize parameters ________________ #
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters

# ________________ compute forward propagation ________________ #
def forward_propagation(X, parameters, activation):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1

    # there are 3 options for the function g()
    if activation == "tanh":
        A1 = np.tanh(Z1)
    elif activation == "sigmoid":
        A1 = sigmoid(Z1)
    elif activation == "relu":
        A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # store values for the back_propagation usage later
    temp_cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }

    return A2, temp_cache
```

### How a neural network learn

Training our neural network involves identifying the optimal parameters (W1, b1, W2, b2) that minimize the discrepancy between prediction and ground truth. The key question is how to quantify this discrepancy or error. To evaluate this error, we use what's called a cost function $J$ as follows:
$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$

In python code

```python
# define helper functions in utils_1batch.py
def compute_cost(A2, Y):
    # get the number of examples
    m = Y.shape[1]

    # compute the loss function
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    # sum of loss funtions = the cost function
    cost = -np.sum(logprobs) / m

    cost = float(np.squeeze(cost))

    return cost
```

</br>

Once the cost can be computed, the goal will be to minimize the cost. In other words, we search for a solution to minimize the variance between the prediction and the ground truth (i.e., maximizing the likelihood). It is where gradient descent comes in. In this project, we will implement a vanilla version of the gradient descent algorithm (i.e., applying gradient descent through the entire batch of data with one fixed learning rate). If you're curious about different techniques regarding gradient descent, see <a href="https://cs231n.github.io/neural-networks-3/#anneal"> cs231n</a>.

For gradient descent to work, it needs the gradients (i.e., the vector of derivatives) concerning the parameters as follows

<p align="center">
  <a href="">
    <img src="/input/viz/grad_summary.png" width="660" alt=""/>
  </a>
</p>

The calculation of these gradients is achieved through the backpropagation algorithm, an efficient method that begins at the output and works its way backwards to determine the gradients (for more details, see <a href="https://en.wikipedia.org/wiki/Backpropagation">more about backpropagation</a>). The parameters are updated simultaneously until the minimum cost is determined.

In python code:

```python
# define helper functions in utils_1batch.py
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
```

</br>

Bring everything together to make a 2-layer NN as follows:

```python
# define helper functions in utils_1batch.py
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
```

</br>

### Test different selection of model configuration

A commonly asked question from beginners when learning NN is, what are the optimal hyperparameters to use? A fairly simple architecture like this 2-layer NN allows experiments for multiple choices of hyperparameters.

```python
from utils_1batch import *

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
```

### Examining results

</br>

<p align="center">
  <a href="">
    <img src="/output/viz/output0.png" width="880" alt=""/>
  </a>
</p>

</br>

The visualization reveals a broad spectrum of outcomes when using different configurations for the same neural network architecture to solve the same problem. Some configurations stand out, achieving high accuracy levels above 90%, and within this group, there are exceptional cases where accuracy surpasses 95%. These high-performing configurations are prime candidates for further investigation. The next analytical step would be to filter out configurations with an accuracy greater than 95% and conduct a more focused comparison between the training and development datasets

A common practice in model development is that we don't want overfitting issues when a model performs greatly on the training dataset while fitting poorly on the development one. Therefore, we remove the configurations meeting two conditions:

-   Accuracies of > 95%; and
-   The difference between the training and development datasets is > 1%.

The 1% difference is subjective; however, it is reasonably good in this exercise.

</br>

<p align="center">
  <a href="">
    <img src="/output/viz/output1.png" width="880" alt=""/>
  </a>
</p>

</br>

After filtering, we now have several configurations left. The next step is considering how much training time each configuration takes.

</br>

<p align="center">
  <a href="">
    <img src="/output/viz/output2.png" width="880" alt=""/>
  </a>
</p>

</br>

From the scatter plot, we can observe a significant variation in the training times for different configurations. Even though the model performance on the goodness of fit is very similar, some configurations require a lot of resources for training, while others can be trained very fast. In solving real-world tasks, the ones requiring fewer resources are likely preferred. I will now filter out the configurations with a training time of < 30 seconds. Therefore, after filtering, two candidates are retained.

</br>

| Number of nodes | Activation | Learning rate | Iterations | Train |  Dev  | Train time | Configuration index |
| :-------------: | :--------: | :-----------: | :--------: | :---: | :---: | :--------: | :-----------------: |
|        3        |    tanh    |      0.6      |   10000    | 95.12 | 94.25 |  4.527224  |         126         |
|        5        |  sigmoid   |      0.1      |   100000   | 95.12 | 94.25 | 20.912593  |         271         |

</br>

Looking at the two candidates, I will choose the one with much less training time required. Finally, in this experiment, I will try the chosen configuration on a new simulation dataset to see how it performs on completely new data.

</br>

Data simulation

```python
N = 500
noise = 0.25
# load and visualize data
X, Y = load_data(N, noise)
X = X.T
Y =  Y.reshape(1, Y.shape[0])
```

Testing on the newly simulated data

```python
import pickle

with open('../output/data/parameters/parameters_3_tanh_one batch_0.6_10000.pkl', 'rb') as file:
    parameters = pickle.load(file)

Y_hat = predict(X.T, parameters, "tanh")
accuracy = compute_accuracy(Y, Y_hat)
print(f"Accuracy: {accuracy} % ")
```

```
Accuracy: 93.6 %
```

</br>

All the steps presented in this examining result section can be found in the file EDA.ipynb.

To wrap things up, this 2-layer neural network will not solve any real-world tasks, but it is an excellent starting point for anyone diving into artificial intelligence, machine learning, and deep learning. It is often the case that you will employ well-known libraries for solving real-world problems. This experiment could clear the mist about what is happening under the hood. It's important to recognize that this is just the tip of the iceberg in the universe of deep learning, which includes advanced concepts like minibatch, learning rate decay, and many more.
