import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
from scipy.special import expit
import pandas as pd

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

y = np.where(y >= 1, 1, y)
mean_value_thal = X['thal'].mean()
X['thal'].fillna(value=mean_value_thal, inplace=True)
mean_value_ca = X['ca'].mean()
X['ca'].fillna(value=mean_value_ca, inplace=True)
scaler = MinMaxScaler()

# Dopasowanie i transformacja danych (np. macierzy cech X)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



class MLP(object):


    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives


        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):

        activations = inputs

        self.activations[0] = activations

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)

            self.activations[i + 1] = activations

        return activations

    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]

            delta = error * self._sigmoid_derivative(activations)

            delta_re = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]

            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations, delta_re)

            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_errors = 0

            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target, output)
            print(f"Epoch: {i}, Errors {sum_errors/len(inputs)}")

    def gradient_descent(self, learningRate=1):

        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate
            self.weights[i] = weights
    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):

        return x * (1.0 - x)

    def _mse(self, target, output):

        return np.average((target - output) ** 2)

    def predict(self, inputs):
        predictions = self.forward_propagate(inputs)
        for i in range(len(predictions)):
            if predictions[i] >= 0.5:
                predictions[i] = 1
            else:
                predictions[i] = 0
        return predictions
if __name__ == "__main__":


    mlp = MLP(13, [10,10], 1)

    mlp.train(X_train, y_train, 200, 0.1)


    # create dummy data

    output = mlp.predict(X_test)
    print(output)

    print(f"Accuracy: {accuracy_score(y_test, output)}")
    #
    # items = np.array([[random() / 2, random() / 2] for _ in range(1000)])
    # targets = np.array([[i[0] + i[1]] for i in items])  # Etykiety dla operacji odejmowania
    # mlp = MLP(2, [10], 1)
    # mlp.train(items, targets, 50, 0.1)
    # input = np.array([0.3, 0.1])
    # target = np.array([0.4])
    # output = mlp.forward_propagate(input)
    #
    # print(" Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
    #
