import numpy as np

# Funkcja aktywacji - softmax
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Pochodna funkcji aktywacji softmax
def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

# Funkcja kosztu - entropia krzyżowa
def cross_entropy_loss(y, y_pred):
    m = y.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss

# Pochodna funkcji kosztu entropii krzyżowej
def cross_entropy_loss_derivative(y, y_pred):
    m = y.shape[0]
    grad = y_pred
    grad[range(m), y] -= 1
    grad /= m
    return grad

# Klasa reprezentująca sieć neuronową
class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        # Inicjalizacja wag i biasów dla warstw ukrytych
        self.hidden_layer_sizes = hidden_layer_sizes
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def feedforward(self, X):
        # Propagacja w przód
        self.layer_inputs = []
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            layer_input = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(layer_input)
            if i == len(self.weights) - 1:
                # Dla warstwy wyjściowej używamy softmax
                layer_output = softmax(layer_input)
            else:
                # Dla warstw ukrytych używamy sigmoidalnej funkcji aktywacji
                layer_output = 1 / (1 + np.exp(-layer_input))
            self.layer_outputs.append(layer_output)
        return self.layer_outputs[-1]

    def backpropagate(self, X, y, learning_rate):
        m = X.shape[0]
        gradients = []

        # Obliczanie gradientów dla warstwy wyjściowej
        gradient_output = self.layer_outputs[-1]
        gradient_output[range(m), y] -= 1
        gradient_output /= m
        gradients.insert(0, gradient_output)

        # Obliczanie gradientów dla warstw ukrytych
        for i in reversed(range(len(self.weights) - 1)):
            gradient_hidden = gradients[0].dot(self.weights[i + 1].T)
            gradient_hidden *= self.layer_inputs[i] * (1 - self.layer_inputs[i])  # Pochodna dla sigmoidalnej funkcji aktywacji
            gradients.insert(0, gradient_hidden)

        # Aktualizacja wag i biasów
        for i in range(len(self.weights)):
            self.weights[i] -= self.layer_outputs[i].T.dot(gradients[i]) * learning_rate
            self.biases[i] -= np.sum(gradients[i], axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            loss = cross_entropy_loss(y, output)
            self.backpropagate(X, y, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

# Przykładowe dane wejściowe
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Przykładowe dane wyjściowe
y = np.array([0, 1, 1, 0])

# Inicjalizacja i trening sieci neuronowej z trzema warstwami ukrytymi
input_size = 2
hidden_layer_sizes = [4, 3, 2]
output_size = 2
nn = NeuralNetwork(input_size, hidden_layer_sizes, output_size)
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Testowanie sieci
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data_point in test_data:
    prediction = nn.feedforward(data_point)
    print(f"Input: {data_point} Predicted Output: {prediction}")
