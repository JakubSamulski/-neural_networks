import numpy as np
from random import random





if __name__ == "__main__":
    items = np.array([[random() / 2, random() / 2] for _ in range(1000)])
    targets = np.array([[i[0] - i[1]] for i in items])  # Etykiety dla operacji odejmowania

    # Tworzenie nowej sieci MLP z jedną warstwą ukrytą
    updated_mlp = MLP(2, [10], 1)

    # Trenowanie sieci
    updated_mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.2])

    print()
    print("Our network believes that {} - {} is equal to {}".format(input[0], input[1], output[0]))