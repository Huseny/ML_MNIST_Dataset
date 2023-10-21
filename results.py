import numpy as np
from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression
from matplotlib import pyplot as plt
from experiment import Experiment


def try_naive_bayes(e: Experiment):
    train_images, train_labels, test_images, test_labels = e.load_mnist()

    train_images = (train_images > 0).astype(int)
    test_images = (test_images > 0).astype(int)

    class_probabilities = e.calculate_class_probabilities(train_labels)
    smoothing = [0.001, 0.01, 0.1, 1, 10, 100, 200, 400, 600, 800, 1000]
    accuracies = []
    naive_bayes = NaiveBayes()
    for laplace_smoothing in smoothing:
        pixel_probabilities = e.calculate_pixel_probabilities(
            train_images, train_labels, laplace_smoothing
        )

        predicted_labels = naive_bayes.predict(
            test_images, class_probabilities, pixel_probabilities
        )

        accuracy = e.calculate_accuracy(predicted_labels, test_labels)
        accuracies.append(accuracy)

    plt.plot(smoothing, accuracies, marker="o")
    plt.xlabel("Smoothing")
    plt.ylabel("Accuracy (%)")
    plt.title("Smoothing vs Accuracy in Naive Bayes for MNIST")
    plt.grid(True)
    plt.show()


def try_logistic(e: Experiment):
    logistic = LogisticRegression()
    x_train, y_train, x_test, y_test = e.load_mnist()
    np.random.seed(1024)

    num_inputs = x_train.shape[1]
    num_classes = len(set(y_train))
    param = logistic.initialize(num_inputs, num_classes)

    learning_rates = [0.001, 0.01, 0.1, 1.0, 1.5, 5, 10, 100, 1000]
    accuracies = []
    for learning_rate in learning_rates:
        predictions, accuracy = logistic.train(
            param, x_train, y_train, x_test, y_test, learning_rate
        )
        accuracies.append(accuracy)

    plt.plot(learning_rates, accuracies, marker="o")
    plt.xlabel("Learning Rates")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning Rate vs Accuracy in Logistic Regression for MNIST")
    plt.grid(True)
    plt.show()


e = Experiment()
try_naive_bayes(e)
