import numpy as np
from mnist import MNIST


class Experiment:
    def load_mnist(self):
        mndata = MNIST("dataset")
        x_train, y_train = mndata.load_training()
        x_test, y_test = mndata.load_testing()
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    def calculate_class_probabilities(self, labels):
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        total_labels = len(labels)
        class_probabilities = {}
        for label, count in zip(unique_labels, label_counts):
            class_probabilities[label] = count / total_labels
        return class_probabilities

    def calculate_pixel_probabilities(self, images, labels, laplace_smoothing):
        unique_labels = np.unique(labels)
        pixel_probabilities = {}
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_images = images[label_indices]
            total_pixels = np.sum(label_images)
            pixel_probabilities[label] = (
                np.sum(label_images, axis=0) + laplace_smoothing
            ) / (total_pixels + len(label_images) * laplace_smoothing)
        return pixel_probabilities

    def calculate_accuracy(self, predicted_labels, true_labels):
        correct_count = 0
        total_count = len(predicted_labels)

        for image_id, predicted_label in predicted_labels.items():
            if predicted_label == true_labels[image_id]:
                correct_count += 1

        accuracy = correct_count / total_count
        return accuracy * 100
