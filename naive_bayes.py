import numpy as np


class NaiveBayes:
    def predict(self, images, class_probabilities, pixel_probabilities):
        predicted_labels = {}
        for image_id, image in enumerate(images):
            label_probabilities = {}
            for label, class_prob in class_probabilities.items():
                pixel_prob = pixel_probabilities[label]
                probabilities = np.sum(
                    np.log(pixel_prob) * image + np.log(1 - pixel_prob) * (1 - image)
                ) + np.log(class_prob)
                label_probabilities[label] = probabilities
            predicted_label = max(label_probabilities, key=label_probabilities.get)
            predicted_labels[image_id] = predicted_label
        return predicted_labels
