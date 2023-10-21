import numpy as np


class LogisticRegression:
    def __init__(self) -> None:
        pass

    def initialize(self, num_inputs, num_classes):
        w = np.random.randn(num_classes, num_inputs) / np.sqrt(num_classes * num_inputs)
        b = np.random.randn(num_classes, 1) / np.sqrt(num_classes)
        param = {"w": w, "b": b}
        return param

    def softmax(self, z):
        z -= np.max(z)
        exp_list = np.exp(z)
        result = exp_list / np.sum(exp_list)
        result = result.reshape((len(z), 1))
        return result

    def calculate_accuracy(self, param, x_data, y_data):
        w = param["w"].transpose()
        dist = np.array(
            [
                np.squeeze(self.softmax(np.matmul(x_data[i], w)))
                for i in range(len(y_data))
            ]
        )

        result = np.argmax(dist, axis=1)
        accuracy = np.sum(result == y_data) / float(len(y_data))

        return result, accuracy * 100

    def train(self, param, x_train, y_train, x_test, y_test, learning_rate):
        mu = 0.9

        w_velocity = np.zeros(param["w"].shape)
        b_velocity = np.zeros(param["b"].shape)

        for epoch in range(10):
            rand_indices = np.random.choice(
                x_train.shape[0], x_train.shape[0], replace=False
            )

            for batch in range(x_train.shape[0] // 128):
                index = rand_indices[128 * batch : 128 * (batch + 1)]
                x_batch = x_train[index]
                y_batch = y_train[index]

                w_grad_list = []
                b_grad_list = []

                for i in range(x_batch.shape[0]):
                    x, y = x_batch[i].reshape((784, 1)), y_batch[i]
                    E = np.zeros((10, 1))
                    E[y][0] = 1
                    pred = self.softmax(np.matmul(param["w"], x) + param["b"])

                    w_grad = E - pred
                    w_grad = -np.matmul(w_grad, x.reshape((1, 784)))
                    w_grad_list.append(w_grad)

                    b_grad = -(E - pred)
                    b_grad_list.append(b_grad)

                dw = sum(w_grad_list) / x_batch.shape[0]
                db = sum(b_grad_list) / x_batch.shape[0]

                w_velocity = mu * w_velocity + learning_rate * dw
                b_velocity = mu * b_velocity + learning_rate * db
                param["w"] -= w_velocity
                param["b"] -= b_velocity

        return self.calculate_accuracy(param, x_test, y_test)
