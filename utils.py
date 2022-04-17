import os
import numpy as np
import matplotlib.pyplot as plt

train_path = os.path.join("D:\\computational learning\\Ex3\\data\\train.csv")
test_path = os.path.join("D:\\computational learning\\Ex3\\data\\test.csv")


def get_sign(x):
    if x > 0:
        out = 1
    else:
        out = -1
    return out


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_data_and_labels = np.loadtxt(train_path, delimiter=',')
    train_data = train_data_and_labels[:, 0:2]
    train_labels = train_data_and_labels[:, 2]
    test_data_and_labels = np.loadtxt(test_path, delimiter=',')
    test_data = test_data_and_labels[:, 0:2]
    test_labels = test_data_and_labels[:, 2]
    return train_data, train_labels, test_data, test_labels


def scatter_train_data(train_data: np.ndarray, train_labels: np.ndarray) -> None:
    positive_set = train_data[train_labels == 1, :]
    negative_set = train_data[train_labels == -1, :]

    fig, ax = plt.subplots()
    # scatter positive labels
    ax.scatter(positive_set[:, 0], positive_set[:, 1], c='green', label='+1')
    # scatter negative labels
    ax.scatter(negative_set[:, 0], negative_set[:, 1], c='red', label='-1')

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("scatter of train data")
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.savefig('scatter_of_train_data.png')
    plt.show()


def calculate_K(X, kernel, kernel_parameter):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel(X[i], X[j], kernel_parameter)
    return K


def calculate_y_hat(K, alpha, i):
    return np.sign(np.sum(K[:, i] * alpha))


def polynomial_kernel(x, y, q=5):
    return (1 + np.dot(x, y)) ** q


def gaussian_kernel(x, y, sigma=2.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class KernelPerceptron(object):

    def __init__(self, kernel, T=1, kernel_parameter=1):
        self.kernel = kernel
        self.T = T
        self.alpha = None
        self.K = None
        self.kernel_parameter = kernel_parameter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        self.K = calculate_K(X, self.kernel, self.kernel_parameter)
        train_predictions = np.zeros([1, n_samples])

        for t in range(self.T):
            print('starting epoch number ' + str(t) + ' for ' + self.kernel.__name__)
            number_of_fails = 0
            for i in range(n_samples):
                y_hat = self.predict(i, y)
                train_predictions[0, i] = y_hat
                if y_hat != y[i]:
                    self.alpha[i] += 0.5 * (y[i] - y_hat)
                    number_of_fails += 1
            print('finished epoch number ' + str(t) + ' for ' + self.kernel.__name__ + ', number of training '
                                                                                       'fails : ' + str(
                number_of_fails))
        return train_predictions

    def predict(self, i, y):
        # return get_sign(np.sum(self.K[:, i] * self.alpha * y[i]))
        return get_sign(np.sum(self.K[:, i] * self.alpha))

    def predict_test_label(self, test_data, sample_data):
        out = 0
        n_samples, n_features = test_data.shape
        for i in range(n_samples):
            out += self.alpha[i] * self.kernel(test_data[i], sample_data)
        return get_sign(out)

    def test(self, test_data: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
        number_of_errors = 0
        n_samples, n_features = test_data.shape
        prediction_array = np.zeros(test_labels.shape)
        self.K = calculate_K(test_data, self.kernel, self.kernel_parameter)
        for sample_index in range(n_samples):
            sample_data = test_data[sample_index, :]
            y_hat = self.predict_test_label(test_data, sample_data)
            test_label = test_labels[sample_index]
            if y_hat != test_label:
                number_of_errors += 1
            prediction_array[sample_index] = y_hat
        print('number of testing fails for ' + self.kernel.__name__ + ' : ' + str(number_of_errors))
        return prediction_array

    def plot_results(self, train_data, train_labels, train_predictions, test_data, test_labels, test_predictions):
        fig, ax = plt.subplots()
        # scatter correct +1 train predictions
        correct_positive_train_labels_index = np.logical_and(train_labels == train_predictions, train_labels == 1)
        ax.scatter(train_data[correct_positive_train_labels_index[0, :], 0],
                   train_data[correct_positive_train_labels_index[0, :], 1], c='blue', label='correct +1', marker='o')
        # scatter correct train negative predictions
        correct_negative_train_labels_index = np.logical_and(train_labels == train_predictions, train_labels == -1)
        ax.scatter(train_data[correct_negative_train_labels_index[0, :], 0],
                   train_data[correct_negative_train_labels_index[0, :], 1], c='orange', label='correct -1', marker='o')
        # scatter incorrect train predictions
        incorrect_train_labels_index = train_labels != train_predictions
        ax.scatter(train_data[incorrect_train_labels_index[0, :], 0], train_data[incorrect_train_labels_index[0, :], 1],
                   c='green', label='incorrect train', marker='o')

        # scatter correct +1 test predictions
        correct_positive_test_labels_index = np.logical_and(test_labels == test_predictions, test_labels == 1)
        ax.scatter(test_data[correct_positive_test_labels_index, 0], test_data[correct_positive_test_labels_index, 1],
                   c='blue', label='correct +1', marker='*')
        # scatter correct test negative predictions
        correct_negative_test_labels_index = np.logical_and(test_labels == test_predictions, test_labels == -1)
        ax.scatter(test_data[correct_negative_test_labels_index, 0], test_data[correct_negative_test_labels_index, 1],
                   c='orange', label='correct -1', marker='*')
        # scatter incorrect test predictions
        incorrect_test_labels_index = test_labels != test_predictions
        ax.scatter(test_data[incorrect_test_labels_index, 0], test_data[incorrect_test_labels_index, 1], c='green',
                   label='incorrect test', marker='*')

        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("scatter of data and prediction for " + str(self.kernel.__name__) + " ")
        ax.legend(loc="upper left")
        ax.grid(True)
        plt.savefig('scatter of data and prediction for ' + str(self.kernel.__name__) + ' .png')
        plt.show()
