import os
import utils
from utils import KernelPerceptron, gaussian_kernel, polynomial_kernel


def main():
    train_data, train_labels, test_data, test_labels = utils.load_data()
    utils.scatter_train_data(train_data, train_labels)

    # gaussian_perceptron
    sigma = 1
    gaussian_perceptron = KernelPerceptron(kernel=gaussian_kernel, T=50, kernel_parameter=sigma)
    train_predictions = gaussian_perceptron.fit(train_data, train_labels)
    test_predictions = gaussian_perceptron.test(test_data, test_labels)
    gaussian_perceptron.plot_results(train_data, train_labels, train_predictions, test_data, test_labels,
                                     test_predictions)

    # polynomial_perceptron
    q = 5
    polynomial_perceptron = KernelPerceptron(kernel=polynomial_kernel, T=50, kernel_parameter=q)
    train_predictions = polynomial_perceptron.fit(train_data, train_labels)
    test_predictions = polynomial_perceptron.test(test_data, test_labels)
    polynomial_perceptron.plot_results(train_data, train_labels, train_predictions, test_data, test_labels,
                                       test_predictions)


if __name__ == '__main__':
    main()
