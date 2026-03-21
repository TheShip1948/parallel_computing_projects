#include <MiniDNN.h>
#include "../../utils/mnist_loader.h"
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <chrono>

using namespace MiniDNN;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

int main()
{
    // Load MNIST test data
    // Paths are relative to the execution directory (tutorials/test4)
    std::string test_images_path = "../../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
    std::string test_labels_path = "../../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";

    std::cout << "Loading MNIST test data..." << std::endl;
    mnist_loader test_data(test_images_path, test_labels_path);

    if (test_data.size() == 0) {
        std::cerr << "Failed to load MNIST test data!" << std::endl;
        return 1;
    }

    int n_test_original = test_data.size();
    int n_test = n_test_original * 5;
    std::cout << "Original Test size: " << n_test_original << std::endl;
    std::cout << "Extended Test size: " << n_test << " (5x repeat)" << std::endl;

    // Create a network object
    Network net;

    // Load the pre-trained network from NetFolder
    std::cout << "Loading network from NetFolder..." << std::endl;
    try {
        net.read_net("NetFolder", "NetFile");
    } catch (const std::exception& e) {
        std::cerr << "Failed to load network: " << e.what() << std::endl;
        return 1;
    }

    // Map to Eigen format for all test images (repeated 5 times)
    int n_pixels = test_data.rows() * test_data.cols();

    Matrix x_test(n_pixels, n_test);
    Eigen::RowVectorXi y_test(n_test);

    for (int i = 0; i < n_test; i++) {
        std::vector<double> image = test_data.images(i % n_test_original);
        for (int j = 0; j < n_pixels; j++) {
            x_test(j, i) = image[j];
        }
        y_test(i) = test_data.labels(i % n_test_original);
    }

    // Predict with timing
    std::cout << "Predicting " << n_test << " images..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix pred = net.predict(x_test);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;

    // Report results
    int correct_count = 0;
    for (int i = 0; i < n_test; i++) {
        int max_idx;
        pred.col(i).maxCoeff(&max_idx);
        if (max_idx == y_test(i)) {
            correct_count++;
        }
    }

    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Prediction Step Time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Correct Predictions:  " << correct_count << " / " << n_test << std::endl;
    std::cout << "Accuracy:             " << (double)correct_count / n_test * 100.0 << "%" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    return 0;
}
