#include <MiniDNN.h>
#include "../../utils/mnist_loader.h"
#include <Eigen/Core>
#include <iostream>
#include <vector>

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

    std::cout << "Test size: " << test_data.size() << std::endl;

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

    // Map to Eigen format for the first 5 images
    int n_test = 5;
    int n_pixels = test_data.rows() * test_data.cols();

    Matrix x_test(n_pixels, n_test);
    Eigen::RowVectorXi y_test(n_test);

    for (int i = 0; i < n_test; i++) {
        std::vector<double> image = test_data.images(i);
        for (int j = 0; j < n_pixels; j++) {
            x_test(j, i) = image[j];
        }
        y_test(i) = test_data.labels(i);
    }

    // Predict
    std::cout << "Predicting 5 images..." << std::endl;
    Matrix pred = net.predict(x_test);

    // Report results
    std::cout << "-------------------------------------------" << std::endl;
    for (int i = 0; i < n_test; i++) {
        int max_idx;
        pred.col(i).maxCoeff(&max_idx);
        
        int actual = y_test(i);
        int predicted = max_idx;
        bool is_correct = (actual == predicted);

        std::cout << "Image " << i + 1 << ":" << std::endl;
        std::cout << "  Actual Number:    " << actual << std::endl;
        std::cout << "  Predicted Number: " << predicted << std::endl;
        std::cout << "  Result:           " << (is_correct ? "TRUE" : "FALSE") << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
    }

    return 0;
}
