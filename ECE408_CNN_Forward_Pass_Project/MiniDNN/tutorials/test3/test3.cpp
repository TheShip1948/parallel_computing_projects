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
    // Load MNIST data
    // Paths are relative to the execution directory (tutorials/test3)
    std::string train_images_path = "../../data/train-images-idx3-ubyte/train-images.idx3-ubyte";
    std::string train_labels_path = "../../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
    std::string test_images_path = "../../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
    std::string test_labels_path = "../../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";

    std::cout << "Loading MNIST data..." << std::endl;
    mnist_loader train_data(train_images_path, train_labels_path);
    mnist_loader test_data(test_images_path, test_labels_path);

    if (train_data.size() == 0 || test_data.size() == 0) {
        std::cerr << "Failed to load MNIST data!" << std::endl;
        return 1;
    }

    std::cout << "Train size: " << train_data.size() << std::endl;
    std::cout << "Test size: " << test_data.size() << std::endl;

    // Map to Eigen format
    // x: pixels x observations
    // y: 1 x observations (labels)
    int n_train = train_data.size();
    int n_test = test_data.size();
    int n_pixels = train_data.rows() * train_data.cols();

    Matrix x_train(n_pixels, n_train);
    Eigen::RowVectorXi y_train(n_train);

    for (int i = 0; i < n_train; i++) {
        std::vector<double> image = train_data.images(i);
        for (int j = 0; j < n_pixels; j++) {
            x_train(j, i) = image[j];
        }
        y_train(i) = train_data.labels(i);
    }

    Matrix x_test(n_pixels, n_test);
    Eigen::RowVectorXi y_test(n_test);

    for (int i = 0; i < n_test; i++) {
        std::vector<double> image = test_data.images(i);
        for (int j = 0; j < n_pixels; j++) {
            x_test(j, i) = image[j];
        }
        y_test(i) = test_data.labels(i);
    }

    // Create a LeNet-5 style network
    Network net;
    // Layer 1: Conv (28x28x1 -> 24x24x6, 5x5 filter)
    Layer* layer1 = new Convolutional<ReLU>(28, 28, 1, 6, 5, 5);
    // Layer 2: Max Pool (24x24x6 -> 12x12x6, 2x2 pool)
    Layer* layer2 = new MaxPooling<ReLU>(24, 24, 6, 2, 2);
    // Layer 3: Conv (12x12x6 -> 8x8x16, 5x5 filter)
    Layer* layer3 = new Convolutional<ReLU>(12, 12, 6, 16, 5, 5);
    // Layer 4: Max Pool (8x8x16 -> 4x4x16, 2x2 pool)
    Layer* layer4 = new MaxPooling<ReLU>(8, 8, 16, 2, 2);
    // Layer 5: FC (256 -> 120)
    Layer* layer5 = new FullyConnected<ReLU>(4 * 4 * 16, 120);
    // Layer 6: FC (120 -> 84)
    Layer* layer6 = new FullyConnected<ReLU>(120, 84);
    // Layer 7: FC (84 -> 10)
    Layer* layer7 = new FullyConnected<Softmax>(84, 10);

    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);
    net.add_layer(layer4);
    net.add_layer(layer5);
    net.add_layer(layer6);
    net.add_layer(layer7);

    // Set output layer (cross-entropy for multi-class classification)
    net.set_output(new MultiClassEntropy());

    // Optimizer
    RMSProp opt;
    opt.m_lrate = 0.001;

    // Callback for progress
    VerboseCallback callback;
    net.set_callback(callback);

    // Initialize parameters
    net.init(0, 0.01, 123);

    // Train the model
    std::cout << "Starting training..." << std::endl;
    // Batch size 128, 5 epochs
    net.fit(opt, x_train, y_train, 128, 5, 123);

    // Evaluate on test data
    Matrix pred = net.predict(x_test);
    int correct = 0;
    for (int i = 0; i < n_test; i++) {
        int max_idx;
        pred.col(i).maxCoeff(&max_idx);
        if (max_idx == y_test(i)) correct++;
    }

    std::cout << "Test accuracy: " << (double)correct / n_test * 100.0 << "%" << std::endl;

    // Export the network to NetFolder
    std::cout << "Saving network to NetFolder..." << std::endl;
    net.export_net("./NetFolder/", "NetFile");

    return 0;
}
