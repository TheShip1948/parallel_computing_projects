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

    std::vector<int> counts = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};


    int n_pixels = test_data.rows() * test_data.cols();

    struct Result {
        int count;
        double cpu_time;
        double gpu_time;
        double cpu_l0_time;
        double cpu_l2_time;
        double gpu_l0_time;
        double gpu_l2_time;
        double cpu_conv_time;
        double gpu_conv_time;
        double cpu_acc;
        double gpu_acc;
    };
    std::vector<Result> results;

    for (int n_test : counts) {
        std::cout << "\n===========================================" << std::endl;
        std::cout << "Benchmarking with " << n_test << " images..." << std::endl;
        std::cout << "===========================================" << std::endl;

        // Map to Eigen format
        Matrix x_test(n_pixels, n_test);
        Eigen::RowVectorXi y_test(n_test);

        for (int i = 0; i < n_test; i++) {
            std::vector<double> image = test_data.images(i % n_test_original);
            for (int j = 0; j < n_pixels; j++) {
                x_test(j, i) = image[j];
            }
            y_test(i) = test_data.labels(i % n_test_original);
        }

        std::vector<const Layer*> layers = net.get_layers();

        // --- CPU Prediction ---
        std::cout << "Running CPU strategy..." << std::endl;
        for (const Layer* const_layer : layers) {
            if (const_layer->layer_type() == "Convolutional") {
                Layer* layer = const_cast<Layer*>(const_layer);
                auto conv_layer = dynamic_cast<Convolutional<ReLU>*>(layer);
                if (conv_layer) {
                    conv_layer->set_strategy(new Convolutional<ReLU>::CPUForwardStrategy());
                }
            }
        }

        auto start_cpu = std::chrono::high_resolution_clock::now();
        Matrix pred_cpu = net.predict(x_test);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

        double cpu_conv_time = 0;
        double cpu_l0_time = 0;
        double cpu_l2_time = 0;
        std::cout << "CPU Convolution Layer Breakdown:" << std::endl;
        int layer_idx = 0;
        for (const Layer* const_layer : layers) {
            if (const_layer->layer_type() == "Convolutional") {
                auto conv_layer = dynamic_cast<const Convolutional<ReLU>*>(const_layer);
                if (conv_layer) {
                    double l_time = conv_layer->get_last_conv_time() / 1000.0; // convert ms to s
                    cpu_conv_time += l_time;
                    if (layer_idx == 0) cpu_l0_time = l_time;
                    if (layer_idx == 2) cpu_l2_time = l_time;
                    std::cout << "  Layer " << layer_idx << " (Conv): " << l_time << " s" << std::endl;
                }
            }
            layer_idx++;
        }
        std::cout << "Total CPU Conv Time: " << cpu_conv_time << " s" << std::endl;

        // --- GPU Prediction ---
        std::cout << "Running GPU strategy..." << std::endl;
        for (const Layer* const_layer : layers) {
            if (const_layer->layer_type() == "Convolutional") {
                Layer* layer = const_cast<Layer*>(const_layer);
                auto conv_layer = dynamic_cast<Convolutional<ReLU>*>(layer);
                if (conv_layer) {
                    conv_layer->set_strategy(new Convolutional<ReLU>::GPUForwardStrategy());
                }
            }
        }

        auto start_gpu = std::chrono::high_resolution_clock::now();
        Matrix pred_gpu = net.predict(x_test);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

        double gpu_conv_time = 0;
        double gpu_l0_time = 0;
        double gpu_l2_time = 0;
        std::cout << "GPU Convolution Layer Breakdown:" << std::endl;
        layer_idx = 0;
        for (const Layer* const_layer : layers) {
            if (const_layer->layer_type() == "Convolutional") {
                auto conv_layer = dynamic_cast<const Convolutional<ReLU>*>(const_layer);
                if (conv_layer) {
                    double l_time = conv_layer->get_last_conv_time() / 1000.0; // convert ms to s
                    gpu_conv_time += l_time;
                    if (layer_idx == 0) gpu_l0_time = l_time;
                    if (layer_idx == 2) gpu_l2_time = l_time;
                    std::cout << "  Layer " << layer_idx << " (Conv): " << l_time << " s" << std::endl;
                }
            }
            layer_idx++;
        }
        std::cout << "Total GPU Conv Time: " << gpu_conv_time << " s" << std::endl;

        // Calc accuracy
        int correct_cpu = 0;
        int correct_gpu = 0;
        for (int i = 0; i < n_test; i++) {
            int max_idx_cpu, max_idx_gpu;
            pred_cpu.col(i).maxCoeff(&max_idx_cpu);
            pred_gpu.col(i).maxCoeff(&max_idx_gpu);
            if (max_idx_cpu == y_test(i)) correct_cpu++;
            if (max_idx_gpu == y_test(i)) correct_gpu++;
        }

        results.push_back({
            n_test, 
            cpu_time, 
            gpu_time, 
            cpu_l0_time,
            cpu_l2_time,
            gpu_l0_time,
            gpu_l2_time,
            cpu_conv_time,
            gpu_conv_time,
            (double)correct_cpu / n_test * 100.0, 
            (double)correct_gpu / n_test * 100.0
        });

        std::cout << "Batch " << n_test << " done. Speedup: " << cpu_time / gpu_time << "x (Conv Speedup: " << cpu_conv_time / gpu_conv_time << "x)" << std::endl;
    }

    // Final Report
    std::cout << "\n\n" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "FINAL PERFORMANCE REPORT" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    
    // Header
    printf("%-10s | %-12s | %-12s | %-10s | %-10s | %-10s | %-10s | %-8s | %-8s | %-8s | %-10s\n", 
           "Images", "CPU Time(s)", "GPU Time(s)", "C-L0(s)", "C-L2(s)", "G-L0(s)", "G-L2(s)", "L0-X", "L2-X", "Conv-X", "Total-X");
    
    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& res : results) {
        double l0_x = res.cpu_l0_time / res.gpu_l0_time;
        double l2_x = res.cpu_l2_time / res.gpu_l2_time;
        double conv_x = res.cpu_conv_time / res.gpu_conv_time;
        double total_x = res.cpu_time / res.gpu_time;

        printf("%-10d | %-12.4f | %-12.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-8.2fx | %-8.2fx | %-8.2fx | %-10.2fx\n", 
               res.count, res.cpu_time, res.gpu_time, res.cpu_l0_time, res.cpu_l2_time, res.gpu_l0_time, res.gpu_l2_time, l0_x, l2_x, conv_x, total_x);
    }
    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    return 0;
}
