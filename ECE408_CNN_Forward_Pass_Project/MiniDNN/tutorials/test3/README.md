# MNIST Training Tutorial (test3)

This tutorial demonstrates how to train a LeNet-5 style Convolutional Neural Network (CNN) on the MNIST digit dataset using the MiniDNN library.

## Objective
- Load the MNIST training and test datasets.
- Construct a CNN architecture (Convolutional, MaxPooling, and FullyConnected layers).
- Train the model using the RMSProp optimizer.
- Evaluate the model's accuracy on the test set.
- **Save the trained model weights** to the `NetFolder` for future use.

## Architecture
The network follows a classic LeNet-5 structure:
1. **Convolutional Layer**: 28x28x1 -> 24x24x6 (5x5 filter + ReLU)
2. **MaxPooling Layer**: 24x24x6 -> 12x12x6 (2x2 pool + ReLU)
3. **Convolutional Layer**: 12x12x6 -> 8x8x16 (5x5 filter + ReLU)
4. **MaxPooling Layer**: 8x8x16 -> 4x4x16 (2x2 pool + ReLU)
5. **FullyConnected Layer**: 256 -> 120 (ReLU)
6. **FullyConnected Layer**: 120 -> 84 (ReLU)
7. **FullyConnected Layer**: 84 -> 10 (Softmax)

## How to Build and Run
On Windows (with CUDA/NVCC installed):
1. Run `build.bat`.
2. The executable will be created in `build/test3.exe`.
3. The program will automatically train for 5 epochs and save the results to `NetFolder/`.
