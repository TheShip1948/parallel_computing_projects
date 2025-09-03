#include "matrix.h"

int main() {
    std::vector<std::vector<int>> A = {{1,2,3},{4,5,6}};
    std::vector<std::vector<int>> B = {{7,8},{9,10},{11,12}};

    auto result = multiply_matrices(A, B);
    printMatrix(result);
}
