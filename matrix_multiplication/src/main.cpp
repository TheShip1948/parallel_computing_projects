#include "matrix_multiply_brute_force.h" 
#include "matrix_multiply_rows.h"
#include "matrix_multiplier_cacheoblivious.h"

int main() {
    std::vector<std::vector<int>> A = {{1,2,3},{4,5,6}};
    std::vector<std::vector<int>> B = {{7,8},{9,10},{11,12}};

    auto result = multiply_matrices_brute_force(A, B);
    printMatrix(result);

    MatrixMultiplierRows  multiplier(A, B);
    result = multiplier.parallel_multiply_rows(4);
    printMatrix(result);

    MatrixMultiplierCacheOblivious multiplier_cacheoblivious(A, B);
    result = multiplier_cacheoblivious.parallel_multiply_cacheoblivious(4);
    printMatrix(result);
    
    return 0;
}
