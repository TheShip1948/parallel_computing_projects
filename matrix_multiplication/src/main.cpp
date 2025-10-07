#include "matrix_multiply_brute_force.h" 
#include "matrix_multiply_rows.h"
#include "matrix_multiplier_cacheoblivious.h"
#include "utility.h"

int main() {
    std::vector<std::vector<int>> A = {{1,2,3},{4,5,6}};
    std::vector<std::vector<int>> B = {{7,8},{9,10},{11,12}};

    MatrixMultiplierBruteForce multiplier_brute_force(A,B); 
    auto result = multiplier_brute_force.multiply_matrices_brute_force();
    utils::print_matrix(result);

    MatrixMultiplierRows  multiplier(A, B);
    result = multiplier.parallel_multiply_rows(4);
    utils::print_matrix(result);

    MatrixMultiplierCacheOblivious multiplier_cacheoblivious(A, B);
    result = multiplier_cacheoblivious.parallel_multiply_cacheoblivious(4);
    utils::print_matrix(result);
    
    return 0;
}
