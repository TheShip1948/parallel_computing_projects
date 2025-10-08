#ifndef MATRIX_EXCEPTIONS_H
#define MATRIX_EXCEPTIONS_H

#include <stdexcept>
#include <string>

class MatrixDimensionError : public std::exception {
private:
    std::string message;
public:
    MatrixDimensionError(const std::string& msg);
    const char* what() const noexcept override;
};

#endif // MATRIX_EXCEPTIONS_H