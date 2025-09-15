#include "matrix_exceptions.h"

MatrixDimensionError::MatrixDimensionError(const std::string& msg) : message(msg) {}

const char* MatrixDimensionError::what() const noexcept {
    return message.c_str();
}