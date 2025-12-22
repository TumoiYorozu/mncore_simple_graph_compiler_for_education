#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

using namespace std;

// Matrix and Vector types using std::array
template<size_t R, size_t C>
using Matrix = array<array<float, C>, R>;

template<size_t N>
using Vector = array<float, N>;

// Template-based matrix operations

// Zero initialization
template<size_t R, size_t C>
Matrix<R, C> zeros() {
    Matrix<R, C> result{};
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] = 0.0f;
        }
    }
    return result;
}

template<size_t N>
Vector<N> zeros() {
    Vector<N> result{};
    for (int i = 0; i < N; i++) {
        result[i] = 0.0f;
    }
    return result;
}

// Matrix transpose
template<size_t R, size_t C>
Matrix<C, R> trans(const Matrix<R, C>& a) {
    Matrix<C, R> result = zeros<C, R>();
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[j][i] = a[i][j];
        }
    }
    return result;
}

// Matrix subtraction
template<size_t R, size_t C>
Matrix<R, C> sub(const Matrix<R, C>& a, const Matrix<R, C>& b) {
    Matrix<R, C> result;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

// Element-wise multiplication
template<size_t R, size_t C>
Matrix<R, C> mul_elem(const Matrix<R, C>& a, const Matrix<R, C>& b) {
    Matrix<R, C> result;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

// Scalar multiplication
template<size_t R, size_t C>
Matrix<R, C> mul_constant(const Matrix<R, C>& a, float c) {
    Matrix<R, C> result;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] = a[i][j] * c;
        }
    }
    return result;
}

// Element-wise log (renamed to avoid conflict with std::log)
template<size_t R, size_t C>
Matrix<R, C> log(const Matrix<R, C>& a) {
    Matrix<R, C> result;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] = std::log(a[i][j]);
        }
    }
    return result;
}

// Element-wise exponential
template<size_t R, size_t C>
Matrix<R, C> exp(const Matrix<R, C>& x) {
    Matrix<R, C> result;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] = exp(x[i][j]);
        }
    }
    return result;
}


// Reduce sum along axis 0 (sum columns)
template<size_t R, size_t C>
Vector<C> reduce_sum(const Matrix<R, C>& a, int axis = 0) {
    if (axis != 0) {
        throw std::invalid_argument("Only axis 0 is supported for reduce_sum");
    }
    Vector<C> result = zeros<C>();
    for (int j = 0; j < C; j++) {
        float sum = 0.0f;
        for (int i = 0; i < R; i++) {
            sum += a[i][j];
        }
        result[j] = sum;
    }
    return result;
}

// Sum all elements
template<size_t R, size_t C>
float sum(const Matrix<R, C>& a) {
    float sum = 0.0f;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            sum += a[i][j];
        }
    }
    return sum;
}

// Vector subtraction
template<size_t N>
Vector<N> sub(const Vector<N>& a, const Vector<N>& b) {
    Vector<N> result;
    for (int i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Vector scalar multiplication
template<size_t N>
Vector<N> mul_constant(const Vector<N>& a, float c) {
    Vector<N> result;
    for (int i = 0; i < N; i++) {
        result[i] = a[i] * c;
    }
    return result;
}

// Matrix multiplication with bias: C = A * B + bias
template<int R, int K, int C>
Matrix<R, C> Gemm(const Matrix<R, K>& a, const Matrix<K, C>& b, const Vector<C>& bias) {
    Matrix<R, C> result = zeros<R, C>();
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum + bias[j];
        }
    }
    return result;
}

// Matrix multiplication without bias
template<int R, int K, int C>
Matrix<R, C> matmul(const Matrix<R, K>& a, const Matrix<K, C>& b) {
    Vector<C> zero_bias = zeros<C>();
    return Gemm<R, K, C>(a, b, zero_bias);
}

// ReLU activation
template<size_t R, size_t C>
Matrix<R, C> relu(const Matrix<R, C>& x) {
    Matrix<R, C> result = x;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] = max(0.0f, result[i][j]);
        }
    }
    return result;
}

// ReLU gradient
template<size_t R, size_t C>
Matrix<R, C> relu_grad(const Matrix<R, C>& grad, const Matrix<R, C>& x) {
    Matrix<R, C> result = grad;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            if (x[i][j] <= 0) result[i][j] = 0;
        }
    }
    return result;
}

template<size_t R, size_t C>
Matrix<R, C> div_rowvec(const Matrix<R, C>& mat, const Vector<R>& vec) {
    Matrix<R, C> result = mat;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] /= vec[i];
        }
    }
    return result;
}

// Broadcast addition: Matrix + Vector (adds vector to each row)
template<size_t R, size_t C>
Matrix<R, C> add_colvec(const Matrix<R, C>& mat, const Vector<C>& vec) {
    Matrix<R, C> result = mat;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] += vec[j];
        }
    }
    return result;
}

template<size_t R, size_t C>
Matrix<R, C> sub_rowvec(const Matrix<R, C>& mat, const Vector<R>& vec) {
    Matrix<R, C> result = mat;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i][j] -= vec[i];
        }
    }
    return result;
}


template<size_t R, size_t C>
Vector<R> row_max(const Matrix<R, C>& x) {
    Vector<R> result;
    for (int i = 0; i < R; i++) {
        result[i] = *max_element(x[i].begin(), x[i].end());
    }
    return result;
}

template<size_t R, size_t C>
Vector<R> row_sum(const Matrix<R, C>& a) {
    Vector<R> result = zeros<R>();
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[i] += a[i][j];
        }
    }
    return result;
}


// Column-wise sum (sum along rows for each column)
template<size_t R, size_t C>
Vector<C> col_sum(const Matrix<R, C>& a) {
    Vector<C> result = zeros<C>();
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            result[j] += a[i][j];
        }
    }
    return result;
}

template<size_t R, size_t C>
float gather_sum(const Matrix<R, C> &A, const std::array<int, R> &I) {
    float res = 0.0f;
    for (int i = 0; i < R; i++) {
        res += A[i][I[i]];
    }
    return res;
}


// Softmax
// 256, 16
template<size_t R, size_t C>
Matrix<R, C> softmax(const Matrix<R, C>& x) {
    Vector<R> max_vals = row_max(x);
    Matrix<R, C> x_shifted = sub_rowvec(x, max_vals);
    Matrix<R, C> exp_x = exp(x_shifted);
    Vector<R> sum_exp = row_sum(exp_x);
    Matrix<R, C> result = div_rowvec(exp_x, sum_exp);
    return result;
}

// Memory I/O operations
template<size_t R, size_t C, typename T>
array<array<T, C>, R> load(const T* ptr) {
    array<array<T, C>, R> mat;
    memcpy(mat.data(), ptr, R * C * sizeof(T));
    return mat;
}

template<size_t N, typename T>
array<T, N> load(const T* ptr) {
    array<T, N> vec;
    memcpy(vec.data(), ptr, N * sizeof(T));
    return vec;
}

template<size_t R, size_t C, typename T>
void save(const array<array<T, C>, R>& mat, T* ptr) {
    memcpy(ptr, mat.data(), R * C * sizeof(T));
}

template<size_t N, typename T>
void save(const array<T, N>& vec, T* ptr) {
    memcpy(ptr, vec.data(), N * sizeof(T));
}

// Save scalar value
template<typename T>
inline void save(const T v, T* ptr) {
    *ptr = v;
}
