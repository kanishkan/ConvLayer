#include "stdint.h"
#include "tceops.h"

// Pointer meth macro's
#define IND2(j, i, I) ((j)*(I) + (i)) // Never ever delete the () around variables!!
#define IND3(k, j, J, i, I) IND2(IND2(k, j, J), i, I)
#define IND4(l, k, K, j, J, i, I) IND2(IND3(l, k, K, j, J), i, I)

// Convolution layer dimensions
#define dimM 128
#define dimC dimM
#define dimH 22 // Assume input is already zero-padded
#define dimW dimH
#define dimR 3
#define dimS dimR

// Second address space
// #define _pmem __attribute__((address_space(1)))

// Data buffers
int8_t input[dimC*dimH*dimW]; // Assume input is already zero-padded
int8_t weight[dimM*dimC*dimR*dimS];
int32_t bias[dimM];
int8_t output[dimM*(dimH-dimR+1)*(dimW-dimS+1)];


void conv2d_relu(int8_t* restrict output, int8_t* restrict input_, int8_t* restrict weight_, int32_t* restrict bias_,
                 int M, int C, int H, int W, int R, int S) {
    int E = H - R + 1;
    int F = W - S + 1;

    for (int m = 0; m < M; m++) { // Output channels
        for (int h = 0; h < E; h++) { // Output height
            for (int w = 0; w < F; w++) { // Output width
                int32_t acc = bias_[m]; // Initialize accumulator with bias
                for (int c = 0; c < C; c++) { // Input channels
                    for (int r = 0; r < R; r++) { // Filter weight
                        for (int s = 0; s < S; s++) { // Filter width
                            acc += input_[IND3(c, h + r, H, w + s, W)] * weight_[IND4(m, c, C, r, R, s, S)];
                            // _TCE_MAC(acc, input_[IND3(c, h + r, H, w + s, W)],  weight_[IND4(m, c, C, r, R, s, S)], acc);
                        }
                    }
                }

                // ReLU activation
                if (acc < 0) {
                    acc = 0;
                }
                // Quantize back to 8-bits
                output[IND3(m, h, E, w, F)] = acc >> 11;
            }
        }
    }
}


int main() {
    conv2d_relu(output, input, weight, bias, dimM, dimC, dimH, dimW, dimR, dimS);
    return 0;
}
