#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int TLD_WINDOW_SIZE = 5;

#define VAR_FILT_BLOCK_SIZE 192
#define ENS_CLS_BLOCK_SIZE 192

#ifdef NO_ERROR_CHECK
    #define cudaCheckErrors(msg)
#else
    #define cudaCheckErrors(msg) \
        do { \
            cudaError_t __err = cudaGetLastError(); \
            if(__err != cudaSuccess) { \
                fprintf(stderr, "Error:%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(__err)); \
            if((msg)) \
                exit(1); \
            } \
        } while (0)
#endif

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if(__err != cudaSuccess) { \
            fprintf(stderr, "Error:%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(__err)); \
        if((msg)) \
            exit(1); \
        } \
    } while (0)

struct is_negative
{
    __host__ __device__
    bool operator()(const int x)
    {
        return x < 0;
    }
};

#endif /* CUDA_COMMON_H_ */
