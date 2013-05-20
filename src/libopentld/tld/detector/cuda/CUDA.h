#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
