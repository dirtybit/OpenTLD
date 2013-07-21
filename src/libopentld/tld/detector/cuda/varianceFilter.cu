/*  Copyright (c) 2013 Sertac Olgunsoylu
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

// Ugly work-around to resolve conflict between CUDA and SSE
#undef __SSE2__

#include "CUDA.h"
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <opencv2/gpu/gpu.hpp>

void createIndexArray(int * idxArr, int n) {
    thrust::device_ptr<int> dev_ptr;

    dev_ptr = thrust::device_pointer_cast(idxArr);
    thrust::sequence(dev_ptr, dev_ptr + n);
}

__global__ void __cudaVarianceFilter(cv::gpu::PtrStep<int> integralImg, cv::gpu::PtrStep<unsigned long long> integralImg_squared,
                                     int * windows_d, int * d_inWinIndices, int numInWins, float minVar)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < numInWins) {
        int winIdx = d_inWinIndices[idx];
        int * win = &windows_d[winIdx * TLD_WINDOW_SIZE];

        int x1 = win[0]+1;
        int y1 = win[1]+1;
        int w = win[2];
        int h = win[3];
        int x2 = x1 + w - 1;
        int y2 = y1 + h - 1;
        float area = w * h;

        float mX  = (integralImg(y2, x2) - integralImg(y1-1, x2) - integralImg(y2, x1-1) + integralImg(y1-1, x1-1)) / area;
        unsigned long long l = (integralImg_squared(y2, x2) - integralImg_squared(y1-1, x2) - integralImg_squared(y2, x1-1) + integralImg_squared(y1-1, x1-1));
        float mX2 = l / area;
        float variance = mX2 - mX * mX;

        if(variance < minVar)
            d_inWinIndices[idx] = -1;
    }
}

void cudaVarianceFilter(cv::gpu::GpuMat integralImg, cv::gpu::GpuMat integralImg_squared,
                               int * windows_d, int * d_inWinIndices, int &numInWins, float minVar)
{
    cudaEvent_t finished;
    cudaEventCreate(&finished);
    dim3 gridSize(ceil(numInWins / (float)VAR_FILT_BLOCK_SIZE));
    dim3 blockSize(VAR_FILT_BLOCK_SIZE);
    __cudaVarianceFilter<<<gridSize, blockSize>>>(integralImg, integralImg_squared, windows_d, d_inWinIndices, numInWins, minVar);
    cudaCheckErrors(0);
    cudaEventRecord(finished);
    cudaEventSynchronize(finished);
    thrust::device_ptr<int> idxArr = thrust::device_pointer_cast(d_inWinIndices);
    thrust::device_ptr<int> end = thrust::remove_if(idxArr, idxArr + numInWins, is_negative());
    numInWins = end - idxArr;
}
