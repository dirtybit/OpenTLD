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
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <opencv2/gpu/gpu.hpp>

__global__ void __calcTreeConfidence(cv::gpu::PtrStep<unsigned char> img, int *windows_d,
                                     int *d_inWinIndices, int numInWins, float *features_d,
                                     float *posteriors_d, float *confidences_d, int numTrees, int numFeatures, int numIndices)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int id = idx / 16;

    if(id < numInWins) {
        int treeIdx = idx - id * 16;
        int winIdx = d_inWinIndices[id];

        int * win = &windows_d[winIdx * TLD_WINDOW_SIZE];
        int x = win[0];
        int y = win[1];
        int w = win[2];
        int h = win[3];

        float * feature;
        int fx1;
        int fy1;
        int fx2;
        int fy2;

        int index = 0;

        for(int i = 0; i < numFeatures; i++)
        {
            feature = &features_d[4 * numFeatures * treeIdx + 4 * i];
            fx1 = x + (w - 1) * feature[0];
            fy1 = y + (h - 1) * feature[1];
            fx2 = x + (w - 1) * feature[2];
            fy2 = y + (h - 1) * feature[3];
            index <<= 1;

            int fp0 = img(fy1, fx1);
            int fp1 = img(fy2, fx2);

            if(fp0 > fp1)
            {
                index |= 1;
            }
        }

        float conf = posteriors_d[treeIdx * numIndices + index];
        confidences_d[id * numTrees + treeIdx] = conf;
    }
}

__global__ void __classifyWindows(int *d_inWinIndices, int numInWins, float *confidences_d, int numTrees)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < numInWins) {
        float conf = 0.0;

        for(int i = 0; i < numTrees; i++)
        {
            conf += confidences_d[idx * numTrees + i];
        }

        if(conf < 0.5)
            d_inWinIndices[idx] = -1;
    }
}


void cudaEnsembleClassifier(cv::gpu::GpuMat img, int *windows_d, int *d_inWinIndices,
                            int &numInWins, float *features_d, float*posteriors_d, float *confidences_d, int numTrees, int numFeatures, int numIndices)
{
    cudaEvent_t finished;
    cudaEventCreate(&finished);
    dim3 gridSize((numInWins + 11)/12);
    dim3 blockSize(ENS_CLS_BLOCK_SIZE);
    __calcTreeConfidence<<<gridSize, blockSize>>>(img, windows_d, d_inWinIndices, numInWins, features_d, posteriors_d, confidences_d, numTrees, numFeatures, numIndices);
    gridSize.x = (numInWins + ENS_CLS_BLOCK_SIZE - 1) / ENS_CLS_BLOCK_SIZE;
    __classifyWindows<<<gridSize, blockSize>>>(d_inWinIndices, numInWins, confidences_d, numTrees);
    cudaCheckErrors(0);
    cudaEventRecord(finished);
    cudaEventSynchronize(finished);

    thrust::device_ptr<int> idxArr = thrust::device_pointer_cast(d_inWinIndices);
    thrust::device_ptr<int> end = thrust::remove_if(idxArr, idxArr + numInWins, is_negative());
    numInWins = end - idxArr;
}

