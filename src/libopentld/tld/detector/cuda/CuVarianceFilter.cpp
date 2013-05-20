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

#include "CuVarianceFilter.h"

#include "IntegralImage.h"
#include "DetectorCascade.h"
#include "cuda_runtime.h"
#include "NPP_staging.hpp"
#include "opencv2/gpu/device/warp.hpp"


using namespace cv;
using namespace cv::gpu;

extern void cudaVarianceFilter(GpuMat integralImg, GpuMat integralImg_squared,
                               int * windows_d, int * d_inWinIndices, int &numInWins, float minVar);


namespace tld
{

namespace cuda
{

CuVarianceFilter::CuVarianceFilter()
{
    windows_d = NULL;
}

CuVarianceFilter::~CuVarianceFilter()
{
}

void CuVarianceFilter::integral(const GpuMat &src, GpuMat &sum)
{
    GpuMat buffer;
    integralBuffered(src, buffer, sum);
}

void CuVarianceFilter::integralBuffered(const GpuMat &src, GpuMat &buffer, GpuMat &sum)
{
    NcvSize32u roiSize;
    roiSize.width = src.cols;
    roiSize.height = src.rows;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cv::gpu::getDevice());

    Ncv32u bufSize;
    nppiStIntegralGetSize_8u32u(roiSize, &bufSize, prop);
    ensureSizeIsEnough(1, bufSize, CV_8UC1, buffer);

    nppiStIntegral_8u32u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>()), static_cast<int>(src.step), sum.ptr<Ncv32u>(), static_cast<int>(sum.step), roiSize, buffer.ptr<Ncv8u>(), bufSize, prop);

    cudaDeviceSynchronize();
}

void CuVarianceFilter::sqrIntegral(const GpuMat &src, GpuMat &sqsum)
{
    CV_Assert(src.type() == CV_8U);

    NcvSize32u roiSize;
    roiSize.width = src.cols;
    roiSize.height = src.rows;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cv::gpu::getDevice());

    Ncv32u bufSize;
    nppiStSqrIntegralGetSize_8u64u(roiSize, &bufSize, prop);
    GpuMat buf(1, bufSize, CV_8U);

    nppiStSqrIntegral_8u64u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>(0)), static_cast<int>(src.step),
            sqsum.ptr<Ncv64u>(0), static_cast<int>(sqsum.step), roiSize, buf.ptr<Ncv8u>(0), bufSize, prop);

    cudaDeviceSynchronize();
}


void CuVarianceFilter::filter(const GpuMat &img, int *d_inWinIndices, int &numInWins)
{
    if(!enabled) return;

    integral(img, integralImg);
    sqrIntegral(img, sqrIntegralImg);

//    FILE *out = fopen("./cuda_integ", "w");

//    cv::Mat integ(sqrIntegralImg);

//    for(int i = 0; i < integ.rows; i++) {
//        for(int j = 0; j < integ.cols; j++)
//            fprintf(out, "%lld ", integ.at<long long>(i, j));
//        fprintf(out, "\n");
//    }

//    exit(0);

    cudaVarianceFilter(integralImg, sqrIntegralImg, windows_d, d_inWinIndices, numInWins, minVar);
}

void CuVarianceFilter::setImgSize(int w, int h)
{
    integralImg.create(h+1, w+1, CV_32SC1);
    sqrIntegralImg.create(h+1, w+1, CV_64FC1);
}

} /* namespace cuda */

} /* namespace tld */
