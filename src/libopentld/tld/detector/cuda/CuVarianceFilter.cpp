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


using namespace cv;
using namespace cv::gpu;

extern void cudaVarianceFilter(PtrStepSz<int> integralImg, PtrStep<double> integralImg_squared,
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

void CuVarianceFilter::filter(const GpuMat &img, int *d_inWinIndices, int &numInWins)
{
    cv::gpu::GpuMat integralImg;
    cv::gpu::GpuMat integralImg_squared;

    if(!enabled) return;
    gpu::integral(img, integralImg);
    gpu::sqrIntegral(img, integralImg_squared);

    cudaVarianceFilter(integralImg, integralImg_squared, windows_d, d_inWinIndices, numInWins, minVar);
}

} /* namespace cuda */

} /* namespace tld */
