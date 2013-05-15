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

#ifndef _CU_VARIANCEFILTER_H_
#define _CU_VARIANCEFILTER_H_

#include <thrust/device_ptr.h>

#include <opencv/cv.h>
#include <opencv2/gpu/gpu.hpp>

#include "IVarianceFilter.h"
#include "DetectionResult.h"

namespace tld
{

namespace cuda
{

class CuVarianceFilter : public IVarianceFilter
{
public:
    int *windows_d;

    CuVarianceFilter();
    virtual ~CuVarianceFilter();

    virtual void filter(const cv::gpu::GpuMat &img, int *d_inWinIndices, int &numInWins);
};

} /* namespace cuda */

} /* namespace tld */
#endif /* _CU_VARIANCEFILTER_H_ */
