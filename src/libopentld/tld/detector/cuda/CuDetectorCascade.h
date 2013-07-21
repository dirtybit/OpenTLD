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

#ifndef CU_DETECTORCASCADE_H_
#define CU_DETECTORCASCADE_H_

#include "CUDA.h"

#include "IDetectorCascade.h"
#include "CuVarianceFilter.h"
#include "CuEnsembleClassifier.h"
#include "Clustering.h"
#include "NNClassifier.h"


namespace tld
{

namespace cuda
{

class CuDetectorCascade : public IDetectorCascade
{
    int *windows_d;
    int * d_inWinIndices;
    int * qualifiedWins;

public:
    CuDetectorCascade();
    ~CuDetectorCascade();

    void init();
    void initWindowOffsets();
    void initWindowsAndScales();
    void propagateMembers();
    void release();
    void cleanPreviousData();
    void detect(const cv::Mat &img);
    void setImgSize(int w, int h, int step);
};

} /* namespace cuda */

} /* namespace tld */
#endif /* CU_DETECTORCASCADE_H_ */
