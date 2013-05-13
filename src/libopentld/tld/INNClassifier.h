/*  Copyright 2011 AIT Austrian Institute of Technology
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
/*
 * INNClassifier.h
 *
 *  Created on: May 13, 2013
 *      Author: Sertac Olgunsoylu
 */

#ifndef INNCLASSIFIER_H_
#define INNCLASSIFIER_H_

#include <vector>

#include <opencv/cv.h>

#include "NormalizedPatch.h"
#include "DetectionResult.h"

namespace tld
{

class INNClassifier
{
public:
    bool enabled;

    int *windows;
    float thetaFP;
    float thetaTP;
    DetectionResult *detectionResult;
    std::vector<NormalizedPatch>* falsePositives;
    std::vector<NormalizedPatch>* truePositives;

    virtual void release() = 0;
    virtual float classifyBB(const cv::Mat &img, cv::Rect *bb) = 0;
    virtual void learn(std::vector<NormalizedPatch> patches) = 0;
};

} /* namespace tld */
#endif /* INNCLASSIFIER_H_ */
