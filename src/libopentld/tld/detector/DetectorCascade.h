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
 * DetectorCascade.h
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 *
 *  Modified on: May 13, 2013
 *      Author: Sertac Olgunsoylu
 */

#ifndef DETECTORCASCADE_H_
#define DETECTORCASCADE_H_

#include "IDetectorCascade.h"
#include "DetectionResult.h"
//#include "ForegroundDetector.h"
#include "VarianceFilter.h"
#include "EnsembleClassifier.h"
#include "Clustering.h"
#include "NNClassifier.h"


namespace tld
{

class DetectorCascade : public IDetectorCascade
{
public:

    DetectorCascade();
    virtual ~DetectorCascade();

    virtual void init();
    virtual void initWindowOffsets();
    virtual void initWindowsAndScales();
    virtual void propagateMembers();
    virtual void release();
    virtual void cleanPreviousData();
    virtual void detect(const cv::Mat &img);
};

} /* namespace tld */
#endif /* DETECTORCASCADE_H_ */
