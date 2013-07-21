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
 * IDetectorCascade.h
 *
 *  Created on: May 13, 2013
 *      Author: Sertac Olgunsoylu
 */

#ifndef IDETECTORCASCADE_H_
#define IDETECTORCASCADE_H_

#include "DetectionResult.h"
//#include "ForegroundDetector.h"
#include "IVarianceFilter.h"
#include "IEnsembleClassifier.h"
#include "INNClassifier.h"
#include "Clustering.h"


namespace tld
{

//Constants
static const int TLD_WINDOW_SIZE = 5;
static const int TLD_WINDOW_OFFSET_SIZE = 6;

//TODO: Convert this to a function
#define sub2idx(x,y,imgWidthStep) ((int) (floor((x)+0.5) + floor((y)+0.5)*(imgWidthStep)))

class IDetectorCascade
{
protected:
    //Working data
    int numScales;
    cv::Size *scales;
public:
    //Configurable members
    int minScale;
    int maxScale;
    bool useShift;
    float shift;
    int minSize;
    int numFeatures;
    int numTrees;

    //Needed for init
    int imgWidth;
    int imgHeight;
    int imgWidthStep;
    int objWidth;								
    int objHeight;								

    int numWindows;
    int *windows;
    int *windowOffsets;

    //State data
    bool initialised;

    DetectionResult *detectionResult;

    //Components of Detector Cascade
    //ForegroundDetector *foregroundDetector;
    IVarianceFilter *varianceFilter;
    IEnsembleClassifier *ensembleClassifier;
    Clustering *clustering;
    INNClassifier *nnClassifier;

    virtual void init() = 0;
    virtual void initWindowOffsets() = 0;
    virtual void initWindowsAndScales() = 0;
    virtual void propagateMembers() = 0;
    virtual void release() = 0;						
    virtual void cleanPreviousData() = 0;			
    virtual void detect(const cv::Mat &img) = 0;	
    virtual void setImgSize(int w, int h, int step) { imgWidth = w; imgHeight = h; imgWidthStep = step; }
};

} /* namespace tld */
#endif /* IDETECTORCASCADE_H_ */
