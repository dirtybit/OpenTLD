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
 * IEnsembleClassifier.h
 *
 *  Created on: May 13, 2013
 *      Author: Sertac Olgunsoylu
 */

#ifndef IENSEMBLECLASSIFIER_H_
#define IENSEMBLECLASSIFIER_H_

#include <opencv/cv.h>

namespace tld
{

class IEnsembleClassifier
{
public:
    bool enabled;

    //Configurable members
    int numTrees;
    int numFeatures;

    int imgWidthStep;
    int numScales;
    cv::Size *scales;

    int *windowOffsets;
    int *featureOffsets;
    float *features;

    int numIndices;

    float *posteriors;
    int *positives;
    int *negatives;

    DetectionResult *detectionResult;

    virtual void init() = 0;
    virtual void initFeatureLocations() = 0;
    virtual void initFeatureOffsets() = 0;
    virtual void initPosteriors() = 0;
    virtual void release() = 0;
    virtual void updatePosterior(int treeIdx, int idx, int positive, int amount) = 0;
    virtual void learn(int *boundary, int positive, int *featureVector) = 0;
};

} /* namespace tld */
#endif /* IENSEMBLECLASSIFIER_H_ */
