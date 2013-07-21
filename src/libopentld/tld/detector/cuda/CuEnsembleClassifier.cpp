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

#include "DetectorCascade.h"

#include <cstdlib>
#include <cmath>

#include <opencv/cv.h>
#include "CUDA.h"

#include "CuEnsembleClassifier.h"

extern void cudaEnsembleClassifier(GpuMat img, int *windows_d, int *d_inWinIndices,
                                   int &numInWins, float *features_d, float*posteriors_d, float *confidences_d, int numTrees, int numFeatures, int numIndices);

using namespace std;
using namespace cv;

namespace tld
{

namespace cuda
{

CuEnsembleClassifier::CuEnsembleClassifier()
{
    features = NULL;
    featureOffsets = NULL;
    posteriors = NULL;
    positives = NULL;
    negatives = NULL;
    numTrees = 10;
    numFeatures = 13;
    enabled = true;
    features_d = NULL;
}

CuEnsembleClassifier::~CuEnsembleClassifier()
{
    release();
}

void CuEnsembleClassifier::init()
{
    numIndices = pow(2.0f, numFeatures);

    initFeatureLocations();
    initFeatureOffsets();
    initPosteriors();
}

void CuEnsembleClassifier::initGPU()
{
    int feature_size = 2 * 2 * numFeatures * numTrees * sizeof(float);
    int posterior_size = numTrees * numIndices * sizeof(float);
    int confidences_size = numWindows * numTrees * sizeof(float);

    cudaMalloc((void **) &features_d, feature_size);
    cudaMalloc((void **) &posteriors_d, posterior_size);
    cudaMalloc((void **) &confidences_d, confidences_size);
    cudaMemcpy(features_d, features, feature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(posteriors_d, posteriors, posterior_size, cudaMemcpyHostToDevice);

}

void CuEnsembleClassifier::release()
{
    delete[] features;
    features = NULL;
    delete[] featureOffsets;
    featureOffsets = NULL;
    delete[] posteriors;
    posteriors = NULL;
    delete[] positives;
    positives = NULL;
    delete[] negatives;
    negatives = NULL;
    cudaFree(features_d);
    features_d = NULL;
}

/*
 * Generates random measurements in the format <x1,y1,x2,y2>
 */
void CuEnsembleClassifier::initFeatureLocations()
{
    int size = 2 * 2 * numFeatures * numTrees;

    features = new float[size];

    for(int i = 0; i < size; i++)
    {
        features[i] = rand() / (float)RAND_MAX;
    }

}

//Creates offsets that can be added to bounding boxes
//offsets are contained in the form delta11, delta12,... (combined index of dw and dh)
//Order: scale.tree->feature
void CuEnsembleClassifier::initFeatureOffsets()
{
    // Not implemented (not necessary)
    initGPU();
}

void CuEnsembleClassifier::initPosteriors()
{
    posteriors = new float[numTrees * numIndices];
    positives = new int[numTrees * numIndices];
    negatives = new int[numTrees * numIndices];

    for(int i = 0; i < numTrees; i++)
    {
        for(int j = 0; j < numIndices; j++)
        {
            posteriors[i * numIndices + j] = 0;
            positives[i * numIndices + j] = 0;
            negatives[i * numIndices + j] = 0;
        }
    }
}

void CuEnsembleClassifier::filter(const GpuMat &img, int *d_inWinIndices, int &numInWins)
{
    cudaEnsembleClassifier(img, windows_d, d_inWinIndices, numInWins, features_d, posteriors_d, confidences_d, numTrees, numFeatures, numIndices);
}

void CuEnsembleClassifier::updatePosterior(int treeIdx, int idx, int positive, int amount)
{
    int arrayIndex = treeIdx * numIndices + idx;
    (positive) ? positives[arrayIndex] += amount : negatives[arrayIndex] += amount;
    posteriors[arrayIndex] = ((float) positives[arrayIndex]) / (positives[arrayIndex] + negatives[arrayIndex]) / 10.0;
}

void CuEnsembleClassifier::learn(int *boundary, int positive, int *featureVector)
{
    // Not implemented
}

} /* namespace cuda */

} /* namespace tld */
