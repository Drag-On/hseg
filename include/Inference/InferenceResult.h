//
// Created by jan on 28.08.16.
//

#ifndef HSEG_INFERENCERESULT_H
#define HSEG_INFERENCERESULT_H

#include <Image/Image.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <Inference/GraphOptimizer/GraphOptimizer.h>

/**
 * Stores the final result from inference
 */
struct InferenceResult
{
public:
    LabelImage labeling; //< Class labeling
    LabelImage superpixels; //< Superpixel segmentation
};


#endif //HSEG_INFERENCERESULT_H
