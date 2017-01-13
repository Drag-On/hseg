//
// Created by jan on 28.08.16.
//

#ifndef HSEG_INFERENCERESULT_H
#define HSEG_INFERENCERESULT_H

#include <Image/Image.h>

/**
 * Stores the final result from inference
 */
struct InferenceResult
{
public:
    LabelImage labeling; //< Class labeling
    LabelImage superpixels; //< Superpixel segmentation
    //std::vector<Cluster> clusters; //< Cluster representatives
    uint32_t numIter = 0; //< Amount of iterations until convergence
};


#endif //HSEG_INFERENCERESULT_H
