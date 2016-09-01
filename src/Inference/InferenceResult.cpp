//
// Created by jan on 01.09.16.
//

#include "Inference/InferenceResult.h"

InferenceResult::InferenceResult(EnergyFunction const& energy)
        : clusterer(energy),
          optimizer(energy)
{
}