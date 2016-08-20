//
// Created by jan on 20.08.16.
//

#include <helper/coordinate_helper.h>
#include "GraphOptimizer/GraphOptimizer.h"

GraphOptimizer::GraphOptimizer(EnergyFunction const& energy) noexcept
        : m_energy(energy)
{
}

LabelImage const& GraphOptimizer::labeling() const
{
    return m_labeling;
}
