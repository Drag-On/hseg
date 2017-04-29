/**********************************************************
 * @file   DiminishingStepSizeRule.h
 * @author jan
 * @date   29.04.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/
#ifndef HSEG_DIMINISHINGSTEPSIZERULE_H
#define HSEG_DIMINISHINGSTEPSIZERULE_H

#include <cstddef>
#include "IStepSizeRule.h"

class DiminishingStepSizeRule : public IStepSizeRule
{
public:
    explicit DiminishingStepSizeRule(float baseLr, size_t t = 0) noexcept;

    void update(Weights& w, Weights const& gradient) override;

private:
    size_t m_t = 0;
    float const m_base = 1.f;
};


#endif //HSEG_DIMINISHINGSTEPSIZERULE_H
