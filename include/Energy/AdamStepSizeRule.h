/**********************************************************
 * @file   AdamStepSizeRule.h
 * @author jan
 * @date   29.04.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/
#ifndef HSEG_ADAMSTEPSIZERULE_H
#define HSEG_ADAMSTEPSIZERULE_H

#include "Energy/Weights.h"
#include <Energy/IStepSizeRule.h>
#include <cstddef>

class AdamStepSizeRule : public IStepSizeRule
{
public:
    AdamStepSizeRule(float alpha, float beta1, float beta2, float eps, size_t numClasses, size_t featDim,
                     size_t t = 0) noexcept;

    void update(Weights& w, Weights const& gradient) override;

    bool write(std::string const& folder) override;

    bool read(std::string const& folder, size_t t) override;

private:
    float const m_alpha;
    float const m_beta1;
    float const m_beta2;
    float const m_eps;
    size_t m_t = 0;
    Weights m_firstMoment;
    Weights m_secondMoment;
};


#endif //HSEG_ADAMSTEPSIZERULE_H
