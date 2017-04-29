/**********************************************************
 * @file   AdamStepSizeRule.cpp
 * @author jan
 * @date   29.04.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/
#include "Energy/AdamStepSizeRule.h"

AdamStepSizeRule::AdamStepSizeRule(float alpha, float beta1, float beta2, float eps, size_t numClasses, size_t featDim,
                                   size_t t) noexcept
    : m_alpha(alpha),
      m_beta1(beta1),
      m_beta2(beta2),
      m_eps(eps),
      m_t(t),
      m_firstMoment(numClasses, featDim),
      m_secondMoment(numClasses, featDim)
{
}

void AdamStepSizeRule::update(Weights& w, Weights const& gradient)
{
    Weights grad = gradient;

    // Update biased 1st and 2nd moment estimates
    m_firstMoment = m_firstMoment * m_beta1 + grad * (1 - m_beta1);
    grad.squareElements();
    m_secondMoment = m_secondMoment * m_beta2 + grad * (1 - m_beta2);

    // Update weights
    float const curAlpha = m_alpha * std::sqrt(1 - std::pow(m_beta2, m_t + 1)) / (1 - std::pow(m_beta1, m_t + 1));
    auto sqrtSecondMomentVector = m_secondMoment;
    sqrtSecondMomentVector.sqrt();
    w -= (m_firstMoment * curAlpha) / (sqrtSecondMomentVector + m_eps);

    m_t++;
}
