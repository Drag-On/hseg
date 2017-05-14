/**********************************************************
 * @file   DiminishingStepSizeRule.cpp
 * @author jan
 * @date   29.04.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/

#include "typedefs.h"
#include "Energy/Weights.h"
#include "Energy/DiminishingStepSizeRule.h"

DiminishingStepSizeRule::DiminishingStepSizeRule(float baseLr, size_t t) noexcept
    : m_t(t),
      m_base(baseLr)
{
}

void DiminishingStepSizeRule::update(Weights& w, Weights const& gradient)
{
    w -= gradient * (m_base / std::sqrt(m_t+1.f));

    m_t++;
}

bool DiminishingStepSizeRule::write(std::string const& /*folder*/)
{
    return true;
}

bool DiminishingStepSizeRule::read(std::string const& /*folder*/, size_t /*t*/)
{
    return true;
}

