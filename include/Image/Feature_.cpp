//
// Created by jan on 12.01.17.
//

#include <boost/mpl/size_t.hpp>
#include <assert.h>
#include "Feature_.h"

unsigned long Feature_::size() const
{
    return m_features.size();
}

float Feature_::operator()(size_t i) const
{
    assert(i < m_features.size());
    return m_features[i];
}
