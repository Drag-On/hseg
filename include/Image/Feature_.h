//
// Created by jan on 12.01.17.
//

#ifndef HSEG_FEATURE_H
#define HSEG_FEATURE_H


#include <vector>

class Feature_
{
public:
    size_t size() const;

    float operator()(size_t i) const;

private:
    std::vector<float> m_features;

    friend class FeatureImage;
};


#endif //HSEG_FEATURE_H
