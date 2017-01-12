//
// Created by jan on 12.01.17.
//

#ifndef HSEG_FEATUREIMAGE_H
#define HSEG_FEATUREIMAGE_H

#include <string>
#include <vector>
#include <typedefs.h>
#include "Feature_.h"

class FeatureImage
{
public:
    FeatureImage() = default;

    FeatureImage(std::string const& filename);

    bool read(std::string const& filename);

    Coord width() const;

    Coord height() const;

    Coord dim() const;

    Feature_ const& at(Coord x, Coord y) const;

    Feature_ const& atSite(SiteId i) const;

private:
    Coord m_width;
    Coord m_height;
    Coord m_dim;
    std::vector<Feature_> m_features;
};


#endif //HSEG_FEATUREIMAGE_H
