//
// Created by jan on 18.08.16.
//

#ifndef HSEG_CLUSTERER_H
#define HSEG_CLUSTERER_H

#include "Cluster.h"

class Clusterer
{
public:
    void run(size_t numClusters, size_t numLabels, RGBImage const& rgb, LabelImage const& labels);

    LabelImage const& clustership() const;

private:
    std::vector<Cluster> m_clusters;
    LabelImage m_clustership;
    float m_gamma = 1000.f; // TODO: Find good mixing coefficient
    float m_conv = 0.01f;

    inline float delta(Label l1, Label l2) const
    {
        if(l1 == l2)
            return 0;
        else
            return 1;
    }

    void initPrototypes(RGBImage const& rgb, LabelImage const& labels);

    void allocatePrototypes(RGBImage const& rgb, LabelImage const& labels);

    int reallocatePrototypes(RGBImage const& rgb, LabelImage const& labels);

    size_t findClosestCluster(Feature const& feature, Label classLabel) const;
};


#endif //HSEG_CLUSTERER_H
