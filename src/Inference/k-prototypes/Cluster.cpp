//
// Created by jan on 18.08.16.
//

#include "Inference/k-prototypes/Cluster.h"

void Cluster::update(std::vector<Feature> const& features, LabelImage const& labeling)
{
    mean = Feature();
    std::vector<Cost> labelCosts(numClasses, 0.f);
    for (SiteId s : allocated)
    {
        // Sum up features of all allocated sites
        mean += features[s];

        // Compute cost for picking the cluster label
        Label const l1 = labeling.atSite(s);
        if (l1 < numClasses)
        {
            for (Label l2 = 0; l2 < numClasses; ++l2)
            {
                Cost localLabelCost = classDistance(l1, l2) + classData(s, l2);
                labelCosts[l2] += localLabelCost;
            }
        }
    }
    if (!allocated.empty())
        mean /= allocated.size();
    label = std::distance(labelCosts.begin(), std::min_element(labelCosts.begin(), labelCosts.end()));
}

