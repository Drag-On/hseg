//
// Created by jan on 20.08.16.
//

#ifndef HSEG_PROPERTIES_H
#define HSEG_PROPERTIES_H

#include "BaseProperties.h"

PROPERTIES_DEFINE(Hseg,
	GROUP_DEFINE(clustering,
		PROP_DEFINE(size_t, numClusters, 200)
	)
	GROUP_DEFINE(weights,
		PROP_DEFINE(float, unary, 5.)
		PROP_DEFINE(float, pairwise, 500.)
		PROP_DEFINE(float, pairwiseSigmaSq, 0.05)
		PROP_DEFINE(float, spGamma, 80.)
		PROP_DEFINE(float, spatial, 0.7)
		PROP_DEFINE(float, color, 1. - spatial)
	)
	GROUP_DEFINE(convergence,
		PROP_DEFINE(float, cluster, 0.001)
		PROP_DEFINE(float, overall, 0.001)
	)
)

#endif //HSEG_PROPERTIES_H
