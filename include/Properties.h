//
// Created by jan on 20.08.16.
//

#ifndef HSEG_PROPERTIES_H
#define HSEG_PROPERTIES_H

#include "BaseProperties.h"

PROPERTIES_DEFINE(Hseg,
	GROUP_DEFINE(clustering,
		PROP_DEFINE(size_t, numClusters, 200u)
	)
	GROUP_DEFINE(convergence,
		PROP_DEFINE(float, cluster, 0.001f)
		PROP_DEFINE(float, overall, 0.001f)
	)
)

#endif //HSEG_PROPERTIES_H
