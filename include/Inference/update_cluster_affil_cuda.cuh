
#ifndef UPDATE_CLUSTER_AFFIL_CUDA
#define UPDATE_CLUSTER_AFFIL_CUDA

#include <Image/FeatureImage.h>
#include <thrust/device_vector.h>
#include <Image/Image.h>
#include "Cluster.h"

template<typename EnergyFun>
class UpdateClusterAffilCuda
{
public:
    UpdateClusterAffilCuda(EnergyFun const& energyFun, FeatureImage const* features);

    void updateClusterAffiliation(LabelImage& outClustering, LabelImage const& labeling, std::vector<Cluster> const& clusters);

private:
    thrust::device_vector<float> m_features_device;
    thrust::device_vector<float> m_ho_weights_device;
    thrust::device_vector<float> m_feat_weights_device;
    Label m_numLabels;
    uint32_t m_feat_dim;
};


void test();

#endif //UPDATE_CLUSTER_AFFIL_CUDA
