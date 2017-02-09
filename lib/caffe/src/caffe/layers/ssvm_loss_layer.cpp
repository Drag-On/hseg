#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/ssvm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        weights_.read("weights.dat");
    }

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        // Compute the loss
    }

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    }


#ifdef CPU_ONLY
    STUB_GPU(SSVMLossLayer);
#endif

    INSTANTIATE_CLASS(SSVMLossLayer);
    REGISTER_LAYER_CLASS(SSVMLoss);
}