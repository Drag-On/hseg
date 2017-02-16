#ifndef CAFFE_SSVM_LOSS_LAYER_HPP_
#define CAFFE_SSVM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "Energy/Weights.h"
#include <Inference/InferenceResult.h>
#include <Image/FeatureImage.h>

namespace caffe {
    template <typename Dtype>
    class SSVMLossLayer : public LossLayer<Dtype>
    {
    public:
        explicit SSVMLossLayer(const LayerParameter& param)
                : LossLayer<Dtype>(param),
                  weights_(21, 512)
        {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "SSVMLoss"; }
        virtual inline int ExactNumBottomBlobs() const { return 3; }
        virtual inline int ExactNumTopBlobs() const { return -1; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
//        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                                 const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        Weights weights_;
        ClusterId numClusters_ = 30;
        float eps_ = 10.f;
        int maxIter_ = 50;
        int const numClasses_ = 21;

        FeatureImage features_;
        InferenceResult gtResult_;
        InferenceResult predResult_;
    };
}


#endif //CAFFE_SSVM_LOSS_LAYER_HPP_
