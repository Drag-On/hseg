#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/ssvm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "Image/FeatureImage.h"
#include <Energy/EnergyFunction.h>
#include <Inference/InferenceIterator.h>
#include <Energy/LossAugmentedEnergyFunction.h>

namespace caffe {

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        weights_.read("weights.dat");
    }

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        // Copy over the features
        FeatureImage features(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels());
        for(Coord x = 0; x < features.width(); ++x)
        {
            for(Coord y = 0; y < features.height(); ++y)
            {
                for(Coord c = 0; c < features.dim(); ++c)
                    features.at(x, y)[c] = bottom[0]->data_at(1, c, y, x);
            }
        }

        // Copy over label image
        LabelImage gt(bottom[1]->width(), bottom[1]->height());
        for(Coord x = 0; x < features.width(); ++x)
        {
            for(Coord y = 0; y < features.height(); ++y)
            {
                gt.at(x, y) = bottom[1]->data_at(1, 0, y, x);
            }
        }

        // Find latent variables that best explain the ground truth
        EnergyFunction energy(&weights_, numClusters_);
        InferenceIterator<EnergyFunction> gtInference(&energy, &features, eps_, maxIter_);
        gtResult_ = gtInference.runOnGroundTruth(gt);

        // Predict with loss-augmented energy
        LossAugmentedEnergyFunction lossEnergy(&weights_, &gt, numClusters_);
        InferenceIterator<LossAugmentedEnergyFunction> inference(&lossEnergy, &features, eps_, maxIter_);
        predResult_ = inference.run();

        // Compute energy without weights on the ground truth
        auto gtEnergy = energy.giveEnergyByWeight(features, gt, gtResult_.clustering, gtResult_.clusters);
        // Compute energy without weights on the prediction
        auto predEnergy = energy.giveEnergyByWeight(features, predResult_.labeling, predResult_.clustering, predResult_.clusters);

        // Compute upper bound on this image
        auto gtEnergyCur = weights_ * gtEnergy;
        auto predEnergyCur = weights_ * predEnergy;
        float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, numClasses_);
        float loss = LossAugmentedEnergyFunction::computeLoss(predResult_.labeling, predResult_.clustering, gt, predResult_.clusters,
                                                              lossFactor, numClasses_);
        float sampleLoss = (loss - predEnergyCur) + gtEnergyCur;

        top[0]->mutable_cpu_data()[0] = sampleLoss;
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