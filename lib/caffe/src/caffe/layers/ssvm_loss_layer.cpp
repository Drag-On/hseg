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
        features_ = FeatureImage(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels());
        for(Coord x = 0; x < features_.width(); ++x)
        {
            for(Coord y = 0; y < features_.height(); ++y)
            {
                for(Coord c = 0; c < features_.dim(); ++c)
                    features_.at(x, y)[c] = bottom[0]->data_at(0, c, y, x);
            }
        }

        // Copy over label image
        LabelImage gt(bottom[1]->width(), bottom[1]->height());
        for(Coord x = 0; x < features_.width(); ++x)
        {
            for(Coord y = 0; y < features_.height(); ++y)
            {
                // Round because of float imprecision
                gt.at(x, y) = std::round(bottom[1]->data_at(0, 0, y, x));
                assert(gt.at(x, y) >= 0);
                assert(gt.at(x, y) < 21);
            }
        }

        // Find latent variables that best explain the ground truth
        EnergyFunction energy(&weights_, numClusters_);
        InferenceIterator<EnergyFunction> gtInference(&energy, &features_, eps_, maxIter_);
        gtResult_ = gtInference.runOnGroundTruth(gt);

        // Predict with loss-augmented energy
        LossAugmentedEnergyFunction lossEnergy(&weights_, &gt, numClusters_);
        InferenceIterator<LossAugmentedEnergyFunction> inference(&lossEnergy, &features_, eps_, maxIter_);
        predResult_ = inference.run();

        // Compute energy without weights on the ground truth
        auto gtEnergy = energy.giveEnergyByWeight(features_, gt, gtResult_.clustering, gtResult_.clusters);
        // Compute energy without weights on the prediction
        auto predEnergy = energy.giveEnergyByWeight(features_, predResult_.labeling, predResult_.clustering, predResult_.clusters);

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

        if (propagate_down[1]) {
            LOG(FATAL) << this->type()
                       << " Layer cannot backpropagate to label inputs.";
        }
        if (propagate_down[0]) {

            EnergyFunction energy(&weights_, numClusters_);
            FeatureImage gradGt(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels());
            energy.computeFeatureGradient(gradGt, gtResult_.labeling, gtResult_.clustering, gtResult_.clusters, features_);
            FeatureImage gradPred(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels());
            energy.computeFeatureGradient(gradPred, predResult_.labeling, predResult_.clustering, predResult_.clusters, features_);

            // Compute gradient
            gradGt.subtract(gradPred);

            // Write back
            for(Coord x = 0; x < features_.width(); ++x)
            {
                for(Coord y = 0; y < features_.height(); ++y)
                {
                    for(Coord c = 0; c < features_.dim(); ++c)
                        *(bottom[0]->mutable_cpu_diff_at(1, c, y, x)) = gradGt.at(x, y)[c];
                }
            }
        }
    }


#ifdef CPU_ONLY
//    STUB_GPU(SSVMLossLayer);
#endif

    INSTANTIATE_CLASS(SSVMLossLayer);
    REGISTER_LAYER_CLASS(SSVMLoss);
}