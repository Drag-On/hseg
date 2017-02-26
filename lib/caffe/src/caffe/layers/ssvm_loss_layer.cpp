#include <algorithm>

#include "caffe/layers/ssvm_loss_layer.hpp"

#include <Energy/EnergyFunction.h>
#include <Inference/InferenceIterator.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <thread>
#include <future>

namespace caffe {

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        bool ok = weights_.read("weights.dat");
        CHECK(ok) << "Couldn't read \"weights.dat\".";
    }

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::Reshape(vector<Blob<Dtype>*> const& bottom, vector<Blob<Dtype>*> const& top)
    {
        LossLayer<Dtype>::Reshape(bottom, top);
    }

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        // Pre-allocate memory
        features_.resize(bottom[0]->num(), FeatureImage(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels()));
        gt_.resize(bottom[0]->num(), LabelImage(bottom[1]->width(), bottom[1]->height()));
        gtResult_.resize(bottom[0]->num());
        predResult_.resize(bottom[0]->num());

        std::vector<std::future<float>> futures;
        // For every image in the batch
        for (int i = 0; i < bottom[0]->num(); ++i)
        {
            futures.emplace_back(
                    std::async(std::launch::async, [&, i]()
                               {
                                   // Copy over the features
                                   FeatureImage& featImg = features_[i];
                                   for (Coord x = 0; x < bottom[0]->width(); ++x)
                                   {
                                       for (Coord y = 0; y < bottom[0]->height(); ++y)
                                       {
                                           for (Coord c = 0; c < bottom[0]->channels(); ++c)
                                               featImg.at(x, y)[c] = bottom[0]->data_at(i, c, y, x);
                                       }
                                   }

                                   // Copy over label image
                                   LabelImage& gt = gt_[i];
                                   gt = LabelImage(bottom[1]->width(), bottom[1]->height()); // Overwrite downscaled version of previous iteration
                                   for (Coord x = 0; x < bottom[1]->width(); ++x)
                                   {
                                       for (Coord y = 0; y < bottom[1]->height(); ++y)
                                       {
                                           // Round because of float imprecision
                                           gt.at(x, y) = std::round(bottom[1]->data_at(i, 0, y, x));
                                       }
                                   }
                                   gt.rescale(featImg.width(), featImg.height(), false);

                                   // Find latent variables that best explain the ground truth
                                   EnergyFunction energy(&weights_, numClusters_);
                                   std::future<bool> futureGtResult = std::async(std::launch::async, [&]()
                                   {
                                       InferenceIterator<EnergyFunction> gtInference(&energy, &featImg, eps_, maxIter_);
                                       gtResult_[i] = gtInference.runOnGroundTruth(gt);
                                       return true;
                                   });

                                   // Predict with loss-augmented energy
                                   LossAugmentedEnergyFunction lossEnergy(&weights_, &gt, numClusters_);
                                   InferenceIterator<LossAugmentedEnergyFunction> inference(&lossEnergy, &featImg, eps_,
                                                                                            maxIter_);
                                   predResult_[i] = inference.run();
                                   InferenceResult& predResult = predResult_[i];
                                   futureGtResult.get(); // Wait until prediction on gt is done
                                   InferenceResult& gtResult = gtResult_[i];

                                   // Compute energy without weights on the ground truth
                                   auto gtEnergy = energy.giveEnergyByWeight(featImg, gt, gtResult.clustering,
                                                                             gtResult.clusters);
                                   // Compute energy without weights on the prediction
                                   auto predEnergy = energy.giveEnergyByWeight(featImg, predResult.labeling,
                                                                               predResult.clustering,
                                                                               predResult.clusters);

                                   // Compute upper bound on this image
                                   auto gtEnergyCur = weights_ * gtEnergy;
                                   auto predEnergyCur = weights_ * predEnergy;
                                   float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, numClasses_);
                                   float loss = LossAugmentedEnergyFunction::computeLoss(predResult.labeling,
                                                                                         predResult.clustering, gt,
                                                                                         predResult.clusters,
                                                                                         lossFactor, numClasses_);
                                   float sampleLoss = (loss - predEnergyCur) + gtEnergyCur;
                                   return sampleLoss;
                               }));
        }

        float loss = 0;
        for (int j = 0; j < futures.size(); ++j)
            loss += futures[j].get();
        loss /= futures.size();

        top[0]->mutable_cpu_data()[0] = loss;
    }

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        if (propagate_down[1]) {
            LOG(FATAL) << this->type()
                       << " Layer cannot backpropagate to label inputs.";
        }
        if (propagate_down[0])
        {
            // For every image in the batch
            for (int i = 0; i < bottom[0]->num(); ++i)
            {
                EnergyFunction energy(&weights_, numClusters_);
                FeatureImage gradGt(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels());
                energy.computeFeatureGradient(gradGt, gtResult_[i].labeling, gtResult_[i].clustering,
                                              gtResult_[i].clusters, features_[i]);
                FeatureImage gradPred(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels());
                energy.computeFeatureGradient(gradPred, predResult_[i].labeling, predResult_[i].clustering,
                                              predResult_[i].clusters, features_[i]);

                // Compute gradient
                gradGt.subtract(gradPred);

                // Write back
                for (Coord x = 0; x < features_[i].width(); ++x)
                {
                    for (Coord y = 0; y < features_[i].height(); ++y)
                    {
                        Label l = gt_[i].at(x, y);
                        for (Coord c = 0; c < features_[i].dim(); ++c)
                        {
                            if (l < numClasses_)
                                *(bottom[0]->mutable_cpu_diff_at(i, c, y, x)) = gradGt.at(x, y)[c];
                            else
                                *(bottom[0]->mutable_cpu_diff_at(i, c, y, x)) = 0;
                        }
                    }
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