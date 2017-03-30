#include <algorithm>

#include "caffe/layers/ssvm_loss_layer.hpp"

#include <Energy/EnergyFunction.h>
#include <Inference/InferenceIterator.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <thread>
#include <future>
#include <helper/image_helper.h>

namespace caffe {

    template <typename Dtype>
    void SSVMLossLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        std::string weightsPath = this->layer_param_.svm_loss_layer_param().svm_weight_file();
        bool ok = weights_.read(weightsPath);
        CHECK(ok) << "Couldn't read \"" << weightsPath << "\".";

        numClusters_ = this->layer_param_.svm_loss_layer_param().num_clusters();
        numClasses_ = this->layer_param_.svm_loss_layer_param().num_classes();
        eps_ = this->layer_param_.svm_loss_layer_param().eps();
        maxIter_ = this->layer_param_.svm_loss_layer_param().max_iter();

        LossLayer<Dtype>::LayerSetUp(bottom, top);
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
        validRegions_.resize(bottom[0]->num());

        std::vector<std::future<float>> futures;
        // For every image in the batch
        for (int i = 0; i < bottom[0]->num(); ++i)
        {
            futures.emplace_back(
                    std::async(std::launch::async, [&, i]()
                               {
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

                                   // Copy over the features
                                   FeatureImage& featImg = features_[i];
                                   featImg = FeatureImage(bottom[0]->width(), bottom[0]->height(), bottom[0]->channels());
                                   for (Coord x = 0; x < bottom[0]->width(); ++x)
                                   {
                                       for (Coord y = 0; y < bottom[0]->height(); ++y)
                                       {
                                           for (Coord c = 0; c < bottom[0]->channels(); ++c)
                                               featImg.at(x, y)[c] = bottom[0]->data_at(i, c, y, x);
                                       }
                                   }

                                   // Rescale ground truth to feature image dimensions (without interpolation)
                                   gt.rescale(featImg.width(), featImg.height(), false);

                                   // Crop to valid region
                                   cv::Rect bb = computeValidRegion(gt);
                                   validRegions_[i] = bb;
                                   FeatureImage features_cropped(bb.width, bb.height, featImg.dim());
                                   LabelImage gt_cropped(bb.width, bb.height);
                                   for(Coord x = bb.x; x < bb.width; ++x)
                                   {
                                       for (Coord y = bb.y; y < bb.height; ++y)
                                       {
                                           gt_cropped.at(x - bb.x, y - bb.y) = gt.at(x, y);
                                           features_cropped.at(x - bb.x, y - bb.y) = featImg.at(x, y);
                                       }
                                   }
                                   gt = gt_cropped;
                                   featImg = features_cropped;

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

                                   // Compute energy without weights on the prediction
                                   auto predEnergy = energy.giveEnergyByWeight(featImg, predResult.labeling,
                                                                               predResult.clustering,
                                                                               predResult.clusters);

                                   futureGtResult.get(); // Wait until prediction on gt is done
                                   InferenceResult& gtResult = gtResult_[i];

                                   // Compute energy without weights on the ground truth
                                   auto gtEnergy = energy.giveEnergyByWeight(featImg, gt, gtResult.clustering,
                                                                             gtResult.clusters);

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
        loss /= bottom[0]->num();

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
                FeatureImage gradGt(gtResult_[i].labeling.width(), gtResult_[i].labeling.height(), bottom[0]->channels());
                energy.computeFeatureGradient(gradGt, gtResult_[i].labeling, gtResult_[i].clustering,
                                              gtResult_[i].clusters, features_[i]);
                FeatureImage gradPred(predResult_[i].labeling.width(), predResult_[i].labeling.height(), bottom[0]->channels());
                energy.computeFeatureGradient(gradPred, predResult_[i].labeling, predResult_[i].clustering,
                                              predResult_[i].clusters, features_[i]);

                // Compute gradient
                gradGt.subtract(gradPred);

                // Write back
                for (Coord x = 0; x < bottom[0]->width(); ++x)
                {
                    for (Coord y = 0; y < bottom[0]->height(); ++y)
                    {
                        cv::Rect const bb = validRegions_[i];
                        if(bb.contains(cv::Point(x, y)))
                        {
                            Label l = gt_[i].at(x - bb.x, y - bb.y);
                            for (Coord c = 0; c < features_[i].dim(); ++c)
                            {
                                if (l < numClasses_)
                                    *(bottom[0]->mutable_cpu_diff_at(i, c, y, x)) = gradGt.at(x - bb.x, y - bb.y)[c];
                                else
                                    *(bottom[0]->mutable_cpu_diff_at(i, c, y, x)) = 0;
                            }
                        }
                        else
                        {
                            for (Coord c = 0; c < features_[i].dim(); ++c)
                                *(bottom[0]->mutable_cpu_diff_at(i, c, y, x)) = 0;
                        }
                    }
                }
            }
        }
    }




//#ifdef CPU_ONLY
//    STUB_GPU(SSVMLossLayer);
//#endif

    template <typename Dtype>
    cv::Rect SSVMLossLayer<Dtype>::computeValidRegion(LabelImage const& gt) const
    {
        cv::Rect bb = helper::image::computeValidBox(gt, numClasses_);
        return bb;
    }

    INSTANTIATE_CLASS(SSVMLossLayer);
    REGISTER_LAYER_CLASS(SSVMLoss);
}