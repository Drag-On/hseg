//
// Created by jan on 01.09.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <helper/clustering_helper.h>
#include <Inference/InferenceIterator.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

PROPERTIES_DEFINE(TrainDistPred,
                  GROUP_DEFINE(dataset,
                               PROP_DEFINE_A(std::string, list, "", -l)
                               GROUP_DEFINE(path,
                                            PROP_DEFINE_A(std::string, img, "", --img)
                                            PROP_DEFINE_A(std::string, gt, "", --gt)
                               )
                               GROUP_DEFINE(extension,
                                            PROP_DEFINE_A(std::string, img, ".mat", --img_ext)
                                            PROP_DEFINE_A(std::string, gt, ".png", --gt_ext)
                               )
                               GROUP_DEFINE(constants,
                                            PROP_DEFINE_A(uint32_t, numClasses, 21, --numClasses)
                                            PROP_DEFINE_A(uint32_t, featDim, 512, --featDim)
                               )
                  )
                  GROUP_DEFINE(train,
                               PROP_DEFINE_A(float, C, 0.1, -C)
                               GROUP_DEFINE(iter,
                                            PROP_DEFINE_A(uint32_t, end, 1000, --end)
                               )
                               GROUP_DEFINE(rate,
                                            PROP_DEFINE_A(float, base, 0.001f, --rate)
                                            PROP_DEFINE_A(float, T, 200, -T)
                               )
                  )
                  GROUP_DEFINE(param,
                               PROP_DEFINE_A(ClusterId, numClusters, 100, --numClusters)
                               PROP_DEFINE_A(float, eps, 0, --eps)
                  )
                  PROP_DEFINE_A(std::string, weights, "", -w)
                  PROP_DEFINE_A(std::string, img, "", -i)
                  PROP_DEFINE_A(std::string, out, "out/", -o)
                  PROP_DEFINE_A(std::string, propertiesFile, "properties/hseg_train_dist_pred.info", -p)
)

enum ErrorCode
{
    SUCCESS = 0,
    CANT_READ_IMAGE = 1,
    CANT_READ_GT = 2,
    IMAGE_GT_DONT_MATCH = 3,
    INVALID_PRED_LABELING = 4,
    INVALID_PRED_SP = 5,
    INVALID_GT_SP = 6,
    CANT_WRITE_PRED_LABELING = 7,
    CANT_WRITE_PRED_SP = 8,
    CANT_WRITE_GT_SP = 9,
};

int main(int argc, char* argv[])
{
    // Read properties
    TrainDistPredProperties properties;
    properties.fromCmd(argc, argv);
    properties.read(properties.propertiesFile);
    properties.fromCmd(argc, argv); // This is so the property file location can be read via command line, however,
                                    // the command line arguments should overwrite anything written in the file,
                                    // therefore read it in again.
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // Check if there already is a result file
    boost::filesystem::path imgNamePath(properties.img);
    std::string imgName = imgNamePath.filename().stem().string();
    std::string labelPath = properties.out + "labeling/" + imgName + ".png";
    std::string clusterPath = properties.out + "clustering/" + imgName + ".dat";
    std::string clusterGtPath = properties.out + "clustering_gt/" + imgName + ".dat";
    if(boost::filesystem::exists(labelPath) && boost::filesystem::exists(clusterPath) && boost::filesystem::exists(clusterGtPath))
    {
        std::cout << "Result for \"" << properties.img << "\" already exists in \"" << properties.out << "\". Skipping." << std::endl;
        return SUCCESS;
    }

    Label const numClasses = properties.dataset.constants.numClasses;
    uint32_t const featDim = properties.dataset.constants.featDim;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(256);
    Weights curWeights(numClasses, featDim);
    if(!curWeights.read(properties.weights))
    {
        std::cout << "Couldn't read current weights from \"" << properties.weights << "\"" << std::endl;
        std::cout << "Using default weights. This is only right if this is the first iteration." << std::endl;
    }

    // Load images etc...
    std::string featFileName = properties.dataset.path.img + properties.img + properties.dataset.extension.img;
    FeatureImage features;
    if(!features.read(featFileName))
    {
        std::cerr << "Unable to read features from \"" << featFileName << "\"" << std::endl;
        return CANT_READ_IMAGE;
    }

    std::string gtFileName = properties.dataset.path.gt + properties.img + properties.dataset.extension.gt;
    LabelImage gt;
    auto errCode = helper::image::readPalettePNG(gtFileName, gt, nullptr);
    if(errCode != helper::image::PNGError::Okay)
    {
        std::cerr << "Unable to read ground truth from \"" << gtFileName << "\". Error Code: " << (int) errCode << std::endl;
        return CANT_READ_GT;
    }

    if(features.width() != gt.width() || features.height() != gt.height() || featDim != properties.dataset.constants.featDim)
    {
        std::cerr << "Image \"" << properties.img << "\" and its ground truth don't match." << std::endl;
        return IMAGE_GT_DONT_MATCH;
    }

    // Find latent variables that best explain the ground truth
    EnergyFunction energy(&curWeights, properties.param.numClusters);
    InferenceIterator<EnergyFunction> gtInference(&energy, &features, properties.param.eps);
    InferenceResult gtResult = gtInference.runOnGroundTruth(gt);

    // Check validity
    if (gtResult.clustering.width() != features.width() || gtResult.clustering.height() != features.height())
    {
        std::cerr << "Predicted ground truth clustering invalid." << std::endl;
        return INVALID_GT_SP;
    }

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction lossEnergy(&curWeights, &gt, properties.param.numClusters);
    InferenceIterator<LossAugmentedEnergyFunction> inference(&lossEnergy, &features);
    InferenceResult result = inference.run();

    // Check validity
    if (result.labeling.width() != features.width() || result.labeling.height() != features.height())
    {
        std::cerr << "Predicted labeling invalid." << std::endl;
        return INVALID_PRED_LABELING;
    }
    if (result.clustering.width() != features.width() || result.clustering.height() != features.height())
    {
        std::cerr << "Predicted clustering invalid." << std::endl;
        return INVALID_PRED_SP;
    }

    // Store results
    // Predicted labeling
    errCode = helper::image::writePalettePNG(labelPath, result.labeling, cmap);
    if(errCode != helper::image::PNGError::Okay)
    {
        std::cerr << "Couldn't write predicted labeling to \"" << labelPath << "\". Error Code: " << (int) errCode << std::endl;
        return CANT_WRITE_PRED_LABELING;
    }
    // Predicted clustering
    if(!helper::clustering::write(clusterPath, result.clustering, result.clusters))
    {
        std::cerr << "Couldn't write predicted clustering to \"" << clusterPath << "\"." << std::endl;
        return CANT_WRITE_PRED_SP;
    }
    helper::image::colorize(result.clustering, cmap).write(clusterPath + ".png"); // This is just so it can be viewed easily
    // Groundtruth clustering
    if(!helper::clustering::write(clusterGtPath, gtResult.clustering, gtResult.clusters))
    {
        std::cerr << "Couldn't write predicted clustering to \"" << clusterGtPath << "\"." << std::endl;
        return CANT_WRITE_GT_SP;
    }
    helper::image::colorize(gtResult.clustering, cmap).write(clusterGtPath + ".png");

    return SUCCESS;
}