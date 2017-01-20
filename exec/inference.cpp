//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <Timer.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Accuracy/ConfusionMatrix.h>
#include <boost/filesystem/operations.hpp>

PROPERTIES_DEFINE(Inference,
                  GROUP_DEFINE(dataset,
                               GROUP_DEFINE(path,
                                            PROP_DEFINE_A(std::string, gt, "", --gt)
                               )
                               GROUP_DEFINE(extension,
                                       PROP_DEFINE_A(std::string, gt, ".png", --gt_ext)
                               )
                               GROUP_DEFINE(constants,
                                            PROP_DEFINE_A(uint32_t, numClasses, 21, --numClasses)
                                            PROP_DEFINE_A(uint32_t, featDim, 512, --featDim)
                               )
                  )
                  GROUP_DEFINE(param,
                               PROP_DEFINE_A(std::string, weights, "", -w)
                  )
                  PROP_DEFINE_A(std::string, image, "", --img)
                  PROP_DEFINE_A(std::string, outDir, "", --out)
)

enum ERRCODE
{
    SUCCESS=0,
    CANT_READ_FILE,
};

int main(int argc, char** argv)
{
    // Read properties
    InferenceProperties properties;
    properties.read("properties/hseg_infer.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    Weights weights(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
    if(!weights.read(properties.param.weights))
    {
        std::cerr << "Couldn't read weights from \"" << properties.param.weights << "\". Using random weights instead." << std::endl;
        weights.randomize();
    }

    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(256ul);

    // Load image
    FeatureImage features;
    if(!features.read(properties.image))
    {
        std::cerr << "Unable to read features from \"" << properties.image << "\"" << std::endl;
        return CANT_READ_FILE;
    }

    // Create energy function
    EnergyFunction energyFun(&weights);

    // Do the inference!
    Timer t(true);
    InferenceIterator<EnergyFunction> inference(&energyFun, &features);
    auto result = inference.runDetailed();
    t.pause();

    std::cout << "Computed " << result.numIter << " iterations (" << t.elapsed<Timer::milliseconds>() << ")" << std::endl;
    std::cout << "Energy after each iteration: " << std::endl;
    for(size_t i = 0; i < result.energy.size(); ++i)
        std::cout << i << "\t" << result.energy[i] << std::endl;

    // Write results to disk
    std::string filename = boost::filesystem::path(properties.image).stem().string();
    auto errCode = helper::image::writePalettePNG(properties.outDir + filename + ".png", result.labelings.back(), cmap);
    if(errCode != helper::image::PNGError::Okay)
        std::cerr << "Couldn't write prediction to \"" << properties.outDir + filename + ".png" << "\". Error Code: " << (int) errCode << std::endl;
    if(!helper::image::writeMarginals(properties.outDir + filename + ".marginals", result.marginals.back()))
        std::cerr << "Couldn't write marginals to \"" << properties.outDir + filename + ".marginals" << "\"." << std::endl;

    return SUCCESS;
}