//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <helper/image_helper.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <Timer.h>
#include <boost/filesystem/path.hpp>
#include <Energy/feature_weights.h>

PROPERTIES_DEFINE(Clustering,
                  PROP_DEFINE_A(size_t, numClusters, 300, -c)
                  PROP_DEFINE(std::string, image, "")
                  PROP_DEFINE(std::string, labeling, "")
                  GROUP_DEFINE(batch,
                               PROP_DEFINE_A(std::string, listFile, "", -l)
                               PROP_DEFINE(std::string, imgPath, "")
                               PROP_DEFINE(std::string, imgExtension, ".jpg")
                               PROP_DEFINE(std::string, labelPath, "")
                               PROP_DEFINE(std::string, labelExtension, ".png")
                  )
                  PROP_DEFINE_A(std::string, out, "out.png", -o)
                  PROP_DEFINE(bool, showResult, true)
                  GROUP_DEFINE(weights,
                               PROP_DEFINE_A(std::string, weightsFile, "", -w)
                               PROP_DEFINE(std::string, featureWeightFile, "")
                               PROP_DEFINE(float, feature, 1.f)
                               PROP_DEFINE(float, label, 30.f)
                  )
)

int process(std::string const& imgFile, std::string const& labelFile, ClusteringProperties const& properties,
            helper::image::ColorMap const& cmap, helper::image::ColorMap const& cmap2, EnergyFunction const& energy)
{
    std::cout << "Processing image " << imgFile << std::endl;

    // Load images
    RGBImage rgb;
    rgb.read(imgFile);
    if (rgb.pixels() == 0)
    {
        std::cerr << "Couldn't load image " << imgFile << std::endl;
        return -1;
    }
    CieLabImage cieLab = rgb.getCieLabImg();
    RGBImage labelingRGB;
    labelingRGB.read(labelFile);
    if(labelingRGB.pixels() == 0)
    {
        std::cerr << "Couldn't load labeling image " << labelFile << std::endl;
        return -2;
    }
    else if(labelingRGB.width() != rgb.width() || labelingRGB.height() != rgb.height())
    {
        std::cerr << "Color image and labeling don't match up!" << std::endl;
        return -3;
    }

    // Convert labeling back to indices
    LabelImage labeling = helper::image::decolorize(labelingRGB, cmap);

    // Do the clustering
    Timer t(true);
    Clusterer<EnergyFunction> clusterer(energy, cieLab, labeling, properties.numClusters);
    size_t iter = clusterer.run(labeling);
    t.pause();

    std::cout << "Converged after " << iter << " iterations (" << t.elapsed<Timer::milliseconds>() << ")" << std::endl;


    cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
    cv::Mat labelMat = static_cast<cv::Mat>(labelingRGB);
    cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(clusterer.clustership(), cmap2));

    // Write results to disk
    std::string outFile = properties.out;
    if(properties.image.empty())
        outFile += boost::filesystem::path(imgFile).stem().string() + ".png";
    if(!cv::imwrite(outFile, spMat))
        std::cerr << "Couldn't write result to " << outFile << std::endl;

    // Show results
    if(properties.showResult)
    {
        cv::imshow("rgb", rgbMat);
        cv::imshow("labeling", labelMat);
        cv::imshow("superpixels", spMat);
        cv::waitKey();
    }
    return 0;
}

std::vector<std::string> readFileNames(std::string const& listFile)
{
    std::vector<std::string> list;
    std::ifstream in(listFile, std::ios::in);
    if (in.is_open())
    {
        std::string line;
        while (std::getline(in, line))
            list.push_back(line);
        in.close();
    }
    return list;
}

int main(int argc, char** argv)
{
    // Read properties
    ClusteringProperties properties;
    properties.read("properties/clustering.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;

    Weights weights(numClasses, 1, 1, properties.weights.feature, properties.weights.label);
    weights.read(properties.weights.weightsFile);
    std::cout << "Used weights:" << std::endl;
    std::cout << weights << std::endl;
    UnaryFile fakeUnary;
    Matrix5 featureWeights = readFeatureWeights(properties.weights.featureWeightFile);
    featureWeights = featureWeights.inverse();
    std::cout << "Used feature weights: " << std::endl;
    std::cout << featureWeights << std::endl;

    EnergyFunction energyFun(fakeUnary, weights, 0.05f, featureWeights);

    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap cmap2 = helper::image::generateColorMap(properties.numClusters);

    if(properties.image.empty())
    {
        // Batch process
        auto fileNames = readFileNames(properties.batch.listFile);
        for(size_t i = 0; i < fileNames.size(); ++i)
        {
            std::string imgFile = properties.batch.imgPath + fileNames[i] + properties.batch.imgExtension;
            std::string labelFile = properties.batch.labelPath + fileNames[i] + properties.batch.labelExtension;
            process(imgFile, labelFile, properties, cmap, cmap2, energyFun);
        }
    }
    else
    {
        return process(properties.image, properties.labeling, properties, cmap, cmap2, energyFun);
    }
}