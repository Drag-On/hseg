//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <helper/image_helper.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <Timer.h>

PROPERTIES_DEFINE(Clustering,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(std::string, image, "")
                  PROP_DEFINE(std::string, labeling, "")
)

int main()
{
    // Read properties
    ClusteringProperties properties;
    properties.read("properties/clustering.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;

    Weights weights(numClasses); // TODO: Load weights instead of always using the default ones
    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max(256ul, properties.numClusters));

    // Load images
    RGBImage rgb;
    rgb.read(properties.image);
    if (rgb.pixels() == 0)
    {
        std::cerr << "Couldn't load image " << properties.image << std::endl;
        return -1;
    }
    CieLabImage cieLab = rgb.getCieLabImg();
    RGBImage labelingRGB;
    labelingRGB.read(properties.labeling);
    if(labelingRGB.pixels() == 0)
    {
        std::cerr << "Couldn't load labeling image " << properties.labeling << std::endl;
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
    UnaryFile fakeUnary;
    EnergyFunction energyFun(fakeUnary, weights);

    Timer t(true);
    Clusterer clusterer(energyFun);
    size_t iter = clusterer.run(properties.numClusters, numClasses, cieLab, labeling);
    t.pause();

    std::cout << "Converged after " << iter << " iterations (" << t.elapsed<Timer::milliseconds>() << ")" << std::endl;

    // Show results
    cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
    cv::Mat labelMat = static_cast<cv::Mat>(labelingRGB);
    cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(clusterer.clustership(), cmap));
    cv::imshow("rgb", rgbMat);
    cv::imshow("labeling", labelMat);
    cv::imshow("superpixels", spMat);
    cv::waitKey();
}