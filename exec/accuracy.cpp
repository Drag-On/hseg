//
// Created by jan on 15.09.16.
//

#include <BaseProperties.h>
#include <helper/image_helper.h>
#include <helper/clustering_helper.h>
#include <Accuracy/ConfusionMatrix.h>
#include <boost/filesystem/path.hpp>
#include <Energy/LossAugmentedEnergyFunction.h>

PROPERTIES_DEFINE(Accuracy,
                  GROUP_DEFINE(dataset,
                               PROP_DEFINE_A(std::string, list, "", -l)
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
                               PROP_DEFINE_A(ClusterId, numClusters, 100, --numClusters)
                  )
                  GROUP_DEFINE(train,
                               PROP_DEFINE_A(float, C, 0.1, -C)
                               PROP_DEFINE_A(bool, useClusterLoss, true, --useClusterLoss)
                  )
                  PROP_DEFINE_A(bool, scaleGt, false, --scale_gt)
                  PROP_DEFINE_A(std::string, inDir, "", --in)
                  PROP_DEFINE_A(std::string, inClusterDir, "", --in_cluster)
                  PROP_DEFINE_A(std::string, outDir, "./", --out)
)

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

enum ErrorCode
{
    ERR_OK = 0,
    ERR_EMPTY_FILE_LIST,
    ERR_IMAGE_LOAD,
    ERR_CLUSTERING_LOAD,
    ERR_IMAGE_MISMATCH,
    ERR_CANT_READ_WEIGHTS,
};

struct ImageAccuracyData
{
    std::string name;
    float rawAccuracy = 0.f;
    float iouAccuracy = 0.f;
};

int main(int argc, char** argv)
{
    // Read properties
    AccuracyProperties properties;
    properties.read("properties/hseg_accy.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    std::string fileListName = boost::filesystem::path(properties.dataset.list).stem().string();
    auto fileNames = readFileNames(properties.dataset.list);
    if(fileNames.empty())
    {
        std::cerr << "Empty file list." << std::endl;
        return ERR_EMPTY_FILE_LIST;
    }

    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(256ul);
    ConfusionMatrix accuracy(properties.dataset.constants.numClasses);
    float loss = 0;
    size_t rawPxCorrect = 0;
    size_t rawPixelCount = 0;
    float meanCorrectPercentage = 0;

    std::vector<ImageAccuracyData> imageAccData;

    for(auto const& f : fileNames)
    {
        std::string const& predFilename = properties.inDir + f + properties.dataset.extension.gt;
        std::string const& cluFilename = properties.inClusterDir + f + ".dat";
        std::string const& gtFilename = properties.dataset.path.gt + f + properties.dataset.extension.gt;

        // Load images
        LabelImage pred;
        auto errCode = helper::image::readPalettePNG(predFilename, pred, nullptr);
        if(errCode != helper::image::PNGError::Okay)
        {
            std::cerr << "Couldn't load image \"" << predFilename << "\". Error Code: " << (int) errCode << std::endl;
            return ERR_IMAGE_LOAD;
        }
        LabelImage gt;
        errCode = helper::image::readPalettePNG(gtFilename, gt, nullptr);
        if(errCode != helper::image::PNGError::Okay)
        {
            std::cerr << "Couldn't load image \"" << gtFilename << "\". Error Code: " << (int) errCode << std::endl;
            return ERR_IMAGE_LOAD;
        }

        if(properties.scaleGt)
            gt.rescale(pred.width(), pred.height(), false);

        if(pred.width() != gt.width() || pred.height() != gt.height() || pred.pixels() == 0)
        {
            std::cerr << "Prediction and ground truth don't match (" << pred.width() << "x" << pred.height()
                      << " vs. " << gt.width() << "x" << gt.height() << ")" << std::endl;
            return ERR_IMAGE_MISMATCH;
        }

        LabelImage clustering;
        std::vector<Cluster> clusters;
        if(properties.param.numClusters > 0 && !properties.inClusterDir.empty() && !helper::clustering::read(cluFilename, clustering, clusters))
        {
            std::cerr << "Couldn't load clustering from \"" << cluFilename << "\"" << std::endl;
            return ERR_CLUSTERING_LOAD;
        }

        accuracy.join(pred, gt);

        size_t imgRawPxCorrect = 0;
        size_t imgRawPxCount = 0;

        ConfusionMatrix localConfusion(properties.dataset.constants.numClasses, pred, gt);

        // Compute loss
        float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.dataset.constants.numClasses);
        loss += LossAugmentedEnergyFunction::computeLoss(pred, clustering, gt, clusters, lossFactor,
                                                             properties.dataset.constants.numClasses, properties.train.useClusterLoss);
        // Raw percentage
        for (size_t i = 0; i < gt.pixels(); ++i)
            if (gt.atSite(i) < properties.dataset.constants.numClasses)
            {
                imgRawPxCount++;
                if(gt.atSite(i) == pred.atSite(i))
                    imgRawPxCorrect++;
            }

        float rawPercentage = static_cast<float>(imgRawPxCorrect) / imgRawPxCount;
        meanCorrectPercentage += rawPercentage;
        ImageAccuracyData imgDat;
        imgDat.name = f;
        imgDat.rawAccuracy = rawPercentage;
        localConfusion.accuracies(&imgDat.iouAccuracy);
        imageAccData.push_back(imgDat);

        rawPixelCount += imgRawPxCount;
        rawPxCorrect += imgRawPxCorrect;
    }

    loss *= properties.train.C / fileNames.size();

    auto const& acc = accuracy.accuracies();
    float meanIoUWithoutBg = std::accumulate(acc.begin() + 1, acc.end(), 0.f);
    meanIoUWithoutBg /= acc.size() - 1;

    std::cout << accuracy << std::endl;
    std::cout << "Mean IoU w/o BG: " << meanIoUWithoutBg << std::endl;
    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Raw px percentage: " << (100.f * rawPxCorrect) / rawPixelCount << " % (" << rawPxCorrect << "/"
              << rawPixelCount << ")" << std::endl;
    std::cout << "Mean px percentage: " << (100.f * meanCorrectPercentage) / fileNames.size() << " %" << std::endl;
    std::ofstream out(properties.outDir + fileListName + "_accuracy.txt");
    if(out.is_open())
    {
        out << properties << std::endl << std::endl;
        out << accuracy << std::endl;
        out << "Loss: " << loss << std::endl << std::endl;

//        std::sort(imageAccData.begin(), imageAccData.end(),
//                  [](ImageAccuracyData const& a, ImageAccuracyData const& b) { return a.rawAccuracy > b.rawAccuracy; });
        out << "Top List:" << std::endl;
        out << std::setw(12) << "name\t" << std::setw(12) << "raw px accy\t" << std::setw(12) << "iou accy" << std::endl;
        for(auto const& i : imageAccData)
            out << std::setw(12) << i.name << "\t" << std::setw(12) << i.rawAccuracy << "\t" << std::setw(12) << i.iouAccuracy << std::endl;

        out.close();
    }
    else
        std::cerr << "Couldn't write accuracy to \"" + properties.outDir << "\"" << std::endl;

    cv::Mat confusionMat = static_cast<cv::Mat>(accuracy);
    if(!cv::imwrite(properties.outDir + fileListName + "_confusion.png", confusionMat))
        std::cerr << "Couldn't write confusion matrix to \"" + properties.outDir << "\"" << std::endl;

    return ERR_OK;
}