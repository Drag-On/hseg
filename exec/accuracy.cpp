//
// Created by jan on 15.09.16.
//

#include <BaseProperties.h>
#include <helper/image_helper.h>
#include <Accuracy/ConfusionMatrix.h>
#include <boost/filesystem/path.hpp>
#include <Energy/LossAugmentedEnergyFunction.h>

PROPERTIES_DEFINE(Accuracy,
                  PROP_DEFINE_A(std::string, fileList, "", -l)
                  PROP_DEFINE_A(std::string, predDir, "", -p)
                  PROP_DEFINE_A(std::string, gtDir, "", -g)
                  PROP_DEFINE_A(std::string, outDir, "./", -o)
                  PROP_DEFINE_A(float, C, 0.1f, -C)
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
    ERR_IMAGE_MISMATCH,
};

struct ImageAccuracyData
{
    std::string name;
    float rawAccuracy = 0.f;
};

int main(int argc, char** argv)
{
    // Read properties
    AccuracyProperties properties;
    properties.read("properties/accuracy.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    std::string fileListName = boost::filesystem::path(properties.fileList).stem().string();
    auto fileNames = readFileNames(properties.fileList);
    if(fileNames.empty())
    {
        std::cerr << "Empty file list." << std::endl;
        return ERR_EMPTY_FILE_LIST;
    }

    size_t const numClasses = 21ul;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(256ul);
    ConfusionMatrix accuracy(numClasses);
    float loss = 0;
    size_t rawPxCorrect = 0;
    size_t rawPixelCount = 0;
    float meanCorrectPercentage = 0;

    std::vector<ImageAccuracyData> imageAccData;

    for(auto const& f : fileNames)
    {
        std::string const& predFilename = properties.predDir + f + ".png";
        std::string const& gtFilename = properties.gtDir + f + ".png";

        // Load images
        RGBImage predRGB;
        predRGB.read(predFilename);
        if (predRGB.pixels() == 0)
        {
            std::cerr << "Couldn't load image " << predFilename << std::endl;
            return ERR_IMAGE_LOAD;
        }
        LabelImage pred = helper::image::decolorize(predRGB, cmap);

        RGBImage gtRGB;
        gtRGB.read(gtFilename);
        if (gtRGB.pixels() == 0)
        {
            std::cerr << "Couldn't load image " << gtFilename << std::endl;
            return ERR_IMAGE_LOAD;
        }
        LabelImage gt = helper::image::decolorize(gtRGB, cmap);

        if(pred.width() != gt.width() || pred.height() != gt.height() || pred.pixels() == 0)
        {
            std::cerr << "Prediction and ground truth don't match." << std::endl;
            return ERR_IMAGE_MISMATCH;
        }

        accuracy.join(pred, gt);

        size_t imgRawPxCorrect = 0;
        size_t imgRawPxCount = 0;

        // Compute loss
        float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, numClasses);
        loss += LossAugmentedEnergyFunction::computeLoss(pred, gt, lossFactor, numClasses);
        for (size_t i = 0; i < gt.pixels(); ++i)
            if (gt.atSite(i) < numClasses)
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
        imageAccData.push_back(imgDat);

        rawPixelCount += imgRawPxCount;
        rawPxCorrect += imgRawPxCorrect;
    }

    loss *= properties.C / fileNames.size();

    std::cout << accuracy << std::endl;
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

        std::sort(imageAccData.begin(), imageAccData.end(),
                  [](ImageAccuracyData const& a, ImageAccuracyData const& b) { return a.rawAccuracy > b.rawAccuracy; });
        out << "Top List:" << std::endl;
        out << "---------" << std::endl;
        for(auto const& i : imageAccData)
            out << i.name << "\t" << i.rawAccuracy << std::endl;

        out.close();
    }
    else
        std::cerr << "Couldn't write accuracy to \"" + properties.outDir << "\"" << std::endl;

    cv::Mat confusionMat = static_cast<cv::Mat>(accuracy);
    if(!cv::imwrite(properties.outDir + fileListName + "_confusion.png", confusionMat))
        std::cerr << "Couldn't write confusion matrix to \"" + properties.outDir << "\"" << std::endl;

    return ERR_OK;
}