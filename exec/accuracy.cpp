//
// Created by jan on 15.09.16.
//

#include <BaseProperties.h>
#include <helper/image_helper.h>
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
                  )
                  GROUP_DEFINE(train,
                               PROP_DEFINE_A(float, C, 0.1, -C)
                  )
                  PROP_DEFINE_A(std::string, inDir, "", --in)
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
    ERR_IMAGE_MISMATCH,
    ERR_CANT_READ_WEIGHTS,
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

    std::string fileListName = boost::filesystem::path(properties.dataset.list).stem().string();
    auto fileNames = readFileNames(properties.dataset.list);
    if(fileNames.empty())
    {
        std::cerr << "Empty file list." << std::endl;
        return ERR_EMPTY_FILE_LIST;
    }

    Weights weights(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
    if(!weights.read(properties.param.weights))
    {
        std::cerr << "Couldn't read weights file \"" << properties.param.weights << "\"" << std::endl;
        return ERR_CANT_READ_WEIGHTS;
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

        if(pred.width() != gt.width() || pred.height() != gt.height() || pred.pixels() == 0)
        {
            std::cerr << "Prediction and ground truth don't match." << std::endl;
            return ERR_IMAGE_MISMATCH;
        }

        accuracy.join(pred, gt);

        size_t imgRawPxCorrect = 0;
        size_t imgRawPxCount = 0;

        // Compute loss
        float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.dataset.constants.numClasses);
        loss += LossAugmentedEnergyFunction::computeLoss(pred, gt, lossFactor, properties.dataset.constants.numClasses);
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
        imageAccData.push_back(imgDat);

        rawPixelCount += imgRawPxCount;
        rawPxCorrect += imgRawPxCorrect;
    }

    loss *= properties.train.C / fileNames.size();

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