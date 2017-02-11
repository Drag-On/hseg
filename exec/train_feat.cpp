//
// Created by jan on 03.02.17.
//

#include <BaseProperties.h>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <Image/Image.h>
#include <Image/FeatureImage.h>
#include <helper/image_helper.h>

PROPERTIES_DEFINE(TrainFeat,
                  PROP_DEFINE_A(bool, useGPU, false, --useGPU)
                  PROP_DEFINE_A(std::string, filename, "", --filename)
                  PROP_DEFINE_A(std::string, cmpPath, "", --comparePath)
                  PROP_DEFINE_A(std::string, imgPath, "", --imgPath)
                  PROP_DEFINE_A(std::string, gtPath, "", --gtPath)
                  PROP_DEFINE_A(std::string, prototxt, "", --prototxt)
                  PROP_DEFINE_A(std::string, model, "", --model)
)

cv::Mat forward(caffe::Net<float>& net, cv::Mat patch, cv::Mat gt)
{
    caffe::Blob<float>* input_layer = net.input_blobs()[0];
    caffe::Blob<float>* input_layer_gt = net.input_blobs()[1];
    caffe::Blob<float>* output_layer = net.output_blobs()[0];

    CHECK(patch.cols == input_layer->width() && patch.rows == input_layer->height()
          && patch.channels() == input_layer->channels())
    << "Patch doesn't have the right dimensions.";

    CHECK(gt.cols == input_layer_gt->width() && gt.rows == input_layer_gt->height()
          && gt.channels() == input_layer_gt->channels())
    << "Ground truth doesn't have the right dimensions.";

    input_layer->Reshape(1, 3, patch.rows, patch.cols);
    input_layer_gt->Reshape(1, 3, gt.rows, gt.cols);
    net.Reshape();

    // Copy over image
    std::vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += input_layer->width() * input_layer->height();
    }
    cv::split(patch, input_channels);

    CHECK(reinterpret_cast<float*>(input_channels.at(0).data) == net.input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

    // Copy over ground truth
    std::copy((float*)gt.data, (float*)gt.dataend, input_layer_gt->mutable_cpu_data());

    net.ForwardPrefilled();

    std::cout << "Loss: " << *output_layer->cpu_data() << std::endl;

    // Copy results back
//    const float* begin = output_layer->cpu_data();
    cv::Mat scores(output_layer->height(), output_layer->width(), CV_32FC(output_layer->channels()));
//    for(int y = 0; y < scores.rows; ++y)
//    {
//        for(int x = 0; x < scores.cols; ++x)
//        {
//            for(int c = 0; c < scores.channels(); ++c)
//                scores.ptr<float>(y)[scores.channels()*x+c] = *(begin + (x + y * output_layer->width() + c * output_layer->width() * output_layer->height()));
//        }
//    }

    return scores;
}

cv::Mat cropPatch(caffe::Net<float>& net, unsigned int x, unsigned int y, cv::Mat const& img)
{
    caffe::Blob<float>* input_layer = net.input_blobs()[0];
    unsigned int patch_w = input_layer->width();
    unsigned int patch_h =  input_layer->height();
    if(static_cast<int>(x + patch_w) > img.cols)
        patch_w = img.cols - x;
    if(static_cast<int>(y + patch_h) > img.rows)
        patch_h = img.rows - y;
    cv::Mat patch = img(cv::Rect(x, y, patch_w, patch_h));

    return patch;
}

cv::Mat padPatch(caffe::Net<float>& net, cv::Mat const& img)
{
    caffe::Blob<float>* input_layer = net.input_blobs()[0];
    unsigned int patch_w = img.cols;
    unsigned int patch_h =  img.rows;

    // Pad with zeros
    unsigned int const pad_w = input_layer->width() - patch_w;
    unsigned int const pad_h = input_layer->height() - patch_h;
    cv::Mat padded_img(input_layer->height(), input_layer->width(), img.type());
    cv::copyMakeBorder(img, padded_img, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0));

    return padded_img;
}

cv::Mat preImg(caffe::Net<float>& net, unsigned int x, unsigned int y, cv::Mat const& rgb_cv)
{
    cv::Mat patch = cropPatch(net, x, y, rgb_cv);

    // Subtract mean
    float const mean_r = 123.680f;
    float const mean_g = 116.779f;
    float const mean_b = 103.939f;

    cv::Mat channels[3];
    cv::split(patch, channels);
    channels[2] -= mean_r;
    channels[1] -= mean_g;
    channels[0] -= mean_b;
    cv::Mat normalized_img(patch.rows, patch.cols, CV_32FC3);
    cv::merge(channels, 3, normalized_img);

    cv::Mat padded_img = padPatch(net, normalized_img);

    return padded_img;
}

enum ErrorCode
{
    SUCCESS = 0,
    CANT_LOAD_IMAGE,
    CANT_LOAD_GT,
};

int main(int argc, char** argv)
{
    // Read properties
    TrainFeatProperties properties;
    properties.read("properties/hseg_train_feat.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // Read in feature map
    FeatureImage stored_features(properties.cmpPath + properties.filename + ".mat");

    // Setup protobuf logging
    ::google::InitGoogleLogging(argv[0]);

    // Init network
    if(properties.useGPU)
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
    else
        caffe::Caffe::set_mode(caffe::Caffe::CPU);

    caffe::Net<float> net(properties.prototxt, caffe::Phase::TRAIN);
    net.CopyTrainedLayersFrom(properties.model);

    std::cout << "#in: " << net.num_inputs() << std::endl;
    std::cout << "#out: " << net.num_outputs() << std::endl;

    caffe::Blob<float>* input_layer = net.input_blobs()[0];
    std::cout << "in_channels: " << input_layer->channels() << std::endl;
    std::cout << "in_width: " << input_layer->width() << std::endl;
    std::cout << "in_height: " << input_layer->height() << std::endl;

    caffe::Blob<float>* gt_layer = net.input_blobs()[1];
    std::cout << "gt_channels: " << gt_layer->channels() << std::endl;
    std::cout << "gt_width: " << gt_layer->width() << std::endl;
    std::cout << "gt_height: " << gt_layer->height() << std::endl;

    caffe::Blob<float>* output_layer = net.output_blobs()[0];
    std::cout << "out_channels: " << output_layer->channels() << std::endl;
    std::cout << "out_width: " << output_layer->width() << std::endl;
    std::cout << "out_height: " << output_layer->height() << std::endl;

    // Load an image
    RGBImage rgb;
    if(!rgb.read(properties.imgPath + properties.filename + ".jpg"))
    {
        std::cerr << "Unable to load image \"" << properties.imgPath + properties.filename + ".jog" << "\"." << std::endl;
        return CANT_LOAD_IMAGE;
    }
    cv::Mat rgb_cv = static_cast<cv::Mat>(rgb);
    rgb_cv.convertTo(rgb_cv, CV_32FC3);

    std::cout << "im_channels: " << rgb_cv.channels() << std::endl;
    std::cout << "im_width: " << rgb_cv.cols << std::endl;
    std::cout << "im_height: " << rgb_cv.rows << std::endl;

    // Load ground truth
    LabelImage gt;
    helper::image::PNGError err = helper::image::readPalettePNG(properties.gtPath + properties.filename + ".png", gt, nullptr);
    if(err != helper::image::PNGError::Okay)
    {
        std::cerr << "Unable to load ground truth \"" << properties.gtPath + properties.filename + ".png" << "\". Error Code: " << (int) err << std::endl;
        return CANT_LOAD_GT;
    }
    cv::Mat gt_cv = static_cast<cv::Mat>(gt);

    // Scale to base size
    unsigned int const base_size = 512;
    unsigned int const long_side = base_size + 1;
    unsigned int new_rows = long_side;
    unsigned int new_cols = long_side;
    if(rgb_cv.rows > rgb_cv.cols)
        new_cols = std::round(long_side / (float)rgb_cv.rows * rgb_cv.cols);
    else
        new_rows = std::round(long_side / (float)rgb_cv.cols * rgb_cv.rows);
    cv::resize(rgb_cv, rgb_cv, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
    cv::resize(gt_cv, gt_cv, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_NEAREST);

    // Crop out parts that have the right dimensions
    CHECK(input_layer->width() == input_layer->height()) << "Input must be square";
    CHECK(output_layer->width() == output_layer->height()) << "Output must be square";
    float const stride_rate = 2.f / 3.f;
    float const crop_size = input_layer->width();
    float const stride = std::ceil(crop_size * stride_rate);
    float const feature_factor = output_layer->width() / static_cast<float>(input_layer->width());
    unsigned int const data_width = static_cast<unsigned int>(std::floor(rgb_cv.cols * feature_factor));
    unsigned int const data_height = static_cast<unsigned int>(std::floor(rgb_cv.rows * feature_factor));
    cv::Mat data(data_height, data_width, CV_32FC(output_layer->channels()), cv::Scalar(0));
    cv::Mat count(data_height, data_width, CV_32FC(1), cv::Scalar(0));
    for(unsigned int y = 0; y < rgb_cv.rows; y += stride)
    {
        for(unsigned int x = 0; x < rgb_cv.cols; x += stride)
        {
            unsigned int s_x = x;
            unsigned int s_y = y;

            // Pad image if necessary and subtract mean
            if(x + input_layer->width() > rgb_cv.cols)
                s_x = std::max(0, rgb_cv.cols - input_layer->width());
            if(y + input_layer->height() > rgb_cv.rows)
                s_y = std::max(0, rgb_cv.rows - input_layer->height());
            cv::Mat padded_img = preImg(net, s_x, s_y, rgb_cv);
            cv::Mat padded_gt = padPatch(net, cropPatch(net, s_x, s_y, gt_cv));
            cv::resize(padded_gt, padded_gt, cv::Size(60, 60), 0, 0, cv::INTER_NEAREST);

            // Run it through the network
            auto features = forward(net, padded_img, padded_gt);
            cv::flip(padded_img, padded_img, 1);
            auto scores_flip = forward(net, padded_img, padded_gt);
//            cv::flip(scores_flip, scores_flip, 1);
//            features += scores_flip;
//            features /= 2;
//
//            // Remove parts that are padded
//            cv::Rect roi(0, 0, features.cols, features.rows);
//            unsigned int f_x = static_cast<unsigned int>(std::floor(s_x * feature_factor));
//            unsigned int f_y = static_cast<unsigned int>(std::floor(s_y * feature_factor));
//            if(f_x + features.cols > data_width)
//                roi.width = data_width - f_x;
//            if(f_y + features.rows > data_height)
//                roi.height = data_height - f_y;
//            cv::Mat croppedFeatures = features(roi);
//
//            // Add to combined score map
//            data(cv::Rect(f_x, f_y, roi.width, roi.height)) += croppedFeatures;
//            count(cv::Rect(f_x, f_y, roi.width, roi.height)) += cv::Scalar_<float>(1);
        }
    }
    //data /= count;
//    std::vector<cv::Mat> channels(data.channels());
//    cv::split(data, channels);
//    for (cv::Mat chan : channels)
//        chan /= count;
//    cv::merge(channels, data);

    // Check whether loaded features are similar to computed features
//    CHECK_EQ(data.cols, stored_features.width());
//    CHECK_EQ(data.rows, stored_features.height());
//    CHECK_EQ(data.channels(), stored_features.dim());
//    float accy = 0.f;
//    float min = std::numeric_limits<float>::max();
//    float max = std::numeric_limits<float>::min();
//    float min_stored = std::numeric_limits<float>::max();
//    float max_stored = std::numeric_limits<float>::min();
//    float max_diff = 0;
//    size_t total = 0;
//    for(int x = 0; x < data.cols; ++x)
//    {
//        for(int y = 0; y < data.rows; ++y)
//        {
//            for(int c = 0; c < data.channels(); ++c)
//            {
//                float const cur = data.ptr<float>(y)[data.channels() * x + c];
//                float const cur_stored = stored_features.at(x, y)(c);
//                float const diff = std::abs(cur - cur_stored);
//                accy += diff;
//                total++;
//                if(cur < min)
//                    min = cur;
//                if(cur > max)
//                    max = cur;
//                if(cur_stored < min_stored)
//                    min_stored = cur_stored;
//                if(cur_stored > max_stored)
//                    max_stored = cur_stored;
//                if(diff > max_diff)
//                    max_diff = diff;
//            }
//        }
//    }
//    accy /= total;
//
//    std::cout << "Accy: " << accy << std::endl;
//    std::cout << "Range: " << min << " - " << max << " (" << min_stored << " - " << max_stored << ")" << std::endl;
//    std::cout << "Max diff: " << max_diff << std::endl;
//
//    cv::imshow("Layer 0", channels[0]);
//    cv::imshow("Layer 1", channels[1]);
//    cv::waitKey();


    return 0;
}
