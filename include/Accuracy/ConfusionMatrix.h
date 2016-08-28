//
// Created by jan on 26.08.16.
//

#ifndef HSEG_CONFUSIONMATRIX_H
#define HSEG_CONFUSIONMATRIX_H


#include <cstddef>
#include <vector>
#include <Image/Image.h>

class ConfusionMatrix
{
public:
    /**
     * Creates a confusion matrix for a given number of classes
     * @param numClasses Amount of classes
     */
    explicit ConfusionMatrix(size_t numClasses) noexcept;

    /**
     * Creates a confusion matrix from a labeling and the ground truth
     * @note This will ignore any index that is larger than the given number of classes
     * @param numClasses Amount of classes
     * @param labeling Label image
     * @param groundTruth Ground truth image
     */
    ConfusionMatrix(size_t numClasses, LabelImage const& labeling, LabelImage const& groundTruth);

    /**
     * Gives the amount of pixels that have been classified as a certain label but should have been another one
     * @param trueLabel True label
     * @param inferredLabel Inferred label
     * @return The amount of pixels
     */
    size_t at(Label trueLabel, Label inferredLabel) const;

    /**
     * Computes the accuracies of each class according to this confusion matrix
     * @details This uses the intersection / union measure, i.e.
     *              true positives / (true positives + false positives + false negatives)
     * @param[out] mean If this is not a nullptr the mean value will be stored here
     * @return The computed accuracies per class
     */
    std::vector<float> accuracies(float* mean = nullptr) const;

    /**
     * Adds more data to this confusion matrix
     * @param labeling Label image
     * @param groundTruth Ground truth image
     */
    void join(LabelImage const& labeling, LabelImage const& groundTruth);

private:
    size_t m_numClasses;
    std::vector<size_t> m_mat; // First index is true label, second index is inferred label

    size_t& at(Label trueLabel, Label inferredLabel);
};

std::ostream& operator<<(std::ostream& stream, ConfusionMatrix const& cf);

#endif //HSEG_CONFUSIONMATRIX_H
