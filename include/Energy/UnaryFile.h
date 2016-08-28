//
// Created by jan on 17.08.16.
//

#ifndef HSEG_PROBABILITYFILE_H
#define HSEG_PROBABILITYFILE_H

#include <string>
#include <vector>
#include <Image/Image.h>

/**
 * Loads the scores for an image from file
 */
class UnaryFile
{
public:
    /**
     * Default constructor
     */
    UnaryFile() = default;

    /**
     * Loads the data from file
     * @param filename File to load from
     */
    UnaryFile(std::string const& filename);

    /**
     * @return True in case the object is valid (i.e. the scores have been properly loaded), false otherwise
     */
    bool isValid() const;

    /**
     * @return Width of the image
     */
    size_t width() const;

    /**
     * @return Height of the image
     */
    size_t height() const;

    /**
     * @return Amount of classes
     */
    size_t classes() const;

    /**
     * Retrieve the score of a certain pixel-label combination
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Class label
     * @return Score of the given combination
     */
    float at(size_t x, size_t y, size_t c) const;

    /**
     * Computes the class label with the maximum score at a certain pixel
     * @param x X coordinate
     * @param y Y coordinate
     * @return Class label with maximum score at the given pixel
     */
    Label maxLabelAt(size_t x, size_t y) const;

    /**
     * Computes a labeling by taking he maximum score at every pixel
     * @return The best labeling according to the scores
     */
    LabelImage maxLabeling() const;

private:
    bool m_valid = false;
    size_t m_width = 0;
    size_t m_height = 0;
    size_t m_classes = 21; // Is currently not written to the files
    std::vector<float> m_data;
};


#endif //HSEG_PROBABILITYFILE_H
