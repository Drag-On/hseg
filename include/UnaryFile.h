//
// Created by jan on 17.08.16.
//

#ifndef HSEG_PROBABILITYFILE_H
#define HSEG_PROBABILITYFILE_H

#include <string>
#include <vector>

/**
 * Loads the scores for an image from file
 */
class UnaryFile
{
public:
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
    int width() const;

    /**
     * @return Height of the image
     */
    int height() const;

    /**
     * @return Amount of classes
     */
    int classes() const;

    /**
     * Retrieve the score of a certain pixel-label combination
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Class label
     * @return Score of the given combination
     */
    float at(int x, int y, int c) const;

    /**
     * Computes the class label with the maximum score at a certain pixel
     * @param x X coordinate
     * @param y Y coordinate
     * @return Class label with maximum score at the given pixel
     */
    int maxLabelAt(int x, int y) const;

private:
    bool m_valid = false;
    int m_width = 0;
    int m_height = 0;
    int m_classes = 21; // Is currently not written to the files
    std::vector<float> m_data;
};


#endif //HSEG_PROBABILITYFILE_H
