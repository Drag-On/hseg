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
    inline bool isValid() const
    {
        return m_valid;
    }

    /**
     * @return Width of the image
     */
    inline size_t width() const
    {
        return m_width;
    }

    /**
     * @return Height of the image
     */
    inline size_t height() const
    {
        return m_height;
    }

    /**
     * @return Amount of classes
     */
    inline size_t classes() const
    {
        return m_classes;
    }

    /**
     * Retrieve the score of a certain pixel-label combination
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Class label
     * @return Score of the given combination
     */
    inline float at(size_t x, size_t y, size_t c) const
    {
        assert(x + (y * m_width) + (c * m_width * m_height) < m_data.size());
        return m_data[x + (y * m_width) + (c * m_width * m_height)];
    }

    /**
     * Retrieve the score of a certain pixel-label combination
     * @param s Site
     * @param c Class label
     * @return Score of the given combination
     */
    inline float atSite(size_t s, size_t c) const
    {
        assert(s < m_width * m_height);
        assert(s + (c * m_width * m_height) < m_data.size());
        return m_data[s + (c * m_width * m_height)];
    }

    /**
     * Reads in unary data from file
     * @param filename File to read from
     * @return True in case of success, otherwise false
     */
    bool read(std::string const& filename);

    /**
     * Writes the unary data to file
     * @param filename File to write to
     * @return True in case of success, otherwise false
     * @note Always fails if this unary is marked as invalid
     */
    bool write(std::string const& filename);

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
