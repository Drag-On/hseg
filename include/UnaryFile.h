//
// Created by jan on 17.08.16.
//

#ifndef HSEG_PROBABILITYFILE_H
#define HSEG_PROBABILITYFILE_H

#include <string>
#include <vector>

class UnaryFile
{
public:
    UnaryFile(std::string const& filename);

    bool isValid() const;

    int width() const;

    int height() const;

    int classes() const;

    float at(int x, int y, int c) const;

    int maxLabelAt(int x, int y) const;

private:
    bool m_valid = false;
    int m_width = 0;
    int m_height = 0;
    int m_classes = 21; // Is currently not written to the files
    std::vector<float> m_data;
};


#endif //HSEG_PROBABILITYFILE_H
