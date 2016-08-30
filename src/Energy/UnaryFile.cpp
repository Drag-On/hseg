//
// Created by jan on 17.08.16.
//

#include <fstream>
#include <cstring>
#include <boost/endian/conversion.hpp>
#include "Energy/UnaryFile.h"

UnaryFile::UnaryFile(std::string const& filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
    if (file.is_open())
    {
        long int fileSize = file.tellg();
        char fileHeader[12];
        file.seekg(0, std::ios::beg);
        file.read(fileHeader, 12);

        if (std::strncmp(fileHeader, "PROB", 4) != 0)
        {
            m_valid = false;
            file.close();
            return;
        }
        m_height = static_cast<size_t>(boost::endian::little_to_native(*reinterpret_cast<int*>(fileHeader + 4)));
        m_width = static_cast<size_t>(boost::endian::little_to_native(*reinterpret_cast<int*>(fileHeader + 8)));

        m_data.resize((static_cast<size_t>(fileSize) - 12) / sizeof(float));
        file.seekg(12, std::ios::beg);
        file.read(reinterpret_cast<char*>(m_data.data()), fileSize - 12);
        file.close();

        m_valid = true;
    }
}

bool UnaryFile::isValid() const
{
    return m_valid;
}

size_t UnaryFile::width() const
{
    return m_width;
}

size_t UnaryFile::height() const
{
    return m_height;
}

size_t UnaryFile::classes() const
{
    return m_classes;
}

float UnaryFile::at(size_t x, size_t y, size_t c) const
{
    assert(x + (y * m_width) + (c * m_width * m_height) < m_data.size());
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}

float UnaryFile::atSite(size_t s, size_t c) const
{
    assert(s < m_width * m_height);
    assert(s + (c * m_width * m_height) < m_data.size());
    return m_data[s + (c * m_width * m_height)];
}

Label UnaryFile::maxLabelAt(size_t x, size_t y) const
{
    Label maxLabel = 0;
    float maxVal = std::numeric_limits<float>::min();
    for (Label l = 0; l < m_classes; ++l)
    {
        if (at(x, y, l) > maxVal)
        {
            maxVal = at(x, y, l);
            maxLabel = l;
        }
    }
    return maxLabel;
}

LabelImage UnaryFile::maxLabeling() const
{
    LabelImage labeling(m_width, m_height);
    for (size_t x = 0; x < m_width; ++x)
        for (size_t y = 0; y < m_height; ++y)
            labeling.at(x, y) = maxLabelAt(x, y);
    return labeling;
}
