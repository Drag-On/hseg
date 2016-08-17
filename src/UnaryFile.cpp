//
// Created by jan on 17.08.16.
//

#include <fstream>
#include <cstring>
#include <boost/endian/conversion.hpp>
#include "UnaryFile.h"

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
        m_height = boost::endian::little_to_native(*reinterpret_cast<int*>(fileHeader + 4));
        m_width = boost::endian::little_to_native(*reinterpret_cast<int*>(fileHeader + 8));

        m_data.resize((fileSize - 12) / sizeof(float));
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

int UnaryFile::width() const
{
    return m_width;
}

int UnaryFile::height() const
{
    return m_height;
}

int UnaryFile::classes() const
{
    return m_classes;
}

float UnaryFile::at(int x, int y, int c) const
{
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}

int UnaryFile::maxLabelAt(int x, int y) const
{
    int maxLabel = 0;
    float maxVal = std::numeric_limits<float>::min();
    for(int l = 0; l < m_classes; ++l)
    {
        if(at(x, y, l) > maxVal)
        {
            maxVal = at(x, y, l);
            maxLabel = l;
        }
    }
    return maxLabel;
}
