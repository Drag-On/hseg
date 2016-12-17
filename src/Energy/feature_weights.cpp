//
// Created by jan on 20.10.16.
//

#include <fstream>
#include <iostream>
#include <iomanip>
#include "Energy/feature_weights.h"

Matrix5 readFeatureWeights(std::string const& filename)
{
    Matrix5 weights = Matrix5::Identity();

    std::ifstream in(filename);
    if(in.is_open())
    {
        std::string line;
        std::getline(in, line);
        if(line != "feature_weights")
        {
            in.close();
            std::cerr << "File \"" << filename << "\" does not contain feature weights." << std::endl;
            return weights;
        }
        size_t y = 0;
        while(std::getline(in, line))
        {
            std::istringstream iss(line);
            if(!(iss >> weights(y,0)>> weights(y,1)>> weights(y,2)>> weights(y,3)>> weights(y,4)))
            {
                in.close();
                std::cerr << "Feature weights file \"" << filename << "\" is invalid." << std::endl;
                return weights;
            }
            y++;
        }
        in.close();
    }
    else
        std::cerr << "Can't open feature weights from \"" << filename << "\"." << std::endl;

    return weights;
}

bool writeFeatureWeights(std::string const& filename, Matrix5 const& weights)
{
    std::ofstream out(filename);
    if(out.is_open())
    {
        out << "feature_weights" << std::endl;
        out << weights << std::endl;
        out.close();
        return true;
    }

    return false;
}
