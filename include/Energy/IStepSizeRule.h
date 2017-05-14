/**********************************************************
 * @file   IStepSizeRule.h
 * @author jan
 * @date   29.04.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/
#ifndef HSEG_ISTEPSIZERULE_H
#define HSEG_ISTEPSIZERULE_H

class Weights;

class IStepSizeRule
{
public:
    virtual void update(Weights& w, Weights const& gradient) = 0;

    virtual bool write(std::string const& folder) = 0;

    virtual bool read(std::string const& folder, size_t t) = 0;
};

#endif //HSEG_ISTEPSIZERULE_H
