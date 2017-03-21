/**********************************************************
 * @file   utility.h
 * @author jan
 * @date   21.03.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/
#ifndef HSEG_UTILITY_H
#define HSEG_UTILITY_H

namespace helper
{
    namespace utility
    {
        template <typename T, typename U>
        bool is_equal(const T &t, const U &u)
        {
            return t == u;
        }

        template <typename T, typename U, typename... Others>
        bool is_equal(const T &t, const U &u, Others const &... args)
        {
            return (t == u) && is_equal(u, args...);
        }
    }
}

#endif //HSEG_UTILITY_H
