#pragma once
#include "Types.hpp"
#include <cmath>
#include <cstddef>

namespace vix::ai::ml
{

    inline double mse(const Vec &y, const Vec &yhat)
    {
        const std::size_t n = y.size();
        if (n == 0 || yhat.size() != n)
            return std::numeric_limits<double>::infinity();
        double s = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            double e = yhat[i] - y[i];
            s += e * e;
        }
        return s / static_cast<double>(n);
    }

    // accuracy binaire sur y in {0,1}, yhat probas -> seuil 0.5
    inline double accuracy01(const Vec &y, const Vec &yhat)
    {
        const std::size_t n = y.size();
        if (n == 0 || yhat.size() != n)
            return 0.0;
        std::size_t ok = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            int p = yhat[i] >= 0.5 ? 1 : 0;
            ok += (p == static_cast<int>(y[i]));
        }
        return static_cast<double>(ok) / static_cast<double>(n);
    }

} // namespace vix::ai::ml
