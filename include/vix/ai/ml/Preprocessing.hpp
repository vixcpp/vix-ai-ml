#pragma once
#include "Types.hpp"
#include <tuple>
#include <algorithm>
#include <cmath>

namespace vix::ai::ml
{

    struct StandardScaler
    {
        Vec mean;
        Vec std;

        void fit(const Mat &X)
        {
            const auto m = nrows(X), d = ncols(X);
            mean.assign(d, 0.0);
            std.assign(d, 0.0);
            if (m == 0 || d == 0)
                return;
            for (std::size_t j = 0; j < d; ++j)
            {
                double s = 0;
                for (std::size_t i = 0; i < m; ++i)
                    s += X[i][j];
                mean[j] = s / static_cast<double>(m);
                double v = 0;
                for (std::size_t i = 0; i < m; ++i)
                {
                    double z = X[i][j] - mean[j];
                    v += z * z;
                }
                std[j] = std::sqrt(v / static_cast<double>(m > 1 ? m - 1 : 1));
                if (std[j] == 0)
                    std[j] = 1.0; // éviter division par 0 (feature constante)
            }
        }
        Mat transform(const Mat &X) const
        {
            Mat Z = X;
            const auto m = nrows(X), d = ncols(X);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < d; ++j)
                    Z[i][j] = (X[i][j] - mean[j]) / std[j];
            return Z;
        }
        Mat fit_transform(const Mat &X)
        {
            fit(X);
            return transform(X);
        }
    };

} // namespace vix::ai::ml
