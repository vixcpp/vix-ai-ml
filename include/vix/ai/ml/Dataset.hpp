#pragma once
#include <vector>
#include <cstddef>

namespace vix::ai::ml
{

    struct Dataset
    {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<std::vector<double>> X;

        std::size_t size_supervised() const noexcept { return x.size(); }
        std::size_t size_unsupervised() const noexcept { return X.size(); }

        // alias pratique
        std::size_t size() const noexcept { return size_supervised(); }
    };

} // namespace vix::ai::ml
