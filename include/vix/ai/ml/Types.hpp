#pragma once
#include <vector>
#include <cstddef>

namespace vix::ai::ml
{
    using Vec = std::vector<double>;
    using Mat = std::vector<Vec>; // Mat[row][col]
    using Idxs = std::vector<std::size_t>;

    inline std::size_t nrows(const Mat &m) { return m.size(); }
    inline std::size_t ncols(const Mat &m) { return m.empty() ? 0 : m[0].size(); }
}
