#pragma once
#include <cstddef>

namespace vix::ai::ml
{

    class KMeans
    {
    public:
        explicit KMeans(std::size_t k = 2) : k_(k) {}

        void fit()
        {
            // no-op: juste pour compilation
        }

        std::size_t predict() const { return 0; }
        std::size_t k() const noexcept { return k_; }

    private:
        std::size_t k_;
    };

} // namespace vix::ai::ml
