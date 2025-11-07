#pragma once
#include "Types.hpp"
#include <string>
#include <optional>

namespace vix::ai::ml
{

    struct Dataset
    {
        // Supervised
        Mat X; // features (n_samples x n_features)
        Vec y; // targets  (n_samples)
        // Unsupervised
        Mat U; // unlabeled features

        std::size_t size_supervised() const noexcept { return X.size(); }
        std::size_t size_unsupervised() const noexcept { return U.size(); }
        std::size_t n_features() const noexcept { return X.empty() ? 0 : X[0].size(); }

        static std::optional<Dataset> from_csv(
            const std::string &path,
            bool has_header = true,
            int target_col = -1 // -1 -> no target column (unsupervised)
        );
    };

} // namespace vix::ai::ml
