#pragma once
#include "Model.hpp"
#include <random>
#include <limits>

namespace vix::ai::ml
{

    class KMeans final : public Model
    {
    public:
        explicit KMeans(std::size_t k = 2, std::size_t max_iters = 100) : k_(k), max_iters_(max_iters) {}

        void fit(const Mat &X) override
        {
            const auto m = nrows(X), d = ncols(X);
            if (m == 0 || d == 0 || k_ == 0)
                return;

            // init: k centroïdes aléatoires
            std::mt19937 rng(42);
            std::uniform_int_distribution<std::size_t> dist(0, m - 1);
            centers_.assign(k_, Vec(d, 0.0));
            for (std::size_t c = 0; c < k_; ++c)
                centers_[c] = X[dist(rng)];

            Idxs assign(m, 0);
            for (std::size_t it = 0; it < max_iters_; ++it)
            {
                bool changed = false;

                // E-step: assignation
                for (std::size_t i = 0; i < m; ++i)
                {
                    std::size_t best = 0;
                    double bestd = std::numeric_limits<double>::infinity();
                    for (std::size_t c = 0; c < k_; ++c)
                    {
                        double d2 = sqdist(X[i], centers_[c]);
                        if (d2 < bestd)
                        {
                            bestd = d2;
                            best = c;
                        }
                    }
                    if (assign[i] != best)
                    {
                        assign[i] = best;
                        changed = true;
                    }
                }
                // M-step: recalcul des centres
                Mat newC(k_, Vec(d, 0.0));
                Vec cnt(k_, 0.0);
                for (std::size_t i = 0; i < m; ++i)
                {
                    auto c = assign[i];
                    cnt[c] += 1.0;
                    for (std::size_t j = 0; j < d; ++j)
                        newC[c][j] += X[i][j];
                }
                for (std::size_t c = 0; c < k_; ++c)
                {
                    if (cnt[c] > 0)
                        for (std::size_t j = 0; j < d; ++j)
                            newC[c][j] /= cnt[c];
                    else
                        newC[c] = centers_[c]; // cluster vide: garder ancien centre
                }
                centers_.swap(newC);
                if (!changed)
                    break;
            }
        }

        // renvoie l'indice du centre le plus proche
        std::size_t predict_label(const Vec &x) const
        {
            std::size_t best = 0;
            double bestd = std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < k_; ++c)
            {
                double d2 = sqdist(x, centers_[c]);
                if (d2 < bestd)
                {
                    bestd = d2;
                    best = c;
                }
            }
            return best;
        }

        double predict_one(const Vec &x) const override
        {
            // valeur numérique pour interface Model (cluster id)
            return static_cast<double>(predict_label(x));
        }

        const Mat &centers() const noexcept { return centers_; }
        std::size_t k() const noexcept { return k_; }

    private:
        static double sqdist(const Vec &a, const Vec &b)
        {
            double s = 0;
            for (std::size_t j = 0; j < a.size(); ++j)
            {
                double z = a[j] - b[j];
                s += z * z;
            }
            return s;
        }
        std::size_t k_;
        std::size_t max_iters_;
        Mat centers_;
    };

} // namespace vix::ai::ml
