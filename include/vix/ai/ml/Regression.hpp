#pragma once
#include "Model.hpp"
#include <vector>
#include <random>
#include <limits>
#include <iomanip>

namespace vix::ai::ml
{

    // y ≈ a*x + b  (implémentation multi-features: y ≈ w·x + b)
    class LinearRegression final : public Model
    {
    public:
        void fit(const Mat &X, const Vec &y) override
        {
            const std::size_t m = nrows(X);
            const std::size_t d = ncols(X);
            if (m == 0 || d == 0 || y.size() != m)
                return;

            w_.assign(d, 0.0);
            b_ = 0.0;
            const double lr = learning_rate_;
            const std::size_t iters = max_iters_;

            for (std::size_t it = 0; it < iters; ++it)
            {
                // gradients
                Vec gw(d, 0.0);
                double gb = 0.0;
                for (std::size_t i = 0; i < m; ++i)
                {
                    double yhat = dot(X[i], w_) + b_;
                    double err = yhat - y[i];
                    for (std::size_t j = 0; j < d; ++j)
                        gw[j] += err * X[i][j];
                    gb += err;
                }
                const double inv_m = 1.0 / static_cast<double>(m);
                for (std::size_t j = 0; j < d; ++j)
                    w_[j] -= lr * (gw[j] * inv_m);
                b_ -= lr * (gb * inv_m);
            }
        }

        double predict_one(const Vec &x) const override { return dot(x, w_) + b_; }

        void set_hyperparams(double lr, std::size_t iters)
        {
            learning_rate_ = lr;
            max_iters_ = iters;
        }

        void save(std::ostream &os) const override
        {
            os << std::setprecision(17) << w_.size() << " " << b_ << "\n";
            for (double wi : w_)
                os << wi << " ";
            os << "\n";
        }
        void load(std::istream &is) override
        {
            std::size_t d = 0;
            is >> d >> b_;
            w_.assign(d, 0.0);
            for (std::size_t j = 0; j < d; ++j)
                is >> w_[j];
        }

    private:
        static double dot(const Vec &a, const Vec &b)
        {
            double s = 0;
            const std::size_t d = a.size();
            for (std::size_t j = 0; j < d; ++j)
                s += a[j] * b[j];
            return s;
        }
        Vec w_;
        double b_{0.0};
        double learning_rate_{0.05};
        std::size_t max_iters_{2000};
    };

    // (stub) LogisticRegression — interface compatible; entraînement à venir
    class LogisticRegression final : public Model
    {
    public:
        void fit(const Mat &X, const Vec &y) override
        {
            (void)X;
            (void)y; /* TODO */
        }
        double predict_one(const Vec &x) const override
        {
            (void)x;
            return 0.5;
        } // proba
    };

} // namespace vix::ai::ml
