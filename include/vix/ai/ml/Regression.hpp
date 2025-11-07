#pragma once
#include "Model.hpp"
#include <vector>
#include <random>
#include <limits>
#include <iomanip>

namespace vix::ai::ml
{

    /**
     * @brief Ordinary Least Squares (OLS) linear regression.
     *
     * This model fits a linear function y ≈ w·x + b by minimizing the mean squared error (MSE)
     * using (mini-)batch gradient descent. It supports:
     *  - Multi-feature inputs (X: n_samples x n_features)
     *  - Serialization (save/load) of learned parameters
     *  - A convenience 1D constructor (a, b) and predict_scalar(x) for quick demos
     *
     * Typical usage:
     * @code
     * LinearRegression lr;
     * lr.set_hyperparams(0.1, 3000);
     * lr.fit(X, y);
     * auto yhat = lr.predict(Xtest);
     * @endcode
     *
     * Convenience 1D usage:
     * @code
     * LinearRegression lr(2.0, 1.0); // y ≈ 2x + 1
     * double v = lr.predict_scalar(4.0); // ~9.0
     * @endcode
     *
     * Notes:
     *  - This MVP uses plain gradient descent (no regularization, no adaptive LR).
     *  - Features are assumed to be on comparable scales; consider standardizing X.
     *  - For large-scale problems, prefer minibatching or closed-form solutions when viable.
     */
    class LinearRegression final : public Model
    {
    public:
        /**
         * @brief Default constructor: parameters are uninitialized (w = {}, b = 0).
         *        Call fit() before using predict(), unless you set weights by another means.
         */
        LinearRegression() = default;

        /**
         * @brief Convenience 1D constructor that initializes a single-weight model y ≈ a*x + b.
         * @param a Slope for the single-feature case.
         * @param b Intercept term.
         *
         * This is intended for quick demos/tests. Internally it sets w_ = {a} and b_ = b.
         * Multi-feature training via fit() remains fully supported.
         */
        LinearRegression(double a, double b)
        {
            w_.assign(1, a);
            b_ = b;
        }

        /**
         * @brief Fit the model using batch gradient descent on MSE.
         * @param X Feature matrix (n_samples x n_features).
         * @param y Target vector (n_samples).
         *
         * Complexity: O(iters * n_samples * n_features).
         * Assumes X has consistent row sizes. If n_features == 0 or sizes mismatch, fit() is a no-op.
         */
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
                    const double yhat = dot(X[i], w_) + b_;
                    const double err = yhat - y[i];
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

        /**
         * @brief Predict a single sample (vector) x -> yhat.
         * @param x Feature vector (size == n_features).
         * @return Predicted scalar output.
         */
        double predict_one(const Vec &x) const override { return dot(x, w_) + b_; }

        /**
         * @brief Scalar helper for the 1D case (x is treated as a single-feature vector).
         * @param x Single scalar feature.
         * @return Predicted scalar output.
         *
         * Uses predict_one({x}) so it remains consistent with the base Model API.
         */
        double predict_scalar(double x) const { return predict_one(Vec{x}); }

        /**
         * @brief Set basic optimization hyperparameters.
         * @param lr    Learning rate for gradient descent.
         * @param iters Number of gradient steps.
         */
        void set_hyperparams(double lr, std::size_t iters)
        {
            learning_rate_ = lr;
            max_iters_ = iters;
        }

        /**
         * @brief Save learned parameters in a simple text format.
         * @param os Output stream.
         *
         * Format:
         *   <d> <b>\n
         *   w0 w1 ... w(d-1)\n
         */
        void save(std::ostream &os) const override
        {
            os << std::setprecision(17) << w_.size() << " " << b_ << "\n";
            for (double wi : w_)
                os << wi << " ";
            os << "\n";
        }

        /**
         * @brief Load model parameters saved via save().
         * @param is Input stream.
         */
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

        Vec w_;                       ///< Weight vector (size = n_features). For 1D convenience ctor, size = 1.
        double b_{0.0};               ///< Intercept term.
        double learning_rate_{0.05};  ///< Learning rate for gradient descent.
        std::size_t max_iters_{2000}; ///< Number of gradient steps.
    };

    // (stub) LogisticRegression — kept for future implementation
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
        } // probability placeholder
    };

} // namespace vix::ai::ml
