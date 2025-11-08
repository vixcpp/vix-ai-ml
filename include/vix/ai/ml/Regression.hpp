#pragma once
#include "Model.hpp"
#include <vector>
#include <random>
#include <limits>
#include <iomanip>
#include <algorithm> // shuffle, max_element
#include <cmath>     // std::abs

namespace vix::ai::ml
{

    class LinearRegression final : public Model
    {
    public:
        LinearRegression() = default;

        // Convenience 1D : y ~= a*x + b
        LinearRegression(double a, double b)
        {
            w_.assign(1, a);
            b_ = b;
        }

        // -----------------------------
        // Hyperparamètres “prod”
        // -----------------------------
        void set_hyperparams(double lr,
                             std::size_t iters,
                             std::size_t batch_size = 0, // 0 => full batch
                             double l2 = 0.0,            // régularisation L2 (ridge)
                             bool shuffle = true,
                             double tol = 1e-8,            // min amélioration (early stop)
                             std::size_t patience = 20,    // iters sans amélioration
                             std::size_t verbose_every = 0 // 0 = silencieux
        )
        {
            learning_rate_ = lr;
            max_iters_ = iters;
            batch_size_ = batch_size;
            l2_ = l2;
            shuffle_ = shuffle;
            tol_ = tol;
            patience_ = patience;
            verbose_every_ = verbose_every;
        }

        // -----------------------------
        // Entraînement : Gradient Descent (mini-batch possible)
        // -----------------------------
        void fit(const Mat &X, const Vec &y) override
        {
            const std::size_t m = nrows(X);
            const std::size_t d = ncols(X);
            if (m == 0 || d == 0 || y.size() != m)
                return;

            w_.assign(d, 0.0);
            b_ = 0.0;

            std::vector<std::size_t> idx(m);
            for (std::size_t i = 0; i < m; ++i)
                idx[i] = i;

            std::mt19937 rng(42);

            const double lr = learning_rate_;
            const std::size_t iters = max_iters_;
            const std::size_t B = (batch_size_ == 0 || batch_size_ > m) ? m : batch_size_;

            // Early stopping
            double best_loss = std::numeric_limits<double>::infinity();
            std::size_t bad_rounds = 0;
            Mat batchX;
            Vec batchY;

            for (std::size_t it = 0; it < iters; ++it)
            {
                if (shuffle_)
                    std::shuffle(idx.begin(), idx.end(), rng);

                // itération sur mini-batches
                for (std::size_t start = 0; start < m; start += B)
                {
                    const std::size_t end = std::min(start + B, m);
                    const std::size_t curB = end - start;

                    // Préparer batch
                    batchX.assign(curB, Vec(d, 0.0));
                    batchY.assign(curB, 0.0);
                    for (std::size_t r = 0; r < curB; ++r)
                    {
                        const auto i = idx[start + r];
                        batchX[r] = X[i];
                        batchY[r] = y[i];
                    }

                    // gradients
                    Vec gw(d, 0.0);
                    double gb = 0.0;

                    for (std::size_t r = 0; r < curB; ++r)
                    {
                        const double yhat = dot(batchX[r], w_) + b_;
                        const double err = yhat - batchY[r];
                        for (std::size_t j = 0; j < d; ++j)
                            gw[j] += err * batchX[r][j];
                        gb += err;
                    }

                    const double invB = 1.0 / static_cast<double>(curB);

                    // Ajout régularisation L2 sur w (pas sur b)
                    if (l2_ > 0.0)
                    {
                        for (std::size_t j = 0; j < d; ++j)
                            gw[j] += l2_ * w_[j] * static_cast<double>(curB);
                    }

                    // Pas de gradient (moyenne)
                    for (std::size_t j = 0; j < d; ++j)
                        w_[j] -= lr * (gw[j] * invB);
                    b_ -= lr * (gb * invB);
                }

                // Early stopping : calcul de la loss (MSE + L2)
                const double cur_loss = loss_with_l2(X, y);
                if (verbose_every_ && (it % verbose_every_ == 0))
                {
                    // std::cout << "[it=" << it << "] loss=" << cur_loss << "\n";
                }

                const double gain = best_loss - cur_loss;
                if (gain > tol_)
                {
                    best_loss = cur_loss;
                    bad_rounds = 0;
                }
                else
                {
                    if (++bad_rounds >= patience_)
                        break; // stop anticipé
                }
            }
        }

        // -----------------------------
        // Forme fermée (Normal Equation) avec Ridge
        // -----------------------------
        // Résout argmin ||Xw + b - y||^2 + l2*||w||^2
        // Implémentation : on ajoute une colonne de 1 à X (pour le biais),
        // puis on résout (Z^T Z + Lambda) theta = Z^T y,
        // avec Lambda = diag(l2, ..., l2, 0) — pas de L2 sur le biais.
        void fit_closed_form(const Mat &X, const Vec &y, double l2 = 0.0)
        {
            const std::size_t m = nrows(X);
            const std::size_t d = ncols(X);
            if (m == 0 || d == 0 || y.size() != m)
                return;

            // Construire Z = [X | 1]
            Mat Z(m, Vec(d + 1, 0.0));
            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t j = 0; j < d; ++j)
                    Z[i][j] = X[i][j];
                Z[i][d] = 1.0; // colonne de 1 pour le biais
            }

            // A = Z^T Z, b = Z^T y
            Mat A(d + 1, Vec(d + 1, 0.0));
            Vec bvec(d + 1, 0.0);

            for (std::size_t i = 0; i < d + 1; ++i)
            {
                for (std::size_t j = i; j < d + 1; ++j)
                {
                    double s = 0.0;
                    for (std::size_t r = 0; r < m; ++r)
                        s += Z[r][i] * Z[r][j];
                    A[i][j] = A[j][i] = s;
                }
                double t = 0.0;
                for (std::size_t r = 0; r < m; ++r)
                    t += Z[r][i] * y[r];
                bvec[i] = t;
            }

            // Régularisation : L2 sur les d premières diag (poids), pas sur biais (dernier)
            if (l2 > 0.0)
            {
                for (std::size_t j = 0; j < d; ++j)
                    A[j][j] += l2;
            }

            // Résoudre A * theta = bvec (Gauss avec pivot partiel)
            Vec theta = gaussian_solve(A, bvec); // taille d+1

            // Séparer w et b
            w_.assign(d, 0.0);
            for (std::size_t j = 0; j < d; ++j)
                w_[j] = theta[j];
            b_ = theta[d];
        }

        // -----------------------------
        // Prédiction
        // -----------------------------
        double predict_one(const Vec &x) const override { return dot(x, w_) + b_; }
        double predict_scalar(double x) const { return predict_one(Vec{x}); }

        // -----------------------------
        // Sérialisation (format inchangé)
        // -----------------------------
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

        // Accès utilitaires
        const Vec &weights() const noexcept { return w_; }
        double bias() const noexcept { return b_; }
        double l2() const noexcept { return l2_; }

    private:
        // --- helpers numériques ---
        static double dot(const Vec &a, const Vec &b)
        {
            double s = 0.0;
            const std::size_t d = a.size();
            for (std::size_t j = 0; j < d; ++j)
                s += a[j] * b[j];
            return s;
        }

        double mse_only(const Mat &X, const Vec &y) const
        {
            const std::size_t m = nrows(X);
            if (m == 0)
                return 0.0;
            double s = 0.0;
            for (std::size_t i = 0; i < m; ++i)
            {
                const double e = (dot(X[i], w_) + b_) - y[i];
                s += e * e;
            }
            return s / static_cast<double>(m);
        }

        double loss_with_l2(const Mat &X, const Vec &y) const
        {
            double loss = mse_only(X, y);
            if (l2_ > 0.0)
            {
                double norm2 = 0.0;
                for (double wi : w_)
                    norm2 += wi * wi;
                loss += l2_ * norm2; // biais non régularisé
            }
            return loss;
        }

        // Solveur linéaire très simple : Gauss avec pivot partiel
        static Vec gaussian_solve(Mat A, Vec b)
        {
            const std::size_t n = A.size();
            // Augmenter [A|b]
            for (std::size_t i = 0; i < n; ++i)
                A[i].push_back(b[i]);

            // Elimination
            for (std::size_t col = 0; col < n; ++col)
            {
                // pivot partiel
                std::size_t piv = col;
                double best = std::abs(A[col][col]);
                for (std::size_t r = col + 1; r < n; ++r)
                {
                    double v = std::abs(A[r][col]);
                    if (v > best)
                    {
                        best = v;
                        piv = r;
                    }
                }
                if (best == 0.0)
                    continue; // singulier (on laisse passer)

                if (piv != col)
                    std::swap(A[piv], A[col]);

                // normaliser ligne
                const double diag = A[col][col];
                for (std::size_t c = col; c <= n; ++c)
                    A[col][c] /= diag;

                // éliminer autres lignes
                for (std::size_t r = 0; r < n; ++r)
                {
                    if (r == col)
                        continue;
                    const double f = A[r][col];
                    if (f == 0.0)
                        continue;
                    for (std::size_t c = col; c <= n; ++c)
                        A[r][c] -= f * A[col][c];
                }
            }

            Vec x(n, 0.0);
            for (std::size_t i = 0; i < n; ++i)
                x[i] = A[i][n];
            return x;
        }

        // --- paramètres ---
        Vec w_;
        double b_{0.0};

        // hyperparams
        double learning_rate_{0.05};
        std::size_t max_iters_{2000};
        std::size_t batch_size_{0};
        double l2_{0.0};
        bool shuffle_{true};
        double tol_{1e-8};
        std::size_t patience_{20};
        std::size_t verbose_every_{0};
    };

    // (stub) LogisticRegression — inchangé
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
        }
    };

} // namespace vix::ai::ml
