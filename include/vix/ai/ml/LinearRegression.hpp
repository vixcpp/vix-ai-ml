#pragma once
#include "Model.hpp"

namespace vix::ai::ml
{

    class LinearRegression final : public Model
    {
    public:
        // y ≈ a*x + b (ultra basique, sans entraînement réel ici)
        LinearRegression(double a = 1.0, double b = 0.0) : a_(a), b_(b) {}

        double predict(double x) const override { return a_ * x + b_; }

        // mini API factice pour l’exemple
        void set_params(double a, double b)
        {
            a_ = a;
            b_ = b;
        }

    private:
        double a_{};
        double b_{};
    };

} // namespace vix::ai::ml