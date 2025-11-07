#pragma once
#include "Types.hpp"
#include <istream>
#include <ostream>

namespace vix::ai::ml
{

    class Model
    {
    public:
        virtual ~Model() = default;

        // Interfaces minimales (supervised/unsupervised)
        virtual void fit(const Mat &X, const Vec &y)
        {
            (void)X;
            (void)y;
        }
        virtual void fit(const Mat &X) { (void)X; } // p.ex. KMeans

        virtual double predict_one(const Vec &x) const
        {
            (void)x;
            return 0.0;
        }
        virtual Vec predict(const Mat &X) const
        {
            Vec out;
            out.reserve(nrows(X));
            for (const auto &row : X)
                out.push_back(predict_one(row));
            return out;
        }

        // Sérialisation textuelle très simple
        virtual void save(std::ostream &os) const { (void)os; }
        virtual void load(std::istream &is) { (void)is; }
    };

} // namespace vix::ai::ml
