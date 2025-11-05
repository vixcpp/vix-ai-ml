#pragma once
#include <vector>

namespace vix::ai::ml
{

    class Model
    {
    public:
        virtual ~Model() = default;
        virtual double predict(double x) const = 0; // interface minimale
    };

} // namespace vix::ai::ml