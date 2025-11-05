#include <cstdlib>
#include <iostream>
#include <vix/ai/ml/Regression.hpp>
#include <vix/ai/ml/Clustering.hpp>
#include <vix/ai/ml/Dataset.hpp>

int main()
{
    using namespace vix::ai::ml;

    LinearRegression lr{2.0, 1.0};
    if (lr.predict(2.0) != 5.0)
    {
        std::cerr << "LinearRegression predict failed\n";
        return EXIT_FAILURE;
    }

    KMeans km{3};
    km.fit();
    if (km.k() != 3)
    {
        std::cerr << "KMeans k() failed\n";
        return EXIT_FAILURE;
    }

    Dataset ds;
    ds.x = {0.0, 1.0};
    ds.y = {1.0, 3.0};
    ds.X = {{0.0, 0.0}, {1.0, 1.0}};
    if (ds.size_supervised() != 2 || ds.size_unsupervised() != 2)
    {
        std::cerr << "Dataset size check failed\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
