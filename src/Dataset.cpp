#include "vix/ai/ml/Dataset.hpp"
#include <fstream>
#include <sstream>

namespace vix::ai::ml
{

    static bool parse_line(const std::string &line, std::vector<std::string> &out)
    {
        out.clear();
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ','))
            out.push_back(cell);
        return !out.empty();
    }

    std::optional<Dataset> Dataset::from_csv(const std::string &path, bool has_header, int target_col)
    {
        std::ifstream f(path);
        if (!f.is_open())
            return std::nullopt;

        Dataset ds{};
        std::string line;
        std::vector<std::string> cells;
        if (has_header && std::getline(f, line))
        { /* skip */
        }

        while (std::getline(f, line))
        {
            if (!parse_line(line, cells))
                continue;
            // cast
            std::vector<double> row;
            row.reserve(cells.size());
            for (const auto &c : cells)
                row.push_back(std::stod(c));

            if (target_col >= 0)
            {
                // supervised
                std::vector<double> feat;
                feat.reserve(row.size() - 1);
                for (int j = 0; j < (int)row.size(); ++j)
                {
                    if (j == target_col)
                    {
                        ds.y.push_back(row[j]);
                    }
                    else
                        feat.push_back(row[j]);
                }
                ds.X.push_back(std::move(feat));
            }
            else
            {
                // unsupervised
                ds.U.push_back(std::move(row));
            }
        }
        return ds;
    }

} // namespace vix::ai::ml
