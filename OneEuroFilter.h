#pragma once
#include <Eigen/Dense>
#include <cmath>

class OneEuroFilter {
public:
    OneEuroFilter(double freq, double min_cutoff = 1.0, double beta = 0.0, double d_cutoff = 1.0);

    Eigen::Vector3d operator()(const Eigen::Vector3d& x, double timestamp_ms);

private:
    double freq;
    double min_cutoff;
    double beta;
    double d_cutoff;

    Eigen::Vector3d x_prev;
    Eigen::Vector3d dx_prev;
    bool first_time;
    double last_time;

    double alpha(double cutoff) const;
};
