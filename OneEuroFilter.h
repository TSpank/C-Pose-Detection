#pragma once
#include <Eigen/Dense>
#include <cmath>

class OneEuroFilter {
public:
    // Default constructor for array initialization
    OneEuroFilter() : OneEuroFilter(30.0, 0.008, 0.001, 0.7) {}
   
    static OneEuroFilter Hands() {
        return OneEuroFilter(30.0, 0.03, 0.007, 0.7);
    }
    
    OneEuroFilter(double freq, double min_cutoff = 0.008, double beta = 0.001, double d_cutoff = 0.7);

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
