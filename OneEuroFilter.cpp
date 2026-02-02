#include "OneEuroFilter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


OneEuroFilter::OneEuroFilter(double freq, double min_cutoff, double beta, double d_cutoff)
    : freq(freq), min_cutoff(min_cutoff), beta(beta), d_cutoff(d_cutoff),
      x_prev(Eigen::Vector3d::Zero()), dx_prev(Eigen::Vector3d::Zero()), first_time(true), last_time(0.0) {}

Eigen::Vector3d OneEuroFilter::operator()(const Eigen::Vector3d& x, double timestamp_ms) {
    double dt;
    if (first_time) {
        dt = 1.0 / freq;
    } else {
        dt = (timestamp_ms - last_time) / 1000.0;
        if (dt > 0.0) freq = 1.0 / dt;
    }
    last_time = timestamp_ms;

    if (first_time) {
        x_prev = x;
        first_time = false;
        return x;
    }

    Eigen::Vector3d dx = (x - x_prev) * freq;
    double alpha_d = alpha(d_cutoff);
    Eigen::Vector3d dx_hat = alpha_d * dx + (1.0 - alpha_d) * dx_prev;

    Eigen::Array3d cutoff = Eigen::Array3d::Constant(min_cutoff) + beta * dx_hat.array().abs();
    Eigen::Array3d alpha_x = cutoff.unaryExpr([this](double c){ return alpha(c); });
    Eigen::Vector3d x_hat = alpha_x.matrix().cwiseProduct(x) + (Eigen::Vector3d::Ones() - alpha_x.matrix()).cwiseProduct(x_prev);

    x_prev = x_hat;
    dx_prev = dx_hat;

    return x_hat;
}

double OneEuroFilter::alpha(double cutoff) const {
    double tau = 1.0 / (2.0 * M_PI * cutoff);
    return 1.0 / (1.0 + tau * freq);
}

