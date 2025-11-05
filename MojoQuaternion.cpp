/*
 * File: MojoQuaternion.cpp
 * Project: math
 * File Created: Friday, 1st September 2023 11:16:08 am
 * -----
 * Last Modified: Friday, 1st September 2023 11:16:11 am
 * Modified By: Jeremy E. Coppin (jec@scopetechnology.com>)
 * -----
 * Copyright 2023 - 2023 Scope Logistical Solutions, Scope Logistical Solutions
 */

#include "MojoQuaternion.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mojo_quaternion
{

    using mojo_math::EULER_ALGORITHM;

auto quaternion::to_string() const -> std::string
{
    std::ostringstream oss;
    oss << "quaternion(" << std::fixed << std::setprecision(3) << w << "," << x << "," << y << "," << z << ")";
    return oss.str();
}

auto quaternion::toJson() const -> json
{
    json details = {
        {"w", w},
        {"x", x},
        {"y", y},
        {"z", z}};

    json jquat = {
        {"quat", details}};

    return jquat;
}

auto quaternion::dot(const quaternion &q) const -> double
{
    return quaternion::dot(*this, q);
}

// Static method to calculate the dot product of two quaternions
auto quaternion::dot(const quaternion &q1, const quaternion &q2)-> double
{
    double dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

    // Ensure the dot product is clamped to the range [-1.0, 1.0] due to floating-point precision
    return std::min(1.0, std::max(-1.0, dot));
}

auto quaternion::divide(const quaternion &q, double s) -> quaternion
{
    // Check for division by zero
    if (s == 0.0)
    {
        std::cerr << "Error: Division by zero!\n";
        // Return a default quaternion or handle the error as appropriate.
        return quaternion();
    }

    // Divide each component of q by s
    return quaternion(q.w / s, q.x / s, q.y / s, q.z / s);
}

auto quaternion::multiply(const quaternion &q1, const quaternion &q2) -> quaternion
{
    double w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    double x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    double y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    double z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return quaternion(w, x, y, z);
}

auto quaternion::inverse() const -> quaternion
{
    return quaternion::inverse(*this);
}

auto quaternion::inverse(const quaternion &q) -> quaternion
{
    return quaternion(q.w, -q.x, -q.y, -q.z);
}

auto quaternion::absolute() const -> double
{
    return quaternion::absolute(*this);
}

auto quaternion::absolute(const quaternion &q) -> double
{
    return std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

auto quaternion::normalize(const quaternion &q) -> quaternion
{
    quaternion result;
    double q_abs = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);

    if (q_abs != 0.0)
    {
        double norm = 1.0 / q_abs;
        result.w = q.w * norm;
        result.x = q.x * norm;
        result.y = q.y * norm;
        result.z = q.z * norm;
    }
    else
    {
        // Handle the case where q_abs is zero to avoid division by zero
        std::cerr << "Error: Division by zero in quaternion normalization\n";
    }

    return result;
}

auto quaternion::rotate(const quaternion &q) const -> quaternion
{
    return quaternion::rotate(*this, q);
}

auto quaternion::rotate(const quaternion &q1, const quaternion &q2) -> quaternion
{
    quaternion qr = multiply(q2, q1);
    quaternion qr_i = inverse(q2);
    quaternion qa = multiply(qr, qr_i);
    return qa;
}

auto quaternion::log(const quaternion &q) -> quaternion
{
    double v{0};
    double f{0};
    double b{0};

    b = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    quaternion _q(1.0, 0.0, 0.0, 0.0);

    if (std::abs(b) <= _QUATERNION_EPS * std::abs(q.w))
    {
        if (q.w < 0.0)
        {
            if (std::abs(q.w + 1.0) > _QUATERNION_EPS)
            {
                _q.w = std::log(-q.w);
                _q.x = M_PI;
                _q.y = 0.0;
                _q.z = 0.0;
                return _q;
            }
            else
            {
                _q.w = 0.0;
                _q.x = M_PI;
                _q.y = 0.0;
                _q.z = 0.0;
                return _q;
            }
        }
        else
        {
            _q.w = std::log(-q.w);
            _q.x = 0.0;
            _q.y = 0.0;
            _q.z = 0.0;
            return _q;
        }
    }
    else
    {
        v = std::atan2(b, q.w);
        f = v / b;
        _q.w = std::log(q.w * q.w + b * b) / 2.0;
        _q.x = f * q.x;
        _q.y = f * q.y;
        _q.z = f * q.z;
        return _q;
    }
}

// Exponential function for a quaternion
auto quaternion::q_exp(const quaternion &q) -> quaternion
{
    quaternion _q;

    double vnorm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);

    if (vnorm > _QUATERNION_EPS)
    {
        double s = std::sin(vnorm) / vnorm;
        double e = std::exp(q.w);
        _q.w = e * std::cos(vnorm);
        _q.x = e * s * q.x;
        _q.y = e * s * q.y;
        _q.z = e * s * q.z;
        return _q;
    }
    else
    {
        _q.w = std::exp(q.w);
        return _q;
    }
}

auto quaternion::slerp(const quaternion &q1, double t) const -> quaternion
{
    return slerp(*this, q1, t);
}

auto quaternion::slerp(const quaternion &q0, const quaternion &q1, double t) -> quaternion
{
    quaternion q;

    // Calculate the cosine of the angle between q0 and q1
    double cosHalfTheta = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z;

    // Check if the quaternions are almost parallel
    if (std::abs(cosHalfTheta) >= 1.0)
    {
        return q0;
    }

    // Ensure that we always take the shortest path
    bool reverse_q1 = false;
    if (cosHalfTheta < 0.0)
    {
        reverse_q1 = true;
        cosHalfTheta = -cosHalfTheta;
    }

    // Calculate half the angle between q0 and q1
    double halfTheta = std::acos(cosHalfTheta);
    double sinHalfTheta = std::sqrt(1.0 - cosHalfTheta * cosHalfTheta);

    // If the quaternions are almost collinear, use linear interpolation
    if (std::abs(sinHalfTheta) < 0.001)
    {
        if (!reverse_q1)
        {
            q.w = (1 - t) * q0.w + t * q1.w;
            q.x = (1 - t) * q0.x + t * q1.x;
            q.y = (1 - t) * q0.y + t * q1.y;
            q.z = (1 - t) * q0.z + t * q1.z;
        }
        else
        {
            q.w = (1 - t) * q0.w - t * q1.w;
            q.x = (1 - t) * q0.x - t * q1.x;
            q.y = (1 - t) * q0.y - t * q1.y;
            q.z = (1 - t) * q0.z - t * q1.z;
        }
        return q;
    }

    // Perform SLERP
    double A = std::sin((1 - t) * halfTheta) / sinHalfTheta;
    double B = std::sin(t * halfTheta) / sinHalfTheta;

    if (!reverse_q1)
    {
        q.w = A * q0.w + B * q1.w;
        q.x = A * q0.x + B * q1.x;
        q.y = A * q0.y + B * q1.y;
        q.z = A * q0.z + B * q1.z;
    }
    else
    {
        q.w = A * q0.w - B * q1.w;
        q.x = A * q0.x - B * q1.x;
        q.y = A * q0.y - B * q1.y;
        q.z = A * q0.z - B * q1.z;
    }

    return q;
}

auto quaternion::as_rotation_vector() const -> vector3
{
    return as_rotation_vector(*this);
}

// Function to calculate the rotation vector from a quaternion
auto quaternion::as_rotation_vector(const quaternion &q) -> vector3
{
    vector3 v;
    quaternion n = normalize(q);
    quaternion ql = log(n);
    v.x = 2 * ql.x;
    v.y = 2 * ql.y;
    v.z = 2 * ql.z;
    return v;
}

auto quaternion::as_rotation_vector_offset(const quaternion &offset) const -> vector3
{
    return as_rotation_vector_offset(*this, offset);
}


// Function to apply an offset to a rotation vector
auto quaternion::as_rotation_vector_offset(const quaternion &q, const quaternion &offset) -> vector3
{
    vector3 v1 = as_rotation_vector(q);
    vector3 v2 = as_rotation_vector(offset);
    v1.offset(v2);
    return v1;
}


auto quaternion::as_landmark() const -> vector3
{
    return as_landmark(*this);
}

// Function to calculate the rotation vector from a quaternion
auto quaternion::as_landmark(const quaternion &q) -> vector3
{
    vector3 v;
    quaternion q_unit(0.0, 0.0, 0.0, -1.0);
    quaternion qr(-0.70711, 0.0, 0.0, 0.70711);
    quaternion q_out = rotate(q_unit, multiply(qr, q));
    v.x = q_out.x;
    v.y = q_out.y;
    v.z = q_out.z;
    return v;
}

auto quaternion::as_landmark_offset(const quaternion &offset) const -> vector3
{
    return as_landmark_offset(*this, offset);
}

// Function to apply an offset to a landmark vector
auto quaternion::as_landmark_offset(const quaternion &q, const quaternion &offset) -> vector3
{
    vector3 v1 = as_landmark(q);
    vector3 v2 = as_landmark(offset);
    v1.offset( v2);
    return v1;
}

// auto quaternion::as_matrix() const -> std::array<std::array<double, 3>, 3> 
auto quaternion::as_matrix() const -> Eigen::Matrix3d
{
    return as_matrix(*this);
}
    // Function to convert a quaternion to a 3x3 rotation matrix
// auto quaternion::as_matrix(const quaternion &q) -> std::array<std::array<double, 3>, 3>
auto quaternion::as_matrix(const quaternion &q) -> Eigen::Matrix3d
{
    // std::array<std::array<double, 3>, 3> r = {};
    // r.fill({0.0,0.0,0.0});
    Eigen::Matrix3d r;
    r.fill(0.0);

    quaternion q_n = normalize(q);
    double x = q_n.x;
    double y = q_n.y;
    double z = q_n.z;
    double w = q_n.w;
    r(0, 0) = 1 - 2 * (y * y + z * z);
    r(0, 1) = 2 * x * y - 2 * w * z;
    r(0, 2) = 2 * x * z + 2 * w * y;
    r(1, 0) = 2 * x * y + 2 * w * z;
    r(1, 1) = 1 - 2 * (x * x + z * z);
    r(1, 2) = 2 * y * z - 2 * w * x;
    r(2, 0) = 2 * x * z - 2 * w * y;
    r(2, 1) = 2 * y * z + 2 * w * x;
    r(2, 2) = 1 - 2 * (x * x + y * y);
    return r;
}

// Function to calculate intrinsic rotations alpha, beta, and gamma from a quaternion
auto quaternion::to_euler(const quaternion &q) -> vector3
{
    return to_euler(q, EULER_ALGORITHM::ZXY); 
}

auto quaternion::to_euler(EULER_ALGORITHM type) const -> vector3
{
    return to_euler(*this, type);
}
    // Function to calculate intrinsic rotations alpha, beta, and gamma from a quaternion with specified algorithm
auto quaternion::to_euler(const quaternion &q, EULER_ALGORITHM type) -> vector3
{
    double roll{0};
    double pitch{0};
    double yaw{0};
    quaternion q_n = normalize(q);
    auto r = as_matrix(q_n);

    // Convert quaternion to rotation matrix
    // Fill in the rotation matrix here based on the quaternion elements

    switch (type)
    {
    case EULER_ALGORITHM::XYZ:
        pitch = std::asin(r(0, 2));
        if (std::abs(r(0, 2)) < 0.99999)
        {
            roll = std::atan2(-r(1, 2), r(2, 2));
            yaw = std::atan2(-r(0, 1), r(0, 0));
        }
        else
        {
            roll = std::atan2(r(2, 1), r(1, 1));
            yaw = 0.0;
        }

        break;

    case EULER_ALGORITHM::YXZ:
        roll = std::asin(-r(1, 2));
        if (std::abs(r(1, 2)) < 0.99999)
        {
            pitch = std::atan2(r(0, 2), r(2, 2));
            yaw = std::atan2(r(1, 0), r(1, 1));
        }
        else
        {
            pitch = std::atan2(-r(2, 0), r(0, 0));
            yaw = 0.0;
        }

        break;

    case EULER_ALGORITHM::ZYX:
        pitch = std::asin(-r(2, 0));
        if (std::abs(r(2, 0)) < 0.99999)
        {
            roll = std::atan2(r(2, 1), r(2, 2));
            yaw = std::atan2(r(1, 0), r(0, 0));
        }
        else
        {
            roll = 0.0;
            yaw = std::atan2(-r(0, 1), r(1, 1));
        }
        break;

    case EULER_ALGORITHM::ZXY:
        roll = std::asin(r(2, 1));
        if (std::abs(r(2, 1)) < 0.99999)
        {
            pitch = std::atan2(r(2, 0), r(2, 2));
            yaw = std::atan2(-r(0, 1), r(1, 1));
        }
        else
        {
            pitch = 0.0;
            yaw = std::atan2(r(1, 0), r(0, 0));
        }
        break;

    case EULER_ALGORITHM::YZX:
        yaw = std::asin(r(1, 0));
        if (std::abs(r(1, 0)) < 0.99999)
        {
            roll = std::atan2(-r(1, 2), r(1, 1));
            pitch = std::atan2(-r(2, 0), r(0, 0));
        }
        else
        {
            roll = 0.0;
            pitch = std::atan2(r(0, 2), r(2, 2));
        }
        break;

    case EULER_ALGORITHM::XZY:
        yaw = std::asin(-r(0, 1));
        if (std::abs(r(0, 1)) < 0.99999)
        {
            roll = std::atan2(r(2, 1), r(1, 1));
            pitch = std::atan2(r(0, 2), r(0, 0));
        }
        else
        {
            roll = std::atan2(r(1, 2), r(2, 2));
            pitch = 0.0;
        }
        break;

    default:
        roll = 0.0;
        pitch = 0.0;
        yaw = 0.0;
        break;
    }
    return vector3(roll, pitch, yaw);
}

auto quaternion::to_euler_offset(const quaternion &offset, EULER_ALGORITHM type) const -> vector3
{
    return to_euler_offset(*this, offset, type);
}


auto quaternion::to_euler_offset(const quaternion &q, const quaternion &offset, EULER_ALGORITHM type)->vector3
{
    vector3 v1 = to_euler(q, type);
    vector3 v2 = to_euler(offset, type);
    v1.offset(v2);
    return v1;
}

}