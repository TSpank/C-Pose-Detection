/*
 * File: MojoQuaternion.h
 * Project: math
 * File Created: Wednesday, 30th August 2023 8:19:26 pm
 * Author: Jeremy E. Coppin (jec@scopetechnology.com)
 * -----
 * Last Modified: Wednesday, 30th August 2023 8:21:15 pm
 * Modified By: Jeremy E. Coppin (jec@scopetechnology.com>)
 * -----
 * Copyright 2023 - 2023 Scope Logistical Solutions, Scope Logistical Solutions
 */

#ifndef MATH_MOJOQUATERNION_H_
#define MATH_MOJOQUATERNION_H_


#include "MojoVector.hpp"
#include "MojoMath.hpp"
#include "json.hpp"
#include "eigen3/Eigen/Dense"
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip> // For setting decimal precision
#include <unordered_map>

namespace mojo_quaternion
{

using json = nlohmann::json;
using mojo_vector::vector3;
using mojo_math::EULER_ALGORITHM;

// using Matrix3x3 = std::array<std::array<double, 3>, 3>;
using Matrix3x3 = Eigen::Matrix3d;

constexpr double _QUATERNION_EPS = 10e-14;

struct quaternion 
{
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    // Constructors
    quaternion(double _w, double _x, double _y, double _z) : w(_w), x(_x), y(_y), z(_z) {}

    explicit quaternion(const std::string &str_w, 
                const std::string &str_x, 
                const std::string &str_y, 
                const std::string &str_z) :         
                                            w(std::stod(str_w)),
                                            x(std::stod(str_x)),
                                            y(std::stod(str_y)),
                                            z(std::stod(str_z))
    {}

    explicit quaternion( const vector3 &rotation)  
    {
        vector3 v(rotation.x / 2.0, rotation.y / 2.0, rotation.z / 2.0);
        *this = q_exp(quaternion(0.0, v.x, v.y, v.z));
    }

    explicit quaternion(const Matrix3x3 &rotMatrix)
    {
        double trace = rotMatrix(0, 0) + rotMatrix(1, 1) + rotMatrix(2, 2);
        auto quat = quaternion{0.0, 0.0, 0.0, 0.0};

        if (trace > 0.0)
        {
            double s = std::sqrt(trace + 1.0) * 2.0;
            quat.x = (rotMatrix(2, 1) - rotMatrix(1, 2)) / s;
            quat.y = (rotMatrix(0, 2) - rotMatrix(2, 0)) / s;
            quat.z = (rotMatrix(1, 0) - rotMatrix(0, 1)) / s;
            quat.w = 0.25 * s;
        }
        else if ((rotMatrix(0, 0) > rotMatrix(1, 1)) && (rotMatrix(0, 0) > rotMatrix(2, 2)))
        {
            double s = std::sqrt(1.0 + rotMatrix(0, 0) - rotMatrix(1, 1) - rotMatrix(2, 2)) * 2.0;
            quat.x = 0.25 * s;
            quat.y = (rotMatrix(0, 1) + rotMatrix(1, 0)) / s;
            quat.z = (rotMatrix(0, 2) + rotMatrix(2, 0)) / s;
            quat.w = (rotMatrix(2, 1) - rotMatrix(1, 2)) / s;
        }
        else if (rotMatrix(1, 1) > rotMatrix(2, 2))
        {
            double s = std::sqrt(1.0 + rotMatrix(1, 1) - rotMatrix(0, 0) - rotMatrix(2, 2)) * 2.0;
            quat.x = (rotMatrix(0, 1) + rotMatrix(1, 0)) / s;
            quat.y = 0.25 * s;
            quat.z = (rotMatrix(1, 2) + rotMatrix(2, 1)) / s;
            quat.w = (rotMatrix(0, 2) - rotMatrix(2, 0)) / s;
        }
        else
        {
            double s = std::sqrt(1.0 + rotMatrix(2, 2) - rotMatrix(0, 0) - rotMatrix(1, 1)) * 2.0;
            quat.x = (rotMatrix(0, 2) + rotMatrix(2, 0)) / s;
            quat.y = (rotMatrix(1, 2) + rotMatrix(2, 1)) / s;
            quat.z = 0.25 * s;
            quat.w = (rotMatrix(1, 0) - rotMatrix(0, 1)) / s;
        }

        // Assuming you have a constructor in your Quaternion class
        // *this = quaternion(quat.w, quat.x, quat.y, quat.z);
        *this = quaternion(quat.w, quat.x, quat.y, quat.z);
    }

    static auto from_rotation_vector (const vector3 &rotation) -> quaternion
    {
        vector3 v(rotation.x / 2.0, rotation.y / 2.0, rotation.z / 2.0);
        quaternion q(0.0, v.x, v.y, v.z);
        return q_exp(q);
    }

    // Default constructor
    quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}

    // Copy constructor
    quaternion(const quaternion &other) = default;

    // Move constructor
    quaternion(quaternion &&other) noexcept = default;

    // Destructor
    ~quaternion() = default;

    // Copy assignment operator
    auto operator=(const quaternion &other) -> quaternion & = default;

    // Move assignment operator
    auto operator=(quaternion &&other) noexcept -> quaternion & = default;

    auto operator*(const quaternion &other) const -> quaternion
    {
        return quaternion::multiply(*this, other);
    }   
    
    auto operator/(const double &other) const -> quaternion
    {
        return quaternion::divide(*this, other);
    }

    auto operator==(const quaternion &other) const -> bool
    {
        auto q1 = quaternion::normalize(*this);
        auto q2 = quaternion::normalize(other);

        return ((mojo_math::approxEqual(q1.w, q2.w, 4)) &&
                (mojo_math::approxEqual(q1.x, q2.x, 4)) &&
                (mojo_math::approxEqual(q1.y, q2.y, 4)) &&
                (mojo_math::approxEqual(q1.z, q2.z, 4)));
    }


    [[nodiscard]] auto dot(const quaternion &q) const -> double;
    [[nodiscard]] auto inverse() const -> quaternion;
    [[nodiscard]] auto absolute() const -> double;
    [[nodiscard]] auto rotate(const quaternion &q) const ->  quaternion;
    [[nodiscard]] auto slerp(const quaternion &q1, double t) const -> quaternion;
    [[nodiscard]] auto as_rotation_vector() const -> vector3;
    [[nodiscard]] auto as_rotation_vector_offset(const quaternion &offset) const -> vector3;
    [[nodiscard]] auto as_landmark() const -> vector3;
    [[nodiscard]] auto as_landmark_offset(const quaternion &offset) const -> vector3;
    [[nodiscard]] auto as_matrix() const -> Eigen::Matrix3d;
    // [[nodiscard]] auto as_matrix() const -> std::array<std::array<double, 3>, 3>;
    [[nodiscard]] auto to_euler() const -> vector3;
    [[nodiscard]] auto to_euler(EULER_ALGORITHM type) const -> vector3;
    [[nodiscard]] auto to_euler_offset(const quaternion &offset, EULER_ALGORITHM type) const -> vector3;
    [[nodiscard]] auto to_string() const -> std::string;
    [[nodiscard]] auto toJson() const -> json;
    // Static method to calculate the dot product of two quaternions
    
    [[nodiscard]] static auto dot(const quaternion &q1, const quaternion &q2) -> double;
    [[nodiscard]] static auto divide(const quaternion &q, double s) -> quaternion;
    [[nodiscard]] static auto multiply(const quaternion &q1, const quaternion &q2) -> quaternion;
    [[nodiscard]] static auto inverse(const quaternion &q) -> quaternion;
    [[nodiscard]] static auto absolute(const quaternion &q) -> double;
    [[nodiscard]] static auto normalize(const quaternion &q) -> quaternion;
    [[nodiscard]] static auto rotate(const quaternion &q1, const quaternion &q2) -> quaternion;
    [[nodiscard]] static auto log(const quaternion &q) -> quaternion;
    [[nodiscard]] static auto q_exp(const quaternion &q) -> quaternion;
    [[nodiscard]] static auto slerp(const quaternion &q0, const quaternion &q1, double t) -> quaternion;
    [[nodiscard]] static auto as_rotation_vector(const quaternion &q) -> vector3;
    [[nodiscard]] static auto as_rotation_vector_offset(const quaternion &q, const quaternion &offset) -> vector3;
    [[nodiscard]] static auto as_landmark(const quaternion &q) -> vector3;
    [[nodiscard]] static auto as_landmark_offset(const quaternion &q, const quaternion &offset) -> vector3;
    // [[nodiscard]] static auto as_matrix(const quaternion &q) -> std::array<std::array<double, 3>, 3>;
    [[nodiscard]] static auto as_matrix(const quaternion &q) -> Eigen::Matrix3d;

    [[nodiscard]] static auto to_euler(const quaternion &q) -> vector3;
    [[nodiscard]] static auto to_euler(const quaternion &q, EULER_ALGORITHM type) -> vector3;
    [[nodiscard]] static auto to_euler_offset(const quaternion &q, const quaternion &offset, EULER_ALGORITHM type) -> vector3;

};





}
#endif // MATH_MOJOQUATERNION_H_
