/*
 * File: MojoVector.h
 * Project: math
 * File Created: Friday, 1st September 2023 11:05:03 am
 * -----
 * Last Modified: Friday, 1st September 2023 11:05:43 am
 * Modified By: Jeremy E. Coppin (jec@scopetechnology.com>)
 * -----
 * Copyright 2023 - 2023 Scope Logistical Solutions, Scope Logistical Solutions
 */
#ifndef MATH_MOJOVECTOR_H_
#define MATH_MOJOVECTOR_H_

#include "json.hpp"
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace mojo_vector
{

using json = nlohmann::json;

class vector3
{
public:
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    vector3() = default;
    vector3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    explicit vector3(const json &vectorArray);

    /**
     * @brief Applies an offset to the vector scalar-wise i.e. each axis is offset discretely
     * 
     * @param offset 
     */
    void offset(vector3 offset) 
    {
        x -= offset.x;
        y -= offset.y;
        z -= offset.z;
    }

    auto operator+(const vector3 &other)
    {
        double x = sqrt(pow(this->x, 2) + pow(other.x, 2));
        double y = sqrt(pow(this->y, 2) + pow(other.y, 2));
        double z = sqrt(pow(this->z, 2) + pow(other.z, 2));
        return vector3(x,y,z);
    }

    [[nodiscard]] auto norm() const ->double;
    [[nodiscard]] auto to_string() const -> std::string;
    [[nodiscard]] auto toJson() const -> json;
};

} // namespace mojo_vector

#endif // MATH_MOJOVECTOR_H_

