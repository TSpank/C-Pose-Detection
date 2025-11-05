/*
 * File: MojoVector.cpp
 * Project: math
 * File Created: Friday, 1st September 2023 11:05:21 am
 * -----
 * Last Modified: Friday, 1st September 2023 11:05:36 am
 * Modified By: Jeremy E. Coppin (jec@scopetechnology.com>)
 * -----
 * Copyright 2023 - 2023 Scope Logistical Solutions, Scope Logistical Solutions
 */


#include "MojoVector.hpp"


namespace mojo_vector
{

using json = nlohmann::json;

vector3::vector3(const json &vectorArray)
{
    if (vectorArray.contains("x") && vectorArray.contains("y") && vectorArray.contains("z"))
    {
        x = vectorArray["x"];
        y = vectorArray["y"];
        z = vectorArray["z"];
    }
    else
    {
        x = 0.0;
        y = 0.0;
        z = 0.0;
        std::cerr << "JSON Parse ERROR\n";
    }
}

auto vector3::norm() const -> double
{
    return std::sqrt(x * x + y * y + z * z);
}

auto vector3::to_string() const -> std::string
{
    std::ostringstream oss;
    oss << "vector3(" << std::fixed << std::setprecision(3) << x << "," << y << "," << z << ")";
    return oss.str();
}

auto vector3::toJson() const -> json
{
    json details = {
        {"x", x},
        {"y", y},
        {"z", z}};

    json jquat = {
        {"vector", details}};

    return jquat;
}

} // namespace mojo_vector