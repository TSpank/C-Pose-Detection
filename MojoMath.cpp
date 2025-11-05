/*
 * File: Utils.cpp
 * Project: math
 * File Created: Wednesday, 13th September 2023 9:25:33 am
 * Author: Jeremy E. Coppin (jec@scopetechnology.com)
 * -----
 * Last Modified: Wednesday, 13th September 2023 9:25:39 am
 * Modified By: Jeremy E. Coppin (jec@scopetechnology.com>)
 * -----
 * Copyright 2023 - 2023 Scope Logistical Solutions, Scope Logistical Solutions
 */

#include "MojoMath.hpp"


namespace mojo_math
{


auto approxEqual(double a, double b, int decimalPlaces) -> bool
{
    double epsilon = std::pow(10, -decimalPlaces); // Calculate epsilon based on decimal places
    return std::abs(a - b) < epsilon;
}


} //mojo_math