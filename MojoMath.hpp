/*
 * File: MojoMath.h
 * Project: math
 * File Created: Wednesday, 13th September 2023 9:25:00 am
 * Author: Jeremy E. Coppin (jec@scopetechnology.com)
 * -----
 * Last Modified: Wednesday, 13th September 2023 9:25:08 am
 * Modified By: Jeremy E. Coppin (jec@scopetechnology.com>)
 * -----
 * Copyright 2023 - 2023 Scope Logistical Solutions, Scope Logistical Solutions
 */
#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#include <cmath>

namespace mojo_math
{
    enum class EULER_ALGORITHM
    {
        XYZ,
        YXZ,
        ZYX,
        ZXY,
        YZX,
        XZY
    };

    auto approxEqual(double a, double b, int decimalPlaces) -> bool;
}
#endif // MATH_UTILS_H_