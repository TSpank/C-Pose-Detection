#ifndef KINEMATICS_H
#define KINEMATICS_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <map>                                                                                        
#include <string>
#include <utility> 
#include <nlohmann/json.hpp>


class Kinematics {
private:
    // Store quaternions and angles as maps of body-part -> value per timestep
    std::vector<std::map<std::string, Eigen::Quaterniond>> kinematic_quaternions;
    std::vector<std::map<std::string, Eigen::Vector3d>> kinematic_angles;

    double prev_angle = 0.0;
    bool prev_angle_initialized = false;

    // Private helper: normalize a vector
    static Eigen::Vector3d normalize(const Eigen::Vector3d& v);
    // Projects a vector onto a plane perpendicular to 'axis'

    static Eigen::Vector3d toEulerRadians(const Eigen::Matrix3d& rot_matrix);

    Eigen::Vector3d to_euler(const Eigen::Quaterniond& q, const std::string& euler = "ZYX");

public:
    Kinematics();

    // ========================
    // Torso orientation
    // ========================
    std::pair<
        std::map<std::string, Eigen::Vector3d>,
        std::map<std::string, Eigen::Quaterniond>
    > torso_orientation(const std::map<std::string, Eigen::Vector3d>& pose_data);

    // ========================
    // Head orientation
    // ========================
    std::pair<
        std::map<std::string, Eigen::Vector3d>,
        std::map<std::string, Eigen::Quaterniond>
    > head_orientation(
        const std::map<std::string, Eigen::Vector3d>& pose_data,
        const Eigen::Quaterniond& torso_quat
    );

    // ========================
    // Left Arm orientation
    // ========================
    std::pair<
        std::map<std::string, Eigen::Vector3d>,
        std::map<std::string, Eigen::Quaterniond>
    > left_arm_orientation(
        const std::map<std::string, Eigen::Vector3d>& pose_data,
        const Eigen::Quaterniond& torso_quat
    );

    // ========================
    // Right Arm orientation
    // ========================
    std::pair<
        std::map<std::string, Eigen::Vector3d>,
        std::map<std::string, Eigen::Quaterniond>
    > right_arm_orientation(
        const std::map<std::string, Eigen::Vector3d>& pose_data,
        const Eigen::Quaterniond& torso_quat
    );

    // Compute head & shoulder kinematics and update internal storage
    void kinematics_neck(const std::map<std::string, Eigen::Vector3d>& pose_data);

    // --------------------------
    // New function: generate structured JSON-like map
    // --------------------------
    static std::map<std::string, double> structure_json_from_kinematics_history_angles(
        const std::vector<std::map<std::string, Eigen::Vector3d>>& kinematic_angles_history,
        bool torso_valid = true,
        bool right_hand = true,
        bool arms = true
    );

    static std::map<std::string, nlohmann::json> structure_json_from_kinematics_history_quats(
        const std::vector<std::map<std::string, Eigen::Quaterniond>>& kinematic_quaternions_history
    );

    // Getter for private kinematic_angles
    const std::vector<std::map<std::string, Eigen::Vector3d>>& get_kinematic_angles() const {
        return kinematic_angles;
    }

    // Getter for private kinematic_quaternions
    const std::vector<std::map<std::string, Eigen::Quaterniond>>& get_kinematic_quaternions() const {
        return kinematic_quaternions;
    }

};

#endif
