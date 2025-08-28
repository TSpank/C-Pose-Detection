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
    static Eigen::Vector3d project_onto_plane(const Eigen::Vector3d& v,
                                              const Eigen::Vector3d& axis);
    // Computes the rotation angle of an arm vector relative to a reference along a rotation axis
    static double arm_rotation_angle(
        const Eigen::Vector3d& arm_vec,
        const Eigen::Vector3d& rotation_axis,
        const Eigen::Vector3d& reference
    );

    //builds hand refernce fram in order to calculate hand orientation
    static Eigen::Matrix3d build_hand_reference_frame(
        const Eigen::Vector3d& forearm_vec,
        const Eigen::Vector3d& shoulder_vec
    );

    static std::pair<double, double> vector_to_pitch_yaw(
        const Eigen::Vector3d& arm_vec,
        const Eigen::Matrix3d& torso_matrix
    );

    static Eigen::Vector3d toEulerRadians(const Eigen::Matrix3d& rot_matrix);

    double prev_left_angle;
    double prev_right_angle;
    bool prev_left_angle_initialized;
    bool prev_right_angle_initialized;

public:
    Kinematics();

    // Compute torso orientation; returns two maps: angles and quaternions
    std::pair<
        std::map<std::string, Eigen::Vector3d>,
        std::map<std::string, Eigen::Quaterniond>
    > torso_orientation(const std::map<std::string, Eigen::Vector3d>& pose_data);

    // ========================
    // Left Arm orientation
    // ========================
    static std::pair<
        std::map<std::string, Eigen::Vector3d>,
        std::map<std::string, Eigen::Quaterniond>
    > left_arm_orientation(
        const std::map<std::string, Eigen::Vector3d>& pose_data,
        const Eigen::Quaterniond& torso_quat,
        double left_hand_roll
    );

    // ========================
    // Right Arm orientation
    // ========================
    static std::pair<
        std::map<std::string, Eigen::Vector3d>,
        std::map<std::string, Eigen::Quaterniond>
    > right_arm_orientation(
        const std::map<std::string, Eigen::Vector3d>& pose_data,
        const Eigen::Quaterniond& torso_quat,
        double right_hand_roll
    );

    // Compute head & shoulder kinematics and update internal storage
    void kinematics_neck(const std::map<std::string, Eigen::Vector3d>& pose_data);

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
    // Hand Roll
    // ========================
    double compute_hand_roll(
        const Eigen::Vector3d& index_base,
        const Eigen::Vector3d& wrist,
        const Eigen::Vector3d& pinky_base,
        const Eigen::Vector3d& shoulder_vec,
        const Eigen::Matrix3d* rest_orientation = nullptr,
        bool right = true
    );

    // Get left and right hand orientation
    std::pair<double, double> get_hand_orientation(
        const std::map<std::string, Eigen::Vector3d>& pose_data
    );

    // Optional getters for latest stored values
    std::map<std::string, Eigen::Vector3d> latest_angles() const;
    std::map<std::string, Eigen::Quaterniond> latest_quaternions() const;

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
