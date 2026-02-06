#ifndef KINEMATICS_H
#define KINEMATICS_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <map>                                                                                        
#include <string>
#include <utility> 
#include <nlohmann/json.hpp>
#include "MojoQuaternion.hpp"

// -------------------------------------------------------------
// Struct to hold both quaternions and Euler angles for all poses
// -------------------------------------------------------------
struct PoseResults {
    std::vector<mojo_quaternion::quaternion> quaternions;
    std::vector<Eigen::Vector3d> eulerAngles;
    nlohmann::json eulerJson;
    nlohmann::json planeJson;
};

class Kinematics {
private:
    // Private helper: normalize a vector
    static Eigen::Vector3d normalize(const Eigen::Vector3d& v);

    // Private helper: convert Eigen quaternion to mojo_quaternion
    mojo_quaternion::quaternion to_mojo(const Eigen::Quaterniond& q) const;

    //Previous qauts for slerp
    Eigen::Quaterniond prev_torso_quat_;  // stores last torso quaternion
    bool has_prev_torso_quat_;     

    Eigen::Quaterniond prev_head_quat_;  // stores last head quaternion
    bool has_prev_head_quat_;
    
    Eigen::Quaterniond prev_l1_quat_;  // stores l1 quaternion
    bool has_prev_l1_quat_;     

    Eigen::Quaterniond prev_l2_quat_;  // stores l2 quaternion
    bool has_prev_l2_quat_;   

    Eigen::Quaterniond prev_r1_quat_;  // stores r1 quaternion
    bool has_prev_r1_quat_;     

    Eigen::Quaterniond prev_r2_quat_;  // stores r2 quaternion
    bool has_prev_r2_quat_;   

    Eigen::Quaterniond prev_l2_quat_g_;  // stores l2 quaternion g
    bool has_prev_l2_quat_g_;   

    Eigen::Quaterniond prev_r2_quat_g_;  // stores r2 quaternion g
    bool has_prev_r2_quat_g_;   

    double z_scale;
    double z_scale_mesh;

    //items for good plan angles
    bool right_hand_state_;
    bool left_hand_state_;

    double prev_l_hand_rot_;
    double prev_l_hand_rot_g_;
    double prev_r_hand_rot_;
    double prev_r_hand_rot_g_;

    bool right_arm_aligned_;
    bool left_arm_aligned_;

public:
    Kinematics();

    struct KinematicResults {
    Eigen::Quaterniond torso_quat;
    Eigen::Quaterniond head_quat;
    Eigen::Quaterniond l_upper_quat;
    Eigen::Quaterniond l_lower_quat;
    Eigen::Quaterniond r_upper_quat;
    Eigen::Quaterniond r_lower_quat;
    Eigen::Quaterniond r_lower_quat_g;
    Eigen::Quaterniond l_lower_quat_g;
};

    // ========================
    // Torso orientation
    // ========================
    Eigen::Quaterniond torso_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    double alpha = 0.3
);

    // ========================
    // Head orientation
    // ========================
    Eigen::Quaterniond head_orientation(
    const std::map<std::string, Eigen::Vector3d>& face_mesh_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.3
);

    // ========================
    // Left Arm orientation
    // ========================
    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> left_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.3
);

    // ========================
    // Right Arm orientation
    // ========================
    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> right_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.3
);

    // ========================
    // Neck kinematics
    // ========================
    PoseResults process_kinematics(
        const std::map<std::string, Eigen::Vector3d>& pose_data);
    
    // ================================
    // Strucuture kinematic output
    // ================================
    PoseResults structure_kinematic_output(
        const Eigen::Quaterniond& torso_quat,
        const Eigen::Quaterniond& head_quat,
        const Eigen::Quaterniond& l_upper_quat,
        const Eigen::Quaterniond& l_lower_quat,
        const Eigen::Quaterniond& r_upper_quat,
        const Eigen::Quaterniond& r_lower_quat,
        const Eigen::Quaterniond& r_lower_quat_g,
        const Eigen::Quaterniond& l_lower_quat_g);

    // ================================
    // JSON Conversion Helper
    // ================================    
    std::map<std::string, double>
    avatar_json(
         std::vector<Eigen::Vector3d> euler_angles);

    // ================================
    // JSON Conversion Helper plane angles
    // ================================    
    std::map<std::string, double>
    json_isolated_angles(
         std::vector<mojo_quaternion::quaternion>& quaternions,
         std::vector<Eigen::Vector3d>& euler_angles);

    // Z scaling helpers
    void update_z_scale(const std::map<std::string, Eigen::Vector3d>& pose_data);
    std::map<std::string, Eigen::Vector3d> normalize_z_data(const std::map<std::string, Eigen::Vector3d>& pose_data);

    // Not for app
    // Header
    nlohmann::json structure_json_from_quats(
        const std::vector<std::string>& keys,
        const std::vector<mojo_quaternion::quaternion>& quats
    );

};

#endif
