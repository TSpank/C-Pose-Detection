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
    nlohmann::json rawJson;
};

class Kinematics {
private:
    // Private helper: normalize a vector
    static Eigen::Vector3d normalize(const Eigen::Vector3d& v);

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

    double z_scale;

    double prev_avatar_depth;
    bool has_prev_avatar_depth;

    Eigen::Vector3d prev_r_handvec;
    Eigen::Vector3d prev_l_handvec;

public:
    Kinematics();

    struct KinematicResults {
    Eigen::Quaterniond torso_quat;
    Eigen::Quaterniond head_quat;
    Eigen::Quaterniond l_upper_quat;
    Eigen::Quaterniond l_lower_quat;
    Eigen::Quaterniond r_upper_quat;
    Eigen::Quaterniond r_lower_quat;
};
    // ========================
    // Determine avatar depth
    // ========================
    double estimate_avatar_depth(
    std::map<std::string, Eigen::Vector3d>& pose_data, 
    double alpha = 0.3);

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
    std::pair<Eigen::Quaterniond, Eigen::Quaterniond> left_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.3
);

    // ========================
    // Right Arm orientation
    // ========================
    std::pair<Eigen::Quaterniond, Eigen::Quaterniond> right_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.3
);

    // ========================
    // Neck kinematics
    // ========================
    PoseResults Kinematics::process_kinematics(
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
        const Eigen::Quaterniond& r_lower_quat);

    // ================================
    // JSON Conversion Helper
    // ================================    
    std::map<std::string, double>
    Kinematics::avatar_json(
         std::vector<Eigen::Vector3d> euler_angles);

    // Z scaling helpers
    void update_z_scale(const std::map<std::string, Eigen::Vector3d>& pose_data);
    std::map<std::string, Eigen::Vector3d> normalize_z_data(const std::map<std::string, Eigen::Vector3d>& pose_data);

};

#endif
