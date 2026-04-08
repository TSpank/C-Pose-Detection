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
#include "PoseData.h"

// -------------------------------------------------------------
// Struct to hold both quaternions and Euler angles for all poses
// -------------------------------------------------------------
struct PoseResults {
    std::vector<mojo_quaternion::quaternion> quaternions;
    std::vector<mojo_quaternion::quaternion> global_quaternions;
    std::vector<Eigen::Vector3d> eulerAngles;
    nlohmann::json eulerJson;
    nlohmann::json planeJson;
};

// -------------------------------------------------------------
// Struct for perimeter check results
// -------------------------------------------------------------
    struct PerimeterCheckResult {
    bool skip_head = false;

    bool skip_l_arm = false;
    bool skip_r_arm = false;

    bool skip_hips = false;
    bool skip_l_leg = false;
    bool skip_r_leg = false;
};


class Kinematics {
private:
    // ===========================
    // Configuration Constants
    // ===========================
    static constexpr double DEFAULT_ALPHA = 0.3;
    static constexpr double Z_SCALE_SHOULDER = 0.5;
    static constexpr double Z_SCALE_HEAD = 0.3;
    static constexpr double Z_SCALE_MESH_FACTOR = 0.7;
    static constexpr double ARM_ALIGNMENT_THRESHOLD = 0.8;
    static constexpr double LEG_ALIGNMENT_THRESHOLD = 0.7;
    static constexpr double MAX_PALM_JUMP_PIXELS = 40.0;
    static constexpr double TORSO_ANGLE_THRESHOLD_DEG = 30.0;
    static constexpr double HEAD_STRAIGHTNESS_THRESHOLD = 0.052;
    static constexpr double ARM_VERTICAL_THRESHOLD = 0.85;
    static constexpr double NORM_EPSILON = 1e-12;

    // ===========================
    // Private Helpers
    // ===========================
    // Normalize a vector
    static Eigen::Vector3d normalize(const Eigen::Vector3d& v);

    // Convert Eigen quaternion to mojo_quaternion
    mojo_quaternion::quaternion to_mojo(const Eigen::Quaterniond& q) const;
    
    // Flip quaternion if needed to ensure shortest path interpolation
    static void flip_if_needed(const Eigen::Quaterniond& prev, Eigen::Quaterniond& current);
    
    // Apply SLERP smoothing with automatic flipping
    static Eigen::Quaterniond apply_slerp(
        const Eigen::Quaterniond& prev,
        const Eigen::Quaterniond& current,
        double alpha
    );

    Eigen::Quaterniond prev_torso_quat_g_;  // stores last torso quaternion
    bool has_prev_torso_quat_g_;

    Eigen::Quaterniond prev_hip_quat_g_;    // stores last hip quaternion
    bool has_prev_hip_quat_g_;

    Eigen::Quaterniond prev_head_quat_g_;  // stores last head quaternion
    bool has_prev_head_quat_g_;
    
    Eigen::Quaterniond prev_l1_quat_g_;  // stores l1 quaternion
    bool has_prev_l1_quat_g_;     

    Eigen::Quaterniond prev_l2_quat_g_;  // stores l2 quaternion
    bool has_prev_l2_quat_g_;   

    Eigen::Quaterniond prev_r1_quat_g_;  // stores r1 quaternion
    bool has_prev_r1_quat_g_;     

    Eigen::Quaterniond prev_r2_quat_g_;  // stores r2 quaternion
    bool has_prev_r2_quat_g_;   

    Eigen::Quaterniond prev_l_hand_quat_g_;  // stores left hand quaternion
    bool has_prev_l_hand_quat_g_;

    Eigen::Quaterniond prev_r_hand_quat_g_;  // stores right hand quaternion
    bool has_prev_r_hand_quat_g_;

    Eigen::Quaterniond prev_l1_leg_quat_g_;  // stores l1 quaternion
    bool has_prev_l1_leg_quat_g_;     

    Eigen::Quaterniond prev_l2_leg_quat_g_;  // stores l2 quaternion
    bool has_prev_l2_leg_quat_g_;   

    Eigen::Quaterniond prev_r1_leg_quat_g_;  // stores r1 quaternion
    bool has_prev_r1_leg_quat_g_;     

    Eigen::Quaterniond prev_r2_leg_quat_g_;  // stores r2 quaternion
    bool has_prev_r2_leg_quat_g_;   

    Eigen::Quaterniond prev_l_foot_quat_g_;  // stores left foot quaternion
    bool has_prev_l_foot_quat_g_;

    Eigen::Quaterniond prev_r_foot_quat_g_;  // stores right foot quaternion
    bool has_prev_r_foot_quat_g_;

    //Previous qauts for slerp
    Eigen::Quaterniond prev_torso_quat_;  // stores last torso quaternion
    bool has_prev_torso_quat_;

    Eigen::Quaterniond prev_hip_quat_;    // stores last hip quaternion
    bool has_prev_hip_quat_;

    Eigen::Quaterniond prev_head_quat_;  // stores last head quaternion
    bool has_prev_head_quat_;
    
    Eigen::Quaterniond prev_l1_quat_;  // stores l1 quaternion
    bool has_prev_l1_quat_;     

    Eigen::Quaterniond prev_l2_quat_;  // stores l2 quaternion
    bool has_prev_l2_quat_;   

    Eigen::Quaterniond prev_l2_quat_t;  // stores l2 quaternion
    bool has_prev_l2_quat_t;   

    Eigen::Quaterniond prev_r1_quat_;  // stores r1 quaternion
    bool has_prev_r1_quat_;     

    Eigen::Quaterniond prev_r2_quat_;  // stores r2 quaternion
    bool has_prev_r2_quat_;   

    Eigen::Quaterniond prev_r2_quat_t;  // stores r2 quaternion
    bool has_prev_r2_quat_t;  

    Eigen::Quaterniond prev_l_hand_quat_;  // stores left hand quaternion
    bool has_prev_l_hand_quat_;

    Eigen::Quaterniond prev_r_hand_quat_;  // stores right hand quaternion
    bool has_prev_r_hand_quat_;

    Eigen::Quaterniond prev_l1_leg_quat_;  // stores l1 quaternion
    bool has_prev_l1_leg_quat_;     

    Eigen::Quaterniond prev_l2_leg_quat_;  // stores l2 quaternion
    bool has_prev_l2_leg_quat_;   

    Eigen::Quaterniond prev_r1_leg_quat_;  // stores r1 quaternion
    bool has_prev_r1_leg_quat_;     

    Eigen::Quaterniond prev_r2_leg_quat_;  // stores r2 quaternion
    bool has_prev_r2_leg_quat_;   

    Eigen::Quaterniond prev_l_foot_quat_;  // stores left foot quaternion
    bool has_prev_l_foot_quat_;

    Eigen::Quaterniond prev_r_foot_quat_;  // stores right foot quaternion
    bool has_prev_r_foot_quat_;

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

    bool right_leg_aligned_;
    bool left_leg_aligned_;

    Eigen::Vector3d prev_r_handvec_;
    Eigen::Vector3d prev_l_handvec_;

    Eigen::Quaterniond prev_r_hand_quat_recovery_;
    Eigen::Quaterniond prev_l_hand_quat_recovery_;

    double left_hand_lost_count_;
    double right_hand_lost_count_;


public:
    // ===========================
    // Coordinate System:
    // X-axis: Left to Right (left shoulder to right shoulder)
    // Y-axis: Down to Up (hips to shoulders)
    // Z-axis: Back to Front (away from body)
    // ===========================
    
    Kinematics();
    
    // Reset all state (useful when switching subjects or restarting tracking)
    void reset();

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
    std::pair<Eigen::Quaterniond, Eigen::Quaterniond> torso_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.15,
    bool skip = false
);

    // ========================
    // Hip orientation
    // ========================
    Eigen::Quaterniond hip_orientation(
    const PoseData& pose,
    double alpha = 0.15,
    bool skip = false
);

    // ========================
    // Head orientation
    // ========================
    std::pair<Eigen::Quaterniond, Eigen::Quaterniond> head_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.15,
    bool skip = false
);

    // ========================
    // Left Arm orientation
    // ========================
    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> left_arm_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.15,
    bool skip = false
);

    // ========================
    // Right Arm orientation
    // ========================
    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> right_arm_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& torso_quat,
    double alpha = 0.15,
    bool skip = false
);

    // ========================
    // Right Leg orientation  
    // ========================
    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> right_leg_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& hip_quat,
    double alpha = 0.15,
    bool skip = false
);

    // ========================
    // Left Leg orientation
    // ========================
    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> left_leg_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& hip_quat,
    double alpha = 0.15,
    bool skip = false
);

    // ========================
    // Hand normal vector
    // ========================
    std::pair<std::optional<Eigen::Vector3d>, Eigen::Quaterniond> hand_normal_vector(
    const PoseData& pose,
    bool is_left
);

    // ========================
    // Neck kinematics
    // ========================
    PoseResults process_kinematics(
        const PoseData& pose_data);
    
    // ================================
    // Strucuture kinematic output
    // ================================
    PoseResults structure_kinematic_output(
        const Eigen::Quaterniond& torso_quat,const Eigen::Quaterniond& hip_quat,const Eigen::Quaterniond& head_quat,
        const Eigen::Quaterniond& l_upper_quat,const Eigen::Quaterniond& l_lower_quat,const Eigen::Quaterniond& l_hand_quat,
        const Eigen::Quaterniond& r_upper_quat,const Eigen::Quaterniond& r_lower_quat,const Eigen::Quaterniond& r_hand_quat,
        const Eigen::Quaterniond& r_lower_quat_t,const Eigen::Quaterniond& l_lower_quat_t,
        const Eigen::Quaterniond& l_leg_upper_quat,const Eigen::Quaterniond& l_leg_lower_quat,const Eigen::Quaterniond& l_foot_quat,
        const Eigen::Quaterniond& r_leg_upper_quat,const Eigen::Quaterniond& r_leg_lower_quat,const Eigen::Quaterniond& r_foot_quat,

        const Eigen::Quaterniond& torso_quat_g, const Eigen::Quaterniond& head_quat_g,
        const Eigen::Quaterniond& l_upper_quat_g, const Eigen::Quaterniond& l_lower_quat_g, const Eigen::Quaterniond& l_hand_quat_g,
        const Eigen::Quaterniond& r_upper_quat_g, const Eigen::Quaterniond& r_lower_quat_g, const Eigen::Quaterniond& r_hand_quat_g,
        const Eigen::Quaterniond& l_leg_upper_quat_g, const Eigen::Quaterniond& l_leg_lower_quat_g, const Eigen::Quaterniond& l_foot_quat_g,
        const Eigen::Quaterniond& r_leg_upper_quat_g, const Eigen::Quaterniond& r_leg_lower_quat_g, const Eigen::Quaterniond& r_foot_quat_g);

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
    void update_z_scale(const PoseData& pose_data);
    PoseData normalize_z_data(const PoseData& pose_data);

    // ================================
    // Implement smoothing and local rotation computation for limbs
    // ================================ 
    std::pair<Eigen::Quaterniond, Eigen::Quaterniond> smooth_and_compute_local(
        Eigen::Quaterniond& current_global,
        const Eigen::Quaterniond& parent_global,
        Eigen::Quaterniond& prev_global,
        Eigen::Quaterniond& prev_local,
        bool& has_prev,
        double alpha,
        bool skip = false
    );

    // ================================
    // Check in perimter
    // ================================ 
    bool isPointInPerimeter(const PoseData& pose_data, PoseLandmark lm, double margin_ratio = 0.2);
    inline bool areLandmarksInPerimeter(const PoseData& pose, Kinematics& kin,
                                   PoseLandmark lm1, PoseLandmark lm2, double margin_ratio = 0.2);
    PerimeterCheckResult checkBodyPerimeter(
    const PoseData& pose
);











    // ================================ // ================================ // ================================ // ================================ // ================================ 
    // ================================ // ================================ // ================================ // ================================ // ================================ 
    // ================================ // ================================ // ================================ // ================================ // ================================ 
    // ONly for testing
    // Header
    nlohmann::json structure_json_from_quats(
        const std::vector<std::string>& keys,
        const std::vector<mojo_quaternion::quaternion>& quats
    );











};

#endif
