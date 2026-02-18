#include "kinematic_model.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

#include <Eigen/Geometry>     // For Eigen::Quaterniond
#include <cmath>              // For std::abs, std::atan2, M_PI, std::round
#include <stdexcept>          // For std::invalid_argument

#include <utility> // for std::pair
#include <cctype>  // for std::isupper

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========================
// Constructor
// ========================
Kinematics::Kinematics()
    :prev_torso_quat_(1, 0, 0, 0), has_prev_torso_quat_(false), 
     prev_hip_quat_(1, 0, 0, 0), has_prev_hip_quat_(false),
     prev_head_quat_(1, 0, 0, 0), has_prev_head_quat_(false),
     prev_l1_quat_(1, 0, 0, 0), has_prev_l1_quat_(false),
     prev_l2_quat_(1, 0, 0, 0), has_prev_l2_quat_(false),
     prev_r1_quat_(1, 0, 0, 0), has_prev_r1_quat_(false),
     prev_r2_quat_(1, 0, 0, 0), has_prev_r2_quat_(false),
     prev_r2_quat_g_(1, 0, 0, 0), has_prev_r2_quat_g_(false),
     prev_l2_quat_g_(1, 0, 0, 0), has_prev_l2_quat_g_(false),
     prev_l_hand_quat_(1, 0, 0, 0), has_prev_l_hand_quat_(false),
     prev_r_hand_quat_(1, 0, 0, 0), has_prev_r_hand_quat_(false),

     prev_r_leg_upper_quat_(1, 0, 0, 0), has_prev_r_leg_upper_quat_(false),
     prev_r_leg_lower_quat_(1, 0, 0, 0), has_prev_r_leg_lower_quat_(false),
     prev_r_foot_quat_(1, 0, 0, 0), has_prev_r_foot_quat_(false),
     prev_l_leg_upper_quat_(1, 0, 0, 0), has_prev_l_leg_upper_quat_(false),
     prev_l_leg_lower_quat_(1, 0, 0, 0), has_prev_l_leg_lower_quat_(false),
     prev_l_foot_quat_(1, 0, 0, 0), has_prev_l_foot_quat_(false),

     z_scale(4.0),
     z_scale_mesh(0.7),

     right_hand_state_(false),
     left_hand_state_(false),

     prev_l_hand_rot_(M_PI / 2.0),
     prev_l_hand_rot_g_(M_PI / 2.0),
     prev_r_hand_rot_(M_PI / 2.0),
     prev_r_hand_rot_g_(M_PI / 2.0),

     right_arm_aligned_(false),
     left_arm_aligned_(false),
     right_leg_aligned_(false),
     left_leg_aligned_(false)
     {}
    
// ========================
// Private helper
// ========================
Eigen::Vector3d Kinematics::normalize(const Eigen::Vector3d& v) {
    double norm = v.norm();
    if (norm < 1e-12) return Eigen::Vector3d::Zero();
    return v / norm;
}

// =========================
// Helper function implementation
// =========================
mojo_quaternion::quaternion Kinematics::to_mojo(const Eigen::Quaterniond& q) const {
    // Converts Eigen (w, x, y, z) to mojo_quaternion
    return mojo_quaternion::quaternion(q.w(), q.x(), q.y(), q.z());
}

// Flip quaternion coefficients if dot product is negative (ensures shortest path)
void Kinematics::flip_if_needed(const Eigen::Quaterniond& prev, Eigen::Quaterniond& current) {
    if (prev.coeffs().dot(current.coeffs()) < 0.0) {
        current.coeffs() *= -1.0;
    }
}

// Apply SLERP with automatic flipping for shortest path
Eigen::Quaterniond Kinematics::apply_slerp(
    const Eigen::Quaterniond& prev,
    const Eigen::Quaterniond& current,
    double alpha
) {
    Eigen::Quaterniond result = current;
    flip_if_needed(prev, result);
    return prev.slerp(alpha, result);
}

// Reset all tracking state
void Kinematics::reset() {
    has_prev_torso_quat_ = false;
    has_prev_hip_quat_ = false;
    has_prev_head_quat_ = false;
    has_prev_l1_quat_ = false;
    has_prev_l2_quat_ = false;
    has_prev_r1_quat_ = false;
    has_prev_r2_quat_ = false;
    has_prev_l2_quat_g_ = false;
    has_prev_r2_quat_g_ = false;
    has_prev_l_hand_quat_ = false;
    has_prev_r_hand_quat_ = false;
    has_prev_r_leg_upper_quat_ = false;
    has_prev_r_leg_lower_quat_ = false;
    has_prev_r_foot_quat_ = false;
    has_prev_l_leg_upper_quat_ = false;
    has_prev_l_leg_lower_quat_ = false;
    has_prev_l_foot_quat_ = false;
    
    prev_torso_quat_ = Eigen::Quaterniond::Identity();
    prev_hip_quat_ = Eigen::Quaterniond::Identity();
    prev_head_quat_ = Eigen::Quaterniond::Identity();
    prev_l1_quat_ = Eigen::Quaterniond::Identity();
    prev_l2_quat_ = Eigen::Quaterniond::Identity();
    prev_r1_quat_ = Eigen::Quaterniond::Identity();
    prev_r2_quat_ = Eigen::Quaterniond::Identity();
    prev_l2_quat_g_ = Eigen::Quaterniond::Identity();
    prev_r2_quat_g_ = Eigen::Quaterniond::Identity();
    prev_l_hand_quat_ = Eigen::Quaterniond::Identity();
    prev_r_hand_quat_ = Eigen::Quaterniond::Identity();
    prev_r_leg_upper_quat_ = Eigen::Quaterniond::Identity();
    prev_r_leg_lower_quat_ = Eigen::Quaterniond::Identity();
    prev_r_foot_quat_ = Eigen::Quaterniond::Identity();
    prev_l_leg_upper_quat_ = Eigen::Quaterniond::Identity();
    prev_l_leg_lower_quat_ = Eigen::Quaterniond::Identity();
    prev_l_foot_quat_ = Eigen::Quaterniond::Identity();
    
    z_scale = 4.0;
    z_scale_mesh = 0.7;
    
    left_hand_state_ = false;
    right_hand_state_ = false;
    left_arm_aligned_ = false;
    right_arm_aligned_ = false;
    left_leg_aligned_ = false;
    right_leg_aligned_ = false;
    
    prev_l_hand_rot_ = M_PI / 2.0;
    prev_l_hand_rot_g_ = M_PI / 2.0;
    prev_r_hand_rot_ = M_PI / 2.0;
    prev_r_hand_rot_g_ = M_PI / 2.0;
}


//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Torso orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
Eigen::Quaterniond Kinematics::torso_orientation(
    const PoseData& pose,
    double alpha
)
{
    Eigen::Quaterniond quat(1, 0, 0, 0); // identity

    if (pose.hasMinimumBody()) {
        Eigen::Vector3d l_shoulder = pose[PoseLandmark::LeftShoulder];
        Eigen::Vector3d r_shoulder = pose[PoseLandmark::RightShoulder];
        Eigen::Vector3d l_hip      = pose[PoseLandmark::LeftHip];
        Eigen::Vector3d r_hip      = pose[PoseLandmark::RightHip];

        l_shoulder.z() *= Z_SCALE_SHOULDER;
        r_shoulder.z() *= Z_SCALE_SHOULDER;  
        l_hip.z()      *= Z_SCALE_SHOULDER;
        r_hip.z()      *= Z_SCALE_SHOULDER;

        Eigen::Vector3d y_axis = normalize((l_hip + r_hip)/2 - (l_shoulder + r_shoulder)/2);
        Eigen::Vector3d x_axis = normalize(l_shoulder - r_shoulder);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(y_axis));
        y_axis = normalize(z_axis.cross(x_axis));

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        quat = Eigen::Quaterniond(rot_matrix);
    }
    
    else if (pose.has(PoseLandmark::LeftShoulder) && pose.has(PoseLandmark::RightShoulder)) {
        // Shoulder and chest center
        Eigen::Vector3d l_shoulder = pose[PoseLandmark::LeftShoulder];
        Eigen::Vector3d r_shoulder = pose[PoseLandmark::RightShoulder];
        l_shoulder.z() *= Z_SCALE_SHOULDER;
        r_shoulder.z() *= Z_SCALE_SHOULDER;
        
        // Temporary up direction
        Eigen::Vector3d temp_up(0, 1, 0);  // could also use hips if available
        Eigen::Vector3d x_axis = normalize((l_shoulder - r_shoulder));
        Eigen::Vector3d y_axis = normalize(temp_up);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(y_axis));
        y_axis = normalize(z_axis.cross(x_axis));

        // Build rotation matrix
        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        quat = Eigen::Quaterniond(rot_matrix);
    }

    // Smooth transitions
    if (has_prev_torso_quat_) {
        quat = apply_slerp(prev_torso_quat_, quat, alpha);
    } else {
        has_prev_torso_quat_ = true;
    }

    prev_torso_quat_ = quat;
    return quat;
}

//Hip orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
Eigen::Quaterniond Kinematics::hip_orientation(
    const PoseData& pose,
    double alpha
)
{
    Eigen::Quaterniond quat(1, 0, 0, 0); // identity

    if (pose.hasMinimumBody()) {
        Eigen::Vector3d l_shoulder = pose[PoseLandmark::LeftShoulder];
        Eigen::Vector3d r_shoulder = pose[PoseLandmark::RightShoulder];
        Eigen::Vector3d l_hip      = pose[PoseLandmark::LeftHip];
        Eigen::Vector3d r_hip      = pose[PoseLandmark::RightHip];

        l_shoulder.z() *= Z_SCALE_SHOULDER;
        r_shoulder.z() *= Z_SCALE_SHOULDER;  
        l_hip.z()      *= Z_SCALE_SHOULDER;
        r_hip.z()      *= Z_SCALE_SHOULDER;

        Eigen::Vector3d y_axis = normalize((l_hip + r_hip)/2 - (l_shoulder + r_shoulder)/2);
        Eigen::Vector3d x_axis = normalize(l_hip - r_hip);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(y_axis));
        y_axis = normalize(z_axis.cross(x_axis));

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        quat = Eigen::Quaterniond(rot_matrix);
    }
    
    // Smooth transitions
    if (has_prev_hip_quat_) {
        quat = apply_slerp(prev_hip_quat_, quat, alpha);
    } else {
        has_prev_hip_quat_ = true;
    }

    prev_hip_quat_ = quat;
    return quat;
}



//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Head orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
Eigen::Quaterniond Kinematics::head_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond quat_rel(1, 0, 0, 0); // identity quaternion

    if (pose.hasFaceMesh()) {
        Eigen::Vector3d nose = pose[PoseLandmark::MeshNoseTip];
        Eigen::Vector3d l_trag = pose[PoseLandmark::MeshLeftEarTragus];
        Eigen::Vector3d r_trag = pose[PoseLandmark::MeshRightEarTragus];  

        Eigen::Vector3d head_center = (l_trag + r_trag) / 2.0;

        // Local head coordinate frame
        Eigen::Vector3d z_axis = normalize(head_center - nose);   // forward
        Eigen::Vector3d x_axis = normalize(r_trag - l_trag);      // horizontal
        Eigen::Vector3d y_axis = normalize(z_axis.cross(x_axis)); // vertical
        x_axis = normalize(y_axis.cross(z_axis));                 // re-orthogonalize

            // Build rotation matrix
        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        Eigen::Quaterniond quat(rot_matrix);

        // Compute relative rotation w.r.t torso
        quat_rel = torso_quat.inverse()*quat;
    }
    else if (pose.hasBasicHead()){
            // Get points
        Eigen::Vector3d nose = pose[PoseLandmark::Nose];
        Eigen::Vector3d l_trag = pose[PoseLandmark::LeftEar];
        Eigen::Vector3d r_trag = pose[PoseLandmark::RightEar];

        nose.z()   *= Z_SCALE_HEAD;
        l_trag.z() *= Z_SCALE_HEAD;
        r_trag.z() *= Z_SCALE_HEAD;

        Eigen::Vector3d head_center = (l_trag + r_trag) / 2.0;

        // Local head coordinate frame
        Eigen::Vector3d z_axis = normalize(head_center - nose);   // forward
        Eigen::Vector3d x_axis = normalize(l_trag - r_trag);      // horizontal
        Eigen::Vector3d y_axis = normalize(z_axis.cross(x_axis)); // vertical
        x_axis = normalize(y_axis.cross(z_axis));                 // re-orthogonalize

            // Build rotation matrix
        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        Eigen::Quaterniond quat(rot_matrix);

        // Compute relative rotation w.r.t torso
        quat_rel = torso_quat.inverse()*quat;
    }

    // Apply SLERP smoothing
    if (has_prev_head_quat_) {
        quat_rel = apply_slerp(prev_head_quat_, quat_rel, alpha);
    } else {
        has_prev_head_quat_ = true;
    }

    prev_head_quat_ = quat_rel;
    return quat_rel;
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Hand Orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::pair<std::optional<Eigen::Vector3d>, Eigen::Quaterniond> Kinematics::hand_normal_vector(
    const PoseData& pose,
    bool is_left
) {
    PoseLandmark palm, index, middle, pinky;
    
    if (is_left) {
        palm = PoseLandmark::LeftPalmBase;
        index = PoseLandmark::LeftIndexFingerBase;
        middle = PoseLandmark::LeftMiddleFingerBase;
        pinky = PoseLandmark::LeftPinkyFingerBase;
    } else {
        palm = PoseLandmark::RightPalmBase;
        index = PoseLandmark::RightIndexFingerBase;
        middle = PoseLandmark::RightMiddleFingerBase;
        pinky = PoseLandmark::RightPinkyFingerBase;
    }

    if (!pose.has(palm) || !pose.has(index) || !pose.has(middle) || !pose.has(pinky)) {
        return {std::nullopt, Eigen::Quaterniond::Identity()};
    }

    const Eigen::Vector3d& vec1_start = pose[index];
    const Eigen::Vector3d& vec1_end   = pose[pinky];
    const Eigen::Vector3d& vec2_start = pose[palm];
    const Eigen::Vector3d& vec2_end   = pose[middle];

    // Compute unit vectors following Python implementation
    Eigen::Vector3d dir1 = normalize(vec1_end - vec1_start);
    Eigen::Vector3d dir2 = normalize(vec2_end - vec2_start);
    Eigen::Vector3d dir3 = normalize(dir2.cross(dir1));  // normal vector
    dir2 = normalize(dir1.cross(dir3));                  // re-orthogonalize dir2
    
    // Create rotation matrix [dir3, dir2, dir1]
    Eigen::Matrix3d rot_matrix;
    rot_matrix.col(0) = dir3;
    rot_matrix.col(1) = dir2;
    rot_matrix.col(2) = dir1;
    
    Eigen::Quaterniond hand_quat(rot_matrix);
    hand_quat.normalize();

    return {dir3, hand_quat};
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Left arm orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################

std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::left_arm_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond l_arm_upper_quat(1, 0, 0, 0);
    Eigen::Quaterniond l_arm_lower_quat(1, 0, 0, 0);
    Eigen::Quaterniond l_arm_lower_quat_g(1, 0, 0, 0);
    Eigen::Quaterniond l_hand_quat(1, 0, 0, 0);

    Eigen::Matrix3d torso_matrix = torso_quat.toRotationMatrix();
    Eigen::Vector3d torso_z = torso_matrix.col(2);

    Eigen::Quaterniond upper_quat, lower_quat;

    // Upper arm (shoulder to elbow)
    if (pose.has(PoseLandmark::LeftShoulder) && pose.has(PoseLandmark::LeftElbow)) {
        Eigen::Vector3d l_shoulder = pose[PoseLandmark::LeftShoulder];
        Eigen::Vector3d l_elbow = pose[PoseLandmark::LeftElbow];

        Eigen::Vector3d arm_vec = normalize(l_elbow - l_shoulder);
        Eigen::Vector3d l1_x_axis = normalize(arm_vec.cross(torso_z));
        Eigen::Vector3d l1_z_axis = normalize(l1_x_axis.cross(arm_vec));
        Eigen::Vector3d l1_y_axis = arm_vec;

        Eigen::Matrix3d l1_rot_matrix;
        l1_rot_matrix.col(0) = l1_x_axis;
        l1_rot_matrix.col(1) = l1_y_axis;
        l1_rot_matrix.col(2) = l1_z_axis;

        upper_quat = Eigen::Quaterniond(l1_rot_matrix);
        upper_quat.normalize();
        l_arm_upper_quat = torso_quat.inverse()*upper_quat;
    }

    // Lower arm (elbow to wrist)
    if (pose.hasLeftArm()) {

        // --- HAND OVERRIDE: use LeftPalmBasePixels if available ---
        Eigen::Vector3d l_wrist;
        if (pose.has(PoseLandmark::LeftPalmBase)) {
            const Eigen::Vector3d& left_palm = pose[PoseLandmark::LeftPalmBase];
            Eigen::Vector3d raw_wrist = pose[PoseLandmark::LeftWrist];
            Eigen::Vector3d palm_wrist(
                left_palm.x(),
                left_palm.y(),
                raw_wrist.z()
            );

            // Only compare in image space (x,y)
            double jump = (palm_wrist.head<2>() - raw_wrist.head<2>()).norm();

            if (jump < MAX_PALM_JUMP_PIXELS) {
                // Smooth blend → prevents snapping
                l_wrist = palm_wrist;
            } else {
                // Palm base unreliable this frame
                l_wrist = raw_wrist;
            }
        } else {
            l_wrist = pose[PoseLandmark::LeftWrist];
        }

        Eigen::Vector3d l_elbow = pose[PoseLandmark::LeftElbow];
        Eigen::Vector3d forearm_vec = normalize(l_wrist - l_elbow);

        // recompute upper arm 
        Eigen::Vector3d l_shoulder = pose[PoseLandmark::LeftShoulder];
        Eigen::Vector3d arm_vec = normalize(l_elbow - l_shoulder);
        
        Eigen::Vector3d l1_x_axis = normalize(forearm_vec.cross(arm_vec));
        Eigen::Vector3d l1_z_axis = normalize(l1_x_axis.cross(arm_vec));
        Eigen::Vector3d l1_y_axis = arm_vec;

         // ---- SINGULARITY HANDLING if previous upper arm exists ----
        if (has_prev_l1_quat_) {
            Eigen::Quaterniond prev_upper_global = torso_quat * (prev_l1_quat_);
            Eigen::Vector3d prev_l1_x_axis = prev_upper_global.toRotationMatrix().col(0);

            double elbow_align = arm_vec.dot(forearm_vec);
            if (std::abs(elbow_align) > ARM_ALIGNMENT_THRESHOLD) { // nearly straight arm
                l1_x_axis = prev_l1_x_axis;
                l1_z_axis = normalize(l1_x_axis.cross(arm_vec));
                l1_y_axis = arm_vec;
                left_arm_aligned_ = true;
            }
            else {
                left_arm_aligned_ = false;
            }
        }

        Eigen::Matrix3d l1_rot_matrix;
        l1_rot_matrix.col(0) = l1_x_axis;
        l1_rot_matrix.col(1) = l1_y_axis;
        l1_rot_matrix.col(2) = l1_z_axis;

        upper_quat = Eigen::Quaterniond(l1_rot_matrix);
        upper_quat.normalize();
        l_arm_upper_quat = torso_quat.inverse()*upper_quat;
        
        auto [left_hand_normal_vector, left_hand_quat] = hand_normal_vector(pose, true);

        Eigen::Vector3d l2_x_axis;
        if (left_hand_normal_vector) {
            l2_x_axis = normalize((*left_hand_normal_vector)); // Use positive normal vector
            left_hand_state_ = true;
        }
        else if (l1_x_axis.norm() > 0.1) {
            l2_x_axis = l1_x_axis;
            left_hand_state_ = false;
        }
        else {
            l2_x_axis = Eigen::Vector3d(1.0, 0.0, 0.0);
            left_hand_state_ = false;
        }

        Eigen::Vector3d l2_z_axis = normalize(l2_x_axis.cross(forearm_vec));
        Eigen::Vector3d l2_y_axis = normalize(forearm_vec);
        l2_x_axis = normalize((l2_y_axis).cross(l2_z_axis));

        Eigen::Matrix3d l2_rot_matrix;
        l2_rot_matrix.col(0) = l2_x_axis;
        l2_rot_matrix.col(1) = l2_y_axis;
        l2_rot_matrix.col(2) = l2_z_axis;

        lower_quat = Eigen::Quaterniond(l2_rot_matrix);
        lower_quat.normalize();
        l_arm_lower_quat = upper_quat.inverse()*lower_quat;
        l_arm_lower_quat_g = torso_quat.inverse()*lower_quat;
        
        // Calculate hand quaternion relative to lower arm
        l_hand_quat = lower_quat.inverse() * left_hand_quat;
    }

    auto torso_q = to_mojo(torso_quat);
    auto torso_e = torso_q.to_euler(mojo_math::EULER_ALGORITHM::YXZ);
    if (torso_e.y < (-TORSO_ANGLE_THRESHOLD_DEG * M_PI / 180.0)) {
        l_arm_upper_quat        = Eigen::Quaterniond::Identity();
        l_arm_lower_quat        = Eigen::Quaterniond::Identity();
        l_hand_quat            = Eigen::Quaterniond::Identity();
    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    if (has_prev_l1_quat_) {
        l_arm_upper_quat = apply_slerp(prev_l1_quat_, l_arm_upper_quat, alpha);
    } else {
        has_prev_l1_quat_ = true;
    }

    prev_l1_quat_ = l_arm_upper_quat;

    if (has_prev_l2_quat_) {
        l_arm_lower_quat = apply_slerp(prev_l2_quat_, l_arm_lower_quat, alpha);
    } else {
        has_prev_l2_quat_ = true;
    }

    prev_l2_quat_ = l_arm_lower_quat;

    //r1_quat global
    if (has_prev_l2_quat_g_) {
        l_arm_lower_quat_g = apply_slerp(prev_l2_quat_g_, l_arm_lower_quat_g, alpha);
    } else {
        has_prev_l2_quat_g_ = true;
    }
    prev_l2_quat_g_ = l_arm_lower_quat_g;

    // Apply SLERP smoothing to hand quaternion
    if (has_prev_l_hand_quat_) {
        l_hand_quat = apply_slerp(prev_l_hand_quat_, l_hand_quat, alpha);
    } else {
        has_prev_l_hand_quat_ = true;
    }
    prev_l_hand_quat_ = l_hand_quat;

    return {l_arm_upper_quat, l_arm_lower_quat, l_arm_lower_quat_g, l_hand_quat};
}


//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Right arm orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::right_arm_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond r_arm_upper_quat(1, 0, 0, 0);
    Eigen::Quaterniond r_arm_lower_quat(1, 0, 0, 0);
    Eigen::Quaterniond r_arm_lower_quat_g(1, 0, 0, 0);
    Eigen::Quaterniond r_hand_quat(1, 0, 0, 0);

    Eigen::Matrix3d torso_matrix = torso_quat.toRotationMatrix();
    Eigen::Vector3d torso_z = torso_matrix.col(2);

    Eigen::Quaterniond upper_quat, lower_quat;

    // ========================
    // Upper arm (shoulder → elbow)
    // ========================
    if (pose.has(PoseLandmark::RightShoulder) && pose.has(PoseLandmark::RightElbow)) {
        Eigen::Vector3d r_shoulder = pose[PoseLandmark::RightShoulder];
        Eigen::Vector3d r_elbow    = pose[PoseLandmark::RightElbow];

        Eigen::Vector3d arm_vec = normalize(r_elbow - r_shoulder);
        Eigen::Vector3d r1_x_axis = normalize(arm_vec.cross(torso_z));
        Eigen::Vector3d r1_z_axis = normalize(r1_x_axis.cross(arm_vec));
        Eigen::Vector3d r1_y_axis = arm_vec;

        Eigen::Matrix3d r1_rot_matrix;
        r1_rot_matrix.col(0) = r1_x_axis;
        r1_rot_matrix.col(1) = r1_y_axis;
        r1_rot_matrix.col(2) = r1_z_axis;

        upper_quat = Eigen::Quaterniond(r1_rot_matrix);
        upper_quat.normalize();
        r_arm_upper_quat = torso_quat.inverse() * upper_quat;
    }

    // ========================
    // Lower arm (elbow → wrist)
    // ========================
    if (pose.hasRightArm()) {

        // --- HAND OVERRIDE: use RightPalmBasePixels if available ---
        Eigen::Vector3d r_wrist;
        if (pose.has(PoseLandmark::RightPalmBase)) {
            const Eigen::Vector3d& right_palm = pose[PoseLandmark::RightPalmBase];
            Eigen::Vector3d raw_wrist = pose[PoseLandmark::RightWrist];
            Eigen::Vector3d palm_wrist(
                right_palm.x(),
                right_palm.y(),
                raw_wrist.z()
            );

            // Only compare in image space (x,y)
            double jump = (palm_wrist.head<2>() - raw_wrist.head<2>()).norm();

            if (jump < MAX_PALM_JUMP_PIXELS) {
                // Smooth blend → prevents snapping
                r_wrist = palm_wrist;
            } else {
                // Palm base unreliable this frame
                r_wrist = raw_wrist;
            }
        } else {
            r_wrist = pose[PoseLandmark::RightWrist];
        }

        Eigen::Vector3d r_elbow = pose[PoseLandmark::RightElbow];
        Eigen::Vector3d forearm_vec = normalize(r_wrist - r_elbow);

        // recompute upper arm
        Eigen::Vector3d r_shoulder = pose[PoseLandmark::RightShoulder];
        Eigen::Vector3d arm_vec = normalize(r_elbow - r_shoulder);

        Eigen::Vector3d r1_x_axis = normalize(forearm_vec.cross(arm_vec));
        Eigen::Vector3d r1_z_axis = normalize(r1_x_axis.cross(arm_vec));
        Eigen::Vector3d r1_y_axis = arm_vec;

        // ---- SINGULARITY HANDLING (straight arm) ----
        if (has_prev_r1_quat_) {
            Eigen::Quaterniond prev_upper_global = torso_quat * prev_r1_quat_;
            Eigen::Vector3d prev_r1_x_axis = prev_upper_global.toRotationMatrix().col(0);

            double elbow_align = arm_vec.dot(forearm_vec);
            if (std::abs(elbow_align) > ARM_ALIGNMENT_THRESHOLD) {
                r1_x_axis = prev_r1_x_axis;
                r1_z_axis = normalize(r1_x_axis.cross(arm_vec));
                r1_y_axis = arm_vec;
                right_arm_aligned_ = true;
            }
            else {
                right_arm_aligned_ = false;
            }
        }

        Eigen::Matrix3d r1_rot_matrix;
        r1_rot_matrix.col(0) = r1_x_axis;
        r1_rot_matrix.col(1) = r1_y_axis;
        r1_rot_matrix.col(2) = r1_z_axis;

        upper_quat = Eigen::Quaterniond(r1_rot_matrix);
        upper_quat.normalize();
        r_arm_upper_quat = torso_quat.inverse() * upper_quat;

        // ========================
        // Forearm / wrist orientation
        // ========================
        auto [right_hand_normal_vector, right_hand_quat] = hand_normal_vector(pose, false); //is_left = false for right hand

        Eigen::Vector3d r2_x_axis;
        if (right_hand_normal_vector) {
            r2_x_axis = normalize((*right_hand_normal_vector)); // Use positive normal vector
            right_hand_state_ = true;
            //std::cout << "r2_x_axis: hand normal\n";
        }
        else if (r1_x_axis.norm() > 0.1) {
            r2_x_axis = r1_x_axis;//prev_r_handvec;
            right_hand_state_ = false;
            //std::cout << "r2_x_axis: prev fallback\n";
        }
        else {
            Eigen::Vector3d torso_x = torso_matrix.col(0);  // Use torso X axis like Python
            r2_x_axis = torso_x;
            right_hand_state_ = false;
            //std::cout << "r2_x_axis: torso_x default\n";
        }

        Eigen::Vector3d r2_z_axis = normalize(r2_x_axis.cross(forearm_vec));
        Eigen::Vector3d r2_y_axis = normalize(forearm_vec);
        r2_x_axis = normalize(r2_y_axis.cross(r2_z_axis));

        Eigen::Matrix3d r2_rot_matrix;
        r2_rot_matrix.col(0) = r2_x_axis;
        r2_rot_matrix.col(1) = r2_y_axis;
        r2_rot_matrix.col(2) = r2_z_axis;

        lower_quat = Eigen::Quaterniond(r2_rot_matrix);
        lower_quat.normalize();
        r_arm_lower_quat = upper_quat.inverse() * lower_quat;
        r_arm_lower_quat_g = torso_quat.inverse() * lower_quat;
        
        // Calculate hand quaternion relative to lower arm
        r_hand_quat = lower_quat.inverse() * right_hand_quat;
    }


    auto torso_q = to_mojo(torso_quat);
    auto torso_e = torso_q.to_euler(mojo_math::EULER_ALGORITHM::YXZ);
    if (torso_e.y > (TORSO_ANGLE_THRESHOLD_DEG * M_PI / 180.0)) {  // Use .y for yaw (Y axis rotation)
        r_arm_upper_quat        = Eigen::Quaterniond::Identity();
        r_arm_lower_quat        = Eigen::Quaterniond::Identity();
        r_hand_quat             = Eigen::Quaterniond::Identity();
    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    //r1_quat
    if (has_prev_r1_quat_) {
        r_arm_upper_quat = apply_slerp(prev_r1_quat_, r_arm_upper_quat, alpha);
    } else {
        has_prev_r1_quat_ = true;
    }
    prev_r1_quat_ = r_arm_upper_quat;

    //r2_quat
    if (has_prev_r2_quat_) {
        r_arm_lower_quat = apply_slerp(prev_r2_quat_, r_arm_lower_quat, alpha);
    } else {
        has_prev_r2_quat_ = true;
    }
    prev_r2_quat_ = r_arm_lower_quat;

    //r1_quat global
    if (has_prev_r2_quat_g_) {
        r_arm_lower_quat_g = apply_slerp(prev_r2_quat_g_, r_arm_lower_quat_g, alpha);
    } else {
        has_prev_r2_quat_g_ = true;
    }
    prev_r2_quat_g_ = r_arm_lower_quat_g;

    // Apply SLERP smoothing to hand quaternion
    if (has_prev_r_hand_quat_) {
        r_hand_quat = apply_slerp(prev_r_hand_quat_, r_hand_quat, alpha);
    } else {
        has_prev_r_hand_quat_ = true;
    }
    prev_r_hand_quat_ = r_hand_quat;

    return {r_arm_upper_quat, r_arm_lower_quat, r_arm_lower_quat_g, r_hand_quat};
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Right leg orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::right_leg_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& hip_quat,
    double alpha
) {
    Eigen::Quaterniond r_leg_upper_quat(1, 0, 0, 0);
    Eigen::Quaterniond r_leg_lower_quat(1, 0, 0, 0);
    Eigen::Quaterniond r_foot_quat(1, 0, 0, 0);

    Eigen::Matrix3d hip_matrix = hip_quat.toRotationMatrix();
    Eigen::Vector3d hip_z = hip_matrix.col(2);

    Eigen::Quaterniond upper_quat, lower_quat;

    // ========================
    // Upper leg (hip → knee)
    // ========================
    if (pose.has(PoseLandmark::RightHip) && pose.has(PoseLandmark::RightKnee)) {
        Eigen::Vector3d r_hip = pose[PoseLandmark::RightHip];
        Eigen::Vector3d r_knee = pose[PoseLandmark::RightKnee];

        Eigen::Vector3d thigh_vec = normalize(r_knee - r_hip);

        Eigen::Vector3d r1_x_axis = normalize(thigh_vec.cross(hip_z));
        Eigen::Vector3d r1_z_axis = normalize(r1_x_axis.cross(thigh_vec));
        Eigen::Vector3d r1_y_axis = thigh_vec;

        Eigen::Matrix3d r1_rot_matrix;
        r1_rot_matrix.col(0) = r1_x_axis;
        r1_rot_matrix.col(1) = r1_y_axis;
        r1_rot_matrix.col(2) = r1_z_axis;

        upper_quat = Eigen::Quaterniond(r1_rot_matrix);
        upper_quat.normalize();
        r_leg_upper_quat = hip_quat.inverse() * upper_quat;
    }

    // ========================
    // Lower leg (knee → ankle)
    // ========================
    if (pose.has(PoseLandmark::RightKnee) && pose.has(PoseLandmark::RightAnkle)) {
        Eigen::Vector3d r_knee = pose[PoseLandmark::RightKnee];
        Eigen::Vector3d r_ankle = pose[PoseLandmark::RightAnkle];

        Eigen::Vector3d shin_vec = normalize(r_ankle - r_knee);

        // Recompute upper using shin direction to stabilize knee roll
        if (pose.has(PoseLandmark::RightHip)) {
            Eigen::Vector3d r_hip = pose[PoseLandmark::RightHip];
            Eigen::Vector3d thigh_vec = normalize(r_knee - r_hip);

            Eigen::Vector3d r1_x_axis = normalize(thigh_vec.cross(shin_vec));
            Eigen::Vector3d r1_z_axis = normalize(r1_x_axis.cross(thigh_vec));
            Eigen::Vector3d r1_y_axis = thigh_vec;

            // ---- SINGULARITY PROTECTION (knee straight) ----
            if (has_prev_r_leg_upper_quat_) {
                Eigen::Quaterniond prev_upper_global = hip_quat * prev_r_leg_upper_quat_;
                Eigen::Vector3d prev_r1_x_axis = prev_upper_global.toRotationMatrix().col(0);

                double knee_align = thigh_vec.dot(shin_vec);

                if (std::abs(knee_align) > LEG_ALIGNMENT_THRESHOLD) {
                    r1_x_axis = prev_r1_x_axis;
                    r1_z_axis = normalize(r1_x_axis.cross(thigh_vec));
                    r1_y_axis = thigh_vec;
                    right_leg_aligned_ = true;
                } else {
                    right_leg_aligned_ = false;
                }
            }

            Eigen::Matrix3d r1_rot_matrix;
            r1_rot_matrix.col(0) = r1_x_axis;
            r1_rot_matrix.col(1) = r1_y_axis;
            r1_rot_matrix.col(2) = r1_z_axis;

            upper_quat = Eigen::Quaterniond(r1_rot_matrix);
            upper_quat.normalize();
            r_leg_upper_quat = hip_quat.inverse() * upper_quat;

            // ---- INITIAL LOWER LEG FRAME ----
            Eigen::Vector3d r2_x_axis = r1_x_axis;
            Eigen::Vector3d r2_z_axis = normalize(r2_x_axis.cross(shin_vec));
            Eigen::Vector3d r2_y_axis = shin_vec;

            // ---- FOOT FRAME & LOWER LEG REFINEMENT ----
            if (pose.has(PoseLandmark::RightHeel) && pose.has(PoseLandmark::RightToe)) {
                Eigen::Vector3d r_toe = pose[PoseLandmark::RightToe];
                Eigen::Vector3d r_heel = pose[PoseLandmark::RightHeel];

                // Z-backward system: heel→toe gives negative-Z (backward)
                Eigen::Vector3d foot_vec = normalize(r_heel - r_toe);
                
                // Compute initial foot x-axis (medial/lateral) from shin and foot direction
                Eigen::Vector3d foot_x_axis = normalize(shin_vec.cross(foot_vec));
                
                // Use foot x-axis to refine lower leg frame
                r2_x_axis = foot_x_axis;
                
                // Recompute lower leg frame with foot-informed x-axis
                r2_z_axis = normalize(r2_x_axis.cross(shin_vec));
                r2_y_axis = shin_vec;
                r2_x_axis = normalize(r2_y_axis.cross(r2_z_axis));  // Ensure orthogonality
                
                Eigen::Matrix3d r2_rot_matrix;
                r2_rot_matrix.col(0) = r2_x_axis;
                r2_rot_matrix.col(1) = r2_y_axis;
                r2_rot_matrix.col(2) = r2_z_axis;
                
                lower_quat = Eigen::Quaterniond(r2_rot_matrix);
                lower_quat.normalize();
                
                // Build foot frame with proper orthogonality
                // Z-axis points backward (heel→toe direction in Z-backward system)
                Eigen::Vector3d foot_z_axis = foot_vec;
                Eigen::Vector3d foot_y_axis = normalize(foot_z_axis.cross(foot_x_axis));
                foot_x_axis = normalize(foot_y_axis.cross(foot_z_axis));  // Recompute for orthogonality
                
                Eigen::Matrix3d foot_rot_matrix;
                foot_rot_matrix.col(0) = foot_x_axis;
                foot_rot_matrix.col(1) = foot_y_axis;
                foot_rot_matrix.col(2) = foot_z_axis;
                
                Eigen::Quaterniond foot_quat_abs(foot_rot_matrix);
                foot_quat_abs.normalize();

                r_foot_quat = lower_quat.inverse() * foot_quat_abs;
            } else {
                // No foot data - build lower leg frame without foot refinement
                r2_x_axis = normalize(r2_y_axis.cross(r2_z_axis));
                
                Eigen::Matrix3d r2_rot_matrix;
                r2_rot_matrix.col(0) = r2_x_axis;
                r2_rot_matrix.col(1) = r2_y_axis;
                r2_rot_matrix.col(2) = r2_z_axis;
                
                lower_quat = Eigen::Quaterniond(r2_rot_matrix);
                lower_quat.normalize();
            }

            // ---- LOWER LEG RELATIVE TO UPPER ----
            r_leg_lower_quat = upper_quat.inverse() * lower_quat;
        }
    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    if (has_prev_r_leg_upper_quat_) {
        r_leg_upper_quat = apply_slerp(prev_r_leg_upper_quat_, r_leg_upper_quat, alpha);
    } else {
        has_prev_r_leg_upper_quat_ = true;
    }
    prev_r_leg_upper_quat_ = r_leg_upper_quat;

    if (has_prev_r_leg_lower_quat_) {
        r_leg_lower_quat = apply_slerp(prev_r_leg_lower_quat_, r_leg_lower_quat, alpha);
    } else {
        has_prev_r_leg_lower_quat_ = true;
    }
    prev_r_leg_lower_quat_ = r_leg_lower_quat;

    if (has_prev_r_foot_quat_) {
        r_foot_quat = apply_slerp(prev_r_foot_quat_, r_foot_quat, alpha);
    } else {
        has_prev_r_foot_quat_ = true;
    }
    prev_r_foot_quat_ = r_foot_quat;

    return {r_leg_upper_quat, r_leg_lower_quat, r_foot_quat};
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Left leg orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::left_leg_orientation(
    const PoseData& pose,
    const Eigen::Quaterniond& hip_quat,
    double alpha
) {
    Eigen::Quaterniond l_leg_upper_quat(1, 0, 0, 0);
    Eigen::Quaterniond l_leg_lower_quat(1, 0, 0, 0);
    Eigen::Quaterniond l_foot_quat(1, 0, 0, 0);

    Eigen::Matrix3d hip_matrix = hip_quat.toRotationMatrix();
    Eigen::Vector3d hip_z = hip_matrix.col(2);

    Eigen::Quaterniond upper_quat, lower_quat;

    // ========================
    // Upper leg (hip → knee)
    // ========================
    if (pose.has(PoseLandmark::LeftHip) && pose.has(PoseLandmark::LeftKnee)) {
        Eigen::Vector3d l_hip = pose[PoseLandmark::LeftHip];
        Eigen::Vector3d l_knee = pose[PoseLandmark::LeftKnee];

        Eigen::Vector3d thigh_vec = normalize(l_knee - l_hip);

        Eigen::Vector3d l1_x_axis = normalize(thigh_vec.cross(hip_z));
        Eigen::Vector3d l1_z_axis = normalize(l1_x_axis.cross(thigh_vec));
        Eigen::Vector3d l1_y_axis = thigh_vec;

        Eigen::Matrix3d l1_rot_matrix;
        l1_rot_matrix.col(0) = l1_x_axis;
        l1_rot_matrix.col(1) = l1_y_axis;
        l1_rot_matrix.col(2) = l1_z_axis;

        upper_quat = Eigen::Quaterniond(l1_rot_matrix);
        upper_quat.normalize();
        l_leg_upper_quat = hip_quat.inverse() * upper_quat;
    }

    // ========================
    // Lower leg (knee → ankle)
    // ========================
    if (pose.has(PoseLandmark::LeftKnee) && pose.has(PoseLandmark::LeftAnkle)) {
        Eigen::Vector3d l_knee = pose[PoseLandmark::LeftKnee];
        Eigen::Vector3d l_ankle = pose[PoseLandmark::LeftAnkle];

        Eigen::Vector3d shin_vec = normalize(l_ankle - l_knee);

        // Recompute upper using shin direction to stabilize knee roll
        if (pose.has(PoseLandmark::LeftHip)) {
            Eigen::Vector3d l_hip = pose[PoseLandmark::LeftHip];
            Eigen::Vector3d thigh_vec = normalize(l_knee - l_hip);

            Eigen::Vector3d l1_x_axis = normalize(thigh_vec.cross(shin_vec));
            Eigen::Vector3d l1_z_axis = normalize(l1_x_axis.cross(thigh_vec));
            Eigen::Vector3d l1_y_axis = thigh_vec;

            // ---- SINGULARITY PROTECTION (knee straight) ----
            if (has_prev_l_leg_upper_quat_) {
                Eigen::Quaterniond prev_upper_global = hip_quat * prev_l_leg_upper_quat_;
                Eigen::Vector3d prev_l1_x_axis = prev_upper_global.toRotationMatrix().col(0);

                double knee_align = thigh_vec.dot(shin_vec);

                if (std::abs(knee_align) > LEG_ALIGNMENT_THRESHOLD) {
                    l1_x_axis = prev_l1_x_axis;
                    l1_z_axis = normalize(l1_x_axis.cross(thigh_vec));
                    l1_y_axis = thigh_vec;
                    left_leg_aligned_ = true;
                } else {
                    left_leg_aligned_ = false;
                }
            }

            Eigen::Matrix3d l1_rot_matrix;
            l1_rot_matrix.col(0) = l1_x_axis;
            l1_rot_matrix.col(1) = l1_y_axis;
            l1_rot_matrix.col(2) = l1_z_axis;

            upper_quat = Eigen::Quaterniond(l1_rot_matrix);
            upper_quat.normalize();
            l_leg_upper_quat = hip_quat.inverse() * upper_quat;

            // ---- INITIAL LOWER LEG FRAME ----
            Eigen::Vector3d l2_x_axis = l1_x_axis;
            Eigen::Vector3d l2_z_axis = normalize(l2_x_axis.cross(shin_vec));
            Eigen::Vector3d l2_y_axis = shin_vec;

            // ---- FOOT FRAME & LOWER LEG REFINEMENT ----
            if (pose.has(PoseLandmark::LeftHeel) && pose.has(PoseLandmark::LeftToe)) {
                Eigen::Vector3d l_toe = pose[PoseLandmark::LeftToe];
                Eigen::Vector3d l_heel = pose[PoseLandmark::LeftHeel];

                // Z-backward system: heel→toe gives negative-Z (backward)
                Eigen::Vector3d foot_vec = normalize(l_heel - l_toe);
                
                // Compute initial foot x-axis (medial/lateral) from shin and foot direction
                Eigen::Vector3d foot_x_axis = normalize(shin_vec.cross(foot_vec));
                
                // Use foot x-axis to refine lower leg frame
                l2_x_axis = foot_x_axis;
                
                // Recompute lower leg frame with foot-informed x-axis
                l2_z_axis = normalize(l2_x_axis.cross(shin_vec));
                l2_y_axis = shin_vec;
                l2_x_axis = normalize(l2_y_axis.cross(l2_z_axis));  // Ensure orthogonality
                
                Eigen::Matrix3d l2_rot_matrix;
                l2_rot_matrix.col(0) = l2_x_axis;
                l2_rot_matrix.col(1) = l2_y_axis;
                l2_rot_matrix.col(2) = l2_z_axis;
                
                lower_quat = Eigen::Quaterniond(l2_rot_matrix);
                lower_quat.normalize();
                
                // Build foot frame with proper orthogonality
                // Z-axis points backward (heel→toe direction in Z-backward system)
                Eigen::Vector3d foot_z_axis = foot_vec;
                Eigen::Vector3d foot_y_axis = normalize(foot_z_axis.cross(foot_x_axis));
                foot_x_axis = normalize(foot_y_axis.cross(foot_z_axis));  // Recompute for orthogonality
                
                Eigen::Matrix3d foot_rot_matrix;
                foot_rot_matrix.col(0) = foot_x_axis;
                foot_rot_matrix.col(1) = foot_y_axis;
                foot_rot_matrix.col(2) = foot_z_axis;
                
                Eigen::Quaterniond foot_quat_abs(foot_rot_matrix);
                foot_quat_abs.normalize();

                l_foot_quat = lower_quat.inverse() * foot_quat_abs;
            } else {
                // No foot data - build lower leg frame without foot refinement
                l2_x_axis = normalize(l2_y_axis.cross(l2_z_axis));
                
                Eigen::Matrix3d l2_rot_matrix;
                l2_rot_matrix.col(0) = l2_x_axis;
                l2_rot_matrix.col(1) = l2_y_axis;
                l2_rot_matrix.col(2) = l2_z_axis;
                
                lower_quat = Eigen::Quaterniond(l2_rot_matrix);
                lower_quat.normalize();
            }

            // ---- LOWER LEG RELATIVE TO UPPER ----
            l_leg_lower_quat = upper_quat.inverse() * lower_quat;
        }
    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    if (has_prev_l_leg_upper_quat_) {
        l_leg_upper_quat = apply_slerp(prev_l_leg_upper_quat_, l_leg_upper_quat, alpha);
    } else {
        has_prev_l_leg_upper_quat_ = true;
    }
    prev_l_leg_upper_quat_ = l_leg_upper_quat;

    if (has_prev_l_leg_lower_quat_) {
        l_leg_lower_quat = apply_slerp(prev_l_leg_lower_quat_, l_leg_lower_quat, alpha);
    } else {
        has_prev_l_leg_lower_quat_ = true;
    }
    prev_l_leg_lower_quat_ = l_leg_lower_quat;

    if (has_prev_l_foot_quat_) {
        l_foot_quat = apply_slerp(prev_l_foot_quat_, l_foot_quat, alpha);
    } else {
        has_prev_l_foot_quat_ = true;
    }
    prev_l_foot_quat_ = l_foot_quat;

    return {l_leg_upper_quat, l_leg_lower_quat, l_foot_quat};
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
// calulate Z scaling using 3d face data
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
void Kinematics::update_z_scale(const PoseData& pose) {
        if (pose.hasFaceMesh()) {
            Eigen::Vector3d l_ear = pose[PoseLandmark::MeshLeftEarTragus];
            Eigen::Vector3d r_ear = pose[PoseLandmark::MeshRightEarTragus];
            Eigen::Vector3d head_center = (l_ear + r_ear) / 2.0;
            Eigen::Vector3d nose = pose[PoseLandmark::MeshNoseTip];

            z_scale_mesh = std::abs(nose.z() - (r_ear.z() + l_ear.z()) / 2.0) / std::abs(r_ear.x() - l_ear.x());
            //std::cout << "Z Scale mesh: " << z_scale_mesh << std::endl;
        }


        if (!pose.hasBasicHead() || 
            !pose.has(PoseLandmark::LeftShoulder) || 
            !pose.has(PoseLandmark::RightShoulder)) {
            return; // z_scale already defined
        }

        Eigen::Vector3d nose = pose[PoseLandmark::Nose];
        Eigen::Vector3d l_ear = pose[PoseLandmark::LeftEar];
        Eigen::Vector3d r_ear = pose[PoseLandmark::RightEar];

        Eigen::Vector3d head_center = (l_ear + r_ear) / 2.0;

        // Local head coordinate frame
        Eigen::Vector3d z_axis = normalize(head_center - nose);   // forward
        Eigen::Vector3d x_axis = normalize(l_ear - r_ear);      // horizontal
        Eigen::Vector3d y_axis = normalize(z_axis.cross(x_axis)); // vertical
        x_axis = normalize(y_axis.cross(z_axis));                 // re-orthogonalize

            // Build rotation matrix
        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        Eigen::Quaterniond head_quat(rot_matrix);


        auto to_mojo = [](const Eigen::Quaterniond& q) {
        return mojo_quaternion::quaternion(q.w(), q.x(), q.y(), q.z());
        };

        // --- Head ---
        auto head_q = to_mojo(head_quat);
        auto head_e = head_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);

        // constexpr double RAD2DEG = 180.0 / M_PI;


        // std::cout << "Head Euler (XYZ): "
        //   << "X = " << head_e.x* RAD2DEG << ", "
        //   << "Y = " << head_e.y* RAD2DEG << ", "
        //   << "Z = " << head_e.z* RAD2DEG << std::endl;

        if (std::abs(head_e.x) < HEAD_STRAIGHTNESS_THRESHOLD &&
            std::abs(head_e.y) < HEAD_STRAIGHTNESS_THRESHOLD &&
            std::abs(head_e.z) < HEAD_STRAIGHTNESS_THRESHOLD)
        {
            z_scale = Z_SCALE_MESH_FACTOR*(std::abs(nose.z() - (r_ear.z() + l_ear.z()) / 2.0) /  std::abs(r_ear.x() - l_ear.x()));
            // std::cout << "Internal Z Scale: " << z_scale << std::endl;
        }
    }

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Apply Z normalization to pose data that has capitalised names, hand data is non capitalised
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
PoseData 
Kinematics::normalize_z_data(const PoseData& pose)
{
    PoseData normalized = pose;  // Copy all data

    // Normalize body landmarks (uppercase enum names)
    for (uint8_t i = static_cast<uint8_t>(PoseLandmark::Nose); 
         i <= static_cast<uint8_t>(PoseLandmark::RightAnkle); ++i) {
        PoseLandmark landmark = static_cast<PoseLandmark>(i);
        if (normalized.has(landmark) && z_scale != 0.0) {
            normalized[landmark].z() /= z_scale;
        }
    }

    // Normalize face mesh landmarks
    if (z_scale_mesh != 0.0) {
        if (normalized.has(PoseLandmark::MeshNoseTip)) {
            normalized[PoseLandmark::MeshNoseTip].z() /= z_scale_mesh;
        }
        if (normalized.has(PoseLandmark::MeshLeftEarTragus)) {
            normalized[PoseLandmark::MeshLeftEarTragus].z() /= z_scale_mesh;
        }
        if (normalized.has(PoseLandmark::MeshRightEarTragus)) {
            normalized[PoseLandmark::MeshRightEarTragus].z() /= z_scale_mesh;
        }
    }

    // Hand landmarks remain unchanged (not normalized)
    
    return normalized;
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Process kinematics
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
PoseResults Kinematics::process_kinematics(
        const PoseData& pose_data)
    {
    // Early return with identity pose if no valid timestamp
    if (pose_data.timestamp_ms == 0) {
        PoseResults empty_results;
        empty_results.quaternions.reserve(17);  // torso, hip, head, 4*arms, 2*hands, 6*legs
        empty_results.eulerAngles.reserve(15);  // torso, hip, head, 4*arms, 2*hands, 6*legs
        // Fill with identity quaternions
        for (int i = 0; i < 17; ++i) {
            empty_results.quaternions.push_back(to_mojo(Eigen::Quaterniond::Identity()));
        }
        for (int i = 0; i < 15; ++i) {
            empty_results.eulerAngles.push_back(Eigen::Vector3d::Zero());
        }
        return empty_results;
    }

    update_z_scale(pose_data);
    auto normalized_pose = normalize_z_data(pose_data);

    // Compute orientations
    Eigen::Quaterniond torso_quat = torso_orientation(normalized_pose);
    Eigen::Quaterniond hip_quat   = hip_orientation(normalized_pose);
    Eigen::Quaterniond head_quat  = head_orientation(normalized_pose, torso_quat);

    auto [l_upper_quat, l_lower_quat, l_lower_quat_g, l_hand_quat] = left_arm_orientation(normalized_pose, torso_quat);
    auto [r_upper_quat, r_lower_quat, r_lower_quat_g, r_hand_quat] = right_arm_orientation(normalized_pose, torso_quat);

    // Compute leg orientations
    auto [l_leg_upper_quat, l_leg_lower_quat, l_foot_quat] = left_leg_orientation(normalized_pose, hip_quat);
    auto [r_leg_upper_quat, r_leg_lower_quat, r_foot_quat] = right_leg_orientation(normalized_pose, hip_quat);

    // Return as mojo quaternion vector
    auto kinematic_output = structure_kinematic_output(
        torso_quat, hip_quat, head_quat,
        l_upper_quat, l_lower_quat, l_hand_quat,
        r_upper_quat, r_lower_quat, r_hand_quat,
        r_lower_quat_g, l_lower_quat_g,
        l_leg_upper_quat, l_leg_lower_quat, l_foot_quat,
        r_leg_upper_quat, r_leg_lower_quat, r_foot_quat
    );

    return kinematic_output;
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Structure kinematic output and extract euler angles
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
PoseResults Kinematics::structure_kinematic_output(
    const Eigen::Quaterniond& torso_quat,
    const Eigen::Quaterniond& hip_quat,
    const Eigen::Quaterniond& head_quat,
    const Eigen::Quaterniond& l_upper_quat,
    const Eigen::Quaterniond& l_lower_quat,
    const Eigen::Quaterniond& l_hand_quat,
    const Eigen::Quaterniond& r_upper_quat,
    const Eigen::Quaterniond& r_lower_quat,
    const Eigen::Quaterniond& r_hand_quat,
    const Eigen::Quaterniond& r_lower_quat_g,
    const Eigen::Quaterniond& l_lower_quat_g,
    const Eigen::Quaterniond& l_leg_upper_quat,
    const Eigen::Quaterniond& l_leg_lower_quat,
    const Eigen::Quaterniond& l_foot_quat,
    const Eigen::Quaterniond& r_leg_upper_quat,
    const Eigen::Quaterniond& r_leg_lower_quat,
    const Eigen::Quaterniond& r_foot_quat)
{
    PoseResults results;
    results.quaternions.reserve(17);  // torso, hip, head, 4*arms, 2*hands, 6*legs
    results.eulerAngles.reserve(15);  // torso, hip, head, 4*arms, 2*hands, 6*legs

    // --- Torso ---
    auto torso_q = to_mojo(torso_quat);
    auto torso_e = torso_q.to_euler(mojo_math::EULER_ALGORITHM::YXZ);
    results.quaternions.push_back(torso_q);
    results.eulerAngles.push_back(Eigen::Vector3d(torso_e.x, torso_e.z, torso_e.y));

    // --- Hip ---
    auto hip_q = to_mojo(hip_quat);
    auto hip_e = hip_q.to_euler(mojo_math::EULER_ALGORITHM::YXZ);
    results.quaternions.push_back(hip_q);
    results.eulerAngles.push_back(Eigen::Vector3d(hip_e.x, hip_e.z, hip_e.y));

    // --- Head ---
    auto head_q = to_mojo(head_quat);
    auto head_e = head_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(head_q);
    results.eulerAngles.push_back(Eigen::Vector3d(head_e.x, head_e.z, head_e.y));

    // --- Left arm (upper) ---
    auto l_upper_q = to_mojo(r_upper_quat);
    auto l_upper_e = l_upper_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(l_upper_q);
    results.eulerAngles.push_back(Eigen::Vector3d(-l_upper_e.z, -l_upper_e.x, -l_upper_e.y));

    // --- Left arm (lower) ---
    auto l_lower_q = to_mojo(r_lower_quat);
    auto l_lower_q_g = to_mojo(r_lower_quat_g);
    auto l_lower_e = l_lower_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(l_lower_q);
    results.quaternions.push_back(l_lower_q_g);
    results.eulerAngles.push_back(Eigen::Vector3d(-l_lower_e.z, -l_lower_e.x, -l_lower_e.y));

    // --- Right arm (upper) ---
    auto r_upper_q = to_mojo(l_upper_quat);
    auto r_upper_e = r_upper_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(r_upper_q);
    results.eulerAngles.push_back(Eigen::Vector3d(r_upper_e.z, -r_upper_e.x, r_upper_e.y));

    // --- Right arm (lower) ---
    auto r_lower_q = to_mojo(l_lower_quat);
    auto r_lower_q_g = to_mojo(l_lower_quat_g);
    auto r_lower_e = r_lower_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(r_lower_q);
    results.quaternions.push_back(r_lower_q_g);
    results.eulerAngles.push_back(Eigen::Vector3d(r_lower_e.z, -r_lower_e.x, r_lower_e.y));

    // --- Hand quaternions ---
    auto l_hand_q = to_mojo(r_hand_quat);
    auto r_hand_q = to_mojo(l_hand_quat);
    results.quaternions.push_back(l_hand_q);
    results.quaternions.push_back(r_hand_q);

    // --- Hand Euler angles ---
    auto l_hand_e = l_hand_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    auto r_hand_e = r_hand_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.eulerAngles.push_back(Eigen::Vector3d(-l_hand_e.z, l_hand_e.x, -l_hand_e.y));
    results.eulerAngles.push_back(Eigen::Vector3d(r_hand_e.z, -r_hand_e.x, r_hand_e.y));

    // --- Left leg quaternions ---
    auto l_leg_upper_q = to_mojo(l_leg_upper_quat);
    auto l_leg_lower_q = to_mojo(l_leg_lower_quat);
    auto l_foot_q = to_mojo(l_foot_quat);
    results.quaternions.push_back(l_leg_upper_q);
    results.quaternions.push_back(l_leg_lower_q);
    results.quaternions.push_back(l_foot_q);

    // --- Left leg Euler angles ---
    auto l_leg_upper_e = l_leg_upper_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    auto l_leg_lower_e = l_leg_lower_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    auto l_foot_e = l_foot_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.eulerAngles.push_back(Eigen::Vector3d(-l_leg_upper_e.z, -l_leg_upper_e.x, -l_leg_upper_e.y));
    results.eulerAngles.push_back(Eigen::Vector3d(-l_leg_lower_e.z, -l_leg_lower_e.x, -l_leg_lower_e.y));
    results.eulerAngles.push_back(Eigen::Vector3d(-l_foot_e.z, -l_foot_e.x, -l_foot_e.y));

    // --- Right leg quaternions ---
    auto r_leg_upper_q = to_mojo(r_leg_upper_quat);
    auto r_leg_lower_q = to_mojo(r_leg_lower_quat);
    auto r_foot_q = to_mojo(r_foot_quat);
    results.quaternions.push_back(r_leg_upper_q);
    results.quaternions.push_back(r_leg_lower_q);
    results.quaternions.push_back(r_foot_q);

    // --- Right leg Euler angles ---
    auto r_leg_upper_e = r_leg_upper_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    auto r_leg_lower_e = r_leg_lower_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    auto r_foot_e = r_foot_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.eulerAngles.push_back(Eigen::Vector3d(r_leg_upper_e.z, -r_leg_upper_e.x, r_leg_upper_e.y));
    results.eulerAngles.push_back(Eigen::Vector3d(r_leg_lower_e.z, -r_leg_lower_e.x, r_leg_lower_e.y));
    results.eulerAngles.push_back(Eigen::Vector3d(r_foot_e.z, -r_foot_e.x, r_foot_e.y));

    results.eulerJson = avatar_json(results.eulerAngles);
    results.planeJson = json_isolated_angles(results.quaternions, results.eulerAngles);

    return results;
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Create json for website model
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::map<std::string, double> 
Kinematics::avatar_json(
    std::vector<Eigen::Vector3d> euler_angles) 
    {
    std::map<std::string, double> json_angles;

    // --- Torso ---
    json_angles["theta_torso_pitch_r"] = euler_angles[0].x();
    json_angles["theta_torso_tilt_r"]  = euler_angles[0].y();
    json_angles["theta_torso_yaw_r"]   = euler_angles[0].z();
    json_angles["theta_torso_roll_r"]  = 0.0;
    json_angles["theta_torso_bend_r"]  = 0.0;

    // --- Hip ---
    json_angles["theta_hip_pitch_r"] = euler_angles[1].x();
    json_angles["theta_hip_tilt_r"]  = euler_angles[1].y();
    json_angles["theta_hip_yaw_r"]   = euler_angles[1].z();
    json_angles["theta_hip_roll_r"]  = 0.0;
    json_angles["theta_hip_bend_r"]  = 0.0;

    // --- Head ---
    json_angles["theta_head_pitch_h"] = euler_angles[2].x();
    json_angles["theta_head_roll_h"]  = euler_angles[2].y();
    json_angles["theta_head_yaw_h"]   = euler_angles[2].z();

    // --- Right arm ---
    json_angles["theta_armright_upper_alpha"] = euler_angles[5].x();
    json_angles["theta_armright_upper_beta"]  = euler_angles[5].y();
    json_angles["theta_armright_upper_gamma"] = euler_angles[5].z();

    json_angles["theta_armright_lower_alpha"] = euler_angles[6].x();
    json_angles["theta_armright_lower_beta"]  = euler_angles[6].y();
    json_angles["theta_armright_lower_gamma"] = euler_angles[6].z();

    
    // --- Left arm ---
    json_angles["theta_armleft_upper_alpha"] = euler_angles[3].x();
    json_angles["theta_armleft_upper_beta"]  = euler_angles[3].y();
    json_angles["theta_armleft_upper_gamma"] = euler_angles[3].z();

    json_angles["theta_armleft_lower_alpha"] = euler_angles[4].x();
    json_angles["theta_armleft_lower_beta"]  = euler_angles[4].y();
    json_angles["theta_armleft_lower_gamma"] = euler_angles[4].z();

    // --- Hand angles ---
    json_angles["theta_handleft_alpha"] = euler_angles[7].x();
    json_angles["theta_handleft_beta"]  = euler_angles[7].y();
    json_angles["theta_handleft_gamma"] = euler_angles[7].z();

    json_angles["theta_handright_alpha"] = euler_angles[8].x();
    json_angles["theta_handright_beta"]  = euler_angles[8].y();
    json_angles["theta_handright_gamma"] = euler_angles[8].z();

    // --- Left leg ---
    json_angles["theta_legleft_upper_alpha"] = euler_angles[9].x();
    json_angles["theta_legleft_upper_beta"]  = euler_angles[9].y();
    json_angles["theta_legleft_upper_gamma"] = euler_angles[9].z();

    json_angles["theta_legleft_lower_alpha"] = euler_angles[10].x();
    json_angles["theta_legleft_lower_beta"]  = euler_angles[10].y();
    json_angles["theta_legleft_lower_gamma"] = euler_angles[10].z();

    json_angles["theta_footleft_alpha"] = euler_angles[11].x();
    json_angles["theta_footleft_beta"]  = euler_angles[11].y();
    json_angles["theta_footleft_gamma"] = euler_angles[11].z();

    // --- Right leg ---
    json_angles["theta_legright_upper_alpha"] = euler_angles[12].x();
    json_angles["theta_legright_upper_beta"]  = euler_angles[12].y();
    json_angles["theta_legright_upper_gamma"] = euler_angles[12].z();

    json_angles["theta_legright_lower_alpha"] = euler_angles[13].x();
    json_angles["theta_legright_lower_beta"]  = euler_angles[13].y();
    json_angles["theta_legright_lower_gamma"] = euler_angles[13].z();

    json_angles["theta_footright_alpha"] = euler_angles[14].x();
    json_angles["theta_footright_beta"]  = euler_angles[14].y();
    json_angles["theta_footright_gamma"] = euler_angles[14].z();

    return json_angles;
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
// Return angles in planes (torso, head, arms)
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::map<std::string, double>
Kinematics::json_isolated_angles(
    std::vector<mojo_quaternion::quaternion>& quaternions,
    std::vector<Eigen::Vector3d>& euler_angles)
{
    std::map<std::string, double> json;

    // -------------------------
    // Torso & Head (from Euler angles)
    // -------------------------
    json["theta_torso_pitch_r"] = euler_angles[0].x();
    json["theta_torso_tilt_r"]  = euler_angles[0].y();
    json["theta_torso_yaw_r"]   = euler_angles[0].z();
    json["theta_torso_roll_r"]  = 0.0;
    json["theta_torso_bend_r"]  = 0.0;

    json["theta_hip_pitch_r"] = euler_angles[1].x();
    json["theta_hip_tilt_r"]  = euler_angles[1].y();
    json["theta_hip_yaw_r"]   = euler_angles[1].z();
    json["theta_hip_roll_r"]  = 0.0;
    json["theta_hip_bend_r"]  = 0.0;

    // -------------------------
    // Helper: convert mojo -> Eigen
    // -------------------------
    auto to_eigen = [](const mojo_quaternion::quaternion& q) {
        Eigen::Quaterniond eq(q.w, q.x, q.y, q.z);
        eq.normalize(); // safe normalization
        return eq;
    };

    // -------------------------
    // quaternions
    // -------------------------
    Eigen::Quaterniond head_q = to_eigen(quaternions[2]);
    Eigen::Quaterniond l_upper = to_eigen(quaternions[3]);
    Eigen::Quaterniond l_lower = to_eigen(quaternions[4]);
    Eigen::Quaterniond l_lower_g = to_eigen(quaternions[5]);
    Eigen::Quaterniond r_upper = to_eigen(quaternions[6]);
    Eigen::Quaterniond r_lower = to_eigen(quaternions[7]);
    Eigen::Quaterniond r_lower_g = to_eigen(quaternions[8]);
    Eigen::Quaterniond l_hand_q = to_eigen(quaternions[9]);
    Eigen::Quaterniond r_hand_q = to_eigen(quaternions[10]);
    Eigen::Quaterniond l_leg_upper_q = to_eigen(quaternions[11]);
    Eigen::Quaterniond l_leg_lower_q = to_eigen(quaternions[12]);
    Eigen::Quaterniond l_foot_q = to_eigen(quaternions[13]);
    Eigen::Quaterniond r_leg_upper_q = to_eigen(quaternions[14]);
    Eigen::Quaterniond r_leg_lower_q = to_eigen(quaternions[15]);
    Eigen::Quaterniond r_foot_q = to_eigen(quaternions[16]);

    // -------------------------
    // Head
    // -------------------------

    // Local head axes
    Eigen::Vector3d head_forward_local(0, 0, 1);
    Eigen::Vector3d head_up_local(0, 1, 0);

    // Rotate to world space
    Eigen::Vector3d head_forward = head_q * head_forward_local;
    Eigen::Vector3d head_up      = head_q * head_up_local;

    // Unit vectors for axes
    Eigen::Vector3d X(1, 0, 0);
    Eigen::Vector3d Y(0, 1, 0);
    Eigen::Vector3d Z(0, 0, 1);

    const double eps = 1e-8;
    // --- YAW: forward projected onto XZ plane
    Eigen::Vector3d head_fwd_xz = head_forward - head_forward.dot(Y) * Y;
    head_fwd_xz.normalize(); // Eigen handles small norm automatically

    double theta_head_yaw = std::atan2(head_fwd_xz.dot(X),
                                       head_fwd_xz.dot(Z));

    // --- PITCH: forward projected onto YZ plane
    Eigen::Vector3d head_fwd_yz = head_forward - head_forward.dot(X) * X;
    head_fwd_yz.normalize();

    double theta_head_pitch = std::atan2(-head_fwd_yz.dot(Y),
                                         head_fwd_yz.dot(Z));

    // --- ROLL: up vector projected onto XY plane
    Eigen::Vector3d head_up_xy = head_up - head_up.dot(Z) * Z;
    head_up_xy.normalize();

    double theta_head_roll = std::atan2(head_up_xy.dot(X),
                                        head_up_xy.dot(Y));

    // Assign to "JSON" map
    json["theta_head_pitch_h"] = theta_head_pitch;
    json["theta_head_yaw_h"]   = theta_head_yaw;
    json["theta_head_roll_h"]  = theta_head_roll * 1.2;

    //////////////////////////////////////////////////////////////////////////////////////////////////////

    Eigen::Vector3d torso_forward = Eigen::Vector3d(0, 0, 1);
    Eigen::Vector3d arm_axis_local(0, 1, 0);   // +Y along the arm
    Eigen::Vector3d arm_forward_local(1, 0, 0); // along forearm for rotatio

    // -------------------------
    // LEFT SHOULDER
    // -------------------------
    Eigen::Vector3d arm_dir_l = l_upper * arm_axis_local;

    // Flexion/Extension
    Eigen::Vector3d arm_sag_l(0, arm_dir_l.y(), arm_dir_l.z());
    arm_sag_l.normalize();
    double theta_flex_l = std::atan2(arm_sag_l.z(), arm_sag_l.y());

    // Abduction/Adduction (sign flipped for left side)
    Eigen::Vector3d arm_cor_l(arm_dir_l.x(), arm_dir_l.y(), 0);
    arm_cor_l.normalize();
    double theta_abduct_l = std::atan2(-arm_cor_l.x(), arm_cor_l.y());

    // Horizontal rotation (around Y)
    Eigen::Vector3d arm_horiz_l(arm_dir_l.x(), 0, arm_dir_l.z());
    arm_horiz_l.normalize();
    double theta_yaw_l = std::atan2(-arm_horiz_l.z(), -arm_horiz_l.x());

    // Internal/External rotation
    Eigen::Vector3d arm_forward_l = l_upper * arm_forward_local;
    Eigen::Vector3d arm_proj_l = arm_forward_l - arm_forward_l.dot(arm_dir_l) * arm_dir_l;
    arm_proj_l.normalize();
    double theta_rot_l = std::atan2(
        torso_forward.cross(arm_proj_l).dot(arm_dir_l),
        torso_forward.dot(arm_proj_l)
    );

    if (right_arm_aligned_) {
        theta_rot_l = NAN;
    } 

    if (std::abs(arm_dir_l.y()) >= ARM_VERTICAL_THRESHOLD) {
        theta_yaw_l = NAN;
    }

    json["theta_armleft_upper_flexion"] = -theta_flex_l;
    json["theta_armleft_upper_abduction"]  = theta_abduct_l;
    json["theta_armleft_upper_rotation"] = theta_rot_l;
    json["theta_armleft_upper_yaw"]   = theta_yaw_l;

    // -------------------------
    // RIGHT SHOULDER
    // -------------------------
    Eigen::Vector3d arm_dir_r = r_upper * arm_axis_local;
    Eigen::Vector3d arm_forward_r_local(-1, 0, 0);

    // Flexion/Extension
    Eigen::Vector3d arm_sag_r(0, arm_dir_r.y(), arm_dir_r.z());
    arm_sag_r.normalize();
    double theta_flex_r = std::atan2(arm_sag_r.z(), arm_sag_r.y());

    // Abduction/Adduction
    Eigen::Vector3d arm_cor_r(arm_dir_r.x(), arm_dir_r.y(), 0);
    arm_cor_r.normalize();
    double theta_abduct_r = std::atan2(arm_cor_r.x(), arm_cor_r.y());

    // Horizontal rotation
    Eigen::Vector3d arm_horiz_r(arm_dir_r.x(), 0, arm_dir_r.z());
    arm_horiz_r.normalize();
    double theta_yaw_r = std::atan2(-arm_horiz_r.z(), arm_horiz_r.x());

    // Internal/External rotation
    Eigen::Vector3d arm_forward_r = r_upper * arm_forward_r_local;
    Eigen::Vector3d arm_proj_r = arm_forward_r - arm_forward_r.dot(arm_dir_r) * arm_dir_r;
    arm_proj_r.normalize();
    double theta_rot_r = std::atan2(
        torso_forward.cross(arm_proj_r).dot(arm_dir_r),
        torso_forward.dot(arm_proj_r)
    );

    if (left_arm_aligned_) {
        theta_rot_r = NAN;
    } 

    if (std::abs(arm_dir_r.y()) >= ARM_VERTICAL_THRESHOLD) {
        theta_yaw_r = NAN;
    }

    json["theta_armright_upper_flexion"] = -theta_flex_r;
    json["theta_armright_upper_abduction"]  = theta_abduct_r;
    json["theta_armright_upper_rotation"] = -theta_rot_r; // match Python
    json["theta_armright_upper_yaw"]   = theta_yaw_r;

    // -------------------------
    // LEFT ELBOW
    // -------------------------
    Eigen::Vector3d lower_dir_l = l_lower * arm_axis_local;
    Eigen::Vector3d upper_axis(0, 1, 0);

    double cos_angle_l = upper_axis.dot(lower_dir_l);
    cos_angle_l = std::clamp(cos_angle_l, -1.0, 1.0);
    double theta_elbow_flex_l = std::acos(cos_angle_l);

    Eigen::Vector3d palm_l = l_lower * Eigen::Vector3d(-1, 0, 0);
    Eigen::Vector3d palm_proj_l = palm_l;  // explicit copy like Python
    palm_proj_l.z() = 0; // project onto torso frontal plane
    palm_proj_l /= (palm_proj_l.norm() + 1e-8);  // explicit normalization with epsilon
    double theta_pronation_l = std::atan2(-palm_proj_l.x(), -palm_proj_l.y());

    Eigen::Vector3d palm_l_g = l_lower_g * Eigen::Vector3d(-1, 0, 0);
    Eigen::Vector3d palm_proj_l_g = palm_l_g;  // explicit copy like Python  
    palm_proj_l_g.z() = 0;
    palm_proj_l_g /= (palm_proj_l_g.norm() + 1e-8);  // explicit normalization with epsilon
    double theta_pronation_l_g = std::atan2(-palm_proj_l_g.x(), -palm_proj_l_g.y());

    json["theta_armleft_lower_flexion"]  = theta_elbow_flex_l;
    if (right_hand_state_) {  // left hand active
        json["theta_armleft_lower_rotation"]   = theta_pronation_l;
        json["theta_armleft_lower_rotation_g"] = theta_pronation_l_g;

        prev_l_hand_rot_   = theta_pronation_l;
        prev_l_hand_rot_g_ = theta_pronation_l_g;
    } else {
        json["theta_armleft_lower_rotation"]   = prev_l_hand_rot_;
        json["theta_armleft_lower_rotation_g"] = prev_l_hand_rot_g_;
    }

    // -------------------------
    // RIGHT ELBOW
    // -------------------------
    Eigen::Vector3d lower_dir_r = r_lower * arm_axis_local;

    double cos_angle_r = upper_axis.dot(lower_dir_r);
    cos_angle_r = std::clamp(cos_angle_r, -1.0, 1.0);
    double theta_elbow_flex_r = std::acos(cos_angle_r);

    Eigen::Vector3d palm_r = r_lower * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d palm_proj_r = palm_r;  // explicit copy like Python
    palm_proj_r.z() = 0;
    palm_proj_r /= (palm_proj_r.norm() + 1e-8);  // explicit normalization with epsilon
    double theta_pronation_r = std::atan2(palm_proj_r.x(), -palm_proj_r.y());

    Eigen::Vector3d palm_r_g = r_lower_g * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d palm_proj_r_g = palm_r_g;  // explicit copy like Python
    palm_proj_r_g.z() = 0;
    palm_proj_r_g /= (palm_proj_r_g.norm() + 1e-8);  // explicit normalization with epsilon  
    double theta_pronation_r_g = std::atan2(-palm_proj_r_g.x(), -palm_proj_r_g.y());

    json["theta_armright_lower_flexion"]  = theta_elbow_flex_r;

    if (left_hand_state_) {
        json["theta_armright_lower_rotation"]   = theta_pronation_r;
        json["theta_armright_lower_rotation_g"] = theta_pronation_r_g;
        prev_r_hand_rot_   = theta_pronation_r;
        prev_r_hand_rot_g_ = theta_pronation_r_g;
    } else {
        json["theta_armright_lower_rotation"]   = prev_r_hand_rot_;
        json["theta_armright_lower_rotation_g"] = prev_r_hand_rot_g_;
    }

    // -------------------------
    // RIGHT HAND
    // -------------------------
    Eigen::Vector3d hand_fwd_r = r_hand_q * Eigen::Vector3d(0, 1, 0);  // Y along fingers
    Eigen::Vector3d hand_side_r = r_hand_q * Eigen::Vector3d(1, 0, 0); // X across hand

    double hand_theta_flex_r = std::atan2(hand_fwd_r.z(), hand_fwd_r.y());    // Z vs Y
    double hand_theta_abduct_r = std::atan2(hand_side_r.y(), hand_side_r.x()); // Y vs X

    json["handright_abduction"] = hand_theta_flex_r;
    json["handright_flexion"] = hand_theta_abduct_r;

    // -------------------------
    // LEFT HAND
    // -------------------------
    Eigen::Vector3d hand_fwd_l = l_hand_q * Eigen::Vector3d(0, 1, 0);  // Y along fingers
    Eigen::Vector3d hand_side_l = l_hand_q * Eigen::Vector3d(1, 0, 0); // X across hand

    double hand_theta_flex_l = std::atan2(hand_fwd_l.z(), hand_fwd_l.y());    // Z vs Y
    double hand_theta_abduct_l = std::atan2(hand_side_l.y(), hand_side_l.x()); // Y vs X

    json["handleft_abduction"] = hand_theta_flex_l;
    json["handleft_flexion"] = hand_theta_abduct_l;

    // ####################################################
    // ######################################################
    // #legs
    // ########################################################
    // ####################################################

    Eigen::Vector3d leg_axis_local(0, 1, 0);   // +Y points down the leg
    Eigen::Vector3d leg_forward_local_L(1, 0, 0);
    Eigen::Vector3d leg_forward_local_R(-1, 0, 0);
    Eigen::Vector3d expected_forward(0, 0, 1);

    // =====================
    // LEFT HIP
    // =====================
    Eigen::Vector3d leg_direction_l = l_leg_upper_q * leg_axis_local;

    // Flexion / Extension
    Eigen::Vector3d leg_sagittal_l(0, leg_direction_l.y(), leg_direction_l.z());
    leg_sagittal_l.normalize();
    double theta_hipflex_l = std::atan2(leg_sagittal_l.z(), leg_sagittal_l.y());

    // Abduction / Adduction
    Eigen::Vector3d leg_coronal_l(leg_direction_l.x(), leg_direction_l.y(), 0);
    leg_coronal_l.normalize();
    double theta_hipabd_l = std::atan2(-leg_coronal_l.x(), leg_coronal_l.y());

    // Horizontal yaw
    Eigen::Vector3d leg_horizontal_l(leg_direction_l.x(), 0, leg_direction_l.z());
    leg_horizontal_l.normalize();
    double theta_hipyaw_l = std::atan2(-leg_horizontal_l.z(), -leg_horizontal_l.x());

    // Internal / External rotation (twist only, local)
    Eigen::Vector3d leg_forward_l = l_leg_upper_q * leg_forward_local_L;
    Eigen::Vector3d leg_forward_proj_l = leg_forward_l - leg_forward_l.dot(leg_direction_l) * leg_direction_l;
    leg_forward_proj_l.normalize();

    double theta_hiprot_l = std::atan2(
        expected_forward.cross(leg_forward_proj_l).dot(leg_direction_l),
        expected_forward.dot(leg_forward_proj_l)
    );

    json["l_upper_leg_flexion"] = -theta_hipflex_l;
    json["l_upper_leg_abduction"] = theta_hipabd_l;
    if (right_leg_aligned_) {
        json["l_upper_leg_rotation"] = std::numeric_limits<double>::quiet_NaN();
    } else {
        json["l_upper_leg_rotation"] = theta_hiprot_l;
    }

    double verticality = std::abs(leg_direction_l.y());  // Y component
    if (verticality < 0.85) {  // leg not too vertical
        json["l_upper_leg_yaw"] = theta_hipyaw_l;
    } else {
        json["l_upper_leg_yaw"] = std::numeric_limits<double>::quiet_NaN();
    }

    // =====================
    // RIGHT HIP
    // =====================
    Eigen::Vector3d leg_direction_r = r_leg_upper_q * leg_axis_local;

    // Flexion / Extension
    Eigen::Vector3d leg_sagittal_r(0, leg_direction_r.y(), leg_direction_r.z());
    leg_sagittal_r.normalize();
    double theta_hipflex_r = std::atan2(leg_sagittal_r.z(), leg_sagittal_r.y());

    // Abduction / Adduction
    Eigen::Vector3d leg_coronal_r(leg_direction_r.x(), leg_direction_r.y(), 0);
    leg_coronal_r.normalize();
    double theta_hipabd_r = std::atan2(leg_coronal_r.x(), leg_coronal_r.y());

    // Horizontal yaw
    Eigen::Vector3d leg_horizontal_r(leg_direction_r.x(), 0, leg_direction_r.z());
    leg_horizontal_r.normalize();
    double theta_hipyaw_r = std::atan2(-leg_horizontal_r.z(), leg_horizontal_r.x());

    // Twist (local only)
    Eigen::Vector3d leg_forward_r = r_leg_upper_q * leg_forward_local_R;
    Eigen::Vector3d leg_forward_proj_r = leg_forward_r - leg_forward_r.dot(leg_direction_r) * leg_direction_r;
    leg_forward_proj_r.normalize();

    double theta_hiprot_r = std::atan2(
        expected_forward.cross(leg_forward_proj_r).dot(leg_direction_r),
        expected_forward.dot(leg_forward_proj_r)
    );

    json["r_upper_leg_flexion"] = -theta_hipflex_r;
    json["r_upper_leg_abduction"] = -theta_hipabd_r;
    if (left_leg_aligned_) {
        json["r_upper_leg_rotation"] = std::numeric_limits<double>::quiet_NaN();
    } else {
        json["r_upper_leg_rotation"] = -theta_hiprot_r;
    }

    verticality = std::abs(leg_direction_r.y());  // Y component
    if (verticality < 0.85) {  // leg not too vertical
        json["r_upper_leg_yaw"] = theta_hipyaw_r;
    } else {
        json["r_upper_leg_yaw"] = std::numeric_limits<double>::quiet_NaN();
    }

    // =====================
    // RIGHT KNEE
    // =====================
    Eigen::Vector3d lower_leg_direction_r = r_leg_lower_q * leg_axis_local;

    Eigen::Vector3d upper_leg_axis(0, 1, 0);

    double cos_knee_r = upper_leg_axis.dot(lower_leg_direction_r);
    cos_knee_r = std::clamp(cos_knee_r, -1.0, 1.0);

    double theta_knee_flex_r = std::acos(cos_knee_r);

    json["r_lower_leg_flexion"] = theta_knee_flex_r;

    Eigen::Vector3d foot_local(1, 0, 0);
    Eigen::Vector3d foot_dir_r = r_leg_lower_q * foot_local;

    Eigen::Vector3d foot_proj = foot_dir_r;
    foot_proj.z() = 0;
    foot_proj.normalize();

    double theta_tibia_rot_r = std::atan2(
        foot_proj.x(),
        -foot_proj.y()
    );

    json["r_lower_leg_rotation"] = theta_tibia_rot_r;

    // =====================
    // LEFT KNEE
    // =====================
    Eigen::Vector3d lower_leg_direction_l = l_leg_lower_q * leg_axis_local;

    double cos_knee_l = upper_leg_axis.dot(lower_leg_direction_l);
    cos_knee_l = std::clamp(cos_knee_l, -1.0, 1.0);

    double theta_knee_flex_l = std::acos(cos_knee_l);

    json["l_lower_leg_flexion"] = theta_knee_flex_l;

    Eigen::Vector3d foot_local_l(-1, 0, 0);
    Eigen::Vector3d foot_dir_l = l_leg_lower_q * foot_local_l;

    Eigen::Vector3d foot_proj_l = foot_dir_l;
    foot_proj_l.z() = 0;
    foot_proj_l.normalize();

    double theta_tibia_rot_l = std::atan2(
        -foot_proj_l.x(),
        -foot_proj_l.y()
    );

    json["l_lower_leg_rotation"] = theta_tibia_rot_l;

    // FFFEEEEETTTTTT

    Eigen::Vector3d foot_axis_local(0, 1, 0);        // +Y down foot
    Eigen::Vector3d foot_forward_local_R(-1, 0, 0);  // matches right-side convention

    // Foot direction (down axis)
    Eigen::Vector3d foot_direction_r = r_foot_q * foot_axis_local;

    // --------------------------------
    // Ankle Flexion / Extension
    // --------------------------------
    Eigen::Vector3d lower_leg_axis(0, 1, 0);  // tibia axis

    double cos_ankle_r = lower_leg_axis.dot(foot_direction_r);
    cos_ankle_r = std::clamp(cos_ankle_r, -1.0, 1.0);

    double theta_ankle_flex_r = std::acos(cos_ankle_r);

    json["ankleright_flexion"] = theta_ankle_flex_r;

    Eigen::Vector3d foot_direction_l = l_foot_q * foot_axis_local;

    // --------------------------------
    // Ankle Flexion / Extension
    // --------------------------------
    double cos_ankle_l = lower_leg_axis.dot(foot_direction_l);
    cos_ankle_l = std::clamp(cos_ankle_l, -1.0, 1.0);

    double theta_ankle_flex_l = std::acos(cos_ankle_l);

    json["ankleleft_flexion"] = theta_ankle_flex_l;

    return json;
}

nlohmann::json Kinematics::structure_json_from_quats(
    const std::vector<std::string>& keys,
    const std::vector<mojo_quaternion::quaternion>& quats
) {
    nlohmann::json json_quat = nlohmann::json::object();

    const size_t count = std::min(keys.size(), quats.size());

    for (size_t i = 0; i < count; ++i) {
        const auto& q = quats[i];

        json_quat[keys[i]] = {
            {"w", static_cast<double>(q.w)},
            {"x", static_cast<double>(q.x)},
            {"y", static_cast<double>(q.y)},
            {"z", static_cast<double>(q.z)}
        };
    }

    return json_quat;
}