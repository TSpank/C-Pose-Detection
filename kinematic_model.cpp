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
     prev_head_quat_(1, 0, 0, 0), has_prev_head_quat_(false),
     prev_l1_quat_(1, 0, 0, 0), has_prev_l1_quat_(false),
     prev_l2_quat_(1, 0, 0, 0), has_prev_l2_quat_(false),
     prev_r1_quat_(1, 0, 0, 0), has_prev_r1_quat_(false),
     prev_r2_quat_(1, 0, 0, 0), has_prev_r2_quat_(false),
     prev_r2_quat_g_(1, 0, 0, 0), has_prev_r2_quat_g_(false),
     prev_l2_quat_g_(1, 0, 0, 0), has_prev_l2_quat_g_(false),

     z_scale(4.0),
     z_scale_mesh(0.7),
     prev_l_handvec(0,0,0),
     prev_r_handvec(0,0,0), 

     prev_avatar_depth(0.0),
     has_prev_avatar_depth(false){}
    
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

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Estiamte avatar depth based on hip distance
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
double Kinematics::estimate_avatar_depth(std::map<std::string, Eigen::Vector3d>& pose_data, double alpha)
{
    auto normalize_to_0_2 = [](double x, double min_val, double max_val) -> double {
        double normalized = 2.0 * (max_val - x) / (max_val - min_val);
        // Clamp between 0 and 2
        if (normalized < 0.0) normalized = 0.0;
        if (normalized > 2.0) normalized = 2.0;
        return normalized;
    };

    // Ensure both hips exist
    if (pose_data.count("LeftHip") && pose_data.count("RightHip") && pose_data.count("video_width")) {
        double width = pose_data["video_width"].x();

        Eigen::Vector3d l_hip = pose_data["LeftHip"];
        Eigen::Vector3d r_hip = pose_data["RightHip"];

        double length = (l_hip - r_hip).norm();
        double dist = normalize_to_0_2(length, width / 8.0, width / 3.0);

        // Apply exponential smoothing (like Python version)
        if (has_prev_avatar_depth)
            prev_avatar_depth = (alpha) * prev_avatar_depth + alpha * dist;
        else
            prev_avatar_depth = dist;
    }
    // Return last known value (or 0 if no data yet)
    return prev_avatar_depth;
}


//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Torso orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
Eigen::Quaterniond Kinematics::torso_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    double alpha
)
{
    Eigen::Quaterniond quat(1, 0, 0, 0); // identity

    std::vector<std::string> keys1 = {"LeftShoulder", "RightShoulder", "LeftHip", "RightHip"};
    std::vector<std::string> keys2 = {"LeftShoulder", "RightShoulder"};

    bool has_keys1 = std::all_of(keys1.begin(), keys1.end(), [&](const std::string& k){
        return body_pts.find(k) != body_pts.end();
    });

    bool has_keys2 = std::all_of(keys2.begin(), keys2.end(), [&](const std::string& k){
        return body_pts.find(k) != body_pts.end();
    });

    if (has_keys1) {
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d l_hip      = body_pts.at("LeftHip");
        Eigen::Vector3d r_hip      = body_pts.at("RightHip");

        l_shoulder.z() *= 0.5;
        r_shoulder.z() *= 0.5;  
        l_hip.z()      *= 0.5;
        r_hip.z()      *= 0.5;

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
    
    else if (has_keys2) {
        // Shoulder and chest center
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        l_shoulder.z() *= 0.5;
        r_shoulder.z() *= 0.5;
        
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
        if (prev_torso_quat_.coeffs().dot(quat.coeffs()) < 0.0) {
            quat.coeffs() *= -1.0;
        }
        quat = prev_torso_quat_.slerp(alpha, quat);
    } else {
        has_prev_torso_quat_ = true;
    }

    prev_torso_quat_ = quat;
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
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond quat_rel(1, 0, 0, 0); // identity quaternion

    // Required keys
    std::vector<std::string> required_keys_face_mesh = {"mesh_NoseTip"};
    std::vector<std::string> required_keys = {"Nose", "LeftEar", "RightEar"};

    bool has_mesh_keys = std::all_of(required_keys_face_mesh.begin(), required_keys_face_mesh.end(), [&](const std::string& k){
        return body_pts.find(k) != body_pts.end();
    });

    bool has_required_keys = std::all_of(required_keys.begin(), required_keys.end(), [&](const std::string& k){
        return body_pts.find(k) != body_pts.end();
    });

    if (has_mesh_keys) {
        Eigen::Vector3d nose = body_pts.at("mesh_NoseTip");
        Eigen::Vector3d l_trag = body_pts.at("mesh_LeftEarTragus");
        Eigen::Vector3d r_trag = body_pts.at("mesh_RightEarTragus");  

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
    else if (has_required_keys){
            // Get points
        Eigen::Vector3d nose = body_pts.at("Nose");
        Eigen::Vector3d l_trag = body_pts.at("LeftEar");
        Eigen::Vector3d r_trag = body_pts.at("RightEar");

        nose.z()   *= 0.3;
        l_trag.z() *= 0.3;
        r_trag.z() *= 0.3;

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
        if (prev_head_quat_.coeffs().dot(quat_rel.coeffs()) < 0.0) {
            quat_rel.coeffs() *= -1.0;
        }
        quat_rel = prev_head_quat_.slerp(alpha, quat_rel);
    } else {
        has_prev_head_quat_ = true;
    }

    prev_head_quat_ = quat_rel;
    return quat_rel;
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Hand normal vector
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::optional<Eigen::Vector3d> hand_normal_vector(
    const std::map<std::string, Eigen::Vector3d>& pose_data,
    const std::string& side
) {

    std::vector<std::string> required_keys = {
        side + "IndexFingerBase",
        side + "MiddleFingerBase",
        side + "PalmBase",
        side + "PinkyFingerBase"
    };

    for (const auto& key : required_keys) {
        if (pose_data.find(key) == pose_data.end()) {
            return std::nullopt;  // equivalent to Python None
        }
    }

    const Eigen::Vector3d& vec1_start = pose_data.at(side + "IndexFingerBase");
    const Eigen::Vector3d& vec1_end   = pose_data.at(side + "PinkyFingerBase");
    const Eigen::Vector3d& vec2_start = pose_data.at(side + "PalmBase");
    const Eigen::Vector3d& vec2_end   = pose_data.at(side + "MiddleFingerBase");

    Eigen::Vector3d dir1 = vec1_end - vec1_start;
    Eigen::Vector3d dir2 = vec2_end - vec2_start;

    return dir1.cross(dir2);  // Eigen::Vector3d result
}


//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Left arm orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::left_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond l_arm_upper_quat(1, 0, 0, 0);
    Eigen::Quaterniond l_arm_lower_quat(1, 0, 0, 0);
    Eigen::Quaterniond l_arm_lower_quat_g(1, 0, 0, 0);

    Eigen::Matrix3d torso_matrix = torso_quat.toRotationMatrix();
    Eigen::Vector3d torso_z = torso_matrix.col(2);

    auto has_keys = [&](const std::vector<std::string>& keys) {
        return std::all_of(keys.begin(), keys.end(),
                           [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });
    };

    Eigen::Quaterniond upper_quat, lower_quat;

    // Upper arm (shoulder to elbow)
    std::vector<std::string> upper_keys = {"LeftShoulder", "LeftElbow"};
    if (has_keys(upper_keys)) {
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d l_elbow = body_pts.at("LeftElbow");

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
    std::vector<std::string> lower_keys = {"LeftElbow", "LeftWrist"};
    if (has_keys(lower_keys) && has_keys(upper_keys)) {

        // --- HAND OVERRIDE: use leftPalmBase_pixels if available ---
        Eigen::Vector3d l_wrist;
        const double MAX_JUMP_PX = 20.0;   // pixel gate
        if (body_pts.count("leftPalmBase_pixels")) {
            const Eigen::Vector3d& left_palm = body_pts.at("leftPalmBase_pixels");
            Eigen::Vector3d raw_wrist = body_pts.at("LeftWrist");
            Eigen::Vector3d palm_wrist(
                left_palm.x(),
                left_palm.y(),
                raw_wrist.z()
            );

            // Only compare in image space (x,y)
            double jump = (palm_wrist.head<2>() - raw_wrist.head<2>()).norm();

            if (jump < MAX_JUMP_PX) {
                // Smooth blend → prevents snapping
                l_wrist = palm_wrist;
            } else {
                // Palm base unreliable this frame
                l_wrist = raw_wrist;
            }
        } else {
            l_wrist = body_pts.at("LeftWrist");
        }

        Eigen::Vector3d l_elbow = body_pts.at("LeftElbow");
        Eigen::Vector3d forearm_vec = normalize(l_wrist - l_elbow);

        // recompute upper arm 
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d arm_vec = normalize(l_elbow - l_shoulder);
        
        Eigen::Vector3d l1_x_axis = normalize(forearm_vec.cross(arm_vec));
        Eigen::Vector3d l1_z_axis = normalize(l1_x_axis.cross(arm_vec));
        Eigen::Vector3d l1_y_axis = arm_vec;

         // ---- SINGULARITY HANDLING if previous upper arm exists ----
        if (has_prev_l1_quat_) {
            Eigen::Quaterniond prev_upper_global = torso_quat * (prev_l1_quat_);
            Eigen::Vector3d prev_l1_x_axis = prev_upper_global.toRotationMatrix().col(0);

            double elbow_align = arm_vec.dot(forearm_vec);
            if (std::abs(elbow_align) > 0.8) { // nearly straight arm
                l1_x_axis = prev_l1_x_axis;
                l1_z_axis = normalize(l1_x_axis.cross(arm_vec));
                l1_y_axis = arm_vec;
            }
        }

        Eigen::Matrix3d l1_rot_matrix;
        l1_rot_matrix.col(0) = l1_x_axis;
        l1_rot_matrix.col(1) = l1_y_axis;
        l1_rot_matrix.col(2) = l1_z_axis;

        upper_quat = Eigen::Quaterniond(l1_rot_matrix);
        upper_quat.normalize();
        l_arm_upper_quat = torso_quat.inverse()*upper_quat;
        
        std::optional<Eigen::Vector3d> left_hand_normal_vector = hand_normal_vector(body_pts, "left");

        Eigen::Vector3d l2_x_axis;
        if (left_hand_normal_vector) {
            std::cout << "l2_x_axis: hand normal\n";
            l2_x_axis = normalize(-(*left_hand_normal_vector));
            prev_l_handvec = l2_x_axis;
        }
        else if (prev_l_handvec.norm() > 1e-6) {
            std::cout << "l2_x_axis: prev fallback\n";
            l2_x_axis = l1_x_axis;//prev_l_handvec;
        }
        else {
            std::cout << "l2_x_axis: default\n";
            l2_x_axis = Eigen::Vector3d(1.0, 0.0, 0.0);
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
    }

    auto torso_q = to_mojo(torso_quat);
    auto torso_e = torso_q.to_euler(mojo_math::EULER_ALGORITHM::YXZ);
    if (torso_e.y < (-30.0*M_PI / 180.0)) {
        l_arm_upper_quat        = Eigen::Quaterniond::Identity();
        l_arm_lower_quat        = Eigen::Quaterniond::Identity();
    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    if (has_prev_l1_quat_) {
        if (prev_l1_quat_.coeffs().dot(l_arm_upper_quat.coeffs()) < 0.0) {
            l_arm_upper_quat.coeffs() *= -1.0;
        }
        l_arm_upper_quat = prev_l1_quat_.slerp(alpha, l_arm_upper_quat);
    } else {
        has_prev_l1_quat_ = true;
    }

    prev_l1_quat_ = l_arm_upper_quat;

    if (has_prev_l2_quat_) {
        if (prev_l2_quat_.coeffs().dot(l_arm_lower_quat.coeffs()) < 0.0) {
            l_arm_lower_quat.coeffs() *= -1.0;
        }
        l_arm_lower_quat = prev_l2_quat_.slerp(alpha, l_arm_lower_quat);
    } else {
        has_prev_l2_quat_ = true;
    }

    prev_l2_quat_ = l_arm_lower_quat;

    //r1_quat global
    if (has_prev_l2_quat_g_) {
        if (prev_r2_quat_g_.coeffs().dot(l_arm_lower_quat_g.coeffs()) < 0.0) {
            l_arm_lower_quat.coeffs() *= -1.0;
        }
        l_arm_lower_quat_g = prev_l2_quat_g_.slerp(alpha, l_arm_lower_quat_g);
    } else {
        has_prev_l2_quat_g_ = true;
    }
    prev_l2_quat_g_ = l_arm_lower_quat_g;

    return {l_arm_upper_quat, l_arm_lower_quat, l_arm_lower_quat_g};
}


//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Right arm orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::right_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond r_arm_upper_quat(1, 0, 0, 0);
    Eigen::Quaterniond r_arm_lower_quat(1, 0, 0, 0);
    Eigen::Quaterniond r_arm_lower_quat_g(1, 0, 0, 0);

    Eigen::Matrix3d torso_matrix = torso_quat.toRotationMatrix();
    Eigen::Vector3d torso_z = torso_matrix.col(2);

    auto has_keys = [&](const std::vector<std::string>& keys) {
        return std::all_of(keys.begin(), keys.end(),
                           [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });
    };

    Eigen::Quaterniond upper_quat, lower_quat;

    // ========================
    // Upper arm (shoulder → elbow)
    // ========================
    std::vector<std::string> upper_keys = {"RightShoulder", "RightElbow"};
    if (has_keys(upper_keys)) {
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d r_elbow    = body_pts.at("RightElbow");

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
    std::vector<std::string> lower_keys = {"RightElbow", "RightWrist"};
    if (has_keys(lower_keys) && has_keys(upper_keys)) {

        // --- HAND OVERRIDE: use rightPalmBase_pixels if available ---
        // --- HAND OVERRIDE: use leftPalmBase_pixels if available ---
        Eigen::Vector3d r_wrist;
        const double MAX_JUMP_PX = 20.0;   // pixel gate
        if (body_pts.count("rightPalmBase_pixels")) {
            const Eigen::Vector3d& right_palm = body_pts.at("rightPalmBase_pixels");
            Eigen::Vector3d raw_wrist = body_pts.at("RightWrist");
            Eigen::Vector3d palm_wrist(
                right_palm.x(),
                right_palm.y(),
                raw_wrist.z()
            );

            // Only compare in image space (x,y)
            double jump = (palm_wrist.head<2>() - raw_wrist.head<2>()).norm();

            if (jump < MAX_JUMP_PX) {
                // Smooth blend → prevents snapping
                r_wrist = palm_wrist;
            } else {
                // Palm base unreliable this frame
                r_wrist = raw_wrist;
            }
        } else {
            r_wrist = body_pts.at("RightWrist");
        }

        Eigen::Vector3d r_elbow = body_pts.at("RightElbow");
        Eigen::Vector3d forearm_vec = normalize(r_wrist - r_elbow);

        // recompute upper arm
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d arm_vec = normalize(r_elbow - r_shoulder);

        Eigen::Vector3d r1_x_axis = normalize(forearm_vec.cross(arm_vec));
        Eigen::Vector3d r1_z_axis = normalize(r1_x_axis.cross(arm_vec));
        Eigen::Vector3d r1_y_axis = arm_vec;

        // ---- SINGULARITY HANDLING (straight arm) ----
        if (has_prev_r1_quat_) {
            Eigen::Quaterniond prev_upper_global = torso_quat * prev_r1_quat_;
            Eigen::Vector3d prev_r1_x_axis = prev_upper_global.toRotationMatrix().col(0);

            double elbow_align = arm_vec.dot(forearm_vec);
            if (std::abs(elbow_align) > 0.8) {
                r1_x_axis = prev_r1_x_axis;
                r1_z_axis = normalize(r1_x_axis.cross(arm_vec));
                r1_y_axis = arm_vec;
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
        std::optional<Eigen::Vector3d> right_hand_normal_vector =
            hand_normal_vector(body_pts, "right");

        Eigen::Vector3d r2_x_axis;
        if (right_hand_normal_vector) {
            r2_x_axis = normalize(-(*right_hand_normal_vector));
            prev_r_handvec = r2_x_axis;
            //std::cout << "r2_x_axis: hand normal\n";
        }
        else if (prev_r_handvec.norm() > 1e-6) {
            r2_x_axis = r1_x_axis;//prev_r_handvec;
            //std::cout << "r2_x_axis: prev fallback\n";
        }
        else {
            r2_x_axis = Eigen::Vector3d(1.0, 0.0, 0.0);
            //std::cout << "r2_x_axis: default\n";
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
    }


    auto torso_q = to_mojo(torso_quat);
    auto torso_e = torso_q.to_euler(mojo_math::EULER_ALGORITHM::YXZ);
    if (torso_e.y > (30.0*M_PI / 180.0)) {
        r_arm_upper_quat        = Eigen::Quaterniond::Identity();
        r_arm_lower_quat        = Eigen::Quaterniond::Identity();
    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    //r1_quat
    if (has_prev_r1_quat_) {
        if (prev_r1_quat_.coeffs().dot(r_arm_upper_quat.coeffs()) < 0.0) {
            r_arm_upper_quat.coeffs() *= -1.0;
        }
        r_arm_upper_quat = prev_r1_quat_.slerp(alpha, r_arm_upper_quat);
    } else {
        has_prev_r1_quat_ = true;
    }
    prev_r1_quat_ = r_arm_upper_quat;

    //r2_quat
    if (has_prev_r2_quat_) {
        if (prev_r2_quat_.coeffs().dot(r_arm_lower_quat.coeffs()) < 0.0) {
            r_arm_lower_quat.coeffs() *= -1.0;
        }
        r_arm_lower_quat = prev_r2_quat_.slerp(alpha, r_arm_lower_quat);
    } else {
        has_prev_r2_quat_ = true;
    }
    prev_r2_quat_ = r_arm_lower_quat;

    //r1_quat global
    if (has_prev_r2_quat_g_) {
        if (prev_r2_quat_g_.coeffs().dot(r_arm_lower_quat_g.coeffs()) < 0.0) {
            r_arm_lower_quat.coeffs() *= -1.0;
        }
        r_arm_lower_quat_g = prev_r2_quat_g_.slerp(alpha, r_arm_lower_quat_g);
    } else {
        has_prev_r2_quat_g_ = true;
    }
    prev_r2_quat_g_ = r_arm_lower_quat_g;

    return {r_arm_upper_quat, r_arm_lower_quat, r_arm_lower_quat_g};
}

//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
// calulate Z scaling using 3d face data
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
void Kinematics::update_z_scale(const std::map<std::string, Eigen::Vector3d>& pose_data) {
        // Required keys
        auto has_keys = [&](const std::vector<std::string>& keys) {
        return std::all_of(keys.begin(), keys.end(),
                           [&](const std::string& k){ return pose_data.find(k) != pose_data.end(); });
        };

        std::vector<std::string> keys_mesh = {"mesh_NoseTip", "mesh_LeftEarTragus", "mesh_RightEarTragus"};
        if (has_keys(keys_mesh)){
            Eigen::Vector3d l_ear = pose_data.at("mesh_LeftEarTragus");
            Eigen::Vector3d r_ear = pose_data.at("mesh_RightEarTragus");
            Eigen::Vector3d head_center = (l_ear + r_ear) / 2.0;
            Eigen::Vector3d nose = pose_data.at("mesh_NoseTip");

            z_scale_mesh = std::abs(nose.z() - (r_ear.z() + l_ear.z()) / 2.0) / std::abs(r_ear.x() - l_ear.x());
            //std::cout << "Z Scale mesh: " << z_scale_mesh << std::endl;
            }


        std::vector<std::string> keys = {"LeftEar", "RightEar", "Nose", "LeftShoulder", "RightShoulder"};
        if (!has_keys(keys)){
            return; // z_scale already defined
        }

        Eigen::Vector3d nose = pose_data.at("Nose");
        Eigen::Vector3d l_ear = pose_data.at("LeftEar");
        Eigen::Vector3d r_ear = pose_data.at("RightEar");

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

        if (std::abs(head_e.x) < 0.052 &&
            std::abs(head_e.y) < 0.052 &&
            std::abs(head_e.z) < 0.052)
        {
            z_scale = 0.7*(std::abs(nose.z() - (r_ear.z() + l_ear.z()) / 2.0) /  std::abs(r_ear.x() - l_ear.x()));
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
std::map<std::string, Eigen::Vector3d> 
Kinematics::normalize_z_data(const std::map<std::string, Eigen::Vector3d>& pose_data)

{
    std::map<std::string, Eigen::Vector3d> normalized;

    for (const auto& [key, vec] : pose_data) {
        Eigen::Vector3d v = vec;

        // Check if first character is uppercase
        if ((!key.empty() && std::isupper(static_cast<unsigned char>(key[0]))) && (z_scale != 0.0)) {
            v.z() /= z_scale;
        }
        if ((!key.empty() && key[0] == 'm') && (z_scale_mesh != 0.0)){
            v.z() /= z_scale_mesh;
        }
        normalized[key] = v;  // store the (possibly modified) vector
    }
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
        const std::map<std::string, Eigen::Vector3d>& pose_data)
    {

    update_z_scale(pose_data);
    auto body_pts = normalize_z_data(pose_data);

    // Compute orientations
    Eigen::Quaterniond torso_quat = torso_orientation(body_pts);
    Eigen::Quaterniond head_quat  = head_orientation(body_pts, torso_quat);

    auto [l_upper_quat, l_lower_quat, l_lower_quat_g] = left_arm_orientation(body_pts, torso_quat);
    auto [r_upper_quat, r_lower_quat, r_lower_quat_g] = right_arm_orientation(body_pts, torso_quat);

    // Return as mojo quaternion vector
    auto kinematic_output = structure_kinematic_output(
        torso_quat, head_quat,
        l_upper_quat, l_lower_quat,
        r_upper_quat, r_lower_quat, 
        r_lower_quat_g, l_lower_quat_g
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
    const Eigen::Quaterniond& head_quat,
    const Eigen::Quaterniond& l_upper_quat,
    const Eigen::Quaterniond& l_lower_quat,
    const Eigen::Quaterniond& r_upper_quat,
    const Eigen::Quaterniond& r_lower_quat, 
    const Eigen::Quaterniond& r_lower_quat_g,
    const Eigen::Quaterniond& l_lower_quat_g)
{
    PoseResults results;
    results.quaternions.reserve(6);
    results.eulerAngles.reserve(6);

    // --- Torso ---
    auto torso_q = to_mojo(torso_quat);
    auto torso_e = torso_q.to_euler(mojo_math::EULER_ALGORITHM::YXZ);
    results.quaternions.push_back(torso_q);
    results.eulerAngles.push_back(Eigen::Vector3d(torso_e.x, torso_e.z, torso_e.y));

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

    // --- Head ---
    json_angles["theta_head_pitch_h"] = euler_angles[1].x();
    json_angles["theta_head_roll_h"]  = euler_angles[1].y();
    json_angles["theta_head_yaw_h"]   = euler_angles[1].z();

    // --- Right arm ---
    json_angles["theta_armright_upper_alpha"] = euler_angles[4].x();
    json_angles["theta_armright_upper_beta"]  = euler_angles[4].y();
    json_angles["theta_armright_upper_gamma"] = euler_angles[4].z();

    json_angles["theta_armright_lower_alpha"] = euler_angles[5].x();
    json_angles["theta_armright_lower_beta"]  = euler_angles[5].y();
    json_angles["theta_armright_lower_gamma"] = euler_angles[5].z();

    
    // --- Left arm ---
    json_angles["theta_armleft_upper_alpha"] = euler_angles[2].x();
    json_angles["theta_armleft_upper_beta"]  = euler_angles[2].y();
    json_angles["theta_armleft_upper_gamma"] = euler_angles[2].z();

    json_angles["theta_armleft_lower_alpha"] = euler_angles[3].x();
    json_angles["theta_armleft_lower_beta"]  = euler_angles[3].y();
    json_angles["theta_armleft_lower_gamma"] = euler_angles[3].z();

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

    json["theta_head_pitch_h"] = euler_angles[1].x();
    json["theta_head_roll_h"]  = euler_angles[1].y();
    json["theta_head_yaw_h"]   = euler_angles[1].z();

    // -------------------------
    // Helper: convert mojo -> Eigen
    // -------------------------
    auto to_eigen = [](const mojo_quaternion::quaternion& q) {
        Eigen::Quaterniond eq(q.w, q.x, q.y, q.z);
        eq.normalize(); // safe normalization
        return eq;
    };

    // -------------------------
    // Arm quaternions
    // -------------------------
    Eigen::Quaterniond l_upper = to_eigen(quaternions[2]);
    Eigen::Quaterniond l_lower = to_eigen(quaternions[3]);
    Eigen::Quaterniond l_lower_g = to_eigen(quaternions[4]);
    Eigen::Quaterniond r_upper = to_eigen(quaternions[5]);
    Eigen::Quaterniond r_lower = to_eigen(quaternions[6]);
    Eigen::Quaterniond r_lower_g = to_eigen(quaternions[7]);

    Eigen::Vector3d torso_forward = Eigen::Vector3d(0, 0, 1);

    Eigen::Vector3d arm_axis_local(0, 1, 0);   // +Y along the arm
    Eigen::Vector3d arm_forward_local(1, 0, 0); // along forearm for rotation

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
    double theta_yaw_l = std::atan2(arm_horiz_l.z(), -arm_horiz_l.x());

    // Internal/External rotation
    Eigen::Vector3d arm_forward_l = l_upper * arm_forward_local;
    Eigen::Vector3d arm_proj_l = arm_forward_l - arm_forward_l.dot(arm_dir_l) * arm_dir_l;
    arm_proj_l.normalize();
    double theta_rot_l = std::atan2(
        torso_forward.cross(arm_proj_l).dot(arm_dir_l),
        torso_forward.dot(arm_proj_l)
    );

    json["theta_armleft_upper_alpha"] = theta_flex_l;
    json["theta_armleft_upper_beta"]  = theta_abduct_l;
    json["theta_armleft_upper_gamma"] = theta_rot_l;
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
    double theta_yaw_r = std::atan2(arm_horiz_r.z(), arm_horiz_r.x());

    // Internal/External rotation
    Eigen::Vector3d arm_forward_r = r_upper * arm_forward_r_local;
    Eigen::Vector3d arm_proj_r = arm_forward_r - arm_forward_r.dot(arm_dir_r) * arm_dir_r;
    arm_proj_r.normalize();
    double theta_rot_r = std::atan2(
        torso_forward.cross(arm_proj_r).dot(arm_dir_r),
        torso_forward.dot(arm_proj_r)
    );

    json["theta_armright_upper_alpha"] = theta_flex_r;
    json["theta_armright_upper_beta"]  = theta_abduct_r;
    json["theta_armright_upper_gamma"] = -theta_rot_r; // match Python
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
    palm_l.z() = 0; // project onto torso frontal plane
    palm_l.normalize();
    double theta_pronation_l = std::atan2(palm_l.x(), palm_l.y());

    Eigen::Vector3d palm_l_g = l_lower_g * Eigen::Vector3d(-1, 0, 0);
    palm_l_g.z() = 0;
    palm_l_g.normalize();
    double theta_pronation_l_g = std::atan2(palm_l_g.x(), palm_l_g.y());

    json["theta_armleft_lower_beta"]  = theta_elbow_flex_l;
    json["theta_armleft_lower_gamma"] = -theta_pronation_l;
    json["theta_armleft_lower_gamma_g"] = -theta_pronation_l_g;

    // -------------------------
    // RIGHT ELBOW
    // -------------------------
    Eigen::Vector3d lower_dir_r = r_lower * arm_axis_local;

    double cos_angle_r = upper_axis.dot(lower_dir_r);
    cos_angle_r = std::clamp(cos_angle_r, -1.0, 1.0);
    double theta_elbow_flex_r = std::acos(cos_angle_r);

    Eigen::Vector3d palm_r = r_lower * Eigen::Vector3d(1, 0, 0);
    palm_r.z() = 0;
    palm_r.normalize();
    double theta_pronation_r = std::atan2(palm_r.x(), palm_r.y());

    Eigen::Vector3d palm_r_g = r_lower_g * Eigen::Vector3d(1, 0, 0);
    palm_r_g.z() = 0;
    palm_r_g.normalize();
    double theta_pronation_r_g = std::atan2(palm_r_g.x(), palm_r_g.y());

    json["theta_armright_lower_beta"]  = theta_elbow_flex_r;
    json["theta_armright_lower_gamma"] = theta_pronation_r;
    json["theta_armright_lower_gamma_g"] = theta_pronation_r_g;

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