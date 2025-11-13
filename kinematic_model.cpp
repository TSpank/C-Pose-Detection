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
     z_scale(0.0),
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
            prev_avatar_depth = (1.0 - alpha) * prev_avatar_depth + alpha * dist;
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
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d ref(0,1,0);

        Eigen::Vector3d x_axis = normalize(l_shoulder - r_shoulder);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(ref));
        if (z_axis.dot(ref) < 0) z_axis = -z_axis;
        Eigen::Vector3d y_axis = normalize(z_axis.cross(x_axis));

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        quat = Eigen::Quaterniond(rot_matrix);
    }

    // Smooth transitions
    if (has_prev_torso_quat_) {
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
    const std::map<std::string, Eigen::Vector3d>& face_mesh_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond head_quat(1, 0, 0, 0); // identity quaternion

    // Required keys
    std::vector<std::string> required_keys = {"Nose", "LeftEar", "RightEar"};
    for (auto& k : required_keys) {
        if (face_mesh_pts.find(k) == face_mesh_pts.end()) {
            return head_quat;  // early return with identity
        }
    }

    // Get points
    Eigen::Vector3d nose = face_mesh_pts.at("Nose");
    Eigen::Vector3d l_trag = face_mesh_pts.at("LeftEar");
    Eigen::Vector3d r_trag = face_mesh_pts.at("RightEar");

    if (face_mesh_pts.find("NoseTip") != face_mesh_pts.end()) {
        nose = face_mesh_pts.at("NoseTip");
        l_trag = face_mesh_pts.at("LeftEarTragus");
        r_trag = face_mesh_pts.at("RightEarTragus");   
    }

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
    Eigen::Quaterniond quat_rel = torso_quat.inverse()*quat;

    // Apply SLERP smoothing
    if (has_prev_head_quat_) {
        head_quat = prev_head_quat_.slerp(alpha, quat_rel);
    } else {
        head_quat = quat_rel;
        has_prev_head_quat_ = true;
    }

    prev_head_quat_ = head_quat;
    return head_quat;
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
        side + "PalmBase"
    };

    for (const auto& key : required_keys) {
        if (pose_data.find(key) == pose_data.end()) {
            return std::nullopt;  // equivalent to Python None
        }
    }

    const Eigen::Vector3d& vec1_start = pose_data.at(side + "IndexFingerBase");
    const Eigen::Vector3d& vec1_end   = pose_data.at(side + "MiddleFingerBase");
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
std::pair<Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::left_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond quat_default(1, 0, 0, 0);
    Eigen::Quaterniond l_arm_upper_quat = quat_default;
    Eigen::Quaterniond l_arm_lower_quat = quat_default;

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
        Eigen::Vector3d x_axis = normalize(arm_vec.cross(torso_z));
        Eigen::Vector3d z_axis = normalize(x_axis.cross(arm_vec));
        Eigen::Vector3d y_axis = arm_vec;

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        upper_quat = Eigen::Quaterniond(rot_matrix);
        l_arm_upper_quat.normalize();
        l_arm_upper_quat = torso_quat.inverse()*upper_quat;
    }

    // Lower arm (elbow to wrist)
    std::vector<std::string> lower_keys = {"LeftElbow", "LeftWrist"};
    if (has_keys(lower_keys)) {

        Eigen::Vector3d l_elbow = body_pts.at("LeftElbow");
        Eigen::Vector3d l_wrist = body_pts.at("LeftWrist");
        Eigen::Vector3d forearm_vec = normalize(l_wrist - l_elbow);

        // recompute upper arm 
        if (has_keys(upper_keys)) {
            Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
            Eigen::Vector3d l_elbow = body_pts.at("LeftElbow");
            Eigen::Vector3d arm_vec = normalize(l_elbow - l_shoulder);
            
            Eigen::Vector3d x_axis = normalize(forearm_vec.cross(arm_vec));
            Eigen::Vector3d z_axis = normalize(x_axis.cross(arm_vec));
            Eigen::Vector3d y_axis = arm_vec;

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            upper_quat = Eigen::Quaterniond(rot_matrix);
            l_arm_upper_quat.normalize();
            l_arm_upper_quat = torso_quat.inverse()*upper_quat;
        }

        std::optional<Eigen::Vector3d> left_hand_normal_vector = hand_normal_vector(body_pts, "left");
        if (left_hand_normal_vector) {
            Eigen::Vector3d x_axis = normalize(-(*left_hand_normal_vector));
            Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
            Eigen::Vector3d y_axis = normalize(forearm_vec);
            x_axis = normalize((y_axis).cross(z_axis));

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            lower_quat = Eigen::Quaterniond(rot_matrix);
            lower_quat.normalize();
            l_arm_lower_quat = upper_quat.inverse()*lower_quat;

            prev_l_handvec = x_axis;
        }
        else if (prev_l_handvec.norm() > 1e-6) {
            // Use previous hand normal vector to estimate current orientation
            Eigen::Vector3d x_axis = prev_l_handvec;
            Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
            Eigen::Vector3d y_axis = normalize(forearm_vec);
            x_axis = normalize((y_axis).cross(z_axis));

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            lower_quat = Eigen::Quaterniond(rot_matrix);
            lower_quat.normalize();
            l_arm_lower_quat = upper_quat.inverse()*lower_quat;
        }
        else{
            Eigen::Vector3d x_axis(1.0, 0.0, 0.0);
            Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
            Eigen::Vector3d y_axis = normalize(forearm_vec);
            x_axis = normalize((y_axis).cross(z_axis));

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            lower_quat = Eigen::Quaterniond(rot_matrix);
            lower_quat.normalize();
            l_arm_lower_quat = upper_quat.inverse()*lower_quat;
        }

    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    if (has_prev_l1_quat_) {
        l_arm_upper_quat = prev_l1_quat_.slerp(alpha, l_arm_upper_quat);
    } else {
        has_prev_l1_quat_ = true;
    }
    prev_l1_quat_ = l_arm_upper_quat;

    if (has_prev_l2_quat_) {
        l_arm_lower_quat = prev_l2_quat_.slerp(alpha, l_arm_lower_quat);
    } else {
        has_prev_l2_quat_ = true;
    }

    prev_l2_quat_ = l_arm_lower_quat;
    
    return {l_arm_upper_quat, l_arm_lower_quat};
}


//################################################################################################
//################################################################################################
//------------------------------------------------------------------------------------------------
//Right arm orientation
//------------------------------------------------------------------------------------------------
//################################################################################################
//################################################################################################
std::pair<Eigen::Quaterniond, Eigen::Quaterniond> Kinematics::right_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& body_pts,
    const Eigen::Quaterniond& torso_quat,
    double alpha
) {
    Eigen::Quaterniond quat_default(1, 0, 0, 0);
    Eigen::Quaterniond r_arm_upper_quat = quat_default;
    Eigen::Quaterniond r_arm_lower_quat = quat_default;

    Eigen::Matrix3d torso_matrix = torso_quat.toRotationMatrix();
    Eigen::Vector3d torso_z = torso_matrix.col(2);

    auto has_keys = [&](const std::vector<std::string>& keys) {
        return std::all_of(keys.begin(), keys.end(),
                           [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });
    };

    Eigen::Quaterniond upper_quat, lower_quat;

    // Upper arm (shoulder to elbow)
    std::vector<std::string> upper_keys = {"RightShoulder", "RightElbow"};
    if (has_keys(upper_keys)) {
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d r_elbow = body_pts.at("RightElbow");

        Eigen::Vector3d arm_vec = normalize(r_elbow - r_shoulder);
        Eigen::Vector3d x_axis = normalize(arm_vec.cross(torso_z));
        Eigen::Vector3d z_axis = normalize(x_axis.cross(arm_vec));
        Eigen::Vector3d y_axis = arm_vec;

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        upper_quat = Eigen::Quaterniond(rot_matrix);
        r_arm_upper_quat =  torso_quat.inverse()*upper_quat;
    }

    // Lower arm (elbow to wrist)
    std::vector<std::string> lower_keys = {"RightElbow", "RightWrist"};
    if (has_keys(lower_keys)) {
        Eigen::Vector3d r_elbow = body_pts.at("RightElbow");
        Eigen::Vector3d r_wrist = body_pts.at("RightWrist");
        Eigen::Vector3d forearm_vec = normalize(r_wrist - r_elbow);

        if (has_keys(upper_keys)) {
            Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
            Eigen::Vector3d r_elbow = body_pts.at("RightElbow");
            Eigen::Vector3d arm_vec = normalize(r_elbow - r_shoulder);
            
            Eigen::Vector3d x_axis = normalize(forearm_vec.cross(arm_vec));
            Eigen::Vector3d z_axis = normalize(x_axis.cross(arm_vec));
            Eigen::Vector3d y_axis = arm_vec;

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            upper_quat = Eigen::Quaterniond(rot_matrix);
            r_arm_upper_quat.normalize();
            r_arm_upper_quat = torso_quat.inverse()*upper_quat;
        }

        std::optional<Eigen::Vector3d> right_hand_normal_vector = hand_normal_vector(body_pts, "right");
        if (right_hand_normal_vector) {
            Eigen::Vector3d x_axis = normalize(-*right_hand_normal_vector);
            Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
            Eigen::Vector3d y_axis = forearm_vec;
            x_axis = normalize((y_axis).cross(z_axis));

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            lower_quat = Eigen::Quaterniond(rot_matrix);
            lower_quat.normalize();
            r_arm_lower_quat = upper_quat.inverse()*lower_quat;

            prev_r_handvec = x_axis;
        }
        else if (prev_r_handvec.norm() > 1e-6) {
            // Use previous hand normal vector to estimate current orientation
            Eigen::Vector3d x_axis = prev_r_handvec;
            Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
            Eigen::Vector3d y_axis = normalize(forearm_vec);
            x_axis = normalize((y_axis).cross(z_axis));

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            lower_quat = Eigen::Quaterniond(rot_matrix);
            lower_quat.normalize();
            r_arm_lower_quat = upper_quat.inverse()*lower_quat;
        }
        else{
            Eigen::Vector3d x_axis(1.0, 0.0, 0.0);
            Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
            Eigen::Vector3d y_axis = normalize(forearm_vec);
            x_axis = normalize((y_axis).cross(z_axis));

            Eigen::Matrix3d rot_matrix;
            rot_matrix.col(0) = x_axis;
            rot_matrix.col(1) = y_axis;
            rot_matrix.col(2) = z_axis;

            lower_quat = Eigen::Quaterniond(rot_matrix);
            lower_quat.normalize();
            r_arm_lower_quat = upper_quat.inverse()*lower_quat;
        }
    }

    // ========================
    // Apply SLERP smoothing
    // ========================
    if (has_prev_r1_quat_) {
        r_arm_upper_quat = prev_r1_quat_.slerp(alpha, r_arm_upper_quat);
    } else {
        has_prev_r1_quat_ = true;
    }
    prev_r1_quat_ = r_arm_upper_quat;

    if (has_prev_r2_quat_) {
        r_arm_lower_quat = prev_r2_quat_.slerp(alpha, r_arm_lower_quat);
    } else {
        has_prev_r2_quat_ = true;
    }
    prev_r2_quat_ = r_arm_lower_quat;

    return {r_arm_upper_quat, r_arm_lower_quat};
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

        std::vector<std::string> keys = {"LeftEar", "RightEar", "Nose", "LeftShoulder", "RightShoulder"};
        if (!has_keys(keys)){
            return; // z_scale already defined
        }

        Eigen::Vector3d l_ear = pose_data.at("LeftEar");
        Eigen::Vector3d r_ear = pose_data.at("RightEar");
        Eigen::Vector3d head_center = (l_ear + r_ear) / 2.0;
        Eigen::Vector3d nose = pose_data.at("Nose");

        Eigen::Vector3d l_shoulder = pose_data.at("LeftShoulder");
        Eigen::Vector3d r_shoulder = pose_data.at("RightShoulder");
        Eigen::Vector3d chest = (l_shoulder + r_shoulder) / 2.0;

        double head_roll_check = std::abs(std::atan2(l_ear.y() - r_ear.y(), l_ear.x() - r_ear.x()) * 180.0 / M_PI);
        double yaw_check = std::abs(std::abs(nose.x() - l_ear.x()) - std::abs(nose.x() - r_ear.x()));
        double pitch_check = std::abs(head_center.y() - nose.y());
       
        if (head_roll_check < 10 && yaw_check < 30 && pitch_check < 20) {
            z_scale = std::abs(nose.z() - (r_ear.z() + l_ear.z()) / 2.0) / std::abs(r_ear.x() - l_ear.x());
            //std::cout << "Internal Z Scale: " << z_scale << std::endl;
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
    // If z_scale is not defined, just return a copy of the original
    if (z_scale == 0.0) {
        return pose_data;
    }

    std::map<std::string, Eigen::Vector3d> normalized;

    for (const auto& [key, vec] : pose_data) {
        Eigen::Vector3d v = vec;

        // Check if first character is uppercase
        if (!key.empty() && std::isupper(static_cast<unsigned char>(key[0]))) {
            v.z() /= z_scale;
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

    auto [l_upper_quat, l_lower_quat] = left_arm_orientation(body_pts, torso_quat);
    auto [r_upper_quat, r_lower_quat] = right_arm_orientation(body_pts, torso_quat);

    // Return as mojo quaternion vector
    auto kinematic_output = structure_kinematic_output(
        torso_quat, head_quat,
        l_upper_quat, l_lower_quat,
        r_upper_quat, r_lower_quat
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
    const Eigen::Quaterniond& r_lower_quat)
{
    PoseResults results;
    results.quaternions.reserve(6);
    results.eulerAngles.reserve(6);

    auto to_mojo = [](const Eigen::Quaterniond& q) {
        return mojo_quaternion::quaternion(q.w(), q.x(), q.y(), q.z());
    };

    // --- Torso ---
    auto torso_q = to_mojo(torso_quat);
    auto torso_e = torso_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
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
    auto l_lower_e = l_lower_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(l_lower_q);
    results.eulerAngles.push_back(Eigen::Vector3d(-l_lower_e.z, -l_lower_e.x, -l_lower_e.y));

    // --- Right arm (upper) ---
    auto r_upper_q = to_mojo(l_upper_quat);
    auto r_upper_e = r_upper_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(r_upper_q);
    results.eulerAngles.push_back(Eigen::Vector3d(r_upper_e.z, -r_upper_e.x, r_upper_e.y));

    // --- Right arm (lower) ---
    auto r_lower_q = to_mojo(l_lower_quat);
    auto r_lower_e = r_lower_q.to_euler(mojo_math::EULER_ALGORITHM::XYZ);
    results.quaternions.push_back(r_lower_q);
    results.eulerAngles.push_back(Eigen::Vector3d(r_lower_e.z, -r_lower_e.x, r_lower_e.y));

    results.rawJson = avatar_json(results.eulerAngles);

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
    auto to_mojo = [](const Eigen::Quaterniond& q) {
        return mojo_quaternion::quaternion(q.w(), q.x(), q.y(), q.z());
    };

    std::map<std::string, double> json_angles;

    // --- Torso ---
    json_angles["theta_torso_pitch_r"] = euler_angles[0][0];
    json_angles["theta_torso_tilt_r"]  = euler_angles[0][1];
    json_angles["theta_torso_yaw_r"]   = euler_angles[0][2];
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

