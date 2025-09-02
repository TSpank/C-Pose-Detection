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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========================
// Constructor
// ========================
Kinematics::Kinematics()
    : prev_left_angle(0.0),
      prev_right_angle(0.0),
      prev_left_angle_initialized(false),
      prev_right_angle_initialized(false)
{
    // Constructor body can be empty if no other initialization needed
}

// ========================
// Private helper
// ========================
Eigen::Vector3d Kinematics::normalize(const Eigen::Vector3d& v) {
    double norm = v.norm();
    if (norm < 1e-12) return Eigen::Vector3d::Zero();
    return v / norm;
}

Eigen::Vector3d Kinematics::project_onto_plane(const Eigen::Vector3d& v,
                                          const Eigen::Vector3d& axis) {
        return v - v.dot(axis) * axis;
    }

double Kinematics::arm_rotation_angle(const Eigen::Vector3d& arm_vec,
                                     const Eigen::Vector3d& rotation_axis,
                                     const Eigen::Vector3d& reference) {
        Eigen::Vector3d axis = rotation_axis.normalized();

        Eigen::Vector3d arm_proj = project_onto_plane(arm_vec, axis);
        Eigen::Vector3d ref_proj = project_onto_plane(reference, axis);

        if (arm_proj.norm() < 1e-8) return 0.0;

        arm_proj.normalize();
        ref_proj.normalize();

        // Dot product clamped to [-1,1] to avoid numerical errors
        double dot = std::clamp(ref_proj.dot(arm_proj), -1.0, 1.0);
        double angle = std::acos(dot);

        // Determine the sign using cross product
        double sign = std::copysign(1.0, axis.dot(ref_proj.cross(arm_proj)));

        return sign * angle;
    }

std::pair<double, double> Kinematics::vector_to_pitch_yaw(
        const Eigen::Vector3d& arm_vec,
        const Eigen::Matrix3d& torso_matrix)
    {
        // Extract torso axes
        Eigen::Vector3d torso_x = torso_matrix.col(0);
        Eigen::Vector3d torso_y = torso_matrix.col(1);
        Eigen::Vector3d torso_z = torso_matrix.col(2);

        // --- yaw: projection in torso XY-plane (remove z component) ---
        Eigen::Vector3d arm_proj_xy = arm_vec - (arm_vec.dot(torso_z)) * torso_z;
        arm_proj_xy.normalize();

        double yaw = std::atan2(
            torso_z.dot(torso_x.cross(arm_proj_xy)),
            torso_x.dot(arm_proj_xy)
        );

        // --- pitch: elevation out of torso plane ---
        double pitch = std::asin(arm_vec.dot(torso_z));

        return {yaw, pitch};
    }

Eigen::Matrix3d Kinematics::build_hand_reference_frame(
    const Eigen::Vector3d& forearm_vec,
    const Eigen::Vector3d& shoulder_vec
) {
    auto normalize = [](const Eigen::Vector3d& v) -> Eigen::Vector3d {
        return v.normalized();
    };

    // Z-axis aligned with forearm
    Eigen::Vector3d z_axis = normalize(forearm_vec);

    // Y-axis: cross of z_axis and shoulder vector (palm normal)
    Eigen::Vector3d y_axis = normalize(z_axis.cross(shoulder_vec));

    // Adjust Y to match Python version: [0, y, |z|]
    y_axis = Eigen::Vector3d(0.0, y_axis.y(), std::abs(y_axis.z()));
    y_axis.normalize(); // normalize after modification

    // X-axis: re-orthogonalize
    Eigen::Vector3d x_axis = normalize(y_axis.cross(z_axis));

    // Construct 3x3 rotation matrix: columns = [x, y, z]
    Eigen::Matrix3d hand_frame;
    hand_frame.col(0) = x_axis;
    hand_frame.col(1) = y_axis;
    hand_frame.col(2) = z_axis;

    return hand_frame;
}

Eigen::Vector3d canonicalEuler(const Eigen::Quaterniond& q) {
    // Create a new quaternion instead of modifying the const input
    Eigen::Quaterniond quat = q;

    // Flip quaternion to consistent branch
    if (quat.w() < 0) quat = Eigen::Quaterniond(-quat.w(), -quat.x(), -quat.y(), -quat.z());

    // Compute angles using atan2 / asin manually
    double roll  = atan2(2*(quat.w()*quat.x() + quat.y()*quat.z()), 1 - 2*(quat.x()*quat.x() + quat.y()*quat.y()));
    double pitch = asin( std::clamp(2*(quat.w()*quat.y() - quat.z()*quat.x()), -1.0, 1.0) );
    double yaw   = atan2(2*(quat.w()*quat.z() + quat.x()*quat.y()), 1 - 2*(quat.y()*quat.y() + quat.z()*quat.z()));

    // Wrap angles to [-pi, pi] if needed
    Eigen::Vector3d euler(yaw, pitch, roll); // adjust order to match ZYX

    return euler;
}
// ========================
// Torso orientation
// ========================
std::pair<
    std::map<std::string, Eigen::Vector3d>,
    std::map<std::string, Eigen::Quaterniond>

> Kinematics::torso_orientation(const std::map<std::string, Eigen::Vector3d>& pose_data) 
{
    std::map<std::string, Eigen::Vector3d> body_pts;
        for (auto& [key, vec] : pose_data) {
        if (key.rfind("poses/", 0) == 0) { // starts with "body_pts/"
            std::string subkey = key.substr(std::string("poses/").size());
            body_pts[subkey] = vec;
        }
    }
    Eigen::Quaterniond torso_default(1,0,0,0); // identity quaternion (w,x,y,z)
    Eigen::Vector3d euler_default(0,0,0);

    std::map<std::string, Eigen::Vector3d> angle_map;
    std::map<std::string, Eigen::Quaterniond> quat_map;

    
    std::vector<std::string> keys1 = {"LeftShoulder","RightShoulder","LeftHip","RightHip", "No3D"};
    std::vector<std::string> keys2 = {"LeftShoulder","RightShoulder"};

    bool has_keys1 = true;
    for (auto& k : keys1) {
        if (body_pts.find(k) == body_pts.end()) {
            has_keys1 = false;
            break;
        }
    }

    bool has_keys2 = true;
    for (auto& k : keys2) {
        if (body_pts.find(k) == body_pts.end()) {
            has_keys2 = false;
            break;
        }
    }

    if (has_keys1) {
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d l_hip = body_pts.at("LeftHip");
        Eigen::Vector3d r_hip = body_pts.at("RightHip");

        Eigen::Vector3d y_axis = normalize((l_hip + r_hip)/2 - (l_shoulder + r_shoulder)/2);
        Eigen::Vector3d x_axis = normalize(l_shoulder - r_shoulder);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(y_axis));
        y_axis = normalize(z_axis.cross(x_axis));

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        Eigen::Quaterniond quat(rot_matrix);
        Eigen::Vector3d euler = canonicalEuler(quat);
        
        angle_map["torso"] = euler;
        quat_map["torso"]  = quat;
    }
    else if (has_keys2) {
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d ref(0,1,0);

        Eigen::Vector3d x_axis = normalize(l_shoulder - r_shoulder);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(ref));
        if (z_axis.dot(ref) > 0) z_axis = -z_axis;
        Eigen::Vector3d y_axis = normalize(z_axis.cross(x_axis));

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        Eigen::Quaterniond quat(rot_matrix);
        Eigen::Vector3d euler = canonicalEuler(quat);

        angle_map["torso"] = euler;
        quat_map["torso"]  = quat;
    }
    else {
        // Fall back to default identity orientation
        angle_map["torso"] = euler_default;
        quat_map["torso"]  = torso_default;
    }

    return {angle_map, quat_map};
}

// ========================
// Head Orientation
// ========================
std::pair<
    std::map<std::string, Eigen::Vector3d>,
    std::map<std::string, Eigen::Quaterniond>
>
Kinematics::head_orientation(
    const std::map<std::string, Eigen::Vector3d>& pose_data,
    const Eigen::Quaterniond& torso_quat
) {
    std::map<std::string, Eigen::Vector3d> head_euler_map;
    std::map<std::string, Eigen::Quaterniond> head_quat_map;

    Eigen::Vector3d head_euler(0, 0, 0);
    Eigen::Quaterniond head_quat(1,0,0,0); // identity quaternion

    // Extract face mesh points
    std::map<std::string, Eigen::Vector3d> face_mesh_pts;
    for (const auto& [key, vec] : pose_data) {
        if (key.rfind("poses/", 0) == 0) { // starts with "face_mesh_pts/"
            std::string subkey = key.substr(std::string("poses/").size());
            face_mesh_pts[subkey] = vec;
        }
    }

    // for (const auto& [key, vec] : face_mesh_pts) {
    // std::cout << key << ": [" 
    //           << vec.x() << ", " 
    //           << vec.y() << ", " 
    //           << vec.z() << "]\n";
    // }

    // Required keys
    std::vector<std::string> required_keys = {"Nose", "LeftEar", "RightEar"};

    for (auto& k : required_keys) {
        if (face_mesh_pts.find(k) == face_mesh_pts.end()) {
            // std::cout << "head_euler: ["
            //         << head_euler.x() << ", "
            //         << head_euler.y() << ", "
            //         << head_euler.z() << "]" << std::endl;

            // std::cout << "head_quat: ["
            //         << head_quat.w() << ", "
            //         << head_quat.x() << ", "
            //         << head_quat.y() << ", "
            //         << head_quat.z() << "]" << std::endl;
            head_euler_map["head"] = head_euler;
            head_quat_map["head"] = head_quat;
            return {head_euler_map, head_quat_map};
        }
    }

    // Get points
    Eigen::Vector3d nose = face_mesh_pts.at("Nose");
    Eigen::Vector3d l_trag = face_mesh_pts.at("LeftEar");
    Eigen::Vector3d r_trag = face_mesh_pts.at("RightEar");
    Eigen::Vector3d head_center = (l_trag + r_trag) / 2.0;

    // Local head coordinate frame
    Eigen::Vector3d z_axis = normalize(head_center - nose);    // forward
    Eigen::Vector3d x_axis = normalize(l_trag - r_trag);      // horizontal

    // Check for near-parallel vectors
    if (std::abs(z_axis.dot(x_axis)) > 0.99) {
        std::cerr << "Warning: Forward and horizontal axes are too parallel!\n";
    }

    Eigen::Vector3d y_axis = normalize(z_axis.cross(x_axis)); // vertical
    x_axis = normalize(y_axis.cross(z_axis));                 // re-orthogonalize

    // Build rotation matrix
    Eigen::Matrix3d rot_matrix;
    rot_matrix.col(0) = x_axis;
    rot_matrix.col(1) = y_axis;
    rot_matrix.col(2) = z_axis;

    Eigen::Quaterniond quat(rot_matrix);

    // Compute relative rotation w.r.t torso
    Eigen::Quaterniond q_rel = quat * torso_quat.inverse();

    head_quat = q_rel;
    head_euler = canonicalEuler(head_quat);

    head_euler_map["head"] = head_euler;
    head_quat_map["head"] = head_quat;

    // std::cout << "head_euler: ["
    //       << head_euler.x() << ", "
    //       << head_euler.y() << ", "
    //       << head_euler.z() << "]" << std::endl;

    // std::cout << "head_quat: ["
    //         << head_quat.w() << ", "
    //         << head_quat.x() << ", "
    //         << head_quat.y() << ", "
    //         << head_quat.z() << "]" << std::endl;

    return {head_euler_map, head_quat_map};
}

// ========================
// Left Arm Orientation
// ========================
std::pair<
    std::map<std::string, Eigen::Vector3d>,
    std::map<std::string, Eigen::Quaterniond>
>
Kinematics::left_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& pose_data,
    const Eigen::Quaterniond& torso_quat,
    double left_hand_roll
) {
    std::map<std::string, Eigen::Vector3d> angle_map;
    std::map<std::string, Eigen::Quaterniond> quat_map;

    Eigen::Vector3d euler_default(0,0,0);
    Eigen::Quaterniond quat_default(1,0,0,0); // identity

    std::map<std::string, Eigen::Vector3d> body_pts;
    for (auto& [key, vec] : pose_data) {
        if (key.rfind("poses/", 0) == 0) {
            std::string subkey = key.substr(std::string("poses/").size());
            body_pts[subkey] = vec;
        }
    }

    std::vector<std::string> upper_keys = {"LeftShoulder", "LeftElbow"};
    std::vector<std::string> lower_keys = {"LeftElbow", "LeftWrist"};
    std::vector<std::string> hip_keys   = {"LeftHip", "RightHip","lock2D"};

    auto has_keys = [&](const std::vector<std::string>& keys){
        return std::all_of(keys.begin(), keys.end(),
                           [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });
    };

    Eigen::Vector3d l_arm_upper_euler = euler_default;
    Eigen::Vector3d l_arm_lower_euler = euler_default;
    Eigen::Quaterniond l_arm_upper_quat = quat_default;
    Eigen::Quaterniond l_arm_lower_quat = quat_default;

    Eigen::Matrix3d torso_matrix = torso_quat.toRotationMatrix();
    Eigen::Vector3d torso_x = torso_matrix.col(0);
    Eigen::Vector3d torso_y = torso_matrix.col(1);
    Eigen::Vector3d torso_z = torso_matrix.col(2);

    // Upper arm
    if (has_keys(upper_keys)) {
        Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
        Eigen::Vector3d l_elbow = body_pts.at("LeftElbow");
        Eigen::Vector3d arm_vec = normalize(l_elbow - l_shoulder);

        Eigen::Vector3d x_axis = normalize(arm_vec.cross(torso_z));
        Eigen::Vector3d z_axis = normalize(x_axis.cross(arm_vec));
        Eigen::Vector3d y_axis = normalize(arm_vec);

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        l_arm_upper_quat = Eigen::Quaterniond(rot_matrix)*torso_quat.inverse();  
        l_arm_upper_euler = canonicalEuler(l_arm_upper_quat);

        if (!has_keys(hip_keys)) {
        }
    }

    // Lower arm
    if (has_keys(lower_keys)) {
        Eigen::Vector3d l_elbow = body_pts.at("LeftElbow");
        Eigen::Vector3d l_wrist = body_pts.at("LeftWrist");
        Eigen::Vector3d forearm_vec = normalize(l_wrist - l_elbow);

        Eigen::Vector3d x_axis = normalize(forearm_vec.cross(torso_z));
        Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
        Eigen::Vector3d y_axis = normalize(forearm_vec);

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        l_arm_lower_quat = Eigen::Quaterniond(rot_matrix)*torso_quat.inverse();;  
        l_arm_lower_euler = canonicalEuler(l_arm_lower_quat);

        if (!has_keys(hip_keys)) {
        }
    }

    angle_map["l_arm_upper"] = l_arm_upper_euler;
    angle_map["l_arm_lower"] = l_arm_lower_euler;
    quat_map["l_arm_upper"]  = l_arm_upper_quat;
    quat_map["l_arm_lower"]  = l_arm_lower_quat;

    return {angle_map, quat_map};
}
// ========================
// Right Arm Orientation
// ========================
std::pair<
    std::map<std::string, Eigen::Vector3d>,
    std::map<std::string, Eigen::Quaterniond>
>
Kinematics::right_arm_orientation(
    const std::map<std::string, Eigen::Vector3d>& pose_data,
    const Eigen::Quaterniond& torso_quat,
    double right_hand_roll
) {
    std::map<std::string, Eigen::Vector3d> angle_map;
    std::map<std::string, Eigen::Quaterniond> quat_map;

    Eigen::Vector3d euler_default(0,0,0);
    Eigen::Quaterniond quat_default(1,0,0,0); // identity

    std::map<std::string, Eigen::Vector3d> body_pts;
    for (auto& [key, vec] : pose_data) {
        if (key.rfind("poses/", 0) == 0) {
            std::string subkey = key.substr(std::string("poses/").size());
            body_pts[subkey] = vec;
        }
    }

    std::vector<std::string> upper_keys = {"RightShoulder", "RightElbow"};
    std::vector<std::string> lower_keys = {"RightElbow", "RightWrist"};
    std::vector<std::string> hip_keys   = {"LeftHip", "RightHip", "lock2D"};

    auto has_keys = [&](const std::vector<std::string>& keys){
        return std::all_of(keys.begin(), keys.end(),
                           [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });
    };

    Eigen::Vector3d r_arm_upper_euler = euler_default;
    Eigen::Vector3d r_arm_lower_euler = euler_default;
    Eigen::Quaterniond r_arm_upper_quat = quat_default;
    Eigen::Quaterniond r_arm_lower_quat = quat_default;

    Eigen::Matrix3d torso_matrix = torso_quat.toRotationMatrix();
    Eigen::Vector3d torso_x = torso_matrix.col(0);
    Eigen::Vector3d torso_y = torso_matrix.col(1);
    Eigen::Vector3d torso_z = torso_matrix.col(2);

    // Upper arm
    if (has_keys(upper_keys)) {
        Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
        Eigen::Vector3d r_elbow = body_pts.at("RightElbow");
        Eigen::Vector3d arm_vec = normalize(r_elbow - r_shoulder);

        Eigen::Vector3d x_axis = normalize(arm_vec.cross(torso_z));
        Eigen::Vector3d z_axis = normalize(x_axis.cross(arm_vec));
        Eigen::Vector3d y_axis = normalize(arm_vec);

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        r_arm_upper_quat = Eigen::Quaterniond(rot_matrix);
        r_arm_upper_euler = canonicalEuler(r_arm_upper_quat)*-1;

        if (!has_keys(hip_keys)) {
        }
    }

    // Lower arm
    if (has_keys(lower_keys)) {
        Eigen::Vector3d r_elbow = body_pts.at("RightElbow");
        Eigen::Vector3d r_wrist = body_pts.at("RightWrist");
        Eigen::Vector3d forearm_vec = normalize(r_wrist - r_elbow);

        Eigen::Vector3d x_axis = normalize(forearm_vec.cross(torso_z));
        Eigen::Vector3d z_axis = normalize(x_axis.cross(forearm_vec));
        Eigen::Vector3d y_axis = normalize(forearm_vec);

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        r_arm_lower_quat = Eigen::Quaterniond(rot_matrix);
        r_arm_lower_euler = canonicalEuler(r_arm_lower_quat)*-1;

        if (!has_keys(hip_keys)) {
        }
    }
    
    angle_map["r_arm_upper"] = r_arm_upper_euler;
    angle_map["r_arm_lower"] = r_arm_lower_euler;
    quat_map["r_arm_upper"]  = r_arm_upper_quat;
    quat_map["r_arm_lower"]  = r_arm_lower_quat;

    return {angle_map, quat_map};
}


// ========================
// Hand Orientation
// ========================
double Kinematics::compute_hand_roll(
    const Eigen::Vector3d& index_base,
    const Eigen::Vector3d& wrist,
    const Eigen::Vector3d& pinky_base,
    const Eigen::Vector3d& shoulder_vec,
    const Eigen::Matrix3d* rest_orientation,
    bool right
) {
    // Validate points
    if ((index_base - wrist).norm() < 1e-8)
        throw std::invalid_argument("index_base and wrist cannot be the same point");
    if ((pinky_base - index_base).norm() < 1e-8)
        throw std::invalid_argument("pinky_base and index_base cannot be the same point");

    // Local hand frame
    Eigen::Vector3d z_axis = normalize(index_base - wrist);
    Eigen::Vector3d x_axis = normalize(pinky_base - index_base);

    if (std::abs(z_axis.dot(x_axis)) > 0.99) return 0.0;

    Eigen::Vector3d y_axis;
    if (!right) {
        y_axis = normalize(x_axis.cross(z_axis));
    } else {
        y_axis = normalize(z_axis.cross(x_axis));
    }
    x_axis = normalize(y_axis.cross(z_axis));

    Eigen::Matrix3d R_now;
    R_now.col(0) = x_axis;
    R_now.col(1) = y_axis;
    R_now.col(2) = z_axis;

    Eigen::Matrix3d R_ref = rest_orientation ? *rest_orientation : build_hand_reference_frame(z_axis, shoulder_vec);

    Eigen::Quaterniond r_now(R_now);
    Eigen::Quaterniond r_ref(R_ref);
    Eigen::Quaterniond r_rel = r_now * r_ref.inverse();

    Eigen::Vector3d quat_vec(r_rel.x(), r_rel.y(), r_rel.z());
    double quat_w = r_rel.w();

    double angle = 2.0 * std::atan2(quat_vec.dot(z_axis), quat_w);

    // Unwrap angle for continuity
    if (!right) {
        if (prev_left_angle_initialized)
            angle += 2 * M_PI * std::round((prev_left_angle - angle) / (2 * M_PI));
        prev_left_angle = angle;
        prev_left_angle_initialized = true;
    } else {
        if (prev_right_angle_initialized)
            angle += 2 * M_PI * std::round((prev_right_angle - angle) / (2 * M_PI));
        prev_right_angle = angle;
        prev_right_angle_initialized = true;
    }

    return angle;
}

std::pair<double, double> Kinematics::get_hand_orientation(
    const std::map<std::string, Eigen::Vector3d>& pose_data
) {
    double left_hand_roll = 0.0;
    double right_hand_roll = 0.0;

    // Extract body_pts/*
    std::map<std::string, Eigen::Vector3d> body_pts;
    for (const auto& [key, vec] : pose_data) {
        const std::string prefix = "poses/";
        if (key.rfind(prefix, 0) == 0) {
            std::string subkey = key.substr(prefix.size());
            body_pts[subkey] = vec;
        }
    }

    // std::cout << "Body_pts" << std::endl;
    // for (const auto& [key, vec] : body_pts) {
    // std::cout << key << ": [" 
    //           << vec.x() << ", " 
    //           << vec.y() << ", " 
    //           << vec.z() << "]\n";
    // }

    // Check required body keys
    std::vector<std::string> required_body_keys = {"LeftShoulder", "RightShoulder"};
    bool body_ok = std::all_of(required_body_keys.begin(), required_body_keys.end(),
                                [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });

    if (!body_ok) return {left_hand_roll, right_hand_roll};

    Eigen::Vector3d l_shoulder = body_pts.at("LeftShoulder");
    Eigen::Vector3d r_shoulder = body_pts.at("RightShoulder");
    Eigen::Vector3d shoulder_vec = normalize(r_shoulder - l_shoulder);

    // Required hand keys
    std::vector<std::string> hand_keys = {"index_base", "pinky_base", "wrist", "ring_base"};

    // Right hand
    bool right_ok = std::all_of(hand_keys.begin(), hand_keys.end(),
                                [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });
    if (right_ok) {
        right_hand_roll = compute_hand_roll(
            body_pts.at("index_base"),
            body_pts.at("wrist"),
            body_pts.at("pinky_base"),
            shoulder_vec,
            nullptr,
            true
        );
    }

    // Left hand
    bool left_ok = std::all_of(hand_keys.begin(), hand_keys.end(),
                               [&](const std::string& k){ return body_pts.find(k) != body_pts.end(); });
    if (left_ok) {
        left_hand_roll = compute_hand_roll(
            body_pts.at("index_base"),
            body_pts.at("wrist"),
            body_pts.at("pinky_base"),
            shoulder_vec,
            nullptr,
            false
        );
    }

    return {left_hand_roll, right_hand_roll};
}


// ========================
// Neck kinematics
// ========================
void Kinematics::kinematics_neck(
    const std::map<std::string, Eigen::Vector3d>& pose_data) 
{
    // Get torso orientation as maps
    auto [torso_angles, torso_quats] = torso_orientation(pose_data);

    Eigen::Quaterniond torso_quat = torso_quats.at("torso");
    auto [head_angles, head_quats] = head_orientation(pose_data,torso_quat);

    std::pair<double, double> hand_rolls = get_hand_orientation(pose_data);

    double left_hand_rotation = hand_rolls.first;
    auto [left_arm_angles, left_arm_quats] = left_arm_orientation(pose_data,torso_quat,left_hand_rotation);

    double right_hand_rotation = hand_rolls.second;
    auto [right_arm_angles, right_arm_quats] = right_arm_orientation(pose_data,torso_quat,right_hand_rotation);

    //Correct Angle Direction
    

    // Append to history
    // Merge torso and head maps
    std::map<std::string, Eigen::Vector3d> merged_angles = torso_angles;
    merged_angles.insert(head_angles.begin(), head_angles.end());
    merged_angles.insert(left_arm_angles.begin(), left_arm_angles.end());
    merged_angles.insert(right_arm_angles.begin(), right_arm_angles.end());
    
    std::map<std::string, Eigen::Quaterniond> merged_quats = torso_quats;
    merged_quats.insert(head_quats.begin(), head_quats.end());
    merged_quats.insert(left_arm_quats.begin(), left_arm_quats.end());
    merged_quats.insert(right_arm_quats.begin(), right_arm_quats.end());

    // Append merged maps to history
    kinematic_angles.push_back(merged_angles);
    kinematic_quaternions.push_back(merged_quats);

    // Print merged_angles
    std::cout << "Merged Angles (degrees): " << std::endl;
    for (const auto& kv : merged_angles) {
        Eigen::Vector3d euler_deg = kv.second * (180.0 / M_PI);
        std::cout << "  " << kv.first << " : ["
                << euler_deg.x() << ", "
                << euler_deg.y() << ", "
                << euler_deg.z() << "]" << std::endl;
    }

}

std::map<std::string, double> Kinematics::structure_json_from_kinematics_history_angles(
    const std::vector<std::map<std::string, Eigen::Vector3d>>& kinematic_angles_history,
    bool torso_valid,
    bool right_hand,
    bool arms
) {
    if (kinematic_angles_history.empty()) return {};

    const auto& kinematic_angles_last = kinematic_angles_history.back();

    auto reverse_yaw_roll_direction = [](const Eigen::Vector3d& v) -> Eigen::Vector3d {
        return Eigen::Vector3d(-v.x(), -v.y(), v.z());
    };

    Eigen::Vector3d torso = kinematic_angles_last.count("torso") ? kinematic_angles_last.at("torso") : Eigen::Vector3d::Zero();
    Eigen::Vector3d head  = kinematic_angles_last.count("head")  ? kinematic_angles_last.at("head")  : Eigen::Vector3d::Zero();

    Eigen::Vector3d l_arm_1 = kinematic_angles_last.count("l_arm_upper") ? kinematic_angles_last.at("l_arm_upper") : Eigen::Vector3d::Zero();
    Eigen::Vector3d r_arm_1 = kinematic_angles_last.count("r_arm_upper") ? kinematic_angles_last.at("r_arm_upper") : Eigen::Vector3d::Zero();
    Eigen::Vector3d l_arm_2 = kinematic_angles_last.count("l_arm_lower") ? kinematic_angles_last.at("l_arm_lower") : Eigen::Vector3d::Zero();
    Eigen::Vector3d r_arm_2 = kinematic_angles_last.count("r_arm_lower") ? kinematic_angles_last.at("r_arm_lower") : Eigen::Vector3d::Zero();

    auto torso_rev = reverse_yaw_roll_direction(torso);
    auto head_rev  = reverse_yaw_roll_direction(head);

    std::map<std::string, double> pose;

    // Torso
    if (torso_valid) {
        pose["theta_torso_pitch_r"] = torso_rev.z();
        pose["theta_torso_tilt_r"]  = torso_rev.x();
        pose["theta_torso_yaw_r"]   = torso_rev.y();
        pose["theta_torso_roll_r"]  = 0.0;
        pose["theta_torso_bend_r"]  = 0.0;
    } else {
        pose["theta_torso_pitch_r"] = 0.0;
        pose["theta_torso_tilt_r"]  = 0.0;
        pose["theta_torso_yaw_r"]   = 0.0;
        pose["theta_torso_roll_r"]  = 0.0;
        pose["theta_torso_bend_r"]  = 0.0;
    }

    // Head
    bool head_valid = true;
    if (head_valid) {
        pose["theta_head_pitch_h"] = head_rev.z();
        pose["theta_head_roll_h"]  = head_rev.x();
        pose["theta_head_yaw_h"]   = head_rev.y();
    } else {
        pose["theta_head_pitch_h"] = 0.0;
        pose["theta_head_roll_h"]  = 0.0;
        pose["theta_head_yaw_h"]   = 0.0;
    }

    // Arms
    arms = true;
    if (arms) {
        pose["theta_armright_upper_alpha"] = r_arm_1.x();
        pose["theta_armright_upper_beta"]  = r_arm_1.y();
        pose["theta_armright_upper_gamma"] = r_arm_1.z();

        pose["theta_armleft_upper_alpha"] = l_arm_1.x();
        pose["theta_armleft_upper_beta"]  = l_arm_1.y();
        pose["theta_armleft_upper_gamma"] = l_arm_1.z();

        pose["theta_armright_lower_alpha"] = r_arm_2.x();
        pose["theta_armright_lower_beta"]  = r_arm_2.y();
        pose["theta_armright_lower_gamma"] = r_arm_2.z();

        pose["theta_armleft_lower_alpha"]  = l_arm_2.x();
        pose["theta_armleft_lower_beta"]   = l_arm_2.y();
        pose["theta_armleft_lower_gamma"]  = -l_arm_2.z();
    } else {
        std::vector<std::string> keys = {
            "theta_armright_upper_alpha", "theta_armright_upper_beta", "theta_armright_upper_gamma",
            "theta_armleft_upper_alpha", "theta_armleft_upper_beta", "theta_armleft_upper_gamma",
            "theta_armright_lower_alpha", "theta_armright_lower_beta", "theta_armright_lower_gamma",
            "theta_armleft_lower_alpha", "theta_armleft_lower_beta", "theta_armleft_lower_gamma"
        };
        for (auto& k : keys) pose[k] = 0.0;
    }

    return pose;
}

std::map<std::string, nlohmann::json> Kinematics::structure_json_from_kinematics_history_quats(
    const std::vector<std::map<std::string, Eigen::Quaterniond>>& kinematic_quaternions_history) {

    if (kinematic_quaternions_history.empty()) return {};

    const auto& kinematic_quaternions_last = kinematic_quaternions_history.back();

    std::map<std::string, nlohmann::json> pose;

    // Helper lambda to convert Eigen::Quaterniond to JSON
    auto quat_to_json = [](const Eigen::Quaterniond& quat) -> nlohmann::json {
        return nlohmann::json{{"w", quat.w()}, {"x", quat.x()}, {"y", quat.y()}, {"z", quat.z()}};
    };

    // Extract quaternion data for each body part
    if (kinematic_quaternions_last.count("torso")) {
        pose["torso_quat"] = quat_to_json(kinematic_quaternions_last.at("torso"));
    }
    if (kinematic_quaternions_last.count("head")) {
        pose["head_quat"] = quat_to_json(kinematic_quaternions_last.at("head"));
    }
    if (kinematic_quaternions_last.count("l_arm_upper")) {
        pose["l_arm_upper_quat"] = quat_to_json(kinematic_quaternions_last.at("l_arm_upper"));
    }
    if (kinematic_quaternions_last.count("r_arm_upper")) {
        pose["r_arm_upper_quat"] = quat_to_json(kinematic_quaternions_last.at("r_arm_upper"));
    }
    if (kinematic_quaternions_last.count("l_arm_lower")) {
        pose["l_arm_lower_quat"] = quat_to_json(kinematic_quaternions_last.at("l_arm_lower"));
    }
    if (kinematic_quaternions_last.count("r_arm_lower")) {
        pose["r_arm_lower_quat"] = quat_to_json(kinematic_quaternions_last.at("r_arm_lower"));
    }

    return pose;
}



// ========================
// Getters for latest stored values
// ========================
std::map<std::string, Eigen::Vector3d> Kinematics::latest_angles() const {
    if (!kinematic_angles.empty()) return kinematic_angles.back();
    return {};
}

std::map<std::string, Eigen::Quaterniond> Kinematics::latest_quaternions() const {
    if (!kinematic_quaternions.empty()) return kinematic_quaternions.back();
    return {};
}