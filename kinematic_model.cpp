#include "kinematic_model.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
        if (key.rfind("body_pts/", 0) == 0) { // starts with "body_pts/"
            std::string subkey = key.substr(std::string("body_pts/").size());
            body_pts[subkey] = vec;
        }
    }
    Eigen::Quaterniond torso_default(1,0,0,0); // identity quaternion (w,x,y,z)
    Eigen::Vector3d euler_default(0,0,0);

    std::map<std::string, Eigen::Vector3d> angle_map;
    std::map<std::string, Eigen::Quaterniond> quat_map;

    std::vector<std::string> keys1 = {"l_shoulder","r_shoulder","l_hip","r_hip"};
    std::vector<std::string> keys2 = {"l_shoulder","r_shoulder"};

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
        Eigen::Vector3d l_shoulder = body_pts.at("l_shoulder");
        Eigen::Vector3d r_shoulder = body_pts.at("r_shoulder");
        Eigen::Vector3d l_hip = body_pts.at("l_hip");
        Eigen::Vector3d r_hip = body_pts.at("r_hip");

        Eigen::Vector3d y_axis = normalize((l_hip + r_hip)/2 - (l_shoulder + r_shoulder)/2);
        Eigen::Vector3d x_axis = normalize(l_shoulder - r_shoulder);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(y_axis));
        y_axis = normalize(z_axis.cross(x_axis));

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        Eigen::Quaterniond quat(rot_matrix);
        Eigen::Vector3d euler = rot_matrix.eulerAngles(2,1,0); // ZYX order

        angle_map["torso"] = euler;
        quat_map["torso"]  = quat;
    }
    else if (has_keys2) {
        Eigen::Vector3d l_shoulder = body_pts.at("l_shoulder");
        Eigen::Vector3d r_shoulder = body_pts.at("r_shoulder");
        Eigen::Vector3d ref(0,1,0);

        Eigen::Vector3d x_axis = normalize(l_shoulder - r_shoulder);
        Eigen::Vector3d z_axis = normalize(x_axis.cross(ref));
        if (z_axis.dot(ref) < 0) z_axis = -z_axis;
        Eigen::Vector3d y_axis = normalize(z_axis.cross(x_axis));

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        Eigen::Quaterniond quat(rot_matrix);
        Eigen::Vector3d euler = rot_matrix.eulerAngles(2,1,0); // ZYX order

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
        if (key.rfind("face_mesh_pts/", 0) == 0) { // starts with "face_mesh_pts/"
            std::string subkey = key.substr(std::string("face_mesh_pts/").size());
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
    std::vector<std::string> required_keys = {"nose_bridge", "l_tragus", "r_tragus"};

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
    Eigen::Vector3d nose = face_mesh_pts.at("nose_bridge");
    Eigen::Vector3d l_trag = face_mesh_pts.at("l_tragus");
    Eigen::Vector3d r_trag = face_mesh_pts.at("r_tragus");
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
    head_euler = q_rel.toRotationMatrix().eulerAngles(2,1,0); // ZYX order

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
        if (key.rfind("body_pts/", 0) == 0) {
            std::string subkey = key.substr(std::string("body_pts/").size());
            body_pts[subkey] = vec;
        }
    }

    for (const auto& [key, vec] : body_pts) {
    std::cout << key << ": [" 
              << vec.x() << ", " 
              << vec.y() << ", " 
              << vec.z() << "]\n";
    }

    std::vector<std::string> upper_keys = {"l_shoulder", "l_elbow"};
    std::vector<std::string> lower_keys = {"l_elbow", "l_wrist"};
    std::vector<std::string> hip_keys   = {"l_hip", "r_hip"};

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
        std::cout << "All upper keys found" << std::endl;
        Eigen::Vector3d l_shoulder = body_pts.at("l_shoulder");
        Eigen::Vector3d l_elbow = body_pts.at("l_elbow");
        Eigen::Vector3d arm_vec = normalize(l_elbow - l_shoulder);

        double pitch = arm_rotation_angle(arm_vec, torso_x, torso_z) - M_PI/2.0;
        double yaw   = arm_rotation_angle(arm_vec, torso_y, torso_z);
        double roll  = 0.0;

        Eigen::Quaterniond l_arm_upper_rel =
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        if (!has_keys(hip_keys)) {
            Eigen::Vector3d euler = l_arm_upper_rel.toRotationMatrix().eulerAngles(2,1,0);
            l_arm_upper_rel =
                Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitX());
        }

        l_arm_upper_euler = l_arm_upper_rel.toRotationMatrix().eulerAngles(2,1,0);
        l_arm_upper_quat = l_arm_upper_rel;
    }

    // Lower arm
    if (has_keys(lower_keys)) {
        std::cout << "All lower keys found" << std::endl;
        Eigen::Vector3d l_elbow = body_pts.at("l_elbow");
        Eigen::Vector3d l_wrist = body_pts.at("l_wrist");
        Eigen::Vector3d forearm_vec = normalize(l_wrist - l_elbow);

        double pitch = arm_rotation_angle(forearm_vec, torso_x, torso_z) - M_PI/2.0;
        double yaw   = arm_rotation_angle(forearm_vec, torso_y, torso_z);
        double roll  = 0.0; // could add left_hand_roll if needed

        Eigen::Quaterniond l_arm_lower_rel =
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        if (!has_keys(hip_keys)) {
            Eigen::Vector3d euler = l_arm_lower_rel.toRotationMatrix().eulerAngles(2,1,0);
            l_arm_lower_rel =
                Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitX());
        }

        l_arm_lower_euler = l_arm_lower_rel.toRotationMatrix().eulerAngles(2,1,0);
        l_arm_lower_quat = l_arm_lower_rel;
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

    // Extract only body_pts/*
    std::map<std::string, Eigen::Vector3d> body_pts;
    for (auto& [key, vec] : pose_data) {
        if (key.rfind("body_pts/", 0) == 0) {
            std::string subkey = key.substr(std::string("body_pts/").size());
            body_pts[subkey] = vec;
        }
    }

    std::vector<std::string> upper_keys = {"r_shoulder", "r_elbow"};
    std::vector<std::string> lower_keys = {"r_elbow", "r_wrist"};
    std::vector<std::string> hip_keys   = {"l_hip", "r_hip"};

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
        Eigen::Vector3d r_shoulder = body_pts.at("r_shoulder");
        Eigen::Vector3d r_elbow = body_pts.at("r_elbow");
        Eigen::Vector3d arm_vec = normalize(r_elbow - r_shoulder);

        double pitch = arm_rotation_angle(arm_vec, torso_x, torso_z) - M_PI/2.0;
        double yaw   = arm_rotation_angle(arm_vec, torso_y, torso_z);
        double roll  = 0.0;

        Eigen::Quaterniond r_arm_upper_rel =
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        if (!has_keys(hip_keys)) {
            Eigen::Vector3d euler = r_arm_upper_rel.toRotationMatrix().eulerAngles(2,1,0);
            r_arm_upper_rel =
                Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitX());
        }

        r_arm_upper_euler = r_arm_upper_rel.toRotationMatrix().eulerAngles(2,1,0);
        r_arm_upper_quat = r_arm_upper_rel;
    }

    // Lower arm
    if (has_keys(lower_keys)) {
        Eigen::Vector3d r_elbow = body_pts.at("r_elbow");
        Eigen::Vector3d r_wrist = body_pts.at("r_wrist");
        Eigen::Vector3d forearm_vec = normalize(r_wrist - r_elbow);

        double pitch = arm_rotation_angle(forearm_vec, torso_x, torso_z) - M_PI/2.0;
        double yaw   = arm_rotation_angle(forearm_vec, torso_y, torso_z);
        double roll  = 0.0; // could add right_hand_roll if needed

        Eigen::Quaterniond r_arm_lower_rel =
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        if (!has_keys(hip_keys)) {
            Eigen::Vector3d euler = r_arm_lower_rel.toRotationMatrix().eulerAngles(2,1,0);
            r_arm_lower_rel =
                Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitX());
        }

        r_arm_lower_euler = r_arm_lower_rel.toRotationMatrix().eulerAngles(2,1,0);
        r_arm_lower_quat = r_arm_lower_rel;
    }

    angle_map["r_arm_upper"] = r_arm_upper_euler;
    angle_map["r_arm_lower"] = r_arm_lower_euler;
    quat_map["r_arm_upper"]  = r_arm_upper_quat;
    quat_map["r_arm_lower"]  = r_arm_lower_quat;

    return {angle_map, quat_map};
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

    double left_hand_rotation = 0.0;
    auto [left_arm_angles, left_arm_quats] = left_arm_orientation(pose_data,torso_quat,left_hand_rotation);

    // Append to history
    // Merge torso and head maps
    std::map<std::string, Eigen::Vector3d> merged_angles = torso_angles;
    merged_angles.insert(head_angles.begin(), head_angles.end());
    merged_angles.insert(left_arm_angles.begin(), left_arm_angles.end());
    

    std::map<std::string, Eigen::Quaterniond> merged_quats = torso_quats;
    merged_quats.insert(head_quats.begin(), head_quats.end());
    merged_quats.insert(left_arm_quats.begin(), left_arm_quats.end());

    // Append merged maps to history
    kinematic_angles.push_back(merged_angles);
    kinematic_quaternions.push_back(merged_quats);

    // Print merged_angles
    std::cout << "Merged Angles: " << std::endl;
    for (const auto& kv : merged_angles) {
        std::cout << "  " << kv.first << " : ["
                << kv.second.x() << ", "
                << kv.second.y() << ", "
                << kv.second.z() << "]" << std::endl;
    }

    

}

std::map<std::string, double> Kinematics::structure_json_from_kinematics_history(
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