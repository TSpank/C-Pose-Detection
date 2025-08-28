#include "mqtt/client.h"
#include "mqtt/connect_options.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include "kinematic_model.h"

using json = nlohmann::json;

json structure_pose_payload(const json& pose_json_angles, bool include_camera = false) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    json pose_data = pose_json_angles;
    pose_data["timestamp"] = ms;

    json payload;
    payload["pose"] = pose_data;

    if (include_camera) {
        json camera_data = {
            {"position", {{"x", -1.0}, {"y", 1.6}, {"z", 0.0}}},
            {"target",   {{"x",  0.0}, {"y", 1.6}, {"z", 0.0}}},
            {"mimic", false},
            {"cam_animation", false}
        };
        payload["camera"] = camera_data;
    }

    return payload;
}


// Fix static linking issue
namespace mqtt {
    const std::string message::EMPTY_STR;
}

// MQTT configuration constants...
const std::string SUB_SERVER_ADDRESS("tcp://localhost:1883");
const std::string SUB_CLIENT_ID("listener_client");
const std::string SUB_TOPIC("mediapipe/data");

const std::string PUB_SERVER_ADDRESS("tcp://207.154.244.181");
const std::string PUB_CLIENT_ID("publisher_client");
const std::string PUB_TOPIC("mojo/iOS/1234");



// Recursive helper to traverse JSON and extract 3-element numeric arrays
void extract_pose_data(
    const json& j,
    const std::string& path,
    std::map<std::string, Eigen::Vector3d>& pose_data)
{
    if (j.is_object()) {
        for (auto& [key, val] : j.items()) {
            std::string new_path = path.empty() ? key : path + "/" + key;

            if (val.is_array() && val.size() == 3
                && val[0].is_number() && val[1].is_number() && val[2].is_number())
            {
                pose_data[new_path] = Eigen::Vector3d(
                    val[0].get<double>(),
                    val[1].get<double>(),
                    val[2].get<double>()
                );
            }
            else {
                extract_pose_data(val, new_path, pose_data);
            }
        }
    }
    else if (j.is_array()) {
        for (size_t i = 0; i < j.size(); ++i) {
            std::string new_path = path + "[" + std::to_string(i) + "]";
            extract_pose_data(j[i], new_path, pose_data);
        }
    }
}

int main() {
    Kinematics kinematics;

    // Subscriber setup
    mqtt::client sub_client(SUB_SERVER_ADDRESS, SUB_CLIENT_ID);
    try {
        mqtt::connect_options sub_connOpts;
        sub_connOpts.set_clean_session(true);
        sub_client.connect(sub_connOpts);
        std::cout << "Connected to subscriber MQTT broker at " << SUB_SERVER_ADDRESS << std::endl;
        sub_client.subscribe(SUB_TOPIC, 1);
        std::cout << "Subscribed to topic: " << SUB_TOPIC << std::endl;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Subscriber MQTT error: " << exc.what() << std::endl;
        return 1;
    }

    // Publisher setup
    mqtt::client pub_client(PUB_SERVER_ADDRESS, PUB_CLIENT_ID);
    mqtt::connect_options pub_connOpts;
    pub_connOpts.set_clean_session(true);
    pub_connOpts.set_user_name("scope_mosquitto");
    pub_connOpts.set_password("dektzOWb3pmI");

    try {
        std::cout << "Connecting to publisher MQTT broker..." << std::endl;
        pub_client.connect(pub_connOpts);
        std::cout << "Connected to publisher broker!" << std::endl;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Publisher connection error: " << exc.what() << std::endl;
        return 1;
    }

    // Main loop: receive -> compute -> publish
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(100)) {
        mqtt::const_message_ptr msg(nullptr);
        if (sub_client.try_consume_message_for(&msg, std::chrono::milliseconds(100)) && msg) {
            try {
                std::map<std::string, Eigen::Vector3d> pose_data;
                json json_msg = json::parse(msg->to_string());

                extract_pose_data(json_msg, "", pose_data);

                // for (const auto& [key, vec] : pose_data) {
                //     std::cout << key << ": ("
                //               << vec.x() << ", "
                //               << vec.y() << ", "
                //               << vec.z() << ")\n";
                // }

                kinematics.kinematics_neck(pose_data);

                auto pose_json_angles = Kinematics::structure_json_from_kinematics_history_angles(
                    kinematics.get_kinematic_angles(),
                    true, true, true);

                auto pose_json_quats = Kinematics::structure_json_from_kinematics_history_quats(
                    kinematics.get_kinematic_quaternions());

                json json_out_angles(pose_json_angles);
                json payload_angles = structure_pose_payload(json_out_angles, false);
                json json_out_quats(pose_json_quats);

                // // // DEBUG: print JSON Angles being published 
                // std::cout << "Publishing JSON angles:\n" << json_out_angles.dump(4) << std::endl;

                // // // DEBUG: print JSON quats being published 
                // std::cout << "Publishing JSON quats:\n" << json_out_quats.dump(4) << std::endl;

                auto pubmsg = mqtt::make_message(PUB_TOPIC, payload_angles.dump());
                pubmsg->set_qos(1);
                pub_client.publish(pubmsg);
                std::cout << "Published processed pose to topic: " << PUB_TOPIC << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "Error processing message: " << e.what() << std::endl;
            }
        }
    }

    // Cleanup
    sub_client.disconnect();
    pub_client.disconnect();
    std::cout << "Disconnected from both brokers. Done." << std::endl;
    return 0;
}
