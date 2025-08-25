#include "mqtt/client.h"
#include "mqtt/connect_options.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include "kinematic_model.h"

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

using json = nlohmann::json;

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
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
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

                auto pose_json = Kinematics::structure_json_from_kinematics_history(
                    kinematics.get_kinematic_angles(),
                    true, true, true);

                json json_out(pose_json);
                // // DEBUG: print JSON being published 
                std::cout << "Publishing JSON:\n" << json_out.dump(4) << std::endl;
                auto pubmsg = mqtt::make_message(PUB_TOPIC, json_out.dump());
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
