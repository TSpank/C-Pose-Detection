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

// Subscriber MQTT
const std::string SUB_SERVER_ADDRESS("tcp://localhost:1883");
const std::string SUB_CLIENT_ID("listener_client");
const std::string SUB_TOPIC("mediapipe/data");

// Publisher MQTT
const std::string PUB_SERVER_ADDRESS("tcp://207.154.244.181");
const std::string PUB_CLIENT_ID("publisher_client");
const std::string PUB_TOPIC("mojo/iOS/1234");

int main() {
    Kinematics kinematics;

    // ------------------------
    // Subscriber setup
    // ------------------------
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

    // ------------------------
    // Publisher setup
    // ------------------------
    mqtt::client pub_client(PUB_SERVER_ADDRESS, PUB_CLIENT_ID);

    mqtt::connect_options pub_connOpts;
    pub_connOpts.set_clean_session(true);
    pub_connOpts.set_user_name("scope_mosquitto");   // <-- your username
    pub_connOpts.set_password("dektzOWb3pmI");       // <-- your password

    try {
        std::cout << "Connecting to publisher MQTT broker..." << std::endl;
        pub_client.connect(pub_connOpts);
        std::cout << "Connected to publisher broker!" << std::endl;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Publisher connection error: " << exc.what() << std::endl;
        return 1;
    }

    // ------------------------
    // Main loop: receive -> compute -> publish
    // ------------------------
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
        mqtt::const_message_ptr msg = nullptr;
        if (sub_client.try_consume_message_for(&msg, std::chrono::milliseconds(100)) && msg) {
            try {
                // Parse incoming JSON
                std::map<std::string, Eigen::Vector3d> pose_data;
                auto json_msg = nlohmann::json::parse(msg->to_string());

                for (auto& [top_key, sub_obj] : json_msg.items()) {
                    if (sub_obj.is_object()) {
                        for (auto& [subkey, val] : sub_obj.items()) {
                            if (val.is_array() && val.size() == 3) {
                                std::string combined_key = top_key + "/" + subkey;
                                pose_data[combined_key] = Eigen::Vector3d(
                                    val[0].get<double>(),
                                    val[1].get<double>(),
                                    val[2].get<double>()
                                );
                            }
                        }
                    }
                }

                // Compute kinematics
                kinematics.kinematics_neck(pose_data);

                // Generate JSON for publishing
                std::map<std::string, double> pose_json = Kinematics::structure_json_from_kinematics_history(
                    kinematics.get_kinematic_angles(),
                    true,  // torso_valid
                    true,  // right_hand
                    true   // arms
                );

                // Publish to second MQTT
                nlohmann::json json_out(pose_json);

                // DEBUG: print JSON being published
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

    // ------------------------
    // Cleanup
    // ------------------------
    sub_client.disconnect();
    pub_client.disconnect();
    std::cout << "Disconnected from both brokers. Done." << std::endl;

    return 0;
}
