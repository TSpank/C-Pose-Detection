#include "mqtt/client.h"
#include "mqtt/connect_options.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include "kinematic_model.h"
#include "mqttStickman.h"

using json = nlohmann::json;

// Fix static linking issue
namespace mqtt {
    const std::string message::EMPTY_STR;
}

// MQTT configuration constants...
const std::string SUB_SERVER_ADDRESS("tcp://207.154.244.181:1883");
const std::string SUB_CLIENT_ID("listener_client");
const std::string SUB_TOPIC("mojo/ten/1/t/210/p/52/e/0");
const std::string SUB_USERNAME("scope_mosquitto");
const std::string SUB_PASSWORD("dektzOWb3pmI");

const std::string PUB_SERVER_ADDRESS("tcp://207.154.244.181:1883");
const std::string PUB_CLIENT_ID("publisher_client");
const std::string PUB_TOPIC("mojo/iOS/1234");
const std::string PUB_USERNAME("scope_mosquitto");
const std::string PUB_PASSWORD("dektzOWb3pmI");

const std::string PUB_SERVER_ADDRESS_python("tcp://localhost:1883");
const std::string PUB_CLIENT_ID_python("publisher_client");
const std::string PUB_TOPIC_python("mojo/iOS/1234");
const std::string PUB_TOPIC_python_quats("mojo/iOS/quats");

Eigen::Vector3d normalize(const Eigen::Vector3d& v) {
    double norm = v.norm();
    if (norm == 0.0) return Eigen::Vector3d::Zero();
    return v / norm;
}

// -----------------------------------------------------------------------------
// Determine handedness (Left / Right) from nlohmann::json landmarks
// -----------------------------------------------------------------------------
std::string determine_handedness(const json& hand_pts) {
    // Helper lambda to extract 3D point safely
    auto vec3 = [&](const std::string& name) -> Eigen::Vector3d {
        if (!hand_pts.contains(name))
            throw std::invalid_argument("Missing landmark: " + name);
        const auto& pt = hand_pts.at(name);
        return Eigen::Vector3d(pt.at("x").get<double>(),
                               pt.at("y").get<double>(),
                               pt.at("z").get<double>());
    };

    // Extract and convert to your coordinate system (-y, x, z)
    auto transform = [](const Eigen::Vector3d& v) {
        return Eigen::Vector3d(-v(1), v(0), v(2));
    };

    Eigen::Vector3d thumbTip = transform(vec3("ThumbTip"));
    Eigen::Vector3d palmBase = transform(vec3("PalmBase"));
    Eigen::Vector3d palm_vec = normalize(thumbTip - palmBase);

    Eigen::Vector3d indexFingerBase = transform(vec3("IndexFingerBase"));
    Eigen::Vector3d middleFingerBase = transform(vec3("MiddleFingerBase"));

    // Compute reference vectors
    Eigen::Vector3d dir1 = middleFingerBase - indexFingerBase;
    Eigen::Vector3d dir2 = middleFingerBase - palmBase;
    Eigen::Vector3d ref_vec = normalize(dir1.cross(dir2));

    double handedness_value = palm_vec.dot(ref_vec);

    return (handedness_value > 0.0) ? "Left" : "Right";
}

void extract_pose_data(
    const json& j,
    std::map<std::string, Eigen::Vector3d>& pose_data,
    long long& timestamp)
{
    // === Determine input type ===
    std::string inputType = "Camera";
    if (j.contains("inputType")){
        inputType = j["inputType"].get<std::string>();
        //std::cout << "Input type: " << inputType << std::endl;
    }

    // === Extract hands ===
    if (j.contains("handOutput") && j["handOutput"].contains("hands")) {
        for (auto& [raw_hand_name, landmarks] : j["handOutput"]["hands"].items()) {
            // Copy hand name so we can modify it
            std::string hand_name = raw_hand_name;

            if ((hand_name == "Unknown0")||(hand_name == "Unknown1")) {
                hand_name = determine_handedness(landmarks);
                std::cout << "Determined hand as: " << hand_name << std::endl;
            }

            for (auto& [landmark_name, coords] : landmarks.items()) {
                if (coords.contains("x") && coords.contains("y") && coords.contains("z")) {
                    std::string full_name;
                    if (hand_name == "Right")
                        full_name = "right" + landmark_name;
                    else if (hand_name == "Left")
                        full_name = "left" + landmark_name;
                    else
                        continue;

                    if (inputType == "Camera") {
                        pose_data[full_name] = Eigen::Vector3d(
                        -coords["y"].get<double>(),
                        coords["x"].get<double>(),
                        coords["z"].get<double>());
                    }
                    else {
                        pose_data[full_name] = Eigen::Vector3d(
                        coords["x"].get<double>(),
                        coords["y"].get<double>(),
                        coords["z"].get<double>());
                    };        
                    
                }
            }
        }
    }

    // === Extract pose landmarks and timestamp ===
    if (j.contains("poseOutput")) {

        const auto& poseOutput = j["poseOutput"];

        // --- Extract timestamp if present ---
        if (poseOutput.contains("timestampMs") && poseOutput["timestampMs"].is_number()) {
            timestamp = poseOutput["timestampMs"].get<long long>();
            //std::cout << "Pose timestamp: " << timestamp << std::endl;
        }

        // --- Extract poses ---
        if (poseOutput.contains("poses")) {
            for (auto& [name, coords] : poseOutput["poses"].items()) {
                if (coords.contains("x") && coords.contains("y") && coords.contains("z") && coords.contains("inFrameLikelihood")) {
                    if (coords["inFrameLikelihood"].get<double>() < 0.5) {
                        continue; // Skip low likelihood points
                    };
                    if (inputType == "Camera") {
                        pose_data[name] = Eigen::Vector3d(
                        -coords["y"].get<double>(),
                        coords["x"].get<double>(),
                        coords["z"].get<double>());
                    }
                    else {
                        pose_data[name] = Eigen::Vector3d(
                        coords["x"].get<double>(),
                        coords["y"].get<double>(),
                        coords["z"].get<double>());
                    }; 
                }
            }
        }
    }

    // === Extract face landmarks ===
    if (j.contains("faceOutput") && j["faceOutput"].contains("faces")) {
        for (auto& [name, coords] : j["faceOutput"]["faces"].items()) {
            if (coords.contains("x") && coords.contains("y") && coords.contains("z")) {
                if (inputType == "Camera") {
                        pose_data[name] = Eigen::Vector3d(
                        -coords["y"].get<double>(),
                        coords["x"].get<double>(),
                        coords["z"].get<double>());
                    }
                    else {
                        pose_data[name] = Eigen::Vector3d(
                        coords["x"].get<double>(),
                        coords["y"].get<double>(),
                        coords["z"].get<double>());
                    }; 
            }
        }
    }

    if (j.contains("videoData") && j["videoData"].contains("height") && j["videoData"].contains("height")) {
        int width = j["videoData"]["width"].get<int>();
        int height = j["videoData"]["height"].get<int>();
        pose_data["video_width"] = Eigen::Vector3d(static_cast<double>(height), 0.0, 0.0);
        pose_data["video_height"] = Eigen::Vector3d(static_cast<double>(width), 0.0, 0.0);
    }
}


int main() {
    Kinematics kinematics;

     // === Subscriber setup ===
    mqtt::client sub_client(SUB_SERVER_ADDRESS, SUB_CLIENT_ID);
    mqtt::connect_options sub_connOpts;
    sub_connOpts.set_clean_session(true);
    sub_connOpts.set_user_name(SUB_USERNAME);
    sub_connOpts.set_password(SUB_PASSWORD);

    try {
        std::cout << "Connecting to subscriber MQTT broker..." << std::endl;
        sub_client.connect(sub_connOpts);
        std::cout << "Connected to subscriber MQTT broker at " << SUB_SERVER_ADDRESS << std::endl;

        sub_client.subscribe(SUB_TOPIC, 1);
        std::cout << "Subscribed to topic: " << SUB_TOPIC << std::endl;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Subscriber MQTT error: " << exc.what() << std::endl;
        return 1;
    }

    // === Publisher setup ===
    mqtt::client pub_client(PUB_SERVER_ADDRESS, PUB_CLIENT_ID);
    mqtt::connect_options pub_connOpts;
    pub_connOpts.set_clean_session(true);
    pub_connOpts.set_user_name(PUB_USERNAME);
    pub_connOpts.set_password(PUB_PASSWORD);

    try {
        std::cout << "Connecting to publisher MQTT broker..." << std::endl;
        pub_client.connect(pub_connOpts);
        std::cout << "Connected to publisher broker!" << std::endl;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Publisher connection error: " << exc.what() << std::endl;
        return 1;
    }

    // === Publisher setup python ===
    mqtt::client pub_client_python(PUB_SERVER_ADDRESS_python, PUB_CLIENT_ID_python);
    mqtt::connect_options pub_connOpts_python;
    pub_connOpts_python.set_clean_session(true);

    try {
        std::cout << "Connecting to publisher MQTT broker..." << std::endl;
        pub_client_python.connect(pub_connOpts_python);
        std::cout << "Connected to publisher broker!" << std::endl;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Publisher connection error: " << exc.what() << std::endl;
        return 1;
    }

    // Main loop: receive -> compute -> publish
    auto start = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(1000)) {
        mqtt::const_message_ptr msg(nullptr);
        if (sub_client.try_consume_message_for(&msg, std::chrono::milliseconds(100)) && msg) {
            try {
                // === timestamp ===
                long long timestamp = 0;
       
                std::map<std::string, Eigen::Vector3d> pose_data;
                // Parse incoming JSON message
                json json_msg = json::parse(msg->to_string());

                std::cout << json_msg.dump(4) << std::endl;
          

                extract_pose_data(json_msg, pose_data, timestamp);
                auto kinematic_output = kinematics.process_kinematics(pose_data);
                auto angles_map = kinematics.avatar_json(kinematic_output.eulerAngles);
     
                // // Prepare JSON for angles
                json json_angles;
                for (const auto& [key, value] : angles_map) {
                    json_angles[key] = value;
                }
                
                nlohmann::json payload;
                payload["pose"] = json_angles;
                payload["timestamp"] = timestamp; 

                // // Prepare JSON for quats
                // json json_quats;
                // for (const auto& [key, value] : quats_map) {
                //     json_quats[key] = value;
                // }
                
                // nlohmann::json payload_quat;
                // payload_quat = json_quats;
                // payload_quat["timestamp"] = timestamp;

                // // Publish angles as MQTT message
                auto pubmsg = mqtt::make_message(PUB_TOPIC, payload.dump());
                pubmsg->set_qos(1);
                pub_client.publish(pubmsg);
                // pub_client_python.publish(pubmsg);

                // // Publish quats as MQTT message
                // auto pubmsg_quat = mqtt::make_message(PUB_TOPIC_python_quats, payload_quat.dump());
                // pubmsg_quat->set_qos(1);
                // pub_client_python.publish(pubmsg_quat);

                std::cout << "Published: " << PUB_TOPIC << std::endl;

            } catch (const std::exception& e) {
                                                try {
                                                    std::string msg_str = msg ? msg->to_string() : "<no message>";
                                                    std::cerr << "\n===================== EXCEPTION CAUGHT =====================\n";
                                                    std::cerr << "Exception: " << e.what() << "\n";

                                                    // Type information
                                                    std::cerr << "Exception type: " << typeid(e).name() << "\n";

                                                    // If available, show where in code this happened
                                                    // (only works if compiled with -g)
                                                    std::cerr << "Occurred during message processing.\n";

                                                    // Show the message that caused the error
                                                    std::cerr << "Offending MQTT message payload:\n" << msg_str << "\n";

                                                    // If using nlohmann::json, catch parse errors specifically
                                                    if (auto* je = dynamic_cast<const nlohmann::json::parse_error*>(&e)) {
                                                        std::cerr << "JSON Parse error: " << je->what()
                                                                << "\nByte position: " << je->byte << "\n";
                                                    }

                                                    std::cerr << "===========================================================\n\n";
                                                }
                                                catch (...) {
                                                    std::cerr << "⚠️  Exception while handling another exception!\n";
                                                }
                                            }
        }
    }

    // Cleanup
    sub_client.disconnect();
    pub_client.disconnect();
    std::cout << "Disconnected from both brokers. Done." << std::endl;
    return 0;
}
