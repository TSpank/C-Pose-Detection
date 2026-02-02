#include "mqtt/client.h"
#include "mqtt/connect_options.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "kinematic_model.h"
#include "mqttStickman.h"
#include "OneEuroFilter.h"

using json = nlohmann::json;

// Fix static linking issue
namespace mqtt {
    const std::string message::EMPTY_STR;
}

// MQTT configuration constants...
const std::string SUB_SERVER_ADDRESS("tcp://207.154.244.181:1883");
const std::string SUB_CLIENT_ID("listener_client");
const std::string SUB_TOPIC("mojo_pose_lm/ten/1/t/#");
const std::string SUB_USERNAME("scope_mosquitto");
const std::string SUB_PASSWORD("dektzOWb3pmI");

const std::string PUB_SERVER_ADDRESS("tcp://207.154.244.181:1883");
const std::string PUB_CLIENT_ID("publisher_client");
const std::string PUB_TOPIC("mojo/iOS/thomas");
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

struct VideoInfo {
    bool valid = false;      // True if width/height extracted
    std::string inputType;   // "Camera" or other
    int width = 0;
    int height = 0;
};

VideoInfo extract_video_info_once(const json& j)
{
    static bool hasRun = false;
    static VideoInfo info;   // static so it persists after returning

    if (hasRun)
        return info;         // return the same struct every time

    hasRun = true;
    // --- Input Type ---
    if (j.contains("inputType") && j["inputType"].is_string())
        info.inputType = j["inputType"];
    else
        info.inputType = "Camera";

    // --- Video Dimensions ---
    if (j.contains("videoData") &&
        j["videoData"].contains("width") &&
        j["videoData"].contains("height"))
    {
        info.width  = j["videoData"]["width"].get<int>();
        info.height = j["videoData"]["height"].get<int>();
        info.valid = true;
    }

    return info;
}


void extract_pose_data(
    const json& json_data,
    std::map<std::string, Eigen::Vector3d>& pose_data,
    long long& timestamp,
    const VideoInfo vid)
{
    static std::unordered_map<std::string, OneEuroFilter> filters;
    
    // Clear previous data but keep capacity
    pose_data.clear();
    
    // Handle landmarks nested structure
    json j;
    if (json_data.contains("landmarks")) {
        j = json_data["landmarks"];
    } else {
        j = json_data;
    }
    
    // === Extract hands ===
    if (j.contains("handOutput") && j["handOutput"].contains("hands")) {
        for (auto& [raw_hand_name, landmarks] : j["handOutput"]["hands"].items()) {
            // Copy hand name so we can modify it
            std::string hand_name = raw_hand_name;

            for (auto& [landmark_name, coords] : landmarks.items()) {
                if (coords.contains("x") && coords.contains("y") && coords.contains("z")) {
                    std::string full_name;
                    if (hand_name == "Right")
                        full_name = "right" + landmark_name;
                    else if (hand_name == "Left")
                        full_name = "left" + landmark_name;
                    else
                        continue;

                    pose_data[full_name] = Eigen::Vector3d(
                    coords["x"].get<double>(),
                    coords["y"].get<double>(),
                    coords["z"].get<double>());

                    pose_data[full_name+"_pixels"] = Eigen::Vector3d(
                    coords["x"].get<double>()*vid.width,
                    coords["y"].get<double>()*vid.height,
                    coords["z"].get<double>());

                    if (landmark_name == "Wrist") {
                        // Optimize filter lookup
                        const std::string filter_key = full_name+"_pixels";
                        auto filter_it = filters.find(filter_key);
                        if (filter_it == filters.end()) {
                            filter_it = filters.insert({filter_key, OneEuroFilter(30.0, 0.01, 0.001, 1.0)}).first;
                        }
                        // Apply filter using iterator to avoid second lookup
                        pose_data[full_name+"_pixels"] = filter_it->second(pose_data[full_name+"_pixels"], static_cast<double>(timestamp));
                    }
                          
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

                    if (coords["inFrameLikelihood"].get<double>() < 0.8) {
                        continue; // Skip low likelihood points
                    }
                    else {
                        // Point is valid → add and filter it
                        Eigen::Vector3d pt(
                            coords["x"].get<double>(),
                            coords["y"].get<double>(),
                            coords["z"].get<double>()
                        );

                        // Optimize filter lookup
                        auto filter_it = filters.find(name);
                        if (filter_it == filters.end()) {
                            filter_it = filters.insert({name, OneEuroFilter(30.0, 0.01, 0.001, 1.0)}).first;
                        }

                        // Apply filter using iterator to avoid second lookup
                        pose_data[name] = filter_it->second(pt, static_cast<double>(timestamp));
                    } 
                }
            }
        }
    }

    // === Extract face landmarks ===
    if (j.contains("faceOutput") && j["faceOutput"].contains("faces")) {
        for (auto& [name, coords] : j["faceOutput"]["faces"].items()) {
            if (coords.contains("x") && coords.contains("y") && coords.contains("z")) { 
                std::string key = "mesh_" + name;
                pose_data[key] = Eigen::Vector3d(
                coords["x"].get<double>(),
                coords["y"].get<double>(),
                coords["z"].get<double>());        
            }
        }
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

    VideoInfo vid;
    
    // Pre-allocate JSON objects to avoid repeated allocations
    json json_angles;
    json json_angles_plane;
    nlohmann::json payload_angles;
    nlohmann::json payload_angles_plane;

    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(1000)) {
        mqtt::const_message_ptr msg(nullptr);
        if (sub_client.try_consume_message(&msg) && msg) {
            try {
                // === timestamp ===
                long long timestamp = 0;

                std::map<std::string, Eigen::Vector3d> pose_data;

                // Parse incoming JSON message (avoid dumping for debug)
                json json_msg = json::parse(msg->to_string());

                vid = extract_video_info_once(json_msg);
                extract_pose_data(json_msg, pose_data, timestamp, vid);

                auto kinematic_output = kinematics.process_kinematics(pose_data);
                auto angles_map_plane = kinematics.json_isolated_angles(kinematic_output.quaternions,kinematic_output.eulerAngles);
                auto angles_map = kinematics.avatar_json(kinematic_output.eulerAngles);

     
                // // Prepare JSON for angles (reuse pre-allocated objects)
                json_angles.clear();
                for (const auto& [key, value] : angles_map) {
                    json_angles[key] = value;
                }

                

                payload_angles.clear();
                payload_angles["pose"] = json_angles;
                payload_angles["timestamp"] = timestamp; 
                // // Publish angles as MQTT message
                auto pubmsg_angles = mqtt::make_message(PUB_TOPIC, payload_angles.dump());
                pubmsg_angles->set_qos(0); // Changed to QoS 0 for better performance
                
                pub_client.publish(pubmsg_angles);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // // Prepare JSON for angles (reuse pre-allocated objects)
                json_angles_plane.clear();
                for (const auto& [key, value] : angles_map_plane) {
                    json_angles_plane[key] = value;
                }

                payload_angles_plane.clear();
                payload_angles_plane["pose"] = json_angles_plane;
                payload_angles_plane["timestamp"] = timestamp; 
                // // Publish angles as MQTT message
                auto pubmsg_angles_plane = mqtt::make_message(PUB_TOPIC, payload_angles_plane.dump());
                pubmsg_angles_plane->set_qos(0); // Changed to QoS 0 for better performance
                
                pub_client_python.publish(pubmsg_angles_plane);

                std::cout << "Published: " << PUB_TOPIC << std::endl; // Removed for performance

                // publish quats???????????????????????????????????????????????????????????????????????????????????
                std::vector<std::string> joint_names = {
                                        "torso_quat",
                                        "head_quat",
                                        "r_arm_upper",
                                        "r_arm_lower",
                                        "r_arm_lower_global",
                                        "l_arm_upper",
                                        "l_arm_lower",
                                        "l_arm_lower_global"
                                    };
                nlohmann::json msg = kinematics.structure_json_from_quats(joint_names, kinematic_output.quaternions);

                // // Publish angles as MQTT message
                auto pubmsg_quat = mqtt::make_message(PUB_TOPIC_python_quats, msg.dump());
                pubmsg_quat->set_qos(0); // Changed to QoS 0 for better performance
                
                pub_client.publish(pubmsg_quat);
                pub_client_python.publish(pubmsg_quat);

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
