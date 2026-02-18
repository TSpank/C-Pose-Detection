#include "mqtt/client.h"
#include "mqtt/connect_options.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <memory>
#include <array>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "kinematic_model.h"
#include "mqttStickman.h"
#include "OneEuroFilter.h"
#include "PoseData.h"

using json = nlohmann::json;

// Fix static linking issue
namespace mqtt {
    const std::string message::EMPTY_STR;
}

// MQTT configuration constants...
const std::string SUB_SERVER_ADDRESS("tcp://207.154.244.181:1883");
const std::string SUB_CLIENT_ID("listener_client");
const std::string SUB_TOPIC("mojo_pose_lm/ten/1/t/1/p/1276/#");
const std::string SUB_USERNAME("scope_mosquitto");
const std::string SUB_PASSWORD("dektzOWb3pmI");

const std::string PUB_SERVER_ADDRESS("tcp://207.154.244.181:1883");
const std::string PUB_CLIENT_ID("publisher_client");
const std::string PUB_TOPIC("mojo/iOS/thomasavatar");
const std::string PUB_USERNAME("scope_mosquitto");
const std::string PUB_PASSWORD("dektzOWb3pmI");

const std::string PUB_SERVER_ADDRESS_python("tcp://localhost:1883");
const std::string PUB_CLIENT_ID_python("publisher_client");
const std::string PUB_TOPIC_python("mojo/iOS/1234");
const std::string PUB_TOPIC_python_quats("mojo/iOS/quats");

// ============================================================================
// FAST extraction: O(1) indexed access, cache-friendly memory layout
// Uses PoseData struct with enum-indexed array instead of string map
// ============================================================================
void extract_pose_data_fast(
    const json& json_data,
    PoseData& pose_data)
{
    // Filters stored per-landmark index for O(1) access
    // Static arrays with default constructor (uses default filter params)
    static std::array<OneEuroFilter, POSE_LANDMARK_COUNT> filters;
    static std::array<OneEuroFilter, POSE_LANDMARK_COUNT> pixel_filters;
    
    // Clear previous data
    pose_data.clear();
    
    // Handle landmarks nested structure
    const json* j = &json_data;
    json nested;
    if (json_data.contains("landmarks")) {
        nested = json_data["landmarks"];
        j = &nested;
    }

    // Extract timestamp and dimensions
    if (j->contains("poseOutput") && (*j)["poseOutput"].contains("timestampMs") && 
        (*j)["poseOutput"]["timestampMs"].is_number()) {
        pose_data.timestamp_ms = (*j)["poseOutput"]["timestampMs"].get<int64_t>();
        if (j->contains("videoData")) {
            pose_data.frame_width = (*j)["videoData"]["width"].get<int32_t>();
            pose_data.frame_height = (*j)["videoData"]["height"].get<int32_t>();
        }
    }
    
    double ts = static_cast<double>(pose_data.timestamp_ms);

    // === Extract hands ===
    if (j->contains("handOutput") && (*j)["handOutput"].contains("hands")) {
        for (auto& [raw_hand_name, landmarks] : (*j)["handOutput"]["hands"].items()) {
            bool isLeft = (raw_hand_name == "Left");
            bool isRight = (raw_hand_name == "Right");
            if (!isLeft && !isRight) continue;
            
            const char* prefix = isLeft ? "left" : "right";

            for (auto& [landmark_name, coords] : landmarks.items()) {
                if (!coords.contains("x") || !coords.contains("y") || !coords.contains("z"))
                    continue;

                bool found = false;
                PoseLandmark idx = handLandmarkToEnum(prefix, landmark_name, found);
                if (!found) continue;

                size_t i = static_cast<size_t>(idx);
                
                // Normalized coordinates (filtered)
                Eigen::Vector3d norm_pt(
                    coords["x"].get<double>()*pose_data.frame_width,
                    coords["y"].get<double>()*pose_data.frame_height,
                    coords["z"].get<double>()*pose_data.frame_width
                );
                pose_data.set(idx, norm_pt);
            }
        }
    }
    // === Extract pose landmarks ===
    if (j->contains("poseOutput") && (*j)["poseOutput"].contains("poses")) {
        for (auto& [name, coords] : (*j)["poseOutput"]["poses"].items()) {
            if (!coords.contains("x") || !coords.contains("y") || 
                !coords.contains("z") || !coords.contains("inFrameLikelihood"))
                continue;

            if (coords["inFrameLikelihood"].get<double>() < 0.75)
                continue;

            bool found = false;
            PoseLandmark idx = stringToPoseLandmark(name, found);
            if (!found) continue;

            size_t i = static_cast<size_t>(idx);
            
            Eigen::Vector3d pt(
                coords["x"].get<double>(),
                coords["y"].get<double>(),
                coords["z"].get<double>()
            );
            
            pose_data.set(idx, filters[i](pt, ts));
        }
    }

    // === Extract face mesh landmarks ===
    if (j->contains("faceOutput") && (*j)["faceOutput"].contains("faces")) {
        for (auto& [name, coords] : (*j)["faceOutput"]["faces"].items()) {
            if (!coords.contains("x") || !coords.contains("y") || !coords.contains("z"))
                continue;

            bool found = false;
            PoseLandmark idx = faceMeshToEnum(name, found);
            if (!found) continue;

            // Face mesh: no filtering, direct assignment
            pose_data.set(idx, Eigen::Vector3d(
                coords["x"].get<double>(),
                coords["y"].get<double>(),
                coords["z"].get<double>()
            ));
        }
    }
}

// ============================================================================
// Adapter: Convert PoseData to std::map for backward compatibility
// Use this during transition period, then refactor Kinematics to use PoseData
// ============================================================================
std::map<std::string, Eigen::Vector3d> poseDataToMap(const PoseData& pd) {
    std::map<std::string, Eigen::Vector3d> result;
    
    // Body landmarks
    if (pd.has(PoseLandmark::Nose)) result["Nose"] = pd[PoseLandmark::Nose];
    if (pd.has(PoseLandmark::LeftEar)) result["LeftEar"] = pd[PoseLandmark::LeftEar];
    if (pd.has(PoseLandmark::RightEar)) result["RightEar"] = pd[PoseLandmark::RightEar];
    if (pd.has(PoseLandmark::LeftShoulder)) result["LeftShoulder"] = pd[PoseLandmark::LeftShoulder];
    if (pd.has(PoseLandmark::RightShoulder)) result["RightShoulder"] = pd[PoseLandmark::RightShoulder];
    if (pd.has(PoseLandmark::LeftElbow)) result["LeftElbow"] = pd[PoseLandmark::LeftElbow];
    if (pd.has(PoseLandmark::RightElbow)) result["RightElbow"] = pd[PoseLandmark::RightElbow];
    if (pd.has(PoseLandmark::LeftWrist)) result["LeftWrist"] = pd[PoseLandmark::LeftWrist];
    if (pd.has(PoseLandmark::RightWrist)) result["RightWrist"] = pd[PoseLandmark::RightWrist];
    if (pd.has(PoseLandmark::LeftHip)) result["LeftHip"] = pd[PoseLandmark::LeftHip];
    if (pd.has(PoseLandmark::RightHip)) result["RightHip"] = pd[PoseLandmark::RightHip];
    if (pd.has(PoseLandmark::LeftKnee)) result["LeftKnee"] = pd[PoseLandmark::LeftKnee];
    if (pd.has(PoseLandmark::RightKnee)) result["RightKnee"] = pd[PoseLandmark::RightKnee];
    if (pd.has(PoseLandmark::LeftAnkle)) result["LeftAnkle"] = pd[PoseLandmark::LeftAnkle];
    if (pd.has(PoseLandmark::RightAnkle)) result["RightAnkle"] = pd[PoseLandmark::RightAnkle];
    
    // Face mesh
    if (pd.has(PoseLandmark::MeshNoseTip)) result["mesh_NoseTip"] = pd[PoseLandmark::MeshNoseTip];
    if (pd.has(PoseLandmark::MeshLeftEarTragus)) result["mesh_LeftEarTragus"] = pd[PoseLandmark::MeshLeftEarTragus];
    if (pd.has(PoseLandmark::MeshRightEarTragus)) result["mesh_RightEarTragus"] = pd[PoseLandmark::MeshRightEarTragus];
    
    // Hand landmarks - Left (normalized)
    if (pd.has(PoseLandmark::LeftPalmBase)) result["leftPalmBase"] = pd[PoseLandmark::LeftPalmBase];
    if (pd.has(PoseLandmark::RightPalmBase)) result["rightPalmBase"] = pd[PoseLandmark::RightPalmBase];
    if (pd.has(PoseLandmark::LeftIndexFingerBase)) result["leftIndexFingerBase"] = pd[PoseLandmark::LeftIndexFingerBase];
    if (pd.has(PoseLandmark::LeftMiddleFingerBase)) result["leftMiddleFingerBase"] = pd[PoseLandmark::LeftMiddleFingerBase];
    if (pd.has(PoseLandmark::LeftPinkyFingerBase)) result["leftPinkyFingerBase"] = pd[PoseLandmark::LeftPinkyFingerBase];
    if (pd.has(PoseLandmark::RightIndexFingerBase)) result["rightIndexFingerBase"] = pd[PoseLandmark::RightIndexFingerBase];
    if (pd.has(PoseLandmark::RightMiddleFingerBase)) result["rightMiddleFingerBase"] = pd[PoseLandmark::RightMiddleFingerBase];
    if (pd.has(PoseLandmark::RightPinkyFingerBase)) result["rightPinkyFingerBase"] = pd[PoseLandmark::RightPinkyFingerBase];

    return result;
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
    
    // Pre-allocate JSON objects to avoid repeated allocations
    json json_angles;
    json json_angles_plane;
    nlohmann::json payload_angles;
    nlohmann::json payload_angles_plane;
    
    // Pre-allocate PoseData struct (reused each frame)
    PoseData fast_pose_data;

    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(1000)) {
        mqtt::const_message_ptr msg(nullptr);
        if (sub_client.try_consume_message(&msg) && msg) {
            try {
                // Parse incoming JSON message
                json json_msg = json::parse(msg->to_string());
                
                // Use fast extraction into indexed struct
                extract_pose_data_fast(json_msg, fast_pose_data);
                
                long long timestamp = fast_pose_data.timestamp_ms;

                // Direct PoseData → Kinematics (no conversion overhead!)
                auto kinematic_output = kinematics.process_kinematics(fast_pose_data);
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
                payload_angles_plane["pose_avatar"] = json_angles;
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
