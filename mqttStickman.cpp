#include "mqttStickman.h"
#include <iostream>

using json = nlohmann::json;

mqttStickman::mqttStickman() {
    pub_client = new mqtt::client(PUB_SERVER_ADDRESS, PUB_CLIENT_ID);
    mqtt::connect_options pub_connOpts;
    pub_connOpts.set_clean_session(true);

    try {
        std::cout << "Connecting to publisher MQTT broker..." << std::endl;
        pub_client->connect(pub_connOpts);
        std::cout << "Connected to publisher broker!" << std::endl;
    }
    catch (const mqtt::exception& exc) {
        std::cerr << "Publisher connection error: " << exc.what() << std::endl;
    }
}

mqttStickman::~mqttStickman() {
    if (pub_client) {
        pub_client->disconnect();
        delete pub_client;
    }
}

void mqttStickman::publishPose(const std::vector<mojo_quaternion::quaternion>& quats) {
    if (!pub_client) return;

    // Names for each quaternion
    std::vector<std::string> names = {"torso_quat", "head_quat", "l_arm_upper", "l_arm_lower", "r_arm_upper", "r_arm_lower"};
    json j;

    for (size_t i = 0; i < quats.size() && i < names.size(); ++i) {
        j[names[i]] = {
            {"w", quats[i].w},
            {"x", quats[i].x},
            {"y", quats[i].y},
            {"z", quats[i].z}
        };
    }

    std::string json_str = j.dump();

    mqtt::message_ptr pub_msg = mqtt::make_message(PUB_TOPIC, json_str);
    pub_client->publish(pub_msg);

    std::cout << "Published JSON pose: " << json_str << std::endl;
}
