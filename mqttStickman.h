#ifndef MQTTSTICKMAN_H
#define MQTTSTICKMAN_H

#include <string>
#include <vector>
#include "mqtt/client.h"
#include "mqtt/connect_options.h"
#include <nlohmann/json.hpp>
#include "MojoQuaternion.hpp"  // Your quaternion class

class mqttStickman {
public:
    mqttStickman();
    ~mqttStickman();

    // Publish JSON pose to MQTT
    void publishPose(const std::vector<mojo_quaternion::quaternion>& quats);

private:
    const std::string PUB_SERVER_ADDRESS = "tcp://192.168.0.241:1883"; // replace with your server IP
    const std::string PUB_CLIENT_ID = "publisher_client";
    const std::string PUB_TOPIC = "mediapipe/published";

    mqtt::client* pub_client;
};

#endif // MQTTSTICKMAN_H
