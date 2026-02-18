#ifndef POSE_DATA_H
#define POSE_DATA_H

#include <Eigen/Dense>
#include <cstdint>
#include <array>
#include <bitset>

// ============================================================================
// Memory-efficient pose data structure with O(1) indexed access
// Replaces std::map<std::string, Eigen::Vector3d> for better performance
// ============================================================================

// Pose landmark indices - body
enum class PoseLandmark : uint8_t {
    // Body landmarks
    Nose = 0,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
    LeftHeel,
    RightHeel,
    LeftToe,
    RightToe,
    
    // Face mesh landmarks  
    MeshNoseTip,
    MeshLeftEarTragus,
    MeshRightEarTragus,
    
    // Hand landmarks (normalized)
    LeftPalmBase,
    RightPalmBase,
    LeftIndexFingerBase,
    RightIndexFingerBase,
    LeftMiddleFingerBase,
    RightMiddleFingerBase,
    LeftPinkyFingerBase,
    RightPinkyFingerBase,
    // Must be last
    COUNT
};

// Total number of landmarks
constexpr size_t POSE_LANDMARK_COUNT = static_cast<size_t>(PoseLandmark::COUNT);

// ============================================================================
// Compact pose data structure
// ============================================================================
struct PoseData {
    // Fixed-size array of landmarks - cache-friendly contiguous memory
    alignas(32) std::array<Eigen::Vector3d, POSE_LANDMARK_COUNT> landmarks;
    
    // Bitset to track which landmarks are valid/present
    std::bitset<POSE_LANDMARK_COUNT> valid;
    
    // Metadata
    int64_t timestamp_ms = 0;
    int32_t frame_width = 0;
    int32_t frame_height = 0;
    
    // Default constructor - zero-initialize
    PoseData() {
        for (auto& lm : landmarks) {
            lm.setZero();
        }
        valid.reset();
    }
    
    // Fast O(1) indexed access
    inline Eigen::Vector3d& operator[](PoseLandmark idx) {
        return landmarks[static_cast<size_t>(idx)];
    }
    
    inline const Eigen::Vector3d& operator[](PoseLandmark idx) const {
        return landmarks[static_cast<size_t>(idx)];
    }
    
    // Check if landmark is valid
    inline bool has(PoseLandmark idx) const {
        return valid[static_cast<size_t>(idx)];
    }
    
    // Set landmark value and mark as valid
    inline void set(PoseLandmark idx, const Eigen::Vector3d& value) {
        landmarks[static_cast<size_t>(idx)] = value;
        valid.set(static_cast<size_t>(idx));
    }
    
    // Set landmark value and mark as valid (move version)
    inline void set(PoseLandmark idx, Eigen::Vector3d&& value) {
        landmarks[static_cast<size_t>(idx)] = std::move(value);
        valid.set(static_cast<size_t>(idx));
    }
    
    // Get landmark with bounds checking
    inline const Eigen::Vector3d& at(PoseLandmark idx) const {
        if (!valid[static_cast<size_t>(idx)]) {
            throw std::out_of_range("Landmark not valid");
        }
        return landmarks[static_cast<size_t>(idx)];
    }
    
    // Try to get landmark, returns nullptr if invalid
    inline const Eigen::Vector3d* tryGet(PoseLandmark idx) const {
        if (valid[static_cast<size_t>(idx)]) {
            return &landmarks[static_cast<size_t>(idx)];
        }
        return nullptr;
    }
    
    // Clear all data
    inline void clear() {
        valid.reset();
        timestamp_ms = 0;
    }
    
    // Check if we have minimum required landmarks for kinematics
    inline bool hasMinimumBody() const {
        return has(PoseLandmark::LeftShoulder) && 
               has(PoseLandmark::RightShoulder) &&
               has(PoseLandmark::LeftHip) && 
               has(PoseLandmark::RightHip);
    }
    
    inline bool hasLeftArm() const {
        return has(PoseLandmark::LeftShoulder) && 
               has(PoseLandmark::LeftElbow) && 
               has(PoseLandmark::LeftWrist);
    }
    
    inline bool hasRightArm() const {
        return has(PoseLandmark::RightShoulder) && 
               has(PoseLandmark::RightElbow) && 
               has(PoseLandmark::RightWrist);
    }
    
    inline bool hasFaceMesh() const {
        return has(PoseLandmark::MeshNoseTip) && 
               has(PoseLandmark::MeshLeftEarTragus) && 
               has(PoseLandmark::MeshRightEarTragus);
    }
    
    inline bool hasBasicHead() const {
        return has(PoseLandmark::Nose) && 
               has(PoseLandmark::LeftEar) && 
               has(PoseLandmark::RightEar);
    }
};

// ============================================================================
// String to enum mapping for JSON parsing (compile-time hash would be faster)
// Uses a simple switch for the common landmarks
// ============================================================================
inline PoseLandmark stringToPoseLandmark(const std::string& name, bool& found) {
    found = true;
    
    // Fast path: check first character to narrow down
    if (name.empty()) { found = false; return PoseLandmark::Nose; }
    
    switch (name[0]) {
        case 'N':
            if (name == "Nose") return PoseLandmark::Nose;
            break;
        case 'L':
            if (name == "LeftShoulder") return PoseLandmark::LeftShoulder;
            if (name == "LeftElbow") return PoseLandmark::LeftElbow;
            if (name == "LeftWrist") return PoseLandmark::LeftWrist;
            if (name == "LeftHip") return PoseLandmark::LeftHip;
            if (name == "LeftKnee") return PoseLandmark::LeftKnee;
            if (name == "LeftAnkle") return PoseLandmark::LeftAnkle;
            if (name == "LeftHeel") return PoseLandmark::LeftHeel;
            if (name == "LeftToe") return PoseLandmark::LeftToe;
            if (name == "LeftEar") return PoseLandmark::LeftEar;
            break;
        case 'R':
            if (name == "RightShoulder") return PoseLandmark::RightShoulder;
            if (name == "RightElbow") return PoseLandmark::RightElbow;
            if (name == "RightWrist") return PoseLandmark::RightWrist;
            if (name == "RightHip") return PoseLandmark::RightHip;
            if (name == "RightKnee") return PoseLandmark::RightKnee;
            if (name == "RightAnkle") return PoseLandmark::RightAnkle;
            if (name == "RightHeel") return PoseLandmark::RightHeel;
            if (name == "RightToe") return PoseLandmark::RightToe;
            if (name == "RightEar") return PoseLandmark::RightEar;
            break;
        default:
            break;
    }
    
    found = false;
    return PoseLandmark::Nose;
}

// Hand landmark string mapping
inline PoseLandmark handLandmarkToEnum(const std::string& handPrefix, const std::string& landmarkName, bool& found) {
    found = true;
    bool isLeft = (handPrefix == "left");
    
    if (landmarkName == "PalmBase") {
        return isLeft ? PoseLandmark::LeftPalmBase : PoseLandmark::RightPalmBase;
    }
    if (landmarkName == "IndexFingerBase") {
        return isLeft ? PoseLandmark::LeftIndexFingerBase : PoseLandmark::RightIndexFingerBase;
    }
    if (landmarkName == "MiddleFingerBase") {
        return isLeft ? PoseLandmark::LeftMiddleFingerBase : PoseLandmark::RightMiddleFingerBase;
    }
    if (landmarkName == "PinkyFingerBase") {
        return isLeft ? PoseLandmark::LeftPinkyFingerBase : PoseLandmark::RightPinkyFingerBase;
    }
    
    found = false;
    return PoseLandmark::Nose;
}

// Face mesh landmark mapping
inline PoseLandmark faceMeshToEnum(const std::string& name, bool& found) {
    found = true;
    
    if (name == "NoseTip") return PoseLandmark::MeshNoseTip;
    if (name == "LeftEarTragus") return PoseLandmark::MeshLeftEarTragus;
    if (name == "RightEarTragus") return PoseLandmark::MeshRightEarTragus;
    
    found = false;
    return PoseLandmark::Nose;
}

#endif // POSE_DATA_H
