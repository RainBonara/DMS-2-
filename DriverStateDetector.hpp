#pragma once
#include <chrono>
#include <string>

struct DriverStateDetector {
    const double EAR_THRESHOLD = 0.22;
    const double MAR_THRESHOLD = 0.55;
    std::string eyeState = "UNKNOWN";
    std::string mouthState = "UNKNOWN";
    std::string driverState = "NORMAL";
    // 눈 깜빡임 판단용
    int eyeClosedCount = 0;
    const int EYE_CLOSED_FRAME_THRESHOLD = 2;

    // 졸음 판단용: 눈 감김이 길게 유지될 때
    const int DROWSY_EYE_FRAME_THRESHOLD = 15;

    // 하품 판단용: 입 벌림이 길게 유지될 때
    int mouthOpenCount = 0;
    const int MOUTH_OPEN_FRAME_THRESHOLD = 10;

    // 눈 깜빡임 빈도 계산용
    int blinkCount = 0;
    bool prevEyeClosed = false;

    std::chrono::steady_clock::time_point blinkWindowStart = std::chrono::steady_clock::now();
    const double BLINK_WINDOW_SEC = 10.0;
    double blinkPerMinute = 0.0;

    void update(double ear, double mar, bool hasLandmarks) {
        if (hasLandmarks) {
        // ================================
        // 눈 상태 판단
        // ================================
        bool currentEyeClosed = false;

        if (ear < EAR_THRESHOLD) {
            eyeClosedCount++;
        }
        else {
            eyeClosedCount = 0;
        }

        if (eyeClosedCount >= EYE_CLOSED_FRAME_THRESHOLD) {
            currentEyeClosed = true;
            eyeState = "CLOSED";
        }
        else {
            currentEyeClosed = false;
            eyeState = "OPEN";
        }

        // blink count: OPEN → CLOSED로 바뀌는 순간 카운트
        if (!prevEyeClosed && currentEyeClosed) {
            blinkCount++;
        }

        prevEyeClosed = currentEyeClosed;

        // ================================
        // 입 상태 판단
        // ================================
        if (mar > MAR_THRESHOLD) {
            mouthOpenCount++;
            mouthState = "OPEN";
        }
        else {
            mouthOpenCount = 0;
            mouthState = "NORMAL";
        }

        // ================================
        // Driver 최종 상태 판단
        // ================================
        bool longEyeClosed =
            eyeClosedCount >= DROWSY_EYE_FRAME_THRESHOLD;

        bool longMouthOpen =
            mouthOpenCount >= MOUTH_OPEN_FRAME_THRESHOLD;

        if (longEyeClosed && longMouthOpen) {
            driverState = "DROWSY WARNING";
        }
        else if (longEyeClosed) {
            driverState = "EYE CLOSED";
        }
        else if (longMouthOpen) {
            driverState = "YAWNING";
        }
        else {
            driverState = "NORMAL";
        }
        }
        if (!hasLandmarks) {
            eyeState = "NO FACE";
            mouthState = "NO FACE";
            driverState = "NO FACE";

            eyeClosedCount = 0;
            mouthOpenCount = 0;
            prevEyeClosed = false;
        }

        // ================================
        // 8. blink frequency 계산
        // ================================
        auto now = std::chrono::steady_clock::now();

        double blinkWindowElapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - blinkWindowStart
            ).count() / 1000.0;

        if (blinkWindowElapsed >= BLINK_WINDOW_SEC) {
            blinkPerMinute =
                blinkCount * (60.0 / BLINK_WINDOW_SEC);

            blinkCount = 0;
            blinkWindowStart = now;
        }


    }
};
