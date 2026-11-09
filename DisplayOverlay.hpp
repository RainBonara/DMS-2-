#pragma once
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>
#include "DriverStateDetector.hpp"

inline std::string toFixedString(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

inline void drawOverlay(cv::Mat& outputBgr, const DriverStateDetector& state,
                        double fps, double ear, double leftEAR, double rightEAR, double mar) {
    cv::putText(
        outputBgr,
        "OAK-D + MediaPipe FaceMesh Real-Time",
        cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 255),
        2
    );

    cv::putText(
        outputBgr,
        "FPS: " + std::to_string(static_cast<int>(fps)),
        cv::Point(20, 60),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(255, 255, 0),
        2
    );

    cv::putText(
        outputBgr,
        "EAR: " + toFixedString(ear, 2) +
            "  L: " + toFixedString(leftEAR, 2) +
            "  R: " + toFixedString(rightEAR, 2),
        cv::Point(20, 90),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::putText(
        outputBgr,
        "MAR: " + toFixedString(mar, 2),
        cv::Point(20, 120),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::Scalar eyeColor =
        (state.eyeState == "CLOSED") ? cv::Scalar(0, 0, 255)
                                : cv::Scalar(0, 255, 0);

    if (state.eyeState == "NO FACE") {
        eyeColor = cv::Scalar(255, 255, 255);
    }

    cv::Scalar mouthColor =
        (state.mouthState == "OPEN") ? cv::Scalar(0, 0, 255)
                                : cv::Scalar(0, 255, 0);

    if (state.mouthState == "NO FACE") {
        mouthColor = cv::Scalar(255, 255, 255);
    }

    cv::putText(
        outputBgr,
        "Eye: " + state.eyeState,
        cv::Point(20, 150),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        eyeColor,
        2
    );

    cv::putText(
        outputBgr,
        "Mouth: " + state.mouthState,
        cv::Point(20, 180),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        mouthColor,
        2
    );

    cv::putText(
        outputBgr,
        "Blink Count: " + std::to_string(state.blinkCount),
        cv::Point(20, 210),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::putText(
        outputBgr,
        "Blink/min: " + toFixedString(state.blinkPerMinute, 1),
        cv::Point(20, 240),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::Scalar driverColor;

    if (state.driverState == "NORMAL") {
        driverColor = cv::Scalar(0, 255, 0);
    }
    else if (state.driverState == "NO FACE") {
        driverColor = cv::Scalar(255, 255, 255);
    }
    else {
        driverColor = cv::Scalar(0, 0, 255);
    }

    cv::putText(
        outputBgr,
        "Driver: " + state.driverState,
        cv::Point(20, 275),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        driverColor,
        2
    );

    cv::putText(
        outputBgr,
        "EyeClosedCnt: " + std::to_string(state.eyeClosedCount) +
            "  MouthOpenCnt: " + std::to_string(state.mouthOpenCount),
        cv::Point(20, 305),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        2
    );

}
