#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "depthai/depthai.hpp"
#include <opencv2/opencv.hpp>

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

struct Point2D {
    double x;
    double y;
};

double distancePoint(const Point2D& p1, const Point2D& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

Point2D getLandmarkPoint(
    const mediapipe::NormalizedLandmarkList& landmarks,
    int index,
    int width,
    int height
) {
    const auto& lm = landmarks.landmark(index);

    Point2D p;
    p.x = lm.x() * width;
    p.y = lm.y() * height;
    return p;
}

double calculateEAR(
    const mediapipe::NormalizedLandmarkList& landmarks,
    int width,
    int height,
    bool leftEye
) {
    /*
        MediaPipe FaceMesh 눈 landmark 기준

        왼쪽 눈:
        33, 160, 158, 133, 153, 144

        오른쪽 눈:
        362, 385, 387, 263, 373, 380

        EAR = (세로거리1 + 세로거리2) / (2 * 가로거리)
    */

    int p1, p2, p3, p4, p5, p6;

    if (leftEye) {
        p1 = 33;
        p2 = 160;
        p3 = 158;
        p4 = 133;
        p5 = 153;
        p6 = 144;
    }
    else {
        p1 = 362;
        p2 = 385;
        p3 = 387;
        p4 = 263;
        p5 = 373;
        p6 = 380;
    }

    Point2D point1 = getLandmarkPoint(landmarks, p1, width, height);
    Point2D point2 = getLandmarkPoint(landmarks, p2, width, height);
    Point2D point3 = getLandmarkPoint(landmarks, p3, width, height);
    Point2D point4 = getLandmarkPoint(landmarks, p4, width, height);
    Point2D point5 = getLandmarkPoint(landmarks, p5, width, height);
    Point2D point6 = getLandmarkPoint(landmarks, p6, width, height);

    double vertical1 = distancePoint(point2, point6);
    double vertical2 = distancePoint(point3, point5);
    double horizontal = distancePoint(point1, point4);

    if (horizontal <= 0.0) {
        return 0.0;
    }

    return (vertical1 + vertical2) / (2.0 * horizontal);
}

double calculateMAR(
    const mediapipe::NormalizedLandmarkList& landmarks,
    int width,
    int height
) {
    /*
        MediaPipe FaceMesh 입 landmark 기준

        입 가로:
        61 - 291

        입 세로:
        13 - 14
        81 - 178
        312 - 402

        MAR = 평균 세로거리 / 가로거리
    */

    Point2D leftMouth = getLandmarkPoint(landmarks, 61, width, height);
    Point2D rightMouth = getLandmarkPoint(landmarks, 291, width, height);

    Point2D upper1 = getLandmarkPoint(landmarks, 13, width, height);
    Point2D lower1 = getLandmarkPoint(landmarks, 14, width, height);

    Point2D upper2 = getLandmarkPoint(landmarks, 81, width, height);
    Point2D lower2 = getLandmarkPoint(landmarks, 178, width, height);

    Point2D upper3 = getLandmarkPoint(landmarks, 312, width, height);
    Point2D lower3 = getLandmarkPoint(landmarks, 402, width, height);

    double horizontal = distancePoint(leftMouth, rightMouth);

    if (horizontal <= 0.0) {
        return 0.0;
    }

    double vertical1 = distancePoint(upper1, lower1);
    double vertical2 = distancePoint(upper2, lower2);
    double vertical3 = distancePoint(upper3, lower3);

    return (vertical1 + vertical2 + vertical3) / (3.0 * horizontal);
}

std::string toFixedString(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void drawPoint(
    cv::Mat& frame,
    const mediapipe::NormalizedLandmarkList& landmarks,
    int index,
    int width,
    int height,
    const cv::Scalar& color
) {
    Point2D p = getLandmarkPoint(landmarks, index, width, height);

    cv::circle(
        frame,
        cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)),
        2,
        color,
        -1
    );
}

void drawSelectedMediaPipeLandmarks(
    cv::Mat& frame,
    const mediapipe::NormalizedLandmarkList& landmarks,
    int width,
    int height
) {
    // 눈 landmark 표시
    std::vector<int> leftEye = {33, 160, 158, 133, 153, 144};
    std::vector<int> rightEye = {362, 385, 387, 263, 373, 380};

    // 입 landmark 표시
    std::vector<int> mouth = {61, 291, 13, 14, 81, 178, 312, 402};

    for (int idx : leftEye) {
        drawPoint(frame, landmarks, idx, width, height, cv::Scalar(255, 0, 0));
    }

    for (int idx : rightEye) {
        drawPoint(frame, landmarks, idx, width, height, cv::Scalar(255, 0, 0));
    }

    for (int idx : mouth) {
        drawPoint(frame, landmarks, idx, width, height, cv::Scalar(0, 0, 255));
    }
}

int main(int argc, char** argv) {
    try {
        const std::string graphPath =
            "mediapipe/examples/desktop/oakd_face_mesh/oakd_face_mesh_desktop_live.pbtxt";

        std::cout << "======================================" << std::endl;
        std::cout << "OAK-D Pro + MediaPipe FaceMesh Real-Time" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "Graph: " << graphPath << std::endl;
        std::cout << "Press ESC to exit." << std::endl;

        // ================================
        // 1. MediaPipe graph 로드
        // ================================
        std::string graphConfigContents;

        auto fileStatus = mediapipe::file::GetContents(
            graphPath,
            &graphConfigContents
        );

        if (!fileStatus.ok()) {
            std::cerr << "[ERROR] Failed to read graph file: "
                      << graphPath << std::endl;
            std::cerr << fileStatus.message() << std::endl;
            return -1;
        }

        mediapipe::CalculatorGraphConfig graphConfig =
            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
                graphConfigContents
            );

        mediapipe::CalculatorGraph graph;

        auto initStatus = graph.Initialize(graphConfig);

        if (!initStatus.ok()) {
            std::cerr << "[ERROR] Failed to initialize MediaPipe graph."
                      << std::endl;
            std::cerr << initStatus.message() << std::endl;
            return -1;
        }

        // FaceMesh 결과 영상 stream
        auto outputPollerStatus = graph.AddOutputStreamPoller("output_video");

        if (!outputPollerStatus.ok()) {
            std::cerr << "[ERROR] Failed to add output_video poller."
                      << std::endl;
            std::cerr << outputPollerStatus.status().message() << std::endl;
            return -1;
        }

        auto outputPoller = std::move(outputPollerStatus.value());

        // FaceMesh landmark 좌표 stream
        auto landmarkPollerStatus =
            graph.AddOutputStreamPoller("multi_face_landmarks");

        if (!landmarkPollerStatus.ok()) {
            std::cerr << "[ERROR] Failed to add multi_face_landmarks poller."
                      << std::endl;
            std::cerr << landmarkPollerStatus.status().message() << std::endl;
            return -1;
        }

        auto landmarkPoller = std::move(landmarkPollerStatus.value());

        auto startStatus = graph.StartRun({});

        if (!startStatus.ok()) {
            std::cerr << "[ERROR] Failed to start MediaPipe graph."
                      << std::endl;
            std::cerr << startStatus.message() << std::endl;
            return -1;
        }

        std::cout << "[INFO] MediaPipe FaceMesh graph started." << std::endl;

        // ================================
        // 2. DepthAI Pipeline 생성
        // ================================
        dai::Pipeline pipeline;

        auto camRgb = pipeline.create<dai::node::ColorCamera>();
        auto xoutRgb = pipeline.create<dai::node::XLinkOut>();

        xoutRgb->setStreamName("rgb");

        camRgb->setPreviewSize(640, 480);
        camRgb->setResolution(
            dai::ColorCameraProperties::SensorResolution::THE_1080_P
        );

        camRgb->setInterleaved(true);
        camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
        camRgb->setFps(30);

        camRgb->preview.link(xoutRgb->input);

        dai::Device device(pipeline);
        auto rgbQueue = device.getOutputQueue("rgb", 4, false);

        std::cout << "[INFO] OAK-D Pro started." << std::endl;

        const int frameWidth = 640;
        const int frameHeight = 480;

        int64_t timestampUs = 0;
        const int64_t timestampStepUs = 33333;

        auto prevTime = std::chrono::steady_clock::now();
        double fps = 0.0;

        double leftEAR = 0.0;
        double rightEAR = 0.0;
        double ear = 0.0;
        double mar = 0.0;

        // ================================
        // 3. 판단 기준값
        // ================================
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

        auto blinkWindowStart = std::chrono::steady_clock::now();
        const double BLINK_WINDOW_SEC = 10.0;
        double blinkPerMinute = 0.0;

        while (true) {
            // ================================
            // 4. OAK-D 프레임 수신
            // ================================
            auto oakFrame = rgbQueue->get<dai::ImgFrame>();
            std::vector<uint8_t> data = oakFrame->getData();

            if (data.empty()) {
                std::cout << "[WARN] Empty frame data" << std::endl;
                continue;
            }

            cv::Mat bgrFrame(frameHeight, frameWidth, CV_8UC3, data.data());

            cv::Mat rgbFrame;
            cv::cvtColor(bgrFrame, rgbFrame, cv::COLOR_BGR2RGB);

            // ================================
            // 5. cv::Mat → MediaPipe ImageFrame
            // ================================
            auto inputFrame = std::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB,
                rgbFrame.cols,
                rgbFrame.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary
            );

            cv::Mat inputFrameMat =
                mediapipe::formats::MatView(inputFrame.get());

            rgbFrame.copyTo(inputFrameMat);

            auto addPacketStatus = graph.AddPacketToInputStream(
                "input_video",
                mediapipe::Adopt(inputFrame.release())
                    .At(mediapipe::Timestamp(timestampUs))
            );

            if (!addPacketStatus.ok()) {
                std::cerr << "[ERROR] Failed to add packet to input stream."
                          << std::endl;
                std::cerr << addPacketStatus.message() << std::endl;
                break;
            }

            timestampUs += timestampStepUs;

            // ================================
            // 6. MediaPipe output_video 수신
            // ================================
            mediapipe::Packet outputPacket;

            if (!outputPoller.Next(&outputPacket)) {
                continue;
            }

            const mediapipe::ImageFrame& outputFrame =
                outputPacket.Get<mediapipe::ImageFrame>();

            cv::Mat outputRgb =
                mediapipe::formats::MatView(&outputFrame);

            cv::Mat outputBgr;
            cv::cvtColor(outputRgb, outputBgr, cv::COLOR_RGB2BGR);

            // ================================
            // 7. MediaPipe landmark 수신
            // ================================
            mediapipe::Packet landmarkPacket;

            bool hasLandmarks = false;

            if (landmarkPoller.Next(&landmarkPacket)) {
                const auto& multiFaceLandmarks =
                    landmarkPacket.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

                if (!multiFaceLandmarks.empty()) {
                    hasLandmarks = true;

                    const auto& faceLandmarks = multiFaceLandmarks[0];

                    // EAR/MAR 계산
                    leftEAR = calculateEAR(
                        faceLandmarks,
                        frameWidth,
                        frameHeight,
                        true
                    );

                    rightEAR = calculateEAR(
                        faceLandmarks,
                        frameWidth,
                        frameHeight,
                        false
                    );

                    ear = (leftEAR + rightEAR) / 2.0;

                    mar = calculateMAR(
                        faceLandmarks,
                        frameWidth,
                        frameHeight
                    );

                    // 선택 landmark 표시
                    drawSelectedMediaPipeLandmarks(
                        outputBgr,
                        faceLandmarks,
                        frameWidth,
                        frameHeight
                    );

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

            // ================================
            // 9. FPS 계산
            // ================================
            double elapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - prevTime
                ).count();

            if (elapsedMs > 0.0) {
                fps = 1000.0 / elapsedMs;
            }

            prevTime = now;

            // ================================
            // 10. 화면 표시
            // ================================
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
                (eyeState == "CLOSED") ? cv::Scalar(0, 0, 255)
                                        : cv::Scalar(0, 255, 0);

            if (eyeState == "NO FACE") {
                eyeColor = cv::Scalar(255, 255, 255);
            }

            cv::Scalar mouthColor =
                (mouthState == "OPEN") ? cv::Scalar(0, 0, 255)
                                        : cv::Scalar(0, 255, 0);

            if (mouthState == "NO FACE") {
                mouthColor = cv::Scalar(255, 255, 255);
            }

            cv::putText(
                outputBgr,
                "Eye: " + eyeState,
                cv::Point(20, 150),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                eyeColor,
                2
            );

            cv::putText(
                outputBgr,
                "Mouth: " + mouthState,
                cv::Point(20, 180),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                mouthColor,
                2
            );

            cv::putText(
                outputBgr,
                "Blink Count: " + std::to_string(blinkCount),
                cv::Point(20, 210),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(255, 255, 255),
                2
            );

            cv::putText(
                outputBgr,
                "Blink/min: " + toFixedString(blinkPerMinute, 1),
                cv::Point(20, 240),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(255, 255, 255),
                2
            );

            cv::Scalar driverColor;

            if (driverState == "NORMAL") {
                driverColor = cv::Scalar(0, 255, 0);
            }
            else if (driverState == "NO FACE") {
                driverColor = cv::Scalar(255, 255, 255);
            }
            else {
                driverColor = cv::Scalar(0, 0, 255);
            }

            cv::putText(
                outputBgr,
                "Driver: " + driverState,
                cv::Point(20, 275),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                driverColor,
                2
            );

            cv::putText(
                outputBgr,
                "EyeClosedCnt: " + std::to_string(eyeClosedCount) +
                    "  MouthOpenCnt: " + std::to_string(mouthOpenCount),
                cv::Point(20, 305),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(255, 255, 255),
                2
            );

            cv::imshow(
                "OAK-D Pro MediaPipe FaceMesh EAR MAR Driver State",
                outputBgr
            );

            int key = cv::waitKey(1);

            if (key == 27) {
                break;
            }
        }

        graph.CloseInputStream("input_video");
        graph.WaitUntilDone();

        cv::destroyAllWindows();

        std::cout << "[INFO] Program finished." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << e.what() << std::endl;
        return -1;
    }

    return 0;