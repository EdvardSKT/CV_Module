#include "image_processing.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace {

constexpr float kConfidenceThreshold = 0.45f;
constexpr float kNmsThreshold = 0.45f;
const std::vector<std::string> kClassNames = {
    "Alkalisk Ax2",
    "Alkalisk Ax3",
    "Lithium knappecelle",
    "Super Alkalisk",
};

}  // namespace

PreprocessResult preprocessYOLO(const cv::Mat& image, int input_w, int input_h) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    if (image.channels() != 3) {
        throw std::runtime_error("Expected 3-channel color image");
    }

    // 1. BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    // 2. Compute letterbox resize
    float scale = std::min(
        static_cast<float>(input_w) / static_cast<float>(rgb.cols),
        static_cast<float>(input_h) / static_cast<float>(rgb.rows)
    );

    int new_w = static_cast<int>(std::round(rgb.cols * scale));
    int new_h = static_cast<int>(std::round(rgb.rows * scale));

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 3. Pad to target size
    int pad_x = (input_w - new_w) / 2;
    int pad_y = (input_h - new_h) / 2;
    int pad_right = input_w - new_w - pad_x;
    int pad_bottom = input_h - new_h - pad_y;

    cv::Mat letterboxed;
    cv::copyMakeBorder(
        resized,
        letterboxed,
        pad_y, pad_bottom,
        pad_x, pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    // 4. Convert to float32 and normalize to [0,1]
    cv::Mat float_img;
    letterboxed.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // 5. HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    std::vector<float> tensor(3 * input_w * input_h);
    size_t channel_size = static_cast<size_t>(input_w) * input_h;

    for (int c = 0; c < 3; ++c) {
        std::memcpy(
            tensor.data() + c * channel_size,
            channels[c].data,
            channel_size * sizeof(float)
        );
    }

    return {tensor, scale, pad_x, pad_y};
}

std::vector<Detection> decode_detections(const cv::Mat& output, const PreprocessResult& prep, const cv::Size& image_size) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // The exported YOLO model currently returns a 3D tensor shaped like [1, 8, 8400].
    // We expect: batch=1, 8 values per candidate prediction, and 8400 candidate predictions.
    if (output.dims != 3) {
        return {};
    }

    const int channels = output.size[1];
    const int num_predictions = output.size[2];

    // Each prediction needs at least 4 box values (cx, cy, w, h) and one class score.
    if (channels < 5) {
        return {};
    }

    // Create a 2D view over the raw tensor memory so we can read it as:
    // row = feature channel, column = prediction index.
    const cv::Mat reshaped(channels, num_predictions, CV_32F, const_cast<float*>(output.ptr<float>()));

    for (int i = 0; i < num_predictions; ++i) {
        // YOLO predicts boxes in center format inside the preprocessed model image.
        const float cx = reshaped.at<float>(0, i);
        const float cy = reshaped.at<float>(1, i);
        const float w = reshaped.at<float>(2, i);
        const float h = reshaped.at<float>(3, i);

        int best_class = -1;
        float best_score = 0.0f;

        // Channels 4..end are the class scores. We keep only the strongest class
        // for each candidate prediction before thresholding and NMS.
        for (int c = 4; c < channels; ++c) {
            const float score = reshaped.at<float>(c, i);
            if (score > best_score) {
                best_score = score;
                best_class = c - 4;
            }
        }

        // Drop weak predictions early so they do not clutter later stages.
        if (best_class < 0 || best_score < kConfidenceThreshold) {
            continue;
        }

        // Convert the predicted box from center format to corner format and then
        // undo the letterboxing from preprocessYOLO():
        // 1. subtract the padding added to reach 640x640
        // 2. divide by the resize scale to map back to the original frame
        float x1 = (cx - 0.5f * w - static_cast<float>(prep.pad_x)) / prep.scale;
        float y1 = (cy - 0.5f * h - static_cast<float>(prep.pad_y)) / prep.scale;
        float x2 = (cx + 0.5f * w - static_cast<float>(prep.pad_x)) / prep.scale;
        float y2 = (cy + 0.5f * h - static_cast<float>(prep.pad_y)) / prep.scale;

        // Keep the decoded box inside the image bounds.
        x1 = std::clamp(x1, 0.0f, static_cast<float>(image_size.width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(image_size.height - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(image_size.width - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(image_size.height - 1));

        const int left = static_cast<int>(std::round(x1));
        const int top = static_cast<int>(std::round(y1));
        const int right = static_cast<int>(std::round(x2));
        const int bottom = static_cast<int>(std::round(y2));

        const int width = std::max(0, right - left);
        const int height = std::max(0, bottom - top);
        if (width == 0 || height == 0) {
            continue;
        }

        class_ids.push_back(best_class);
        confidences.push_back(best_score);
        boxes.emplace_back(left, top, width, height);
    }

    // YOLO usually predicts several overlapping boxes for the same object.
    // NMS keeps only the strongest overlapping detections.
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, kConfidenceThreshold, kNmsThreshold, keep);

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (int idx : keep) {
        detections.push_back({class_ids[idx], confidences[idx], boxes[idx]});
    }

    return detections;
}

std::vector<DetectionCenter> postprocess_detection_centers(
    const cv::Mat& output,
    const PreprocessResult& prep,
    const cv::Size& image_size
) {
    const auto detections = decode_detections(output, prep, image_size);

    std::vector<DetectionCenter> centers;
    centers.reserve(detections.size());

    for (const auto& det : detections) {
        const cv::Point center(
            det.box.x + det.box.width / 2,
            det.box.y + det.box.height / 2
        );

        const std::string battery_type =
            det.class_id >= 0 && det.class_id < static_cast<int>(kClassNames.size())
                ? kClassNames[det.class_id]
                : "unknown";

        centers.push_back({center, battery_type, det.confidence});
    }

    return centers;
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);

        const std::string class_name = det.class_id >= 0 && det.class_id < static_cast<int>(kClassNames.size())
            ? kClassNames[det.class_id]
            : "class " + std::to_string(det.class_id);
        const std::string label = class_name + " " + cv::format("%.2f", det.confidence);

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        const int text_top = std::max(0, det.box.y - text_size.height - baseline - 4);
        const cv::Rect text_bg(
            det.box.x,
            text_top,
            text_size.width + 8,
            text_size.height + baseline + 8
        );

        cv::rectangle(image, text_bg, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(
            image,
            label,
            cv::Point(det.box.x + 4, text_bg.y + text_size.height + 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            2
        );
    }
}
