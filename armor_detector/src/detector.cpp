#include "armor_detector/detector.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

namespace armor {
Detector::Detector(
    int binary_threshold,
    int contour_thres,
    Color enemy_color,
    std::string model_path,
    std::string label_path,
    float classifier_threshold,
    const std::array<double, 9>& camera_matrix,
    const std::vector<double>& distortion_coefficients,
    std::vector<std::string> ignore_classes,
    cv::Mat kernel
):
    binary_threshold_(binary_threshold),
    contour_thres_(contour_thres),
    enemy_color_(enemy_color),
    kernel_(kernel) {
    auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
    this->classifier_ = std::make_unique<NumberClassifier>(
        pkg_path + model_path,
        pkg_path + label_path,
        classifier_threshold,
        ignore_classes
    );
    this->pnp_solver_ = std::make_unique<PnPSolver>(camera_matrix, distortion_coefficients);
}

std::vector<Armor> Detector::DetectArmor(const cv::Mat& input) {
    this->preprocessed_image_ = PreprocessImage(input);
    this->lights_ = DetectLight(input);
    this->armors_ = FilterArmor(lights_);

    if (!armors_.empty()) {
        this->classifier_->ExtractNumbers(input, armors_);
        this->classifier_->Classify(armors_);

        for (auto& armor: armors_) {
            this->pnp_solver_->CalculatePose(armor);
        }
    }

    return this->armors_;
}

void Detector::DrawResult(const cv::Mat& input) {
    for (const auto& light: lights_) {
        cv::Point2f vertices[4];
        light.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(input, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::putText(input, std::to_string(light.tilt_angle), light.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    for (const auto& armor: armors_) {
        cv::line(input, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(input, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
        cv::circle(input, armor.center, 3, cv::Scalar(0, 255, 0), 2);
        cv::putText(input, armor.number, armor.center, cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(255, 0, 0), 2);
    }
}

cv::Mat Detector::PreprocessImage(const cv::Mat& input) {
    cv::Mat gray, binary;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, this->binary_threshold_, 255, cv::THRESH_BINARY);
    return binary;
}

std::vector<Light> Detector::DetectLight(const cv::Mat& input) {
    cv::split(input, this->channels_);
    // 敌方颜色通道 - 己方颜色通道
    if (this->enemy_color_ == Color::RED) {
        cv::subtract(this->channels_[2], this->channels_[0], this->color_mask_);
    } else {
        cv::subtract(this->channels_[0], this->channels_[2], this->color_mask_);
    }
    cv::threshold(this->color_mask_, this->light_contour_binary_image_, this->contour_thres_, 255, cv::THRESH_BINARY);

    cv::dilate(this->light_contour_binary_image_, this->light_contour_binary_image_, this->kernel_);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(this->light_contour_binary_image_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::fillPoly(light_contour_binary_image_, contours, cv::Scalar(255));
    }

    cv::bitwise_and(this->preprocessed_image_, this->light_contour_binary_image_, this->light_contour_binary_image_);

    std::vector<Light> lights;
    contours.clear();
    cv::findContours(this->light_contour_binary_image_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour: contours) {
        if (contour.size() < 4) {
            continue;
        }

        Light light(cv::minAreaRect(contour));
        if (IsLight(light) == false) {
            continue;
        }
        lights.push_back(light);
    }

    return lights;
}

std::vector<Armor> Detector::FilterArmor(const std::vector<Light>& lights) {
    std::vector<Armor> armors;
    armors.reserve(lights.size() / 2);
    for (auto& left_light: lights) {
        for (auto& right_light: lights) {
            if (left_light.center.x >= right_light.center.x) {
                continue;
            }
            ArmorType type = CanFormArmor(left_light, right_light);
            if (type != ArmorType::INVALID) {
                armors.push_back(Armor(left_light, right_light, type));
            }
        }
    }
    return armors;
}

bool Detector::IsLight(const Light& light) {
    return (light.length > light.width * 3) && (light.size.area() > 100);
}

ArmorType Detector::CanFormArmor(const Light& left_light, const Light& right_light) {
    // 两灯条的高度比，短 / 长
    float height_ratio = (left_light.length > right_light.length)
        ? (left_light.length / right_light.length)
        : (right_light.length / left_light.length);
    bool height_ratio_valid = height_ratio > 0.8;

    // 两灯条的角度差
    float light_angle_diff = std::abs(left_light.tilt_angle - right_light.tilt_angle);
    bool light_angle_diff_valid = light_angle_diff < 10;

    // 装甲板倾斜角度
    cv::Point2f diff = left_light.center - right_light.center;
    float armor_angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool armor_angle_valid = armor_angle < 30;

    if (height_ratio_valid && light_angle_diff_valid && armor_angle_valid) {
        float average_light_length = (left_light.length + right_light.length) / 2;
        // 两灯条中心点距离 / 平均灯条长度
        float center_distance = cv::norm(left_light.center - right_light.center) / average_light_length;
        return (center_distance > 3.2) ? ArmorType::LARGE : ArmorType::SMALL;
    } else {
        return ArmorType::INVALID;
    }
}

} // namespace armor
