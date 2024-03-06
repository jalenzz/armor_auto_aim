#include "armor_detector/detector.hpp"

namespace armor {
Detector::Detector(
    int binary_threshold,
    int light_contour_threshold,
    Color enemy_color,
    std::string model_path,
    std::string label_path,
    float confidence_threshold,
    const std::vector<double>& camera_matrix,
    const std::vector<double>& distortion_coefficients,
    std::vector<std::string> ignore_classes,
    cv::Mat kernel
):
    binary_threshold_(binary_threshold),
    light_contour_threshold_(light_contour_threshold),
    enemy_color_(enemy_color),
    kernel_(kernel) {
    this->classifier_ = std::make_unique<NumberClassifier>(model_path, label_path, confidence_threshold, ignore_classes);
    this->pnp_solver_ = std::make_unique<PnPSolver>(camera_matrix, distortion_coefficients);
}

Detector::Detector(
    int binary_threshold,
    int light_contour_threshold,
    Color enemy_color,
    std::unique_ptr<NumberClassifier> classifier,
    std::unique_ptr<PnPSolver> pnp_solver,
    cv::Mat kernel
):
    binary_threshold_(binary_threshold),
    light_contour_threshold_(light_contour_threshold),
    enemy_color_(enemy_color),
    kernel_(kernel),
    classifier_(std::move(classifier)),
    pnp_solver_(std::move(pnp_solver)) {}

std::vector<Armor> Detector::DetectArmor(const cv::Mat& input) {
    preprocessed_image_ = PreprocessImage(input);
    lights_ = DetectLight(input);
    armors_ = FilterArmor(lights_);

    if (!armors_.empty()) {
        classifier_->ExtractNumbers(input, armors_);
        classifier_->Classify(armors_);

        for (auto& armor: armors_) {
            pnp_solver_->CalculatePose(armor);
        }
    }

    return armors_;
}

cv::Mat Detector::PreprocessImage(const cv::Mat& input) {
    cv::Mat gray, binary;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, this->binary_threshold_, 255, cv::THRESH_BINARY);
    return binary;
}

std::vector<Light> Detector::DetectLight(const cv::Mat& input) {
    std::vector<Light> lights;
    debug_lights_.clear();

    cv::split(input, channels_);
    // 敌方颜色通道 - 己方颜色通道
    if (enemy_color_ == Color::RED) {
        cv::subtract(channels_[2], channels_[0], color_mask_);
    } else {
        cv::subtract(channels_[0], channels_[2], color_mask_);
    }
    cv::threshold(color_mask_, light_contour_binary_image_, light_contour_threshold_, 255, cv::THRESH_BINARY);
    cv::dilate(light_contour_binary_image_, light_contour_binary_image_, kernel_);
    cv::bitwise_and(preprocessed_image_, light_contour_binary_image_, light_contour_binary_image_);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(this->light_contour_binary_image_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour: contours) {
        if (contour.size() < 4) {
            continue;
        }

        Light light = FormLight(contour);
        if (light.valid == true) {
            lights.push_back(light);
        }
    }

    return lights;
}

std::vector<Armor> Detector::FilterArmor(const std::vector<Light>& lights) {
    std::vector<Armor> armors;
    armors.reserve(lights.size() / 2);
    debug_armors_.clear();

    for (auto& left_light: lights) {
        for (auto& right_light: lights) {
            if (left_light.center.x >= right_light.center.x) {
                continue;
            }

            Armor armor = FormArmor(left_light, right_light);
            if (armor.type != ArmorType::INVALID) {
                armors.push_back(armor);
            }
        }
    }

    return armors;
}

Light Detector::FormLight(const std::vector<cv::Point>& light_contour) {
    Light light(cv::minAreaRect(light_contour));
    bool is_light = (light.length > light.width * 3) && (light.size.area() > 100);
    light.valid = is_light;

    debug_lights_.push_back(light);
    return light;
}

Armor Detector::FormArmor(const Light& left_light, const Light& right_light) {
    Armor armor(left_light, right_light);
    bool light_height_ratio_valid = armor.light_height_ratio > 0.8;
    bool light_angle_diff_valid = armor.light_angle_diff < 10;
    bool angle_valid = armor.angle < 30;

    if (light_height_ratio_valid && light_angle_diff_valid && angle_valid) {
        armor.type = (armor.light_center_distance > 3.2) ? ArmorType::LARGE : ArmorType::SMALL;
    } else {
        armor.type = ArmorType::INVALID;
    }

    debug_armors_.push_back(armor);
    return armor;
}

cv::Mat Detector::GetAllNumbersImage() {
    if (armors_.empty()) {
        return cv::Mat(cv::Size(20, 28), CV_8UC1);
    } else {
        std::vector<cv::Mat> number_imgs;
        number_imgs.reserve(armors_.size());
        for (auto& armor: armors_) {
            number_imgs.emplace_back(armor.number_image);
        }
        cv::Mat all_num_img;
        cv::vconcat(number_imgs, all_num_img);
        return all_num_img;
    }
}

void Detector::UpdateIgnoreClasses(const std::vector<std::string>& ignore_classes) {
    classifier_->UpdateIgnoreClasses(ignore_classes);
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

} // namespace armor
