#pragma once

#include <opencv2/opencv.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace armor {

enum class Color {
    RED,
    BLUE
};

enum class ArmorType {
    SMALL,
    LARGE,
    INVALID
};

// clang-format off
const float SMALL_ARMOR_WIDTH  = 0.135;
const float SMALL_ARMOR_HEIGHT = 0.055;
const float LARGE_ARMOR_WIDTH  = 0.225;
const float LARGE_ARMOR_HEIGHT = 0.055;
// x 垂直装甲板向内，从左下角开始顺时针
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS = {
    { 0, +SMALL_ARMOR_WIDTH / 2, -SMALL_ARMOR_HEIGHT / 2},
    { 0, +SMALL_ARMOR_WIDTH / 2, +SMALL_ARMOR_HEIGHT / 2},
    { 0, -SMALL_ARMOR_WIDTH / 2, +SMALL_ARMOR_HEIGHT / 2},
    { 0, -SMALL_ARMOR_WIDTH / 2, -SMALL_ARMOR_HEIGHT / 2}
};
// x 垂直装甲板向内，从左下角开始顺时针
const std::vector<cv::Point3f> LARGE_ARMOR_POINTS = {
    { 0, +LARGE_ARMOR_WIDTH / 2, -LARGE_ARMOR_HEIGHT / 2 },
    { 0, +LARGE_ARMOR_WIDTH / 2, +LARGE_ARMOR_HEIGHT / 2 },
    { 0, -LARGE_ARMOR_WIDTH / 2, +LARGE_ARMOR_HEIGHT / 2 },
    { 0, -LARGE_ARMOR_WIDTH / 2, -LARGE_ARMOR_HEIGHT / 2 }
};
// clang-format on

// 灯条参数
struct LightParams {
    // 宽高比范围 width / height
    float min_ratio, max_ratio;
};

// 装甲板参数
struct ArmorParams {
    // 左右灯条比例最小值
    float min_light_ratio;
    // 大小装甲板两灯条之间距离 / 灯条平均长度 的阈值
    float min_small_center_distance, max_small_center_distance, min_large_center_distance, max_large_center_distance;
    // 装甲板水平角度
    float max_angle;
};

struct Light: public cv::RotatedRect {
    Light() = default;
    explicit Light(const cv::RotatedRect& light_box):
        cv::RotatedRect(light_box) {
        // 排序灯条四个角点，左上、右上、右下、左下
        light_box.points(point);
        std::sort(point, point + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

        top = (point[0] + point[1]) / 2;
        bottom = (point[2] + point[3]) / 2;
        length = cv::norm(top - bottom);
        width = cv::norm(point[0] - point[1]);
        tilt_angle = std::atan2(top.x - bottom.x, top.y - bottom.y) / CV_PI * 180;
    }

    cv::Point2f point[4];    // 灯条四个角点坐标
    cv::Point2f top, bottom; // 灯条上下边框中点坐标
    float length;            // 灯条长度
    float width;             // 灯条宽度
    float tilt_angle;        // 灯条倾斜角度，相对于垂直面，向右倾斜为正
};

struct Armor {
    Armor() = default;
    explicit Armor(const Light& left_light, const Light& right_light, ArmorType type) {
        this->type = type;
        this->left_light = left_light;
        this->right_light = right_light;
        this->center = (left_light.center + right_light.center) / 2;
    }

    Light left_light, right_light;
    ArmorType type;
    cv::Point2f center;

    cv::Mat number_image;
    std::string number;
    float classification_confidence;
    std::string classification_result;

    geometry_msgs::msg::Pose pose;
    float distance_to_image_center;
};

} // namespace armor
