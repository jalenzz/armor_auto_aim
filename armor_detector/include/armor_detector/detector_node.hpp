#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "armor_detector/detector.hpp"

namespace armor {
class ArmorDetectorNode: public rclcpp::Node {
public:
    explicit ArmorDetectorNode(const rclcpp::NodeOptions& options);

private:
    std::unique_ptr<Detector> CreateDetector();

    void CreateDebugPublishers();

    void DestroyDebugPublishers();

    void UpdateDetectorParameters();

    void PublishDebugInfo(std::vector<Armor>& armors);

    void PublishArmors(std::vector<Armor>& armors);

    bool debug_;
    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;

    std::unique_ptr<Detector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};
} // namespace armor
