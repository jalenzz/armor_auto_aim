#pragma once

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector.hpp"

#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"

namespace armor {
class ArmorDetectorNode: public rclcpp::Node {
public:
    explicit ArmorDetectorNode(const rclcpp::NodeOptions& options);

private:
    std::unique_ptr<Detector> CreateDetector();

    void CreateDebugPublishers();

    void DestroyDebugPublishers();

    void UpdateDetectorParameters();

    void PublishDebugInfo(const sensor_msgs::msg::Image::SharedPtr& image_msg);

    void PublishArmors(std::vector<Armor>& armors);

    bool debug_;
    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
    rclcpp::Publisher<auto_aim_interfaces::msg::DebugLights>::SharedPtr debug_lights_pub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::DebugArmors>::SharedPtr debug_armors_pub_;
    cv::Mat result_image_;
    image_transport::Publisher binary_img_pub_;
    image_transport::Publisher number_img_pub_;
    image_transport::Publisher result_img_pub_;

    std::unique_ptr<Detector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;
};
} // namespace armor
