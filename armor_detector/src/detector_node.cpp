#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_node.hpp"

namespace armor {
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions& options):
    Node("armor_detector", options) {
    detector_ = CreateDetector();

    // 监视 Debug 参数变化
    debug_ = declare_parameter("debug", false);
    if (debug_) {
        CreateDebugPublishers();
    }
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ = debug_param_sub_->add_parameter_callback(
        "debug",
        [this](const rclcpp::Parameter& p) {
            debug_ = p.as_bool();
            debug_ ? CreateDebugPublishers() : DestroyDebugPublishers();
        }
    );

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "image",
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
            auto&& raw_image = cv::Mat(msg->height, msg->width, CV_8UC3, msg->data.data());
            std::vector<Armor> armors;

            if (debug_) {
                UpdateDetectorParameters();
                armors = detector_->DetectArmor(raw_image);
                detector_->DrawResult(raw_image);
                PublishDebugInfo(armors);
            } else {
                armors = detector_->DetectArmor(raw_image);
            }

            PublishArmors(armors);
        }
    );
}

std::unique_ptr<Detector> ArmorDetectorNode::CreateDetector() {
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].step = 1;
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value = 255;
    auto&& binary_threshold = declare_parameter("binary_threshold", 100, param_desc);
    auto&& light_contour_threshold = declare_parameter("light_contour_threshold", 100, param_desc);
    auto&& enemy_color = declare_parameter("enemy_color", 'r');
    auto&& confidence_threshold = declare_parameter("confidence_threshold", 0.7);
    auto&& camera_matrix = declare_parameter("camera_matrix", std::vector<double> { 1302.9388992859376, 0, 609.2298064340857, 0, 2515.6912302455735, 467.0345949712323, 0, 0, 1 });
    auto&& pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
    auto&& model_path = declare_parameter("model_path", "/model/mlp.onnx");
    auto&& label_path = declare_parameter("label_path", "/model/label.txt");
    auto&& distortion_coefficients = declare_parameter("distortion_coefficients", std::vector<double> { 0.9716178021093913, -22.20834732244382, -0.19838225091062828, -0.08828110807170159, 96.16902256363146 });
    auto&& ignore_classes = declare_parameter("ignore_classes", std::vector<std::string> { "negative" });

    return std::make_unique<Detector>(
        binary_threshold,
        light_contour_threshold,
        enemy_color == 'b' ? Color::BLUE : Color::RED,
        pkg_path + model_path,
        pkg_path + label_path,
        confidence_threshold,
        camera_matrix,
        distortion_coefficients,
        ignore_classes
    );
}

void ArmorDetectorNode::CreateDebugPublishers() {}

void ArmorDetectorNode::DestroyDebugPublishers() {}

void ArmorDetectorNode::UpdateDetectorParameters() {}

void ArmorDetectorNode::PublishDebugInfo(std::vector<Armor>& armors) {}

void ArmorDetectorNode::PublishArmors(std::vector<Armor>& armors) {}

} // namespace armor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(armor::ArmorDetectorNode)
