#include "armor_detector/detector_node.hpp"

namespace armor {
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions& options):
    Node("armor_detector", options) {}
} // namespace armor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(armor::ArmorDetectorNode)
