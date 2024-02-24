#ifndef ARMOR_DETECTOR_DETECTOR_NODE_HPP
#define ARMOR_DETECTOR_DETECTOR_NODE_HPP

#include <rclcpp/rclcpp.hpp>

namespace armor {
class ArmorDetectorNode: public rclcpp::Node {
public:
    explicit ArmorDetectorNode(const rclcpp::NodeOptions& options);

private:
    // ...
};
} // namespace armor

#endif // ARMOR_DETECTOR_DETECTOR_NODE_HPP
