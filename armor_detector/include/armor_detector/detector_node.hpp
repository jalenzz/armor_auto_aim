#include <rclcpp/rclcpp.hpp>

namespace armor {
class ArmorDetectorNode: public rclcpp::Node {
public:
    explicit ArmorDetectorNode(const rclcpp::NodeOptions& options);

private:
    // ...
};
} // namespace armor
