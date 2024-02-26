#include <geometry_msgs/msg/pose.hpp>

#include "armor_detector/armor.hpp"

namespace armor {

class PnPSolver {
public:
    explicit PnPSolver(
        const std::array<double, 9>& camera_matrix,
        const std::vector<double>& distortion_coefficients
    );

    void SolvePnP(const Armor& armor);

    /**
     * @brief 计算相机坐标系到装甲板坐标系的旋转矩阵
     * @param pose
     */
    void CalculatePose(Armor& armor);

private:
    cv::Mat camera_matrix_;
    cv::Mat distortion_coefficients_;
    cv::Mat rvec_;
    cv::Mat tvec_;
};

} // namespace armor
