#ifndef ARMOR_DETECTOR_DETECTOR_HPP
#define ARMOR_DETECTOR_DETECTOR_HPP

#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"

namespace armor {

class Detector {
public:
    Detector(
        int binary_threshold,
        int contour_thres,
        Color enemy_color,
        std::string model_path,
        std::string label_path,
        float classifier_threshold,
        std::vector<std::string> ignore_classes = {},
        cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U)
    );

    /**
     * @brief 对输入图片进行装甲板检测
     * @param input 输入图片
     * @return 装甲板的集合
     */
    std::vector<Armor> DetectArmor(const cv::Mat& input);

    /**
     * @brief 在输入图片上绘制装甲板和灯条
     * @param input 输入图片
     */
    void DrawResult(const cv::Mat& input);

private:
    /**
     * @brief 对输入图片进行预处理
     * @param input 输入图片
     * @return 预处理后的图片
     */
    cv::Mat PreprocessImage(const cv::Mat& input);

    /**
     * @brief 从预处理后的图片中检测灯条
     * @param input 预处理后的图片
     * @return 灯条的集合
     */
    std::vector<Light> DetectLight(const cv::Mat& input);

    /**
     * @brief 从灯条集合中筛选出装甲板
     * @param lights 灯条集合
     * @return 装甲板集合
     */
    std::vector<Armor> FilterArmor(const std::vector<Light>& lights);

    /**
     * @brief 判断是否为灯条
     * @param light 待判断的灯条
     * @return 是否为灯条
     */
    bool IsLight(const Light& light);

    /**
     * @brief 判断两个灯条是否能够组成装甲板
     * @param left_light 左灯条
     * @param right_light 右灯条
     * @return 是否能够组成装甲板的类型
     */
    ArmorType CanFormArmor(const Light& left_light, const Light& right_light);

    cv::Mat preprocessed_image_;         // 预处理后的图片
    cv::Mat channels_[3];                // 通道相减模式下的三通道图
    cv::Mat color_mask_;                 // 敌方颜色通道 - 己方颜色通道后的图
    cv::Mat light_contour_binary_image_; // 灯条轮廓为白的二值图
    std::vector<Light> lights_;          // 灯条集合
    std::vector<Armor> armors_;          // 装甲板集合

    int binary_threshold_; // 原图二值化阈值
    int contour_thres_;    // 对通道相减后的灯条轮廓图进行二值化的阈值
    Color enemy_color_;    // 敌方颜色
    cv::Mat kernel_;       // 膨胀核

    std::unique_ptr<NumberClassifier> classifier_;
};

} // namespace armor

#endif // ARMOR_DETECTOR_DETECTOR_HPP_
