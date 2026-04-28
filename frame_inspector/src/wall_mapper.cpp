#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

class WallMapper : public rclcpp::Node {
public:
    WallMapper() : Node("wall_mapper") {

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", 10,
            std::bind(&WallMapper::callback, this, std::placeholders::_1));

        map_ = cv::Mat(size_, size_, CV_8UC3, cv::Scalar(255,255,255));
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

        Cloud::Ptr cloud(new Cloud);
        pcl::fromROSMsg(*msg, *cloud);

        // Remove NaNs
        Cloud::Ptr clean(new Cloud);
        for (const auto& p : cloud->points) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
                clean->points.push_back(p);
        }

        if (clean->empty()) return;

        // --- centroid normalize
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*clean, centroid);

        for (auto &p : clean->points) {
            p.x -= centroid[0];
            p.y -= centroid[1];
        }

        // --- PassThrough filter (height filter)
        Cloud::Ptr filtered(new Cloud);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(clean);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(0.2, 2.5);
        pass.filter(*filtered);

        if (filtered->size() < 100) return;

        // --- Plane segmentation
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setAxis(Eigen::Vector3f(0,1,0));
        seg.setEpsAngle(10.0 * M_PI / 180.0);
        seg.setDistanceThreshold(0.08);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        Cloud::Ptr remaining(new Cloud(*filtered));

        while (remaining->size() > 500) {
            pcl::ModelCoefficients coeff;
            pcl::PointIndices inliers;

            seg.setInputCloud(remaining);
            seg.segment(inliers, coeff);

            if (inliers.indices.size() < 300) break;

            Eigen::Vector3f normal(coeff.values[0],
                                   coeff.values[1],
                                   coeff.values[2]);

            // --- vertical walls only
            if (std::abs(normal.z()) < 0.2) {
                for (int idx : inliers.indices) {
                    const auto &p = remaining->points[idx];

                    int x = size_/2 - static_cast<int>(p.x / res_);
                    int y = size_/2 + static_cast<int>(p.z / res_);

                    if (x >= 0 && y >= 0 && x < size_ && y < size_)
                        map_.at<cv::Vec3b>(y, x) = {0,0,255};
                }
            }

            // remove plane
            extract.setInputCloud(remaining);
            extract.setIndices(
                pcl::make_shared<pcl::PointIndices>(inliers));
            extract.setNegative(true);

            Cloud tmp;
            extract.filter(tmp);
            *remaining = tmp;
        }

        // --- save occasionally
        static int frame = 0;
        if (++frame % 20 == 0) {
            cv::imwrite("map_live.png", map_);
            RCLCPP_INFO(this->get_logger(), "Saved map_live.png");
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

    cv::Mat map_;
    const float res_ = 0.05f;
    const int size_ = 1200;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WallMapper>());
    rclcpp::shutdown();
    return 0;
}
