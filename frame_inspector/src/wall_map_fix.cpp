#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <opencv2/opencv.hpp>
#include <cmath>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;


class WallMapperFixed : public rclcpp::Node {
public:
    WallMapperFixed() : Node("wall_mapper_fixed") {
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rslidar_points", rclcpp::SensorDataQoS(),
            std::bind(&WallMapperFixed::callback, this, std::placeholders::_1));

        map_ = cv::Mat(size_, size_, CV_8UC3, cv::Scalar(255,255,255));
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        Cloud::Ptr cloud(new Cloud);
        pcl::fromROSMsg(*msg, *cloud);
        
        if (cloud->empty()) return;

        Cloud::Ptr clean(new Cloud);
        clean->reserve(cloud->size());
        for (const auto& p : cloud->points) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                clean->points.push_back(p);
            }
        }
        if (clean->empty()) return;

        // Example assumption:
        //  x = forward, y = left/right, z = up/down
        //  Adjust after verifying your actual frame.
        // Range crop in forward direction
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        Cloud::Ptr roi_x(new Cloud);
        pass_x.setInputCloud(clean);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(0.2, 8.0);
        pass_x.filter(*roi_x);
        
        // Height crop
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        Cloud::Ptr roi(new Cloud);
        pass_z.setInputCloud(roi_x);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(0.1, 2.5);
        pass_z.filter(*roi);
        
        if (roi->size() < 100) return;
        
        // Optional downsampling
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        Cloud::Ptr ds(new Cloud);
        
        using Cloud = pcl::PointCloud<pcl::PointXYZ>;

        voxel.setInputCloud(roi);
        voxel.setLeafSize(0.05f, 0.05f, 0.05f);
        voxel.filter(*ds);

        if (ds->size() < 100) return;

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.05);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        Cloud::Ptr remaining(new Cloud(*ds));

        int planes_found = 0;
        while (remaining->size() > 200 && planes_found < 5) {
            pcl::ModelCoefficients coeff;
            pcl::PointIndices inliers;
            seg.setInputCloud(remaining);
            seg.segment(inliers, coeff);

            if (inliers.indices.size() < 150) break;
            if (coeff.values.size() < 4) break;

            float a = coeff.values[0];
            float b = coeff.values[1];
            float c = coeff.values[2];

            // If z is vertical, vertical wall => normal has small z component
            if (std::abs(c) < 0.25f) {
                for (int idx : inliers.indices) {
                    const auto& p = remaining->points[idx];

                    // Top-down map: x forward, y lateral
                    int px = size_/2 + static_cast<int>(p.y / res_);
                    int py = size_ - static_cast<int>(p.x / res_);

                    if (px >= 0 && px < size_ && py >= 0 && py < size_) {
                        map_.at<cv::Vec3b>(py, px) = cv::Vec3b(0, 0, 255);
                    }
                }
                planes_found++;
            }

        extract.setInputCloud(remaining);
        extract.setIndices(pcl::make_shared<pcl::PointIndices>(inliers));
        extract.setNegative(true);
        Cloud tmp;
        extract.filter(tmp);
        *remaining = tmp;
    }

    static int frame = 0;
    if (++frame % 5 == 0) {
        cv::imwrite("map_live_fixed.png", map_);
        RCLCPP_INFO(get_logger(), "Saved map_live_fixed.png");
    }
}

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    cv::Mat map_;
    const float res_ = 0.05f;
    const int size_ = 1200;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WallMapperFixed>());
    rclcpp::shutdown();
    return 0;
}
