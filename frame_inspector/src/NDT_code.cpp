#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cerr << "Usage: ./ndt_demo source.pcd target.pcd\n";
        return -1;
    }

    using PointT = pcl::PointXYZ;

    pcl::PointCloud<PointT>::Ptr source(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr target(new pcl::PointCloud<PointT>());

    if (pcl::io::loadPCDFile<PointT>(argv[1], *source) == -1) {
        std::cerr << "Failed to load source cloud\n";
        return -1;
    }

    if (pcl::io::loadPCDFile<PointT>(argv[2], *target) == -1) {
        std::cerr << "Failed to load target cloud\n";
        return -1;
    }

    // Downsample source
    pcl::PointCloud<PointT>::Ptr filtered_source(new pcl::PointCloud<PointT>());
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setLeafSize(0.2f, 0.2f, 0.2f);
    voxel_grid.setInputCloud(source);
    voxel_grid.filter(*filtered_source);

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);
    ndt.setMaximumIterations(35);

    ndt.setInputSource(filtered_source);
    ndt.setInputTarget(target);

    pcl::PointCloud<PointT>::Ptr output_cloud(new pcl::PointCloud<PointT>());

    Eigen::AngleAxisf init_rotation(0.0f, Eigen::Vector3f::UnitZ());
    Eigen::Translation3f init_translation(0.0f, 0.0f, 0.0f);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

    ndt.align(*output_cloud, init_guess);

    std::cout << "Has converged: " << ndt.hasConverged() << std::endl;
    std::cout << "Fitness score: " << ndt.getFitnessScore() << std::endl;
    std::cout << "Final transformation:\n" << ndt.getFinalTransformation() << std::endl;

    pcl::io::savePCDFileBinary("aligned_ndt.pcd", *output_cloud);
    return 0;
}