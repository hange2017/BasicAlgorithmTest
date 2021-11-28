#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace pcl;

int main(int argc,char** argv){


    string file="/home/hange/Learn/BasicAlgorithmTest/chapter3/points.ply";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile(file,*src);

    pcl::visualization::PCLVisualizer::Ptr vie(new pcl::visualization::PCLVisualizer("vie"));
    vie->addPointCloud(src,"cloud");
    vie->spin();







    std::cout<<"check_rslt.cpp"<<std::endl;
    return 0;
}