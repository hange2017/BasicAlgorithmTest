#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>
//#include <Eigen/SVD>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <g2o/core/auto_differentiation.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/stuff/sampler.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace g2o;

G2O_USE_OPTIMIZATION_LIBRARY(dense);

void load_data(string& file,vector<Eigen::Vector3d>& points_3d,vector<Eigen::Vector2d>& points_2d_0,
    vector<Eigen::Vector2d>& points_2d_1){

    std::ifstream my_in;
    //my_in.open("/home/hange/Learn/BasicAlgorithmTest/chapter2/test_ba.txt",std::ifstream::in);
    my_in.open(file,std::ifstream::in);

    std::string line;
    int linecount=0;
    int total_point=0;//total pnp 3d points num
    //vector<Eigen::Vector3d> points_3d;//3d points
    //vector<Eigen::Vector2d> points_2d_0;//image1/pose1 2d points/observation
    //vector<Eigen::Vector2d> points_2d_1;//image2/pose2 2d points/observation
    while(std::getline(my_in,line)){
        std::stringstream ss(line);
        if(linecount==3){
            string tmp;
            ss>>tmp;
            ss>>total_point;
        }else if(linecount>3 && linecount<=3+total_point){
            Eigen::Vector3d tmp;
            ss>>tmp(0);ss>>tmp(1);ss>>tmp(2);
            points_3d.push_back(tmp);
        }else if(linecount> 3+total_point+1){
            double tmp=0;
            ss>>tmp;
            if(tmp == 0){
                ss>>tmp;
                Eigen::Vector2d point2d;
                ss>>point2d(0);ss>>point2d(1);
                points_2d_0.push_back(point2d);
            }else{
                ss>>tmp;
                Eigen::Vector2d point2d;
                ss>>point2d(0);ss>>point2d(1);
                points_2d_1.push_back(point2d);
            }
        }
        
        ++linecount;
    }
    return;
}

void myPNP(vector<Eigen::Vector3d>& points_3d,vector<Eigen::Vector2d>& points_2d_0,
        Eigen::Matrix3d& K,Eigen::Matrix3d& R,Eigen::Vector3d& t){

    //构建pnp的系数矩阵
    Eigen::MatrixXd A=Eigen::MatrixXd::Zero(points_3d.size()*2,12);
    for(int i=0;i<points_3d.size();i++){
        //i*2
        A(i*2,0) = points_3d[i](0);
        A(i*2,1) = points_3d[i](1);
        A(i*2,2) = points_3d[i](2);
        A(i*2,3) = 1.0;
        A(i*2,8) = -points_3d[i](0)*points_2d_0[i](0);
        A(i*2,9) = -points_3d[i](1)*points_2d_0[i](0);
        A(i*2,10) = -points_3d[i](2)*points_2d_0[i](0);
        A(i*2,11) = -1.0*points_2d_0[i](0);
        //i*2+1
        A(i*2+1,4) = points_3d[i](0);
        A(i*2+1,5) = points_3d[i](1);
        A(i*2+1,6) = points_3d[i](2);
        A(i*2+1,7) = 1.0;
        A(i*2+1,8) = -points_3d[i](0)*points_2d_0[i](1);
        A(i*2+1,9) = -points_3d[i](1)*points_2d_0[i](1);
        A(i*2+1,10) = -points_3d[i](2)*points_2d_0[i](1);
        A(i*2+1,11) = -1.0*points_2d_0[i](1);
    }

    //SVD分解pnp的系数矩阵A，并获得最小二乘解
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A,ComputeFullU | ComputeFullV);
    Eigen::MatrixXd svd_V=svd.matrixV();
    Eigen::VectorXd KT_coeff=svd_V.col(11);
    Eigen::Matrix<double,3,4> KT;
    KT(0,0) = KT_coeff(0);KT(0,1) = KT_coeff(1);KT(0,2) = KT_coeff(2);KT(0,3) = KT_coeff(3);
    KT(1,0) = KT_coeff(4);KT(1,1) = KT_coeff(5);KT(1,2) = KT_coeff(6);KT(1,3) = KT_coeff(7);
    KT(2,0) = KT_coeff(8);KT(2,1) = KT_coeff(9);KT(2,2) = KT_coeff(10);KT(2,3) = KT_coeff(11);

    //QR分解求K\R
    Eigen::Matrix3d KR=KT.block(0,0,3,3);
    Eigen::HouseholderQR<Eigen::Matrix3d> QR_KR_inverse(KR.inverse());
    Eigen::Matrix3d Q_QR = QR_KR_inverse.householderQ();
    Eigen::Matrix3d R_QR = Q_QR.inverse()*KR.inverse();

    //Eigen::Matrix3d R=Q_QR.inverse();//rotation R
    R=Q_QR.inverse();//rotation R
    //Eigen::Matrix3d K=R_QR.inverse();//innertial parameter k
    Eigen::Matrix3d K_tmp=R_QR.inverse();//innertial parameter k
    K = Eigen::Matrix3d::Zero();
    double norm_coeff=1.0/K_tmp(2,2);
    if(K_tmp(0,0)<0){
        R.block(0,0,2,3) = -1.0*R.block(0,0,2,3);

        norm_coeff*=-1;
        K(0,0) = (norm_coeff*K_tmp(0,0)+norm_coeff*K_tmp(1,1))/2.0;K(1,1) = K(0,0);
        K(0,1) = norm_coeff*K_tmp(0,1);
        K(0,2) = (norm_coeff*K_tmp(0,2));K(1,2) = (norm_coeff*K_tmp(1,2));
        K(2,2) = 1;
    }else{
        K(0,0) = (norm_coeff*K_tmp(0,0)+norm_coeff*K_tmp(1,1))/2.0;K(1,1) = K(0,0);
        K(0,1) = norm_coeff*K_tmp(0,1);
        K(0,2) = (norm_coeff*K_tmp(0,2));K(1,2) = (norm_coeff*K_tmp(1,2));
        K(2,2) = 1;
    }
    //求t
    //Eigen::Vector3d t=K.householderQr().solve(KT.block(0,3,3,1));
    t=K.householderQr().solve(abs(norm_coeff)*KT.block(0,3,3,1));
   
    return;
}

int main(int arc,char** argv){

    //load the data
    vector<Eigen::Vector3d> points_3d;//3d points
    vector<Eigen::Vector2d> points_2d_0;//image1/pose1 2d points/observation
    vector<Eigen::Vector2d> points_2d_1;//image2/pose2 2d points/observation

    string file = "/home/hange/Learn/BasicAlgorithmTest/chapter2/test_ba.txt";
    load_data(file,points_3d,points_2d_0,points_2d_1);
    std::cout<<"poins_3d.size():"<<points_3d.size()<<std::endl;
    std::cout<<"points_2d_0.size():"<<points_2d_0.size()<<std::endl;
    std::cout<<"points_2d_1.size():"<<points_2d_1.size()<<std::endl;

    //手写 pnp 求解相机位姿（相机在世界坐标系下的t、R）作为一个BA优化求解结果的对比
    Eigen::Matrix3d K0,K1;
    Eigen::Matrix3d R0,R1;
    Eigen::Vector3d t0,t1;
    myPNP(points_3d,points_2d_0,K0,R0,t0);
    myPNP(points_3d,points_2d_1,K1,R1,t1);
    cout<<"K0:\n"<<K0<<endl;
    cout<<"R0:\n"<<R0<<endl;
    cout<<"t0:\n"<<t0<<endl;
    cout<<"K1:\n"<<K1<<endl;
    cout<<"R1:\n"<<R1<<endl;
    cout<<"t1:\n"<<t1<<endl;
    cout<<"det(R):"<<R0.determinant()<<"  det(R1):"<<R1.determinant()<<endl;

    //opencv 求解pnp获得的相机位姿
    vector<cv::Point3d> points_3d_cv;
    vector<cv::Point2d> points_2d_0_cv,points_2d_1_cv;
    for(int i=0;i<points_3d.size();i++){
        points_3d_cv.push_back(cv::Point3d(points_3d[i](0),points_3d[i](1),points_3d[i](2)));
        points_2d_0_cv.push_back(cv::Point2d(points_2d_0[i](0),points_2d_0[i](1)));
        points_2d_1_cv.push_back(cv::Point2d(points_2d_1[i](0),points_2d_1[i](1)));
    }
    cv::Mat K0_cv,K1_cv;
    cv::Vec2d dist_coeff(0,0);
    cv::Vec3d R0_vec,R1_vec;
    cv::Mat R0_cv,R1_cv;
    cv::Mat t0_cv,t1_cv;
    cv::Mat inliers_0,inliers_1;
    cv::solvePnPRansac(points_3d_cv,points_2d_0_cv,K0_cv,dist_coeff,R0_vec,t0_cv,inliers_0);
    cv::solvePnPRansac(points_3d_cv,points_2d_1_cv,K1_cv,dist_coeff,R1_vec,t1_cv,inliers_1);
    cv::Rodrigues(R0_vec,R0_cv);
    cv::Rodrigues(R1_vec,R1_cv);

    cout<<"K0_cv:\n"<<K0_cv<<endl;
    cout<<"R0_cv:\n"<<R0_cv<<endl;
    cout<<"t0_cv:\n"<<t0_cv<<endl;
    cout<<"K1_cv:\n"<<K1_cv<<endl;
    cout<<"R1_cv:\n"<<R1_cv<<endl;
    cout<<"t1_cv:\n"<<t1_cv<<endl;
    cout<<"inliers num:"<<inliers_0.size()<<endl;

    //Bundle Adjustment(based on reprojection error)
    //用手写pnp的结果作为基于g2o的BA的初始值
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);

    g2o::OptimizationAlgorithmProperty property;
    optimizer.setAlgorithm(
        OptimizationAlgorithmFactory::instance()->construct("lm_dense6_3",property)
    );

    //给图添加相机参数
    double focal_length=0.972222222;
    Eigen::Vector2d principal_point(0,0);
    g2o::CameraParameters* camera_param_1=new g2o::CameraParameters(focal_length,principal_point,0.0);
    g2o::CameraParameters* camera_param_2=new g2o::CameraParameters(focal_length,principal_point,0.0);
    camera_param_1->setId(0);
    camera_param_2->setId(1);
    optimizer.addParameter(camera_param_1);//add the camera parameter
    optimizer.addParameter(camera_param_2);

    //initialization the SE3Quat 为初始化相机位姿做准备
    Eigen::Quaterniond q0(R0),q1(R1);
    g2o::SE3Quat pose0,pose1;
    pose0.setRotation(q0);
    pose0.setTranslation(t0);
    pose1.setRotation(q1);
    pose1.setTranslation(t1);
    cout<<"pose0:\n"<<pose0<<endl;
    cout<<"pose1:\n"<<pose1<<endl;

    //添加相机位姿顶点：初始化相机位姿并添加到地图的顶点中
    int vertex_index=0;
    g2o::VertexSE3Expmap* vertex_se3_0=new g2o::VertexSE3Expmap();
    g2o::VertexSE3Expmap* vertex_se3_1=new g2o::VertexSE3Expmap();
    vertex_se3_0->setEstimate(pose0);
    vertex_se3_1->setEstimate(pose1);
    vertex_se3_0->setId(vertex_index);
    ++vertex_index;
    vertex_se3_1->setId(vertex_index);
    ++vertex_index;
    optimizer.addVertex(vertex_se3_0);
    optimizer.addVertex(vertex_se3_1);

    //添加3D地图点顶点：添加所有的3D地图点map point作为顶点
    vector<g2o::VertexPointXYZ*> point3d_vertexs;
    for(int i=0;i<points_3d.size();i++){
        g2o::VertexPointXYZ* point3d_vertex_tmp=new g2o::VertexPointXYZ();
        point3d_vertex_tmp->setId(vertex_index);
        ++vertex_index;
        point3d_vertex_tmp->setEstimate(points_3d[i]);
        point3d_vertex_tmp->setMarginalized(true);//false????????what the difference
        optimizer.addVertex(point3d_vertex_tmp);
        point3d_vertexs.push_back(point3d_vertex_tmp);
    }

    int edge_index=0;
    //添加camera0重投影误差作为边
    for(int i=0;i<points_2d_0.size();i++){
        g2o::EdgeProjectXYZ2UV* edge=new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(point3d_vertexs[i]));
        //edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(i+2)->second));
        edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(vertex_se3_0));//dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p)
        
        Eigen::Vector2d predict = camera_param_1->cam_map(pose0.map(points_3d[i]));
        edge->setMeasurement(points_2d_0[i]);
        edge->setId(++edge_index);
        edge->setInformation(Eigen::Matrix2d::Identity()) ;
        edge->setParameterId(0, 0);
        optimizer.addEdge(edge);
        
    }

    //添加camera1重投影误差作为边
    for(int i=0;i<points_2d_1.size();i++){
        g2o::EdgeProjectXYZ2UV* edge=new g2o::EdgeProjectXYZ2UV();
        //edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(point3d_vertexs[i]));
        edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(i+2)->second));
        edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(vertex_se3_1));//dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p)
        Eigen::Vector2d predict = camera_param_2->cam_map(pose1.map(points_3d[i]));
        edge->setMeasurement(points_2d_1[i]);
        edge->setId(++edge_index);
        edge->setParameterId(0, 1);//第二个参数1是指g2o::CameraParameters* camera_param_2
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        
    }

    int maxiters=50;
    optimizer.initializeOptimization();
    optimizer.optimize(maxiters);
    cout<<"优化之前的相机位姿："<<endl;
    cout<<"pose0:\n"<<pose0<<endl;
    cout<<"pose1:\n"<<pose1<<endl;
    cout<<"优化之后的相机位姿："<<endl;
    cout<<"vertex_se3_0:\n"<<vertex_se3_0->estimate()<<endl;
    cout<<"vertex_se3_1:\n"<<vertex_se3_1->estimate()<<endl;

    g2o::Parameter* pt0 = optimizer.parameter(0);
    g2o::CameraParameters* cp0=dynamic_cast<g2o::CameraParameters*>(pt0);
    cout<<"focal length:"<<cp0->focal_length<<endl;
    cout<<"principal point:"<<cp0->principle_point<<endl;
    
    cout<<"优化前后的地图点坐标对比："<<endl;
    for(int i=0;i<5;i++){
        g2o::HyperGraph::VertexIDMap::iterator it=optimizer.vertices().find(i+2);
        g2o::VertexPointXYZ* pt = dynamic_cast<g2o::VertexPointXYZ*>(it->second);
        cout<<"("<<points_3d[i].transpose()<<")--->("<<pt->estimate().transpose()<<")"<<endl;
    }


//这个结果是课程代码task2-5_test_bundle_adjustment.cpp里的对比结果
// # Cam 0 #
// Params before BA: 
//   f: 0.972222
//   distortion: 0, 0
//   R: 1 0 0
// 0 1 0
// 0 0 1

//   t: 0 0 0
// Params after BA: 
//   f: 0.97919
//   distortion: -0.0648399, 0.100804
//   R: 1 -0.000287086 -0.000652844
// 0.000284915 0.999993 -0.00429935
// 0.000653779 0.00429922 0.999993

//   t: 0.000631155 0.0298994 0.0223038
// # Cam 1 #
// Params before BA: 
//   f: 0.972222
//   distortion: 0, 0
//   R: 0.999824 -0.0120624 0.0143949
// 0.0123169 0.999767 -0.0177222
// -0.0141778 0.0178964 0.999739

//   t: 0.0785839 0.994591 0.0679174
// Params after BA: 
//   f: 0.978153
//   distortion: -0.0705936, 0.117592
//   R: 0.999808 -0.0125212 0.0150941
// 0.0127197 0.999836 -0.0130745
// -0.0149276 0.0132639 0.999803

//   t: 0.0803135 0.966225 0.0527785




    std::cout<<"chapter2_g2o_BA.cpp"<<std::endl;
    return 0;
}