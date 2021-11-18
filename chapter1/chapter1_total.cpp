#include <iostream>
#include <vector>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace Eigen;

#define sfm_course
//#define cv_compare

//feature matches
template<class T>
vector<DMatch> my_mutual_lowe_ratio_matching(Mat& src1,Mat& src2,vector<KeyPoint>& keypoint1,vector<KeyPoint>& keypoint2,
        Ptr<T>& detector,string windname_pre,DescriptorMatcher::MatcherType dist_type){

    //keypoint descriptor
    cv::Mat descriptor1,descriptor2;
    detector->compute(src1,keypoint1,descriptor1);
    detector->compute(src2,keypoint2,descriptor2);

    //matching keypoint
    Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create(dist_type);
    vector<DMatch> matches;
    matcher->match(descriptor1,descriptor2,matches);

    cv::Mat match_img;
    drawMatches(src1,keypoint1,src2,keypoint2,matches,match_img);
    imshow(windname_pre+"one way matches",match_img);
    waitKey(0);
    
    //mutual nearest matching
    Ptr<DescriptorMatcher> knn_matcher=DescriptorMatcher::create(dist_type);
    vector<vector<DMatch>> knn_matches1,knn_matches2;
    knn_matcher->knnMatch(descriptor1,descriptor2,knn_matches1,2);
    knn_matcher->knnMatch(descriptor2,descriptor1,knn_matches2,2);
    vector<DMatch> mutual_matches;

    //cout<<endl<<"1st query index:"<<knn_matches1[0][0].queryIdx<<" train index:"<<knn_matches1[0][0].trainIdx<<"  dist:"<<knn_matches1[0][0].distance<<endl;
    //cout<<endl<<"1st query index:"<<knn_matches1[0][1].queryIdx<<" train index:"<<knn_matches1[0][1].trainIdx<<"  dist:"<<knn_matches1[0][1].distance<<endl;
    for(int i=0;i<knn_matches1.size();i++){
        if(knn_matches2[knn_matches1[i][0].trainIdx ][0].trainIdx == i){
            mutual_matches.push_back(knn_matches1[i][0]);
        }
    }
    cv::Mat mutual_img;
    drawMatches(src1,keypoint1,src2,keypoint2,mutual_matches,mutual_img);
    imshow(windname_pre+"mutual matches",mutual_img);
    waitKey(0);

    //lowe-ratio(1st-dis/2nd-dis<0.7f)
    vector<DMatch> mutual_loweratio_matches;
    for(int i=0;i<knn_matches1.size();i++){
        if(knn_matches2[knn_matches1[i][0].trainIdx ][0].trainIdx == i){
            if(knn_matches1[i][0].distance/knn_matches1[i][1].distance<0.7f ){
                mutual_loweratio_matches.push_back(knn_matches1[i][0]);
            }
        }
    }
    cv::Mat mutual_lowe_img;
    drawMatches(src1,keypoint1,src2,keypoint2,mutual_loweratio_matches,mutual_lowe_img,cv::Scalar(0,0,255));
    imshow(windname_pre+"mutual lowe ratio matches",mutual_lowe_img);
    imwrite("/home/hange/Learn/BasicAlgorithmTest/build/chapter1/feature_matching.jpg",mutual_lowe_img);
    waitKey(0);

    return mutual_loweratio_matches;
}


//polar geometry constraint to compute F
Eigen::MatrixXf ComputeFundmentalMatrix(vector<KeyPoint>& keypoint1,vector<KeyPoint>& keypoint2){
    cout<<endl<<"ComputeFundmentalMatrix"<<endl;

    //AX=0系数矩阵
    Eigen::MatrixXf A_coeff(keypoint1.size(),9);
    for(int i=0;i<keypoint1.size();i++){
        
        A_coeff(i,0) = keypoint2[i].pt.x*keypoint1[i].pt.x;
        A_coeff(i,1) = keypoint2[i].pt.x*keypoint1[i].pt.y;
        A_coeff(i,2) = keypoint2[i].pt.x;
        A_coeff(i,3) = keypoint2[i].pt.y*keypoint1[i].pt.x;
        A_coeff(i,4) = keypoint2[i].pt.y*keypoint1[i].pt.y;
        A_coeff(i,5) = keypoint2[i].pt.y;
        A_coeff(i,6) = keypoint1[i].pt.x;
        A_coeff(i,7) = keypoint1[i].pt.y;
        A_coeff(i,8) = 1.0f;
    }

    //SVD分解获得F矩阵
    
    Eigen::BDCSVD<Eigen::MatrixXf> svd(A_coeff,ComputeFullU | ComputeFullV);
    Eigen::MatrixXf V=svd.matrixV();

    Eigen::MatrixXf F=V.col(8);
    F.resize(3,3);
    F.transposeInPlace();
    cout<<"F:\n"<<F<<endl;

    //基础矩阵F重构
    Eigen::BDCSVD<Eigen::MatrixXf> svd_f(F,ComputeFullU | ComputeFullV);
    Eigen::MatrixXf U_f = svd_f.matrixU();
    Eigen::MatrixXf V_f = svd_f.matrixV();//svd_f.singularValues()
    Eigen::MatrixXf S_f=Eigen::MatrixXf::Zero(3,3);
    S_f(0,0) = svd_f.singularValues()(0);
    S_f(1,1) = svd_f.singularValues()(1);
    //S_f(2,2) = svd_f.singularValues()(2);
    cout<<"f singularvalues:\n"<<svd_f.singularValues()<<endl;
    F = U_f*S_f*V_f.transpose();
    cout<<"F:\n"<<F<<endl;
    
    //opencv 算出的F矩阵
    vector<Point2f> keypoint1_2f,keypoint2_2f;
    for(int i=0;i<keypoint1.size();i++){
        Point2f p1_tmp,p2_tmp;
        p1_tmp.x = keypoint1[i].pt.x;
        p1_tmp.y = keypoint1[i].pt.y;
        p2_tmp.x = keypoint2[i].pt.x;
        p2_tmp.y = keypoint2[i].pt.y;
        keypoint1_2f.push_back(p1_tmp);
        keypoint2_2f.push_back(p2_tmp);
    }

    

#ifdef sfm_course
    cv::Mat F_cv = findFundamentalMat(keypoint1_2f,keypoint2_2f,FM_8POINT);
#else
    cv::Mat F_cv = findFundamentalMat(keypoint1_2f,keypoint2_2f,FM_RANSAC);
#endif
    cout<<"opencv F_cv:\n"<<F_cv<<endl;
    F = (F_cv.at<double>(2,2)/F(2,2))*F;
    cout<<"compare F:\n"<<F<<endl;

#ifdef sfm_course
    std::cout<<"Result F should be: \n"<<"-0.0315082 -0.63238 0.16121\n"
                                     <<"0.653176 -0.0405703 0.21148\n"
                                     <<"-0.248026 -0.194965 -0.0234573\n" <<std::endl;
#endif        

#ifdef cv_compare
    cv2eigen(F_cv,F);
#endif

    return F;
}


//F+K to compute E matrix
Eigen::MatrixXf ComputeEssentialMatrix(Eigen::MatrixXf& F,Eigen::MatrixXf& K,
        vector<cv::KeyPoint>& keypoint1,vector<cv::KeyPoint>& keypoint2){
    cout<<"ComputeEssentialMatrix"<<endl;

    //E = K^T*F*K
    Eigen::MatrixXf E;
    E = K.transpose()*F*K;
    cout<<"the E:\n"<<E<<endl;

    //SVD decompose and reconstruction
    Eigen::BDCSVD<Eigen::MatrixXf> svd_E(E,ComputeFullU | ComputeFullV);
    Eigen::MatrixXf U_E = svd_E.matrixU();
    Eigen::MatrixXf V_E = svd_E.matrixV();
    Eigen::VectorXf sig_value=svd_E.singularValues();

    Eigen::MatrixXf S_E=Eigen::MatrixXf::Zero(3,3);
    S_E(0,0) = (sig_value(0)+sig_value(1))/2.0;
    S_E(1,1) = (sig_value(0)+sig_value(1))/2.0;
    S_E(2,2) = 0.0;
    E = U_E*S_E*V_E.transpose();
    cout<<"E:\n"<<E<<endl;

    //OpenCV compute E
#ifdef sfm_course
    Point2f principal_point(0,0);
    double focal_length=1;
#else
    Point2f principal_point(325.1,249.7);
    double focal_length=521;
#endif
    Mat E_cv;
    vector<Point2f> keypoint1_pf,keypoint2_pf;
    for(int  i=0;i<keypoint1.size();i++){
        Point2f tmp;
        tmp.x = keypoint1[i].pt.x;
        tmp.y = keypoint1[i].pt.y;
        keypoint1_pf.push_back(tmp);
        tmp.x = keypoint2[i].pt.x;
        tmp.y = keypoint2[i].pt.y;
        keypoint2_pf.push_back(tmp);
    }
    E_cv = cv::findEssentialMat(keypoint1_pf,keypoint2_pf,focal_length,principal_point);
    cout<<"E_cv:\n"<<E_cv<<endl;

#ifdef sfm_course
    std::cout<<"EssentialMatrix should be: \n"
             <<"-0.00490744 -0.0146139 0.34281\n"
             <<"0.0212215 -0.000748851 -0.0271105\n"
             <<"-0.342111 0.0315182 -0.00552454\n";
#endif

#ifdef cv_compare
    cv2eigen(E_cv,E);
#endif  

    return E;
}

Eigen::VectorXf mytriangle(cv::KeyPoint p1,Eigen::MatrixXf& P1,cv::KeyPoint p2,Eigen::MatrixXf& P2
       ){
    //K,p1,P1,p2,P2,vector_R_t[i].first,vector_R_t[i].second
    Eigen::MatrixXf A=Eigen::MatrixXf::Zero(4,4);
    A.row(0) = p1.pt.x*P1.block(2,0,1,4) - P1.block(0,0,1,4);
    A.row(1) = p1.pt.y*P1.block(2,0,1,4) - P1.block(1,0,1,4);
    A.row(2) = p2.pt.x*P2.block(2,0,1,4) - P2.block(0,0,1,4);
    A.row(3) = p2.pt.y*P2.block(2,0,1,4) - P2.block(1,0,1,4);

    //A svd decomposition and get the Least squre rslt
    Eigen::BDCSVD<Eigen::MatrixXf> svd_A(A, ComputeFullV);
    Eigen::VectorXf P = svd_A.matrixV().block(0,3,4,1);
    P = P/P(3);//scale to the last element
    

    return P;
}

bool  mycheck_R_t(Eigen::VectorXf& P,Eigen::MatrixXf& R,Eigen::VectorXf& t){
    //P,vector_R_t[i].first,vector_R_t[i].second
    bool flag=true;
    if(P(2)<0){
        flag=false;
    }else{
        Eigen::MatrixXf R_t=Eigen::MatrixXf::Zero(3,4);
        R_t.block(0,0,3,3) = R;
        R_t.block(0,3,3,1) = t;
        Eigen::VectorXf p_3d=R_t*P;
        if(p_3d(2)<0){
            flag = false;
        }
    }
    return flag;
}


//decompose E ==>R,t
void ComputeR_T(Eigen::MatrixXf& E,vector<KeyPoint>& keypoint1,vector<KeyPoint>& keypoint2,Eigen::MatrixXf& K){

    cout<<"ComputeR_T\n"<<endl;

    Eigen::BDCSVD<Eigen::MatrixXf> svd_E(E,ComputeFullU | ComputeFullV);
    Eigen::MatrixXf U_E = svd_E.matrixU();
    Eigen::MatrixXf V_E = svd_E.matrixV();

    if(U_E.determinant()<0){
        for(int i=0;i<3;i++){
            U_E(i,2) = -U_E(i,2);
        }
    }
    
    if(V_E.determinant()<0){
        for(int i=0;i<3;i++){
            V_E(i,2) = -V_E(i,2);
        }
    }

    Eigen::MatrixXf W=Eigen::MatrixXf::Zero(3,3);
    W(0,1) = -1;W(1,0) = 1;W(2,2) = 1;
    Eigen::MatrixXf Wt=Eigen::MatrixXf::Zero(3,3);
    Wt(0,1) = 1;Wt(1,0) = -1;Wt(2,2) = 1;

    vector<pair<Eigen::MatrixXf,Eigen::VectorXf>> vector_R_t;
    vector_R_t.push_back(pair<Eigen::MatrixXf,Eigen::VectorXf>(U_E*W*V_E.transpose(),U_E.col(2)));
    vector_R_t.push_back(pair<Eigen::MatrixXf,Eigen::VectorXf>(U_E*W*V_E.transpose(),-U_E.col(2)));
    vector_R_t.push_back(pair<Eigen::MatrixXf,Eigen::VectorXf>(U_E*Wt*V_E.transpose(),U_E.col(2)));
    vector_R_t.push_back(pair<Eigen::MatrixXf,Eigen::VectorXf>(U_E*Wt*V_E.transpose(),-U_E.col(2)));

    for(int i=0;i<vector_R_t.size();i++){
        cv::KeyPoint p1,p2;
        p1=keypoint1[0];
        p2=keypoint2[0];
        Eigen::MatrixXf P1=Eigen::MatrixXf::Zero(3,4),P2=Eigen::MatrixXf::Zero(3,4);
        P1.block(0,0,3,3) = Eigen::MatrixXf::Identity(3,3);
        P1 = K*P1;
        P2.block(0,0,3,3) = vector_R_t[i].first;
        P2.block(0,3,3,1) = vector_R_t[i].second;
        P2 = K*P2;
        //triangulation find the 3-D pos of P
        Eigen::VectorXf P= mytriangle(p1,P1,p2,P2);

        //transform P to the C1,C2 coordinate ,check if p1_3d.z>0 && p2_3d.z>0
        bool right_flag = mycheck_R_t(P,vector_R_t[i].first,vector_R_t[i].second);

        if(right_flag){
            cout<<"R:\n"<<vector_R_t[i].first<<"\nt\n:"<<vector_R_t[i].second<<endl;
        }
    }
#ifdef sfm_course
    std::cout<<"Result should be: \n";
    std::cout<<"R: \n"
             << "0.999827 -0.0119578 0.0142419\n"
             << "0.0122145 0.999762 -0.0180719\n"
             << "-0.0140224 0.0182427 0.999735\n";
    std::cout<<"t: \n"
             <<"0.0796625 0.99498 0.0605768\n";
#endif
    //opencv compute R t
    Point2d principal_point(325.1,249.7);
    double focal_length=521;
#ifdef sfm_course
    principal_point = Point2d(0,0);
    focal_length=1;
#endif
    cv::Mat R,t;
    vector<Point2f> keypoint1_pf,keypoint2_pf;
    for(int i=0;i<keypoint2.size();i++){
        keypoint1_pf.push_back(Point2f(keypoint1[i].pt.x,keypoint1[i].pt.y) );
        keypoint2_pf.push_back(Point2f(keypoint2[i].pt.x,keypoint2[i].pt.y) );
    }

    cv::Mat E_cv=(cv::Mat_<double>(3,3)<<E(0,0),E(0,1),E(0,2),E(1,0),E(1,1),E(1,2),E(2,0),E(2,1),E(2,2));
    //eigen2cv(E,E_cv);
    recoverPose(E_cv,keypoint1_pf,keypoint2_pf,R,t,focal_length,principal_point);
    cout<<"cv R:\n"<<R<<endl;
    cout<<"cv t:\n"<<t<<endl;



    return;
}


int main(int argc,char** argv){

    string img_file1="/home/hange/Learn/BasicAlgorithmTest/chapter1/data/1.png";//kxm1.jpg  1.png
    string img_file2="/home/hange/Learn/BasicAlgorithmTest/chapter1/data/2.png";//kxm2.jpg 2.png

    cv::Mat src1,src2;
    src1 = cv::imread(img_file1,cv::IMREAD_COLOR);
    src2 = cv::imread(img_file2,cv::IMREAD_COLOR);
    if(src1.empty() || src2.empty()){
        cout<<endl<<"load data error"<<endl;
        return 0;
    }

    //keypoint detect
    int key_point_num=1000;
    Ptr<SIFT> detector=SIFT::create(key_point_num,3);
    vector<KeyPoint> keypoint1,keypoint2;
    detector->detect(src1,keypoint1);
    detector->detect(src2,keypoint2);

    cv::Mat keypoint1_draw,keypoint2_draw;
    drawKeypoints(src1,keypoint1,keypoint1_draw);
    drawKeypoints(src2,keypoint2,keypoint2_draw);
    imshow("keypoint1_draw",keypoint1_draw);
    imshow("keypoint2_draw",keypoint2_draw);
    cout<<endl<<"keypoint1.size:"<<keypoint1.size()<<endl;
    cout<<endl<<"keypoint2.size:"<<keypoint2.size()<<endl;
    waitKey(0);

    //ORB检测更快,但是sift更稳，而且做最近邻和次近邻比值好设置
    //经过测试发现这种特征点效果比直接的sift特征点匹配效果更好
    Ptr<SIFT> discriptor=SIFT::create(key_point_num,3);
    vector<DMatch> matches = my_mutual_lowe_ratio_matching<SIFT>(src1,src2,keypoint1,keypoint2,discriptor,"SIFT ",DescriptorMatcher::BRUTEFORCE);//DescriptorMatcher::BRUTEFORCE
    vector<KeyPoint> keypoint1_match,keypoint2_match;
    vector<DMatch> matches_match;
    for(int i=0;i<matches.size();i++){
        keypoint1_match.push_back(keypoint1[matches[i].queryIdx]);
        keypoint2_match.push_back(keypoint2[matches[i].trainIdx]);
        DMatch match;
        match.queryIdx = i;
        match.trainIdx = i;
        match.distance = matches[i].distance;
        matches_match.push_back(match);
    }

    // cv::Mat mat_match;
    // drawMatches(src1,keypoint1_match,src2,keypoint2_match,matches_match,mat_match);
    // imshow("matches",mat_match);
    // waitKey(0);
    
#ifdef sfm_course
    keypoint1_match.clear();
    keypoint2_match.clear();
    KeyPoint tmp;
    tmp.pt.x = 0.180123 ; tmp.pt.y = -0.156584;
    keypoint1_match.push_back(tmp);
    tmp.pt.x = 0.291429 ; tmp.pt.y = 0.137662 ;
    keypoint1_match.push_back(tmp);
    tmp.pt.x = -0.170373; tmp.pt.y = 0.0779329;
    keypoint1_match.push_back(tmp);
    tmp.pt.x = 0.235952 ; tmp.pt.y = -0.164956;
    keypoint1_match.push_back(tmp);
    tmp.pt.x = 0.142122 ; tmp.pt.y = -0.216048; 
    keypoint1_match.push_back(tmp);
    tmp.pt.x = -0.463158 ; tmp.pt.y = -0.132632;
    keypoint1_match.push_back(tmp);
    tmp.pt.x = 0.0801864 ; tmp.pt.y = 0.0236417;
    keypoint1_match.push_back(tmp);
    tmp.pt.x = -0.179068; tmp.pt.y = 0.0837119;
    keypoint1_match.push_back(tmp);

    tmp.pt.x = 0.208264 ; tmp.pt.y = -0.035405 ; 
    keypoint2_match.push_back(tmp);
    tmp.pt.x = 0.314848 ; tmp.pt.y = 0.267849 ;
    keypoint2_match.push_back(tmp);
    tmp.pt.x = -0.144499; tmp.pt.y = 0.190208;
    keypoint2_match.push_back(tmp);
    tmp.pt.x = 0.264461 ; tmp.pt.y = -0.0404422; 
    keypoint2_match.push_back(tmp);
    tmp.pt.x = 0.171033; tmp.pt.y = -0.0961747;  
    keypoint2_match.push_back(tmp);
    tmp.pt.x = -0.427861 ; tmp.pt.y = 0.00896567;
    keypoint2_match.push_back(tmp);
    tmp.pt.x = 0.105406; tmp.pt.y = 0.140966; 
    keypoint2_match.push_back(tmp);
    tmp.pt.x = -0.15257; tmp.pt.y = 0.19645 ;
    keypoint2_match.push_back(tmp);
#endif
    //对极几何求解F基础矩阵
    Eigen::MatrixXf F = ComputeFundmentalMatrix(keypoint1_match,keypoint2_match);

#ifdef sfm_course
    F(0,0) = -0.0051918668202215884;
    F(0,1) = -0.015460923969578466;
    F(0,2) = 0.35260470328319654;
    F(1,0) = 0.022451443619913483;
    F(1,1) = -0.00079225386526248181;
    F(1,2) = -0.027885130552744289;
    F(2,0) = -0.35188558059920161;
    F(2,1) = 0.032418724757766811;
    F(2,2) = -0.005524537443406155;
#endif
    //根据内参矩阵K求解本质矩阵E
    Eigen::MatrixXf K(3,3);
    K<<520.9,0,325.1,0,521.0,249.7,0,0,1;
#ifdef sfm_course
    float f=0.972222208;
    K<<f,0,0,0,f,0,0,0,1;
#endif
    Eigen::MatrixXf E = ComputeEssentialMatrix(F,K,keypoint1_match,keypoint2_match);

    ComputeR_T(E,keypoint1_match,keypoint2_match,K);



    destroyAllWindows();
    cout<<"chapter1_feature_matching.cpp"<<endl;
    return 0;
}