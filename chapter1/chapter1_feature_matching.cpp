#include <iostream>
#include <vector>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

template<class T>
void my_mutual_lowe_ratio_matching(Mat& src1,Mat& src2,vector<KeyPoint>& keypoint1,vector<KeyPoint>& keypoint2,
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
            if(knn_matches1[i][0].distance/knn_matches1[i][1].distance<0.75f ){
                mutual_loweratio_matches.push_back(knn_matches1[i][0]);
            }
        }
    }
    cv::Mat mutual_lowe_img;
    drawMatches(src1,keypoint1,src2,keypoint2,mutual_loweratio_matches,mutual_lowe_img,cv::Scalar(0,0,255));
    imshow(windname_pre+"mutual lowe ratio matches",mutual_lowe_img);
    imwrite("/home/hange/Learn/BasicAlgorithmTest/build/chapter1/feature_matching.jpg",mutual_lowe_img);
    waitKey(0);

    return;
}

int main(int argc,char** argv){

    string img_file1="/home/hange/Desktop/data/kxm1.jpg";
    string img_file2="/home/hange/Desktop/data/kxm2.jpg";

    cv::Mat src1,src2;
    src1 = cv::imread(img_file1,cv::IMREAD_COLOR);
    src2 = cv::imread(img_file2,cv::IMREAD_COLOR);
    if(src1.empty() || src2.empty()){
        cout<<endl<<"load data error"<<endl;
        return 0;
    }

    //keypoint detect
    int key_point_num=1000;
    Ptr<ORB> detector=ORB::create(key_point_num,1.2,5);
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
    Ptr<SIFT> detector_sift=SIFT::create(key_point_num,4);
    my_mutual_lowe_ratio_matching<SIFT>(src1,src2,keypoint1,keypoint2,detector_sift,"SIFT ",DescriptorMatcher::BRUTEFORCE);//DescriptorMatcher::BRUTEFORCE

    destroyAllWindows();
    cout<<"chapter1_feature_matching.cpp"<<endl;
    return 0;
}