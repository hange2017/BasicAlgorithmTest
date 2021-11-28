#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/block_solver.h>

using namespace std;
using namespace Eigen;
using namespace g2o;


G2O_USE_OPTIMIZATION_LIBRARY(dense);

//define the optimization vertex
class myvertex:public g2o::BaseVertex<3,Eigen::Vector3d>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        myvertex() {}
        virtual bool read(std::istream& /*is*/) { return false; }
        virtual bool write(std::ostream& /*os*/) const { return false; }
        //initial parameter
        virtual void setToOriginImpl(){
            _estimate<<0,0,0;
        }
        //update parameter
        virtual void oplusImpl(const double* update){
            Eigen::Vector3d::ConstMapType v(update);
            _estimate +=v;
        }
};

//define the optimization edge
class myedge:public g2o::BaseUnaryEdge<1,Eigen::Vector2d, myvertex>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        myedge(){}
        virtual bool read(std::istream& /*is*/) {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }
        virtual bool write(std::ostream& /*os*/) const {
            cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        template<typename T>
        bool operator()(const T* params,T* error)const{
            const T& a=params[0];
            const T& b=params[1];
            const T& c=params[2];

            T predict = a*measurement()(0)*measurement()(0)+b*measurement()(0)+c;
            error[0] = predict - measurement()(1);
            return true;
        }

        G2O_MAKE_AUTO_AD_FUNCTIONS;

};


int main(int argc,char** argv){

    //generate the curve points
    int points_num=200;
    Eigen::VectorXd points_x=Eigen::VectorXd::LinSpaced(points_num,20,30);
    Eigen::VectorXd points_y(points_num);
    double a_param=3.0,b_param=4.0,c_param=5.0;
    for(int i=0;i<points_num;i++){
        points_y(i) = a_param*points_x(i)*points_x(i)+b_param*points_x(i)+c_param+Eigen::VectorXd::Random(1)(0);
       // std::cout<<"x:"<<points_x(i)<<"  y:"<<points_y(i)<<std::endl;
    }

    //
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> block_solver_type;
    typedef g2o::LinearSolverDense<block_solver_type::PoseMatrixType> linear_solver_type;
    auto solver =new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<block_solver_type>(g2o::make_unique<linear_solver_type>()));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    //optimization params
    double a=0,b=0,c=0;

    //add vertex
    myvertex* vertex0=new myvertex();
    vertex0->setId(0);
    //vertex0->setEstimate(Eigen::Vector3d(a,b,c));
    optimizer.addVertex(vertex0);

    //add edge
    for(int i=0;i<points_num;i++){
        myedge* edge=new myedge();
        edge->setId(i);
        edge->setVertex(0,vertex0);
        edge->setMeasurement(Eigen::Vector2d(points_x(i),points_y(i)));
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity());//必须添加
        optimizer.addEdge(edge);
    }

    int maxiter=60;
    optimizer.initializeOptimization();
    optimizer.optimize(maxiter);

    cout<<"a:"<<vertex0->estimate()(0)<<"  b:"<<vertex0->estimate()(1)<<" c:"<<vertex0->estimate()(0)<<endl;


    std::cout<<"chapter2_g2o_curve_fitting"<<std::endl;
    return 0;
}