# BasicAlgorithmTest
test the algorithms learned from course

chapter2迭代：
本次迭代的作业都在chapter2文件夹下：
1、chapter2_g2o_curve_fitting.cpp通过曲线拟合熟悉g2o的使用
2、chapter2_g2o_BA.cpp里首先实现pnp的位姿估计，分为手写部分和OpenCV两个部分，手写部分结果与课程代码       task2-5_test_bundle_adjustment.cpp中的理论结果更相似
3、chapter2_g2o_BA.cpp里以手写pnp的结果作为初始值初始化基于g2o构建的BA，并优化相机位姿和三维点坐标

结果如下：
优化之前的相机位姿：
pose0:
           1 -7.93017e-05  0.000205216   0.00398599
 7.93434e-05            1 -0.000203216  -0.00074192
  -0.0002052  0.000203232            1  -0.00179433
           0            0            0            1

pose1:
  0.999831 -0.0119387  0.0140138  0.0762272
 0.0121885   0.999766 -0.0178751   0.995671
-0.0137971  0.0180429   0.999742  0.0612567
         0          0          0          1

优化之后的相机位姿：
vertex_se3_0:
    0.999997 -0.000952277   0.00221527   -0.0221661
 0.000974289      0.99995  -0.00995664    0.0703978
 -0.00220568   0.00995877     0.999948    0.0132127
           0            0            0            1

vertex_se3_1:
  0.999826 -0.0128116  0.0135814   0.088381
 0.0132111    0.99947 -0.0297409    1.07637
-0.0131931  0.0299152   0.999465  0.0586664
         0          0          0          1

focal length:0.972222
principal point:0
0
优化前后的地图点坐标对比：
( 1.32001 -1.14151  7.08367)--->(1.31801 -1.1417 7.08199)
(0.0191685  0.912243   7.41801)--->(0.0210905    0.9187    7.4171)
(-1.12243 -1.23971  7.19036)--->(-1.11951 -1.23555  7.17788)
( -0.99997 -0.568618   7.27555)--->(-0.998464 -0.565529   7.26821)
(0.0637892  -0.90761    7.3552)--->(0.0636606 -0.904008   7.34775)