 //transform_pic.cc的头文件
#pragma once
#ifndef TRANSFORM_H_
#define TRANSFORM_H_
//----------------------[头文件引用区域]----------------------

#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "basic_gx.h"
//----------------------[名称空间引用区域]----------------------
using namespace cv;
using std::cout;
using std::ofstream;
using std::string;
using std::endl;


//----------------------[全局变量声明区域]----------------------
//声明存储图像的Mat矩阵
//Mat g_src, g_srcGray, g_dst;
//原图，灰度图，目标图
#define OUTPUT_TRUE 1//是=1，否=0   测试中白色点为1，黑色点为0
#define OUTPUT_FALSE 0
#define SMALL_EDGE 0 //大边为1，小边为0.
#define BIG_EDGE 1
#define MAX_PT  255;//像素点最大值
//----------------------[函数声明区域]----------------------
void maxGradAndDirection(const Mat &, Mat &, Mat &);//计算最大梯度和最大梯度方向
Mat maxGrad(const Mat &);//计算最大梯度 0~255
Mat maxGradDirection(const Mat &);//最大梯度方向0~7
Mat samePixelCnt(const Mat &, int p);//计算8邻域内相同点的个数，其中p是阈值；
Mat clusterDistribution(const Mat &, int);//计算8邻域内不同点的个数
Mat bigSmallEdge(const Mat &);//大小边
Mat model32Edge(const Mat & , const int * , int_pr * ,
	int , int* ,int *, int , int );//是陈老师在17.9提出的方法，用于边缘检测
//Mat bigSmallRegionNum(const Mat & , int tag = 0);// 小区域或大区域点个数
Mat binaryZation(const Mat & , int );//参考图二值化
Mat binarytoZero(const Mat &, int );//二值化为0
//int edge_cnt(const Mat & , int p = 0);//统计边缘点个数
//void edge_station_p_arr(const Mat & , int_pr *, int p =0 );//记录边缘点位置
Mat enlarge_edge(const Mat & );//扩增边缘点
Mat Distogram_gray(const Mat & );//为灰度图绘制直方图
Mat discrimin_pow(const Mat & );//找区分度
double find_maxpoint(const Mat &);//找最小像素值点
double find_minpoint(const Mat & );//找最大像素值点
//Mat Distogram_gx(const Mat &, int *,int ml = GRAY_POINT_MAX_VALUE);//自写的直方图绘制函数
void mark_dis_arr(const Mat & g, int *, int);//直方图数组填充函数
Mat Mirror_Mat(const Mat & ,int );//镜像Mat
Mat Invert_Mat(const Mat & );//灰度Mat上下倒置
int * calcu_dis_arr(const int * , int , int );//按一定规则缩放数组
int calcu_dis_arr_height(int, int, int &);//计算绘制高度
Mat copy_mat(const Mat &);//复制Mat图
Mat CannyEdge(const string &);//Canny 边缘检测法
Mat SobelEdge(const string &);//Sobel 边缘检测法
Mat LaplacianEdge(const string &);//拉普拉斯变幻
Mat ScharEdge(const string &);//scharr滤波器
int LocalMaxDirection(const Mat &, int_pr *);//计算8邻域内局部最大点位置
int Int_pr_cpy(const Mat &, int_pr *, int, int_pr *, int, int_pr *);//拼接多个数组
int Int_pr_cpy(const Mat &, int_pr *, int, int_pr *); //复制数组（上面函数的重载版本)
Mat eli_local_max(const Mat & src, const int_pr *);//剔除局部最大点 
void iter_local_max_edge(const Mat &, int, int_pr *);//迭代找出局部最大点
Mat BinZeroParr(const Mat &, const int_pr *);//类的边缘数组将原图二值化
Mat find_max_local(const Mat &, Mat &,int );//寻找局部最大点
Mat findBigSmallArea(const Mat &,  int th);//寻找大小区域，th是阈值
int find_pos_bigSmallArea(const Mat &gray, int x, int y, int & max_p, int & min_p);//findBigSmallArea()函数的一部分//寻找大区域的前后两个位置
void WeightBigSmall(const Mat & , Mat & , int,  int , const int &, const int & );//findBigSmallArea()函数的一部分 给大小区域通过次数加权
Mat voteBigSmall(const Mat & );//投票判断大小区域
Mat voteToFix(const Mat &, const Mat &, Mat &, Mat &, Mat &, int);//修复矛盾点（原灰度图，投票图“findBigSmallArea”的图,大边缘图,小边缘图，矛盾点图,阈值)
bool FixDiff(const Mat & src, const Mat & votemat, Mat & obj, int area_tag, int i, int j);//修复大小区域的灰度值
bool judeAndFixDiff(const Mat & src, const Mat & votemat, Mat & obj, int i, int j);//判断并修复矛盾点
Mat gx_cvt_color(const Mat & gray, int cl);//以BGR三种单色显示,0蓝，1绿，2红
Mat color3_edge(const Mat & big_edge, const Mat & small_edge, const Mat & diff,int );//使用两种颜色表示大小边 大边（红），小边（绿），矛盾点（蓝）

//ps_way中的函数
bool to_gray(const Mat & src, Mat & src_Gray);//将图像转化为灰度图
bool read_src(Mat & src, const string & imgName, int pipe = 0 );//读取图像,第三个参数代表图像读取通道参数
bool src_to_bmp(const Mat & src, const string & imgName);//存储图像为bmp图片
bool show_src(const Mat & src, const string & imgName);//显示图像文件
void output_arr_txt(const Mat &src, const string & txt_name);//将图像数组写入txt文件中
bool read_Graysrc(Mat & gray_src, const string & imgName);//将文件读取为灰度图
void output_arr_csv(const Mat &src, const string & txt_name);//将图像数组写入csv文件中

 //majpyi

Mat sortpixel(const Mat & gray, int th);
int testsortpixel(const Mat & gray, int th, int x, int y);


Mat find_edge(Mat & votemat);
Mat find_edge(Mat & votemat);

Mat find_edgexiao(Mat & votemat);
Mat mfixDiff(const Mat & src, Mat & votemat, Mat & bigm, Mat & smallm, Mat & diffm, int th);
int centerFix(const Mat & gray, int maxcha, int x, int y);
Mat mfindBigSmallArea(const Mat & gray, int th);
Mat color2_vote(const Mat & vote);


Mat fix7vs1(Mat & gray);
int testfix7vs1(Mat & gray, int x, int y);


Mat sort7vs1(const Mat & gray);
Mat find_edgeboth(Mat & votemat);

Mat newsortpixel(const Mat & gray, int th);
void newtestsortpixel(const Mat & gray, int th, int x, int y);

Mat find_edgeboth2(Mat & votemat);
Mat tagdaxiao(Mat & votemat);
//Mat enhance(Mat & gray, Mat & votemat);
//Mat enhance(Mat & gray, Mat & votemat, Mat & yuan);
Mat enhance(Mat & gray, const Mat & votemat, const Mat & yuan);
Mat Mapping(Mat & gray, Mat & tag, Mat & yuan);
void Histogram(Mat & gray);
int histm();
void Histogramdaxiao(Mat & gray, Mat & vote);
Mat color_findBigSmallArea(Mat &gray, IplImage * a, int th);
Mat mcolor_fixDiff(Mat gray, Mat & src, Mat & votemat, Mat & bigm, Mat & smallm, Mat & diffm, int th);
int judge_single(const Mat & gray, int th, int i, int j);
int judge_transition(const Mat & gray, int th, int i, int j);
Mat color_fix7vs1(Mat & src, IplImage * a, Mat &gray);
Mat color_sortpixel(const Mat & gray, Mat & color, IplImage * a, int th);
void Near_attribution(const Mat &gray, Mat &vote, Mat & big,Mat & small, int x, int y, Mat &newgray, Mat &newvote,Mat &newbig ,Mat &newsmall);
int verity(Mat vote,int x,int y);



#endif // !TRANSFORM_H_
//By Ma. 2018/1/30