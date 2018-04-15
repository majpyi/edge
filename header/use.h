//本文件做为ps_way 的头文件，主要用来将一些常用操作简单化，并引用头文件库
#pragma once
#ifndef USE_H_
#define USE_H_
//----------------------[头文件引用区域]----------------------

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


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

//----------------------[函数声明区域]----------------------

bool to_gray(const Mat & src, Mat & src_Gray);//将图像转化为灰度图
bool read_src(Mat & src, const string & imgName ,int pipe);//读取图像,第三个参数代表图像读取通道参数
bool src_to_bmp(const Mat & src, const string & imgName);//存储图像为bmp图片
bool show_src(const Mat & src, const string & imgName);//显示图像文件
void output_arr_txt(const Mat &src, const string & txt_name);//将图像数组写入txt文件中
bool read_Graysrc(Mat & gray_src, const string & imgName);//将文件读取为灰度图
void output_arr_csv(const Mat &src, const string & txt_name);//将图像数组写入csv文件中
#endif // !USE_H_


//By Ghostxiu. 2017/9/28