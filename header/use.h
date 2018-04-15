//���ļ���Ϊps_way ��ͷ�ļ�����Ҫ������һЩ���ò����򵥻���������ͷ�ļ���
#pragma once
#ifndef USE_H_
#define USE_H_
//----------------------[ͷ�ļ���������]----------------------

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


//----------------------[���ƿռ���������]----------------------
using namespace cv;
using std::cout;
using std::ofstream;
using std::string;
using std::endl;

//----------------------[ȫ�ֱ�����������]----------------------
//�����洢ͼ���Mat����
//Mat g_src, g_srcGray, g_dst;
//ԭͼ���Ҷ�ͼ��Ŀ��ͼ

//----------------------[������������]----------------------

bool to_gray(const Mat & src, Mat & src_Gray);//��ͼ��ת��Ϊ�Ҷ�ͼ
bool read_src(Mat & src, const string & imgName ,int pipe);//��ȡͼ��,��������������ͼ���ȡͨ������
bool src_to_bmp(const Mat & src, const string & imgName);//�洢ͼ��ΪbmpͼƬ
bool show_src(const Mat & src, const string & imgName);//��ʾͼ���ļ�
void output_arr_txt(const Mat &src, const string & txt_name);//��ͼ������д��txt�ļ���
bool read_Graysrc(Mat & gray_src, const string & imgName);//���ļ���ȡΪ�Ҷ�ͼ
void output_arr_csv(const Mat &src, const string & txt_name);//��ͼ������д��csv�ļ���
#endif // !USE_H_


//By Ghostxiu. 2017/9/28