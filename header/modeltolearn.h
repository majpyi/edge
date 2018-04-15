//modeltolearn.cc ��ͷ�ļ�
#pragma once
#ifndef MODELTOLEARN_H_
#define MODELTOLEARN_H_
//----------------------[ͷ�ļ���������]----------------------

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "basic_gx.h"
//----------------------[���ƿռ���������]----------------------
using std::cin;
using std::cout;
using std::endl;
using namespace cv;
using std::string;
using std::ifstream;
using std::ofstream;
//----------------------[ȫ�ֱ�����������]----------------------
//�����洢ͼ���Mat����
Mat g_src, g_srcGray, g_dst;
//ԭͼ���Ҷ�ͼ��Ŀ��ͼ
const int PIPE_GRAY = 1;//�趨�Ҷ�ͼͨ����Ϊ1
const int BIN_APART = 127;//��ֵ���ֽ�
const int EDGE8_MODEL_SIZE = 36;//ģ���С
//36��ģ��
const int ModelArr[EDGE8_MODEL_SIZE] = { 0,1,3,5,7,9,11,13,15,17,19,
21,23,25,27,29,31,37,39,43,45,47,51,53,55,59,61,63,
85,87,91,95,111,119,127,255 };
double ModelPercent[EDGE8_MODEL_SIZE];//ģ������Ǳ�Ե��İٷֱ�
const int Thre = 20;//�����ļ�����ֵʱ�ı�Ե��ֵ
const double per_cent = 100.00;//�ٷֱ�
int pic_nums;//ͼƬ����


//----------------------[������������]----------------------
void show_edge_station(const int_pr *, int);//�����Ե��λ��
void show_model_cnt(const int *, const int *, int ,int );//���ģ����������
void single_process(int th = Thre);//���ļ��������
void batch_process(int th = Thre);//������
void test();//����һЩ�������ܵĺ���
void fout_model_cnt(const int *, const int *, int, string,int);//��ģ������д���ļ�
void same_thre_per(int );//ͬ��ֵ��ģ�����ְٷֱ�
void plot_disgram(); //����ֱ��ͼ
void twice_dis_pow();//�������ֶ�ͼ
void dis_pow_try_threhold(int th = 20);//ʹ�����ֶȺ���ֵ��Ե���
void otherway_edge_detect();//ʹ��������Ե��ⷽ��
void org_expert_csv();//�洢csv�ļ�
Mat FindLocalMaxEdge(const string & , int );//Ѱ�Ҿֲ�����Եͼ
Mat FindFriendEdge(const string & pic_name);//ʹ�������ѵķ���17.12.18

//----------------------[�ⲿ��������]----------------------
//transform_pic�еĺ���
void maxGradAndDirection(const Mat &, Mat &, Mat &);//��������ݶȺ�����ݶȷ���
Mat maxGrad(const Mat & );//��������ݶ� 0~255
Mat maxGradDirection(const Mat & );//����ݶȷ���0~7
Mat samePixelCnt(const Mat &, int p);//����8��������ͬ��ĸ���������p����ֵ��
Mat clusterDistribution(const Mat &, int);//Ⱥ�ֲ�
Mat bigSmallEdge(const Mat &);//��С��   

Mat model32Edge(const Mat &, const int *, int_pr *,
int, int*,int *, int, int);//�ǳ���ʦ��17.9����ķ��������ڱ�Ե���
Mat binaryZation(const Mat & src, int p);//�ο�ͼ��ֵ��
Mat binarytoZero(const Mat &, int);//��ֵ��Ϊ0
int edge_cnt(const Mat &, int p = 0);//ͳ�Ʊ�Ե��������Ѷ�ֵ����ר��ͼ
void edge_station_p_arr(const Mat &, int_pr *, int p = 0);//��¼��Ե��λ��
Mat enlarge_edge(const Mat &);//������Ե��
Mat Distogram_gray(const Mat &);//Ϊ�Ҷ�ͼ����ֱ��ͼ
Mat discrimin_pow(const Mat &);//�����ֶ�
double find_maxpoint(const Mat &);//����С����ֵ��
double find_minpoint(const Mat &);//���������ֵ��
Mat add_point(const Mat & gray1, const Mat & gray2);
Mat Distogram_gx(const Mat & , int * , int max_len = GRAY_POINT_MAX_VALUE);//��д��ֱ��ͼ���ƺ���
Mat Mirror_Mat(const Mat &, int rotate = 0 );//����Mat
Mat Invert_Mat(const Mat &);//�Ҷ�Mat���µ���
Mat minus_point(const Mat & , const Mat &);//��ͼ֮��
Mat copy_mat(const Mat & );//����Matͼ
Mat CannyEdge(const string &);//Canny ��Ե��ⷨ
Mat SobelEdge(const string &);//Sobel ��Ե��ⷨ
Mat LaplacianEdge(const string &);//������˹���
Mat ScharEdge(const string &);//scharr�˲���
int LocalMaxDirection(const Mat &, int_pr *);//����8�����ھֲ�����λ��
int Int_pr_cpy(const Mat &, int_pr *, int, int_pr *, int, int_pr *);//ƴ�Ӷ������
int Int_pr_cpy(const Mat &, int_pr *, int, int_pr *); //�������飨���溯�������ذ汾)
Mat eli_local_max(const Mat & src, const int_pr *);//�޳��ֲ����� 
void iter_local_max_edge(const Mat & , int, int_pr * );//�����ҳ��ֲ�����
Mat BinZeroParr(const Mat &, const int_pr *);//��ı�Ե���齫ԭͼ��ֵ��
Mat find_max_local(const Mat &, Mat &,int );//Ѱ�Ҿֲ�����
Mat findBigSmallArea(const Mat &, int th);//Ѱ�Ҵ�С����th����ֵ
int find_pos_bigSmallArea(const Mat &gray, int x, int y, int & max_p, int & min_p);//findBigSmallArea()������һ����//Ѱ�Ҵ������ǰ������λ��
void WeightBigSmall(const Mat &, Mat &, int, int, const int &, const int &);//findBigSmallArea()������һ���� ����С����ͨ��������

Mat voteBigSmall(const Mat &);//ͶƱ�жϴ�С����
bool FixDiff(const Mat & src, const Mat & votemat, Mat & obj, int area_tag, int i, int j);//�޸���С����ĻҶ�ֵ
bool judeAndFixDiff(const Mat & src, const Mat & votemat, Mat & obj, int i, int j);//�жϲ��޸�ì�ܵ�

Mat voteToFix(const Mat &, const Mat &, Mat &, Mat &, Mat &, int);//�޸�ì�ܵ㣨ԭ�Ҷ�ͼ��ͶƱͼ��findBigSmallArea����ͼ,���Եͼ,С��Եͼ��ì�ܵ�ͼ,��ֵ)
Mat gx_cvt_color(const Mat & gray, int cl);//��BGR���ֵ�ɫ��ʾ,0����1�̣�2��
Mat color3_edge(const Mat & big_edge, const Mat & small_edge, const Mat & diff,int tag = 3);//ʹ��������ɫ��ʾ��С�� ��ߣ��죩��С�ߣ��̣���ì�ܵ㣨����
//ps_way�еĺ���

bool to_gray(const Mat & src, Mat & src_Gray);//��ͼ��ת��Ϊ�Ҷ�ͼ
bool read_src(Mat & src, const string & imgName, int pipe = 0);//��ȡͼ��
bool src_to_bmp(const Mat & src, const string & imgName);//�洢ͼ��ΪbmpͼƬ
bool show_src(const Mat & src, const string & imgName);//��ʾͼ���ļ�
bool read_Graysrc(Mat & gray_src, const string & imgName);//���ļ���ȡΪ�Ҷ�ͼ
void output_arr_txt(const Mat &src, const string & txt_name);
Mat bigSmallRegionNum(const Mat &, int tag = 0);// С��������������
void output_arr_csv(const Mat & src, const string & txt_name);//��ͼ������д��csv�ļ���
//basic_gx�еĺ������Ѿ�ȫ��������ͷ�ļ���


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


Mat fix7vs1( Mat & gray);
int testfix7vs1( Mat & gray, int x, int y);


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