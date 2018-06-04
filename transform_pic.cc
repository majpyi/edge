//处理图像的方法
#include "transform.h"
#include <cmath>
#include <algorithm>
//#include <basic_gx.h>
//#include <use.h>

using namespace std;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 华丽的分割线 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   第一部分   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

//----------------------[maxGradAndDirection()函数]---------------------
//计算8邻域内最大梯度 0~255 和 最大梯度方向 0 ~ 7
//maxGradAndDirection（） 包含 maxGrad() 和 maxGradAndDirection（）的功能
//----------------------------------------------------------------------
void maxGradAndDirection(const Mat &src, Mat &obj1, Mat &obj2) {
    //原矩阵，最大梯度矩阵，最大梯度方向矩阵
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float tmpmax = -1, tmp;
            int kmax = -1;//记录8邻域内最大梯度方向
            for (int k = 0; k < 8; ++k) {
                tmp = fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k]));
                if (tmp > tmpmax) {
                    kmax = k;
                    tmpmax = tmp;//tmpmax记录最大梯度
                }
            }
            obj1.at<float>(i, j) = tmpmax;
            obj2.at<float>(i, j) = kmax;
        }
    }

}

//----------------------[maxGrad()函数]---------------------
//计算8邻域内最大梯度 0~255
//----------------------------------------------------------
Mat maxGrad(const Mat &src) {
    //src灰度矩阵，数据类型CV_32F, p为阈值, 输出为0/255
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float tmpmax = -1, tmp;
            for (int k = 0; k < 8; ++k) {
                tmp = fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k]));
                if (tmp > tmpmax) {
                    tmpmax = tmp;//tmpmax记录最大梯度
                }
            }
            obj.at<float>(i, j) = tmpmax;
        }
    }
    return obj;
}

//----------------------[maxGradDirection()函数]---------------------
//计算8邻域内最大梯度 0~7
//------------------------------------------------------------------

Mat maxGradDirection(const Mat &src) {
    //最大梯度方向 0-7
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float tmpmax = -1, tmp;
            int kmax = -1;
            for (int k = 0; k < 8; ++k) {
                tmp = fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k]));
                if (tmp > tmpmax) {
                    kmax = k;
                    tmpmax = tmp;
                }
            }
            obj.at<float>(i, j) = kmax;
        }
    }
    return obj;
}

//----------------------[samePixelCnt()函数]-------------------------
//计算8邻域内相同点的个数，其中p是阈值,有0~8个相似点
//------------------------------------------------------------------

Mat samePixelCnt(const Mat &src, int p) {
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            int cnt = 0;
            for (int k = 0; k < 8; ++k) {
                if (fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k])) <= p) {
                    cnt++;
                }
            }
            obj.at<float>(i, j) = float(cnt);
        }
    }
    return obj;
}


//----------------------[clusterDistribution()函数]---------------------
//群分布 xxx
//---------------------------------------------------------------------
Mat clusterDistribution(const Mat &src, int p) {
    //群分布0~8
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            int cnt = 0;
            for (int k = 0; k < 8; ++k) {
                if (fabs(src.at<float>(i + di[k], j + dj[k]) - src.at<float>(i + di[k + 1], j + dj[k + 1])) > p) {
                    cnt++;
                }
            }
            obj.at<float>(i, j) = float(cnt);
        }
    }
    return obj;
}

//----------------------[bigSmallEdge()函数]---------------------
//群分布 大小边， 大边为 1 白色，小边 为 0 黑色
//--------------------------------------------------------------
//大小边，大边为1白色，小边为0黑色，
Mat bigSmallEdge(const Mat &src) {
    using namespace std;
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float pixel[8 + 1], diff[8] = {0};
            for (int k = 0; k < 8; ++k) {
                pixel[k] = src.at<float>(i + di[k], j + dj[k]);
            }
            sort(pixel, pixel + 8);
            //pixel[8] = pixel[0];
            for (int k = 0; k < 7; ++k) {
                diff[k] = fabs(pixel[k + 1] - pixel[k]);
            }
            int kmax = -1;
            float tmpmax = -1;
            for (int k = 0; k < 7; ++k) {
                if (diff[k] > tmpmax) {
                    tmpmax = diff[k];
                    kmax = k;
                }
            }
            if (src.at<float>(i, j) <= pixel[kmax]) {
                obj.at<float>(i, j) = SMALL_EDGE;//小边为0
            } else if (src.at<float>(i, j) >= pixel[kmax + 1]) {
                obj.at<float>(i, j) = BIG_EDGE;//大边为1
            } else if ((src.at<float>(i, j) - pixel[kmax]) >= (pixel[kmax + 1] - src.at<float>(i, j))) {
                obj.at<float>(i, j) = SMALL_EDGE;
            } else {
                obj.at<float>(i, j) = BIG_EDGE;
            }
        }
    }
    return obj;
}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 华丽的分割线 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   第二部分   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[model32Edge()函数]---------------------
//model32edge()是陈老师在17.9提出的方法，用于边缘检测
//返回Mat型的图，保存32个模板的点
//函数有七个参数：1.Mat型的灰度图 2.存储模板的常量int数组，
//3.存储边缘点位置的数组pair<int,int>  4.模板数组大小int
//5.统计模版是边缘点个数的int型数组 6. 统计图像内模板点出现的int数组
//7.边缘点数量edges_num 8.阈值p
//--------------------------------------------------------------
Mat model32Edge(const Mat &src, const int *model, int_pr *edge_station,
                int models_num, int *m_point, int *m_edgepo, int edges_num, int p) {
    //p是阈值
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    int edges = 0; //统计到第几个边缘点
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            bit8 b8;
            for (int k = 0; k < 8; ++k) {
                if (fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k])) <= p) {
//                    b8[k] = 1;//1是内部点
                } else {
//                    b8[k] = 0;//0是边
                }
            }
            bit8 min_b8 = min_bit8(b8);//保存为36模版之一
            int temp_b8 = min_b8.to_ulong();//暂存模版值
            obj.at<float>(i, j) = float(temp_b8);//都保存为模板点（即缩减为36种模型之一）

//统计模版点数量（二分查找）
#if(1)
            //用二分查找找到模板点所处位置，并把模板点次数加1
            int model_stat = binary_cut_search(model, models_num, temp_b8);
            m_point[model_stat]++;
            //模板点是边缘点的数量
            if (edges < edges_num && i == edge_station[edges].first && j == edge_station[edges].second) {
                //在边缘点位置，统计模版数量
                edges++;
                m_edgepo[model_stat]++;
            }
#endif

        }
    }
    return obj;
}


//----------------------[bigSmallRegionNum()函数]-----------------------
//bigSmallRegionNum()大小区域点个数
//参数 1 为 Mat 类型 原图，参数2 int类型 确定返回大区域 还是小区域
//---------------------------------------------------------------------

Mat bigSmallRegionNum(const Mat &src, int tag) {
    //0返回小区域，1返回大区域,默认参数为0，返回小区域点个数
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float pixel[8 + 1], diff[8] = {0};

            for (int k = 0; k <= 8; ++k) {
                //pexel保存一圈九个点
                pixel[k] = src.at<float>(i + di[k], j + dj[k]);
            }
            for (int k = 0; k < 8; ++k) {
                //一圈内相邻两个点相减
                diff[k] = fabs(pixel[k + 1] - pixel[k]);
            }

            float tmp;
            int kmax = -1, nummax, posmax;
            float tmpmax = -1;
            for (int k = 0; k < 8; ++k) {
                if (diff[k] > tmpmax) {
                    tmpmax = diff[k];//记录差值最大的点
                    kmax = k;//记录点的位置
                }
            }

            int kmin = -1, nummin, posmin;
            float tmpmin = 1;
            for (int k = 0; k < 8; ++k) {
                if (diff[k] < tmpmin) {
                    tmpmin = diff[k];
                    kmin = k;
                }
            }

            if (kmax < kmin) {
                //大区域
                nummax = kmin - kmax;
                tmp = -1;
                posmax = -1;
                for (int k = kmax + 1; k <= kmin; ++k) {
                    float cha = fabs(src.at<float>(i + di[k], j + dj[k]) - src.at<float>(i + di[k + 1], j + dj[k + 1]));
                    if (cha > tmp) {
                        tmp = cha;
                        posmax = k;
                    }
                }
                //小区域
                nummin = 8 - nummax;
                tmp = -1;
                posmin = -1;
                for (int k = kmin + 1; k < 8; ++k) {
                    float cha = fabs(src.at<float>(i + di[k], j + dj[k]) - src.at<float>(i + di[k + 1], j + dj[k + 1]));
                    if (cha > tmp) {
                        tmp = cha;
                        posmin = k;
                    }
                }
                for (int k = 0; k <= kmax; ++k) {
                    float cha = fabs(src.at<float>(i + di[k], j + dj[k]) - src.at<float>(i + di[k + 1], j + dj[k + 1]));
                    if (cha > tmp) {
                        tmp = cha;
                        posmin = k;
                    }
                }
            } else {
                //小区域
                nummin = kmax - kmin;
                tmp = -1;
                posmin = -1;
                for (int k = kmin + 1; k <= kmax; ++k) {
                    float cha = fabs(src.at<float>(i + di[k], j + dj[k]) - src.at<float>(i + di[k + 1], j + dj[k + 1]));
                    if (cha > tmp) {
                        tmp = cha;
                        posmin = k;
                    }
                }
                //大区域
                nummax = 8 - nummin;
                tmp = -1;
                posmax = -1;
                for (int k = kmax + 1; k < 8; ++k) {
                    float cha = fabs(src.at<float>(i + di[k], j + dj[k]) - src.at<float>(i + di[k + 1], j + dj[k + 1]));
                    if (cha > tmp) {
                        tmp = cha;
                        posmax = k;
                    }
                }
                for (int k = 0; k <= kmin; ++k) {
                    float cha = fabs(src.at<float>(i + di[k], j + dj[k]) - src.at<float>(i + di[k + 1], j + dj[k + 1]));
                    if (cha > tmp) {
                        tmp = cha;
                        posmax = k;
                    }
                }
            }
            if (tag == 0) {
                obj.at<float>(i, j) = float(nummax);
            } else if (tag == 1) {
                obj.at<float>(i, j) = float(nummin);
            }

        }
    }
    return obj;
}


//----------------------[binaryZation()函数]-----------------------
//binaryZation()参考图二值化 0/1
//参数 1 为 Mat 类型 原图，参数2 int类型 确定返回大区域 还是小区域
//---------------------------------------------------------------------

Mat binaryZation(const Mat &src, int p) {
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_TRUE));
    int rows = src.rows;
    int cols = src.cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) {
                obj.at<float>(i, j) = OUTPUT_FALSE;
            } else if (src.at<float>(i, j) > p) {
                obj.at<float>(i, j) = OUTPUT_FALSE;
            }
        }
    }
    return obj;
}


//----------------------[binarytoZero()函数]-----------------------
//binarytoZero() 二值化为0
//小于阈值 p 的记为0
//---------------------------------------------------------------------

Mat binarytoZero(const Mat &src, int p) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (src.at<float>(i, j) <= p) {
                obj.at<float>(i, j) = OUTPUT_FALSE;//边
            } else {
                obj.at<float>(i, j) = INSIDE_WHITE;//无关点
            }
        }
    }
    return obj;
}


//----------------------[edge_cnt()函数]-----------------------
//edge_cnt()统计边缘点个数
//一般认为等于0是边缘点，p的值默认为0，在声明时初始化为0
//-------------------------------------------------------------

int edge_cnt(const Mat &src, int p) {
    int rows = src.rows;
    int cols = src.cols;
    int edge_p = 0;
    int edge_per = p;//边缘点代数值
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (src.at<float>(i, j) == edge_per) {
                edge_p++;
            }

        }
    }
    return edge_p;
}

//-----------------[edge_station_p_arr()函数]--------------------
//edge_station_p_arr()用pair数组记录位置
//Mat类型保存二值化的专家图，int_ptr类型为pair<int, int>记录数组位置
//一般认为等于0是边缘点，p的值默认为0，在声明时初始化为0
//-------------------------------------------------------------

void edge_station_p_arr(const Mat &src, int_pr *e_station, int p) {
    int rows = src.rows;
    int cols = src.cols;
    int edge_per = p;//边缘点代数值
    int edge_p = 0;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (src.at<float>(i, j) == edge_per) {
                e_station[edge_p] = make_pair(i, j);
                edge_p++;
            }

        }
    }
}

//----------------------[enlarge_edge()函数]-----------------------
//扩增边缘点
//----------------------------------------------------------------
Mat enlarge_edge(const Mat &src) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            obj.at<float>(i, j) = src.at<float>(i, j);
            if (src.at<float>(i, j) == OUTPUT_FALSE) {
                obj.at<float>(i + 1, j) = OUTPUT_FALSE;
                obj.at<float>(i, j + 1) = OUTPUT_FALSE;
                obj.at<float>(i - 1, j) = OUTPUT_FALSE;
                obj.at<float>(i, j - 1) = OUTPUT_FALSE;
            }

        }
    }
    return obj;

}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 华丽的分割线 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   第三部分   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[Distogram_gray()函数]--------------------
//绘制直方图
//----------------------------------------------------------------
Mat Distogram_gray(const Mat &gray) {
    //为计算直方图配置变量
    //首先是需要计算的图像的通道，就是需要计算图像的哪个通道（bgr空间需要确定计算 b或g货r空间）
    int channels = 0;
    int scale = 1;//控制柱状图的宽度
    //然后是配置输出的结果存储的 空间 ，用MatND类型来存储结果
    MatND dstHist;
    int size = 256;
    //接下来是直方图的每一个维度的 柱条的数目（就是将数值分组，共有多少组）
    int histSize[] = {256};       //如果这里写成int histSize = 256;   那么下面调用计算直方图的函数的时候，该变量需要写 &histSize
    //最后是确定每个维度的取值范围，就是横坐标的总数
    //首先得定义一个变量用来存储 单个维度的 数值的取值范围
    float midRanges[] = {0, 256};
    const float *ranges[] = {midRanges};

    calcHist(&gray, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);

    //calcHist  函数调用结束后，dstHist变量中将储存了 直方图的信息  用dstHist的模版函数 at<Type>(i)得到第i个柱条的值
    //at<Type>(i, j)得到第i个并且第j个柱条的值

    //开始直观的显示直方图——绘制直方图
    //首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像
    Mat hist_img = Mat::zeros(size, size * scale, CV_8UC3);
    //因为任何一个图像的某个像素的总个数，都有可能会有很多，会超出所定义的图像的尺寸，针对这种情况，先对个数进行范围的限制
    //先用 minMaxLoc函数来得到计算直方图后的像素的最大个数
    double g_dHistMaxValue, g_dHistMinValue;
    minMaxLoc(dstHist, &g_dHistMinValue, &g_dHistMaxValue, 0, 0);
    //将像素的个数整合到 图像的最大范围内
    //遍历直方图得到的数据
    for (int i = 0; i < 256; i++) {
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);

        //line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));
        rectangle(hist_img, Point(i * scale, size - 1), Point((i + 1) * scale - 1, size - value),
                  CV_RGB(255, 255, 255));
    }
    return hist_img;

}

//----------------------[discrimin_pow()函数]--------------------
//区分度
//取min{+max_a(i)-a(i+1）,-max_a(i)-a(i+1）}
//---------------------------------------------------------------
Mat discrimin_pow(const Mat &gray) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int negative_max, postive_max;
            negative_max = postive_max = OUTPUT_FALSE;
            //计算区分度
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) -
                           gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                    }
                } else {
                    if (diff > postive_max) {
                        postive_max = diff;
                    }
                }
            }
            int dis_pow = fabs(postive_max) < fabs(negative_max)
                          ? fabs(postive_max) : fabs(negative_max);
            obj.at<float>(i, j) = dis_pow;
        }
    }
    return obj;

}


//----------------------[find_maxpoint()函数]--------------------
//找图中最大点
//---------------------------------------------------------------
double find_maxpoint(const Mat &gray) {
    int maxp = 0;
    int rows = gray.rows;
    int cols = gray.cols;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (gray.at<float>(i, j) > maxp) {
                maxp = gray.at<float>(i, j);
            }
        }
    }
    return maxp;
}

//----------------------[find_minpoint()函数]--------------------
//找图中最小点
//---------------------------------------------------------------
double find_minpoint(const Mat &gray) {
    int minp = gray.at<float>(1, 1);
    int rows = gray.rows;
    int cols = gray.cols;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (gray.at<float>(i, j) < minp) {
                minp = gray.at<float>(i, j);
            }
        }
    }
    return minp;
}


//----------------------[add_point()函数]--------------------
//叠加
//---------------------------------------------------------------
Mat add_point(const Mat &gray1, const Mat &gray2) {

    int rows = gray1.rows;
    int cols = gray1.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            obj.at<float>(i, j) = gray1.at<float>(i, j) + gray2.at<float>(i, j);

        }
    }
    return obj;
}


//----------------------[minus_point()函数]--------------------
//求差
//---------------------------------------------------------------
Mat minus_point(const Mat &gray1, const Mat &gray2) {

    int rows = gray1.rows;
    int cols = gray1.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            obj.at<float>(i, j) = fabs(gray1.at<float>(i, j) - gray2.at<float>(i, j));

        }
    }
    return obj;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 华丽的分割线 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   第四部分   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[Distogram_gx()函数]--------------------
//ghostxiu自写的直方图绘制函数
//-------------------------------------------------------------
Mat Distogram_gx(const Mat &gray, int *dis_arr, int max_len) {
    int scale = 8;//柱状图宽度


    mark_dis_arr(gray, dis_arr, max_len);//计算区分度直方图数组

    //缩放数组
    int zero_num = dis_arr[0];
    //cout << "zero : " << zero_num << endl;
    dis_arr[0] = OUTPUT_FALSE;//置0
    int max_point_num = max_arr_num(dis_arr, max_len);
    //zoom_dis_arr是缩放后的数组,height是绘图高度
    int *zoom_dis_arr = new int[max_len];
    if ((zero_num / max_point_num) < 20) {
        dis_arr[0] = zero_num;
        max_point_num = max_arr_num(dis_arr, max_len);
    }

    zoom_dis_arr = calcu_dis_arr(dis_arr, max_len, max_point_num);

    //绘制高度
    int height = max_arr_num(zoom_dis_arr, max_len);
    height = calcu_dis_arr_height(max_len, height, scale);

    if (zoom_dis_arr[0] == 0) {
        zoom_dis_arr[0] = height - 1;
    }

    Mat obj(height, max_len * scale, CV_32F, Scalar(OUTPUT_FALSE));//存储直方图信息的数组

    for (int i = 0; i < max_len; ++i) {
        int scl_i = i * scale;//每次增加scale宽度
        for (int p = 0; p < scale; ++p) {
            obj.at<float>(zoom_dis_arr[i], scl_i + p) = INSIDE_WHITE;//直方图封顶
        }

        for (int j = 0; j < zoom_dis_arr[i]; ++j) {
            obj.at<float>(j, scl_i) = INSIDE_WHITE;//左边
            obj.at<float>(j, scl_i + scale - 1) = INSIDE_WHITE;//右边
        }
    }
    dis_arr[0] = zero_num;

    delete[] zoom_dis_arr;//回收缩放数组的空间
    Mat in_obj = Invert_Mat(obj);
    return in_obj;
}

//----------------------[Mirror_Mat()函数]--------------------
//Mat矩阵旋转函数，0行列互换，1，
//-----------------------------------------------------------
Mat Mirror_Mat(const Mat &org_mat, int rotate_tag) {
    int rows = org_mat.rows;
    int cols = org_mat.cols;
    Mat row_to_cows(cols, rows, CV_32F, Scalar(OUTPUT_FALSE));
    Mat rotate(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat row_roate(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            row_to_cows.at<float>(j, i) = org_mat.at<float>(i, j);
            rotate.at<float>(i, j) = org_mat.at<float>(rows - i - 1, j);
            row_roate.at<float>(i, j) = org_mat.at<float>(rows - i - 1, cols - j - 1);
        }
    }
    if (rotate_tag == 0) {
        return row_to_cows;
    } else if (rotate_tag == 1) {
        return rotate;
    } else {
        return row_roate;
    }

}

//----------------------[Mirror_Mat()函数]--------------------
//上下倒置
//-----------------------------------------------------------
Mat Invert_Mat(const Mat &org_mat) {
    int rows = org_mat.rows;
    int cols = org_mat.cols;
    Mat rotate(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            rotate.at<float>(i, j) = org_mat.at<float>(rows - i - 1, j);
        }
    }

    return rotate;


}


//----------------------[mark_dis_arr()函数]------------------
//记录直方图数组
//-----------------------------------------------------------
void mark_dis_arr(const Mat &gray, int *dis_arr, int max_len) {
    //dis_arr是存储直方图绘制信息的二维数组
    std::fill(&dis_arr[0], &dis_arr[0] + max_len, OUTPUT_FALSE);
    int rows = gray.rows;
    int cols = gray.cols;

    //记录直方图数组元素

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int ij_value = gray.at<float>(i, j);
            dis_arr[ij_value]++;
        }
    }

}


//----------------------[calcu_dis_arr()函数]------------------
//为了方便观察，将直方图按一定规则放缩
//-----------------------------------------------------------
int *calcu_dis_arr(const int *arr, int max_len, int max_point_num) {
    int zoom_scale = -(max_point_num / max_len);//计算缩放比例
    int max_num = 10000;
    int add_min = 100;
    int *arr_c = new int[max_len];
    if (zoom_scale > 0) {
        for (int i = 0; i < max_len; ++i) {
            arr_c[i] = arr[i] * zoom_scale;
        }
    }
        /*
	else if (zoom_scale < 0)
	{

		for (int i = 0; i < max_len; ++i)
		{

			if (arr[i] > 10)
			{
				arr_c[i] = arr[i];
			}
			else if ((max_point_num / arr[i]) < 100 )
			{

				arr_c[i] = int(arr[i] / abs(zoom_scale));
			}
			else if ((max_point_num / arr[i]) < 1000 && zoom_scale > 10)
			{
				int temp = int(abs(zoom_scale/5));

				arr_c[i] = int(arr[i] / temp);
			}
			else if ((max_point_num / arr[i]) < 10000 && zoom_scale > 50)
			{
				int temp = int(abs(zoom_scale / 20));

				arr_c[i] = int(arr[i] / temp);
			}
			else
			{
				arr_c[i] = arr[i];

			}


		}
	}
	*/
    else if (zoom_scale < 0) {
        for (int i = 0; i < max_len; ++i) {
            arr_c[i] = arr[i] + add_min;
            if (arr_c[i] > max_num) {
                arr_c[i] = add_min;
            }
            arr_c[i] = int(arr_c[i] / abs(zoom_scale));
        }

    } else {
        std::cerr << "Error Zoom Scale !" << endl;
    }
    return arr_c;

}


//----------------------[calcu_dis_arr_height()函数]------------------
//计算显示高度
//-------------------------------------------------------------------
int calcu_dis_arr_height(int max_len, int height, int &scale) {
    if (max_len > GRAY_POINT_MAX_VALUE) {
        height += max_len / 4;
        scale /= 2;
    } else {
        height += max_len;
    }

    return height;
}


Mat copy_mat(const Mat &original) {
    Mat obj(original.rows, original.cols, CV_32F, Scalar(0));
    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            obj.at<float>(i, j) = original.at<float>(i, j);
        }
    }
    return obj;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 华丽的分割线 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   第五部分   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

//----------------------[Canny_edge()函数]------------------
//Canny 边缘检测法 2017.11.14
//---------------------------------------------------------
Mat CannyEdge(const string &pic_name) {
    //使用Canny算子进行边缘检测
    Mat org;
    // 【1】读取原图像
    read_src(org, pic_name);


    Mat canny_src = org.clone();
    /*
	Mat dst_c, edge,gray;
	// 【2】创建与src同类型和大小的矩阵(dst)
	dst_c.create(canny_src.size(), canny_src.type());
	// 【3】转为灰度图像
	cvtColor(canny_src, gray, COLOR_RGB2GRAY);


	// 【3】先用使用 3x3内核来降噪
	blur(gray, edge, Size(3, 3));

	// 【4】运行Canny算子
	Canny(edge, edge, 3, 9, 3);

	// 【5】将g_dstImage内的所有元素设置为0
	dst_c = Scalar::all(0);

	// 【6】使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中
	canny_src.copyTo(dst_c, edge);
	return dst_c;
	*/
    Canny(org, canny_src, 150, 100, 3);
    return canny_src;

}

//----------------------[Canny_edge()函数]------------------
//Sobel 边缘检测法 2017.11.14
//---------------------------------------------------------
Mat SobelEdge(const string &pic_name) {
    //使用Sobel算子进行边缘检测
    //【1】创建 grad_x 和 grad_y 矩阵
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, dst;

    //【2】载入原始图
    Mat org;
    read_src(org, pic_name, 1);
    //【3】求 X方向梯度
    Sobel(org, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    //imshow("【效果图】 X方向Sobel", abs_grad_x);

    //【4】求Y方向梯度
    Sobel(org, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);

    convertScaleAbs(grad_y, abs_grad_y);
    //imshow("【效果图】Y方向Sobel", abs_grad_y);

    //【5】合并梯度(近似)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
    return dst;
}

//----------------------[LaplacianEdge()函数]------------------
//拉普拉斯变幻 2017.11.14
//------------------------------------------------------------

Mat LaplacianEdge(const string &pic_name) {
    //【0】变量定义
    Mat src_l, src_l_gray, dst_l, abs_dst_l;
    //【1】读取源文件

    read_src(src_l, pic_name);
    //【2】高斯滤波消除噪声
    GaussianBlur(src_l, src_l, Size(3, 3), 0, 0, BORDER_DEFAULT);
    //【3】转为灰度图
    cvtColor(src_l, src_l_gray, COLOR_RGB2GRAY);
    //【4】使用拉普拉斯
    Laplacian(src_l_gray, dst_l, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    //【5】计算绝对值并保存为8位
    convertScaleAbs(dst_l, abs_dst_l);

    return abs_dst_l;
}

//----------------------[scharr()函数]------------------
//scharr滤波器 2017.11.14
//-----------------------------------------------------
Mat ScharEdge(const string &pic_name) {
    //【0】变量定义
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, dst_s;
    //【1】读取源文件
    Mat org;
    read_src(org, pic_name);
    //【2】求X方向梯度
    Scharr(org, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    //【3】求Y方向梯度
    Scharr(org, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    //【4】近似合并梯度
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst_s);


    return dst_s;
}


//----------------------[LocalMaxDirection()函数]---------------------
//计算8邻域内局部最大点位置 2017.11.16
//输入为1.Mat灰度图像 和 2.pair<int, int> 位置矩阵
//返回值为数组长度
//-------------------------------------------------------------------
int LocalMaxDirection(const Mat &src, int_pr *dis_dir) {
    //src灰度矩阵，数据类型CV_32F, p为阈值, 输出为0/255
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    int maxl_cnt = 0;
    //统计局部最大点数量
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            for (int k = 0; k < 8; ++k) {
                if (src.at<float>(i, j) <= src.at<float>(i + di[k], j + dj[k])) {
                    continue;
                } else {
                    maxl_cnt++;
                }
            }
        }
    }

    //记录局部最大点位置
    dis_dir = new int_pr[maxl_cnt];
    int max_l_cnt = 0;
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            for (int k = 0; k < 8; ++k) {
                if (src.at<float>(i, j) <= src.at<float>(i + di[k], j + dj[k])) {
                    continue;
                } else {
                    dis_dir[max_l_cnt] = make_pair(i, j);
                    max_l_cnt++;
                }
            }
        }
    }

    if (maxl_cnt = max_l_cnt) {
        return maxl_cnt;
    } else {
        return OUTPUT_FALSE;
    }

}

//----------------------[LocalMaxDirection()函数]---------------------
//拼接多个位置数组 2017.11.16
//-------------------------------------------------------------------
int Int_pr_cpy(const Mat &src, int_pr *dir_1, int len1, int_pr *dir_2, int len2, int_pr *dir) {
    dir = new int_pr[len1 + len2];
    int l1, l2, l;
    l1 = l2 = l = 0;
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            if (i == dir_1[l1].first && j == dir_1[l1].second) {
                dir[l] = make_pair(i, j);
                l++;
            } else if (i == dir_2[l2].first && j == dir_2[l2].second) {
                dir[l] = make_pair(i, j);
                l++;
            }
        }
    }
    return l;
}

//----------------------[LocalMaxDirection()函数]---------------------
//复制数组（上面函数的重载版本) 2017.11.17
//-------------------------------------------------------------------
int Int_pr_cpy(const Mat &src, int_pr *dir_1, int len1, int_pr *dir) {
    dir = new int_pr[len1];
    int l1, l;
    l1 = l = 0;
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            if (i == dir_1[l1].first && j == dir_1[l1].second) {
                dir[l] = make_pair(i, j);
                l++;
            }
        }
    }
    return l;
}


//----------------------[eli_local_max()函数]---------------------
//剔除局部最大点 2017.11.17
//---------------------------------------------------------------
Mat eli_local_max(const Mat &src, const int_pr *loc_dir) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    int top = 0;//记录最大值位置数组位置
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (i == loc_dir[top].first && j == loc_dir[top].second) {
                obj.at<float>(i, j) = OUTPUT_FALSE;
                top++;
            } else {
                obj.at<float>(i, j) = src.at<float>(i, j);
            }
        }
    }
    return obj;
}


//----------------------[eli_local_max()函数]---------------------
//迭代找出局部最大点 2017.11.17
//输入1.const Mat & 型 的图（一般为区分度图） 2.int 型 迭代次数
//返回值 pair<int,int> 类型，记录局部最大点次数
//---------------------------------------------------------------

void iter_local_max_edge(const Mat &dis_pow, int times, int_pr *edge_loc_max) {
    int_pr *local_max_pt = new int_pr[1];
    int_pr *tmp_add = new int_pr[1];
    int len, len_t;
    Mat tmp_mat = dis_pow;
    for (int i = 0; i < times; ++i) {
        cout << "第" << i + 1 << "次寻找局部最大点" << endl;
        int_pr *tmp_max_arr = new int_pr[1];
        int len_tmp = LocalMaxDirection(tmp_mat, tmp_max_arr);
        tmp_mat = eli_local_max(tmp_mat, tmp_max_arr);
        if (i > 0) {
            len = Int_pr_cpy(tmp_mat, tmp_max_arr, len_tmp, tmp_add, len_t, local_max_pt);
            len_t = Int_pr_cpy(tmp_mat, local_max_pt, len, tmp_add);
        } else {
            len_t = Int_pr_cpy(tmp_mat, local_max_pt, len, tmp_add);
        }
        string src_name = "第";
        src_name.append(num_to_string(i)).append("次寻找图");
        show_src(tmp_mat, src_name);
        delete[] tmp_max_arr;
    }

    len = Int_pr_cpy(tmp_mat, tmp_add, len_t, edge_loc_max);
    delete[] local_max_pt;
    delete[] tmp_add;
}


//----------------------[BinZeroParr()函数]-----------------------
//BinZeroParr() 经过pair<int,int> 类的边缘数组将原图二值化为0和255
//---------------------------------------------------------------
Mat BinZeroParr(const Mat &src, const int_pr *edge_arr_loc) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    int top = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == edge_arr_loc[top].first && j == edge_arr_loc[top].second) {
                obj.at<float>(i, j) = OUTPUT_FALSE;//边
                top++;
            } else {
                obj.at<float>(i, j) = INSIDE_WHITE;//无关点
            }
        }
    }
    return obj;
}

//----------------------[find_max_local()函数]-----------------------
//find_max_local() 寻找局部最大点
//-------------------------------------------------------------------
Mat find_max_local(const Mat &src, Mat &tag, int thre) {
    int rows = src.rows;
    int cols = src.cols;
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};

    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    int count = 0;
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            for (int k = 0; k < 8; ++k) {
                if (src.at<float>(i, j) <= src.at<float>(i + di[k], j + dj[k])) {
                    obj.at<float>(i, j) = src.at<float>(i, j);
                    continue;
                } else {
                    if (k == 7) {
                        if (src.at<float>(i, j) < thre) {
                            continue;
                        }
                        obj.at<float>(i, j) = OUTPUT_FALSE;
                        tag.at<float>(i, j) = INSIDE_WHITE;
                        count++;
                    }
                }
            }
        }
    }
    cout << count << endl;
    return obj;
}

//----------------------[find_max_local()函数]-----------------------
//localArea_edge() 求局部最大边缘
//-------------------------------------------------------------------
Mat localArea_edge(const Mat &src) {
    int rows = src.rows;
    int cols = src.cols;
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};

    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    int count = 0;
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            int minl, maxl;
            minl = maxl = src.at<float>(i + di[0], j + dj[0]);
            vector<int_pr> max_area;
            vector<int_pr> min_area;
            int max_ct, min_ct;
            max_ct = min_ct = 0;
            for (int k = 1; k < 8; ++k) {
                //记录八邻域最大值和最小值
                if (src.at<float>(i + di[k], j + dj[k]) < minl) {
                    minl = src.at<float>(i + di[k], j + dj[k]);
                } else if (src.at<float>(i + di[k], j + dj[k]) > maxl) {
                    maxl = src.at<float>(i + di[k], j + dj[k]);
                }

                //分成两个区域
                if (src.at<float>(i + di[k], j + dj[k]) > src.at<float>(i, j)) {
                    max_area[max_ct] = make_pair(i + di[k], j + dj[k]);
                    max_ct++;
                } else {
                    min_area[min_ct] = make_pair(i + di[k], j + dj[k]);
                    min_ct++;
                }

                //判断中心点属于哪个区域
                {

                }
                //检查中心点是不是在最上下左右

                {

                }
            }


        }
    }

    return obj;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 华丽的分割线 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[findBigSmallArea()函数]--------------------
//划分大小区域，产生投票图（返回值）12.18
//本函数由 find_pos_bigSmallArea()、WeightBigSmall()两个函数组成
//-----------------------------------------------------------------
Mat findBigSmallArea(const Mat &gray, int th) {


    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int max_p, min_p;
            max_p = min_p = -1;

            int dis_pow = find_pos_bigSmallArea(gray, i, j, max_p, min_p);
            if (dis_pow < th) {//跳过低于阈值的点
                continue;
            }

            WeightBigSmall(gray, obj, i, j, max_p, min_p);
        }
    }

    return obj;
}

//----------------------[find_pos_bigSmallArea()函数]--------------
//findBigSmallArea()函数的一部分
//寻找大区域的前后两个位置
//-----------------------------------------------------------------
int find_pos_bigSmallArea(const Mat &gray, int i, int j, int &max_p, int &min_p) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int negative_max, positive_max;
    negative_max = positive_max = OUTPUT_FALSE;
    //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
    //找最大最小区域
    for (int k = 1; k <= 8; ++k) {
        int diff = gray.at<float>(i + di[k], j + dj[k]) -
                   gray.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //大区域的最后一个点是负最小前面的点
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //大区域的第一个点是正最大后面的点
            }
        }
    }
    int dis_pow = fabs(positive_max) < fabs(negative_max)
                  ? fabs(positive_max) : fabs(negative_max);
    return dis_pow;

}

//----------------------[WeightBigSmall()函数]---------------------
//findBigSmallArea()函数的一部分
//给大小区域通过次数加权
//-----------------------------------------------------------------
void WeightBigSmall(const Mat &gray, Mat &obj, int i, int j, const int &max_p, const int &min_p) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int max1 = max_p;//大区域起始点
    int max2 = min_p;//大区域结束点
    int min1 = max_p - 1;//小区域起始点
    int min2 = min_p + 1;//小区域结束点

    int mindiff_b = MAX_PT;//大区域和中心点的最小差
    int mindiff_s = MAX_PT;//小区域和中心点的最小差

    //大区域部分次数加权（出现一次+10）
    if (max1 <= max2) {
        for (int l = max1; l <= max2; ++l) {
            obj.at<float>(i + di[l], j + dj[l]) += 10;
            mindiff_b = abs_min_num(mindiff_b, gray.at<float>(i, j),
                                    gray.at<float>(i + di[l], j + dj[l]));
        }
    } else {
        for (int l = max1; l <= 7; ++l) {
            obj.at<float>(i + di[l], j + dj[l]) += 10;
            mindiff_b = abs_min_num(mindiff_b, gray.at<float>(i, j),
                                    gray.at<float>(i + di[l], j + dj[l]));
        }
        for (int l = 0; l <= max2; ++l) {
            obj.at<float>(i + di[l], j + dj[l]) += 10;
            mindiff_b = abs_min_num(mindiff_b, gray.at<float>(i, j),
                                    gray.at<float>(i + di[l], j + dj[l]));
        }

    }

    //小区域部分次数加权（出现一次+1）
    if (min2 <= min1) {
        for (int l = min2; l <= min1; ++l) {
            obj.at<float>(i + di[l], j + dj[l]) += 1;
            mindiff_s = abs_min_num(mindiff_s, gray.at<float>(i, j),
                                    gray.at<float>(i + di[l], j + dj[l]));
        }
    } else {
        for (int l = min2; l <= 7; ++l) {
            obj.at<float>(i + di[l], j + dj[l]) += 1;
            mindiff_s = abs_min_num(mindiff_s, gray.at<float>(i, j),
                                    gray.at<float>(i + di[l], j + dj[l]));
        }
        for (int l = 0; l <= min1; ++l) {
            obj.at<float>(i + di[l], j + dj[l]) += 1;
            mindiff_s = abs_min_num(mindiff_s, gray.at<float>(i, j),
                                    gray.at<float>(i + di[l], j + dj[l]));
        }

    }

    //判断中心点本身在8邻域内属于哪一部分，并加权
    obj.at<float>(i, j) += mindiff_b < mindiff_s ? 10 : 1;


}


//----------------------[voteBigSmall()函数]--------------------
//大小区域通过比例判断12.18
//
//--------------------------------------------------------------
Mat voteBigSmall(const Mat &src) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    int min_tag, max_tag;
    min_tag = max_tag = 0;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            max_tag = src.at<float>(i, j) / 10;
            min_tag = int(src.at<float>(i, j)) % 10;
            if (max_tag > min_tag) {
                obj.at<float>(i, j) = 222;
            } else if (min_tag > max_tag) {
                obj.at<float>(i, j) = OUTPUT_TRUE;
            }
        }
    }

    return obj;
}


//----------------------[voteToFix()函数]--------------------
//2018.1.11
//通过投票图找出大小边，矛盾点，并修正
//包含FixDiff()（）,judeAndFixDiff()（）
//-----------------------------------------------------------
Mat voteToFix(const Mat &src, const Mat &votemat, Mat &bigm, Mat &smallm, Mat &diffm, int th) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//这里的OBJ存储修复后的灰度图
    diffm = obj.clone();
    bigm = obj.clone();
    smallm = obj.clone();
    int turn_to_diff = 10;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //【1】判断是否为内部点
            if (votemat.at<float>(i, j) == 0) {//投票值为0,不判断，直接赋值
                obj.at<float>(i, j) = src.at<float>(i, j);
                continue;
            }
            //【2】判断是矛盾点还是大小区域点
            int min_tag, max_tag;
            min_tag = int(int(votemat.at<float>(i, j)) % turn_to_diff);
            max_tag = int(votemat.at<float>(i, j) / turn_to_diff);
            if (max_tag > min_tag) {

                //【3】存入矛盾点或者大小边
                if (min_tag == OUTPUT_FALSE) {
                    //不是矛盾点，存入大边
                    bigm.at<float>(i, j) = OUTPUT_TRUE;
                    obj.at<float>(i, j) = src.at<float>(i, j);
                } else {
                    //是矛盾点，存入矛盾点，并修复灰度值
                    diffm.at<float>(i, j) = OUTPUT_TRUE;
                    //【4】修复灰度值
                    FixDiff(src, votemat, obj, 0, i, j);//0表示大边
                }
            } else if (max_tag < min_tag) {
                if (max_tag == OUTPUT_FALSE) {
                    //不是矛盾点，存入小边
                    smallm.at<float>(i, j) = OUTPUT_TRUE;
                    obj.at<float>(i, j) = src.at<float>(i, j);
                } else {
                    //是矛盾点，存入矛盾点，并修复灰度值
                    diffm.at<float>(i, j) = OUTPUT_TRUE;
                    //【4】修复灰度值
                    FixDiff(src, votemat, obj, 0, i, j);//1表示小边

                }
            } else {
                //存入矛盾点
                diffm.at<float>(i, j) = OUTPUT_TRUE;
                //因为大小区域投票结果相同，首先要判断是哪个区域
                judeAndFixDiff(src, votemat, obj, i, j);

            }

        }
    }
    return obj;
}

//----------------------[FixDiff()函数]--------------------
//2018.1.11
//voteToFix()的子函数
//修复大小区域的灰度值
//参数：原图，投票图，修复图，区域tag(0大，1小）
//--------------------------------------------------------
bool FixDiff(const Mat &src, const Mat &votemat, Mat &obj, int area_tag, int i, int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int turn_to_diff = 10;
    int sum, cnt;
    sum = cnt = 0;
    for (int k = 0; k < 8; ++k) {
        int judge;
        if (area_tag == 0) {
            judge = int(votemat.at<float>(i + di[k], j + dj[k])
                        / turn_to_diff);
        } else if (area_tag == 1) {
            judge = int(int(votemat.at<float>(i + di[k], j + dj[k]))
                        % turn_to_diff);
        }

        if (judge > 0) {
            sum += int(src.at<float>(i + di[k], j + dj[k]));
            cnt++;
        }
    }
    //修复灰度值
    if (cnt == 0) {
        //此类矛盾点暂时无好的处理方式
        obj.at<float>(i, j) = src.at<float>(i, j);
    } else {
        obj.at<float>(i, j) = int(sum / cnt);
    }

    return true;
}

//----------------------[judeAndFixDiff()函数]--------------------
//2018.1.11
//voteToFix()的子函数
//判断并修复矛盾点
//---------------------------------------------------------------
bool judeAndFixDiff(const Mat &src, const Mat &votemat, Mat &obj, int i, int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int min_cl_diff_b, min_cl_diff_s;//大小区域和中心点的最小色差
    int sum_big, cnt_big, sum_small, cnt_small;
    sum_big = sum_small = cnt_big = cnt_small = OUTPUT_FALSE;
    min_cl_diff_b = min_cl_diff_s = MAX_PT;
    int turn_to_diff = 10;
    for (int k = 0; k < 8; ++k) {
        int judge_b = votemat.at<float>(i + di[k], j + dj[k])
                      / turn_to_diff;
        int judge_s = int(votemat.at<float>(i + di[k], j + dj[k]))
                      % turn_to_diff;
        if (judge_b > judge_s && judge_s == OUTPUT_FALSE) {
            //大区域点
            sum_big += src.at<float>(i + di[k], j + dj[k]);
            cnt_big++;
            int temp_diff = fabs(src.at<float>(i + di[k], j + dj[k])
                                 - src.at<float>(i, j));
            if (temp_diff < min_cl_diff_b) {
                min_cl_diff_b = temp_diff;
            }
        } else if (judge_s > judge_b && judge_b == OUTPUT_FALSE) {
            //小区域点
            sum_small += src.at<float>(i + di[k], j + dj[k]);
            cnt_small++;
            int temp_diff = fabs(src.at<float>(i + di[k], j + dj[k])
                                 - src.at<float>(i, j));
            if (temp_diff < min_cl_diff_s) {
                min_cl_diff_s = temp_diff;
            }
        }
    }
    if (min_cl_diff_b > min_cl_diff_s && cnt_big != 0) {
        //矛盾点灰度值近似为大区域
        obj.at<float>(i, j) = sum_big / cnt_big;
    } else if (min_cl_diff_b < min_cl_diff_s && cnt_small != 0) {
        //矛盾点灰度值近似为小区域
        obj.at<float>(i, j) = sum_small / cnt_small;
    } else {
        //周围8个点都是矛盾点
        obj.at<float>(i, j) = src.at<float>(i, j);
    }
    return true;
}


//----------------------[gx_cvt_color()函数]--------------------
//自建的颜色转换函数
//将灰度颜色转化为某个颜色0,b,1,g,2,r
//--------------------------------------------------------------
Mat gx_cvt_color(const Mat &gray, int cl) {
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (gray.at<float>(i, j) == 1) {
                obj.at<Vec3b>(i, j)[cl] = MAX_PT;
            }
        }
    }

    return obj;

}


//----------------------[color3_edge()函数]---------------------
//使用两种颜色表现大小边
//将灰度颜色转化为某个颜色0,b,1,g,2,r
//tag 显示几条边 默认 3，显示三条边
//--------------------------------------------------------------
Mat color3_edge(const Mat &big_edge, const Mat &small_edge, const Mat &diff, int tag) {
    int rows = big_edge.rows;
    int cols = big_edge.cols;
    Mat obj(rows, cols, CV_8UC3, Scalar(255, 255, 255));
//    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (big_edge.at<float>(i, j) == 1) {
//                type3.at<double>(i, j)=2;
                obj.at<Vec3b>(i, j)[2] = MAX_PT;
                //红色，大边
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
            if (small_edge.at<float>(i, j) == 1) {
//                type3.at<double>(i, j)=1;
                obj.at<Vec3b>(i, j)[1] = MAX_PT;
                //绿色，小边
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
            }
            if (diff.at<float>(i, j) == 1 && tag == 3) {
//                type3.at<double>(i, j)=3;
                obj.at<Vec3b>(i, j)[0] = MAX_PT;
                //蓝色,矛盾点
                obj.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
            }
        }
    }

    return obj;

}


//----------------------[find_edgeda()函数]---------------------
//只使用一条边找到大小边相邻的轮廓
//--------------------------------------------------------------
Mat find_edge(Mat &votemat) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = votemat.rows;
    int cols = votemat.cols;
    Mat objda(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat objxiao(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    Mat tu(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    //Mat newobj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));


    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            for (int s = 1; s < 8; s++) {
                if (votemat.at<float>(i + di[s], j + dj[s]) == 222 && votemat.at<float>(i, j) == 222) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 1 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 1) {
                        objda.at<float>(i + di[s], j + dj[s]) += 1;
                    }
                }
                if (votemat.at<float>(i + di[s], j + dj[s]) == 1 && votemat.at<float>(i, j) == 1) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 222 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 222) {
                        objxiao.at<float>(i + di[s], j + dj[s]) -= 1;
                    }
                }
            }

        }
    }


    /*
	for (int i = 1; i < rows - 1; ++i)
	{
	//int num = 0;
	for (int j = 1; j < cols - 1; ++j)
	{
	for (int s = 0; s<8; s++)
	{
	if (obj.at<float>(i + di[s], j + dj[s]) == 1 && obj.at<float>(i, j) == 1)
	{
	newobj.at<float>(i, j) =1 ;
	}
	}

	}
	}
	*/


    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            if (objda.at<float>(i, j) >= 2) {
                tu.at<Vec3b>(i, j)[2] = MAX_PT;
                tu.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
            /*
			if (objxiao.at<float>(i, j) <= -2)
			{
				tu.at<Vec3b>(i, j)[2] = MAX_PT;
				tu.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
				tu.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
			}
			*/


        }
    }
    return tu;

}


//----------------------[find_edgexiao()函数]---------------------
//只使用一条边找到大小边相邻的轮廓
//--------------------------------------------------------------
Mat find_edgexiao(Mat &votemat) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = votemat.rows;
    int cols = votemat.cols;
    Mat objda(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat objxiao(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    Mat tu(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    //Mat newobj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            for (int s = 1; s < 8; s++) {
                if (votemat.at<float>(i + di[s], j + dj[s]) == 1 && votemat.at<float>(i, j) == 1) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 222 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 222) {
                        objxiao.at<float>(i + di[s], j + dj[s]) -= 1;
                    }
                }
            }

        }
    }
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (objxiao.at<float>(i, j) <= -2) {
                tu.at<Vec3b>(i, j)[2] = MAX_PT;
                tu.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
        }
    }
    return tu;
}


//排序法使用的结构体，用来同时记录数值与位置
typedef struct Test {
    int val;
    int pos;
} PIXEL;

bool Cmpare(const PIXEL &a, const PIXEL &b) {
    return a.val < b.val;
}


//----------------------[sortpixel()函数]---------------------
//只使用一条边找到大小边相邻的轮廓,其中包含对中心点的修复centerFix()函数
//--------------------------------------------------------------
Mat sortpixel(const Mat &gray, int th) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放排序法进行标记的大小区
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //存放处理之后的图像灰度值
    Mat maodun(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//记录矛盾点（左右的标记点与自身的不一致）
    tu = gray.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //单像素区域跳过
            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }

            //判断是不是内部点，如果是，则进行跳过
            int negative_max, positive_max;
            negative_max = OUTPUT_FALSE;
            positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) -
                           gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max)
                          ? fabs(positive_max) : fabs(negative_max);

            if (dis_pow < th) {
                continue;
            }



            //组成数组进行排序
            for (int k = 0; k <= 7; k++) {
                round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //排序后找到最大差值的位置
            for (int k = 1; k <= 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }


            /*
			//加入中心点之后进行判断是否是单像素点或者是过渡带
			int c[9];
			c[0] = gray.at<float>(i, j);
			for (int k = 1; k <= 8; k++)
			{
				c[k] = int(gray.at<float>(i + di[k], j + dj[k]));
			}
			sort(c, c + 9, Cmpare1);
			//判断是否是单像素点
			if (premax == 2 || premax == 5)
			{

			}
			//判断是否为过渡带
			*/

            int sumda = 0;
            int sumxiao = 0;
            //排序后在经由最大差值处理之后的分为大小区之后的点进行obj进行大小区标记
            for (int k = 0; k <= premax; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
                sumxiao += round[k].val;
            }
            for (int k = latemax; k <= 7; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
                sumda += round[k].val;
            }


            if (premax == 0) {
                int k = 0;
                tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumda / 7;
                continue;
            }
            if (latemax == 7) {
                int k = 7;
                tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumxiao / 7;
                continue;
            }



            //找到被两个大边夹到中心的小边点
            for (int k = 0; k <= premax; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 1 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 1) {
                    //这里是两个矛盾点相邻时的状况
                    if (obj.at<float>(
                            obj.at<float>(i + di[(round[k].pos - 2 + 8) % 8], j + dj[(round[k].pos - 2 + 8) % 8])) ==
                        0 || obj.at<float>(i + di[(round[k].pos + 2) % 8], j + dj[(round[k].pos + 2) % 8]) == 0) {
                        continue;
                    } else {
                        maodun.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
                        /*
						sumxiao -= round[k].val;
						if(latemax!=7)
						tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumda / (8 - latemax);
						*/
                    }
                }
            }
            //与被两个小边点夹到中心的大边点
            for (int k = latemax; k <= 7; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 0 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 0) {
                    //这里是两个矛盾点相邻时的状况
                    if (obj.at<float>(
                            obj.at<float>(i + di[(round[k].pos - 2 + 8) % 8], j + dj[(round[k].pos - 2 + 8) % 8])) ==
                        1 || obj.at<float>(i + di[(round[k].pos + 2) % 8], j + dj[(round[k].pos + 2) % 8]) == 1) {
                        continue;
                    } else {
                        maodun.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 2;
                        /*
						sumda -= round[k].val;
						if (premax != 0)
							tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumxiao / latemax;
						*/
                    }
                }
            }



            //把周围八邻域之中的矛盾点全部减去之后再求均值
            int reducexiao = 0;
            int reduceda = 0;
            int numxiao = 0;
            int numda = 0;
            for (int k = 0; k <= 7; k++) {
                if (maodun.at<float>(i + di[k], j + dj[k]) == 1);
                {
                    reducexiao += gray.at<float>(i + di[k], j + dj[k]);
                    numxiao++;
                }
                if (maodun.at<float>(i + di[k], j + dj[k]) == 2);
                {
                    reduceda += gray.at<float>(i + di[k], j + dj[k]);
                    numda++;
                }
            }

            for (int k = 0; k <= 7; k++) {
                if (maodun.at<float>(i + di[k], j + dj[k]) == 1);
                {
                    obj.at<float>(i + di[k], j + dj[k]) = (sumxiao - reducexiao) / (latemax - numxiao);
                }
                if (maodun.at<float>(i + di[k], j + dj[k]) == 2);
                {
                    obj.at<float>(i + di[k], j + dj[k]) = (sumda - reduceda) / (8 - latemax - numda);
                }
            }



            //中心点修复
            //tu.at<float>(i, j) = centerFix(gray, maxcha, i, j);
        }
    }
    return tu;
}


//----------------------[testsortpixel()函数]---------------------
//只使用一条边找到大小边相邻的轮廓,其中包含对中心点的修复centerFix()函数
//--------------------------------------------------------------
int testsortpixel(const Mat &gray, int th, int x, int y) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    int i = x;
    int j = y;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放排序法进行标记的大小区
    Mat maodun(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//记录矛盾点（左右的标记点与自身的不一致）
    //单像素区域跳过
//    if (judge_single(gray, 20, i, j) == 1) {
//        return 0;
//    }

    //判断是不是内部点，如果是，则进行跳过
    int negative_max, positive_max;
    negative_max = OUTPUT_FALSE;
    positive_max = OUTPUT_FALSE;
    int max_p, min_p;
    for (int k = 1; k <= 8; ++k) {
        int diff = gray.at<float>(i + di[k], j + dj[k]) -
                   gray.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
            }
        }
    }
//    int dis_pow = fabs(positive_max) < fabs(negative_max)
//                  ? fabs(positive_max) : fabs(negative_max);
//
//    if (dis_pow < th) {
//        return 0;
//    }

    //组成数组进行排序
    for (int k = 0; k <= 7; k++) {
        round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
    }

    cout << "排序前" << endl;
    for (int k = 0; k <= 7; k++) {
        cout << round[k].val << "  ";
    }


    sort(round, round + 8, Cmpare);
    //return round;


    cout << endl << "排序后" << endl;
    for (int k = 0; k <= 7; k++) {
        cout << round[k].val << "  ";
    }


    int maxcha = 0, premax, latemax;
    //排序后找到最大差值的位置
    for (int k = 1; k <= 7; k++) {
        if (round[k + 1].val - round[k].val >= maxcha) {
            maxcha = round[k + 1].val - round[k].val;
            premax = k;
            latemax = k + 1;
        }
    }

    //排序后在经由最大差值处理之后的分为大小区之后的点进行obj进行大小区标记
    for (int k = 0; k <= premax; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
    }
    for (int k = latemax; k <= 7; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
    }


    cout << endl << "标记结果" << endl;
    for (int k = 0; k <= premax; k++) {
        cout << 0 << "  ";
    }
    for (int k = latemax; k <= 7; k++) {
        cout << 1 << "  ";
    }


    cout<<endl;


    //找到被两个大边夹到中心的小边点
    for (int k = 0; k <= premax; k++) {
        if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 1 &&
            obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 1) {
            //这里是两个矛盾点相邻时的状况
            if (obj.at<float>(obj.at<float>(i + di[(round[k].pos - 2 + 8) % 8], j + dj[(round[k].pos - 2 + 8) % 8])) ==
                0 || obj.at<float>(i + di[(round[k].pos + 2) % 8], j + dj[(round[k].pos + 2) % 8]) == 0) {
            } else {
                maodun.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
                return 1;
            }
        }
    }
    //与被两个小边点夹到中心的大边点
    for (int k = latemax; k <= 7; k++) {
        if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 0 &&
            obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 0) {
            //这里是两个矛盾点相邻时的状况
            if (obj.at<float>(obj.at<float>(i + di[(round[k].pos - 2 + 8) % 8], j + dj[(round[k].pos - 2 + 8) % 8])) ==
                1 || obj.at<float>(i + di[(round[k].pos + 2) % 8], j + dj[(round[k].pos + 2) % 8]) == 1) {
            } else {
                maodun.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
                return 1;
            }
        }
    }


    return 0;
}


//----------------------[old_sortpixel()函数]---------------------
//只使用一条边找到大小边相邻的轮廓,其中包含对中心点的修复centerFix()函数,这个函数是与八邻域方法来进行比较的
//--------------------------------------------------------------
Mat old_sortpixel(const Mat &gray, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放排序法进行标记的大小区
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放八邻域处理之后的进行的大小区的标记
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //存放处理之后的图像灰度值
    tu = gray.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }
            //组成数组进行排序
            for (int k = 0; k < 8; k++) {
                round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //排序后找到最大差值的位置
            for (int k = 0; k < 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }

            //排序后在经由最大差值处理之后的分为大小区之后的点进行obj进行大小区标记
            for (int k = 0; k <= premax; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
            }
            for (int k = latemax; k <= 7; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
            }

            //中心点修复

            tu.at<float>(i, j) = centerFix(gray, maxcha, i, j);

            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
            //找最大最小区域
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) -
                           gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //大区域的最后一个点是负最小前面的点
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //大区域的第一个点是正最大后面的点
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max)
                          ? fabs(positive_max) : fabs(negative_max);

            //		int dis_pow = fabs(positive_max) < fabs(negative_max)
            //				? fabs(negative_max) : fabs(positive_max);

            if (dis_pow < th) {//跳过低于阈值的点
                /*
			 for (int k = 0; k < 8; k++)
			 {
			 tu.at<float>(i + di[k], j + dj[k]) = gray.at<float>(i + di[k], j + dj[k]);
			 }
			 */
                //tu.at<float>(i, j) = gray.at<float>(i, j);
                continue;
            }

            ////////////////////////////////////////////////////////////////////////
            int max1 = max_p % 8;
            int max2 = min_p;
            int min1 = max_p - 1;
            int min2 = (min_p + 1) % 8;
            int maxsum = 0;
            int maxnum = 0;
            int minsum = 0;
            int minnum = 0;

            //按照八邻域的方式进行大小区的划分与标记，同时计算各区的总和个数
            if (max1 <= max2) {
                for (int l = max1; l <= max2; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 1;
                    maxsum += gray.at<float>(i + di[l], j + dj[l]);
                    maxnum++;
                }
            } else {
                for (int l = max1; l <= 7; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 1;
                    maxsum += gray.at<float>(i + di[l], j + dj[l]);
                    maxnum++;
                }
                for (int l = 0; l <= max2; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 1;
                    maxsum += gray.at<float>(i + di[l], j + dj[l]);
                    maxnum++;
                }

            }
            if (min2 <= min1) {
                for (int l = min2; l <= min1; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 0;
                    minsum += gray.at<float>(i + di[l], j + dj[l]);
                    minnum++;
                }
            } else {
                for (int l = min2; l <= 7; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 0;
                    minsum += gray.at<float>(i + di[l], j + dj[l]);
                    minnum++;
                }
                for (int l = 0; l <= min1; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 0;
                    minsum += gray.at<float>(i + di[l], j + dj[l]);
                    minnum++;
                }
            }

            //处理排序后与八邻域处理之后的不匹配的值，按照八邻域处理方法的结果的处理有问题点的均值
            for (int k = 0; k < 8; k++) {
                if (obj.at<float>(i + di[k], j + dj[k]) == obj2.at<float>(i + di[k], j + dj[k])) {
                    //tu.at<float>( i + di[k], j + dj[k] ) = gray.at<float>( i + di[k], j + dj[k] );
                    continue;
                } else {
                    if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                        maxsum -= gray.at<float>(i + di[k], j + dj[k]);
                        maxnum--;
                    }
                    if (obj2.at<float>(i + di[k], j + dj[k]) == 0) {
                        minsum -= gray.at<float>(i + di[k], j + dj[k]);
                        minnum--;
                    }
                }
            }
            //进行均值修复
            for (int k = 0; k < 8; k++) {
                if (obj.at<float>(i + di[k], j + dj[k]) != obj2.at<float>(i + di[k], j + dj[k])) {
                    if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                        if (maxnum > 0) {
                            tu.at<float>(i + di[k], j + dj[k]) = maxsum / maxnum;
                        }
                    } else {
                        if (minnum > 0) {
                            tu.at<float>(i + di[k], j + dj[k]) = minsum / minnum;
                        }
                    }
                }
            }
        }
    }
    /*
	for (int i = 0; i <= rows - 1; ++i)
	{
	obj.at<float>(i, 0) =tu.at<float>(i, 0);
	obj.at<float>(i, cols - 1) = tu.at<float>(i, cols - 1);
	}
	for (int j = 0; j <= cols - 1; ++j)
	{
	obj.at<float>(0, j) =tu.at<float>(0, j);
	obj.at<float>(rows - 1, j) = tu.at<float>(rows - 1, j);
	}
	*/
    return tu;
}

//----------------------[old_testsortpixel()函数]---------------------
//输出排序法修复展示结果
//--------------------------------------------------------------
void old_testsortpixel(const Mat &gray, int th, int x, int y) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放排序法进行标记的大小区
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放八邻域处理之后的进行的大小区的标记
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //存放处理之后的图像灰度值
    int i = x;
    int j = y;
    int negative_max, positive_max;
    negative_max = positive_max = OUTPUT_FALSE;
    int max_p, min_p;
    //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
    //找最大最小区域
    for (int k = 1; k <= 8; ++k) {
        int diff = gray.at<float>(i + di[k], j + dj[k]) -
                   gray.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //大区域的最后一个点是负最小前面的点
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //大区域的第一个点是正最大后面的点
            }
        }
    }
    int dis_pow = fabs(positive_max) < fabs(negative_max)
                  ? fabs(positive_max) : fabs(negative_max);
    if (dis_pow < th) {//跳过低于阈值的点

        return;
    }

    cout << "原始数值" << endl;
    //组成数组进行排序
    for (int k = 0; k < 8; k++) {
        round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
        cout << round[k].val << " ";
    }
    cout << endl;
    sort(round, round + 8, Cmpare);

    cout << "排序后的坐标" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].pos << " ";
    }
    cout << endl;
    cout << "排序后的数值" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].val << " ";
    }
    cout << endl;


    //return round;
    int maxcha = 0, premax, latemax;
    //排序后找到最大差值的位置
    for (int k = 0; k < 7; k++) {
        if (round[k + 1].val - round[k].val >= maxcha) {
            maxcha = round[k + 1].val - round[k].val;
            premax = k;
            latemax = k + 1;
        }
    }

    //排序后在经由最大差值处理之后的分为大小区之后的点进行obj进行大小区标记
    for (int k = 0; k <= premax; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
    }
    for (int k = latemax; k <= 7; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
    }

    cout << "排序后的标记" << endl;
    for (int k = 0; k < 8; k++) {
        cout << obj.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;

    ////////////////////////////////////////////////////////////////////////
    int max1 = max_p;
    int max2 = min_p;
    int min1 = max_p - 1;
    int min2 = min_p + 1;
    int maxsum = 0;
    int maxnum = 0;
    int minsum = 0;
    int minnum = 0;

    //按照八邻域的方式进行大小区的划分与标记，同时计算各区的总和个数
    if (max1 <= max2) {
        for (int l = max1; l <= max2; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 1;
            maxsum += gray.at<float>(i + di[l], j + dj[l]);
            maxnum++;
        }
    } else {
        for (int l = max1; l <= 7; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 1;
            maxsum += gray.at<float>(i + di[l], j + dj[l]);
            maxnum++;
        }
        for (int l = 0; l <= max2; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 1;
            maxsum += gray.at<float>(i + di[l], j + dj[l]);
            maxnum++;
        }

    }
    if (min2 <= min1) {
        for (int l = min2; l <= min1; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 0;
            minsum += gray.at<float>(i + di[l], j + dj[l]);
            minnum++;
        }
    } else {
        for (int l = min2; l <= 7; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 0;
            minsum += gray.at<float>(i + di[l], j + dj[l]);
            minnum++;
        }
        for (int l = 0; l <= min1; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 0;
            minsum += gray.at<float>(i + di[l], j + dj[l]);
            minnum++;
        }
    }

    cout << "八邻域的标记结果" << endl;
    for (int k = 0; k < 8; k++) {
        cout << obj2.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;

    //处理排序后与八邻域处理之后的不匹配的值，按照八邻域处理方法的结果的处理有问题点的均值
    for (int k = 0; k < 8; k++) {
        if (obj.at<float>(i + di[k], j + dj[k]) == obj2.at<float>(i + di[k], j + dj[k])) {
            tu.at<float>(i + di[k], j + dj[k]) = gray.at<float>(i + di[k], j + dj[k]);
        } else {
            if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                maxsum -= gray.at<float>(i + di[k], j + dj[k]);
                maxnum--;
            }
            if (obj2.at<float>(i + di[k], j + dj[k]) == 0) {
                minsum -= gray.at<float>(i + di[k], j + dj[k]);
                minnum--;
            }
        }
    }
    //进行均值修复
    for (int k = 0; k < 8; k++) {
        if (obj.at<float>(i + di[k], j + dj[k]) != obj2.at<float>(i + di[k], j + dj[k])) {
            if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                if (maxnum > 0) {
                    tu.at<float>(i + di[k], j + dj[k]) = maxsum / maxnum;
                }
            } else {
                if (minnum > 0) {
                    tu.at<float>(i + di[k], j + dj[k]) = minsum / minnum;
                }
            }
        }
    }

    cout << "修改后的结果" << endl;
    for (int k = 0; k < 8; k++) {
        cout << tu.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;


    return;
}


//----------------------[newsortpixel()函数]---------------------
//处理效果不好，废弃，只使用一条边找到大小边相邻的轮廓,其中包含对中心点的修复centerFix()函数
//--------------------------------------------------------------
Mat newsortpixel(const Mat &gray, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat tu = gray;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放排序法进行标记的大小区
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放八邻域处理之后的进行的大小区的标记
//	Mat  tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //存放处理之后的图像灰度值
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //组成数组进行排序
            for (int k = 0; k < 8; k++) {
                round[k].val = int(tu.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //排序后找到最大差值的位置
            for (int k = 0; k < 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }

            //排序后在经由最大差值处理之后的分为大小区之后的点进行obj进行大小区标记
            for (int k = 0; k <= premax; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
            }
            for (int k = latemax; k <= 7; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
            }

            //中心点修复

            tu.at<float>(i, j) = centerFix(tu, maxcha, i, j);

            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
            //找最大最小区域
            for (int k = 1; k <= 8; ++k) {
                int diff = tu.at<float>(i + di[k], j + dj[k]) -
                           tu.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //大区域的最后一个点是负最小前面的点
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //大区域的第一个点是正最大后面的点
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max)
                          ? fabs(positive_max) : fabs(negative_max);

            //		int dis_pow = fabs(positive_max) < fabs(negative_max)
            //				? fabs(negative_max) : fabs(positive_max);

            if (dis_pow < th) {//跳过低于阈值的点
                /*
			 for (int k = 0; k < 8; k++)
			 {
			 tu.at<float>(i + di[k], j + dj[k]) = gray.at<float>(i + di[k], j + dj[k]);
			 }
			 */
                //tu.at<float>(i, j) = gray.at<float>(i, j);
                continue;
            }

            ////////////////////////////////////////////////////////////////////////
            int max1 = max_p % 8;
            int max2 = min_p;
            int min1 = max_p - 1;
            int min2 = (min_p + 1) % 8;
            int maxsum = 0;
            int maxnum = 0;
            int minsum = 0;
            int minnum = 0;

            //按照八邻域的方式进行大小区的划分与标记，同时计算各区的总和个数
            if (max1 <= max2) {
                for (int l = max1; l <= max2; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 1;
                    maxsum += tu.at<float>(i + di[l], j + dj[l]);
                    maxnum++;
                }
            } else {
                for (int l = max1; l <= 7; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 1;
                    maxsum += tu.at<float>(i + di[l], j + dj[l]);
                    maxnum++;
                }
                for (int l = 0; l <= max2; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 1;
                    maxsum += tu.at<float>(i + di[l], j + dj[l]);
                    maxnum++;
                }

            }
            if (min2 <= min1) {
                for (int l = min2; l <= min1; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 0;
                    minsum += tu.at<float>(i + di[l], j + dj[l]);
                    minnum++;
                }
            } else {
                for (int l = min2; l <= 7; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 0;
                    minsum += tu.at<float>(i + di[l], j + dj[l]);
                    minnum++;
                }
                for (int l = 0; l <= min1; ++l) {
                    obj2.at<float>(i + di[l], j + dj[l]) = 0;
                    minsum += tu.at<float>(i + di[l], j + dj[l]);
                    minnum++;
                }
            }

            //处理排序后与八邻域处理之后的不匹配的值，按照八邻域处理方法的结果的处理有问题点的均值
            for (int k = 0; k < 8; k++) {
                if (obj.at<float>(i + di[k], j + dj[k]) == obj2.at<float>(i + di[k], j + dj[k])) {
                    //tu.at<float>( i + di[k], j + dj[k] ) = gray.at<float>( i + di[k], j + dj[k] );
                    continue;
                } else {
                    if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                        maxsum -= tu.at<float>(i + di[k], j + dj[k]);
                        maxnum--;
                    }
                    if (obj2.at<float>(i + di[k], j + dj[k]) == 0) {
                        minsum -= tu.at<float>(i + di[k], j + dj[k]);
                        minnum--;
                    }
                }
            }
            //进行均值修复
            for (int k = 0; k < 8; k++) {
                if (obj.at<float>(i + di[k], j + dj[k]) != obj2.at<float>(i + di[k], j + dj[k])) {
                    if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                        if (maxnum > 0) {
                            tu.at<float>(i + di[k], j + dj[k]) = maxsum / maxnum;
                        }
                    } else {
                        if (minnum > 0) {
                            tu.at<float>(i + di[k], j + dj[k]) = minsum / minnum;
                        }
                    }
                }
            }
        }
    }
    /*
	for (int i = 0; i <= rows - 1; ++i)
	{
	obj.at<float>(i, 0) =tu.at<float>(i, 0);
	obj.at<float>(i, cols - 1) = tu.at<float>(i, cols - 1);
	}
	for (int j = 0; j <= cols - 1; ++j)
	{
	obj.at<float>(0, j) =tu.at<float>(0, j);
	obj.at<float>(rows - 1, j) = tu.at<float>(rows - 1, j);
	}
	*/
    return tu;
}


//----------------------[newtestsortpixel()函数]---------------------
//处理效果不好，废弃，，输出排序法修复展示结果
//--------------------------------------------------------------
void newtestsortpixel(const Mat &gray, int th, int x, int y) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    Mat tu = gray;
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放排序法进行标记的大小区
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放八邻域处理之后的进行的大小区的标记
    //Mat  tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //存放处理之后的图像灰度值
    int i = x;
    int j = y;
    int negative_max, positive_max;
    negative_max = positive_max = OUTPUT_FALSE;
    int max_p, min_p;
    //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
    //找最大最小区域
    for (int k = 1; k <= 8; ++k) {
        int diff = tu.at<float>(i + di[k], j + dj[k]) -
                   tu.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //大区域的最后一个点是负最小前面的点
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //大区域的第一个点是正最大后面的点
            }
        }
    }
    int dis_pow = fabs(positive_max) < fabs(negative_max)
                  ? fabs(positive_max) : fabs(negative_max);
    if (dis_pow < th) {//跳过低于阈值的点

        return;
    }

    cout << "原始数值" << endl;
    //组成数组进行排序
    for (int k = 0; k < 8; k++) {
        round[k].val = int(tu.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
        cout << round[k].val << " ";
    }
    cout << endl;
    sort(round, round + 8, Cmpare);

    cout << "排序后的坐标" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].pos << " ";
    }
    cout << endl;
    cout << "排序后的数值" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].val << " ";
    }
    cout << endl;


    //return round;
    int maxcha = 0, premax, latemax;
    //排序后找到最大差值的位置
    for (int k = 0; k < 7; k++) {
        if (round[k + 1].val - round[k].val >= maxcha) {
            maxcha = round[k + 1].val - round[k].val;
            premax = k;
            latemax = k + 1;
        }
    }

    //排序后在经由最大差值处理之后的分为大小区之后的点进行obj进行大小区标记
    for (int k = 0; k <= premax; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
    }
    for (int k = latemax; k <= 7; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
    }

    cout << "排序后的标记" << endl;
    for (int k = 0; k < 8; k++) {
        cout << obj.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;

    ////////////////////////////////////////////////////////////////////////
    int max1 = max_p;
    int max2 = min_p;
    int min1 = max_p - 1;
    int min2 = min_p + 1;
    int maxsum = 0;
    int maxnum = 0;
    int minsum = 0;
    int minnum = 0;

    //按照八邻域的方式进行大小区的划分与标记，同时计算各区的总和个数
    if (max1 <= max2) {
        for (int l = max1; l <= max2; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 1;
            maxsum += tu.at<float>(i + di[l], j + dj[l]);
            maxnum++;
        }
    } else {
        for (int l = max1; l <= 7; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 1;
            maxsum += tu.at<float>(i + di[l], j + dj[l]);
            maxnum++;
        }
        for (int l = 0; l <= max2; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 1;
            maxsum += tu.at<float>(i + di[l], j + dj[l]);
            maxnum++;
        }

    }
    if (min2 <= min1) {
        for (int l = min2; l <= min1; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 0;
            minsum += tu.at<float>(i + di[l], j + dj[l]);
            minnum++;
        }
    } else {
        for (int l = min2; l <= 7; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 0;
            minsum += tu.at<float>(i + di[l], j + dj[l]);
            minnum++;
        }
        for (int l = 0; l <= min1; ++l) {
            obj2.at<float>(i + di[l], j + dj[l]) = 0;
            minsum += tu.at<float>(i + di[l], j + dj[l]);
            minnum++;
        }
    }

    cout << "八邻域的标记结果" << endl;
    for (int k = 0; k < 8; k++) {
        cout << obj2.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;

    //处理排序后与八邻域处理之后的不匹配的值，按照八邻域处理方法的结果的处理有问题点的均值
    for (int k = 0; k < 8; k++) {
        if (obj.at<float>(i + di[k], j + dj[k]) == obj2.at<float>(i + di[k], j + dj[k])) {
            tu.at<float>(i + di[k], j + dj[k]) = tu.at<float>(i + di[k], j + dj[k]);
        } else {
            if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                maxsum -= tu.at<float>(i + di[k], j + dj[k]);
                maxnum--;
            }
            if (obj2.at<float>(i + di[k], j + dj[k]) == 0) {
                minsum -= tu.at<float>(i + di[k], j + dj[k]);
                minnum--;
            }
        }
    }
    //进行均值修复
    for (int k = 0; k < 8; k++) {
        if (obj.at<float>(i + di[k], j + dj[k]) != obj2.at<float>(i + di[k], j + dj[k])) {
            if (obj2.at<float>(i + di[k], j + dj[k]) == 1) {
                if (maxnum > 0) {
                    tu.at<float>(i + di[k], j + dj[k]) = maxsum / maxnum;
                }
            } else {
                if (minnum > 0) {
                    tu.at<float>(i + di[k], j + dj[k]) = minsum / minnum;
                }
            }
        }
    }

    cout << "修改后的结果" << endl;
    for (int k = 0; k < 8; k++) {
        cout << tu.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;


    return;
}


//----------------------[mfixDiff()函数]--------------------
//我的fixDiff函数
//
//--------------------------------------------------------
Mat mfixDiff(const Mat &src, Mat &votemat, Mat &bigm, Mat &smallm, Mat &diffm, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//这里的OBJ存储修复后的灰度图
    //Mat diffm(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    //Mat bifm(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    //Mat small(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    diffm = obj.clone();
    bigm = obj.clone();
    smallm = obj.clone();
    //output_arr_csv(diffm, "C://Users//mrmjy//Desktop//diffm.csv");
    int min_tag, max_tag;
    min_tag = max_tag = 0;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int sum = 0;
            int num = 0;
            int min = 255;
            int locate = -1;

            if (votemat.at<float>(i, j) == 0) {//投票值为0,不判断，直接赋值
                obj.at<float>(i, j) = src.at<float>(i, j);
                continue;
            } else if (int(votemat.at<float>(i, j)) % 10 == 0) {
                bigm.at<float>(i, j) = 1;
                obj.at<float>(i, j) = src.at<float>(i, j);
                continue;
            } else if (int(votemat.at<float>(i, j)) / 10 == 0) {
                smallm.at<float>(i, j) = 1;
                obj.at<float>(i, j) = src.at<float>(i, j);
                continue;
            } else {
                diffm.at<float>(i, j) = 1;
                if (int(votemat.at<float>(i, j)) % 10 < int(votemat.at<float>(i, j)) / 10) {
                    for (int s = 0; s < 8; s++) {
                        if (int(votemat.at<float>(i + di[s], j + dj[s])) % 10 == 0 &&
                            int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                            sum += src.at<float>(i + di[s], j + dj[s]);
                            num++;
                        }
                    }
                    if (num != 0) {
                        obj.at<float>(i, j) = sum / num;
                        //votemat.at<float>(i, j) = 10;
                        //bigm.at<float>(i, j) = 1;
                        continue;
                    } else {
                        obj.at<float>(i, j) = src.at<float>(i, j);
                        //diffm.at<float>(i, j) = 1;
                        continue;
                    }
                } else if (int(votemat.at<float>(i, j)) % 10 > int(votemat.at<float>(i, j)) / 10) {
                    for (int s = 0; s < 8; s++) {
                        if (int(votemat.at<float>(i + di[s], j + dj[s])) / 10 == 0 &&
                            int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                            sum += src.at<float>(i + di[s], j + dj[s]);
                            num++;
                        }
                    }
                    if (num != 0) {
                        obj.at<float>(i, j) = sum / num;
                        //votemat.at<float>(i, j) = 1;
                        //smallm.at<float>(i, j) = 1;
                        continue;
                    } else {
                        obj.at<float>(i, j) = src.at<float>(i, j);
                        //diffm.at<float>(i, j) = 1;
                        continue;
                    }
                } else {
                    for (int s = 0; s < 8; s++) {
                        if (int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                            if (int(votemat.at<float>(i + di[s], j + dj[s])) / 10 == 0 ||
                                int(votemat.at<float>(i + di[s], j + dj[s])) % 10 == 0) {
                                if (min > abs(src.at<float>(i + di[s], j + dj[s]) - src.at<float>(i, j))) {
                                    min = abs(src.at<float>(i + di[s], j + dj[s]) - src.at<float>(i, j));
                                    locate = s;
                                }
                            }
                        }
                    }
                    if (locate == -1) {
                        obj.at<float>(i, j) = src.at<float>(i, j);
                        continue;
                    } else if (int(votemat.at<float>(i + di[locate], j + dj[locate])) % 10 == 0) {
                        for (int s = 0; s < 8; s++) {
                            if (int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                                if (int(votemat.at<float>(i + di[s], j + dj[s])) % 10 == 0 &&
                                    int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                                    sum += src.at<float>(i + di[s], j + dj[s]);
                                    num++;
                                }
                            }
                        }
                        if (num != 0) {
                            obj.at<float>(i, j) = sum / num;
                            //votemat.at<float>(i, j) = 10;
                            //bigm.at<float>(i, j) = 1;
                            continue;
                        } else {
                            obj.at<float>(i, j) = src.at<float>(i, j);
                            //diffm.at<float>(i, j) = 1;
                            continue;
                        }
                    } else if (int(votemat.at<float>(i + di[locate], j + dj[locate])) / 10 == 0) {
                        for (int s = 0; s < 8; s++) {
                            if (int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                                if (int(votemat.at<float>(i + di[s], j + dj[s])) / 10 == 0 &&
                                    int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                                    sum += src.at<float>(i + di[s], j + dj[s]);
                                    num++;
                                }
                            }
                        }
                        if (num != 0) {
                            obj.at<float>(i, j) = sum / num;
                            //votemat.at<float>(i, j) = 1;
                            //smallm.at<float>(i, j) = 1;
                            continue;
                        } else {
                            obj.at<float>(i, j) = src.at<float>(i, j);
                            //diffm.at<float>(i, j) = 1;
                            continue;
                        }
                    }

                }

            }

        }
    }

    return obj;
}


bool Cmpare1(const int &a, const int &b) {
    return a < b;
}

//----------------------[centerFix()函数]--------------------
//中心点去噪声
//
//--------------------------------------------------------
int centerFix(const Mat &gray, int maxcha, int x, int y) {
    int i = x;
    int j = y;
    int max = maxcha;
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int a[9];
    a[0] = gray.at<float>(i, j);
    for (int k = 1; k <= 8; k++) {
        a[k] = int(gray.at<float>(i + di[k], j + dj[k]));
    }
    sort(a, a + 9, Cmpare1);
    for (int k = 0; k <= 8; k++) {
        if (a[k] == gray.at<float>(i, j)) {
            if (k == 0) {
                /*
				if (a[k + 1] - a[k] > max)
				{
					return a[k + 1];
					break;
				}
				*/
                return a[k + 1];
                break;
            } else if (k == 8) {
                /*
					if (a[k ] - a[k-1] > max)
					{
						return a[k -1];
						break;
					}
					*/
                return a[k - 1];
                break;
            } else if (a[k] - a[k - 1] > max || a[k + 1] - a[k] > max) {
                if (a[k] - a[k - 1] > a[k + 1] - a[k]) {
                    return a[k + 1];
                    break;
                } else {
                    return a[k - 1];
                    break;
                }
            }
        }
    }
    return gray.at<float>(i, j);
}


//----------------------[mfindBigSmallArea()函数]--------------------
//划分大小区域12.18
//
//-----------------------------------------------------------------
Mat mfindBigSmallArea(const Mat &gray, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            /*
			if (judge_single(gray, 20, i, j) == 1)
			{
				continue;
			}
			*/
            /*
			if (judge_transition(gray, 20, i, j) == 1)
			{
				continue;
			}
			*/
            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
            //找最大最小区域
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //前面是大区域
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //后面是大区域
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
            if (dis_pow < th) {//跳过低于阈值的点
                continue;
            }

            //int min = 999;
            //int locate = -1;
            int max1 = max_p % 8;    //后面是大区域
            int max2 = min_p;    //前面是大区域
            int min1 = (min_p + 1) % 8;  //后面是小区域
            int min2 = max_p - 1;   //前面是小区域


            if (max1 <= max2) {
                for (int l = max1; l <= max2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
            } else {
                for (int l = max1; l <= 7; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
                for (int l = 0; l <= max2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }

            }

            if (min2 >= min1) {
                for (int l = min1; l <= min2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }
            } else {
                for (int l = min1; l <= 7; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }
                for (int l = 0; l <= min2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }

            }

            /*
			for (int s = 0; s < 8; s++)
			{
				if (min > abs(gray.at<float>(i + di[s], j + dj[s]) - gray.at<float>(i, j)))
				{
					min = abs(gray.at<float>(i + di[s], j + dj[s]) - gray.at<float>(i, j));
					locate = s;
				}
			}
			if (int(obj.at<float>(i + di[locate], j + dj[locate])) % 10 > int(obj.at<float>(i + di[locate], j + dj[locate])) / 10)
			{
				obj.at<float>(i + di[locate], j + dj[locate]) += 1;
			}
			else if (int(obj.at<float>(i + di[locate], j + dj[locate])) % 10 < int(obj.at<float>(i + di[locate], j + dj[locate])) / 10)
			{
				obj.at<float>(i +  di[locate], j + dj[locate]) += 10;
			}
			*/
            /*
			while (1)
			{
			if (max1 != max2)
			{
			obj.at<float>(i + di[max1], j + dj[max1]) += 10;
			max1 = max1 + 1;
			}
			if (max1 + 1 > 8)
			{
			break;
			}
			if (max1 == max2)
			{
			obj.at<float>(i + di[max1], j + dj[max1]) += 10;
			break;
			}
			}

			if (max1 != max2) {
			for (int k = 0; k < max2+1; k++)
			{
			obj.at<float>(i + di[k], j + dj[k]) += 10;
			}

			}


			while (1)
			{
			if (min1 != min2)
			{
			obj.at<float>(i + di[min1], j + dj[min1]) += 1;
			min1 = min1 - 1;
			}
			if (min1 - 1 < -1)
			{
			break;
			}
			if (min1 == min2)
			{
			obj.at<float>(i + di[min1], j + dj[min1]) += 1;
			break;
			}
			}

			if (min1 != min2) {
			for (int k = min2; k < 8; k++)
			{
			obj.at<float>(i + di[k], j + dj[k]) += 1;
			}

			}
			*/
        }
    }

    return obj;
}


//----------------------[mfindBigSmallArea()函数]--------------------
//划分大小区域12.18
//
//-----------------------------------------------------------------
void test_mfindBigSmallArea(const Mat &gray, int i,int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p=0, min_p=0;
            //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
            //找最大最小区域
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff <= 0) {
                    if (diff <= negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //前面是大区域
                    }
                } else {
                    if (diff >= positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //后面是大区域
                    }
                }
            }


            //int min = 999;
            //int locate = -1;
            int max1 = max_p % 8;    //后面是大区域
            int max2 = min_p;    //前面是大区域
            int min1 = (min_p + 1) % 8;  //后面是小区域
            int min2 = max_p - 1;   //前面是小区域

            cout<<"大区域: "<<endl;
            if (max1 <= max2) {
                for (int l = max1; l <= max2; ++l) {
                    cout<<gray.at<float>(i + di[l], j + dj[l])<<" ";
                }
            } else {
                for (int l = max1; l <= 7; ++l) {
                    cout<<gray.at<float>(i + di[l], j + dj[l])<<" ";
                }
                for (int l = 0; l <= max2; ++l) {
                    cout<<gray.at<float>(i + di[l], j + dj[l])<<" ";
                }

            }

            cout<<endl;

            cout<<"小区域: "<<endl;
            if (min2 >= min1) {
                for (int l = min1; l <= min2; ++l) {
                    cout<<gray.at<float>(i + di[l], j + dj[l])<<" ";
                }
            } else {
                for (int l = min1; l <= 7; ++l) {
                    cout<<gray.at<float>(i + di[l], j + dj[l])<<" ";
                }
                for (int l = 0; l <= min2; ++l) {
                    cout<<gray.at<float>(i + di[l], j + dj[l])<<" ";
                }
            }

            cout<<endl;
}

//----------------------[color2_vote()函数]---------------------
//使用两种颜色表现投票大小边
//
//
//--------------------------------------------------------------
Mat color2_vote(const Mat &vote) {
    int rows = vote.rows;
    int cols = vote.cols;
    Mat obj(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (vote.at<float>(i, j) == 222) {
                obj.at<Vec3b>(i, j)[2] = MAX_PT;
                //红色，大边
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
            if (vote.at<float>(i, j) == 1) {
                obj.at<Vec3b>(i, j)[1] = MAX_PT;
                //绿色，小边
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
            }
        }
    }

    return obj;

}


//----------------------[fix7vs1()函数]--------------------
////7比1情况下的修正
//
//-----------------------------------------------------------------
Mat fix7vs1(Mat &gray) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat tag(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));   //   标记其中变化的点
    obj = gray.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }
            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //前面是大区域
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //后面是大区域
                    }
                }
            }
            int max1 = max_p % 8;    //后面是大区域
            int max2 = min_p;    //前面是大区域
            int min1 = (min_p + 1) % 8;  //后面是小区域
            int min2 = max_p - 1;   //前面是小区域

            if (max1 == max2)//7比1情况下的修正
            {
                int sum = 0;
                for (int k = 0; k <= 7; ++k) {
                    sum += gray.at<float>(i + di[k], j + dj[k]);
                }
                sum -= gray.at<float>(i + di[max1], j + dj[max1]);
                tag.at<float>(i + di[max1], j + dj[max1]) = 1;
                //obj.at<float>(i, j) = sum / 7;
                obj.at<float>(i + di[max1], j + dj[max1]) = sum / 7;
                //if (i + di[max1] >= 1 && j + dj[max1] >= 1 && i + di[max1] <= rows - 1 && j + dj[max1] <= cols - 1)
                //		if (i >= 2 && j >= 2 && i < rows - 2 && j < cols - 2)
                //   	{
                //			obj.at<float>(i + di[max1], j + dj[max1]) = sum / 7;
                //		}
            } else if (min1 == min2)//7比1情况下的修正
            {
                int sum = 0;
                for (int k = 0; k <= 7; ++k) {
                    sum += gray.at<float>(i + di[k], j + dj[k]);
                }
                sum -= gray.at<float>(i + di[min1], j + dj[min1]);
                tag.at<float>(i + di[min1], j + dj[min1]) = 2;
                //obj.at<float>(i, j) = sum / 7;
                obj.at<float>(i + di[min1], j + dj[min1]) = sum / 7;
                //if (i + di[min1] >= 1 && j + dj[min1] >= 1 && i + di[min1] <= rows - 1&& j + dj[min1] <=cols-1)
                //	if (i >= 2 && j >= 2 && i < rows - 2 && j < cols - 2)
                //		{
                //			obj.at<float>(i + di[min1], j + dj[min1]) = sum / 7;
                //		}
            } else {
                //obj.at<float>(i, j) = gray.at<float>(i, j);
                continue;
            }

        }
    }
    /*
	for (int i = 0; i <= rows - 1; ++i)
	{
		obj.at<float>(i, 0) = gray.at<float>(i, 0);
		obj.at<float>(i, cols - 1) = gray.at<float>(i, cols - 1);
	}
	for (int j = 0; j <= cols - 1; ++j)
	{
		obj.at<float>(0, j) = gray.at<float>(0, j);
		obj.at<float>(rows - 1,j) = gray.at<float>(rows - 1,j);
	}
	*/
    //return obj;
    return tag;
}


//----------------------[testfix7vs1()函数]--------------------
////test7比1情况下的修正
//
//-----------------------------------------------------------------
int testfix7vs1(Mat &gray, int x, int y) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int i = x;
    int j = y;
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    obj = gray.clone();
    int negative_max, positive_max;
    negative_max = positive_max = OUTPUT_FALSE;
    int max_p, min_p;
    //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
    //找最大最小区域
    for (int k = 1; k <= 8; ++k) {
        int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //前面是大区域
                cout << gray.at<float>(i + di[k], j + dj[k]) << "-" << gray.at<float>(i + di[k - 1], j + dj[k - 1])
                     << "min:" << diff << endl;
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //后面是大区域
            }
        }
    }
    int max1 = max_p % 8;    //后面是大区域
    int max2 = min_p;    //前面是大区域
    int min1 = (min_p + 1) % 8;  //后面是小区域
    int min2 = max_p - 1;   //前面是小区域

    if (max1 <= max2) {
        cout << "da" << endl;
        for (int l = max1; l <= max2; ++l) {
            cout << obj.at<float>(i + di[l], j + dj[l]) << " ";
        }
    } else {
        cout << "da" << endl;
        for (int l = max1; l <= 7; ++l) {
            cout << obj.at<float>(i + di[l], j + dj[l]) << " ";
        }
        for (int l = 0; l <= max2; ++l) {
            cout << obj.at<float>(i + di[l], j + dj[l]) << " ";
        }

    }

    if (min2 >= min1) {
        cout << "xiao" << endl;
        for (int l = min1; l <= min2; ++l) {
            cout << obj.at<float>(i + di[l], j + dj[l]) << " ";
        }
    } else {
        cout << "xiao" << endl;
        for (int l = min1; l <= 7; ++l) {
            cout << obj.at<float>(i + di[l], j + dj[l]) << " ";
        }
        for (int l = 0; l <= min2; ++l) {
            cout << obj.at<float>(i + di[l], j + dj[l]) << " ";
        }

    }


    cout << endl;

    if (max1 == max2)//7比1情况下的修正
    {
        int sum = 0;
        for (int k = 0; k <= 7; ++k) {
            sum += gray.at<float>(i + di[k], j + dj[k]);
        }
        sum -= gray.at<float>(i + di[max1], j + dj[max1]);
        obj.at<float>(i, j) = sum / 7;
        //if (i + di[max1] >= 1 && j + dj[max1] >= 1 && i + di[max1] <= rows - 1 && j + dj[max1] <= cols - 1)
//		if (i >= 2 && j >= 2 && i < rows - 2 && j < cols - 2)
//		{
//			obj.at<float>(i + di[max1], j + dj[max1]) = sum / 7;
//		}
        cout << "max1:" << max1 << "    " << gray.at<float>(i + di[max1], j + dj[max1]) << endl;

    } else if (min1 == min2)//7比1情况下的修正
    {
        int sum = 0;
        for (int k = 0; k <= 7; ++k) {
            sum += gray.at<float>(i + di[k], j + dj[k]);
        }
        sum -= gray.at<float>(i + di[min1], j + dj[min1]);
        obj.at<float>(i, j) = sum / 7;
        //if (i + di[min1] >= 1 && j + dj[min1] >= 1 && i + di[min1] <= rows - 1&& j + dj[min1] <=cols-1)
//	   if (i >= 2 && j >= 2 && i < rows - 2 && j < cols - 2)
//  	{
//			obj.at<float>(i + di[min1], j + dj[min1]) = sum / 7;
//		}
        cout << "min:" << min1 << "    " << gray.at<float>(i + di[min1], j + dj[min1]) << endl;

    }



    /*
	else
	{
		obj.at<float>(i, j) = gray.at<float>(i, j);
	}

	cout << "min1:"<<min1 <<endl;
	cout << "min2:" <<min2 <<endl;
	cout << obj.at<float>(i ,j)<< " ";
	*/
    for (int k = 0; k < 8; k++) {
        cout << obj.at<float>(i + di[k], j + dj[k]) << " ";
    }
    return 0;
}

//----------------------[sort7vs1()函数]---------------------
//sort7vs1，结果测试不理想，在几何图形中，会依次修正边缘，导致整个图像被完全修改
//--------------------------------------------------------------
Mat sort7vs1(const Mat &gray) {
    Mat copy = gray;
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //组成数组进行排序

            for (int k = 0; k < 8; k++) {
                round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //排序后找到最大差值的位置
            for (int k = 0; k < 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }

            if (premax == 0) {
                copy.at<float>(i + di[round[0].pos], j + dj[round[0].pos]) = copy.at<float>(i + di[round[1].pos],
                                                                                            j + dj[round[1].pos]);
            }
            if (latemax == 7) {
                copy.at<float>(i + di[round[7].pos], j + dj[round[7].pos]) = copy.at<float>(i + di[round[6].pos],
                                                                                            j + dj[round[6].pos]);
            }

        }
    }
    return copy;
}


//----------------------[fixEdge函数]---------------------
//
//--------------------------------------------------------------
Mat fixEdge(const Mat &src) {
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (src.at<float>(i, j) == 222) {
                for (int k = 0; k <= 7; k++) {

                }
            }

        }
    }
    return obj;
}


//----------------------[find_edgeboth()函数]---------------------
//在一张图中显示大小边
//--------------------------------------------------------------
Mat find_edgeboth(Mat &votemat) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = votemat.rows;
    int cols = votemat.cols;
    Mat objda(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat objxiao(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat tu(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    Mat both(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    //Mat newobj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            for (int s = 1; s < 8; s++) {
                if (votemat.at<float>(i + di[s], j + dj[s]) == 222 && votemat.at<float>(i, j) == 222) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 1 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 1) {
                        objda.at<float>(i + di[s], j + dj[s]) += 1;
                    }
                }
                if (votemat.at<float>(i + di[s], j + dj[s]) == 1 && votemat.at<float>(i, j) == 1) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 222 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 222) {
                        objxiao.at<float>(i + di[s], j + dj[s]) -= 1;
                    }
                }
            }

        }
    }

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (objxiao.at<float>(i, j) <= -2) {
                tu.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[1] = MAX_PT;
            }
            if (objda.at<float>(i, j) >= 2) {
                tu.at<Vec3b>(i, j)[2] = MAX_PT;
                tu.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
        }
    }

    return tu;
}

//----------------------[find_edgeboth2()函数]---------------------
//在一张图中显示大小边
//--------------------------------------------------------------
Mat find_edgeboth2(Mat &votemat) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = votemat.rows;
    int cols = votemat.cols;
    Mat objda(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat objxiao(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat objda1(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat objxiao1(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat tu(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    Mat both(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    //Mat newobj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            for (int s = 1; s < 8; s++) {
                if (votemat.at<float>(i + di[s], j + dj[s]) == 222 && votemat.at<float>(i, j) == 222) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 1 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 1) {
                        objda1.at<float>(i + di[s], j + dj[s]) += 1;
                    }
                }
                if (votemat.at<float>(i + di[s], j + dj[s]) == 1 && votemat.at<float>(i, j) == 1) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 222 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 222) {
                        objxiao1.at<float>(i + di[s], j + dj[s]) -= 1;
                    }
                }
            }
        }
    }

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (objxiao1.at<float>(i, j) <= -2) {
                both.at<float>(i, j) = -1;
            }
            if (objda1.at<float>(i, j) >= 2) {
                both.at<float>(i, j) = 1;
            }
        }
    }

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            for (int s = 1; s < 8; s++) {
                if (both.at<float>(i + di[s], j + dj[s]) == 1 && both.at<float>(i, j) == 1) {
                    if (both.at<float>(i + di[s + 1], j + dj[s + 1]) == -1 ||
                        both.at<float>(i + di[s - 1], j + dj[s - 1]) == -1) {
                        objda.at<float>(i + di[s], j + dj[s]) += 1;
                    }
                }
                if (both.at<float>(i + di[s], j + dj[s]) == -1 && both.at<float>(i, j) == -1) {
                    if (both.at<float>(i + di[s + 1], j + dj[s + 1]) == 1 ||
                        both.at<float>(i + di[s - 1], j + dj[s - 1]) == 1) {
                        objxiao.at<float>(i + di[s], j + dj[s]) -= 1;
                    }
                }
            }
        }
    }
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (objxiao.at<float>(i, j) <= -1) {
                tu.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[1] = MAX_PT;
            }
            if (objda.at<float>(i, j) >= 1) {
                tu.at<Vec3b>(i, j)[2] = MAX_PT;
                tu.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                tu.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
        }
    }
    return tu;
}


/*
//----------------------[drawline()函数]---------------------
//画出直线线条
//--------------------------------------------------------------
Mat drawline(Mat & votemat)
{

	int rows = votemat.rows;
	int cols = votemat.cols;

	Mat tu(rows, cols, CV_8UC3, Scalar(255, 255, 255));

	int d1i[3] = { -1,-1,0 };
	int d1j[3] = { 0,-1,-1 };

	int d2i[3] = { 0,1,1 };
	int d2j[3] = { -1,-1,0 };

	int d3i[3] = { 1,1,0 };
	int d3j[3] = { 0,1,1 };

	int d4i[3] = { 0,-1,-1 };
	int d4j[3] = { 1,1,0 };

	int di[4] = { -1,0,1,0 };
	int dj[4] = { 0,-1,0,1 };

	for (int i = 1; i < rows - 1; ++i)
	{
		for (int j = 1; j < cols - 1; ++j)
		{
		}
	}
}
*/


//----------------------[tagdaxiao()函数]---------------------
//标记大小点
//--------------------------------------------------------------
Mat tagdaxiao(Mat &votemat) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = votemat.rows;
    int cols = votemat.cols;
    Mat objda(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat objxiao(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            for (int s = 1; s < 8; s++) {
                if (votemat.at<float>(i + di[s], j + dj[s]) == 222 && votemat.at<float>(i, j) == 222) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 1 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 1) {
                        objda.at<float>(i + di[s], j + dj[s]) += 1;
                    }
                }
                if (votemat.at<float>(i + di[s], j + dj[s]) == 1 && votemat.at<float>(i, j) == 1) {
                    if (votemat.at<float>(i + di[s + 1], j + dj[s + 1]) == 222 ||
                        votemat.at<float>(i + di[s - 1], j + dj[s - 1]) == 222) {
                        objxiao.at<float>(i + di[s], j + dj[s]) -= 1;
                    }
                }
            }

        }
    }

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (objxiao.at<float>(i, j) <= -2) {
                tu.at<float>(i, j) = -1;
            }
            if (objda.at<float>(i, j) >= 2) {
                tu.at<float>(i, j) = 1;
            }
        }
    }

    return tu;
}


//----------------------[enhance()函数]---------------------
//增强
//--------------------------------------------------------------
Mat enhance(Mat &gray, const Mat &votemat, const Mat &yuan) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = votemat.rows;
    int cols = votemat.cols;
    int da;
    int sumda = 0;
    int numda = 0;
    int xiao;
    int sumxiao = 0;
    int numxiao = 0;
    //Mat tu = yuan;
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (votemat.at<float>(i, j) == 1) {
                sumda += gray.at<float>(i, j);
                numda++;
            }
            if (votemat.at<float>(i, j) == -1) {
                sumxiao += gray.at<float>(i, j);
                numxiao++;
            }
        }
    }
    da = sumda / numda;
    xiao = sumxiao / numxiao;
    int mean = (da + xiao) / 2;

    /*
	if (da < xiao)
	{
		int temp = xiao;
		xiao = da;
		da = temp;
	}
	*/
    cout << "大" << da << endl;
    cout << "小" << xiao << endl;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            /*
			//if (gray.at<float>(i, j) > da)
			if (yuan.at<float>(i, j) > da)
			{
		//	if (gray.at<float>(i, j) >= da && gray.at<float>(i, j) <= 200 )
		//	{
				//tu.at<float>(i, j) += 5;
				tu.at<float>(i, j) = yuan.at<float>(i, j)+20;
				//tu.at<float>(i, j) = 255;
			}
			*/
            if (yuan.at<float>(i, j) > mean)
                //else if(yuan.at<float>(i, j) < xiao )
            {
                tu.at<float>(i, j) = 255;
            } else {
                //tu.at<float>(i, j) = yuan.at<float>(i, j);
                tu.at<float>(i, j) = 0;
            }


        }
    }
    return tu;
}


//----------------------[Mapping()函数]---------------------
//Mapping通过映射使边缘处锐化，使内部区域平滑
//--------------------------------------------------------------
Mat Mapping(Mat &gray, Mat &tag, Mat &yuan) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat mpic(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    //mpic = gray;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (tag.at<float>(i, j) == 1) {
                //mpic.at<float>(i, j) = 255;
                mpic.at<float>(i, j) = gray.at<float>(i, j) + 40;
            } else if (tag.at<float>(i, j) == -1) {
                //mpic.at<float>(i, j) = 0;
                mpic.at<float>(i, j) = gray.at<float>(i, j) - 40;
            } else {
                mpic.at<float>(i, j) = gray.at<float>(i, j);
            }
        }
    }
    return mpic;
}

//----------------------[Histogram()函数]---------------------
//Histogram做并显示直方图
//--------------------------------------------------------------
void Histogram(Mat &gray) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    int a[256] = {0};
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            a[int(gray.at<float>(i, j))]++;
        }
    }
    for (int i = 0; i < 256; i++) {
        if (a[i] != 0) {
            cout << i << ":" << a[i] << endl;
        }
    }
}


//----------------------[Histogramdaxiao()函数]---------------------
//Histogramdaxiao做并显示直方图
//--------------------------------------------------------------
void Histogramdaxiao(Mat &gray, Mat &vote) {
    int di[8 + 1] = {+1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 1] = {+1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = vote.rows;
    int cols = vote.cols;
    int a[256] = {0};
    int b[256] = {0};
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (vote.at<float>(i, j) == 1) {
                a[int(gray.at<float>(i, j))]++;
            }
            if (vote.at<float>(i, j) == -1) {
                b[int(gray.at<float>(i, j))]++;
            }

        }
    }

    cout << "da" << endl;
    for (int i = 0; i < 256; i++) {
        if (a[i] != 0) {
            cout << i << ":" << a[i] << endl;
        }
    }

    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;

    cout << "xiao" << endl;
    for (int i = 0; i < 256; i++) {
        if (b[i] != 0) {
            cout << i << ":" << b[i] << endl;
        }
    }
}


//----------------------[Histogram1()函数]---------------------
//Histogram1做并显示直方图
//--------------------------------------------------------------
int Histogram1() {
    Mat src, gray, hist;                //hist为存储直方图的矩阵
    src = imread("C://Users//mrmjy//Desktop//4.jpg");
    cvtColor(src, gray, CV_BGR2GRAY);   //转换为灰度图

    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    int channels[] = {0};
    bool uniform = true;
    bool accumulate = false;

    /*计算直方图*/
    calcHist(&gray, 1, channels, Mat(), hist, 1, &histSize,
             &histRange, uniform, accumulate);

    /*创建描绘直方图的“图像”，和原图大小一样*/
    int hist_w = src.cols;
    int hist_h = src.rows;
    int bin_w = cvRound((double) hist_w / histSize);

    Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

    /*直方图归一化范围[0，histImage.rows]*/
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /*画直线*/
    for (int i = 1; i < histSize; ++i) {
        //cvRound：类型转换。 这里hist为256*1的一维矩阵，存储的是图像中各个灰度级的归一化值
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow("figure_src", src);
    imshow("figure_hist", histImage);

    waitKey(0);
    return 0;
}


//----------------------[histm()函数]---------------------
//histm()进行直方图均衡化处理
//--------------------------------------------------------------
int histm() {
    Mat srcImage = imread("C://Users//mrmjy//Desktop//5.jpg");
    if (!srcImage.data) {
        printf("图片加载失败!\n");
        return -1;
    }

    //定义灰度图像
    Mat gray;
    cvtColor(srcImage, gray, COLOR_RGB2GRAY);

    namedWindow("原图");
    imshow("原图", gray);

    //开始直方图均化处理
    Mat out;
    equalizeHist(gray, out);
    src_to_bmp(out, "C://Users//mrmjy//Desktop//out.jpg");
    namedWindow("经过直方图均化后处理");
    imshow("经过直方图均化后处理", out);
    waitKey();

    return 0;
}


//----------------------[grow()函数]---------------------
//grow()进行直方图均衡化处理
//--------------------------------------------------------------
int grow(Mat &src, Mat &vote) {
    int di[8 + 2] = {+1, +1, +0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8 + 2] = {+0, +1, +1, +1, 0, -1, -1, -1, +0, +1};
    int rows = vote.rows;
    int cols = vote.cols;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            for (int s = 1; s <= 8; s++) {

                if (vote.at<float>(i + di[s], j + dj[s]) == 1) {
                    if (vote.at<float>(i + di[s - 1], j + dj[s - 1]) == -1) {
                        if (abs(src.at<float>(i, j) - src.at<float>(i + di[s], j + dj[s])) <
                            abs(src.at<float>(i, j) - src.at<float>(i + di[s - 1], j + dj[s - 1]))) {
                            vote.at<float>(i, j) = 1;
                        }
                    }
                    if (vote.at<float>(i + di[s + 1], j + dj[s + 1]) == -1) {
                        if (abs(src.at<float>(i, j) - src.at<float>(i + di[s], j + dj[s])) <
                            abs(src.at<float>(i, j) - src.at<float>(i + di[s + 1], j + dj[s + 1]))) {
                            vote.at<float>(i, j) = 1;
                        }
                    }
                }


            }
        }
    }
    return 0;
}

//----------------------[color_findBigSmallAre()函数]---------------------
//color_findBigSmallAre()对彩色图像划分大小区域
//--------------------------------------------------------------
int color_diff(CvScalar a, CvScalar b) {
    int num = static_cast<int>(abs(a.val[0] - b.val[0]) + abs(a.val[1] - b.val[1]) + abs(a.val[2] - b.val[2]));
    return num;
}

//----------------------[light()函数]---------------------
//light()对彩色图像划分大小区域
//--------------------------------------------------------------
float light(CvScalar a) {
    float num = static_cast<float>(a.val[0] * 0.1 + a.val[1] * 0.6 + a.val[2] * 0.3);
    return num;
}


//----------------------[color_findBigSmallAre()函数]---------------------
//color_findBigSmallAre()对彩色图像划分大小区域
//--------------------------------------------------------------
Mat color_findBigSmallArea(Mat &gray, IplImage *a, int th) {

    //CvScalar s;
    //s = cvGet2D(a, 0, 0);
    //printf("B=%d, G=%d, R=%d\n", int(s.val[0]), int(s.val[1]), int(s.val[2]));

    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;

            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }

            //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
            /*
			int k = 1;
			int diff1 = color_diff(cvGet2D(a, i + di[k], j + dj[k]), cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
			k = 2;
			int diff2 = color_diff(cvGet2D(a, i + di[k], j + dj[k]), cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
			if (diff1 > diff2)
			{
				positive_max = diff1;
				max_p = 1;
				negative_max = diff2;
				min_p = 2;
			}
			else
			{
				positive_max = diff2;
				max_p = 2;
				negative_max = diff1;
				min_p = 1;
			}
			for (int k = 3; k <= 8; ++k)
			{
				//int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
				int diff = color_diff(cvGet2D(a, i + di[k], j + dj[k]), cvGet2D(a, i + di[k-1], j + dj[k-1]));
					if (diff > positive_max)
					{
						negative_max = positive_max;
						min_p = max_p;
						positive_max = diff;
						max_p = k;
						//后面是大区域
					}
					if (diff > negative_max && diff <positive_max)
					{
						negative_max = diff;
						min_p = k;
					}
			}
		//	int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
			if (positive_max <  th)
			{//跳过低于阈值的点
				continue;
			}

			int max1, max2, min1, min2;
			if ( light( cvGet2D( a, i + di[max_p], j + dj[max_p]) ) >= light(cvGet2D( a, i + di[max_p-1], j + dj[max_p-1] ) ) )
			{
				max1 = max_p%8;
				min1 = max_p - 1;
			}
			else
			{
				min1 = max_p%8;
				max1 = max_p - 1;
			}
			if (light(cvGet2D(a, i + di[min_p], j + dj[min_p])) >= light(cvGet2D(a, i + di[min_p - 1], j + dj[min_p - 1])))
			{
				max2 = min_p%8;
				min2 = min_p - 1;
			}
			else
			{
				min2 = min_p%8;
				max2 = min_p - 1;
			}
			*/

            /*
			if (max_p >= min_p)
			{
				for (int l = min_p; l < max_p; l++)
				{
					obj.at<float>(i + di[l], j + dj[l]) += 1;
				}
				for (int l = max_p; l <= 7; ++l)
				{
					obj.at<float>(i + di[l], j + dj[l]) += 10;
				}
				for (int l = 0; l < min_p; ++l)
				{
					obj.at<float>(i + di[l], j + dj[l]) += 10;
				}
			}
			else
			{
				for (int l = max_p; l < min_p; l++)
				{
					obj.at<float>(i + di[l], j + dj[l]) += 1;
				}
				for (int l = min_p; l <= 7; ++l)
				{
					obj.at<float>(i + di[l], j + dj[l]) += 10;
				}
				for (int l = 0; l < max_p; ++l)
				{
					obj.at<float>(i + di[l], j + dj[l]) += 10;
				}
			}
			*/



            for (int k = 1; k <= 8; ++k) {
                //int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                int diff = light(cvGet2D(a, i + di[k], j + dj[k])) - light(cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //前面是大区域
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //后面是大区域
                    }
                }
            }

            int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
            if (dis_pow < th) {//跳过低于阈值的点
                continue;
            }
            int max1 = max_p % 8;    //后面是大区域
            int max2 = min_p;    //前面是大区域
            int min1 = (min_p + 1) % 8;  //后面是小区域
            int min2 = max_p - 1;   //前面是小区域

            if (max1 <= max2) {
                for (int l = max1; l <= max2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
            } else {
                for (int l = max1; l <= 7; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
                for (int l = 0; l <= max2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
            }

            if (min2 >= min1) {
                for (int l = min1; l <= min2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }
            } else {
                for (int l = min1; l <= 7; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }
                for (int l = 0; l <= min2; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }
            }


        }
    }
    return obj;
}


//----------------------[oldcolor_findBigSmallAre()函数]---------------------
//oldcolor_findBigSmallAre()对彩色图像划分大小区域
//--------------------------------------------------------------
Mat oldcolor_findBigSmallArea(Mat &gray, IplImage *a, int th) {

    //CvScalar s;
    //s = cvGet2D(a, 0, 0);
    //printf("B=%d, G=%d, R=%d\n", int(s.val[0]), int(s.val[1]), int(s.val[2]));

    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            //max_p是大区域的第一个点，min_p是大区域最后一个点的位置
            //找最大最小区域
            int k = 1;
            int diff1 = color_diff(cvGet2D(a, i + di[k], j + dj[k]), cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
            k = 2;
            int diff2 = color_diff(cvGet2D(a, i + di[k], j + dj[k]), cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
            if (diff1 > diff2) {
                positive_max = diff1;
                max_p = 1;
                negative_max = diff2;
                min_p = 0;
            } else {
                positive_max = diff2;
                max_p = 2;
                negative_max = diff1;
                min_p = 1;
            }
            for (int k = 3; k <= 8; ++k) {
                //int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                int diff = color_diff(cvGet2D(a, i + di[k], j + dj[k]), cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
                if (diff > positive_max) {
                    negative_max = positive_max;
                    min_p = max_p - 1;
                    positive_max = diff;
                    max_p = k;
                    //后面是大区域
                }
                if (diff > negative_max && diff < positive_max) {
                    negative_max = diff;
                    min_p = k - 1;
                }
            }

            //	int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
            if (positive_max < th) {//跳过低于阈值的点
                continue;
            }

            //	if (light(cvGet2D(a, i + di[k], j + dj[k])) > light(cvGet2D(a, i + di[k], j + dj[k])))
            //	{
            //	}

            if (max_p >= min_p) {
                for (int l = min_p; l < max_p; l++) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }
                for (int l = max_p; l <= 7; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
                for (int l = 0; l < min_p; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
            } else {
                for (int l = max_p; l < min_p; l++) {
                    obj.at<float>(i + di[l], j + dj[l]) += 1;
                }
                for (int l = min_p; l <= 7; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
                for (int l = 0; l < max_p; ++l) {
                    obj.at<float>(i + di[l], j + dj[l]) += 10;
                }
            }

            /*
			int max1, max2, min1, min2;
			max2 = max_p % 8;  	//后面是大区域
			max1 = (max_p -1+8) % 8;  	//前面是大区域
			min1 = (min_p -1+8) % 8;  //后面是小区域
			min2 = min_p  % 8;   //前面是小区域
			if (max1 <= max2)
			{
			for (int l = max1; l <= max2; ++l)
			{
			obj.at<float>(i + di[l], j + dj[l]) += 10;
			}
			}
			else
			{
			for (int l = max1; l <= 7; ++l)
			{
			obj.at<float>(i + di[l], j + dj[l]) += 10;
			}
			for (int l = 0; l <= max2; ++l)
			{
			obj.at<float>(i + di[l], j + dj[l]) += 10;
			}
			}

			if (min2 >= min1)
			{
			for (int l = min1; l <= min2; ++l)
			{
			obj.at<float>(i + di[l], j + dj[l]) += 1;
			}
			}
			else
			{
			for (int l = min1; l <= 7; ++l)
			{
			obj.at<float>(i + di[l], j + dj[l]) += 1;
			}
			for (int l = 0; l <= min2; ++l)
			{
			obj.at<float>(i + di[l], j + dj[l]) += 1;
			}
			}
			*/

        }
    }
    return obj;
}

//----------------------[mcolor_fixDiff()函数]--------------------
//我的mcolor_fixDiff函数
//--------------------------------------------------------
Mat mcolor_fixDiff(Mat gray, Mat &src, Mat &votemat, Mat &bigm, Mat &smallm, Mat &diffm, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = votemat.rows;
    int cols = votemat.cols;
    Mat color(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    color = src.clone();
    Mat colorcopy(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    colorcopy = src.clone();
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//这里的OBJ存储修复后的灰度图
    diffm = obj.clone();
    bigm = obj.clone();
    smallm = obj.clone();
    int min_tag, max_tag;
    min_tag = max_tag = 0;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (judge_single(gray, 20, i, j) == 1) {
                //color.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
                continue;
            }

            //obj.at<Vec3b>(i, j)[2]
            if (votemat.at<float>(i, j) == 0) {
                //投票值为0,不判断，直接赋值
                //obj.at<float>(i, j) = src.at<float>(i, j);
                color.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
                continue;
            } else if (int(votemat.at<float>(i, j)) % 10 == 0) {
                bigm.at<float>(i, j) = 1;
                //obj.at<float>(i, j) = src.at<float>(i, j);
                color.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
                continue;
            } else if (int(votemat.at<float>(i, j)) / 10 == 0) {
                smallm.at<float>(i, j) = 1;
                //obj.at<float>(i, j) = src.at<float>(i, j);
                color.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
                continue;
            } else {
                int num = 0;
                int sumb = 0;
                int sumg = 0;
                int sumr = 0;
                diffm.at<float>(i, j) = 1;
                if (int(votemat.at<float>(i, j)) % 10 < int(votemat.at<float>(i, j)) / 10) {
                    for (int s = 0; s < 8; s++) {
                        if (int(votemat.at<float>(i + di[s], j + dj[s])) % 10 == 0 &&
                            int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                            //sum += src.at<float>(i + di[s], j + dj[s]);
                            num++;
                            sumb += (int) src.at<Vec3b>(i + di[s], j + dj[s])[0];
                            sumg += (int) src.at<Vec3b>(i + di[s], j + dj[s])[1];
                            sumr += (int) src.at<Vec3b>(i + di[s], j + dj[s])[2];
                        }
                    }
                    if (num != 0) {
                        //obj.at<float>(i, j) = sum / num;
                        //votemat.at<float>(i, j) = 10;
                        //bigm.at<float>(i, j) = 1;
                        color.at<Vec3b>(i, j)[0] = (int) sumb / num;
                        color.at<Vec3b>(i, j)[1] = (int) sumg / num;
                        color.at<Vec3b>(i, j)[2] = (int) sumr / num;
                        continue;
                    } else {
                        color.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
                        //diffm.at<float>(i, j) = 1;
                        continue;
                    }
                } else if (int(votemat.at<float>(i, j)) % 10 > int(votemat.at<float>(i, j)) / 10) {
                    for (int s = 0; s < 8; s++) {
                        if (int(votemat.at<float>(i + di[s], j + dj[s])) / 10 == 0 &&
                            int(votemat.at<float>(i + di[s], j + dj[s])) > 0) {
                            //sum += src.at<float>(i + di[s], j + dj[s]);
                            num++;
                            sumb += (int) src.at<Vec3b>(i + di[s], j + dj[s])[0];
                            sumg += (int) src.at<Vec3b>(i + di[s], j + dj[s])[1];
                            sumr += (int) src.at<Vec3b>(i + di[s], j + dj[s])[2];
                        }
                    }
                    if (num != 0) {
                        //obj.at<float>(i, j) = sum / num;
                        //votemat.at<float>(i, j) = 1;
                        //smallm.at<float>(i, j) = 1;
                        color.at<Vec3b>(i, j)[0] = (int) sumb / num;
                        color.at<Vec3b>(i, j)[1] = (int) sumg / num;
                        color.at<Vec3b>(i, j)[2] = (int) sumr / num;
                        continue;
                    } else {
                        color.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
                        //diffm.at<float>(i, j) = 1;
                        continue;
                    }
                } else {
                    color.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
                    continue;
                }
            }


            /*
				else
				{
					for (int s = 0; s < 8; s++)
					{
						if (int(votemat.at<float>(i + di[s], j + dj[s])) > 0)
						{
							if (int(votemat.at<float>(i + di[s], j + dj[s])) / 10 == 0 || int(votemat.at<float>(i + di[s], j + dj[s])) % 10 == 0)
							{
								if (min > abs(src.at<float>(i + di[s], j + dj[s]) - src.at<float>(i, j)))
								{
									min = abs(src.at<float>(i + di[s], j + dj[s]) - src.at<float>(i, j));
									locate = s;
								}
							}
						}
					}
					if (locate == -1)
					{
						obj.at<float>(i, j) = src.at<float>(i, j);
						continue;
					}
					else if (int(votemat.at<float>(i + di[locate], j + dj[locate])) % 10 == 0)
					{
						for (int s = 0; s < 8; s++)
						{
							if (int(votemat.at<float>(i + di[s], j + dj[s])) > 0)
							{
								if (int(votemat.at<float>(i + di[s], j + dj[s])) % 10 == 0 && int(votemat.at<float>(i + di[s], j + dj[s])) > 0)
								{
									sum += src.at<float>(i + di[s], j + dj[s]);
									num++;
								}
							}
						}
						if (num != 0)
						{
							obj.at<float>(i, j) = sum / num;
							//votemat.at<float>(i, j) = 10;
							//bigm.at<float>(i, j) = 1;
							continue;
						}
						else
						{
							obj.at<float>(i, j) = src.at<float>(i, j);
							//diffm.at<float>(i, j) = 1;
							continue;
						}
					}
					else if (int(votemat.at<float>(i + di[locate], j + dj[locate])) / 10 == 0)
					{
						for (int s = 0; s < 8; s++)
						{
							if (int(votemat.at<float>(i + di[s], j + dj[s])) > 0)
							{
								if (int(votemat.at<float>(i + di[s], j + dj[s])) / 10 == 0 && int(votemat.at<float>(i + di[s], j + dj[s])) > 0)
								{
									sum += src.at<float>(i + di[s], j + dj[s]);
									num++;
								}
							}
						}
						if (num != 0)
						{
							obj.at<float>(i, j) = sum / num;
							//votemat.at<float>(i, j) = 1;
							//smallm.at<float>(i, j) = 1;
							continue;
						}
						else
						{
							obj.at<float>(i, j) = src.at<float>(i, j);
							//diffm.at<float>(i, j) = 1;
							continue;
						}
					}

				}

			}
			*/

        }
    }
    return color;
}


//----------------------[judge_single()函数]---------------------
//判断是否是单像素区域
//--------------------------------------------------------------
int judge_single(const Mat &gray, int th, int i, int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    //组成数组进行排序
    for (int k = 0; k < 8; k++) {
        round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
    }
    sort(round, round + 8, Cmpare);
    //return round;
    int maxcha = 0, premax, latemax;
    //排序后找到最大差值的位置
    for (int k = 0; k < 7; k++) {
        if (round[k + 1].val - round[k].val >= maxcha) {
            maxcha = round[k + 1].val - round[k].val;
            premax = k;
            latemax = k + 1;
        }
    }
    if (premax == 1) {
        if ((round[1].pos + 1) % 8 != round[0].pos) {
            return 1;
        }
        if ((round[1].pos - 1) < 0) {
            if (round[0].pos != 7) {
                return 1;
            }
        } else {
            if ((round[1].pos - 1) != round[0].pos) {
                return 1;
            }
        }
    }

    if (latemax == 6) {
        if ((round[6].pos + 1) % 8 != round[7].pos) {
            return 1;
        }
        if ((round[6].pos - 1) < 0) {
            if (round[7].pos != 7) {
                return 1;
            }
        } else {
            if ((round[6].pos - 1) != round[7].pos) {
                return 1;
            }
        }
    }
    return 0;
}


//----------------------[judge_transition()函数]---------------------
//判断是否是过渡像素区域
//--------------------------------------------------------------
int judge_transition(const Mat &gray, int th, int i, int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[9];
    //组成数组进行排序
    round[0].val = gray.at<float>(i, j);
    for (int k = 1; k <= 8; k++) {
        round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
    }
    sort(round, round + 8, Cmpare);
    int cha[9];
    for (int k = 1; k <= 8; k++) {
        cha[k] = round[k].val - round[k - 1].val;
    }
    int s[9];
    for (int k = 1; k <= 8; k++) {
        s[k] = cha[k];
    }
    sort(cha, cha + 9, Cmpare1);
    if (cha[8] == s[3] && cha[7] == s[6]) {
        return 1;
    }
    if (cha[8] == s[6] && cha[7] == s[3]) {
        return 1;
    }
    return 0;
}

//----------------------[color_fix7vs1()函数]--------------------
////彩图7比1情况下的修正
//
//-----------------------------------------------------------------
Mat color_fix7vs1(Mat &src, IplImage *a, Mat &gray) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    Mat tag(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));   //   标记其中变化的点
    obj = gray.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            /*
			if (judge_single(src, 20, i, j) == 1)
			{
				continue;
			}

			if (judge_transition(src, 20, i, j) == 1)
			{
				continue;
			}
			*/
            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            for (int k = 1; k <= 8; ++k) {
                int diff = light(cvGet2D(a, i + di[k], j + dj[k])) - light(cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //前面是大区域
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //后面是大区域
                    }
                }
            }
            int max1 = max_p % 8;    //后面是大区域
            int max2 = min_p;    //前面是大区域
            int min1 = (min_p + 1) % 8;  //后面是小区域
            int min2 = max_p - 1;   //前面是小区域

            if (max1 == max2)//7比1情况下的修正
            {
                int sumb = 0;
                int sumg = 0;
                int sumr = 0;
                for (int k = 0; k <= 7; ++k) {
                    sumb += gray.at<Vec3b>(i + di[k], j + dj[k])[0];
                    sumg += gray.at<Vec3b>(i + di[k], j + dj[k])[1];
                    sumr += gray.at<Vec3b>(i + di[k], j + dj[k])[2];
                }
                sumb -= gray.at<Vec3b>(i + di[max1], j + dj[max1])[0];
                sumg -= gray.at<Vec3b>(i + di[max1], j + dj[max1])[1];
                sumr -= gray.at<Vec3b>(i + di[max1], j + dj[max1])[2];
                tag.at<float>(i + di[max1], j + dj[max1]) = 1;
                obj.at<Vec3b>(i + di[max1], j + dj[max1])[0] = sumb / 7;
                obj.at<Vec3b>(i + di[max1], j + dj[max1])[1] = sumg / 7;
                obj.at<Vec3b>(i + di[max1], j + dj[max1])[2] = sumr / 7;
            } else if (min1 == min2)//7比1情况下的修正
            {
                int sumb = 0;
                int sumg = 0;
                int sumr = 0;
                for (int k = 0; k <= 7; ++k) {
                    sumb += gray.at<Vec3b>(i + di[k], j + dj[k])[0];
                    sumg += gray.at<Vec3b>(i + di[k], j + dj[k])[1];
                    sumr += gray.at<Vec3b>(i + di[k], j + dj[k])[2];

                }
                sumb -= gray.at<Vec3b>(i + di[min1], j + dj[min1])[0];
                sumg -= gray.at<Vec3b>(i + di[min1], j + dj[min1])[1];
                sumr -= gray.at<Vec3b>(i + di[min1], j + dj[min1])[2];
                tag.at<float>(i + di[min1], j + dj[min1]) = 1;
                obj.at<Vec3b>(i + di[min1], j + dj[min1])[0] = sumb / 7;
                obj.at<Vec3b>(i + di[min1], j + dj[min1])[1] = sumg / 7;
                obj.at<Vec3b>(i + di[min1], j + dj[min1])[2] = sumr / 7;
            } else {
                //obj.at<Vec3b>(i, j) = gray.at<Vec3b>(i, j);
                continue;
            }

        }
    }

    for (int i = 0; i <= rows - 1; ++i) {
        obj.at<Vec3b>(i, 0) = gray.at<Vec3b>(i, 0);
        obj.at<Vec3b>(i, cols - 1) = gray.at<Vec3b>(i, cols - 1);
    }
    for (int j = 0; j <= cols - 1; ++j) {
        obj.at<Vec3b>(0, j) = gray.at<Vec3b>(0, j);
        obj.at<Vec3b>(rows - 1, j) = gray.at<Vec3b>(rows - 1, j);
    }

    return obj;
    //return tag;
}


//----------------------[color_sortpixel()函数]--------------------
////彩图排算法情况下的修正
//
//-----------------------------------------------------------------
Mat color_sortpixel(const Mat &gray, Mat &color, IplImage *a, int th) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//存放排序法进行标记的大小区
    Mat tu(rows, cols, CV_8UC3, Scalar(OUTPUT_FALSE)); //存放处理之后的彩色图像
    Mat maodun(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//记录矛盾点（左右的标记点与自身的不一致）
    tu = color.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //单像素区域跳过
            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }

            //判断是不是内部点，如果是，则进行跳过
            int negative_max, positive_max;
            negative_max = OUTPUT_FALSE;
            positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            for (int k = 1; k <= 8; ++k) {
                int diff = light(cvGet2D(a, i + di[k], j + dj[k])) - light(cvGet2D(a, i + di[k - 1], j + dj[k - 1]));
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max)
                          ? fabs(positive_max) : fabs(negative_max);

            if (dis_pow < th) {
                continue;
            }

            //组成数组进行排序
            for (int k = 0; k <= 7; k++) {
                round[k].val = light(cvGet2D(a, i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //排序后找到最大差值的位置
            for (int k = 1; k <= 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }

            /*
			//加入中心点之后进行判断是否是单像素点或者是过渡带
			int c[9];
			c[0] = gray.at<float>(i, j);
			for (int k = 1; k <= 8; k++)
			{
			c[k] = int(gray.at<float>(i + di[k], j + dj[k]));
			}
			sort(c, c + 9, Cmpare1);
			//判断是否是单像素点
			if (premax == 2 || premax == 5)
			{

			}
			//判断是否为过渡带
			*/

            int sumda = 0;
            int sumxiao = 0;

            int sumdab = 0;
            int sumdag = 0;
            int sumdar = 0;

            int sumxiaob = 0;
            int sumxiaog = 0;
            int sumxiaor = 0;

            //排序后在经由最大差值处理之后的分为大小区之后的点进行obj进行大小区标记
            for (int k = 0; k <= premax; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
                sumxiaob += color.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[0];
                sumxiaog += color.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[1];
                sumxiaor += color.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[2];


            }
            for (int k = latemax; k <= 7; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
                sumdab += color.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[0];
                sumdag += color.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[1];
                sumdar += color.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[2];
            }

            if (premax == 0) {
                int k = 0;
                //tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumda / 7;
                tu.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[0] = sumdab / 7;
                tu.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[1] = sumdag / 7;
                tu.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[2] = sumdar / 7;

                continue;
            }
            if (latemax == 7) {
                int k = 7;
                //tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumxiao / 7;
                tu.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[0] = sumxiaob / 7;
                tu.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[1] = sumxiaog / 7;
                tu.at<Vec3b>(i + di[round[k].pos], j + dj[round[k].pos])[2] = sumxiaor / 7;
                continue;
            }

            //找到被两个大边夹到中心的小边点
            for (int k = 0; k <= premax; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 1 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 1) {
                    //这里是两个矛盾点相邻时的状况
                    if (obj.at<float>(
                            obj.at<float>(i + di[(round[k].pos - 2 + 8) % 8], j + dj[(round[k].pos - 2 + 8) % 8])) ==
                        0 || obj.at<float>(i + di[(round[k].pos + 2) % 8], j + dj[(round[k].pos + 2) % 8]) == 0) {
                        continue;
                    } else {
                        maodun.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
                        /*
						sumxiao -= round[k].val;
						if(latemax!=7)
						tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumda / (8 - latemax);
						*/
                    }
                }
            }
            //与被两个小边点夹到中心的大边点
            for (int k = latemax; k <= 7; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 0 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 0) {
                    //这里是两个矛盾点相邻时的状况
                    if (obj.at<float>(
                            obj.at<float>(i + di[(round[k].pos - 2 + 8) % 8], j + dj[(round[k].pos - 2 + 8) % 8])) ==
                        1 || obj.at<float>(i + di[(round[k].pos + 2) % 8], j + dj[(round[k].pos + 2) % 8]) == 1) {
                        continue;
                    } else {
                        maodun.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 2;
                        /*
						sumda -= round[k].val;
						if (premax != 0)
						tu.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = sumxiao / latemax;
						*/
                    }
                }
            }

            //把周围八邻域之中的矛盾点全部减去之后再求均值
            int reducexiaob = 0;
            int reducexiaog = 0;
            int reducexiaor = 0;

            int reducedab = 0;
            int reducedag = 0;
            int reducedar = 0;

            int numxiao = 0;
            int numda = 0;
            for (int k = 0; k <= 7; k++) {
                if (maodun.at<float>(i + di[k], j + dj[k]) == 1);
                {
                    reducexiaob += color.at<Vec3b>(i + di[k], j + dj[k])[0];
                    reducexiaog += color.at<Vec3b>(i + di[k], j + dj[k])[1];
                    reducexiaor += color.at<Vec3b>(i + di[k], j + dj[k])[2];
                    numxiao++;
                }
                if (maodun.at<float>(i + di[k], j + dj[k]) == 2);
                {
                    reducedab += color.at<Vec3b>(i + di[k], j + dj[k])[0];
                    reducedag += color.at<Vec3b>(i + di[k], j + dj[k])[1];
                    reducedar += color.at<Vec3b>(i + di[k], j + dj[k])[2];
                    numda++;
                }
            }

            for (int k = 0; k <= 7; k++) {
                if (maodun.at<float>(i + di[k], j + dj[k]) == 1 && (latemax - numxiao) != 0) {
                    //obj.at<float>(i + di[k], j + dj[k]) = (sumxiao - reducexiao) / (latemax - numxiao);
                    tu.at<Vec3b>(i + di[k], j + dj[k])[0] = (sumxiaob - reducexiaob) / (latemax - numxiao);
                    tu.at<Vec3b>(i + di[k], j + dj[k])[1] = (sumxiaog - reducexiaog) / (latemax - numxiao);
                    tu.at<Vec3b>(i + di[k], j + dj[k])[2] = (sumxiaor - reducexiaor) / (latemax - numxiao);

                }
                if (maodun.at<float>(i + di[k], j + dj[k]) == 2 && (8 - latemax - numda) != 0) {
                    //obj.at<float>(i + di[k], j + dj[k]) = (sumda - reduceda) / (8 - latemax - numda);
                    tu.at<Vec3b>(i + di[k], j + dj[k])[0] = (sumdab - reducedab) / (8 - latemax - numda);
                    tu.at<Vec3b>(i + di[k], j + dj[k])[1] = (sumdag - reducedag) / (8 - latemax - numda);
                    tu.at<Vec3b>(i + di[k], j + dj[k])[2] = (sumdar - reducedar) / (8 - latemax - numda);

                }
            }

            //中心点修复
            //tu.at<float>(i, j) = centerFix(gray, maxcha, i, j);
        }
    }
    /*
	for (int i = 0; i <= rows - 1; ++i)
	{
		tu.at<Vec3b>(i, 0) = gray.at<Vec3b>(i, 0);
		tu.at<Vec3b>(i, cols - 1) = gray.at<Vec3b>(i, cols - 1);
	}
	for (int j = 0; j <= cols - 1; ++j)
	{
		tu.at<Vec3b>(0, j) = gray.at<Vec3b>(0, j);
		tu.at<Vec3b>(rows - 1, j) = gray.at<Vec3b>(rows - 1, j);
	}
	*/
    return tu;
}

//----------------------[Near_attribution()函数]--------------------
//
//根据邻接的大小点进行八邻域内的所有点的归属划分
//-----------------------------------------------------------------

void Near_attribution(const Mat &gray, Mat &diff, Mat &big, Mat &small, int x, int y, Mat &newgray, Mat &newvote,
                      Mat &newbig, Mat &newsmall, Mat &newdiff) {
    int di[8 + 3] = {0, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 3] = {0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    newvote = obj.clone();
    newbig = big.clone();
    newsmall = small.clone();
    newdiff = diff.clone();
    newgray = gray.clone();


    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int da_num = 0;
            int xiao_num = 0;
            int da_sum = 0;
            int xiao_sum = 0;
            for (int k = 0; k < 9; k++) {
                if (big.at<float>(i + di[k], j + dj[k]) == 1) {
                    da_num++;
                    da_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 222;
                    newbig.at<float>(i + di[k], j + dj[k]) = 1;
                }
                if (small.at<float>(i + di[k], j + dj[k]) == 1) {
                    xiao_num++;
                    xiao_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 1;
                    newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                }
            }
            if (da_num && xiao_num) {
                int da = da_sum / da_num;
                int xiao = xiao_sum / xiao_num;
                for (int k = 0; k < 9; k++) {
                    if ((big.at<float>(i + di[k], j + dj[k]) == 0 && small.at<float>(i + di[k], j + dj[k])) == 0) {
                        if (abs(gray.at<float>(i + di[k], j + dj[k]) - da) >
                            abs(gray.at<float>(i + di[k], j + dj[k]) - xiao)) {
                            newvote.at<float>(i + di[k], j + dj[k]) = 1;
                            newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = xiao;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        } else {
                            newvote.at<float>(i + di[k], j + dj[k]) = 222;
                            newbig.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = da;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        }
                    }

                }


            }


        }
    }


}


/*
if(verity(newvote,i,j)==0)
{
for (int k = 1; k < 9; k++) {
if (big.at<float>(i + di[k], j + dj[k]) == 1)
{
newvote.at<float>(i + di[k], j + dj[k]) = 222;
}
if (small.at<float>(i + di[k], j + dj[k]) == 1)
{
newvote.at<float>(i + di[k], j + dj[k]) = 1;
}
if(vote.at<float>(i + di[k], j + dj[k]) == 0)
{
newvote.at<float>(i + di[k], j + dj[k]) = 0;
} else
{
newvote.at<float>(i + di[k], j + dj[k]) = 3;
}
}
}
*/


void Near_attribution_3(const Mat &gray, Mat &diff, Mat &big, Mat &small, int x, int y, Mat &newgray, Mat &newvote,
                        Mat &newbig, Mat &newsmall, Mat &newdiff) {
    int di[8 + 3] = {0, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 3] = {0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    newvote = obj.clone();
    newbig = big.clone();
    newsmall = small.clone();
    newdiff = diff.clone();
    newgray = gray.clone();


    for (int i = rows - 2; i >= 1; --i) {
        for (int j = cols - 2; j >= 1; --j) {
            int da_num = 0;
            int xiao_num = 0;
            int da_sum = 0;
            int xiao_sum = 0;
            for (int k = 0; k < 9; k++) {
                if (big.at<float>(i + di[k], j + dj[k]) == 1) {
                    da_num++;
                    da_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 222;
                    newbig.at<float>(i + di[k], j + dj[k]) = 1;
                }
                if (small.at<float>(i + di[k], j + dj[k]) == 1) {
                    xiao_num++;
                    xiao_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 1;
                    newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                }
            }
            if (da_num && xiao_num) {
                int da = da_sum / da_num;
                int xiao = xiao_sum / xiao_num;
                for (int k = 0; k < 9; k++) {
                    if ((big.at<float>(i + di[k], j + dj[k]) == 0 && small.at<float>(i + di[k], j + dj[k])) == 0) {
                        if (abs(gray.at<float>(i + di[k], j + dj[k]) - da) >
                            abs(gray.at<float>(i + di[k], j + dj[k]) - xiao)) {
                            newvote.at<float>(i + di[k], j + dj[k]) = 1;
                            newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = xiao;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        } else {
                            newvote.at<float>(i + di[k], j + dj[k]) = 222;
                            newbig.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = da;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        }
                    }

                }

            }


        }
    }


}


void Near_attribution_2(const Mat &gray, Mat &diff, Mat &big, Mat &small, int x, int y, Mat &newgray, Mat &newvote,
                        Mat &newbig, Mat &newsmall, Mat &newdiff) {
    int di[8 + 3] = {0, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 3] = {0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    newvote = obj.clone();
    newbig = big.clone();
    newsmall = small.clone();
    newdiff = diff.clone();
    newgray = gray.clone();


    for (int i = 1; i < rows - 1; ++i) {
        for (int j = cols - 2; j >= 1; --j) {
            int da_num = 0;
            int xiao_num = 0;
            int da_sum = 0;
            int xiao_sum = 0;
            for (int k = 0; k < 9; k++) {
                if (big.at<float>(i + di[k], j + dj[k]) == 1) {
                    da_num++;
                    da_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 222;
                    newbig.at<float>(i + di[k], j + dj[k]) = 1;
                }
                if (small.at<float>(i + di[k], j + dj[k]) == 1) {
                    xiao_num++;
                    xiao_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 1;
                    newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                }
            }
            if (da_num && xiao_num) {
                int da = da_sum / da_num;
                int xiao = xiao_sum / xiao_num;
                for (int k = 0; k < 9; k++) {
                    if ((big.at<float>(i + di[k], j + dj[k]) == 0 && small.at<float>(i + di[k], j + dj[k])) == 0) {
                        if (abs(gray.at<float>(i + di[k], j + dj[k]) - da) >
                            abs(gray.at<float>(i + di[k], j + dj[k]) - xiao)) {
                            newvote.at<float>(i + di[k], j + dj[k]) = 1;
                            newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = xiao;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        } else {
                            newvote.at<float>(i + di[k], j + dj[k]) = 222;
                            newbig.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = da;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        }
                    }

                }


            }


        }
    }


}


void Near_attribution_1(const Mat &gray, Mat &diff, Mat &big, Mat &small, int x, int y, Mat &newgray, Mat &newvote,
                        Mat &newbig, Mat &newsmall, Mat &newdiff) {
    int di[8 + 3] = {0, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 3] = {0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    newvote = obj.clone();
    newbig = big.clone();
    newsmall = small.clone();
    newdiff = diff.clone();
    newgray = gray.clone();


    for (int i = rows - 2; i >= 1; --i) {
        for (int j = 1; j < cols - 1; ++j) {
            int da_num = 0;
            int xiao_num = 0;
            int da_sum = 0;
            int xiao_sum = 0;
            for (int k = 0; k < 9; k++) {
                if (big.at<float>(i + di[k], j + dj[k]) == 1) {
                    da_num++;
                    da_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 222;
                    newbig.at<float>(i + di[k], j + dj[k]) = 1;
                }
                if (small.at<float>(i + di[k], j + dj[k]) == 1) {
                    xiao_num++;
                    xiao_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 1;
                    newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                }
            }
            if (da_num && xiao_num) {
                int da = da_sum / da_num;
                int xiao = xiao_sum / xiao_num;
                for (int k = 0; k < 9; k++) {
                    if ((big.at<float>(i + di[k], j + dj[k]) == 0 && small.at<float>(i + di[k], j + dj[k])) == 0) {
                        if (abs(gray.at<float>(i + di[k], j + dj[k]) - da) >
                            abs(gray.at<float>(i + di[k], j + dj[k]) - xiao)) {
                            newvote.at<float>(i + di[k], j + dj[k]) = 1;
                            newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = xiao;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        } else {
                            newvote.at<float>(i + di[k], j + dj[k]) = 222;
                            newbig.at<float>(i + di[k], j + dj[k]) = 1;
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                newgray.at<float>(i + di[k], j + dj[k]) = da;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        }
                    }

                }

            }


        }
    }


}


//----------------------[verify()函数]--------------------
//
// 判断修正后的图是不是连续的分区
//-----------------------------------------------------------------
int verity(Mat vote, int x, int y) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    for (int k = 1; k < 9; k++) {
        if (vote.at<float>(x + di[k], y + dj[k]) != vote.at<float>(x + di[k - 1], y + dj[k - 1]) &&
            vote.at<float>(x + di[k], y + dj[k]) != vote.at<float>(x + di[k + 1], y + dj[k + 1])) {
            return 0;
        }
    }
    return 1;
}

//----------------------[verify()函数]--------------------
//
// 判断如果是内部区域则停止生长
//1.   区域内的色差较小
//2.  区域中的像素点分布没有规律，比较混乱
//-----------------------------------------------------------------
int verity_break(Mat gray, int x, int y, int th) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int negative_max, positive_max;
    negative_max = OUTPUT_FALSE;
    positive_max = OUTPUT_FALSE;
    int max_p, min_p;
    for (int k = 1; k <= 8; ++k) {
        int diff = static_cast<int>(gray.at<float>(x, y) -
                                    gray.at<float>(x, y));
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
            }
        }
    }
    int dis_pow = static_cast<int>(fabs(positive_max) < fabs(negative_max)
                                   ? fabs(positive_max) : fabs(negative_max));

    if (dis_pow < th) {
        return 1;
    }
}
//----------------------[New_Near_attribution()函数]--------------------
//
// 每次得到起始点，就开始向所有四周改变的点延伸
//-----------------------------------------------------------------


void New_Near_attribution(const Mat &gray, Mat &diff, Mat &big, Mat &small, int x, int y, Mat &newgray, Mat &newvote,
                          Mat &newbig, Mat &newsmall, Mat &newdiff, int th) {
    int di[8 + 3] = {0, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 3] = {0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    newvote = obj.clone();
    newbig = big.clone();
    newsmall = small.clone();
    newdiff = diff.clone();
    newgray = gray.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int da_num = 0;
            int xiao_num = 0;
            int da_sum = 0;
            int xiao_sum = 0;
            for (int k = 0; k < 9; k++) {
                if (big.at<float>(i + di[k], j + dj[k]) == 1) {
                    da_num++;
                    da_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 222;
                    newbig.at<float>(i + di[k], j + dj[k]) = 1;
                }
                if (small.at<float>(i + di[k], j + dj[k]) == 1) {
                    xiao_num++;
                    xiao_sum += gray.at<float>(i + di[k], j + dj[k]);
                    newvote.at<float>(i + di[k], j + dj[k]) = 1;
                    newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                }
            }

            if (da_num && xiao_num) {
                int da = da_sum / da_num;
                int xiao = xiao_sum / xiao_num;
                for (int k = 0; k < 9; k++) {
                    if (big.at<float>(i + di[k], j + dj[k]) == 0 && small.at<float>(i + di[k], j + dj[k]) == 0) {
                        if (abs(gray.at<float>(i + di[k], j + dj[k]) - da) >
                            abs(gray.at<float>(i + di[k], j + dj[k]) - xiao)) {
                            newvote.at<float>(i + di[k], j + dj[k]) = 1;
                            newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                            e_Near_attribution(i + di[k], j + dj[k], newgray, newvote, newbig, newsmall, newdiff, k,
                                               th);
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                //                                newgray.at<float>(i + di[k], j + dj[k]) = xiao;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        } else {
                            newvote.at<float>(i + di[k], j + dj[k]) = 222;
                            newbig.at<float>(i + di[k], j + dj[k]) = 1;
                            e_Near_attribution(i + di[k], j + dj[k], newgray, newvote, newbig, newsmall, newdiff, k,
                                               th);
                            if (diff.at<float>(i + di[k], j + dj[k]) == 1) {
                                //                                newgray.at<float>(i + di[k], j + dj[k]) = da;
                                newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                            }
                        }
                    }

                }

            }


        }
    }

}


void
e_Near_attribution(int x, int y, Mat &newgray, Mat &newvote, Mat &newbig, Mat &newsmall, Mat &newdiff, int p, int th) {
    int di[8 + 3] = {0, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 3] = {0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = newgray.rows;
    int cols = newgray.cols;
    if (x == 0 || y == 0 || x == (newgray.rows - 2) || y == (newgray.cols - 2)) {
        return;
    }

    if (verity_break(newgray, x, y, th) == 1) {
        return;
    }

    int i = x;
    int j = y;
    int da_num = 0;
    int xiao_num = 0;
    int da_sum = 0;
    int xiao_sum = 0;
    for (int k = 0; k < 9; k++) {
        if (newbig.at<float>(i + di[k], j + dj[k]) == 1) {
            da_num++;
            da_sum += newgray.at<float>(i + di[k], j + dj[k]);
            newvote.at<float>(i + di[k], j + dj[k]) = 222;
            newbig.at<float>(i + di[k], j + dj[k]) = 1;
        }
        if (newsmall.at<float>(i + di[k], j + dj[k]) == 1) {
            xiao_num++;
            xiao_sum += newgray.at<float>(i + di[k], j + dj[k]);
            newvote.at<float>(i + di[k], j + dj[k]) = 1;
            newsmall.at<float>(i + di[k], j + dj[k]) = 1;
        }
    }

    //	if (da_num && xiao_num && (p==3 || p==4 || p==5 ) ) {
    if (da_num && xiao_num) {
        //        if (p == 3 || p == 4 || p == 5) {
        int da = da_sum / da_num;
        int xiao = xiao_sum / xiao_num;
        for (int k = 0; k < 9; k++) {
            if (newbig.at<float>(i + di[k], j + dj[k]) == 0 && newsmall.at<float>(i + di[k], j + dj[k]) == 0) {
                if (abs(newgray.at<float>(i + di[k], j + dj[k]) - da) >
                    abs(newgray.at<float>(i + di[k], j + dj[k]) - xiao)) {
                    newvote.at<float>(i + di[k], j + dj[k]) = 1;
                    newsmall.at<float>(i + di[k], j + dj[k]) = 1;
                    e_Near_attribution(i + di[k], j + dj[k], newgray, newvote, newbig, newsmall, newdiff, k, th);
                    if (newdiff.at<float>(i + di[k], j + dj[k]) == 1) {
                        //                            newgray.at<float>(i + di[k], j + dj[k]) = xiao;
                        newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                    }
                } else {
                    newvote.at<float>(i + di[k], j + dj[k]) = 222;
                    newbig.at<float>(i + di[k], j + dj[k]) = 1;
                    e_Near_attribution(i + di[k], j + dj[k], newgray, newvote, newbig, newsmall, newdiff, k, th);
                    if (newdiff.at<float>(i + di[k], j + dj[k]) == 1) {
                        //                            newgray.at<float>(i + di[k], j + dj[k]) = da;
                        newdiff.at<float>(i + di[k], j + dj[k]) = 0;
                    }
                }
            } else {
                continue;
            }
            //                else{
            //                    return;
            //                }
        }
        //        }

    } else {
        return;
    }
}


//----------------------[L0g()函数]--------------------
//
//  按照L0g算子的计算进行改写
//-----------------------------------------------------------------
struct Point1 {
    int x;
    int y;
};

Mat L0g(Mat &gray, Mat &tag, long long sum[], long nums[], Mat &color, long long sumb[], long long sumg[],
        long long sumr[]) {
//Mat L0g(Mat &gray, Mat &tag, long sum[], long nums[],Mat &color,long  sumb[],long sumg[],long sumr[]) {
    int rows = gray.rows;
    int cols = gray.cols;
    int num = 4;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
//    long long sum[1000]={0};
//    long nums[1000]={0};
    obj = tag.clone();
//    int di[8 + 3] = {0, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
//    int dj[8 + 3] = {0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int di[9] = {0, +1, -1, -1, +1, +1, -1, 0, 0};
    int dj[9] = {0, +1, -1, +1, -1, 0, 0, +1, -1};

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {


            queue<Point1> qu;

            if (obj.at<float>(i, j) == 0) {
                Point1 a = {i, j};
                qu.push(a);
                obj.at<float>(i, j) = num;
                nums[num]++;
                sum[num] += (int) gray.at<float>(i, j);
                sumb[num] += (int) color.at<Vec3b>(i, j)[0];
                sumg[num] += (int) color.at<Vec3b>(i, j)[1];
                sumr[num] += (int) color.at<Vec3b>(i, j)[2];
                while (!qu.empty()) {
                    a = qu.front();
//                    obj.at<float>(a.x,a.y) = num;
                    qu.pop();
                    for (int k = 0; k < 9; k++) {
                        if (obj.at<float>(a.x + di[k], a.y + dj[k]) == 0) {
//                            tag.at<float>(i + di[k], j + dj[k]) =num;
                            obj.at<float>(a.x + di[k], a.y + dj[k]) = num;
                            nums[num]++;
                            sum[num] += (int) gray.at<float>(a.x + di[k], a.y + dj[k]);
                            sumb[num] += (int) color.at<Vec3b>(a.x + di[k], a.y + dj[k])[0];
                            sumg[num] += (int) color.at<Vec3b>(a.x + di[k], a.y + dj[k])[1];
                            sumr[num] += (int) color.at<Vec3b>(a.x + di[k], a.y + dj[k])[2];
                            a = {a.x + di[k], a.y + dj[k]};
                            qu.push(a);
                        }
                    }
                }

                num++;

            }


        }
    }


    return obj;
}

//----------------------[type3()函数]--------------------
//
//  type3把大小矛盾点分别进行记录
//-----------------------------------------------------------------
Mat type3(const Mat &big_edge, const Mat &small_edge, const Mat &diff) {
    int rows = big_edge.rows;
    int cols = big_edge.cols;
    Mat type3result(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (big_edge.at<float>(i, j) == 1) {
                type3result.at<float>(i, j) = 2;
            }
            if (small_edge.at<float>(i, j) == 1) {
                type3result.at<float>(i, j) = 1;
            }
            if (diff.at<float>(i, j) == 1) {
                type3result.at<float>(i, j) = 3;
            }
        }
    }
    return type3result;
}

//----------------------[smooth_gary()函数]--------------------
//
//  smooth_gray把大小矛盾点分别进行记录
//  obj 是进行标记的数据其中 0代表内部点,1代表小边点,2代表大边点,3代表矛盾点
//  sum 各区域灰度值的总和
//  nums 各区域像素点的总个数
//  gray 存储计算后的个区域像素点的平局值
//-----------------------------------------------------------------
Mat smooth_gray(Mat &obj, long long sum[], long nums[], int gray[]) {
//Mat smooth_gray(Mat &obj, long sum[], long nums[]) {
    int rows = obj.rows;
    int cols = obj.cols;
//    long gray[1000];
    for (int i = 4; i < 1000; i++) {
        gray[i] = 255;
        if (nums[i] != 0)
            gray[i] = sum[i] / nums[i];
    }
    Mat re(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (obj.at<float>(i, j) > 3)
                re.at<float>(i, j) = gray[(int) obj.at<float>(i, j)];
        }
    }
    return re;

}

//----------------------[smooth_color()函数]--------------------
//
//  smooth_color把大小矛盾点分别进行记录
//-----------------------------------------------------------------
Mat smooth_color(Mat &obj, long long sumb[], long long sumg[], long long sumr[], long nums[]) {
//Mat smooth_color(Mat &obj, long sumb[],long sumg[],long sumr[], long nums[]) {
    int rows = obj.rows;
    int cols = obj.cols;
    long b[1000];
    long g[1000];
    long r[1000];
    Mat color(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 4; i < 1000; i++) {
        b[i] = 255;
        g[i] = 255;
        r[i] = 255;
        if (nums[i] != 0) {
            b[i] = sumb[i] / nums[i];
            g[i] = sumg[i] / nums[i];
            r[i] = sumr[i] / nums[i];
        }

    }
//    Mat re(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (obj.at<float>(i, j) > 3)
//            re.at<float>(i, j)=gray[(int)obj.at<float>(i, j)];
            {
                color.at<Vec3b>(i, j)[0] = b[(int) obj.at<float>(i, j)];
                color.at<Vec3b>(i, j)[1] = g[(int) obj.at<float>(i, j)];
                color.at<Vec3b>(i, j)[2] = r[(int) obj.at<float>(i, j)];
            }

        }
    }
    return color;

}

//----------------------[extend_gray()函数]--------------------
//
//  extend_gray把大小矛盾点分别进行记录
//  obj 各区域的分布
//  gray 各区域值的评价灰度值
//-----------------------------------------------------------------
Mat extend_gray(Mat &obj, int gray[], Mat &src, Mat &smooth) {
    int rows = obj.rows;
    int cols = obj.cols;
    Mat re_gray(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    re_gray = smooth.clone();
    int di[9] = {0, +1, -1, -1, +1, +1, -1, 0, 0};
    int dj[9] = {0, +1, -1, +1, -1, 0, 0, +1, -1};
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (obj.at<float>(i, j) <= 3) {
                int num = 0;
                int pos[8] = {0};
                for (int k = 1; k <= 8; k++) {
                    if (obj.at<float>(i + di[k], j + dj[k]) > 3) {
                        pos[num] = (int) obj.at<float>(i + di[k], j + dj[k]);
                        num++;
                    }
                }
                if (num == 1) {
                    re_gray.at<float>(i, j) = gray[pos[0]];
                    obj.at<float>(i, j) = pos[0];
                } else {
                    int min = 255;
                    int tag = 0;
                    for (int m = 0; m < num; m++) {
                        if (abs(gray[pos[m]] - src.at<float>(i, j)) < min) {
                            min = abs(gray[pos[m]] - (int) src.at<float>(i, j));
                            tag = pos[m];
                        }
                    }
                    re_gray.at<float>(i, j) = gray[tag];
                    obj.at<float>(i, j) = tag;
                }
            }
        }
    }

    return re_gray;
}


//   寻找能量值最小函数
Mat Minimum_capacity(Mat &gray, Mat &cha, Mat &bigsmall, Mat &chasum, Mat &two_areas_min_r,Mat &re) {

    int rows = gray.rows;
    int cols = gray.cols;
    Mat re_gray(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    cha = re_gray.clone();
    bigsmall = re_gray.clone();
    chasum = re_gray.clone();
    two_areas_min_r = re_gray.clone();
    re = re_gray.clone();
    int di[8 + 8 + 8] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0, -1, -1, -1, 0, +1, +1, +1, +0, -1, -1, -1, 0, +1, +1};
    int dj[8 + 8 + 8] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1, +1, 0, -1, -1, -1, +0, +1, +1, +1, 0, -1, -1, -1, +0};

    int times = 1;

    int max_cha=0;
    int min_cha=0;



//    int p_cha[8];
//    int q_cha[8];


    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {


            int tag = 0;

            int cha1=0;// 第一部分差值
            int cha2=0;// 第二部分差值

            int p_cha[8];
            int q_cha[8];

            int min = 2147483647;
            int sum = 0;
            int avg_m = 0;
            int avg_n = 0;

            int max1=0;
            int min1=0;
            int max2=0;
            int min2=0;

            for (int k = 1; k <= 8; k++) {
                sum += gray.at<float>(i + di[k], j + dj[k]);
            }


            int pos[5] = {-1, -1, -1, -1, -1};

            // 0 vs 8
            for (int k = 1; k <= 8; k++) {
                int avg = sum / 8;
                int re = 0;
                for (int m = 0; m <= 7; m++) {
                    re += (gray.at<float>(i + di[m], j + dj[m]) - avg) * (gray.at<float>(i + di[m], j + dj[m]) - avg);
                }

                if (min > re) {
                    min = re;
                    avg_m = avg;
                    avg_n = avg;

                    max1=avg;
                    min1=avg;
                    max2=avg;
                    min2=avg;
                    tag =0;
                }

            }

            //1vs7
            for (int k = 1; k <= 8; k++) {
                int avg = (sum - (int) gray.at<float>(i + di[k], j + dj[k])) / 7;
                p_cha[0] = (int) gray.at<float>(i + di[k], j + dj[k]);
                int re = 0;
                int index=0;
                for (int m = 0; m <= 7; m++) {
                    if (m == k%8) continue;
                    re += (gray.at<float>(i + di[m], j + dj[m]) - avg) * (gray.at<float>(i + di[m], j + dj[m]) - avg);
                    q_cha[index++]=gray.at<float>(i + di[m], j + dj[m]);
                }

                if (min > re) {
                    min = re;
                    pos[0] = k % 8;
                    avg_m = avg;
                    avg_n = p_cha[0];
                    cha1 = p_cha[0];
                    sort(q_cha,q_cha+7);
                    cha2=q_cha[6]-q_cha[0];

                    max1=p_cha[0];
                    min1=p_cha[0];
                    max2=q_cha[6];
                    min2=q_cha[0];
                    tag =1;
                }
            }

            //2vs6
            for (int k = 1; k <= 8; k++) {
                int a1 = (int) gray.at<float>(i + di[k], j + dj[k]);
                p_cha[0] = (int) gray.at<float>(i + di[k], j + dj[k]);
                int a2 = (int) gray.at<float>(i + di[k + 1], j + dj[k + 1]);
                p_cha[1] = (int) gray.at<float>(i + di[k + 1], j + dj[k + 1]);

                int avg1 = (a1 + a2) / 2;

                int avg = (sum - a1 - a2) / 6;
                int re = 0;
                int index=0;
                for (int m = 0; m <= 7; m++) {
                    if (m == k%8 || m == (k + 1) % 8) {
                        re += (gray.at<float>(i + di[m], j + dj[m]) - avg1) *
                              (gray.at<float>(i + di[m], j + dj[m]) - avg1);
                    } else{
                        re += (gray.at<float>(i + di[m], j + dj[m]) - avg) *
                              (gray.at<float>(i + di[m], j + dj[m]) - avg);
                        q_cha[index++]=gray.at<float>(i + di[m], j + dj[m]);
                    }
                }

                if (min > re) {
                    min = re;
                    pos[0] = k % 8;
                    pos[1] = (k + 1) % 8;
                    avg_m = avg;
                    avg_n = avg1;
                    sort(p_cha,p_cha+2);
                    cha1=p_cha[1]-p_cha[0];
                    sort(q_cha,q_cha+6);
                    cha2=q_cha[5]-q_cha[0];

                    max1=p_cha[1];
                    min1=p_cha[0];
                    max2=q_cha[5];
                    min2=q_cha[0];
                    tag =2;
                }
            }


            //3vs5

            for (int k = 1; k <= 8; k++) {
                int a1 = (int) gray.at<float>(i + di[k], j + dj[k]);
                p_cha[0] = (int) gray.at<float>(i + di[k], j + dj[k]);

                int a2 = (int) gray.at<float>(i + di[k + 1], j + dj[k + 1]);
                p_cha[1] = (int) gray.at<float>(i + di[k+1], j + dj[k+1]);

                int a3 = (int) gray.at<float>(i + di[k + 2], j + dj[k + 2]);
                p_cha[2] = (int) gray.at<float>(i + di[k+2], j + dj[k+2]);


                int avg1 = (a1 + a2 + a3) / 3;
                int avg = (sum - a1 - a2 - a3) / 5;
                int re = 0;
                int index=0;
                for (int m = 0; m <= 7; m++) {
                    if (m == k%8 || m == (k + 1) % 8 || m == (k + 2) % 8) {
                        re += (gray.at<float>(i + di[m], j + dj[m]) - avg1) *
                              (gray.at<float>(i + di[m], j + dj[m]) - avg1);

                    } else{
                        re += (gray.at<float>(i + di[m], j + dj[m]) - avg) *
                              (gray.at<float>(i + di[m], j + dj[m]) - avg);
                        q_cha[index++]=gray.at<float>(i + di[m], j + dj[m]);
                    }
                }

                if (min > re) {
                    min = re;
                    pos[0] = k % 8;
                    pos[1] = (k + 1) % 8;
                    pos[2] = (k + 2) % 8;
                    avg_m = avg;
                    avg_n = avg1;
                    sort(p_cha,p_cha+3);
                    cha1=p_cha[2]-p_cha[0];
                    sort(q_cha,q_cha+5);
                    cha2=q_cha[4]-q_cha[0];

                    max1=p_cha[2];
                    min1=p_cha[0];
                    max2=q_cha[4];
                    min2=q_cha[0];
                    tag =3;
                }
            }
            //4vs4

            for (int k = 1; k <= 8; k++) {
                int a1 = (int) gray.at<float>(i + di[k], j + dj[k]);
                p_cha[0] = (int) gray.at<float>(i + di[k], j + dj[k]);

                int a2 = (int) gray.at<float>(i + di[k + 1], j + dj[k + 1]);
                p_cha[1] = (int) gray.at<float>(i + di[k+1], j + dj[k+1]);

                int a3 = (int) gray.at<float>(i + di[k + 2], j + dj[k + 2]);
                p_cha[2] = (int) gray.at<float>(i + di[k+2], j + dj[k+2]);

                int a4 = (int) gray.at<float>(i + di[k + 3], j + dj[k + 3]);
                p_cha[3] = (int) gray.at<float>(i + di[k+3], j + dj[k+3]);

                int avg1 = (a1 + a2 + a3 + a4) / 4;
                int avg = (sum - a1 - a2 - a3 - a4) / 4;
                int re = 0;
                int index=0;
                for (int m = 0; m <= 7; m++) {
                    if (m == k%8 || m == (k + 1) % 8 || m == (k + 2) % 8 || m == (k + 3) % 8) {
                        re += (gray.at<float>(i + di[m], j + dj[m]) - avg1) *
                              (gray.at<float>(i + di[m], j + dj[m]) - avg1);

                    } else{
                        re += (gray.at<float>(i + di[m], j + dj[m]) - avg) *
                              (gray.at<float>(i + di[m], j + dj[m]) - avg);
                        q_cha[index++]=gray.at<float>(i + di[m], j + dj[m]);
                    }
                }

                if (min > re) {
//                    p_cha[0] = (int) gray.at<float>(i + di[k], j + dj[k]);
//                    p_cha[1] = (int) gray.at<float>(i + di[k+1], j + dj[k+1]);
//                    p_cha[2] = (int) gray.at<float>(i + di[k+2], j + dj[k+2]);
//                    p_cha[3] = (int) gray.at<float>(i + di[k+3], j + dj[k+3]);

                    min = re;
                    pos[0] = k % 8;
                    pos[1] = (k + 1) % 8;
                    pos[2] = (k + 2) % 8;
                    pos[3] = (k + 3) % 8;
                    avg_m = avg;
                    avg_n = avg1;
                    sort(p_cha,p_cha+4);
                    cha1=p_cha[3]-p_cha[0];
                    sort(q_cha,q_cha+4);
                    cha2=q_cha[3]-q_cha[0];

                    max1=p_cha[3];
                    min1=p_cha[0];
                    max2=q_cha[3];
                    min2=q_cha[0];
                    tag =4;
                }
            }

            mc_smooth(re_gray, gray, pos, i, j, avg_m, avg_n, bigsmall, chasum);


//                 if (i == 51 && j == 58) {
                 if (i == 33 && j == 58) {
                     test_mc(re_gray, gray, pos, i, j, avg_m);
                     times++;
//                     for(int p=0;p<tag;p++){
//                         cout<<p_cha[p]<<" ";
//                     }
//                     cout<<endl;
                     cout << "八邻域区域平均灰度值: " << sum / 8 << endl<<endl;
                     cout << "第一部分最大差值: " << cha1 << endl;
                     cout << "第一部分最大值: " << max1 << endl;
                     cout << "第一部分最小值: " << min1 << endl;

                     cout << endl;

//                     for(int p=0;p<8-tag;p++){
//                         cout<<q_cha[p]<<" ";
//                     }
//
//                     cout<<endl;
                     cout << "第二部分最大差值: " << cha2 << endl;
                     cout << "第二部分最大值: " << max2 << endl;
                     cout << "第二部分最小值: " << min2 << endl;
//                     cout << "两个区域的平均值差值: " << abs(avg_m - avg_n) << endl;

                     cout<<endl;
                     cout<<"排序法的结果"<< endl;
                     testsortpixel(gray, 1, i, j);
                     cout<<endl;
                     cout<<"最大正负方法的结果"<< endl;
                     test_mfindBigSmallArea(gray,i,j);
                     cout<<endl;
                 }



            two_areas_min_r.at<float>(i, j) = two_areas_min(pos, gray, i, j);

            if(cha1>cha2){
                max_cha=cha1;
                min_cha =cha2;
            }
            else {
                max_cha =cha2;
                min_cha = cha1;
            }




            int cha_jun=abs(avg_m - avg_n);
            cha.at<float>(i, j) = cha_jun;
//            if(cha_jun>1.2*max_cha){
            if(avg_m>avg_n+10||avg_n<avg_m-10){
                re.at<float>(i, j)=255;
            }



        }
    }

    return re_gray;

}

//  找到最小能量值之后划分为两个区域,分别进行均值处理
void mc_smooth(Mat &re, Mat &gray, int pos[], int i, int j, int avg_m, int avg_n, Mat &bigsmall, Mat &chasum) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int sum = 0;
    int num = 0;
    int avg = 0;
    int tag_m = 1;
    int tag_n = 1;
    if (avg_m > avg_n) {
        tag_m = 2;
    } else tag_n = 2;

    if (pos[0] == -1) {
        for (int k = 0; k < 8; k++) {
            re.at<float>(i + di[k], j + dj[k]) = avg_m;
            chasum.at<float>(i + di[k], j + dj[k]) += abs(avg_m - gray.at<float>(i + di[k], j + dj[k]));
            bigsmall.at<float>(i + di[k], j + dj[k]) = 0;
        }
        return;
    }

    for (int k = 0; k < 8; k++) {
        re.at<float>(i + di[k], j + dj[k]) = avg_m;
        chasum.at<float>(i + di[k], j + dj[k]) += abs(avg_m - gray.at<float>(i + di[k], j + dj[k]));
        bigsmall.at<float>(i + di[k], j + dj[k]) = tag_m;
    }

//    for (int k = 0; k < 5; k++) {
//        if (pos[k] != -1) {
//            sum += gray.at<float>(i + di[pos[k]], j + dj[pos[k]]);
//            num++;
//        }
//    }

//    avg = sum / num;

    for (int k = 0; k < 5; k++) {
        if (pos[k] != -1) {
            re.at<float>(i + di[pos[k]], j + dj[pos[k]]) = avg;
            chasum.at<float>(i + di[pos[k]], j + dj[pos[k]]) += abs(
                    avg_n - gray.at<float>(i + di[pos[k]], j + dj[pos[k]]));
            chasum.at<float>(i + di[pos[k]], j + dj[pos[k]]) -= abs(
                    avg_m - gray.at<float>(i + di[pos[k]], j + dj[pos[k]]));
            bigsmall.at<float>(i + di[k], j + dj[k]) = tag_n;
        }
    }

    if (pos[0] != -1 && pos[1] == -1) {
        for (int k = 0; k < 8; k++) {
            bigsmall.at<float>(i + di[k], j + dj[k]) = 0;
        }
    }


}


//  找到最小能量值之后划分为两个区域,分别进行均值处理
void test_mc_smooth(Mat &re, Mat &gray, int pos[], int i, int j, int avg_m, int avg_n, Mat &bigsmall, Mat &chasum) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};


    if (pos[0] == -1) {
        for (int k = 0; k < 8; k++) {
            chasum.at<float>(i + di[k], j + dj[k]) += abs(avg_m - gray.at<float>(i + di[k], j + dj[k]));
        }
        return;
    }

    for (int k = 0; k < 8; k++) {
        chasum.at<float>(i + di[k], j + dj[k]) += abs(avg_m - gray.at<float>(i + di[k], j + dj[k]));
    }

    for (int k = 0; k < 5; k++) {
        if (pos[k] != -1) {
            chasum.at<float>(i + di[pos[k]], j + dj[pos[k]]) += abs(
                    avg_n - gray.at<float>(i + di[pos[k]], j + dj[pos[k]]));
            chasum.at<float>(i + di[pos[k]], j + dj[pos[k]]) -= abs(
                    avg_m - gray.at<float>(i + di[pos[k]], j + dj[pos[k]]));
        }
    }


}


//  测试函数只是为了输出结果
void test_mc(Mat &re, Mat &gray, int pos[], int i, int j, int avg_m) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int sum = 0;
    int num = 0;
    int avg = 0;

    cout << "原始区域部分的像素点情况 :  " << endl;
    for (int k = 0; k < 8; k++) {
        re.at<float>(i + di[k], j + dj[k]) = avg_m;
        cout << gray.at<float>(i + di[k], j + dj[k]) << " ";
    }

    for (int k = 0; k < 5; k++) {
        if (pos[k] != -1) {
            sum += gray.at<float>(i + di[pos[k]], j + dj[pos[k]]);
            num++;
        }
    }

    if(num==0){
        cout<<endl<<"所有像素点相同,同一区域"<< endl;
        return;
    }

    avg = sum / num;
    cout<<endl;
    for (int k = 0; k < 5; k++) {
        if (pos[k] != -1) {
            re.at<float>(i + di[pos[k]], j + dj[pos[k]]) = avg;
            cout << pos[k] << " : " << gray.at<float>(i + di[pos[k]], j + dj[pos[k]]) << endl;
        }
    }

    cout << endl << "这一部分的均值:  " << avg;
    cout << endl << "剩下部分的均值:  " << avg_m << endl;

    cout << "两区域的差值:  " << abs(avg_m - avg) << endl;
    cout << endl;

}

//  判断是否是边缘点
Mat mc_judge(Mat &chasum, Mat Two_areas_min) {
    int rows = chasum.rows;
    int cols = chasum.cols;
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    Mat re(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int max = (int) chasum.at<float>(i + di[0], j + dj[0]);

            for (int k = 1; k < 8; k++) {
                if (max < chasum.at<float>(i + di[k], j + dj[k]))
                    max = (int) chasum.at<float>(i + di[k], j + dj[k]);
            }

            if (max/8  < Two_areas_min.at<float>(i, j)) {
                re.at<float>(i, j) = 1;
            }

        }
    }

    return re;
}

//  新的判断是否是边缘点
//Mat mc_judge1(int cha1,int cha2,int chajun) {
//    int rows = chasum.rows;
//    int cols = chasum.cols;
//    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
//    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
//    Mat re(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
//
//    for (int i = 1; i < rows - 1; ++i) {
//        for (int j = 1; j < cols - 1; ++j) {
//            int max = (int) chasum.at<float>(i + di[0], j + dj[0]);
//
//            for (int k = 1; k < 8; k++) {
//                if (max < chasum.at<float>(i + di[k], j + dj[k]))
//                    max = (int) chasum.at<float>(i + di[k], j + dj[k]);
//            }
//
//            if (max/8  < Two_areas_min.at<float>(i, j)) {
//                re.at<float>(i, j) = 1;
//            }
//
//        }
//    }
//
//    return re;
//}

// 两块区域的最小差值
int two_areas_min(int pos[], Mat &gray, int i, int j) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int are[] = {-2, -2, -2, -2, -2};
    int index = 0;
    for (int k = 0; k < 5; k++) {
        if (pos[k] != -1) {
            are[index] = pos[k];
            index++;
        }
    }

    if (pos[0] == -1)
        return 0;

    int num1[index];
    int num2[8 - index];

    int index1 = 0;
    int index2 = 0;
    sort(are, are + index);
    for (int k = 0; k < 8; k++) {
        if (are[index1] == k) {
            num1[index1] = (int) gray.at<float>(i + di[k], j + dj[k]);
            index1++;
        } else {
            num2[index2] = (int) gray.at<float>(i + di[k], j + dj[k]);
            index2++;
        }
    }

    int min = 256;
    for (int m = 0; m < index1; m++) {
        for (int n = 0; n < index2; n++) {
            if (abs(num1[m] - num2[n]) < min) {
                min = abs(num1[m] - num2[n]);
            }
        }
    }


    return min;

}

//   画出边缘点,边缘点显示黑色
Mat mc_edge(Mat &mc_judge) {
    int rows = mc_judge.rows;
    int cols = mc_judge.cols;
    Mat re(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if(mc_judge.at<float>(i,j)==0)
                re.at<float>(i,j)=255;
        }
    }
    return re;
}

//By majpyi. 2018.3.5
