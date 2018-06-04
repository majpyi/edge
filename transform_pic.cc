//����ͼ��ķ���
#include "transform.h"
#include <cmath>
#include <algorithm>
//#include <basic_gx.h>
//#include <use.h>

using namespace std;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ �����ķָ��� ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ��һ����   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

//----------------------[maxGradAndDirection()����]---------------------
//����8����������ݶ� 0~255 �� ����ݶȷ��� 0 ~ 7
//maxGradAndDirection���� ���� maxGrad() �� maxGradAndDirection�����Ĺ���
//----------------------------------------------------------------------
void maxGradAndDirection(const Mat &src, Mat &obj1, Mat &obj2) {
    //ԭ��������ݶȾ�������ݶȷ������
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float tmpmax = -1, tmp;
            int kmax = -1;//��¼8����������ݶȷ���
            for (int k = 0; k < 8; ++k) {
                tmp = fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k]));
                if (tmp > tmpmax) {
                    kmax = k;
                    tmpmax = tmp;//tmpmax��¼����ݶ�
                }
            }
            obj1.at<float>(i, j) = tmpmax;
            obj2.at<float>(i, j) = kmax;
        }
    }

}

//----------------------[maxGrad()����]---------------------
//����8����������ݶ� 0~255
//----------------------------------------------------------
Mat maxGrad(const Mat &src) {
    //src�ҶȾ�����������CV_32F, pΪ��ֵ, ���Ϊ0/255
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float tmpmax = -1, tmp;
            for (int k = 0; k < 8; ++k) {
                tmp = fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k]));
                if (tmp > tmpmax) {
                    tmpmax = tmp;//tmpmax��¼����ݶ�
                }
            }
            obj.at<float>(i, j) = tmpmax;
        }
    }
    return obj;
}

//----------------------[maxGradDirection()����]---------------------
//����8����������ݶ� 0~7
//------------------------------------------------------------------

Mat maxGradDirection(const Mat &src) {
    //����ݶȷ��� 0-7
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

//----------------------[samePixelCnt()����]-------------------------
//����8��������ͬ��ĸ���������p����ֵ,��0~8�����Ƶ�
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


//----------------------[clusterDistribution()����]---------------------
//Ⱥ�ֲ� xxx
//---------------------------------------------------------------------
Mat clusterDistribution(const Mat &src, int p) {
    //Ⱥ�ֲ�0~8
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

//----------------------[bigSmallEdge()����]---------------------
//Ⱥ�ֲ� ��С�ߣ� ���Ϊ 1 ��ɫ��С�� Ϊ 0 ��ɫ
//--------------------------------------------------------------
//��С�ߣ����Ϊ1��ɫ��С��Ϊ0��ɫ��
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
                obj.at<float>(i, j) = SMALL_EDGE;//С��Ϊ0
            } else if (src.at<float>(i, j) >= pixel[kmax + 1]) {
                obj.at<float>(i, j) = BIG_EDGE;//���Ϊ1
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
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ �����ķָ��� ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   �ڶ�����   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[model32Edge()����]---------------------
//model32edge()�ǳ���ʦ��17.9����ķ��������ڱ�Ե���
//����Mat�͵�ͼ������32��ģ��ĵ�
//�������߸�������1.Mat�͵ĻҶ�ͼ 2.�洢ģ��ĳ���int���飬
//3.�洢��Ե��λ�õ�����pair<int,int>  4.ģ�������Сint
//5.ͳ��ģ���Ǳ�Ե�������int������ 6. ͳ��ͼ����ģ�����ֵ�int����
//7.��Ե������edges_num 8.��ֵp
//--------------------------------------------------------------
Mat model32Edge(const Mat &src, const int *model, int_pr *edge_station,
                int models_num, int *m_point, int *m_edgepo, int edges_num, int p) {
    //p����ֵ
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));
    int edges = 0; //ͳ�Ƶ��ڼ�����Ե��
    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            bit8 b8;
            for (int k = 0; k < 8; ++k) {
                if (fabs(src.at<float>(i, j) - src.at<float>(i + di[k], j + dj[k])) <= p) {
//                    b8[k] = 1;//1���ڲ���
                } else {
//                    b8[k] = 0;//0�Ǳ�
                }
            }
            bit8 min_b8 = min_bit8(b8);//����Ϊ36ģ��֮һ
            int temp_b8 = min_b8.to_ulong();//�ݴ�ģ��ֵ
            obj.at<float>(i, j) = float(temp_b8);//������Ϊģ��㣨������Ϊ36��ģ��֮һ��

//ͳ��ģ������������ֲ��ң�
#if(1)
            //�ö��ֲ����ҵ�ģ�������λ�ã�����ģ��������1
            int model_stat = binary_cut_search(model, models_num, temp_b8);
            m_point[model_stat]++;
            //ģ����Ǳ�Ե�������
            if (edges < edges_num && i == edge_station[edges].first && j == edge_station[edges].second) {
                //�ڱ�Ե��λ�ã�ͳ��ģ������
                edges++;
                m_edgepo[model_stat]++;
            }
#endif

        }
    }
    return obj;
}


//----------------------[bigSmallRegionNum()����]-----------------------
//bigSmallRegionNum()��С��������
//���� 1 Ϊ Mat ���� ԭͼ������2 int���� ȷ�����ش����� ����С����
//---------------------------------------------------------------------

Mat bigSmallRegionNum(const Mat &src, int tag) {
    //0����С����1���ش�����,Ĭ�ϲ���Ϊ0������С��������
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    Mat obj(src.rows, src.cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            float pixel[8 + 1], diff[8] = {0};

            for (int k = 0; k <= 8; ++k) {
                //pexel����һȦ�Ÿ���
                pixel[k] = src.at<float>(i + di[k], j + dj[k]);
            }
            for (int k = 0; k < 8; ++k) {
                //һȦ���������������
                diff[k] = fabs(pixel[k + 1] - pixel[k]);
            }

            float tmp;
            int kmax = -1, nummax, posmax;
            float tmpmax = -1;
            for (int k = 0; k < 8; ++k) {
                if (diff[k] > tmpmax) {
                    tmpmax = diff[k];//��¼��ֵ���ĵ�
                    kmax = k;//��¼���λ��
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
                //������
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
                //С����
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
                //С����
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
                //������
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


//----------------------[binaryZation()����]-----------------------
//binaryZation()�ο�ͼ��ֵ�� 0/1
//���� 1 Ϊ Mat ���� ԭͼ������2 int���� ȷ�����ش����� ����С����
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


//----------------------[binarytoZero()����]-----------------------
//binarytoZero() ��ֵ��Ϊ0
//С����ֵ p �ļ�Ϊ0
//---------------------------------------------------------------------

Mat binarytoZero(const Mat &src, int p) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (src.at<float>(i, j) <= p) {
                obj.at<float>(i, j) = OUTPUT_FALSE;//��
            } else {
                obj.at<float>(i, j) = INSIDE_WHITE;//�޹ص�
            }
        }
    }
    return obj;
}


//----------------------[edge_cnt()����]-----------------------
//edge_cnt()ͳ�Ʊ�Ե�����
//һ����Ϊ����0�Ǳ�Ե�㣬p��ֵĬ��Ϊ0��������ʱ��ʼ��Ϊ0
//-------------------------------------------------------------

int edge_cnt(const Mat &src, int p) {
    int rows = src.rows;
    int cols = src.cols;
    int edge_p = 0;
    int edge_per = p;//��Ե�����ֵ
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            if (src.at<float>(i, j) == edge_per) {
                edge_p++;
            }

        }
    }
    return edge_p;
}

//-----------------[edge_station_p_arr()����]--------------------
//edge_station_p_arr()��pair�����¼λ��
//Mat���ͱ����ֵ����ר��ͼ��int_ptr����Ϊpair<int, int>��¼����λ��
//һ����Ϊ����0�Ǳ�Ե�㣬p��ֵĬ��Ϊ0��������ʱ��ʼ��Ϊ0
//-------------------------------------------------------------

void edge_station_p_arr(const Mat &src, int_pr *e_station, int p) {
    int rows = src.rows;
    int cols = src.cols;
    int edge_per = p;//��Ե�����ֵ
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

//----------------------[enlarge_edge()����]-----------------------
//������Ե��
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
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ �����ķָ��� ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ��������   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[Distogram_gray()����]--------------------
//����ֱ��ͼ
//----------------------------------------------------------------
Mat Distogram_gray(const Mat &gray) {
    //Ϊ����ֱ��ͼ���ñ���
    //��������Ҫ�����ͼ���ͨ����������Ҫ����ͼ����ĸ�ͨ����bgr�ռ���Ҫȷ������ b��g��r�ռ䣩
    int channels = 0;
    int scale = 1;//������״ͼ�Ŀ��
    //Ȼ������������Ľ���洢�� �ռ� ����MatND�������洢���
    MatND dstHist;
    int size = 256;
    //��������ֱ��ͼ��ÿһ��ά�ȵ� ��������Ŀ�����ǽ���ֵ���飬���ж����飩
    int histSize[] = {256};       //�������д��int histSize = 256;   ��ô������ü���ֱ��ͼ�ĺ�����ʱ�򣬸ñ�����Ҫд &histSize
    //�����ȷ��ÿ��ά�ȵ�ȡֵ��Χ�����Ǻ����������
    //���ȵö���һ�����������洢 ����ά�ȵ� ��ֵ��ȡֵ��Χ
    float midRanges[] = {0, 256};
    const float *ranges[] = {midRanges};

    calcHist(&gray, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);

    //calcHist  �������ý�����dstHist�����н������� ֱ��ͼ����Ϣ  ��dstHist��ģ�溯�� at<Type>(i)�õ���i��������ֵ
    //at<Type>(i, j)�õ���i�����ҵ�j��������ֵ

    //��ʼֱ�۵���ʾֱ��ͼ��������ֱ��ͼ
    //�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��
    Mat hist_img = Mat::zeros(size, size * scale, CV_8UC3);
    //��Ϊ�κ�һ��ͼ���ĳ�����ص��ܸ��������п��ܻ��кܶ࣬�ᳬ���������ͼ��ĳߴ磬�������������ȶԸ������з�Χ������
    //���� minMaxLoc�������õ�����ֱ��ͼ������ص�������
    double g_dHistMaxValue, g_dHistMinValue;
    minMaxLoc(dstHist, &g_dHistMinValue, &g_dHistMaxValue, 0, 0);
    //�����صĸ������ϵ� ͼ������Χ��
    //����ֱ��ͼ�õ�������
    for (int i = 0; i < 256; i++) {
        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);

        //line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));
        rectangle(hist_img, Point(i * scale, size - 1), Point((i + 1) * scale - 1, size - value),
                  CV_RGB(255, 255, 255));
    }
    return hist_img;

}

//----------------------[discrimin_pow()����]--------------------
//���ֶ�
//ȡmin{+max_a(i)-a(i+1��,-max_a(i)-a(i+1��}
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
            //�������ֶ�
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


//----------------------[find_maxpoint()����]--------------------
//��ͼ������
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

//----------------------[find_minpoint()����]--------------------
//��ͼ����С��
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


//----------------------[add_point()����]--------------------
//����
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


//----------------------[minus_point()����]--------------------
//���
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
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ �����ķָ��� ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ���Ĳ���   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[Distogram_gx()����]--------------------
//ghostxiu��д��ֱ��ͼ���ƺ���
//-------------------------------------------------------------
Mat Distogram_gx(const Mat &gray, int *dis_arr, int max_len) {
    int scale = 8;//��״ͼ���


    mark_dis_arr(gray, dis_arr, max_len);//�������ֶ�ֱ��ͼ����

    //��������
    int zero_num = dis_arr[0];
    //cout << "zero : " << zero_num << endl;
    dis_arr[0] = OUTPUT_FALSE;//��0
    int max_point_num = max_arr_num(dis_arr, max_len);
    //zoom_dis_arr�����ź������,height�ǻ�ͼ�߶�
    int *zoom_dis_arr = new int[max_len];
    if ((zero_num / max_point_num) < 20) {
        dis_arr[0] = zero_num;
        max_point_num = max_arr_num(dis_arr, max_len);
    }

    zoom_dis_arr = calcu_dis_arr(dis_arr, max_len, max_point_num);

    //���Ƹ߶�
    int height = max_arr_num(zoom_dis_arr, max_len);
    height = calcu_dis_arr_height(max_len, height, scale);

    if (zoom_dis_arr[0] == 0) {
        zoom_dis_arr[0] = height - 1;
    }

    Mat obj(height, max_len * scale, CV_32F, Scalar(OUTPUT_FALSE));//�洢ֱ��ͼ��Ϣ������

    for (int i = 0; i < max_len; ++i) {
        int scl_i = i * scale;//ÿ������scale���
        for (int p = 0; p < scale; ++p) {
            obj.at<float>(zoom_dis_arr[i], scl_i + p) = INSIDE_WHITE;//ֱ��ͼ�ⶥ
        }

        for (int j = 0; j < zoom_dis_arr[i]; ++j) {
            obj.at<float>(j, scl_i) = INSIDE_WHITE;//���
            obj.at<float>(j, scl_i + scale - 1) = INSIDE_WHITE;//�ұ�
        }
    }
    dis_arr[0] = zero_num;

    delete[] zoom_dis_arr;//������������Ŀռ�
    Mat in_obj = Invert_Mat(obj);
    return in_obj;
}

//----------------------[Mirror_Mat()����]--------------------
//Mat������ת������0���л�����1��
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

//----------------------[Mirror_Mat()����]--------------------
//���µ���
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


//----------------------[mark_dis_arr()����]------------------
//��¼ֱ��ͼ����
//-----------------------------------------------------------
void mark_dis_arr(const Mat &gray, int *dis_arr, int max_len) {
    //dis_arr�Ǵ洢ֱ��ͼ������Ϣ�Ķ�ά����
    std::fill(&dis_arr[0], &dis_arr[0] + max_len, OUTPUT_FALSE);
    int rows = gray.rows;
    int cols = gray.cols;

    //��¼ֱ��ͼ����Ԫ��

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int ij_value = gray.at<float>(i, j);
            dis_arr[ij_value]++;
        }
    }

}


//----------------------[calcu_dis_arr()����]------------------
//Ϊ�˷���۲죬��ֱ��ͼ��һ���������
//-----------------------------------------------------------
int *calcu_dis_arr(const int *arr, int max_len, int max_point_num) {
    int zoom_scale = -(max_point_num / max_len);//�������ű���
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


//----------------------[calcu_dis_arr_height()����]------------------
//������ʾ�߶�
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
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ �����ķָ��� ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ���岿��   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

//----------------------[Canny_edge()����]------------------
//Canny ��Ե��ⷨ 2017.11.14
//---------------------------------------------------------
Mat CannyEdge(const string &pic_name) {
    //ʹ��Canny���ӽ��б�Ե���
    Mat org;
    // ��1����ȡԭͼ��
    read_src(org, pic_name);


    Mat canny_src = org.clone();
    /*
	Mat dst_c, edge,gray;
	// ��2��������srcͬ���ͺʹ�С�ľ���(dst)
	dst_c.create(canny_src.size(), canny_src.type());
	// ��3��תΪ�Ҷ�ͼ��
	cvtColor(canny_src, gray, COLOR_RGB2GRAY);


	// ��3������ʹ�� 3x3�ں�������
	blur(gray, edge, Size(3, 3));

	// ��4������Canny����
	Canny(edge, edge, 3, 9, 3);

	// ��5����g_dstImage�ڵ�����Ԫ������Ϊ0
	dst_c = Scalar::all(0);

	// ��6��ʹ��Canny��������ı�Եͼg_cannyDetectedEdges��Ϊ���룬����ԭͼg_srcImage����Ŀ��ͼg_dstImage��
	canny_src.copyTo(dst_c, edge);
	return dst_c;
	*/
    Canny(org, canny_src, 150, 100, 3);
    return canny_src;

}

//----------------------[Canny_edge()����]------------------
//Sobel ��Ե��ⷨ 2017.11.14
//---------------------------------------------------------
Mat SobelEdge(const string &pic_name) {
    //ʹ��Sobel���ӽ��б�Ե���
    //��1������ grad_x �� grad_y ����
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, dst;

    //��2������ԭʼͼ
    Mat org;
    read_src(org, pic_name, 1);
    //��3���� X�����ݶ�
    Sobel(org, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    //imshow("��Ч��ͼ�� X����Sobel", abs_grad_x);

    //��4����Y�����ݶ�
    Sobel(org, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);

    convertScaleAbs(grad_y, abs_grad_y);
    //imshow("��Ч��ͼ��Y����Sobel", abs_grad_y);

    //��5���ϲ��ݶ�(����)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
    return dst;
}

//----------------------[LaplacianEdge()����]------------------
//������˹��� 2017.11.14
//------------------------------------------------------------

Mat LaplacianEdge(const string &pic_name) {
    //��0����������
    Mat src_l, src_l_gray, dst_l, abs_dst_l;
    //��1����ȡԴ�ļ�

    read_src(src_l, pic_name);
    //��2����˹�˲���������
    GaussianBlur(src_l, src_l, Size(3, 3), 0, 0, BORDER_DEFAULT);
    //��3��תΪ�Ҷ�ͼ
    cvtColor(src_l, src_l_gray, COLOR_RGB2GRAY);
    //��4��ʹ��������˹
    Laplacian(src_l_gray, dst_l, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    //��5���������ֵ������Ϊ8λ
    convertScaleAbs(dst_l, abs_dst_l);

    return abs_dst_l;
}

//----------------------[scharr()����]------------------
//scharr�˲��� 2017.11.14
//-----------------------------------------------------
Mat ScharEdge(const string &pic_name) {
    //��0����������
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, dst_s;
    //��1����ȡԴ�ļ�
    Mat org;
    read_src(org, pic_name);
    //��2����X�����ݶ�
    Scharr(org, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    //��3����Y�����ݶ�
    Scharr(org, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    //��4�����ƺϲ��ݶ�
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst_s);


    return dst_s;
}


//----------------------[LocalMaxDirection()����]---------------------
//����8�����ھֲ�����λ�� 2017.11.16
//����Ϊ1.Mat�Ҷ�ͼ�� �� 2.pair<int, int> λ�þ���
//����ֵΪ���鳤��
//-------------------------------------------------------------------
int LocalMaxDirection(const Mat &src, int_pr *dis_dir) {
    //src�ҶȾ�����������CV_32F, pΪ��ֵ, ���Ϊ0/255
    int di[8] = {+0, -1, -1, -1, 0, +1, +1, +1};
    int dj[8] = {+1, +1, 0, -1, -1, -1, +0, +1};
    int maxl_cnt = 0;
    //ͳ�ƾֲ���������
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

    //��¼�ֲ�����λ��
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

//----------------------[LocalMaxDirection()����]---------------------
//ƴ�Ӷ��λ������ 2017.11.16
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

//----------------------[LocalMaxDirection()����]---------------------
//�������飨���溯�������ذ汾) 2017.11.17
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


//----------------------[eli_local_max()����]---------------------
//�޳��ֲ����� 2017.11.17
//---------------------------------------------------------------
Mat eli_local_max(const Mat &src, const int_pr *loc_dir) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    int top = 0;//��¼���ֵλ������λ��
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


//----------------------[eli_local_max()����]---------------------
//�����ҳ��ֲ����� 2017.11.17
//����1.const Mat & �� ��ͼ��һ��Ϊ���ֶ�ͼ�� 2.int �� ��������
//����ֵ pair<int,int> ���ͣ���¼�ֲ��������
//---------------------------------------------------------------

void iter_local_max_edge(const Mat &dis_pow, int times, int_pr *edge_loc_max) {
    int_pr *local_max_pt = new int_pr[1];
    int_pr *tmp_add = new int_pr[1];
    int len, len_t;
    Mat tmp_mat = dis_pow;
    for (int i = 0; i < times; ++i) {
        cout << "��" << i + 1 << "��Ѱ�Ҿֲ�����" << endl;
        int_pr *tmp_max_arr = new int_pr[1];
        int len_tmp = LocalMaxDirection(tmp_mat, tmp_max_arr);
        tmp_mat = eli_local_max(tmp_mat, tmp_max_arr);
        if (i > 0) {
            len = Int_pr_cpy(tmp_mat, tmp_max_arr, len_tmp, tmp_add, len_t, local_max_pt);
            len_t = Int_pr_cpy(tmp_mat, local_max_pt, len, tmp_add);
        } else {
            len_t = Int_pr_cpy(tmp_mat, local_max_pt, len, tmp_add);
        }
        string src_name = "��";
        src_name.append(num_to_string(i)).append("��Ѱ��ͼ");
        show_src(tmp_mat, src_name);
        delete[] tmp_max_arr;
    }

    len = Int_pr_cpy(tmp_mat, tmp_add, len_t, edge_loc_max);
    delete[] local_max_pt;
    delete[] tmp_add;
}


//----------------------[BinZeroParr()����]-----------------------
//BinZeroParr() ����pair<int,int> ��ı�Ե���齫ԭͼ��ֵ��Ϊ0��255
//---------------------------------------------------------------
Mat BinZeroParr(const Mat &src, const int_pr *edge_arr_loc) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    int top = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == edge_arr_loc[top].first && j == edge_arr_loc[top].second) {
                obj.at<float>(i, j) = OUTPUT_FALSE;//��
                top++;
            } else {
                obj.at<float>(i, j) = INSIDE_WHITE;//�޹ص�
            }
        }
    }
    return obj;
}

//----------------------[find_max_local()����]-----------------------
//find_max_local() Ѱ�Ҿֲ�����
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

//----------------------[find_max_local()����]-----------------------
//localArea_edge() ��ֲ�����Ե
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
                //��¼���������ֵ����Сֵ
                if (src.at<float>(i + di[k], j + dj[k]) < minl) {
                    minl = src.at<float>(i + di[k], j + dj[k]);
                } else if (src.at<float>(i + di[k], j + dj[k]) > maxl) {
                    maxl = src.at<float>(i + di[k], j + dj[k]);
                }

                //�ֳ���������
                if (src.at<float>(i + di[k], j + dj[k]) > src.at<float>(i, j)) {
                    max_area[max_ct] = make_pair(i + di[k], j + dj[k]);
                    max_ct++;
                } else {
                    min_area[min_ct] = make_pair(i + di[k], j + dj[k]);
                    min_ct++;
                }

                //�ж����ĵ������ĸ�����
                {

                }
                //������ĵ��ǲ���������������

                {

                }
            }


        }
    }

    return obj;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ �����ķָ��� ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//----------------------[findBigSmallArea()����]--------------------
//���ִ�С���򣬲���ͶƱͼ������ֵ��12.18
//�������� find_pos_bigSmallArea()��WeightBigSmall()�����������
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
            if (dis_pow < th) {//����������ֵ�ĵ�
                continue;
            }

            WeightBigSmall(gray, obj, i, j, max_p, min_p);
        }
    }

    return obj;
}

//----------------------[find_pos_bigSmallArea()����]--------------
//findBigSmallArea()������һ����
//Ѱ�Ҵ������ǰ������λ��
//-----------------------------------------------------------------
int find_pos_bigSmallArea(const Mat &gray, int i, int j, int &max_p, int &min_p) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int negative_max, positive_max;
    negative_max = positive_max = OUTPUT_FALSE;
    //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
    //�������С����
    for (int k = 1; k <= 8; ++k) {
        int diff = gray.at<float>(i + di[k], j + dj[k]) -
                   gray.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //����������һ�����Ǹ���Сǰ��ĵ�
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //������ĵ�һ��������������ĵ�
            }
        }
    }
    int dis_pow = fabs(positive_max) < fabs(negative_max)
                  ? fabs(positive_max) : fabs(negative_max);
    return dis_pow;

}

//----------------------[WeightBigSmall()����]---------------------
//findBigSmallArea()������һ����
//����С����ͨ��������Ȩ
//-----------------------------------------------------------------
void WeightBigSmall(const Mat &gray, Mat &obj, int i, int j, const int &max_p, const int &min_p) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int max1 = max_p;//��������ʼ��
    int max2 = min_p;//�����������
    int min1 = max_p - 1;//С������ʼ��
    int min2 = min_p + 1;//С���������

    int mindiff_b = MAX_PT;//����������ĵ����С��
    int mindiff_s = MAX_PT;//С��������ĵ����С��

    //�����򲿷ִ�����Ȩ������һ��+10��
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

    //С���򲿷ִ�����Ȩ������һ��+1��
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

    //�ж����ĵ㱾����8������������һ���֣�����Ȩ
    obj.at<float>(i, j) += mindiff_b < mindiff_s ? 10 : 1;


}


//----------------------[voteBigSmall()����]--------------------
//��С����ͨ�������ж�12.18
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


//----------------------[voteToFix()����]--------------------
//2018.1.11
//ͨ��ͶƱͼ�ҳ���С�ߣ�ì�ܵ㣬������
//����FixDiff()����,judeAndFixDiff()����
//-----------------------------------------------------------
Mat voteToFix(const Mat &src, const Mat &votemat, Mat &bigm, Mat &smallm, Mat &diffm, int th) {
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//�����OBJ�洢�޸���ĻҶ�ͼ
    diffm = obj.clone();
    bigm = obj.clone();
    smallm = obj.clone();
    int turn_to_diff = 10;
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //��1���ж��Ƿ�Ϊ�ڲ���
            if (votemat.at<float>(i, j) == 0) {//ͶƱֵΪ0,���жϣ�ֱ�Ӹ�ֵ
                obj.at<float>(i, j) = src.at<float>(i, j);
                continue;
            }
            //��2���ж���ì�ܵ㻹�Ǵ�С�����
            int min_tag, max_tag;
            min_tag = int(int(votemat.at<float>(i, j)) % turn_to_diff);
            max_tag = int(votemat.at<float>(i, j) / turn_to_diff);
            if (max_tag > min_tag) {

                //��3������ì�ܵ���ߴ�С��
                if (min_tag == OUTPUT_FALSE) {
                    //����ì�ܵ㣬������
                    bigm.at<float>(i, j) = OUTPUT_TRUE;
                    obj.at<float>(i, j) = src.at<float>(i, j);
                } else {
                    //��ì�ܵ㣬����ì�ܵ㣬���޸��Ҷ�ֵ
                    diffm.at<float>(i, j) = OUTPUT_TRUE;
                    //��4���޸��Ҷ�ֵ
                    FixDiff(src, votemat, obj, 0, i, j);//0��ʾ���
                }
            } else if (max_tag < min_tag) {
                if (max_tag == OUTPUT_FALSE) {
                    //����ì�ܵ㣬����С��
                    smallm.at<float>(i, j) = OUTPUT_TRUE;
                    obj.at<float>(i, j) = src.at<float>(i, j);
                } else {
                    //��ì�ܵ㣬����ì�ܵ㣬���޸��Ҷ�ֵ
                    diffm.at<float>(i, j) = OUTPUT_TRUE;
                    //��4���޸��Ҷ�ֵ
                    FixDiff(src, votemat, obj, 0, i, j);//1��ʾС��

                }
            } else {
                //����ì�ܵ�
                diffm.at<float>(i, j) = OUTPUT_TRUE;
                //��Ϊ��С����ͶƱ�����ͬ������Ҫ�ж����ĸ�����
                judeAndFixDiff(src, votemat, obj, i, j);

            }

        }
    }
    return obj;
}

//----------------------[FixDiff()����]--------------------
//2018.1.11
//voteToFix()���Ӻ���
//�޸���С����ĻҶ�ֵ
//������ԭͼ��ͶƱͼ���޸�ͼ������tag(0��1С��
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
    //�޸��Ҷ�ֵ
    if (cnt == 0) {
        //����ì�ܵ���ʱ�޺õĴ���ʽ
        obj.at<float>(i, j) = src.at<float>(i, j);
    } else {
        obj.at<float>(i, j) = int(sum / cnt);
    }

    return true;
}

//----------------------[judeAndFixDiff()����]--------------------
//2018.1.11
//voteToFix()���Ӻ���
//�жϲ��޸�ì�ܵ�
//---------------------------------------------------------------
bool judeAndFixDiff(const Mat &src, const Mat &votemat, Mat &obj, int i, int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int min_cl_diff_b, min_cl_diff_s;//��С��������ĵ����Сɫ��
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
            //�������
            sum_big += src.at<float>(i + di[k], j + dj[k]);
            cnt_big++;
            int temp_diff = fabs(src.at<float>(i + di[k], j + dj[k])
                                 - src.at<float>(i, j));
            if (temp_diff < min_cl_diff_b) {
                min_cl_diff_b = temp_diff;
            }
        } else if (judge_s > judge_b && judge_b == OUTPUT_FALSE) {
            //С�����
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
        //ì�ܵ�Ҷ�ֵ����Ϊ������
        obj.at<float>(i, j) = sum_big / cnt_big;
    } else if (min_cl_diff_b < min_cl_diff_s && cnt_small != 0) {
        //ì�ܵ�Ҷ�ֵ����ΪС����
        obj.at<float>(i, j) = sum_small / cnt_small;
    } else {
        //��Χ8���㶼��ì�ܵ�
        obj.at<float>(i, j) = src.at<float>(i, j);
    }
    return true;
}


//----------------------[gx_cvt_color()����]--------------------
//�Խ�����ɫת������
//���Ҷ���ɫת��Ϊĳ����ɫ0,b,1,g,2,r
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


//----------------------[color3_edge()����]---------------------
//ʹ��������ɫ���ִ�С��
//���Ҷ���ɫת��Ϊĳ����ɫ0,b,1,g,2,r
//tag ��ʾ������ Ĭ�� 3����ʾ������
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
                //��ɫ�����
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
            if (small_edge.at<float>(i, j) == 1) {
//                type3.at<double>(i, j)=1;
                obj.at<Vec3b>(i, j)[1] = MAX_PT;
                //��ɫ��С��
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
            }
            if (diff.at<float>(i, j) == 1 && tag == 3) {
//                type3.at<double>(i, j)=3;
                obj.at<Vec3b>(i, j)[0] = MAX_PT;
                //��ɫ,ì�ܵ�
                obj.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
            }
        }
    }

    return obj;

}


//----------------------[find_edgeda()����]---------------------
//ֻʹ��һ�����ҵ���С�����ڵ�����
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


//----------------------[find_edgexiao()����]---------------------
//ֻʹ��һ�����ҵ���С�����ڵ�����
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


//����ʹ�õĽṹ�壬����ͬʱ��¼��ֵ��λ��
typedef struct Test {
    int val;
    int pos;
} PIXEL;

bool Cmpare(const PIXEL &a, const PIXEL &b) {
    return a.val < b.val;
}


//----------------------[sortpixel()����]---------------------
//ֻʹ��һ�����ҵ���С�����ڵ�����,���а��������ĵ���޸�centerFix()����
//--------------------------------------------------------------
Mat sortpixel(const Mat &gray, int th) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//������򷨽��б�ǵĴ�С��
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //��Ŵ���֮���ͼ��Ҷ�ֵ
    Mat maodun(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//��¼ì�ܵ㣨���ҵı�ǵ�������Ĳ�һ�£�
    tu = gray.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //��������������
            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }

            //�ж��ǲ����ڲ��㣬����ǣ����������
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



            //��������������
            for (int k = 0; k <= 7; k++) {
                round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //������ҵ�����ֵ��λ��
            for (int k = 1; k <= 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }


            /*
			//�������ĵ�֮������ж��Ƿ��ǵ����ص�����ǹ��ɴ�
			int c[9];
			c[0] = gray.at<float>(i, j);
			for (int k = 1; k <= 8; k++)
			{
				c[k] = int(gray.at<float>(i + di[k], j + dj[k]));
			}
			sort(c, c + 9, Cmpare1);
			//�ж��Ƿ��ǵ����ص�
			if (premax == 2 || premax == 5)
			{

			}
			//�ж��Ƿ�Ϊ���ɴ�
			*/

            int sumda = 0;
            int sumxiao = 0;
            //������ھ�������ֵ����֮��ķ�Ϊ��С��֮��ĵ����obj���д�С�����
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



            //�ҵ���������߼е����ĵ�С�ߵ�
            for (int k = 0; k <= premax; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 1 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 1) {
                    //����������ì�ܵ�����ʱ��״��
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
            //�뱻����С�ߵ�е����ĵĴ�ߵ�
            for (int k = latemax; k <= 7; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 0 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 0) {
                    //����������ì�ܵ�����ʱ��״��
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



            //����Χ������֮�е�ì�ܵ�ȫ����ȥ֮�������ֵ
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



            //���ĵ��޸�
            //tu.at<float>(i, j) = centerFix(gray, maxcha, i, j);
        }
    }
    return tu;
}


//----------------------[testsortpixel()����]---------------------
//ֻʹ��һ�����ҵ���С�����ڵ�����,���а��������ĵ���޸�centerFix()����
//--------------------------------------------------------------
int testsortpixel(const Mat &gray, int th, int x, int y) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    int i = x;
    int j = y;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//������򷨽��б�ǵĴ�С��
    Mat maodun(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//��¼ì�ܵ㣨���ҵı�ǵ�������Ĳ�һ�£�
    //��������������
//    if (judge_single(gray, 20, i, j) == 1) {
//        return 0;
//    }

    //�ж��ǲ����ڲ��㣬����ǣ����������
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

    //��������������
    for (int k = 0; k <= 7; k++) {
        round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
    }

    cout << "����ǰ" << endl;
    for (int k = 0; k <= 7; k++) {
        cout << round[k].val << "  ";
    }


    sort(round, round + 8, Cmpare);
    //return round;


    cout << endl << "�����" << endl;
    for (int k = 0; k <= 7; k++) {
        cout << round[k].val << "  ";
    }


    int maxcha = 0, premax, latemax;
    //������ҵ�����ֵ��λ��
    for (int k = 1; k <= 7; k++) {
        if (round[k + 1].val - round[k].val >= maxcha) {
            maxcha = round[k + 1].val - round[k].val;
            premax = k;
            latemax = k + 1;
        }
    }

    //������ھ�������ֵ����֮��ķ�Ϊ��С��֮��ĵ����obj���д�С�����
    for (int k = 0; k <= premax; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
    }
    for (int k = latemax; k <= 7; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
    }


    cout << endl << "��ǽ��" << endl;
    for (int k = 0; k <= premax; k++) {
        cout << 0 << "  ";
    }
    for (int k = latemax; k <= 7; k++) {
        cout << 1 << "  ";
    }


    cout<<endl;


    //�ҵ���������߼е����ĵ�С�ߵ�
    for (int k = 0; k <= premax; k++) {
        if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 1 &&
            obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 1) {
            //����������ì�ܵ�����ʱ��״��
            if (obj.at<float>(obj.at<float>(i + di[(round[k].pos - 2 + 8) % 8], j + dj[(round[k].pos - 2 + 8) % 8])) ==
                0 || obj.at<float>(i + di[(round[k].pos + 2) % 8], j + dj[(round[k].pos + 2) % 8]) == 0) {
            } else {
                maodun.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
                return 1;
            }
        }
    }
    //�뱻����С�ߵ�е����ĵĴ�ߵ�
    for (int k = latemax; k <= 7; k++) {
        if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 0 &&
            obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 0) {
            //����������ì�ܵ�����ʱ��״��
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


//----------------------[old_sortpixel()����]---------------------
//ֻʹ��һ�����ҵ���С�����ڵ�����,���а��������ĵ���޸�centerFix()����,���������������򷽷������бȽϵ�
//--------------------------------------------------------------
Mat old_sortpixel(const Mat &gray, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//������򷨽��б�ǵĴ�С��
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//��Ű�������֮��Ľ��еĴ�С���ı��
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //��Ŵ���֮���ͼ��Ҷ�ֵ
    tu = gray.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {

            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }
            //��������������
            for (int k = 0; k < 8; k++) {
                round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //������ҵ�����ֵ��λ��
            for (int k = 0; k < 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }

            //������ھ�������ֵ����֮��ķ�Ϊ��С��֮��ĵ����obj���д�С�����
            for (int k = 0; k <= premax; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
            }
            for (int k = latemax; k <= 7; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
            }

            //���ĵ��޸�

            tu.at<float>(i, j) = centerFix(gray, maxcha, i, j);

            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
            //�������С����
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) -
                           gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //����������һ�����Ǹ���Сǰ��ĵ�
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //������ĵ�һ��������������ĵ�
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max)
                          ? fabs(positive_max) : fabs(negative_max);

            //		int dis_pow = fabs(positive_max) < fabs(negative_max)
            //				? fabs(negative_max) : fabs(positive_max);

            if (dis_pow < th) {//����������ֵ�ĵ�
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

            //���հ�����ķ�ʽ���д�С���Ļ������ǣ�ͬʱ����������ܺ͸���
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

            //������������������֮��Ĳ�ƥ���ֵ�����հ����������Ľ���Ĵ����������ľ�ֵ
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
            //���о�ֵ�޸�
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

//----------------------[old_testsortpixel()����]---------------------
//��������޸�չʾ���
//--------------------------------------------------------------
void old_testsortpixel(const Mat &gray, int th, int x, int y) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//������򷨽��б�ǵĴ�С��
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//��Ű�������֮��Ľ��еĴ�С���ı��
    Mat tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //��Ŵ���֮���ͼ��Ҷ�ֵ
    int i = x;
    int j = y;
    int negative_max, positive_max;
    negative_max = positive_max = OUTPUT_FALSE;
    int max_p, min_p;
    //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
    //�������С����
    for (int k = 1; k <= 8; ++k) {
        int diff = gray.at<float>(i + di[k], j + dj[k]) -
                   gray.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //����������һ�����Ǹ���Сǰ��ĵ�
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //������ĵ�һ��������������ĵ�
            }
        }
    }
    int dis_pow = fabs(positive_max) < fabs(negative_max)
                  ? fabs(positive_max) : fabs(negative_max);
    if (dis_pow < th) {//����������ֵ�ĵ�

        return;
    }

    cout << "ԭʼ��ֵ" << endl;
    //��������������
    for (int k = 0; k < 8; k++) {
        round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
        cout << round[k].val << " ";
    }
    cout << endl;
    sort(round, round + 8, Cmpare);

    cout << "����������" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].pos << " ";
    }
    cout << endl;
    cout << "��������ֵ" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].val << " ";
    }
    cout << endl;


    //return round;
    int maxcha = 0, premax, latemax;
    //������ҵ�����ֵ��λ��
    for (int k = 0; k < 7; k++) {
        if (round[k + 1].val - round[k].val >= maxcha) {
            maxcha = round[k + 1].val - round[k].val;
            premax = k;
            latemax = k + 1;
        }
    }

    //������ھ�������ֵ����֮��ķ�Ϊ��С��֮��ĵ����obj���д�С�����
    for (int k = 0; k <= premax; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
    }
    for (int k = latemax; k <= 7; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
    }

    cout << "�����ı��" << endl;
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

    //���հ�����ķ�ʽ���д�С���Ļ������ǣ�ͬʱ����������ܺ͸���
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

    cout << "������ı�ǽ��" << endl;
    for (int k = 0; k < 8; k++) {
        cout << obj2.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;

    //������������������֮��Ĳ�ƥ���ֵ�����հ����������Ľ���Ĵ����������ľ�ֵ
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
    //���о�ֵ�޸�
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

    cout << "�޸ĺ�Ľ��" << endl;
    for (int k = 0; k < 8; k++) {
        cout << tu.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;


    return;
}


//----------------------[newsortpixel()����]---------------------
//����Ч�����ã�������ֻʹ��һ�����ҵ���С�����ڵ�����,���а��������ĵ���޸�centerFix()����
//--------------------------------------------------------------
Mat newsortpixel(const Mat &gray, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat tu = gray;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//������򷨽��б�ǵĴ�С��
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//��Ű�������֮��Ľ��еĴ�С���ı��
//	Mat  tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //��Ŵ���֮���ͼ��Ҷ�ֵ
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //��������������
            for (int k = 0; k < 8; k++) {
                round[k].val = int(tu.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //������ҵ�����ֵ��λ��
            for (int k = 0; k < 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }

            //������ھ�������ֵ����֮��ķ�Ϊ��С��֮��ĵ����obj���д�С�����
            for (int k = 0; k <= premax; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
            }
            for (int k = latemax; k <= 7; k++) {
                obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
            }

            //���ĵ��޸�

            tu.at<float>(i, j) = centerFix(tu, maxcha, i, j);

            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p, min_p;
            //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
            //�������С����
            for (int k = 1; k <= 8; ++k) {
                int diff = tu.at<float>(i + di[k], j + dj[k]) -
                           tu.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //����������һ�����Ǹ���Сǰ��ĵ�
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //������ĵ�һ��������������ĵ�
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max)
                          ? fabs(positive_max) : fabs(negative_max);

            //		int dis_pow = fabs(positive_max) < fabs(negative_max)
            //				? fabs(negative_max) : fabs(positive_max);

            if (dis_pow < th) {//����������ֵ�ĵ�
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

            //���հ�����ķ�ʽ���д�С���Ļ������ǣ�ͬʱ����������ܺ͸���
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

            //������������������֮��Ĳ�ƥ���ֵ�����հ����������Ľ���Ĵ����������ľ�ֵ
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
            //���о�ֵ�޸�
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


//----------------------[newtestsortpixel()����]---------------------
//����Ч�����ã�����������������޸�չʾ���
//--------------------------------------------------------------
void newtestsortpixel(const Mat &gray, int th, int x, int y) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    Mat tu = gray;
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//������򷨽��б�ǵĴ�С��
    Mat obj2(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//��Ű�������֮��Ľ��еĴ�С���ı��
    //Mat  tu(rows, cols, CV_32F, Scalar(OUTPUT_FALSE)); //��Ŵ���֮���ͼ��Ҷ�ֵ
    int i = x;
    int j = y;
    int negative_max, positive_max;
    negative_max = positive_max = OUTPUT_FALSE;
    int max_p, min_p;
    //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
    //�������С����
    for (int k = 1; k <= 8; ++k) {
        int diff = tu.at<float>(i + di[k], j + dj[k]) -
                   tu.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //����������һ�����Ǹ���Сǰ��ĵ�
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //������ĵ�һ��������������ĵ�
            }
        }
    }
    int dis_pow = fabs(positive_max) < fabs(negative_max)
                  ? fabs(positive_max) : fabs(negative_max);
    if (dis_pow < th) {//����������ֵ�ĵ�

        return;
    }

    cout << "ԭʼ��ֵ" << endl;
    //��������������
    for (int k = 0; k < 8; k++) {
        round[k].val = int(tu.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
        cout << round[k].val << " ";
    }
    cout << endl;
    sort(round, round + 8, Cmpare);

    cout << "����������" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].pos << " ";
    }
    cout << endl;
    cout << "��������ֵ" << endl;
    for (int k = 0; k < 8; k++) {
        cout << round[k].val << " ";
    }
    cout << endl;


    //return round;
    int maxcha = 0, premax, latemax;
    //������ҵ�����ֵ��λ��
    for (int k = 0; k < 7; k++) {
        if (round[k + 1].val - round[k].val >= maxcha) {
            maxcha = round[k + 1].val - round[k].val;
            premax = k;
            latemax = k + 1;
        }
    }

    //������ھ�������ֵ����֮��ķ�Ϊ��С��֮��ĵ����obj���д�С�����
    for (int k = 0; k <= premax; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 0;
    }
    for (int k = latemax; k <= 7; k++) {
        obj.at<float>(i + di[round[k].pos], j + dj[round[k].pos]) = 1;
    }

    cout << "�����ı��" << endl;
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

    //���հ�����ķ�ʽ���д�С���Ļ������ǣ�ͬʱ����������ܺ͸���
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

    cout << "������ı�ǽ��" << endl;
    for (int k = 0; k < 8; k++) {
        cout << obj2.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;

    //������������������֮��Ĳ�ƥ���ֵ�����հ����������Ľ���Ĵ����������ľ�ֵ
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
    //���о�ֵ�޸�
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

    cout << "�޸ĺ�Ľ��" << endl;
    for (int k = 0; k < 8; k++) {
        cout << tu.at<float>(i + di[k], j + dj[k]) << " ";
    }
    cout << endl;


    return;
}


//----------------------[mfixDiff()����]--------------------
//�ҵ�fixDiff����
//
//--------------------------------------------------------
Mat mfixDiff(const Mat &src, Mat &votemat, Mat &bigm, Mat &smallm, Mat &diffm, int th) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = src.rows;
    int cols = src.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//�����OBJ�洢�޸���ĻҶ�ͼ
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

            if (votemat.at<float>(i, j) == 0) {//ͶƱֵΪ0,���жϣ�ֱ�Ӹ�ֵ
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

//----------------------[centerFix()����]--------------------
//���ĵ�ȥ����
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


//----------------------[mfindBigSmallArea()����]--------------------
//���ִ�С����12.18
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
            //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
            //�������С����
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff < 0) {
                    if (diff < negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //ǰ���Ǵ�����
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //�����Ǵ�����
                    }
                }
            }
            int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
            if (dis_pow < th) {//����������ֵ�ĵ�
                continue;
            }

            //int min = 999;
            //int locate = -1;
            int max1 = max_p % 8;    //�����Ǵ�����
            int max2 = min_p;    //ǰ���Ǵ�����
            int min1 = (min_p + 1) % 8;  //������С����
            int min2 = max_p - 1;   //ǰ����С����


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


//----------------------[mfindBigSmallArea()����]--------------------
//���ִ�С����12.18
//
//-----------------------------------------------------------------
void test_mfindBigSmallArea(const Mat &gray, int i,int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

            int negative_max, positive_max;
            negative_max = positive_max = OUTPUT_FALSE;
            int max_p=0, min_p=0;
            //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
            //�������С����
            for (int k = 1; k <= 8; ++k) {
                int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
                if (diff <= 0) {
                    if (diff <= negative_max) {
                        negative_max = diff;
                        min_p = k - 1;
                        //ǰ���Ǵ�����
                    }
                } else {
                    if (diff >= positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //�����Ǵ�����
                    }
                }
            }


            //int min = 999;
            //int locate = -1;
            int max1 = max_p % 8;    //�����Ǵ�����
            int max2 = min_p;    //ǰ���Ǵ�����
            int min1 = (min_p + 1) % 8;  //������С����
            int min2 = max_p - 1;   //ǰ����С����

            cout<<"������: "<<endl;
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

            cout<<"С����: "<<endl;
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

//----------------------[color2_vote()����]---------------------
//ʹ��������ɫ����ͶƱ��С��
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
                //��ɫ�����
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[1] = OUTPUT_FALSE;
            }
            if (vote.at<float>(i, j) == 1) {
                obj.at<Vec3b>(i, j)[1] = MAX_PT;
                //��ɫ��С��
                obj.at<Vec3b>(i, j)[0] = OUTPUT_FALSE;
                obj.at<Vec3b>(i, j)[2] = OUTPUT_FALSE;
            }
        }
    }

    return obj;

}


//----------------------[fix7vs1()����]--------------------
////7��1����µ�����
//
//-----------------------------------------------------------------
Mat fix7vs1(Mat &gray) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};

    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));
    Mat tag(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));   //   ������б仯�ĵ�
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
                        //ǰ���Ǵ�����
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //�����Ǵ�����
                    }
                }
            }
            int max1 = max_p % 8;    //�����Ǵ�����
            int max2 = min_p;    //ǰ���Ǵ�����
            int min1 = (min_p + 1) % 8;  //������С����
            int min2 = max_p - 1;   //ǰ����С����

            if (max1 == max2)//7��1����µ�����
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
            } else if (min1 == min2)//7��1����µ�����
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


//----------------------[testfix7vs1()����]--------------------
////test7��1����µ�����
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
    //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
    //�������С����
    for (int k = 1; k <= 8; ++k) {
        int diff = gray.at<float>(i + di[k], j + dj[k]) - gray.at<float>(i + di[k - 1], j + dj[k - 1]);
        if (diff < 0) {
            if (diff < negative_max) {
                negative_max = diff;
                min_p = k - 1;
                //ǰ���Ǵ�����
                cout << gray.at<float>(i + di[k], j + dj[k]) << "-" << gray.at<float>(i + di[k - 1], j + dj[k - 1])
                     << "min:" << diff << endl;
            }
        } else {
            if (diff > positive_max) {
                positive_max = diff;
                max_p = k;
                //�����Ǵ�����
            }
        }
    }
    int max1 = max_p % 8;    //�����Ǵ�����
    int max2 = min_p;    //ǰ���Ǵ�����
    int min1 = (min_p + 1) % 8;  //������С����
    int min2 = max_p - 1;   //ǰ����С����

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

    if (max1 == max2)//7��1����µ�����
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

    } else if (min1 == min2)//7��1����µ�����
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

//----------------------[sort7vs1()����]---------------------
//sort7vs1��������Բ����룬�ڼ���ͼ���У�������������Ե����������ͼ����ȫ�޸�
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
            //��������������

            for (int k = 0; k < 8; k++) {
                round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //������ҵ�����ֵ��λ��
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


//----------------------[fixEdge����]---------------------
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


//----------------------[find_edgeboth()����]---------------------
//��һ��ͼ����ʾ��С��
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

//----------------------[find_edgeboth2()����]---------------------
//��һ��ͼ����ʾ��С��
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
//----------------------[drawline()����]---------------------
//����ֱ������
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


//----------------------[tagdaxiao()����]---------------------
//��Ǵ�С��
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


//----------------------[enhance()����]---------------------
//��ǿ
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
    cout << "��" << da << endl;
    cout << "С" << xiao << endl;

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


//----------------------[Mapping()����]---------------------
//Mappingͨ��ӳ��ʹ��Ե���񻯣�ʹ�ڲ�����ƽ��
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

//----------------------[Histogram()����]---------------------
//Histogram������ʾֱ��ͼ
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


//----------------------[Histogramdaxiao()����]---------------------
//Histogramdaxiao������ʾֱ��ͼ
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


//----------------------[Histogram1()����]---------------------
//Histogram1������ʾֱ��ͼ
//--------------------------------------------------------------
int Histogram1() {
    Mat src, gray, hist;                //histΪ�洢ֱ��ͼ�ľ���
    src = imread("C://Users//mrmjy//Desktop//4.jpg");
    cvtColor(src, gray, CV_BGR2GRAY);   //ת��Ϊ�Ҷ�ͼ

    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    int channels[] = {0};
    bool uniform = true;
    bool accumulate = false;

    /*����ֱ��ͼ*/
    calcHist(&gray, 1, channels, Mat(), hist, 1, &histSize,
             &histRange, uniform, accumulate);

    /*�������ֱ��ͼ�ġ�ͼ�񡱣���ԭͼ��Сһ��*/
    int hist_w = src.cols;
    int hist_h = src.rows;
    int bin_w = cvRound((double) hist_w / histSize);

    Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

    /*ֱ��ͼ��һ����Χ[0��histImage.rows]*/
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /*��ֱ��*/
    for (int i = 1; i < histSize; ++i) {
        //cvRound������ת���� ����histΪ256*1��һά���󣬴洢����ͼ���и����Ҷȼ��Ĺ�һ��ֵ
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow("figure_src", src);
    imshow("figure_hist", histImage);

    waitKey(0);
    return 0;
}


//----------------------[histm()����]---------------------
//histm()����ֱ��ͼ���⻯����
//--------------------------------------------------------------
int histm() {
    Mat srcImage = imread("C://Users//mrmjy//Desktop//5.jpg");
    if (!srcImage.data) {
        printf("ͼƬ����ʧ��!\n");
        return -1;
    }

    //����Ҷ�ͼ��
    Mat gray;
    cvtColor(srcImage, gray, COLOR_RGB2GRAY);

    namedWindow("ԭͼ");
    imshow("ԭͼ", gray);

    //��ʼֱ��ͼ��������
    Mat out;
    equalizeHist(gray, out);
    src_to_bmp(out, "C://Users//mrmjy//Desktop//out.jpg");
    namedWindow("����ֱ��ͼ��������");
    imshow("����ֱ��ͼ��������", out);
    waitKey();

    return 0;
}


//----------------------[grow()����]---------------------
//grow()����ֱ��ͼ���⻯����
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

//----------------------[color_findBigSmallAre()����]---------------------
//color_findBigSmallAre()�Բ�ɫͼ�񻮷ִ�С����
//--------------------------------------------------------------
int color_diff(CvScalar a, CvScalar b) {
    int num = static_cast<int>(abs(a.val[0] - b.val[0]) + abs(a.val[1] - b.val[1]) + abs(a.val[2] - b.val[2]));
    return num;
}

//----------------------[light()����]---------------------
//light()�Բ�ɫͼ�񻮷ִ�С����
//--------------------------------------------------------------
float light(CvScalar a) {
    float num = static_cast<float>(a.val[0] * 0.1 + a.val[1] * 0.6 + a.val[2] * 0.3);
    return num;
}


//----------------------[color_findBigSmallAre()����]---------------------
//color_findBigSmallAre()�Բ�ɫͼ�񻮷ִ�С����
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

            //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
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
						//�����Ǵ�����
					}
					if (diff > negative_max && diff <positive_max)
					{
						negative_max = diff;
						min_p = k;
					}
			}
		//	int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
			if (positive_max <  th)
			{//����������ֵ�ĵ�
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
                        //ǰ���Ǵ�����
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //�����Ǵ�����
                    }
                }
            }

            int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
            if (dis_pow < th) {//����������ֵ�ĵ�
                continue;
            }
            int max1 = max_p % 8;    //�����Ǵ�����
            int max2 = min_p;    //ǰ���Ǵ�����
            int min1 = (min_p + 1) % 8;  //������С����
            int min2 = max_p - 1;   //ǰ����С����

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


//----------------------[oldcolor_findBigSmallAre()����]---------------------
//oldcolor_findBigSmallAre()�Բ�ɫͼ�񻮷ִ�С����
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
            //max_p�Ǵ�����ĵ�һ���㣬min_p�Ǵ��������һ�����λ��
            //�������С����
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
                    //�����Ǵ�����
                }
                if (diff > negative_max && diff < positive_max) {
                    negative_max = diff;
                    min_p = k - 1;
                }
            }

            //	int dis_pow = fabs(positive_max) < fabs(negative_max) ? fabs(positive_max) : fabs(negative_max);
            if (positive_max < th) {//����������ֵ�ĵ�
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
			max2 = max_p % 8;  	//�����Ǵ�����
			max1 = (max_p -1+8) % 8;  	//ǰ���Ǵ�����
			min1 = (min_p -1+8) % 8;  //������С����
			min2 = min_p  % 8;   //ǰ����С����
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

//----------------------[mcolor_fixDiff()����]--------------------
//�ҵ�mcolor_fixDiff����
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
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//�����OBJ�洢�޸���ĻҶ�ͼ
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
                //ͶƱֵΪ0,���жϣ�ֱ�Ӹ�ֵ
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


//----------------------[judge_single()����]---------------------
//�ж��Ƿ��ǵ���������
//--------------------------------------------------------------
int judge_single(const Mat &gray, int th, int i, int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    //��������������
    for (int k = 0; k < 8; k++) {
        round[k].val = int(gray.at<float>(i + di[k], j + dj[k]));
        round[k].pos = k;
    }
    sort(round, round + 8, Cmpare);
    //return round;
    int maxcha = 0, premax, latemax;
    //������ҵ�����ֵ��λ��
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


//----------------------[judge_transition()����]---------------------
//�ж��Ƿ��ǹ�����������
//--------------------------------------------------------------
int judge_transition(const Mat &gray, int th, int i, int j) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[9];
    //��������������
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

//----------------------[color_fix7vs1()����]--------------------
////��ͼ7��1����µ�����
//
//-----------------------------------------------------------------
Mat color_fix7vs1(Mat &src, IplImage *a, Mat &gray) {
    int di[8 + 1] = {+0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 1] = {+1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    Mat obj(rows, cols, CV_8UC3, Scalar(255, 255, 255));
    Mat tag(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));   //   ������б仯�ĵ�
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
                        //ǰ���Ǵ�����
                    }
                } else {
                    if (diff > positive_max) {
                        positive_max = diff;
                        max_p = k;
                        //�����Ǵ�����
                    }
                }
            }
            int max1 = max_p % 8;    //�����Ǵ�����
            int max2 = min_p;    //ǰ���Ǵ�����
            int min1 = (min_p + 1) % 8;  //������С����
            int min2 = max_p - 1;   //ǰ����С����

            if (max1 == max2)//7��1����µ�����
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
            } else if (min1 == min2)//7��1����µ�����
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


//----------------------[color_sortpixel()����]--------------------
////��ͼ���㷨����µ�����
//
//-----------------------------------------------------------------
Mat color_sortpixel(const Mat &gray, Mat &color, IplImage *a, int th) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int rows = gray.rows;
    int cols = gray.cols;
    PIXEL round[8];
    Mat obj(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//������򷨽��б�ǵĴ�С��
    Mat tu(rows, cols, CV_8UC3, Scalar(OUTPUT_FALSE)); //��Ŵ���֮��Ĳ�ɫͼ��
    Mat maodun(rows, cols, CV_32F, Scalar(OUTPUT_FALSE));//��¼ì�ܵ㣨���ҵı�ǵ�������Ĳ�һ�£�
    tu = color.clone();
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            //��������������
            if (judge_single(gray, 20, i, j) == 1) {
                continue;
            }

            //�ж��ǲ����ڲ��㣬����ǣ����������
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

            //��������������
            for (int k = 0; k <= 7; k++) {
                round[k].val = light(cvGet2D(a, i + di[k], j + dj[k]));
                round[k].pos = k;
            }
            sort(round, round + 8, Cmpare);
            //return round;
            int maxcha = 0, premax, latemax;
            //������ҵ�����ֵ��λ��
            for (int k = 1; k <= 7; k++) {
                if (round[k + 1].val - round[k].val >= maxcha) {
                    maxcha = round[k + 1].val - round[k].val;
                    premax = k;
                    latemax = k + 1;
                }
            }

            /*
			//�������ĵ�֮������ж��Ƿ��ǵ����ص�����ǹ��ɴ�
			int c[9];
			c[0] = gray.at<float>(i, j);
			for (int k = 1; k <= 8; k++)
			{
			c[k] = int(gray.at<float>(i + di[k], j + dj[k]));
			}
			sort(c, c + 9, Cmpare1);
			//�ж��Ƿ��ǵ����ص�
			if (premax == 2 || premax == 5)
			{

			}
			//�ж��Ƿ�Ϊ���ɴ�
			*/

            int sumda = 0;
            int sumxiao = 0;

            int sumdab = 0;
            int sumdag = 0;
            int sumdar = 0;

            int sumxiaob = 0;
            int sumxiaog = 0;
            int sumxiaor = 0;

            //������ھ�������ֵ����֮��ķ�Ϊ��С��֮��ĵ����obj���д�С�����
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

            //�ҵ���������߼е����ĵ�С�ߵ�
            for (int k = 0; k <= premax; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 1 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 1) {
                    //����������ì�ܵ�����ʱ��״��
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
            //�뱻����С�ߵ�е����ĵĴ�ߵ�
            for (int k = latemax; k <= 7; k++) {
                if (obj.at<float>(i + di[(round[k].pos - 1 + 8) % 8], j + dj[(round[k].pos - 1 + 8) % 8]) == 0 &&
                    obj.at<float>(i + di[(round[k].pos + 1) % 8], j + dj[(round[k].pos + 1) % 8]) == 0) {
                    //����������ì�ܵ�����ʱ��״��
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

            //����Χ������֮�е�ì�ܵ�ȫ����ȥ֮�������ֵ
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

            //���ĵ��޸�
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

//----------------------[Near_attribution()����]--------------------
//
//�����ڽӵĴ�С����а������ڵ����е�Ĺ�������
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


//----------------------[verify()����]--------------------
//
// �ж��������ͼ�ǲ��������ķ���
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

//----------------------[verify()����]--------------------
//
// �ж�������ڲ�������ֹͣ����
//1.   �����ڵ�ɫ���С
//2.  �����е����ص�ֲ�û�й��ɣ��Ƚϻ���
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
//----------------------[New_Near_attribution()����]--------------------
//
// ÿ�εõ���ʼ�㣬�Ϳ�ʼ���������ܸı�ĵ�����
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


//----------------------[L0g()����]--------------------
//
//  ����L0g���ӵļ�����и�д
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

//----------------------[type3()����]--------------------
//
//  type3�Ѵ�Сì�ܵ�ֱ���м�¼
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

//----------------------[smooth_gary()����]--------------------
//
//  smooth_gray�Ѵ�Сì�ܵ�ֱ���м�¼
//  obj �ǽ��б�ǵ��������� 0�����ڲ���,1����С�ߵ�,2�����ߵ�,3����ì�ܵ�
//  sum ������Ҷ�ֵ���ܺ�
//  nums ���������ص���ܸ���
//  gray �洢�����ĸ��������ص��ƽ��ֵ
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

//----------------------[smooth_color()����]--------------------
//
//  smooth_color�Ѵ�Сì�ܵ�ֱ���м�¼
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

//----------------------[extend_gray()����]--------------------
//
//  extend_gray�Ѵ�Сì�ܵ�ֱ���м�¼
//  obj ������ķֲ�
//  gray ������ֵ�����ۻҶ�ֵ
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


//   Ѱ������ֵ��С����
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

            int cha1=0;// ��һ���ֲ�ֵ
            int cha2=0;// �ڶ����ֲ�ֵ

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
                     cout << "����������ƽ���Ҷ�ֵ: " << sum / 8 << endl<<endl;
                     cout << "��һ��������ֵ: " << cha1 << endl;
                     cout << "��һ�������ֵ: " << max1 << endl;
                     cout << "��һ������Сֵ: " << min1 << endl;

                     cout << endl;

//                     for(int p=0;p<8-tag;p++){
//                         cout<<q_cha[p]<<" ";
//                     }
//
//                     cout<<endl;
                     cout << "�ڶ���������ֵ: " << cha2 << endl;
                     cout << "�ڶ��������ֵ: " << max2 << endl;
                     cout << "�ڶ�������Сֵ: " << min2 << endl;
//                     cout << "���������ƽ��ֵ��ֵ: " << abs(avg_m - avg_n) << endl;

                     cout<<endl;
                     cout<<"���򷨵Ľ��"<< endl;
                     testsortpixel(gray, 1, i, j);
                     cout<<endl;
                     cout<<"������������Ľ��"<< endl;
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

//  �ҵ���С����ֵ֮�󻮷�Ϊ��������,�ֱ���о�ֵ����
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


//  �ҵ���С����ֵ֮�󻮷�Ϊ��������,�ֱ���о�ֵ����
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


//  ���Ժ���ֻ��Ϊ��������
void test_mc(Mat &re, Mat &gray, int pos[], int i, int j, int avg_m) {
    int di[8 + 2] = {+1, +0, -1, -1, -1, 0, +1, +1, +1, +0};
    int dj[8 + 2] = {+1, +1, +1, 0, -1, -1, -1, +0, +1, +1};
    int sum = 0;
    int num = 0;
    int avg = 0;

    cout << "ԭʼ���򲿷ֵ����ص���� :  " << endl;
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
        cout<<endl<<"�������ص���ͬ,ͬһ����"<< endl;
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

    cout << endl << "��һ���ֵľ�ֵ:  " << avg;
    cout << endl << "ʣ�²��ֵľ�ֵ:  " << avg_m << endl;

    cout << "������Ĳ�ֵ:  " << abs(avg_m - avg) << endl;
    cout << endl;

}

//  �ж��Ƿ��Ǳ�Ե��
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

//  �µ��ж��Ƿ��Ǳ�Ե��
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

// �����������С��ֵ
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

//   ������Ե��,��Ե����ʾ��ɫ
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
