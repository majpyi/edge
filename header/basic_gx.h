#pragma once
#ifndef BASIC_GX_H_
#define BASIC_GX_H_
//basic_gx��ͷ�ļ�,�����ٵ�һЩ����
//----------------------[ͷ�ļ���������]----------------------

#include <iostream>
#include <bitset>
#include <vector>
#include <string>
//#include <io.h>
//#include <direct.h>
#include <cstring>
#include <sstream> //num_to_string()��ostringstream��Ҫ
#include <iomanip>//��ʽ�����
#include <cmath>
using namespace std;

//----------------------[���ƿռ���������]----------------------

using std::bitset;
using std::string;
using std::vector;
using std::pair;
using std::make_pair;
//----------------------[ȫ�ֱ�����������]----------------------
typedef bitset<8> bit8;
typedef pair<int, int> int_pr;//���ļ�
typedef pair<int_pr, int> int_pr_pr;//Ƕ��ʹ�ö��ļ�
const int GRAY_POINT_MAX_VALUE = 256;
const int INSIDE_WHITE = 255;//��ɫ
const int EDGE_BLACK = 0;//��ɫ

//----------------------[������������]---------------------------


//---------------------[��ͨ������������]--------------------------
bit8 min_bit8(const bit8 & bt8);//��8λ���е���Сֵ

void Catch_fileName(string, vector<string> & files);//��ȡ�ļ���
													//����ʽ��ȡ�ļ���
void GetFilesName(string, string, vector<string>&);//�ݹ����Ŀ¼���ļ���

void create_dir(string);//�ݹ鴴���ļ���



//---------------------[ģ�溯����������]--------------------------


//----------------------[num_to_string()����]---------------------
//������תΪ�ַ���
//---------------------------------------------------------------
template<class T>//ģ�溯��������ת��Ϊ�ַ���
string num_to_string(const T & t)
{
	using namespace std;
	ostringstream buf;
	buf << t;
	return buf.str();
}



//ģ�溯���Ķ��������������棩
//---------------------[binary_cut_search()����]--------------------
//ģ�溯���Ķ��������������棩����ֵ����
//����������ģ�����飬int��Ԫ�ظ�����ģ������Ҫ���ҵ�Ԫ��ֵ
//-----------------------------------------------------------------
template <class T>//ģ�溯���Ķ��������������棩
int binary_cut_search(const T * a, int n, const T & x)
{
	int left = 0;
	int right = n - 1;

	while (left <= right)
	{
		int mid = (left + right) / 2;
		if (x > a[mid])
		{
			left = mid + 1;
		}
		else if (x < a[mid])
		{
			right = mid - 1;
		}
		else
		{
			return mid;
		}
	}
	return -1;
}

//---------------------[max_arr_num()����]--------------------
//ģ�溯���������������ֵ
//-----------------------------------------------------------------
template <class T>
T max_arr_num(const T *a, int n)
{
	T max_num = 0;

	for (int i = 0; i < n; ++i)
	{
		if (a[i] > max_num)
		{
			max_num = a[i];
		}
	}
	return max_num;
}

//---------------------[min_arr_num()����]--------------------
//ģ�溯��������������Сֵ
//-----------------------------------------------------------------
template <class T>
T min_arr(const T *a, int n)
{
	T min_num = 0;

	for (int i = 0; i < n; ++i)
	{
		if (a[i] > min_num)
		{
			min_num = a[i];
		}
	}
	return min_num;
}

//---------------------[sum_arr_num()����]--------------------
//������Ԫ��֮��
//-----------------------------------------------------------------
template <class T>
T sum_arr_num(const T *a, int n)
{
	T sum_num = a[0];

	for (int i = 1; i < n; ++i)
	{
			sum_num += a[i];
	}
	return sum_num;
}


//---------------------[zoom_arr()����]--------------------
//��������������Ԫ��
//--------------------------------------------------------
template <class T>
T * zoom_arr(const T * arr, int size, int scale)
{

	T *arr_c = new T[size];
	if (scale > 0)
	{
		for (int i = 0; i < size; ++i)
		{
			arr_c[i] = int(arr[i] * scale);
		}
	}
	else if (scale < 0)
	{
		for (int i = 0; i < size; ++i)
		{
			arr_c[i] = int(arr[i] / abs(scale));
		}
	}
	else
	{
		std::cerr << "Error Scale !" << endl;
	}
	return arr_c;
}


//---------------------[abs_min_num()����]--------------------
//abs_min_num()�ж��������е�һ������ǰ�������Ĳ�ֵ����Сֵ����������������Сֵ
//-----------------------------------------------------------
template <typename T1,typename T2>
auto abs_min_num(T1 min_org, T2 num_a, T2 num_b = 0 )  -> decltype(min_org*num_a)
{
	decltype (min_org*num_a) temp;
	temp = fabs(num_a - num_b);
	temp = (temp < min_org ? temp : min_org);
	return temp;
}

#endif // !BASIC_GX_H_
//By Ghostxiu. 2018/1/11