#pragma once
#ifndef BASIC_GX_H_
#define BASIC_GX_H_
//basic_gx的头文件,依赖少的一些操作
//----------------------[头文件引用区域]----------------------

#include <iostream>
#include <bitset>
#include <vector>
#include <string>
//#include <io.h>
//#include <direct.h>
#include <cstring>
#include <sstream> //num_to_string()中ostringstream需要
#include <iomanip>//格式化输出
#include <cmath>
using namespace std;

//----------------------[名称空间引用区域]----------------------

using std::bitset;
using std::string;
using std::vector;
using std::pair;
using std::make_pair;
//----------------------[全局变量声明区域]----------------------
typedef bitset<8> bit8;
typedef pair<int, int> int_pr;//对文件
typedef pair<int_pr, int> int_pr_pr;//嵌套使用对文件
const int GRAY_POINT_MAX_VALUE = 256;
const int INSIDE_WHITE = 255;//白色
const int EDGE_BLACK = 0;//黑色

//----------------------[函数声明区域]---------------------------


//---------------------[普通函数声明区域]--------------------------
bit8 min_bit8(const bit8 & bt8);//求8位数中的最小值

void Catch_fileName(string, vector<string> & files);//获取文件名
													//按格式获取文件名
void GetFilesName(string, string, vector<string>&);//递归查找目录下文件名

void create_dir(string);//递归创建文件夹



//---------------------[模版函数声明区域]--------------------------


//----------------------[num_to_string()函数]---------------------
//将数字转为字符串
//---------------------------------------------------------------
template<class T>//模版函数，数字转化为字符串
string num_to_string(const T & t)
{
	using namespace std;
	ostringstream buf;
	buf << t;
	return buf.str();
}



//模版函数的二分搜索（迭代版）
//---------------------[binary_cut_search()函数]--------------------
//模版函数的二分搜索（迭代版），按值查找
//三个参数：模版数组，int型元素个数，模版类型要查找的元素值
//-----------------------------------------------------------------
template <class T>//模版函数的二分搜索（迭代版）
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

//---------------------[max_arr_num()函数]--------------------
//模版函数，求数组中最大值
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

//---------------------[min_arr_num()函数]--------------------
//模版函数，求数组中最小值
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

//---------------------[sum_arr_num()函数]--------------------
//求数组元素之和
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


//---------------------[zoom_arr()函数]--------------------
//按比例缩放数组元素
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


//---------------------[abs_min_num()函数]--------------------
//abs_min_num()判断三个数中第一个数和前两个数的差值的最小值，或者两个数的最小值
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