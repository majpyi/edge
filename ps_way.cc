//实现简化一些常用的OPENCV方法，从而避免调用错误
#include "use.h"
//注意引用头文件时，使用的文件地址是相对于源文件，而读和写文件是相对于工程
//----------------------[main()函数]----------------------
//程序入口
//-------------------------------------------------------
#if(0)//关闭main函数，仅用于测试
int main(int argc, char ** argv)
{
	//读取原图
	read_src(g_src, "lena.jpg");
	//显示原图
	show_src(g_src, "Origin pic");

	//转化原图为灰度图
	to_gray(g_src, g_srcGray);

	//显示灰度图
	show_src(g_srcGray, "Gray pic");
	//储存灰度图文件
	src_to_bmp(g_srcGray, "output//lena_gray.bmp");

	//将图像以数组形式写入txt文件

	//output_arr_txt(g_srcGray, "output//g_srcGray.txt");


	waitKey(0); // 等待一次按键，程序结束
	return 0;
}
#endif

//----------------------[to_gray()函数]---------------------
//图像转灰度图
//----------------------------------------------------------
bool to_gray(const Mat & src, Mat & src_Gray)
{
	if (src.empty())//读取失败时
	{
		cout << "Could not open or find the image" << endl;
		return false;
	}
	cvtColor(src, src_Gray, CV_RGB2GRAY);//把图片转化为灰度图
	return true;
}


//----------------------[read_src()函数]----------------------
//图像读取函数
//三个参数，引用的Mat型，读取常量String型文件名，p是通道数，默认0为不处理通道
//将图像文件读取进入Mat型，并转化为32位浮点型
//可选作为单通道图像处理 p = 1 
//------------------------------------------------------------
bool read_src(Mat & src, const string & imgName ,int p)
{
	if (p == 0)
	{//常规读取方式
		src = imread(imgName.c_str(), IMREAD_COLOR);
	}
	else if (p == 1)
	{//按单通道读取
		src = imread(imgName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	}
	if (p == 1)
	{
		src.convertTo(src, CV_32F);
	}
	//图像数值匹配统一定义为32位浮点类型
	if (src.empty())//读取失败时
	{
		cout << "Could not open or find the image" << std::endl;
		return false;
	}
	return true;
}

//----------------------[src_to_bmp()函数]----------------------
//将Mat缓存中的图片信息写入到bmp
//-------------------------------------------------------------
bool src_to_bmp(const Mat & src, const string & imgName)
{
	if (src.empty())//读取失败时
	{
		cout << "Could not open or find the image" << endl;
		return false;
	}
	imwrite(imgName, src);
	return true;
}

//----------------------[show_src()函数]----------------------
//图像显示函数
//------------------------------------------------------------
bool show_src(const Mat & src, const string & imgName)
{
	if (src.empty())//读取失败时
	{
		cout << "Could not open or find the image" << endl;
		return false;
	}
	namedWindow(imgName);
	imshow(imgName, src);
	return true;
}


//--------------------[output_arr_txt()函数]--------------------
//图像数组写入txt文件
//----------------------------------------------------------------
void output_arr_txt(const Mat &src, const string & txt_name)
{
	
	if (src.empty())//读取失败时
	{
		cout << "Could not open or find the image!" << endl;
		return;
	}
	
	ofstream fout;
	fout.open(txt_name);
	if (!fout)
	{
		cout << "File Could Not Open!" << endl;
		return;
	}
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			fout << src.at<float>(i, j) << " \t";
		}
		fout << endl;

	}
	cout << "Picture success written in txt:"
		<< txt_name << endl;
	fout.close();
}


//----------------------[read_Graysrc()函数]----------------------
//直接将图像读取为灰度图
//------------------------------------------------------------
bool read_Graysrc(Mat & gray_src, const string & imgName)
{
	gray_src = imread(imgName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	if (gray_src.empty())//读取失败时
	{
		cout << "Could not open or find the image" << std::endl;
		return false;
	}
	return true;
}



//--------------------[output_arr_csv()函数]--------------------
//图像数组写入excel文件
//----------------------------------------------------------------
void output_arr_csv(const Mat &src, const string & txt_name)
{

	if (src.empty())//读取失败时
	{
		cout << "Could not open or find the image!" << endl;
		return;
	}

	ofstream fout;
	fout.open(txt_name );
	if (!fout)
	{
		cout << "File Could Not Open!" << endl;
		return;
	}
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			fout << src.at<float>(i, j);
			if (j != src.cols - 1)
			{
				fout << ",";
			}
				
		}
		fout << endl;

	}
	cout << "Picture success written in txt:"
		<< txt_name << endl;
	fout.close();
}

//By Ghostxiu. 2017/9/28
//2017/9/28 完成，封装并简化一些opencv的本身功能以方便使用