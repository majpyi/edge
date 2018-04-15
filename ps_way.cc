//ʵ�ּ�һЩ���õ�OPENCV�������Ӷ�������ô���
#include "use.h"
//ע������ͷ�ļ�ʱ��ʹ�õ��ļ���ַ�������Դ�ļ���������д�ļ�������ڹ���
//----------------------[main()����]----------------------
//�������
//-------------------------------------------------------
#if(0)//�ر�main�����������ڲ���
int main(int argc, char ** argv)
{
	//��ȡԭͼ
	read_src(g_src, "lena.jpg");
	//��ʾԭͼ
	show_src(g_src, "Origin pic");

	//ת��ԭͼΪ�Ҷ�ͼ
	to_gray(g_src, g_srcGray);

	//��ʾ�Ҷ�ͼ
	show_src(g_srcGray, "Gray pic");
	//����Ҷ�ͼ�ļ�
	src_to_bmp(g_srcGray, "output//lena_gray.bmp");

	//��ͼ����������ʽд��txt�ļ�

	//output_arr_txt(g_srcGray, "output//g_srcGray.txt");


	waitKey(0); // �ȴ�һ�ΰ������������
	return 0;
}
#endif

//----------------------[to_gray()����]---------------------
//ͼ��ת�Ҷ�ͼ
//----------------------------------------------------------
bool to_gray(const Mat & src, Mat & src_Gray)
{
	if (src.empty())//��ȡʧ��ʱ
	{
		cout << "Could not open or find the image" << endl;
		return false;
	}
	cvtColor(src, src_Gray, CV_RGB2GRAY);//��ͼƬת��Ϊ�Ҷ�ͼ
	return true;
}


//----------------------[read_src()����]----------------------
//ͼ���ȡ����
//�������������õ�Mat�ͣ���ȡ����String���ļ�����p��ͨ������Ĭ��0Ϊ������ͨ��
//��ͼ���ļ���ȡ����Mat�ͣ���ת��Ϊ32λ������
//��ѡ��Ϊ��ͨ��ͼ���� p = 1 
//------------------------------------------------------------
bool read_src(Mat & src, const string & imgName ,int p)
{
	if (p == 0)
	{//�����ȡ��ʽ
		src = imread(imgName.c_str(), IMREAD_COLOR);
	}
	else if (p == 1)
	{//����ͨ����ȡ
		src = imread(imgName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	}
	if (p == 1)
	{
		src.convertTo(src, CV_32F);
	}
	//ͼ����ֵƥ��ͳһ����Ϊ32λ��������
	if (src.empty())//��ȡʧ��ʱ
	{
		cout << "Could not open or find the image" << std::endl;
		return false;
	}
	return true;
}

//----------------------[src_to_bmp()����]----------------------
//��Mat�����е�ͼƬ��Ϣд�뵽bmp
//-------------------------------------------------------------
bool src_to_bmp(const Mat & src, const string & imgName)
{
	if (src.empty())//��ȡʧ��ʱ
	{
		cout << "Could not open or find the image" << endl;
		return false;
	}
	imwrite(imgName, src);
	return true;
}

//----------------------[show_src()����]----------------------
//ͼ����ʾ����
//------------------------------------------------------------
bool show_src(const Mat & src, const string & imgName)
{
	if (src.empty())//��ȡʧ��ʱ
	{
		cout << "Could not open or find the image" << endl;
		return false;
	}
	namedWindow(imgName);
	imshow(imgName, src);
	return true;
}


//--------------------[output_arr_txt()����]--------------------
//ͼ������д��txt�ļ�
//----------------------------------------------------------------
void output_arr_txt(const Mat &src, const string & txt_name)
{
	
	if (src.empty())//��ȡʧ��ʱ
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


//----------------------[read_Graysrc()����]----------------------
//ֱ�ӽ�ͼ���ȡΪ�Ҷ�ͼ
//------------------------------------------------------------
bool read_Graysrc(Mat & gray_src, const string & imgName)
{
	gray_src = imread(imgName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	if (gray_src.empty())//��ȡʧ��ʱ
	{
		cout << "Could not open or find the image" << std::endl;
		return false;
	}
	return true;
}



//--------------------[output_arr_csv()����]--------------------
//ͼ������д��excel�ļ�
//----------------------------------------------------------------
void output_arr_csv(const Mat &src, const string & txt_name)
{

	if (src.empty())//��ȡʧ��ʱ
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
//2017/9/28 ��ɣ���װ����һЩopencv�ı������Է���ʹ��