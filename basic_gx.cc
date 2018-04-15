#include "basic_gx.h"

//----------------------[min_bit8()函数]---------------------
//求8位数中的最小值
//-----------------------------------------------------------
bit8 min_bit8(const bit8 & bt8)
{
    //求8位数中的最小值，存在min_b8中
    bit8 b8 = bt8;
    int min_b8 = b8.to_ulong();
    for (int i = 1; i < 8; ++i)
    {
        //注意此处的b8和前面的位和声明刚好是反的
        bool b80_f = b8[0];
        b8 = b8 >> 1;
        b8[7] = b80_f;
        int b = b8.to_ulong();
        if (b < min_b8)
        {
            min_b8 = b;
        }

    }
    b8 = bit8(min_b8);
    return b8;
}


/*

//----------------------[Catch_fileName()函数]---------------------
//读取文件名称 ，包括限定文件格式类型和不限定格式类型的重载
//版本1 两个参数 1 string类型 路径		2 vector<string> 类型 文件名
//版本2 三个参数 1 string类型 路径		2 vector<string> 类型 文件名
//			   3 string类型 文件格式
//----------------------------------------------------------------
void Catch_fileName(string path, vector<string>& files)
{
    long   hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;//用来存储文件信息的结构体
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)  //第一次查找
    {
        do
        {
            if ((fileinfo.attrib &  _A_SUBDIR))  //如果查找到的是文件夹
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)  //进入文件夹查找
                {
                    files.push_back(p.assign(path).append("\\").append(fileinfo.name));
                    Catch_fileName(p.assign(path).append("\\").append(fileinfo.name), files);
                }
            }
            else //如果查找到的不是是文件夹
            {
                files.push_back(p.assign(fileinfo.name));  //将文件路径保存，也可以只保存文件名:    p.assign(path).append("\\").append(fileinfo.name)
            }

        } while (_findnext(hFile, &fileinfo) == 0);

        _findclose(hFile); //结束查找
    }

}
//重载版本



//---------------------[GetFilesName()函数]--------------------
//递归查找目录下查找文件名
//------------------------------------------------------------
void GetFilesName(string path, string exd, vector<string>& files)
{
    using namespace std;
    //文件句柄
    long   hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
    string pathName, exdName;

    if (0 != strcmp(exd.c_str(), ""))
    {
        exdName = "\\*." + exd;
    }
    else
    {
        exdName = "\\*";
    }

    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
    {
        do
        {
            //如果是文件夹中仍有文件夹,迭代之
            //如果不是,加入列表
            if ((fileinfo.attrib &  _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    GetFilesName(pathName.assign(path).append("\\").append(fileinfo.name), exd, files);
            }
            else
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}


//---------------------[create_dir()函数]--------------------
//递归创建所需文件夹（如果没有）
//----------------------------------------------------------
void create_dir(string fileName)
{
    const char *tag;
    for (tag = fileName.data(); *tag; tag++)
    {
        if (*tag == '\\' || *tag == '//')
        {
            char buf[1000], path[1000];
            strcpy(buf, fileName.c_str());
            buf[strlen(fileName.c_str()) - strlen(tag) + 1] = NULL;
            strcpy(path, buf);
            if (_access(path, 6) == -1)
            {
                _mkdir(path);  //创建成功返回0 不成功返回-1
            }
        }
    }
}

 */

//By Ghostxiu. 2017/10/21
