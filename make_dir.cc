//创建文件路径中所需却不存在的文件夹
#include <iostream>
#include <string>
//#include <io.h>
//#include <direct.h>
#include <cstring>
#include <zconf.h>

using namespace std;
void create_dir(string fileName);
/*
int main(int argc, char* argv[])
{
	string fileName="psp\\1\\2\\3\\a.txt";
	create_dir(fileName);


	return 0;
}
*/

void create_dir(string fileName)
{
	const char *tag;
	for(tag = fileName.data();*tag;tag++)
	{
		if (*tag=='\\')
		{
			char buf[1000],path[1000];
			strcpy(buf,fileName.c_str());
			buf[strlen(fileName.c_str())-strlen(tag)+1]=NULL;
			strcpy(path,buf);
			if (access(path,6)==-1)
			{
//				mkdir(path);  //创建成功返回0 不成功返回-1
			}
		}
	}
}
