//实现陈老师9.6提出的模版识别边缘检测的问题的主程序
#include <modeltolearn.h>
#include <basic_gx.h>

int main()
{

	//test();//基本功能测试函数，在函数主体中操作需要打开的功能


	//单文件处理过程
	//single_process();

//批处理
#if(0)	
	int p_th = 5;//阈值递进单位
	
	for (int i = 1; i <= 5; ++i)
	{
		//初始化，记录模板出现边缘点的百分比数组
		std::fill(&ModelPercent[0], 
			&ModelPercent[0] + EDGE8_MODEL_SIZE, 0.0);
		//批处理过程多文件
		batch_process(p_th*i);

		//输出同阈值下多图片模板出现的百分比
		same_thre_per(p_th*i);
	
	}
#endif
//使用区分度阈值测定边缘
#if(0)	
	int p_th = 5;//阈值递进单位

	for (int i = 1; i <= 5; ++i)
	{

		//批处理过程多文件
		dis_pow_try_threhold(p_th*i);

	}
#endif
	//绘制直方图
	//plot_disgram();

	//Mat loc_edge_mat = FindLocalMaxEdge("input\\brain\\1.jpg",5);//局部最大区分边缘法
	//twice_dis_pow();//保存二次区分度图
	//otherway_edge_detect();//使用其他边缘检测的方法
	//org_expert_csv();//存储csv文件

	FindFriendEdge("/Users/Quantum/Desktop/edge/input/4.jpg");//找朋友方法找最大值点
//	waitKey();


	return 0;
}

//---------------------[FindFriendEdge()函数]--------------------
//使用找朋友的方法寻找边缘点
//-----------------------------------------------------------------
Mat FindFriendEdge( const string & pic_name)
{

	read_src(g_srcGray, pic_name, PIPE_GRAY);
	output_arr_csv(g_srcGray, "/Users/Quantum/Desktop/edge/out/原图.csv");
	src_to_bmp(g_srcGray, "/Users/Quantum/Desktop/edge/out/g_srcGray.jpg");
	//show_src(g_srcGray/255, "原图");
	int threhold = 20;//阈值

/*
	//彩色图像处理
	Mat color_pic=imread("/Users/Quantum/Desktop/edge/out/4.jpg", CV_LOAD_IMAGE_COLOR);
	IplImage* img = cvLoadImage("/Users/Quantum/Desktop/edge/out/4.jpg", 1);


	Mat cfix7vs1 = color_fix7vs1(g_srcGray,img, color_pic);
	src_to_bmp(cfix7vs1, "/Users/Quantum/Desktop/edge/out/color_fix7vs.jpg");
	//output_arr_csv(cfix7vs1, "/Users/Quantum/Desktop/edge/out/cfix7vs1.csv");


	Mat csortpixel = color_sortpixel(g_srcGray, color_pic, img, threhold);
	//Mat csortpixel = color_sortpixel(g_srcGray, cfix7vs1, img, threhold);
	src_to_bmp(csortpixel, "/Users/Quantum/Desktop/edge/out/csortpixel.jpg");
	

	//以彩色图像来分区
	Mat vote_c1 =	color_findBigSmallArea(g_srcGray, img,20);
	output_arr_csv(vote_c1, "/Users/Quantum/Desktop/edge/out/cc.csv");
	Mat newbs_mat255 = voteBigSmall(vote_c1);
	Mat both_edge225 = find_edgeboth2(newbs_mat255);
	//show_src(both_edge2, "both_edge2");
	src_to_bmp(both_edge225, "/Users/Quantum/Desktop/edge/out/color_both_edge.jpg");

	//以灰度值来分区
	Mat vote_matr = mfindBigSmallArea(g_srcGray, threhold);
	output_arr_csv(vote_matr, "/Users/Quantum/Desktop/edge/out/gg.csv");


	Mat BigArea_raw_c1, SmallArea_raw_c1, diff_m_raw_c1;
	Mat color = mcolor_fixDiff(g_srcGray, color_pic, vote_c1, BigArea_raw_c1, SmallArea_raw_c1, diff_m_raw_c1, 10);
	//Mat color = mcolor_fixDiff(g_srcGray,color_pic, vote_matr,BigArea_raw_c1,SmallArea_raw_c1,diff_m_raw_c1,90);
	src_to_bmp(color, "/Users/Quantum/Desktop/edge/out/color.jpg");

*/



	/*
	//没有进行修复之前的结果展示
	Mat BigArea_raw, SmallArea_raw, diff_m_raw;
	Mat vote_mat_raw = mfindBigSmallArea(g_srcGray, threhold);
	output_arr_csv(vote_mat_raw, "/Users/Quantum/Desktop/edge/out/原图投票图.csv");
	//将属于最大最小区域的点标记区分出最大区域
	Mat bs_mat_raw= voteBigSmall(vote_mat_raw);
	//修理矛盾点
	Mat fix_src_raw = mfixDiff(g_srcGray, vote_mat_raw, BigArea_raw, SmallArea_raw, diff_m_raw, threhold);
	//修正之后的投票图
	Mat fixvote_mat_raw = mfindBigSmallArea(fix_src_raw, threhold);
	//三色显示
	Mat edge3color_raw = color3_edge(BigArea_raw, SmallArea_raw, diff_m_raw);
	src_to_bmp(edge3color_raw, "/Users/Quantum/Desktop/edge/out/原图三色边缘图.jpg");
	Mat color2_edge_raw = color2_vote(bs_mat_raw);
	src_to_bmp(color2_edge_raw, "/Users/Quantum/Desktop/edge/out/原图双色边缘图.jpg");
*/




//废弃的修正方法，sort7vs1的情况进行修正有问题
/* 
//迭代按照7vs1的结果进行修正,包含排序法的7vs1
Mat  fix= fix7vs1(g_srcGray);
Mat sort7vs = sort7vs1(fix);
output_arr_csv(fix, "/Users/Quantum/Desktop/edge/out/7vs1修正.csv");
Mat  fix1 = fix7vs1(sort7vs);
Mat sort7vs11 = sort7vs1(fix1);
//output_arr_csv(fix1, "/Users/Quantum/Desktop/edge/out/7vs1修正1.csv");
Mat  fix2 = fix7vs1(sort7vs11);
Mat sort7vs12 = sort7vs1(fix2);
//output_arr_csv(fix2, "/Users/Quantum/Desktop/edge/out/7vs1修正2.csv");
Mat  fix3 = fix7vs1(sort7vs12);
Mat sort7vs13 = sort7vs1(fix3);
//output_arr_csv(fix3, "/Users/Quantum/Desktop/edge/out/7vs1修正3.csv");
Mat  fix4 = fix7vs1(sort7vs13);
Mat sort7vs14= sort7vs1(fix4);
//output_arr_csv(fix4, "/Users/Quantum/Desktop/edge/out/7vs1修正4.csv");
Mat  fix5 = fix7vs1(sort7vs14);
Mat sort7vs15= sort7vs1(fix5);
output_arr_csv(sort7vs15, "/Users/Quantum/Desktop/edge/out/7vs1修正5.csv");
*/


/*
//迭代按照7vs1的结果进行修正
Mat  fix= fix7vs1(g_srcGray);
//testfix7vs1(g_srcGray,4,14);
//output_arr_csv(fix, "/Users/Quantum/Desktop/edge/out/7vs1修正.csv");

output_arr_csv(fix, "/Users/Quantum/Desktop/edge/out/7vs1修正.csv");
//src_to_bmp(fix, "/Users/Quantum/Desktop/edge/out/7vs1修正.jpg");


Mat  fix1 = fix7vs1(fix);
//output_arr_csv(fix1, "/Users/Quantum/Desktop/edge/out/7vs1修正1.csv");
Mat  fix2 = fix7vs1(fix1);
//output_arr_csv(fix2, "/Users/Quantum/Desktop/edge/out/7vs1修正2.csv");
Mat  fix3 = fix7vs1(fix2);
//output_arr_csv(fix3, "/Users/Quantum/Desktop/edge/out/7vs1修正3.csv");
Mat  fix4 = fix7vs1(fix3);
//output_arr_csv(fix4, "/Users/Quantum/Desktop/edge/out/7vs1修正4.csv");
Mat  fix5 = fix7vs1(fix4);
output_arr_csv(fix5, "/Users/Quantum/Desktop/edge/out/7vs1修正5.csv");
*/


/*
//迭代通过排序修正，其中包含centerfix的对中心点的修正
Mat fixgray5 = sortpixel(g_srcGray, threhold);
//Mat fixgray5 = sortpixel(fix5, threhold);
output_arr_csv(fixgray5, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后1.csv");
Mat fixgray4 = sortpixel(fixgray5, threhold);
//output_arr_csv(fixgray4, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后2.csv");
Mat fixgray3 = sortpixel(fixgray4, threhold);
//output_arr_csv(fixgray3, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后3.csv");
Mat fixgray2 = sortpixel(fixgray3, threhold);
//output_arr_csv(fixgray2, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后4.csv");
Mat fixgray1 = sortpixel(fixgray2, threhold);
//output_arr_csv(fixgray1, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后5.csv");


//Mat fixgray = sortpixel(g_srcGray, threhold);
//Mat fixgray = sortpixel(fix, threhold);
Mat fixgray = sortpixel(fixgray1, threhold);


output_arr_csv(fixgray, "/Users/Quantum/Desktop/edge/out/sortpixel.csv");
src_to_bmp(fixgray, "/Users/Quantum/Desktop/edge/out/sortpixel.jpg");

*/

//测试
/*
int di[8 ] = { +1,+0, -1, -1, -1, 0, +1, +1 };
int dj[8 ] = { +1,+1, +1, 0, -1, -1, -1, +0 };

for (int k = 0; k <= 7; k++)
{
	cout << k << endl;
	testsortpixel(g_srcGray, threhold, 3 - 1+di[k], 5 - 1+dj[k]);
	cout << endl << endl;;
}
*/

//testsortpixel(g_srcGray, threhold, 3-1, 5-1);



/*处理效果不好，废弃
	Mat fixgray5 = newsortpixel(g_srcGray, threhold);
	//Mat fixgray5 = sortpixel(fix5, threhold);
	output_arr_csv(fixgray5, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后1.csv");
	Mat fixgray4 = newsortpixel(fixgray5, threhold);
	output_arr_csv(fixgray4, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后2.csv");
	Mat fixgray3 = newsortpixel(fixgray4, threhold);
	output_arr_csv(fixgray3, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后3.csv");
	Mat fixgray2 = newsortpixel(fixgray3, threhold);
	output_arr_csv(fixgray2, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后4.csv");
	Mat fixgray1 = newsortpixel(fixgray2, threhold);
	output_arr_csv(fixgray1, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后5.csv");
	Mat fixgray = newsortpixel(fixgray1, threhold);
	output_arr_csv(fixgray, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后.csv");
	*/





//对fix7vs1的测试函数
//testfix7vs1(g_srcGray, 36, 20);

//Mat fixgray = fix5;
//Mat fixgray = g_srcGray;    


//testsortpixel(g_srcGray, threhold, 245, 54);
//Mat fixgray = sortpixel(g_srcGray, threhold);
//output_arr_csv(fixgray, "/Users/Quantum/Desktop/edge/out/排序法首先处理之后.csv"); 



/*   原始处理的方法，不进行排序的处理      
//进行投票，分别记录属于最大、最小区域的次数
Mat vote_mat = findBigSmallArea(g_srcGray, threhold);
output_arr_csv(vote_mat, "/Users/Quantum/Desktop/edge/out/投票图.csv");
//将属于最大最小区域的点标记区分出最大区域
Mat bs_mat = voteBigSmall(vote_mat);
output_arr_csv(bs_mat, "/Users/Quantum/Desktop/edge/out/大小区域.csv");
//修理矛盾点
Mat fix_src = voteToFix(g_srcGray, vote_mat, BigArea, SmallArea,diff_m,threhold);
output_arr_csv(fix_src, "/Users/Quantum/Desktop/edge/out/原图修正.csv");
//三色显示
Mat edge3color = color3_edge(BigArea, SmallArea,diff_m);
//Mat edge2color = color3_edge(BigArea, SmallArea, diff_m,2);
show_src(edge3color,"大小区域3色显示");
src_to_bmp(edge3color, "/Users/Quantum/Desktop/edge/out/3色边缘图.jpg");
//src_to_bmp(edge2color, "/Users/Quantum/Desktop/edge/out/4.jpg2色边缘图.jpg");
*/



/*
//师兄的方法
	//进行投票，分别记录属于最大、最小区域的次数
	Mat vote_mat = findBigSmallArea(fixgray, threhold);
	output_arr_csv(vote_mat, "/Users/Quantum/Desktop/edge/out/投票图.csv");
	//将属于最大最小区域的点标记区分出最大区域
	Mat bs_mat = voteBigSmall(vote_mat);
	output_arr_csv(bs_mat, "/Users/Quantum/Desktop/edge/out/大小区域.csv");
	//修理矛盾点
	Mat fix_src = voteToFix(fixgray, vote_mat, BigArea, SmallArea, diff_m, threhold);
	output_arr_csv(fix_src, "/Users/Quantum/Desktop/edge/out/原图修正.csv");
	//三色显示
	Mat edge3color = color3_edge(BigArea, SmallArea, diff_m);
	//Mat edge2color = color3_edge(BigArea, SmallArea, diff_m,2);
	show_src(edge3color, "大小区域3色显示");
	src_to_bmp(edge3color, "/Users/Quantum/Desktop/edge/out/3色边缘图.jpg");
	//src_to_bmp(edge2color, "/Users/Quantum/Desktop/edge/out/4.jpg2色边缘图.jpg");


	//   调用师兄的修正函数处理
	Mat newBigArea, newSmallArea, newdiff_m;
	Mat newvote_mat = findBigSmallArea(fix_src, threhold);
	Mat newbs_mat = voteBigSmall(newvote_mat);
	Mat newfix_src = voteToFix(fix_src, newvote_mat, newBigArea, newSmallArea, newdiff_m, threhold);
	Mat newedge3color = color3_edge(newBigArea, newSmallArea, newdiff_m);

	Mat newBigArea3, newSmallArea3, newdiff_m3;
	Mat newvote_mat3 = findBigSmallArea(newfix_src, threhold);
	Mat newbs_mat3 = voteBigSmall(newvote_mat3);
	Mat newfix_src3 = voteToFix(newfix_src, newvote_mat3, newBigArea3, newSmallArea3, newdiff_m3, threhold);
	Mat newedge3color3 = color3_edge(newBigArea3, newSmallArea3, newdiff_m3);

	Mat newBigArea4, newSmallArea4, newdiff_m4;
	Mat newvote_mat4 = findBigSmallArea(newfix_src3, threhold);
	Mat newbs_mat4 = voteBigSmall(newvote_mat4);
	Mat newfix_src4 = voteToFix(newfix_src3, newvote_mat4, newBigArea4, newSmallArea4, newdiff_m4, threhold);
	Mat newedge3color4 = color3_edge(newBigArea4, newSmallArea4, newdiff_m4);

	Mat newBigArea5, newSmallArea5, newdiff_m5;
	Mat newvote_mat5 = findBigSmallArea(newfix_src4, threhold);
	Mat newbs_mat5 = voteBigSmall(newvote_mat5);
	Mat newfix_src5 = voteToFix(newfix_src4, newvote_mat5, newBigArea5, newSmallArea5, newdiff_m5, threhold);
	Mat newedge3color5 = color3_edge(newBigArea5, newSmallArea5, newdiff_m5);
*/
	


//   我的方法处理
	//进行投票，分别记录属于最大、最小区域的次数
Mat BigArea, SmallArea, diff_m;
Mat vote_mat = mfindBigSmallArea(g_srcGray, threhold);
//Mat vote_mat = mfindBigSmallArea(fixgray, threhold);
	output_arr_csv(vote_mat, "/Users/Quantum/Desktop/edge/out/投票图.csv");
	//将属于最大最小区域的点标记区分出最大区域
	Mat bs_mat = voteBigSmall(vote_mat);
	output_arr_csv(bs_mat, "/Users/Quantum/Desktop/edge/out/大小区域.csv");
	//修理矛盾点
Mat fix_src = mfixDiff(g_srcGray, vote_mat, BigArea, SmallArea, diff_m, threhold);
	//Mat fix_src = mfixDiff(fix5, vote_mat, BigArea, SmallArea, diff_m, threhold);
//	output_arr_csv(fix_src, "/Users/Quantum/Desktop/edge/out/原图修正.csv");

	//修正之后的投票图
	//Mat fixvote_mat = mfindBigSmallArea(fix_src, threhold);
//	output_arr_csv(fixvote_mat, "/Users/Quantum/Desktop/edge/out/修正后投票图.csv");


/*
	//三色显示
	Mat edge3color = color3_edge(BigArea, SmallArea, diff_m);
	//Mat edge2color = color3_edge(BigArea, SmallArea, diff_m,2);
	//show_src(edge3color, "大小区域3色显示");
	//src_to_bmp(edge3color, "/Users/Quantum/Desktop/edge/out/双色边缘图.jpg");
	//src_to_bmp(edge2color, "/Users/Quantum/Desktop/edge/out/4.jpg2色边缘图.jpg");

	Mat color2_edge1111 = color2_vote(bs_mat);
	show_src(color2_edge1111, "color2_edge1111");
	src_to_bmp(color2_edge1111, "/Users/Quantum/Desktop/edge/out/color2_edge1111.jpg");

	output_arr_csv(fix_src, "/Users/Quantum/Desktop/edge/out/fix_src.csv");
	output_arr_csv(vote_mat, "/Users/Quantum/Desktop/edge/out/fix_src_vote_mat.csv");
	src_to_bmp(fix_src, "/Users/Quantum/Desktop/edge/out/fix_src.jpg");
	*/


	    //迭代的过程
	//调用我自己的修正的方法进行处理的结果
	Mat newBigArea, newSmallArea, newdiff_m;
	Mat newvote_mat = mfindBigSmallArea(fix_src, threhold);
	Mat newbs_mat = voteBigSmall(newvote_mat);
	Mat newfix_src = mfixDiff(fix_src, newvote_mat, newBigArea, newSmallArea, newdiff_m, threhold);
	Mat newedge3color = color3_edge(newBigArea, newSmallArea, newdiff_m);

	Mat newBigArea3, newSmallArea3, newdiff_m3;
	Mat newvote_mat3 = mfindBigSmallArea(newfix_src, threhold);
	Mat newbs_mat3 = voteBigSmall(newvote_mat3);
	Mat newfix_src3 = mfixDiff(newfix_src, newvote_mat3, newBigArea3, newSmallArea3, newdiff_m3, threhold);
	Mat newedge3color3 = color3_edge(newBigArea3, newSmallArea3, newdiff_m3);

	Mat newBigArea4, newSmallArea4, newdiff_m4;
	Mat newvote_mat4 = mfindBigSmallArea(newfix_src3, threhold);
	Mat newbs_mat4 = voteBigSmall(newvote_mat4);
	Mat newfix_src4 = mfixDiff(newfix_src3, newvote_mat4, newBigArea4, newSmallArea4, newdiff_m4, threhold);
	Mat newedge3color4 = color3_edge(newBigArea4, newSmallArea4, newdiff_m4);

	Mat newBigArea5, newSmallArea5, newdiff_m5;
	Mat newvote_mat5 = mfindBigSmallArea(newfix_src4, threhold);
	Mat newbs_mat5 = voteBigSmall(newvote_mat5);
	Mat newfix_src5 = mfixDiff(newfix_src4, newvote_mat5, newBigArea5, newSmallArea5, newdiff_m5, threhold);
	Mat newedge3color5 = color3_edge(newBigArea5, newSmallArea5, newdiff_m5);

	//output_arr_csv(newfix_src4, "/Users/Quantum/Desktop/edge/out/修正后结果.csv");

	//show_src(edge3color, "大小区域双色显示");
	//src_to_bmp(edge3color, "/Users/Quantum/Desktop/edge/out/1三色边缘图.jpg");

	/*
	//查看矛盾点
	output_arr_csv(vote_mat, "/Users/Quantum/Desktop/edge/out/tag1.csv");
	output_arr_csv(newvote_mat, "/Users/Quantum/Desktop/edge/out/tag2.csv");
	output_arr_csv(newvote_mat3, "/Users/Quantum/Desktop/edge/out/tag3.csv");
	output_arr_csv(newvote_mat4, "/Users/Quantum/Desktop/edge/out/tag4.csv");
	output_arr_csv(newvote_mat5, "/Users/Quantum/Desktop/edge/out/tag5.csv");
	//output_arr_csv(fixgray, "/Users/Quantum/Desktop/edge/out/csv1.csv");
	output_arr_csv(fix_src, "/Users/Quantum/Desktop/edge/out/csv2.csv");
	output_arr_csv(newfix_src, "/Users/Quantum/Desktop/edge/out/csv3.csv");
	output_arr_csv(newfix_src3, "/Users/Quantum/Desktop/edge/out/csv4.csv");
	output_arr_csv(newfix_src4, "/Users/Quantum/Desktop/edge/out/csv5.csv");
	output_arr_csv(newfix_src5, "/Users/Quantum/Desktop/edge/out/csv6.csv");
	*/
	/*
	//show_src(newedge3color, "new大小区域双色显示");
	src_to_bmp(newedge3color, "/Users/Quantum/Desktop/edge/out/2三色边缘图.jpg");

	//show_src(newedge3color3, "3大小区域双色显示");
	src_to_bmp(newedge3color3, "/Users/Quantum/Desktop/edge/out/3三色边缘图.jpg");

	//show_src(newedge3color4, "4大小区域双色显示");
	src_to_bmp(newedge3color4, "/Users/Quantum/Desktop/edge/out/4三色边缘图.jpg");

	//show_src(newedge3color5, "5大小区域双色显示");
	src_to_bmp(newedge3color5, "/Users/Quantum/Desktop/edge/out/5三色边缘图.jpg");

	Mat color2_edge =  color2_vote(bs_mat);
	//show_src(color2_edge, "color2_edge");
	src_to_bmp(color2_edge, "/Users/Quantum/Desktop/edge/out/5双色边缘图.jpg");

	Mat findedge = find_edge(bs_mat);
	//show_src(findedge, "findedge");
	src_to_bmp(findedge, "/Users/Quantum/Desktop/edge/out/findgeda.jpg");
	
	Mat findedgexiao = find_edgexiao(bs_mat);
	//show_src(findedgexiao, "findedge");
	src_to_bmp(findedgexiao, "/Users/Quantum/Desktop/edge/out/findgexiao.jpg");

	Mat both_edge = find_edgeboth(bs_mat);
	//show_src(both_edge, "both_edge");
	src_to_bmp(both_edge, "/Users/Quantum/Desktop/edge/out/最终both_edge.jpg");

	Mat both_edge2 = find_edgeboth2(bs_mat);
	//show_src(both_edge2, "both_edge2");
	src_to_bmp(both_edge2, "/Users/Quantum/Desktop/edge/out/最终both_edge2.jpg");
	*/



	Mat both_edge = find_edgeboth(bs_mat);
	//show_src(both_edge, "both_edge");
	src_to_bmp(both_edge, "/Users/Quantum/Desktop/edge/out/最终both_edge.jpg");

	Mat both_edge2 = find_edgeboth2(bs_mat);
	//show_src(both_edge2, "both_edge2");
	src_to_bmp(both_edge2, "/Users/Quantum/Desktop/edge/out/最终both_edge2.jpg");


	Mat both_edge5 = find_edgeboth(newbs_mat5);
	//show_src(both_edge, "both_edge");
	src_to_bmp(both_edge5, "/Users/Quantum/Desktop/edge/out/最终both_edge5.jpg");

	Mat both_edge25 = find_edgeboth2(newbs_mat5);
	//show_src(both_edge2, "both_edge2");
	src_to_bmp(both_edge25, "/Users/Quantum/Desktop/edge/out/最终both_edge25.jpg");


	//src_to_bmp(fix_src, "/Users/Quantum/Desktop/edge/out/newfix_src5.jpg");

	//Histogramdaxiao(g_srcGray,tag);
	//histm();

    //canny算子的效果展示
    Mat canny = CannyEdge(pic_name);
    src_to_bmp(canny, "/Users/Quantum/Desktop/edge/out/canny.jpg");
    Mat Sobel = SobelEdge(pic_name);
    src_to_bmp(Sobel, "/Users/Quantum/Desktop/edge/out/Sobel.jpg");
    Mat Lap =LaplacianEdge(pic_name);
    src_to_bmp(Lap, "/Users/Quantum/Desktop/edge/out/Laplacian.jpg");
    Mat Sch = ScharEdge(pic_name);
    src_to_bmp(Sch, "/Users/Quantum/Desktop/edge/out/Schare.jpg");


	/*
	Mat tag = tagdaxiao(newbs_mat5);
	output_arr_csv(tag, "/Users/Quantum/Desktop/edge/out/tag.csv");
	Mat xiqfu = enhance(newfix_src5, tag,g_srcGray);
	//Mat xiqfu = enhance(g_srcGray, tag);
	src_to_bmp(xiqfu, "/Users/Quantum/Desktop/edge/out/enhance.jpg");
	output_arr_csv(xiqfu, "/Users/Quantum/Desktop/edge/out/enhance.csv");
	output_arr_csv(newvote_mat5, "/Users/Quantum/Desktop/edge/out/Last投票图.csv");
	output_arr_csv(g_srcGray, "/Users/Quantum/Desktop/edge/out/g_srcGray1111.csv");
	output_arr_csv(tag, "/Users/Quantum/Desktop/edge/out/tag1111.csv");
	*/


	//Histogram(g_srcGray);
	// 映射算法
	/*
	Mat mpic = Mapping(g_srcGray, tag, g_srcGray);
	src_to_bmp(mpic, "/Users/Quantum/Desktop/edge/out/mpic.jpg");
	output_arr_csv(mpic, "/Users/Quantum/Desktop/edge/out/mpic.csv");
	*/


	//   附近归属
	Mat newgray;
    Mat na_vote;
    Mat newbig;
    Mat newsmall;
	Near_attribution(g_srcGray,bs_mat,BigArea,SmallArea,1,1,newgray,na_vote,newbig,newsmall);
	output_arr_csv(na_vote, "/Users/Quantum/Desktop/edge/out/na_vote.csv");
	Mat newgray2;
    Mat na_vote2;
    Mat newbig2;
    Mat newsmall2;
	Near_attribution(g_srcGray,na_vote,newbig,newsmall,1,1,newgray2,na_vote2,newbig2,newsmall2);
	output_arr_csv(na_vote2, "/Users/Quantum/Desktop/edge/out/na_vote2.csv");


    Mat na_both_edge = find_edgeboth2(na_vote);
    src_to_bmp(na_both_edge, "/Users/Quantum/Desktop/edge/out/na_vote.jpg");
    Mat na_both_edge2 = find_edgeboth2(na_vote2);
    src_to_bmp(na_both_edge2, "/Users/Quantum/Desktop/edge/out/na_vote2.jpg");




    return bs_mat;

}

//By Ghostxiu. 2017/11 /17