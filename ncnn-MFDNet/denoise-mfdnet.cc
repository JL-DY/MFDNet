#include "net.h"
#include "gpu.h"
#include <sys/time.h>  
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>

// 自定义层
class MFDNet_haar : public ncnn::Layer
{
public:
    MFDNet_haar()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        
        //#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr1 = bottom_blob.channel(p % channels).row(0);
            const float* ptr2 = bottom_blob.channel(p % channels).row(1);
            const float* ptr3 = bottom_blob.channel(p % channels).row(0) + 1;
            const float* ptr4 = bottom_blob.channel(p % channels).row(1) + 1;
            float* outptr = top_blob.channel(p);
        
            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    if(p < channels){
                        *outptr = (*ptr1 + *ptr2 + *ptr3 + *ptr4)/2;

                        outptr += 1;
                        ptr1 += 2;
                        ptr2 += 2;
                        ptr3 += 2;
                        ptr4 += 2;
                    }
                    if(channels<=p && p<(channels*2)){
                        *outptr = (*ptr3 + *ptr4 - *ptr1 - *ptr2)/2;

                        outptr += 1;
                        ptr1 += 2;
                        ptr2 += 2;
                        ptr3 += 2;
                        ptr4 += 2;
                    }
                    if((channels*2)<=p && p<(channels*3)){
                        *outptr = (*ptr2 + *ptr4 - *ptr1 - *ptr3)/2;

                        outptr += 1;
                        ptr1 += 2;
                        ptr2 += 2;
                        ptr3 += 2;
                        ptr4 += 2;
                    }
                    if((channels*3)<=p && p<(channels*4)){
                        *outptr = (*ptr1 + *ptr4 - *ptr2 - *ptr3)/2;

                        outptr += 1;
                        ptr1 += 2;
                        ptr2 += 2;
                        ptr3 += 2;
                        ptr4 += 2;
                    }
                }

                ptr1 += w;
                ptr2 += w;
                ptr3 += w;
                ptr4 += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(MFDNet_haar)

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void pretty_print(const ncnn::Mat &m){
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z=0; z<m.d; z++)
        {
            for (int y=0; y<1; y++) //行，h
            {
                for (int x=0; x<10; x++) //列，w
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

template <typename T1, typename T2>
void cv_print(const cv::Mat &m){
    for(int i=0; i<10; i++){
        std::cout << static_cast<T1>(m.at<T2>(i)) << std::endl;
    }
    printf("------------------------\n");
}

cv::Mat denoise_mfdnet(ncnn::Net &mfdnet, const cv::Mat& bgr)
{    
    auto start = GetCurrentUS();
    float duration = 0.0;
    cv::Mat nor_img;
    bgr.convertTo(nor_img, CV_32FC1, 1.0/255.0f); 
    ncnn::Mat in(nor_img.cols, nor_img.rows, 1, (void*)nor_img.data);
    const float mean_vals[1] = { 0.0f };
    const float norm_vals1[1] = { 255.0f };

    start = GetCurrentUS();
    ncnn::Extractor ex = mfdnet.create_extractor();
    ncnn::Mat out;
    ex.input("inputs", in);
    ex.extract("outputs", out);
    out.substract_mean_normalize(mean_vals, norm_vals1);
    duration = (GetCurrentUS()-start) / 1000.0;
    std::cout<<"infer_duration:"<<duration<<std::endl;
    
    start = GetCurrentUS();
    cv::Mat cv_out(out.h, out.w, CV_8UC1);
    out.to_pixels(cv_out.data, ncnn::Mat::PIXEL_GRAY);
    duration = (GetCurrentUS() - start) / 1000.0;
    std::cout<<"post_duration:"<<duration<<std::endl;
    
    return cv_out;
}

int main(int argc, char** argv)
{
    if (argc != 9)
    {
        std::cout << "please input:\n"
                  << "model_param\n"
                  << "model_bin\n"
                  << "input img directory\n"
                  << "output img directory\n"
                  << "repeat\n"
                  << "use_gpu(1:use)\n"
                  << "num_threads\n"
                  << "fp16\n"
                  <<std::endl;
        
        return -1;
    }

    //加载参数
    const char* model_param = argv[1];
    const char* model_bin = argv[2];
    const char* imagepath = argv[3];
    const char* output_img_path = argv[4];
    int repeat = atoi(argv[5]);
    int use_gpu = atoi(argv[6]);
    int threads = atoi(argv[7]);
    bool fp16 = atoi(argv[8]) ? true : false;

    //获取路径中所有图像
    std::vector<cv::String> img_files;
    cv::glob(imagepath, img_files);
    if(img_files.empty()){
        std::cout << "no image need process..." << std::endl;
        return -1;
    }

    //时间测试初始化
    double sum_duration = 0.0;
    auto start1 = GetCurrentUS();
    auto denoise_duration = 0.0;

    //模型创建和设置，所有设置在模型加载前完成
    ncnn::Net mfdnet;
    mfdnet.opt.num_threads = threads;
    mfdnet.opt.use_vulkan_compute = use_gpu ? true : false;
    mfdnet.opt.use_fp16_packed = fp16;
    mfdnet.opt.use_fp16_storage = fp16;
    mfdnet.opt.use_fp16_arithmetic = fp16;
    mfdnet.register_custom_layer("MFDNet_haar", MFDNet_haar_layer_creator); //注册自定义层

    //模型及参数加载
    if (mfdnet.load_param(model_param))
        exit(-1);
    if (mfdnet.load_model(model_bin))
        exit(-1);
    
    //保存预测结果
    cv::Mat result_img;
    cv::String result_str;
    cv::Mat model_out;
    cv::Mat ycrcb;
    cv::Mat channels[3];
    cv::Mat y_channel;
    cv::Mat cr_channel;
    cv::Mat cb_channel;
    int start_index;

    //测试推理时间
    for(int i=0; i<repeat; i++){
        cv::String img_path = img_files[i%img_files.size()];
        cv::Mat m = cv::imread(img_path);
        if (m.empty())
        {
            std::cout << "cv::imread" << img_path << "failed" << std::endl;
            return -1;
        }

        cv::cvtColor(m, ycrcb, cv::COLOR_BGR2YCrCb);
        cv::split(ycrcb, channels);
        y_channel = channels[0];
        start1 = GetCurrentUS();
        model_out = denoise_mfdnet(mfdnet, y_channel);
        denoise_duration = (GetCurrentUS() - start1) / 1000.0;
        channels[0] = model_out;
        cv::merge(channels, 3, result_img);
        cv::cvtColor(result_img, result_img, cv::COLOR_YCrCb2BGR);
        sum_duration += denoise_duration;

        // 获取图片名称，并设置保存路径
        start_index = img_path.find_last_of('/');
        result_str = cv::String(output_img_path) + "/" + img_path.substr(start_index + 1);
        cv::imwrite(result_str, result_img);
    }
    std::cout << "avg_duration:" << sum_duration/repeat << "ms" << std::endl;

    return 0;
}
