/**
 * YOLOv5n NCNN 推理（PNNX 模型），无 OpenCV 依赖
 * 用法：./yolo_detect <param> <bin> <rgb_file> [conf] [nms]
 * 输出：JSON 检测结果到 stdout
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ncnn/net.h>

const int MODEL_W = 640, MODEL_H = 640, NUM_CLASSES = 9;

const char* CLASS_NAMES[] = {
    "helmet","no-helmet","vest","no-vest",
    "mask","gloves","no-gloves","person","boots"
};

struct Detection { float x1,y1,x2,y2; int label; float prob; };

// --- 读取 Python 端预处理的 RGB 数据 ---
unsigned char* read_data(const char* path, int* pre_w, int* pre_h, int* c, int* o_w, int* o_h) {
    FILE* fp = fopen(path, "rb");
    if (!fp) exit(1);
    fread(pre_w, sizeof(int), 1, fp);
    fread(pre_h, sizeof(int), 1, fp);
    fread(c,      sizeof(int), 1, fp);
    fread(o_w,    sizeof(int), 1, fp);
    fread(o_h,    sizeof(int), 1, fp);
    size_t size = (*pre_w) * (*pre_h) * (*c);
    unsigned char* data = (unsigned char*)malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    return data;
}

// --- 预处理：构造 ncnn::Mat ---
ncnn::Mat preprocess(unsigned char* rgb, int w, int h) {
    ncnn::Mat in = ncnn::Mat::from_pixels(rgb, ncnn::Mat::PixelType::PIXEL_RGB, w, h);
    const float mean[3] = {0,0,0}, norm[3] = {1.0f/255, 1.0f/255, 1.0f/255};
    in.substract_mean_normalize(mean, norm);
    return in;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 4) { return 1; }
    float conf_thresh = (argc>=5) ? atof(argv[4]) : 0.25f;
    float nms_thresh  = (argc>=6) ? atof(argv[5]) : 0.45f;

    // 1. 加载模型
    ncnn::Net net;
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0) return 1;

    // 2. 读取 RGB 数据（含原图尺寸用于坐标反算）
    int pre_w, pre_h, c, orig_w, orig_h;
    unsigned char* rgb = read_data(argv[3], &pre_w, &pre_h, &c, &orig_w, &orig_h);

    // 3. 预处理 + 推理
    ncnn::Mat in = preprocess(rgb, pre_w, pre_h);
    free(rgb);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    // 4. 解码（PNNX 输出已是成品：坐标=像素、obj/class=已过 sigmoid）
    float* feat = (float*)out.data;
    std::vector<Detection> proposals;

    // letterbox 左上角对齐 → pad_x = pad_y = 0，直接除以 scale
    float scale = fminf((float)pre_w / orig_w, (float)pre_h / orig_h);

    for (int i = 0; i < out.h; i++) {
        float* row = feat + i * 14;
        float obj = row[4];
        int best_c = 0; float best_s = 0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (row[5+j] > best_s) { best_s = row[5+j]; best_c = j; }
        }
        float conf = obj * best_s;
        if (conf < conf_thresh) continue;

        float xc = row[0], yc = row[1], w = row[2], h = row[3];
        float x1 = (xc - w/2) / scale;
        float y1 = (yc - h/2) / scale;
        float x2 = (xc + w/2) / scale;
        float y2 = (yc + h/2) / scale;
        x1 = fmaxf(0, fminf(x1, orig_w-1));
        y1 = fmaxf(0, fminf(y1, orig_h-1));
        x2 = fmaxf(0, fminf(x2, orig_w-1));
        y2 = fmaxf(0, fminf(y2, orig_h-1));

        proposals.push_back({x1,y1,x2,y2,best_c,conf});
    }

    // 5. NMS per class → results
    std::vector<Detection> results;
    for (int j = 0; j < NUM_CLASSES; j++) {
        std::vector<Detection> cls;
        for (auto& d : proposals) if (d.label == j) cls.push_back(d);
        std::sort(cls.begin(), cls.end(), [](auto& a, auto& b){ return a.prob > b.prob; });
        std::vector<int> kept;
        for (size_t k = 0; k < cls.size(); k++) {
            bool sup = false;
            for (int m : kept) {
                float ix1 = fmaxf(cls[k].x1, cls[m].x1), iy1 = fmaxf(cls[k].y1, cls[m].y1);
                float ix2 = fminf(cls[k].x2, cls[m].x2), iy2 = fminf(cls[k].y2, cls[m].y2);
                if (ix1 >= ix2 || iy1 >= iy2) continue;
                float inter = (ix2-ix1)*(iy2-iy1);
                float ak = (cls[k].x2-cls[k].x1)*(cls[k].y2-cls[k].y1);
                float am = (cls[m].x2-cls[m].x1)*(cls[m].y2-cls[m].y1);
                if (inter/(ak+am-inter) > nms_thresh) { sup = true; break; }
            }
            if (!sup) { kept.push_back(k); results.push_back(cls[k]); }
        }
    }

    // 6. 输出 JSON
    printf("{\"detections\":[");
    for (size_t i = 0; i < results.size(); i++) {
        auto& d = results[i];
        printf("{\"x1\":%.1f,\"y1\":%.1f,\"x2\":%.1f,\"y2\":%.1f,\"label\":%d,\"name\":\"%s\",\"prob\":%.3f}",
               d.x1, d.y1, d.x2, d.y2, d.label, CLASS_NAMES[d.label], d.prob);
        if (i < results.size()-1) printf(",");
    }
    printf("],\"count\":%zu}\n", results.size());
    return 0;
}
