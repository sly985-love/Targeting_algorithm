# -*- coding: utf-8 -*-
import cv2
import math

filename_txt = "3.txt"
# 0 target; 1 hole ;2 kernel; 3 ten;
with open(filename_txt, "r+", encoding="utf-8", errors="ignore") as f:
    x = 0
    y = 0
    x_h = []
    y_h = []
    x_K = 0
    y_k = 0
    w_k = 0
    h_K = 0
    x_t = 0
    y_t = 0
    w_t = 0
    h_t = 0
    x_vv = []
    y_vv = []
    index2 = []
    for line in f:
        aa = line.split(" ")
        if (aa[0] == '0'):
            x_t = float(aa[1]) * 1000
            y_t = float(aa[2]) * 1000
            w_t = float(aa[3]) * 1000
            h_t = float(aa[4]) * 1000
        elif (aa[0] == '1'):
            x = float(aa[1]) * 1000
            x_h.append(x)
            y = float(aa[2]) * 1000
            y_h.append(y)
        elif (aa[0] == '3'):
            x_K = float(aa[1]) * 1000
            y_k = float(aa[2]) * 1000
            w_k = float(aa[3]) * 1000
            h_K = float(aa[4]) * 1000
        else:
            print("靶心获取不准暂时不处理")
    print(x_h, y_h, x_K, y_k, w_k, h_K, x_t, y_t, w_t, h_t)
    for i in range(0, len(x_h)):
        print(i, x_h[i], y_h[i], x_K, y_k, w_k, h_K, x_t, y_t, w_t, h_t)
        x_t0 = x_t - w_t / 2
        y_t0 = y_t - h_t / 2
        x_t1 = x_t + w_t / 2
        y_t1 = y_t + h_t / 2
        # 环间距
        d_r = (w_k + h_K) / 4
        print("环间距", d_r)
        # 计算弹孔到靶心的距离
        d_kh = math.sqrt((x_h[i] - x_K) * (x_h[i] - x_K) + (y_h[i] - y_k) * (y_h[i] - y_k))
        print("弹靶心距", d_kh)
        # 根据弹靶心距离与环间距的比例，计算输出精度为0.1的环值
        score_n = 11 - d_kh / d_r
        print(str("环值" + '%.1f' % (score_n)) + "环")
        # 根据靶面坐标，靶心坐标，弹孔坐标，输出相对于靶心的八方位方位信息
        score_a = "null"
        if (x_t0 < x_h[i] < x_K and y_h[i] == y_k):
            score_a = '偏左方'
        elif (x_t0 < x_h[i] < x_K and y_t0 < y_h[i] < y_k):
            score_a = "偏左上方"
        elif (x_h[i] == x_K and y_t0 < y_h[i] < y_k):
            score_a = "偏上方"
        elif (x_K < x_h[i] < x_t1 and y_t0 < y_h[i] < y_k):
            score_a = "偏右上方"
        elif (x_K < x_h[i] < x_t1 and y_h[i] == y_k):
            score_a = "偏右方"
        elif (x_t0 < x_h[i] < x_K and y_k < y_h[i] < y_t1):
            score_a = "偏左下方"
        elif (x_h[i] == x_K and y_k < y_h[i] < y_t1):
            score_a = "偏下方"
        elif (x_K < x_h[i] < x_t1 and y_k < y_h[i] < y_t1):
            score_a = "偏右下方"
        else:
            print("出界了")
        print("方向:" + str(score_a))
        x_vk = 610
        y_vK = 645
        # 计算相对距离系数
        k_x = round(float(x_h[i]) / float(x_K), 4)
        k_y = round(float(y_h[i]) / float(y_k), 4)
        print("比例系数", k_x, k_y)
        # 判断弹孔相对于靶心的四方位
        x_vh = k_x * x_vk
        y_vh = k_y * y_vK
        print("虚拟图像上靶点坐标", x_vh, y_vh)
        index = [(int(x_vh), int(y_vh))]
        path = r"scr.jpeg"
        image = cv2.imread(path)
        # 循环列表，添加多个点到图片上
        for coor in index:
            print(coor)
            cv2.circle(image, coor, 10, (0, 0, 255), -1)  # 中心坐标,半径,颜色(BGR),线宽(若为-1,即为填充颜色)
        # 保存图片
        cv2.imwrite('result' + str(int(i)) + '.jpg', image)
        result = []
        result.append({"number": i, 'score_direction': score_a, 'score_grade': score_n})
        print(result)
        print("******************************")
        x_vv.append(int(x_vh))
        y_vv.append(int(y_vh))
    resultall=[]
    for i in range(0, len(x_vv)):
        index2.append((x_vv[i], y_vv[i]))


    print(index2)
    # 循环列表，添加多个点到图片上
    for coor in index2:
        cv2.circle(image, coor, 10, (0, 0, 255), -1)  # 中心坐标,半径,颜色(BGR),线宽(若为-1,即为填充颜色)
    # 保存图片
    cv2.imwrite('result' + str(int(len(x_h))) + '.jpg', image)


# detections.append({'class': cls, 'conf': conf, 'position': xywh})

# [521.3539999999999, 505.99000000000007, 406.25, 436.71900000000005] [683.7959999999999, 541.2040000000001, 672.685, 849.074] 434.375 532.4069999999999 108.333 181.481 440.88500000000005 495.833 533.854 912.037
# 0 521.3539999999999 683.7959999999999 434.375 532.4069999999999 108.333 181.481 440.88500000000005 495.833 533.854 912.037
# 环间距 72.45349999999999
# 弹靶心距 174.59660867840472
# 环值8.6环
# 方向:偏右下方
# 比例系数 1.2002 1.2843
# 虚拟图像上靶点坐标 732.122 828.3735
# (732, 828)
# ******************************
# 1 505.99000000000007 541.2040000000001 434.375 532.4069999999999 108.333 181.481 440.88500000000005 495.833 533.854 912.037
# 环间距 72.45349999999999
# 弹靶心距 72.15327736146163
# 环值10.0环
# 方向:偏右下方
# 比例系数 1.1649 1.0165
# 虚拟图像上靶点坐标 710.589 655.6424999999999
# (710, 655)
# ******************************
# 2 406.25 672.685 434.375 532.4069999999999 108.333 181.481 440.88500000000005 495.833 533.854 912.037
# 环间距 72.45349999999999
# 弹靶心距 143.06967851015816
# 环值9.0环
# 方向:偏左下方
# 比例系数 0.9353 1.2635
# 虚拟图像上靶点坐标 570.533 814.9575000000001
# (570, 814)
# ******************************
# 3 436.71900000000005 849.074 434.375 532.4069999999999 108.333 181.481 440.88500000000005 495.833 533.854 912.037
# 环间距 72.45349999999999
# 弹靶心距 316.6756751394082
# 环值6.6环
# 方向:偏右下方
# 比例系数 1.0054 1.5948
# 虚拟图像上靶点坐标 613.2940000000001 1028.646
# (613, 1028)
# ******************************
# [(732, 828), (710, 655), (570, 814), (613, 1028)]
