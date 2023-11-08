# 环值判定
# 输入yolov5得到的有效信息
# 输入靶面的左上和右下坐标，靶心坐标，弹孔坐标
import cv2
import math

#      xc        yc      w       h
# 1  521.354 683.796 9.375 15.7407
# 3  434.375 532.407 108.333 181.481
# 0  440.885 495.833 533.854 912.037
# 类别为1（弹孔），输出左上，右下坐标，计算弹孔的中心坐标
# 类别为3（10环），输出左上，右下坐标，计算十环中心坐标作为靶心，计算两边的平均值的一半作为环间距
# 类别为0（靶面），输出左上，右下坐标
# 现在有一个问题就是靶心坐标没办法直接输出了，所以我就只能先用环间距来计算靶心坐标
# 靶面的左上坐标
x_t = 440.885
y_t = 495.833
w_t = 533.854
h_t = 912.037
# x_t0 = 440.885 - 533.854 / 2
# y_t0 = 495.833 - 912.037 / 2
# # 靶面的右上坐标
# x_t1 = 440.885 + 533.854 / 2
# y_t1 = 495.833 + 912.037 / 2
# 靶心坐标
x_K = 434.375
y_k = 532.407
w_k = 108.333
h_K = 181.481
# 弹孔坐标
x_h = 521.354
y_h = 683.796

print(x_h, y_h, x_K, y_k, w_k, h_K, x_t, y_t, w_t, h_t)


# 521.354 683.796 434.375 532.407 108.333 181.481 440.885 495.833 533.854 912.037
def score(x_h, y_h, x_K, y_k, w_k, h_K, x_t, y_t, w_t, h_t):
    x_t0 = x_t - w_t / 2
    y_t0 = y_t - h_t / 2
    x_t1 = x_t + w_t / 2
    y_t1 = y_t + h_t / 2
    # 环间距
    d_r = (w_k + h_K) / 4
    print("环间距", d_r)
    # 计算弹孔到靶心的距离
    d_kh = math.sqrt((x_h - x_K) * (x_h - x_K) + (y_h - y_k) * (y_h - y_k))
    print("弹靶心距", d_kh)
    # 根据弹靶心距离与环间距的比例，计算输出精度为0.1的环值
    score_n = 11 - d_kh / d_r
    print(str("环值" + '%.1f' % (score_n)) + "环")
    # 根据靶面坐标，靶心坐标，弹孔坐标，输出相对于靶心的八方位方位信息
    score_a = "null"
    if (x_t0 < x_h < x_K and y_h == y_k):
        score_a = '偏左方'
    elif (x_t0 < x_h < x_K and y_t0 < y_h < y_k):
        score_a = "偏左上方"
    elif (x_h == x_K and y_t0 < y_h < y_k):
        score_a = "偏上方"
    elif (x_K < x_h < x_t1 and y_t0 < y_h < y_k):
        score_a = "偏右上方"
    elif (x_K < x_h < x_t1 and y_h == y_k):
        score_a = "偏右方"
    elif (x_t0 < x_h < x_K and y_k < y_h < y_t1):
        score_a = "偏左下方"
    elif (x_h == x_K and y_k < y_h < y_t1):
        score_a = "偏下方"
    elif (x_K < x_h < x_t1 and y_k < y_h < y_t1):
        score_a = "偏右下方"
    else:
        print("出界了")
    print("方向:" + str(score_a))
    # 计算弹孔在虚拟靶面上的坐标，并在虚拟图像上画点
    x_vk = 610
    y_vK = 645
    # 计算相对距离系数
    k_x = round(float(x_h) / float(x_K), 4)
    k_y = round(float(y_h) / float(y_k), 4)
    print("比例系数", k_x, k_y)
    # 判断弹孔相对于靶心的四方位
    x_vh = k_x * x_vk
    y_vh = k_y * y_vK
    print("虚拟图像上靶点坐标", x_vh, y_vh)
    index = [(int(x_vh), int(y_vh))]
    path = r"plot_vpic/scr.jpeg"
    image = cv2.imread(path)
    print(image.shape)
    # 循环列表，添加多个点到图片上
    for coor in index:
        cv2.circle(image, coor, 10, (0, 0, 255), -1)  # 中心坐标,半径,颜色(BGR),线宽(若为-1,即为填充颜色)
    # 保存图片
    cv2.imwrite(r"plot_vpic/plot.jpeg", image)
    return score_a, score_n


a, b = score(x_h, y_h, x_K, y_k, w_k, h_K, x_t, y_t, w_t, h_t)
# 521.354 683.796 434.375 532.407 108.333 181.481 440.885 495.833 533.854 912.037
# 环间距 72.45349999999999
# 弹靶心距 174.59660867840478
# 环值8.6环
# 方向:偏右下方
# 比例系数 1.2002 1.2843
# 虚拟图像上靶点坐标 732.122 828.3735
# (1104, 1200, 3)