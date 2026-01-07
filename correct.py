import cv2
import numpy as np
import os
import sys

# ================= 配置区域 =================
DEFAULT_ASPECT_RATIO = 1.0 #长宽比
# ===========================================

# 全局变量
points = []          # 存储选点
img_display = None   # 显示用图
img_undistorted = None # 去畸变后的图
window_distort = "1. Lens Correction (Enter to Confirm)"
window_select = "2. Select 4 Points"
window_result = "3. Result Preview"

def get_user_aspect_ratio():
    """
    启动时询问用户期望的真实物理长宽比
    """
    print("\n========= 参数设置 =========")
    print("请输入目标图像的【真实长宽比】 (Width / Height)")
    print("  - 正方形网格纸请输入: 1")
    print("  - 只需要看清晰度不关心比例，直接回车 (默认1.0)")
    user_input = input(f"请输入 [默认 {DEFAULT_ASPECT_RATIO}]: ").strip()
    try:
        if not user_input:
            return DEFAULT_ASPECT_RATIO
        ratio = float(user_input)
        if ratio <= 0:
            print("比例必须大于0，使用默认值。")
            return DEFAULT_ASPECT_RATIO
        return ratio
    except ValueError:
        print("输入无效，使用默认值。")
        return DEFAULT_ASPECT_RATIO

def undistort_image(img, k_value):
    """
    简单的径向畸变矫正函数
    k_value: 畸变系数，由滑块提供
    """
    h, w = img.shape[:2]
    # 构造一个假定的相机内参矩阵 (假设光心在图像中心)
    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
    # 畸变系数向量 (k1, k2, p1, p2, k3)
    # 这里我们只调节 k1，通常足以应对大部分简单畸变
    dist_coeffs = np.array([k_value, 0, 0, 0, 0], dtype=np.float32)
    
    # 使用 OpenCV 进行矫正
    return cv2.undistort(img, cam_matrix, dist_coeffs)

def manual_undistort_gui(img_origin):
    """
    阶段1：手动去畸变 GUI
    返回：去畸变后的图像, 是否跳过标志
    """
    def nothing(x): pass

    cv2.namedWindow(window_distort, cv2.WINDOW_NORMAL)
    # 创建滑块：范围从 -100 到 100，初始值 0 (映射到 -0.5 到 0.5 的系数)
    cv2.createTrackbar("K1 (Distort)", window_distort, 100, 200, nothing)
    
    current_img = img_origin.copy()
    
    print(f"   >>> [步骤1] 调节滑块使线条变直。按 [Enter] 确认，按 [D] 跳过此图。")

    while True:
        # 获取滑块值并归一化
        val = cv2.getTrackbarPos("K1 (Distort)", window_distort)
        k1 = (val - 100) * 0.005 # 将 0-200 映射为 -0.5 到 0.5

        # 实时计算畸变 (为了流畅度，大图可以考虑缩放显示，这里保持原图以求精准)
        undistorted = undistort_image(img_origin, k1)
        
        # 显示提示
        display = undistorted.copy()
        cv2.putText(display, f"K1: {k1:.3f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_distort, display)

        key = cv2.waitKey(10) & 0xFF
        
        if key == 13: # Enter 键
            cv2.destroyWindow(window_distort)
            return undistorted, False
        
        elif key == ord('d'): # Skip
            cv2.destroyWindow(window_distort)
            return None, True
            
        elif key == ord('q'): # Quit
            cv2.destroyAllWindows()
            sys.exit()

def order_points(pts):
    """ 对4个点排序: 左上, 右上, 右下, 左下 """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def refresh_select_display():
    """ 刷新选点界面的显示 """
    global img_display, img_undistorted, points
    img_display = img_undistorted.copy()
    for i, pt in enumerate(points):
        x, y = pt
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img_display, str(i + 1), (x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow(window_select, img_display)

def select_points_click(event, x, y, flags, param):
    """ 鼠标回调 """
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        refresh_select_display()

def four_point_transform_fixed_ratio(image, pts, target_ratio):
    """
    带强制长宽比的透视变换
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算最长边作为基准宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 根据用户设定的比例，强制计算高度
    # Ratio = Width / Height  =>  Height = Width / Ratio
    dst_width = maxWidth
    dst_height = int(dst_width / target_ratio)

    dst = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (dst_width, dst_height))
    return warped

def process_batch(input_dir, output_dir, target_ratio):
    global points, img_undistorted, img_display

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not file_list:
        print("文件夹为空。")
        return

    print(f"==========================================")
    print(f"开始处理 {len(file_list)} 张图片 | 锁定长宽比: {target_ratio}")
    print(f"==========================================\n")

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(input_dir, filename)
        name_part, ext_part = os.path.splitext(filename)
        output_filename = f"{name_part}_new{ext_part}"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(file_list)}] 跳过已存在: {filename}")
            continue
        
        print(f"[{idx+1}/{len(file_list)}] 处理中: {filename}")
        img_raw = cv2.imread(file_path)
        if img_raw is None: continue

        # --- 步骤 1: 畸变矫正 ---
        img_undistorted, skip_flag = manual_undistort_gui(img_raw)
        if skip_flag:
            print("   [跳过] 用户选择跳过。")
            continue

        # --- 步骤 2: 选点循环 ---
        points = []
        cv2.namedWindow(window_select, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_select, select_points_click)
        refresh_select_display()
        
        selection_done = False
        should_skip_img = False

        while not selection_done:
            key = cv2.waitKey(20) & 0xFF
            if len(points) == 4: selection_done = True
            
            if key == ord('r'): # 撤销点
                if points: points.pop(); refresh_select_display()
            elif key == ord('d'): # 跳过
                should_skip_img = True; selection_done = True
            elif key == ord('q'): sys.exit()

        cv2.destroyWindow(window_select)
        if should_skip_img: continue

        # 步骤 3: 预览与保存 加含一键对比) 
        try:
            pts_np = np.array(points, dtype="float32")
            # 使用带比例强制的变换
            warped_img = four_point_transform_fixed_ratio(img_undistorted, pts_np, target_ratio)
            
            # 准备对比用的原图 (带红点标记的)
            comparison_base = img_undistorted.copy()
            for i, pt in enumerate(points):
                cv2.circle(comparison_base, (pt[0], pt[1]), 8, (0, 0, 255), -1)
                cv2.line(comparison_base, points[i], points[(i+1)%4], (0,255,0), 2)
            
            # 缩放 marked 原图以适应屏幕显示 (如果原图太大)
            # 这里不做缩放，仅在显示窗口中利用 WINDOW_NORMAL 缩放
            
            is_showing_warped = True
            print("   >>> [预览] 按 [S]保存, [C]按住/切换对比原图, [R]重选, [D]跳过")

            cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
            
            while True:
                # 根据状态显示不同图像
                if is_showing_warped:
                    cv2.imshow(window_result, warped_img)
                    cv2.setWindowTitle(window_result, "3. Result (Corrected) - Press 'C' to Compare")
                else:
                    cv2.imshow(window_result, comparison_base)
                    cv2.setWindowTitle(window_result, "3. Result (Original) - Press 'C' to Corrected")

                key = cv2.waitKey(0) & 0xFF

                if key == ord('s'): # 保存
                    cv2.imwrite(output_path, warped_img)
                    print(f"   [已保存] {output_filename}")
                    break
                elif key == ord('c'): # 切换对比
                    is_showing_warped = not is_showing_warped
                elif key == ord('r'): # 重做 (回到步骤2，保留点)
                    print("   [提示] 重选模式：点将保留，进入编辑界面...")
                    
                    # 重新进入简易编辑模式
                    cv2.namedWindow(window_select, cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(window_select, select_points_click)
                    refresh_select_display()
                    
                    # 小循环：再次选点
                    reselect_done = False
                    while not reselect_done:
                        k2 = cv2.waitKey(20) & 0xFF
                        if len(points) == 4: reselect_done = True
                        if k2 == ord('r') and points: points.pop(); refresh_select_display()
                        if k2 == ord('q'): sys.exit()
                    
                    cv2.destroyWindow(window_select)
                    # 重新计算 warped_img
                    pts_np = np.array(points, dtype="float32")
                    warped_img = four_point_transform_fixed_ratio(img_undistorted, pts_np, target_ratio)
                    # 更新对比图
                    comparison_base = img_undistorted.copy()
                    for i, pt in enumerate(points):
                        cv2.circle(comparison_base, (pt[0], pt[1]), 8, (0, 0, 255), -1)
                        cv2.line(comparison_base, points[i], points[(i+1)%4], (0,255,0), 2)
                    is_showing_warped = True
                    # 循环继续，重新进入预览

                elif key == ord('d'): # 跳过
                    print("   [跳过] 不保存。")
                    break
                elif key == ord('q'): 
                    sys.exit()
            
            cv2.destroyWindow(window_result)

        except Exception as e:
            print(f"   错误: {e}")
            continue

    cv2.destroyAllWindows()
    print("\n所有任务完成。")

if __name__ == "__main__":
    # 1. 路径设置
    INPUT_FOLDER = r"D:\\Desktop\\phone" 
    OUTPUT_FOLDER = r"D:\\Desktop\\correct"
    
    # 2. 获取用户设定的比例
    target_ar = get_user_aspect_ratio()


    # 3. 开始处理
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER, target_ar)