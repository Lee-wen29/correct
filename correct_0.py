import cv2
import numpy as np
import os
import sys

# ================= 配置区域 =================
# 默认长宽比 (宽/高)。
# 正方形网格=1.0, A4纸=1.414
DEFAULT_ASPECT_RATIO = 1.0 
# ===========================================

# 全局变量
points = []            # 存储选点 [x, y]
img_raw = None         # 原始图
img_undistorted = None # 当前去畸变后的图
current_k = 0.0        # 当前畸变系数
window_main = "Main Interface"
window_result = "Result Preview"

def get_user_aspect_ratio():
    """ 启动时询问长宽比 """
    print("\n========= 参数设置 =========")
    print("请输入目标图像的【真实长宽比】 (Width / Height)")
    print("  - 正方形网格纸请输入: 1")
    print("  - A4纸横向: 1.414 | 纵向: 0.707")
    user_input = input(f"请输入 [默认 {DEFAULT_ASPECT_RATIO}]: ").strip()
    try:
        if not user_input: return DEFAULT_ASPECT_RATIO
        return float(user_input)
    except:
        return DEFAULT_ASPECT_RATIO

def undistort_image_fast(img, k):
    """ 
    快速去畸变函数 
    k: 畸变系数 (-0.5 ~ 0.5)
    """
    h, w = img.shape[:2]
    # 建立简单的内参矩阵，假设光心在图像中心
    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
    # 畸变向量 (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(img, cam_matrix, dist_coeffs)

def auto_correction_heuristic(img):
    """
    自动化畸变校正算法
    原理：网格纸应该由直线组成。我们尝试不同的K值，
    利用霍夫变换检测直线，得分最高的K值即为最优解。
    """
    print("   [Auto] 正在分析图像线条，请稍候...", end="", flush=True)
    
    # 为了速度，缩小图像进行计算
    scale = 0.5
    small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    h, w = small_img.shape[:2]
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    best_k = 0.0
    max_score = -1
    
    # 搜索范围：-0.4 到 0.4 (涵盖大多数广角和枕形畸变)
    # 步长 0.05
    search_range = np.arange(-0.40, 0.41, 0.05)
    
    for k in search_range:
        # 1. 尝试去畸变
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
        dist_coeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)
        undist = cv2.undistort(gray, cam_matrix, dist_coeffs)
        
        # 2. 边缘检测
        edges = cv2.Canny(undist, 50, 150)
        
        # 3. 霍夫直线变换 (检测长直线)
        # threshold=100, minLineLength=w/4 (只要长线), maxLineGap=10
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                minLineLength=int(w/5), maxLineGap=20)
        
        # 4. 计算得分：检测到的直线总长度
        score = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                score += length
        
        if score > max_score:
            max_score = score
            best_k = k
            
    print(f" 完成! 推荐 K={best_k:.2f}")
    return best_k

def order_points(pts):
    """ 排序: 左上, 右上, 右下, 左下 """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform_fixed(image, pts, target_ratio):
    """ 透视变换 (强制长宽比) """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 根据比例计算高度
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

def mouse_callback(event, x, y, flags, param):
    """ 鼠标选点逻辑 """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])

def nothing(x): pass

def process_batch(input_dir, output_dir, target_ratio):
    global points, img_raw, img_undistorted, current_k

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not file_list:
        print("文件夹为空。")
        return

    print(f"==========================================")
    print(f"开始处理 {len(file_list)} 张图片")
    print("【操作说明】")
    print("  1. 滑块: 实时调整畸变，让线条变直")
    print("  2. A键:  自动计算畸变 (Auto)")
    print("  3. 鼠标: 点击4个角")
    print("  4. R键:  撤销选点")
    print("  5. S键:  (选满4点后) 确认并预览结果")
    print("  6. D键:  跳过当前图")
    print(f"==========================================\n")

    cv2.namedWindow(window_main, cv2.WINDOW_NORMAL)
    # 创建滑块: 0-200 对应 k -0.5 到 0.5
    cv2.createTrackbar("Distort K", window_main, 100, 200, nothing)
    cv2.setMouseCallback(window_main, mouse_callback)

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(input_dir, filename)
        name_part, ext_part = os.path.splitext(filename)
        output_filename = f"{name_part}_new{ext_part}"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(file_list)}] 已存在，跳过: {filename}")
            continue
        
        print(f"[{idx+1}/{len(file_list)}] 处理中: {filename}")
        img_raw = cv2.imread(file_path)
        if img_raw is None: continue

        # 初始化
        points = []
        current_k = 0.0
        cv2.setTrackbarPos("Distort K", window_main, 100) # 重置滑块
        
        # 初始去畸变
        img_undistorted = img_raw.copy()
        
        step_done = False # 是否完成当前图
        skip_current = False

        while not step_done:
            # === 主循环：渲染界面 ===
            
            # 1. 获取滑块值并计算 k
            trackbar_val = cv2.getTrackbarPos("Distort K", window_main)
            new_k = (trackbar_val - 100) * 0.005
            
            # 2. 如果 K 发生变化，重新计算图像
            #    注意：为了性能，不要每一帧都 undistort，只在值变化时处理
            #    但是为了代码简单且响应及时，这里每一帧处理。
            #    如果卡顿，可加逻辑判断 if abs(new_k - current_k) > 1e-4:
            current_k = new_k
            img_undistorted = undistort_image_fast(img_raw, current_k)
            
            # 3. 绘制点和提示
            display_img = img_undistorted.copy()
            
            # 绘制已选点
            for i, pt in enumerate(points):
                cv2.circle(display_img, (pt[0], pt[1]), 5, (0, 0, 255), -1)
                cv2.putText(display_img, str(i + 1), (pt[0]+10, pt[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 画连线辅助
                if i > 0:
                    cv2.line(display_img, points[i-1], pt, (0, 255, 0), 1)
            if len(points) == 4:
                cv2.line(display_img, points[3], points[0], (0, 255, 0), 1)
                cv2.putText(display_img, "Press 'S' to Preview", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示畸变数值
            cv2.putText(display_img, f"K: {current_k:.3f}", (20, h_raw := display_img.shape[0]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow(window_main, display_img)
            
            # === 按键处理 ===
            key = cv2.waitKey(10) & 0xFF

            if key == ord('a'): # Auto Undistort
                best_k = auto_correction_heuristic(img_raw)
                # 更新滑块位置，主循环会自动更新图像
                slider_pos = int((best_k / 0.005) + 100)
                slider_pos = max(0, min(200, slider_pos))
                cv2.setTrackbarPos("Distort K", window_main, slider_pos)
            
            elif key == ord('r'): # Undo Point
                if points: points.pop()
                
            elif key == ord('d'): # Skip Image
                skip_current = True
                step_done = True
                
            elif key == ord('q'): # Quit
                cv2.destroyAllWindows()
                sys.exit()
                
            elif key == ord('s') or key == 13: # Save/Confirm (Enter or S)
                if len(points) == 4:
                    # 进入预览对比阶段
                    result_valid = preview_and_save(img_undistorted, points, target_ratio, output_path, output_filename)
                    if result_valid:
                        step_done = True # 完成，去下一张
                    else:
                        # 结果不满意，返回主循环继续编辑
                        # 点和畸变系数都保留
                        pass
                else:
                    print("   [提示] 请先选满4个点")

        if skip_current:
            print("   [跳过]...")
            
    cv2.destroyAllWindows()
    print("所有处理完成。")

def preview_and_save(img, pts, ratio, path, filename):
    """
    预览、对比、保存逻辑
    返回: True(已保存/跳过，进入下一张), False(需重做)
    """
    try:
        pts_np = np.array(pts, dtype="float32")
        warped = four_point_transform_fixed(img, pts_np, ratio)
        
        # 准备对比原图 (画框)
        compare_base = img.copy()
        for i in range(4):
            cv2.line(compare_base, pts[i], pts[(i+1)%4], (0,0,255), 2)
            
        cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
        show_warped = True
        
        print("   >>> [预览] S:保存  C:对比原图  R:返回修改  D:跳过")
        
        while True:
            if show_warped:
                cv2.imshow(window_result, warped)
                cv2.setWindowTitle(window_result, "Result (Corrected) - Press 'C'")
            else:
                cv2.imshow(window_result, compare_base)
                cv2.setWindowTitle(window_result, "Result (Original) - Press 'C'")
                
            k = cv2.waitKey(0) & 0xFF
            
            if k == ord('s'): # Save
                cv2.imwrite(path, warped)
                print(f"   [保存成功] {filename}")
                cv2.destroyWindow(window_result)
                return True
            
            elif k == ord('c'): # Compare
                show_warped = not show_warped
                
            elif k == ord('r'): # Retry
                print("   [返回] 回到编辑界面")
                cv2.destroyWindow(window_result)
                return False
            
            elif k == ord('d'): # Discard
                print("   [跳过] 不保存")
                cv2.destroyWindow(window_result)
                return True
                
            elif k == ord('q'):
                sys.exit()
                
    except Exception as e:
        print(f"   运算错误: {e}")
        return False

if __name__ == "__main__":
    # 路径配置
    INPUT_FOLDER = r"D:\\Desktop\\phone" 
    OUTPUT_FOLDER = r"D:\\Desktop\\correct"
    
    # 交互输入
    target_ar = get_user_aspect_ratio()
    
    # 运行
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER, target_ar)