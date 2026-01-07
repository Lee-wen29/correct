import cv2
import numpy as np
import os
import sys

# ================= 全局变量 =================
points = []          # 存储当前图像的坐标点
img_display = None   # 用于显示的图像（带标记）
img_raw = None       # 原始纯净图像
window_name = "Image Calibration"
# ===========================================

def order_points(pts):
    """
    (标准算法) 对四个点进行排序：左上，右上，右下，左下
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    (标准算法) 透视变换核心逻辑
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def refresh_display():
    """
    辅助函数：根据当前的 points 列表，重新在原图上绘制所有的点和序号
    用于 '撤销' 操作后的画面刷新
    """
    global img_display, img_raw, points
    # 每次刷新都拿原图覆盖，相当于擦除之前的标记
    img_display = img_raw.copy()
    
    for i, pt in enumerate(points):
        x, y = pt
        # 画红点
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        # 画序号 (i+1)
        cv2.putText(img_display, str(i + 1), (x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow(window_name, img_display)

def click_event(event, x, y, flags, param):
    """
    鼠标回调：记录点击位置
    """
    global points
    
    # 只有当点数少于4个时才允许添加
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        refresh_display() # 添加点后刷新显示

def process_batch(input_dir, output_dir):
    global img_raw, img_display, points

    if not os.path.exists(input_dir):
        print(f"错误: 输入文件夹不存在 -> {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹 -> {output_dir}")

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not file_list:
        print("文件夹为空。")
        return

    print(f"==========================================")
    print(f"开始处理 {len(file_list)} 张图片")
    print(f"==========================================")
    print("【操作按键说明】")
    print("  鼠标左键 : 选点")
    print("  R 键     : 撤销上一个选点 (Undo)")
    print("  D 键     : 跳过当前图片 (Discard)")
    print("  S 键     : 保存校正结果 (Save)")
    print("  Q 键     : 退出程序 (Quit)")
    print(f"==========================================\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    total_files = len(file_list)
    
    for idx, filename in enumerate(file_list):
        file_path = os.path.join(input_dir, filename)
        
        # 构建输出路径
        name_part, ext_part = os.path.splitext(filename)
        output_filename = f"{name_part}_new{ext_part}"
        output_path = os.path.join(output_dir, output_filename)

        # 检测是否已存在（断点续传）
        if os.path.exists(output_path):
            print(f"[{idx+1}/{total_files}] 已存在，自动跳过: {filename}")
            continue
        
        print(f"[{idx+1}/{total_files}] 正在处理: {filename}")

        img_raw = cv2.imread(file_path)
        if img_raw is None:
            continue
        
        # 初始化当前图片状态
        points = [] 
        refresh_display() # 显示初始图片

        should_skip_current_image = False # 标志位：是否跳过当前图

        # === 核心处理循环 (针对单张图) ===
        while True:
            # --- 阶段 1: 选点模式 (直到选满4个点) ---
            selection_done = False
            while not selection_done:
                key = cv2.waitKey(20) & 0xFF
                
                # [条件] 选满4个点，自动进入下一阶段
                if len(points) == 4:
                    selection_done = True
                
                # [R键] 撤销上一个点
                if key == ord('r'):
                    if len(points) > 0:
                        points.pop() # 移除最后一个
                        refresh_display()
                        print(f"   [撤销] 剩余点数: {len(points)}")
                    else:
                        print("   [提示] 没有点可以撤销了")

                # [D键] 中途跳过
                elif key == ord('d'):
                    print("   [跳过] 用户选择跳过此图。")
                    should_skip_current_image = True
                    selection_done = True # 结束选点循环，并在后面处理跳过逻辑

                # [Q键] 退出
                elif key == ord('q'):
                    print("\n>>> 退出程序。")
                    cv2.destroyAllWindows()
                    sys.exit()

            # 如果用户在选点阶段按了 'd'，跳出外层循环，处理下一张
            if should_skip_current_image:
                break

            # --- 阶段 2: 预览校正结果 ---
            try:
                pts_np = np.array(points, dtype="float32")
                warped_img = four_point_transform(img_raw, pts_np)
                
                cv2.imshow(window_name, warped_img)
                print("   >>> 预览校正结果。 [S]保存  [R]返回微调  [D]跳过不保存")
                
                # 等待确认
                valid_key_pressed = False
                while not valid_key_pressed:
                    key_result = cv2.waitKey(0) & 0xFF
                    
                    if key_result == ord('s'):
                        # 保存
                        cv2.imwrite(output_path, warped_img)
                        print(f"   [已保存] {output_filename}")
                        valid_key_pressed = True
                        should_skip_current_image = True # 利用这个标志位跳出外层 while True
                        
                    elif key_result == ord('d'):
                        # 跳过
                        print("   [跳过] 不保存结果。")
                        valid_key_pressed = True
                        should_skip_current_image = True # 利用这个标志位跳出外层 while True

                    elif key_result == ord('r'):
                        # 返回修改：保留当前点，回到阶段1
                        print("   [返回] 回到编辑模式...")
                        refresh_display() # 恢复显示带有4个点的原图
                        valid_key_pressed = True
                        # 不设置 should_skip，循环会回到 while True 开头，再次进入阶段1
                    
                    elif key_result == ord('q'):
                        cv2.destroyAllWindows()
                        sys.exit()

            except Exception as e:
                print(f"   计算错误 (点可能重合): {e}, 请按R重选")
                refresh_display()
                # 异常后回到阶段1

            # 检查是否完成了当前图的处理（保存或跳过）
            if should_skip_current_image:
                break

    print("\n==========================================")
    print("所有图像处理完毕！")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 原始图片文件夹
    INPUT_FOLDER = r"D:\\Desktop\\phone" 
    
    # 2. 结果保存文件夹
    OUTPUT_FOLDER = r"D:\\Desktop\\correct"
    # ===========================================

    process_batch(INPUT_FOLDER, OUTPUT_FOLDER)