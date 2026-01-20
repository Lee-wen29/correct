import cv2
import numpy as np
import os
import sys

# ================= 配置区域 =================
# 默认长宽比 (宽/高)。通常圆形基板校正后应为正圆，所以默认 1.0
DEFAULT_ASPECT_RATIO = 1.0 
# ===========================================

# 全局变量
points = []          # 模式1用的点
# 【新增】Mode 1 鼠标当前位置记录，用于画十字线
mode1_curr_pos = (-1, -1) 

img_display = None   # 显示用图
img_undistorted = None # 去畸变后的图

# 窗口名称
window_distort = "1. Lens Correction"
window_select_mode1 = "2. Select 4 Points (Mode 1)"
window_select_mode2 = "2. Draw Rect & Fine Tune (Mode 2)"
window_result = "3. Result Preview"

# --- 模式2 全局变量 ---
mode2_state = {
    "phase": 0,          # 0: 等待绘制初始矩形, 1: 顶点编辑模式
    "rect_start": None,  # 初始矩形起点
    "rect_curr": None,   # 初始矩形当前点
    "pts": [],           # 4个顶点坐标 [TL, TR, BR, BL]
    "selected_idx": -1,  # 当前选中的顶点索引 (-1表示无)
    "dragging": False,   # 是否正在拖拽顶点
    "moving_all": False, # 是否正在整体移动
    "last_mouse": (0,0), # 上次鼠标位置 (用于整体移动)
    "current_mouse_pos": (-1, -1) # 【新增】记录当前鼠标位置用于画十字线
}

def get_user_aspect_ratio():
    print("\n========= 参数设置 =========")
    print("请输入目标图像校正后的【真实长宽比】 (Width / Height)")
    print("  - 目标是圆形或正方形请输入: 1")
    user_input = input(f"请输入 [默认 {DEFAULT_ASPECT_RATIO}]: ").strip()
    try:
        if not user_input: return DEFAULT_ASPECT_RATIO
        return float(user_input)
    except ValueError:
        return DEFAULT_ASPECT_RATIO

def undistort_image(img, k_value):
    h, w = img.shape[:2]
    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
    dist_coeffs = np.array([k_value, 0, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(img, cam_matrix, dist_coeffs)

def manual_undistort_gui(img_origin):
    """ 阶段1：手动去畸变 """
    def nothing(x): pass
    cv2.namedWindow(window_distort, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("K1 (Distort)", window_distort, 100, 200, nothing)
    
    print(f"   >>> [步骤1] 去畸变。Enter确认, D跳过。")

    while True:
        val = cv2.getTrackbarPos("K1 (Distort)", window_distort)
        k1 = (val - 100) * 0.005
        undistorted = undistort_image(img_origin, k1)
        
        display = undistorted.copy()
        cv2.putText(display, f"K1: {k1:.3f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_distort, display)

        key = cv2.waitKey(10) & 0xFF
        if key == 13: # Enter
            cv2.destroyWindow(window_distort)
            return undistorted, False
        elif key == ord('d'): 
            cv2.destroyWindow(window_distort)
            return None, True
        elif key == ord('q'): sys.exit()

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

# ==============================================
#                 【新增】辅助绘图功能
# ==============================================
def draw_cursor_crosshair(img, x, y):
    """ 在图像上绘制贯穿全图的十字准星 """
    if x < 0 or y < 0: return
    h, w = img.shape[:2]
    # 颜色设为浅绿色 (100, 255, 100)，线宽1，使用虚线效果不太好实现，实线更清晰
    color = (100, 255, 100)
    # 绘制水平线
    cv2.line(img, (0, y), (w, y), color, 1)
    # 绘制垂直线
    cv2.line(img, (x, 0), (x, h), color, 1)

# ==============================================
#                 模式1：传统四点选择
# ==============================================
def refresh_mode1_display():
    global img_display, img_undistorted, points, mode1_curr_pos
    img_display = img_undistorted.copy()
    
    # 1. 绘制已选点
    for i, pt in enumerate(points):
        cv2.circle(img_display, (pt[0], pt[1]), 5, (0, 0, 255), -1)
        cv2.putText(img_display, str(i + 1), (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 【新增】2. 绘制十字光标
    draw_cursor_crosshair(img_display, mode1_curr_pos[0], mode1_curr_pos[1])

    cv2.imshow(window_select_mode1, img_display)

def mode1_mouse(event, x, y, flags, param):
    global points, mode1_curr_pos
    
    # 【修改】始终记录鼠标当前位置，用于画十字线
    mode1_curr_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
    
    # 【修改】移动或点击都刷新
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
        refresh_mode1_display()

def run_mode1():
    global points, mode1_curr_pos
    points = []
    mode1_curr_pos = (-1, -1) # 重置光标位置
    cv2.namedWindow(window_select_mode1, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_select_mode1, mode1_mouse)
    refresh_mode1_display()
    
    print("   >>> [模式1] 依次点击4个角点")
    while True:
        k = cv2.waitKey(20) & 0xFF
        if len(points) == 4:
            cv2.destroyWindow(window_select_mode1)
            return np.array(points, dtype="float32")
        if k == ord('r') and points: 
            points.pop() 
            refresh_mode1_display()
        elif k == ord('q'): sys.exit()
        elif k == ord('d'): cv2.destroyWindow(window_select_mode1); return None

# ==============================================
#                 模式2：绘制矩形 + 透视内切圆微调
# ==============================================
def get_perspective_circle_contour(pts, steps=100):
    """ 
    计算给定4个顶点(透视平面)下的内切圆轮廓 
    """
    if len(pts) != 4: return []
    src_rect = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float32)
    dst_rect = np.array(pts, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    
    t = np.linspace(0, 2*np.pi, steps)
    circle_x = 0.5 + 0.5 * np.cos(t)
    circle_y = 0.5 + 0.5 * np.sin(t)
    ones = np.ones_like(t)
    src_circle_pts = np.stack([circle_x, circle_y, ones], axis=0)
    dst_circle_pts_h = M @ src_circle_pts
    dst_circle_pts = dst_circle_pts_h[:2, :] / dst_circle_pts_h[2, :]
    return dst_circle_pts.T.astype(np.int32)

def refresh_mode2_display():
    global img_display, img_undistorted, mode2_state
    img_display = img_undistorted.copy()
    
    # --- 阶段 0: 绘制初始矩形 ---
    if mode2_state["phase"] == 0:
        cv2.putText(img_display, "Step 1: Drag to draw a box around the object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if mode2_state["rect_start"] and mode2_state["rect_curr"]:
            cv2.rectangle(img_display, mode2_state["rect_start"], mode2_state["rect_curr"], (0, 255, 0), 2)
            
    # --- 阶段 1: 顶点编辑 (核心) ---
    elif mode2_state["phase"] == 1:
        pts = mode2_state["pts"]
        
        # 1. 绘制外部四边形框
        for i in range(4):
            p1 = tuple(pts[i].astype(int))
            p2 = tuple(pts[(i+1)%4].astype(int))
            cv2.line(img_display, p1, p2, (255, 200, 0), 1)
            
        # 2. 绘制透视内切圆
        circle_cnt = get_perspective_circle_contour(pts)
        if len(circle_cnt) > 0:
            cv2.polylines(img_display, [circle_cnt], True, (0, 255, 255), 2)

        # 3. 绘制4个顶点控制柄
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i == mode2_state["selected_idx"] else (0, 255, 0)
            cv2.circle(img_display, tuple(p.astype(int)), 6, color, -1)
            
        cv2.putText(img_display, "Step 2: Drag corners to fit the INNER CIRCLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img_display, "L-Drag: Corner | R-Drag: Move All | [Arrows]: Fine Tune", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

    # 【新增】绘制十字光标 (在所有图层最上方)
    mx, my = mode2_state["current_mouse_pos"]
    draw_cursor_crosshair(img_display, mx, my)

    cv2.imshow(window_select_mode2, img_display)

def mode2_mouse(event, x, y, flags, param):
    global mode2_state
    
    # 【修改】始终更新鼠标位置并刷新，以便显示十字线
    mode2_state["current_mouse_pos"] = (x, y)
    need_refresh = False

    # === 阶段 0: 初始拉框 ===
    if mode2_state["phase"] == 0:
        if event == cv2.EVENT_LBUTTONDOWN:
            mode2_state["rect_start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            # 【修改】这里原本只在有rect_start时更新，现在为了十字线始终更新
            if mode2_state["rect_start"]:
                mode2_state["rect_curr"] = (x, y)
            need_refresh = True
        elif event == cv2.EVENT_LBUTTONUP and mode2_state["rect_start"]:
            x1, y1 = mode2_state["rect_start"]
            x2, y = x, y
            lx, rx = sorted([x1, x2])
            ty, by = sorted([y1, y])
            mode2_state["pts"] = [np.array([lx, ty]), np.array([rx, ty]), 
                                  np.array([rx, by]), np.array([lx, by])]
            mode2_state["phase"] = 1
            need_refresh = True

    # === 阶段 1: 顶点精调 ===
    elif mode2_state["phase"] == 1:
        pts = mode2_state["pts"]
        
        if event == cv2.EVENT_MOUSEMOVE:
            # 无论是否拖拽，都要刷新以显示十字线和高亮
            need_refresh = True 
            
            if mode2_state["dragging"] and mode2_state["selected_idx"] != -1:
                mode2_state["pts"][mode2_state["selected_idx"]] = np.array([x, y])
            elif mode2_state["moving_all"]:
                dx = x - mode2_state["last_mouse"][0]
                dy = y - mode2_state["last_mouse"][1]
                for i in range(4):
                    mode2_state["pts"][i] += np.array([dx, dy])
                mode2_state["last_mouse"] = (x, y)
            else:
                # 悬停高亮
                min_d = float('inf')
                sel = -1
                for i, p in enumerate(pts):
                    d = np.linalg.norm(p - np.array([x, y]))
                    if d < 20: 
                        if d < min_d:
                            min_d = d
                            sel = i
                if sel != mode2_state["selected_idx"]:
                    mode2_state["selected_idx"] = sel

        elif event == cv2.EVENT_LBUTTONDOWN:
            if mode2_state["selected_idx"] != -1:
                mode2_state["dragging"] = True
        
        elif event == cv2.EVENT_LBUTTONUP:
            mode2_state["dragging"] = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            mode2_state["moving_all"] = True
            mode2_state["last_mouse"] = (x, y)
            
        elif event == cv2.EVENT_RBUTTONUP:
            mode2_state["moving_all"] = False

    if need_refresh:
        refresh_mode2_display()

def run_mode2():
    global mode2_state
    # 初始化状态
    mode2_state = {
        "phase": 0,
        "rect_start": None, "rect_curr": None,
        "pts": [], "selected_idx": -1,
        "dragging": False, "moving_all": False, "last_mouse": (0,0),
        "current_mouse_pos": (-1, -1) # 【新增】
    }
    
    cv2.namedWindow(window_select_mode2, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_select_mode2, mode2_mouse)
    refresh_mode2_display()
    
    print("\n   >>> [模式2] 操作说明：")
    print("       1. [鼠标左键拖拽] 绘制一个矩形框住目标。")
    print("       2. 松开后进入【精调模式】，矩形变为4个控制点。")
    print("       3. 拖动4个角，使中间的【内切圆】完美贴合基板边缘。")
    print("       4. 选中某个点后，可用【方向键】微调。Enter完成。")
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        
        if mode2_state["phase"] == 1 and mode2_state["selected_idx"] != -1:
            idx = mode2_state["selected_idx"]
            updated = False
            step = 1
            if key == ord('w') or key == ord('i'): 
                mode2_state["pts"][idx][1] -= step; updated = True
            elif key == ord('s') or key == ord('k'):
                mode2_state["pts"][idx][1] += step; updated = True
            elif key == ord('a') or key == ord('j'):
                mode2_state["pts"][idx][0] -= step; updated = True
            elif key == ord('d') or key == ord('l'):
                mode2_state["pts"][idx][0] += step; updated = True
            
            if updated:
                refresh_mode2_display()

        if key == 13: # Enter
            if mode2_state["phase"] == 1 and len(mode2_state["pts"]) == 4:
                cv2.destroyWindow(window_select_mode2)
                return np.array(mode2_state["pts"], dtype="float32")
            else:
                print("   [提示] 请先绘制并调整好矩形。")
        elif key == ord('r'): # Reset
            mode2_state["phase"] = 0
            mode2_state["rect_start"] = None
            refresh_mode2_display()
        elif key == ord('q'): sys.exit()
        elif key == ord('d'): cv2.destroyWindow(window_select_mode2); return None


# ==============================================
#                 通用转换与批处理 (未修改)
# ==============================================
def four_point_transform_fixed_ratio(image, pts, target_ratio):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

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
    global img_undistorted

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not file_list: print("文件夹为空。"); return

    print(f"=== 开始处理 {len(file_list)} 张图片 ===")

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"corrected_{filename}")

        if os.path.exists(output_path): 
            print(f"[{idx+1}] 跳过已存在: {filename}"); continue
        
        print(f"\n[{idx+1}/{len(file_list)}] 处理中: {filename}")
        img_raw = cv2.imread(file_path)
        if img_raw is None: continue

        # 1. 畸变校正
        img_undistorted, skip = manual_undistort_gui(img_raw)
        if skip: continue

        # 2. 模式选择
        h, w = img_undistorted.shape[:2]
        msg_img = np.zeros((200, 500, 3), dtype=np.uint8)
        cv2.putText(msg_img, "Select Mode:", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(msg_img, "1. Four Points Click", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(msg_img, "2. Draw Rect & Fine Tune", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.namedWindow("Mode Select", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Mode Select", msg_img)
        
        mode = 0
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord('1'): mode=1; break
            if k == ord('2'): mode=2; break
            if k == ord('q'): sys.exit()
        cv2.destroyWindow("Mode Select")

        # 执行对应模式
        if mode == 1:
            pts_np = run_mode1()
        else:
            pts_np = run_mode2()

        if pts_np is None: continue

        # 3. 预览与保存
        try:
            warped = four_point_transform_fixed_ratio(img_undistorted, pts_np, target_ratio)
            
            comp = img_undistorted.copy()
            pts_int = pts_np.astype(int)
            for i in range(4):
                cv2.line(comp, tuple(pts_int[i]), tuple(pts_int[(i+1)%4]), (0,255,0), 2)
            
            if mode == 2:
                circ_cnt = get_perspective_circle_contour(pts_np)
                cv2.polylines(comp, [circ_cnt], True, (0,255,255), 1)

            cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)
            show_warp = True
            print("   >>> [预览] S保存, C对比, D跳过")
            
            saved = False
            while True:
                if show_warp:
                    cv2.imshow(window_result, warped)
                    cv2.setWindowTitle(window_result, "3. Result (Corrected) - Press C")
                else:
                    cv2.imshow(window_result, comp)
                    cv2.setWindowTitle(window_result, "3. Result (Original) - Press C")

                k = cv2.waitKey(0) & 0xFF
                if k == ord('s'): 
                    cv2.imwrite(output_path, warped)
                    saved = True; break
                elif k == ord('c'): show_warp = not show_warp
                elif k == ord('d'): break
                elif k == 27: sys.exit()
            
            cv2.destroyWindow(window_result)
            if saved: print(f"   已保存: {output_path}")

        except Exception as e:
            print(f"   错误: {e}")

    cv2.destroyAllWindows()
    print("\n所有任务完成。")

if __name__ == "__main__":
    # 路径配置
    INPUT_FOLDER = r"D:\\Desktop\\phone" 
    OUTPUT_FOLDER = r"D:\\Desktop\\correct"
    
    target_ar = get_user_aspect_ratio()
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER, target_ar)

###############增加十字光标指示