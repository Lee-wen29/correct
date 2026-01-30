import cv2
import numpy as np
import os
import sys

# ================= 配置区域 =================
# 默认长宽比 (宽/高)。通常圆形基板校正后应为正圆，所以默认 1.0
DEFAULT_ASPECT_RATIO = 1.0 
# ===========================================

# 全局变量
points = []          # 最终确定的4个点
img_display = None   # 显示用图
img_undistorted = None # 去畸变后的图

# 窗口名称
window_distort = "1. Lens Correction"
window_select_mode1 = "2. Select 4 Points (Mode 1)"
window_select_mode2 = "2. Draw Rect & Fine Tune (Mode 2)"
window_result = "3. Result Preview"

# 【新增】Mode 1 的复杂状态管理
mode1_state = {
    "step_lines": [],       # 当前正在寻找的角点的辅助线列表，存 [(p1, p2), (p3, p4)]
    "drag_start": None,     # 当前正在画的线的起点
    "drag_curr": None,      # 当前正在画的线的终点
    "curr_mouse": (-1, -1), # 十字光标位置
    "intersection": None    # 计算出的交点
}

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
    "current_mouse_pos": (-1, -1)
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
    """ 阶段1：手动去畸变 (修改版：返回 k1 值) """
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
            # 修改：返回处理后的图，K1值，以及是否跳过的标志
            return undistorted, k1, False
        elif key == ord('d'): 
            cv2.destroyWindow(window_distort)
            return None, 0, True
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
#                 辅助绘图与数学功能
# ==============================================
def draw_cursor_crosshair(img, x, y):
    """ 绘制十字光标 """
    if x < 0 or y < 0: return
    h, w = img.shape[:2]
    color = (100, 255, 100)
    cv2.line(img, (0, y), (w, y), color, 1)
    cv2.line(img, (x, 0), (x, h), color, 1)

def compute_line_intersection(line1, line2):
    p1, p2 = np.array(line1[0]), np.array(line1[1])
    p3, p4 = np.array(line2[0]), np.array(line2[1])
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None # 平行
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))

# ==============================================
#          模式1：辅助线交点定位法
# ==============================================
def refresh_mode1_display():
    global img_display, img_undistorted, points, mode1_state
    img_display = img_undistorted.copy()
    h, w = img_display.shape[:2]
    
    for i, pt in enumerate(points):
        cv2.circle(img_display, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        cv2.putText(img_display, f"P{i+1}", (int(pt[0]) + 10, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if mode1_state["drag_start"] and mode1_state["drag_curr"]:
        p1 = mode1_state["drag_start"]
        p2 = mode1_state["drag_curr"]
        cv2.line(img_display, p1, p2, (255, 0, 0), 2)
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        cv2.line(img_display, p2, (p2[0]+dx*10, p2[1]+dy*10), (255, 0, 0), 1)

    for line in mode1_state["step_lines"]:
        cv2.line(img_display, line[0], line[1], (0, 255, 0), 2)
        p1, p2 = line[0], line[1]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        ext_p1 = (int(p1[0]-dx*10), int(p1[1]-dy*10))
        ext_p2 = (int(p2[0]+dx*10), int(p2[1]+dy*10))
        cv2.line(img_display, ext_p1, ext_p2, (0, 255, 0), 1)

    if len(mode1_state["step_lines"]) == 2:
        inter = compute_line_intersection(mode1_state["step_lines"][0], mode1_state["step_lines"][1])
        mode1_state["intersection"] = inter
        if inter:
            cv2.circle(img_display, inter, 6, (255, 0, 255), -1)
            cv2.circle(img_display, inter, 10, (255, 0, 255), 2)
            cv2.putText(img_display, "Intersection Found!", (inter[0]+15, inter[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    else:
        mode1_state["intersection"] = None

    info_y = 30
    if len(points) < 4:
        cv2.putText(img_display, f"Finding Point {len(points)+1}/4...", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        info_y += 30
        if len(mode1_state["step_lines"]) == 0:
            cv2.putText(img_display, ">> Drag to draw Line 1 (Edge 1)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        elif len(mode1_state["step_lines"]) == 1:
            cv2.putText(img_display, ">> Drag to draw Line 2 (Edge 2)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        elif len(mode1_state["step_lines"]) == 2:
            cv2.putText(img_display, ">> Press [Enter] to Confirm Point", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(img_display, ">> Press [R] to Redraw lines", (10, info_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cx, cy = mode1_state["curr_mouse"]
    draw_cursor_crosshair(img_display, cx, cy)

    cv2.imshow(window_select_mode1, img_display)

def mode1_mouse(event, x, y, flags, param):
    global points, mode1_state
    
    mode1_state["curr_mouse"] = (x, y)
    need_refresh = False

    if len(points) < 4:
        if len(mode1_state["step_lines"]) < 2:
            if event == cv2.EVENT_LBUTTONDOWN:
                mode1_state["drag_start"] = (x, y)
                mode1_state["drag_curr"] = (x, y)
                need_refresh = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if mode1_state["drag_start"]:
                    mode1_state["drag_curr"] = (x, y)
                    need_refresh = True
                else:
                    need_refresh = True
            elif event == cv2.EVENT_LBUTTONUP:
                if mode1_state["drag_start"]:
                    start_pt = mode1_state["drag_start"]
                    end_pt = (x, y)
                    dist = np.linalg.norm(np.array(start_pt) - np.array(end_pt))
                    if dist > 5: 
                        mode1_state["step_lines"].append((start_pt, end_pt))
                    
                    mode1_state["drag_start"] = None
                    mode1_state["drag_curr"] = None
                    need_refresh = True
        else:
            if event == cv2.EVENT_MOUSEMOVE:
                need_refresh = True

    if need_refresh:
        refresh_mode1_display()

def run_mode1():
    global points, mode1_state
    points = []
    mode1_state = {
        "step_lines": [], 
        "drag_start": None, "drag_curr": None, 
        "curr_mouse": (-1, -1), "intersection": None
    }
    
    cv2.namedWindow(window_select_mode1, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_select_mode1, mode1_mouse)
    refresh_mode1_display()
    
    print("   >>> [模式1] 辅助线交点定位法")
    
    while True:
        k = cv2.waitKey(20) & 0xFF
        if len(points) == 4:
            cv2.destroyWindow(window_select_mode1)
            return np.array(points, dtype="float32")
        
        if k == 13: 
            if mode1_state["intersection"] is not None:
                points.append(list(mode1_state["intersection"]))
                mode1_state["step_lines"] = []
                mode1_state["intersection"] = None
                refresh_mode1_display()
        elif k == ord('r'):
            if points and len(mode1_state["step_lines"]) == 0:
                points.pop()
            mode1_state["step_lines"] = []
            mode1_state["intersection"] = None
            refresh_mode1_display()
        elif k == ord('q'): sys.exit()
        elif k == ord('d'): cv2.destroyWindow(window_select_mode1); return None

# ==============================================
#                 模式2：绘制矩形 + 透视内切圆微调
# ==============================================
def get_perspective_circle_contour(pts, steps=100):
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
    
    if mode2_state["phase"] == 0:
        cv2.putText(img_display, "Step 1: Drag to draw a box around the object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if mode2_state["rect_start"] and mode2_state["rect_curr"]:
            cv2.rectangle(img_display, mode2_state["rect_start"], mode2_state["rect_curr"], (0, 255, 0), 2)
            
    elif mode2_state["phase"] == 1:
        pts = mode2_state["pts"]
        for i in range(4):
            p1 = tuple(pts[i].astype(int))
            p2 = tuple(pts[(i+1)%4].astype(int))
            cv2.line(img_display, p1, p2, (255, 200, 0), 1)
        circle_cnt = get_perspective_circle_contour(pts)
        if len(circle_cnt) > 0:
            cv2.polylines(img_display, [circle_cnt], True, (0, 255, 255), 2)
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i == mode2_state["selected_idx"] else (0, 255, 0)
            cv2.circle(img_display, tuple(p.astype(int)), 6, color, -1)
            
        cv2.putText(img_display, "Step 2: Drag corners to fit the INNER CIRCLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img_display, "L-Drag: Corner | R-Drag: Move All | [Arrows]: Fine Tune", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

    mx, my = mode2_state["current_mouse_pos"]
    draw_cursor_crosshair(img_display, mx, my)
    cv2.imshow(window_select_mode2, img_display)

def mode2_mouse(event, x, y, flags, param):
    global mode2_state
    mode2_state["current_mouse_pos"] = (x, y)
    need_refresh = False

    if mode2_state["phase"] == 0:
        if event == cv2.EVENT_LBUTTONDOWN:
            mode2_state["rect_start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
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

    elif mode2_state["phase"] == 1:
        pts = mode2_state["pts"]
        if event == cv2.EVENT_MOUSEMOVE:
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
                min_d = float('inf')
                sel = -1
                for i, p in enumerate(pts):
                    d = np.linalg.norm(p - np.array([x, y]))
                    if d < 20: 
                        if d < min_d: min_d = d; sel = i
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
    mode2_state = {
        "phase": 0,
        "rect_start": None, "rect_curr": None,
        "pts": [], "selected_idx": -1,
        "dragging": False, "moving_all": False, "last_mouse": (0,0),
        "current_mouse_pos": (-1, -1)
    }
    cv2.namedWindow(window_select_mode2, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_select_mode2, mode2_mouse)
    refresh_mode2_display()
    
    print("\n   >>> [模式2] 操作说明：")
    
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
            if updated: refresh_mode2_display()

        if key == 13: # Enter
            if mode2_state["phase"] == 1 and len(mode2_state["pts"]) == 4:
                cv2.destroyWindow(window_select_mode2)
                return np.array(mode2_state["pts"], dtype="float32")
        elif key == ord('r'): 
            mode2_state["phase"] = 0
            mode2_state["rect_start"] = None
            refresh_mode2_display()
        elif key == ord('q'): sys.exit()
        elif key == ord('d'): cv2.destroyWindow(window_select_mode2); return None


# ==============================================
#                 通用转换与批处理 (已修改)
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

    # 用于存储第一张图（模板）的参数
    template_params_set = False
    fixed_k1 = 0.0
    fixed_pts = None

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"corrected_{filename}")

        if os.path.exists(output_path): 
            print(f"[{idx+1}] 跳过已存在: {filename}"); continue
        
        print(f"\n[{idx+1}/{len(file_list)}] 处理中: {filename}")
        img_raw = cv2.imread(file_path)
        if img_raw is None: continue

        # =======================================================
        # 逻辑分支：如果还没有设置模板参数，则手动操作；否则自动处理
        # =======================================================
        if not template_params_set:
            # --- 手动模式 ---
            print("   >>> 正在进行参数设定 (此图参数将应用于后续所有图片)")
            
            # 1. 畸变校正 GUI
            # 注意：manual_undistort_gui 现在返回 (img, k1, skip)
            img_undistorted, k1_val, skip = manual_undistort_gui(img_raw)
            if skip: continue

            # 2. 模式选择
            h, w = img_undistorted.shape[:2]
            msg_img = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(msg_img, "Select Mode:", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(msg_img, "1. Auxiliary Lines (Intersect)", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(msg_img, "2. Rect & Inner Circle", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
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

            # 3. 预览与保存 (确认模板参数)
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
                print("   >>> [预览] S保存并应用到全部, C对比, D跳过当前")
                
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
                        saved = True
                        break
                    elif k == ord('c'): show_warp = not show_warp
                    elif k == ord('d'): break
                    elif k == 27: sys.exit()
                
                cv2.destroyWindow(window_result)
                
                if saved: 
                    print(f"   已保存: {output_path}")
                    # === 核心修改：保存参数，锁定后续操作 ===
                    fixed_k1 = k1_val
                    fixed_pts = pts_np
                    template_params_set = True
                    print("   >>> 参数已锁定！后续图片将自动处理...")
                    # ===================================

            except Exception as e:
                print(f"   错误: {e}")

        else:
            # =======================================================
            # 自动批处理模式 (使用保存的 fixed_k1 和 fixed_pts)
            # =======================================================
            try:
                # 1. 自动去畸变
                img_undistorted = undistort_image(img_raw, fixed_k1)
                
                # 2. 自动透视变换
                warped = four_point_transform_fixed_ratio(img_undistorted, fixed_pts, target_ratio)
                
                # 3. 自动保存
                cv2.imwrite(output_path, warped)
                print(f"   [自动处理] 已保存: {output_path}")
                
            except Exception as e:
                print(f"   [自动处理失败]: {e}")

    cv2.destroyAllWindows()
    print("\n所有任务完成。")

if __name__ == "__main__":
    # 路径配置
    INPUT_FOLDER = r"D:\\Desktop\\A" 
    OUTPUT_FOLDER = r"D:\\Desktop\\B"
    #INPUT_FOLDER = r"D:\\Desktop\\phone" 
    #OUTPUT_FOLDER = r"D:\\Desktop\\correct"
    
    target_ar = get_user_aspect_ratio()
    process_batch(INPUT_FOLDER, OUTPUT_FOLDER, target_ar)
#########固定参数,E:/Test/wood_ring/python.exe F:\Project\correct\correct_0_5.py