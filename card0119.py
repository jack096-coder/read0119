import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io

# --- 影像處理函數優化版 ---

def detect_corner_markers(img_crop_bgr):
    """辨識 A1 定位點：強化方形特徵偵測"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    # 使用自適應二值化處理背景不均
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue # 過濾過小雜訊
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4: # 方形定位點
            points = approx.reshape(4, 2).tolist()
            detected_squares.append(points)
    return detected_squares

def detect_bubbles(img_crop_bgr):
    """辨識 A2, A3 氣泡：加入圓性過濾以提高準確率"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    # 平滑化處理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用 Canny 邊緣偵測配合霍夫圓變換
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=20,    # 氣泡間的最小距離
        param1=50,
        param2=30,     # 偵測靈敏度，調高可減少誤判
        minRadius=10,   
        maxRadius=35
    )
    
    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # (center_x, center_y, radius)
            detected_circles.append((int(i[0]), int(i[1]), int(i[2])))
    return detected_circles

def draw_results_on_image(pil_image, results, region_offsets):
    """繪製結果：使用較粗的線條確保可視度"""
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() 

    # A1 定位點
    if 'A1_value' in results:
        off_x, off_y = region_offsets.get('A1', (0, 0))
        for square in results['A1_value']:
            abs_pts = (np.array(square) + [off_x, off_y]).astype(np.int32)
            cv2.polylines(img_cv, [abs_pts.reshape((-1, 1, 2))], True, (0, 0, 255), 5)

    # A2, A3 氣泡
    for key in ['A2_value', 'A3_value']:
        if key in results:
            region = key.split('_')[0]
            off_x, off_y = region_offsets.get(region, (0, 0))
            for (cx, cy, r) in results[key]:
                # 畫出氣泡的外切紅框
                cv2.rectangle(img_cv, (cx + off_x - r, cy + off_y - r), 
                              (cx + off_x + r, cy + off_y + r), (0, 0, 255), 3)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# --- Streamlit UI 邏輯 ---

st.set_page_config(page_title="OMR 答案卡區域校準系統", layout="wide")

# 初始化 Session State
for key in ['img_file', 'original_image', 'resized_image', 'scale_factor', 'zones', 'cropping_mode', 'temp_box', 'recognition_results', 'result_image']:
    if key not in st.session_state:
        if key == 'zones': st.session_state[key] = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
        else: st.session_state[key] = None

st.title("🎯 答案卡精確辨識與校準")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("1. 檔案上傳")
    uploaded_file = st.file_uploader("請上傳空白答案卡", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        if st.session_state.img_file != uploaded_file:
            st.session_state.img_file = uploaded_file
            st.session_state.original_image = Image.open(uploaded_file)
            
            # 全版縮放預覽邏輯
            display_width = 800
            orig_w, orig_h = st.session_state.original_image.size
            w_ratio = display_width / orig_w
            st.session_state.resized_image = st.session_state.original_image.resize((display_width, int(orig_h * w_ratio)), Image.LANCZOS)
            st.session_state.scale_factor = 1 / w_ratio
            
            # 重置狀態
            st.session_state.zones = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
            st.session_state.cropping_mode = None
            st.session_state.recognition_results = {}
            st.session_state.result_image = None

        st.markdown("### 2. 標示區域")
        def set_mode(mode): st.session_state.cropping_mode = mode

        for zone in ['A1', 'A2', 'A3', 'A4']:
            label = {"A1":"定位點","A2":"基本資料","A3":"選擇題","A4":"手寫區"}[zone]
            c1, c2 = st.columns([3, 1])
            is_active = st.session_state.cropping_mode == zone
            c1.button(f"標示 {zone} ({label})", on_click=set_mode, args=(zone,), 
                      type="primary" if is_active else "secondary", use_container_width=True)
            if st.session_state.zones[zone]: c2.success("OK")

        st.divider()
        if st.button("🚀 開始辨識", type="primary", use_container_width=True, disabled=not all(st.session_state.zones.values())):
            with st.spinner("執行深度掃描中..."):
                try:
                    full_cv = cv2.cvtColor(np.array(st.session_state.original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
                    results = {}
                    offsets = {}
                    scale = st.session_state.scale_factor
                    
                    for z in ['A1', 'A2', 'A3']:
                        box = st.session_state.zones[z]
                        rx, ry, rw, rh = int(box['left']*scale), int(box['top']*scale), int(box['width']*scale), int(box['height']*scale)
                        crop = full_cv[ry:ry+rh, rx:rx+rw]
                        offsets[z] = (rx, ry)
                        results[f"{z}_value"] = detect_corner_markers(crop) if z=='A1' else detect_bubbles(crop)

                    # A4 座標記錄
                    box4 = st.session_state.zones['A4']
                    results['A4_value'] = [(int(box4['left']*scale), int(box4['top']*scale)), (int((box4['left']+box4['width'])*scale), int((box4['top']+box4['height'])*scale))]

                    st.session_state.recognition_results = results
                    st.session_state.result_image = draw_results_on_image(st.session_state.original_image, results, offsets)
                    st.session_state.cropping_mode = None
                    st.success("辨識完成！")
                except Exception as e:
                    st.error(f"辨識失敗: {str(e)}")

with col_right:
    if st.session_state.resized_image:
        mode = st.session_state.cropping_mode
        if mode:
            # --- 計算 5cm 初始選取框 ---
            # A4 寬 21cm，預覽圖寬度為 W。 5cm = W * (5/21)
            img_w, img_h = st.session_state.resized_image.size
            cm5_px = int(img_w * (5 / 21))
            
            # 置中座標
            start_x = (img_w - cm5_px) // 2
            start_y = (img_h - cm5_px) // 2
            
            st.info(f"正在設定 {mode}：請調整藍框覆蓋目標區域")
            box_data = st_cropper(
                st.session_state.resized_image,
                realtime_update=True,
                box_color='#0000FF', # 藍色選取框
                aspect_ratio=None,
                default_coords=(start_x, start_y, cm5_px, cm5_px), # 5cm 置中初始大小
                return_type='box',
                key=f"cropper_{mode}"
            )
            if box_data:
                st.session_state.temp_box = box_data
            
            if st.button(f"確認儲存 {mode} 範圍", use_container_width=True):
                st.session_state.zones[mode] = st.session_state.temp_box
                st.session_state.cropping_mode = None
                st.rerun()

        elif st.session_state.result_image:
            st.image(st.session_state.result_image, caption="辨識結果預覽 (紅框標註偵測到的項目)")
        else:
            st.image(st.session_state.resized_image, caption="全版預覽")
