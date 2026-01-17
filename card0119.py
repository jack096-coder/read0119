import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io

# --- 影像處理核心：高精準辨識演算法 ---

def detect_corner_markers(img_crop_bgr):
    """辨識定位點 (A1)：尋找實心方形"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            detected_squares.append(approx.reshape(4, 2).tolist())
    return detected_squares

def detect_bubbles(img_crop_bgr):
    """辨識氣泡 (A2, A3)：使用圓性過濾 (Circularity) 排除文字干擾"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_circles = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 50 or peri == 0: continue
        
        # 圓性計算公式: 4 * PI * Area / Perimeter^2
        circularity = 4 * np.pi * area / (peri * peri)
        
        # 氣泡通常圓性 > 0.7 且面積在特定範圍內
        if 0.65 < circularity < 1.2: 
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if 8 < radius < 30: # 限制氣泡半徑大小
                detected_circles.append((int(x), int(y), int(radius)))
                
    return detected_circles

def draw_results_on_image(pil_image, results, region_offsets):
    """在原圖繪製標註"""
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() 

    for key, color, thickness in [('A1_value', (0,0,255), 4), ('A2_value', (0,0,255), 2), ('A3_value', (0,0,255), 2)]:
        if key in results:
            region = key.split('_')[0]
            off_x, off_y = region_offsets.get(region, (0, 0))
            for item in results[key]:
                if key == 'A1_value':
                    pts = (np.array(item) + [off_x, off_y]).astype(np.int32)
                    cv2.polylines(img_cv, [pts], True, color, thickness)
                else:
                    cx, cy, r = item
                    cv2.rectangle(img_cv, (cx + off_x - r, cy + off_y - r), (cx + off_x + r, cy + off_y + r), color, thickness)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# --- Streamlit UI 介面 ---

st.set_page_config(page_title="答案卡辨識與校準系統", layout="wide")

# 初始化 Session 狀態
for k in ['img_file', 'original_image', 'resized_image', 'scale_factor', 'zones', 'cropping_mode', 'temp_box', 'recognition_results', 'result_image']:
    if k not in st.session_state:
        st.session_state[k] = {'A1': None, 'A2': None, 'A3': None, 'A4': None} if k == 'zones' else None

st.title("🎯 答案卡精確辨識系統")

col_left, col_right = st.columns([1.5, 2.5])

with col_left:
    st.header("1. 檔案上傳")
    uploaded_file = st.file_uploader("上傳空白答案卡圖片", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        if st.session_state.img_file != uploaded_file:
            st.session_state.img_file = uploaded_file
            st.session_state.original_image = Image.open(uploaded_file)
            
            # 縮放預覽
            display_width = 850
            orig_w, orig_h = st.session_state.original_image.size
            w_ratio = display_width / orig_w
            st.session_state.resized_image = st.session_state.original_image.resize((display_width, int(orig_h * w_ratio)), Image.LANCZOS)
            st.session_state.scale_factor = 1 / w_ratio
            st.session_state.zones = {k: None for k in st.session_state.zones}
            st.session_state.cropping_mode = None

        st.markdown("### 2. 標示區域")
        st.caption("點擊「標示」出現藍框，調整後按右側「確定」儲存。")
        
        for zone in ['A1', 'A2', 'A3', 'A4']:
            label = {"A1":"定位點","A2":"基本資料","A3":"選擇題","A4":"手寫區"}[zone]
            # --- UI 修改：將確定按鈕移至旁邊 ---
            c_btn, c_status, c_ok = st.columns([1.8, 0.4, 1])
            
            is_active = st.session_state.cropping_mode == zone
            c_btn.button(f"標示 {zone} {label}", key=f"btn_{zone}", 
                         type="primary" if is_active else "secondary", 
                         use_container_width=True, 
                         on_click=lambda z=zone: st.session_state.update({"cropping_mode": z}))
            
            if st.session_state.zones[zone]:
                c_status.markdown("✅")
            
            # 只有在當前模式下才顯示「確定」鍵
            if is_active:
                if c_ok.button("確定", key=f"ok_{zone}", type="primary", use_container_width=True):
                    st.session_state.zones[zone] = st.session_state.temp_box
                    st.session_state.cropping_mode = None
                    st.rerun()

        st.divider()
        if st.button("🚀 開始執行辨識", type="primary", use_container_width=True, disabled=not all(st.session_state.zones.values())):
            with st.spinner("辨識中..."):
                full_cv = cv2.cvtColor(np.array(st.session_state.original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
                results, offsets = {}, {}
                scale = st.session_state.scale_factor
                for z in ['A1', 'A2', 'A3']:
                    box = st.session_state.zones[z]
                    rx, ry, rw, rh = int(box['left']*scale), int(box['top']*scale), int(box['width']*scale), int(box['height']*scale)
                    crop = full_cv[ry:ry+rh, rx:rx+rw]
                    offsets[z] = (rx, ry)
                    results[f"{z}_value"] = detect_corner_markers(crop) if z=='A1' else detect_bubbles(crop)
                
                box4 = st.session_state.zones['A4']
                results['A4_value'] = [(int(box4['left']*scale), int(box4['top']*scale)), (int((box4['left']+box4['width'])*scale), int((box4['top']+box4['height'])*scale))]
                st.session_state.result_image = draw_results_on_image(st.session_state.original_image, results, offsets)
                st.session_state.recognition_results = results
                st.success("辨識完成")

with col_right:
    if st.session_state.resized_image:
        mode = st.session_state.cropping_mode
        if mode:
            # 計算 5cm 選取框 (假設 A4 寬 21cm)
            img_w, _ = st.session_state.resized_image.size
            cm5_px = int(img_w * (5 / 21))
            start_x, start_y = (img_w - cm5_px) // 2, 100
            
            st.info(f"正在設定 {mode} 範圍：請拖曳藍色邊框")
            # --- 修正選取框樣式 ---
            box_data = st_cropper(
                st.session_state.resized_image,
                realtime_update=True,
                box_color='#0000FF', # 純藍色線條
                aspect_ratio=None,
                default_coords=(start_x, start_y, cm5_px, cm5_px),
                return_type='box',
                key=f"crop_comp_{mode}"
            )
            st.session_state.temp_box = box_data
        elif st.session_state.result_image:
            st.image(st.session_state.result_image, caption="辨識結果 (圓性過濾已排除非氣泡區域)")
        else:
            st.image(st.session_state.resized_image, caption="全版預覽")
