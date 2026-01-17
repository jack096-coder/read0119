import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
import io

# --- 影像處理函數區域 (OpenCV) ---

def preprocess_image(pil_image):
    """將 PIL 圖片轉換為 OpenCV 格式並轉為灰階"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    return open_cv_image, gray

def detect_corner_markers(img_crop_bgr):
    """辨識黑色方形定位點 (A1)"""
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: # 放寬最小面積限制
            continue
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            points = approx.reshape(4, 2).tolist()
            detected_squares.append(points)
    return detected_squares

def detect_bubbles(img_crop_bgr):
    """辨識圓形氣泡 (A2, A3)"""
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=15,    # 稍微放寬距離限制
        param1=100,
        param2=25,     # 調整靈敏度
        minRadius=8,   
        maxRadius=40
    )
    
    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            detected_circles.append((int(i[0]), int(i[1]), int(i[2])))
    return detected_circles

def draw_results_on_image(pil_image, results, region_offsets):
    """將辨識結果畫在原始圖片上"""
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() 

    # 繪製 A1 方塊
    if 'A1_value' in results:
        offset_x, offset_y = region_offsets.get('A1', (0, 0))
        for square in results['A1_value']:
            abs_points = np.array(square) + [offset_x, offset_y]
            pts = abs_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img_cv, [pts], True, (0, 0, 255), 2)

    # 繪製 A2, A3 圓圈
    for region_key in ['A2_value', 'A3_value']:
        if region_key in results:
            region_name = region_key.split('_')[0]
            offset_x, offset_y = region_offsets.get(region_name, (0, 0))
            for (cx, cy, r) in results[region_key]:
                abs_cx = cx + offset_x
                abs_cy = cy + offset_y
                cv2.rectangle(img_cv, (abs_cx - r, abs_cy - r), (abs_cx + r, abs_cy + r), (0, 0, 255), 2)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# --- Streamlit 主程式 ---

st.set_page_config(page_title="答案卡辨識系統", layout="wide")

# 初始化 Session State
if 'img_file' not in st.session_state:
    st.session_state.img_file = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'zones' not in st.session_state:
    st.session_state.zones = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
if 'cropping_mode' not in st.session_state:
    st.session_state.cropping_mode = None
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = {}
if 'result_image' not in st.session_state:
    st.session_state.result_image = None

st.title("📝 答案卡全版標示與辨識")

col_left, col_right = st.columns([1, 2])

# --- 左側欄位 ---
with col_left:
    st.header("1. 上傳與設定")
    uploaded_file = st.file_uploader("請上傳空白答案卡 (jpg, png)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # 當上傳新檔案時重置
        if st.session_state.img_file != uploaded_file:
            st.session_state.img_file = uploaded_file
            st.session_state.original_image = Image.open(uploaded_file)
            st.session_state.zones = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
            st.session_state.cropping_mode = None
            st.session_state.recognition_results = {}
            st.session_state.result_image = None
            
        st.success("圖片已載入，請依序標示區域。")
        
        def set_crop_mode(mode):
            st.session_state.cropping_mode = mode

        # 按鈕區
        st.markdown("### 2. 標示區域")
        st.caption("點擊按鈕後，請在右側圖片畫出對應範圍")

        b1, s1 = st.columns([3, 1])
        b1.button("標示 A1 (定位點)", on_click=set_crop_mode, args=('A1',), use_container_width=True)
        if st.session_state.zones['A1']: s1.success("OK")
        
        b2, s2 = st.columns([3, 1])
        b2.button("標示 A2 (基本資料)", on_click=set_crop_mode, args=('A2',), use_container_width=True)
        if st.session_state.zones['A2']: s2.success("OK")

        b3, s3 = st.columns([3, 1])
        b3.button("標示 A3 (選擇題)", on_click=set_crop_mode, args=('A3',), use_container_width=True)
        if st.session_state.zones['A3']: s3.success("OK")
        
        b4, s4 = st.columns([3, 1])
        b4.button("標示 A4 (手寫區)", on_click=set_crop_mode, args=('A4',), use_container_width=True)
        if st.session_state.zones['A4']: s4.success("OK")

        st.markdown("---")
        
        # 辨識按鈕
        all_marked = all(st.session_state.zones.values())
        if st.button("開始辨識", disabled=not all_marked, type="primary", use_container_width=True):
            if st.session_state.original_image:
                with st.spinner("辨識中..."):
                    try:
                        results = {}
                        region_offsets = {}
                        # 使用 BGR 格式進行 OpenCV 處理
                        full_img_cv = cv2.cvtColor(np.array(st.session_state.original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
                        
                        # 處理各個區域
                        # A1
                        box = st.session_state.zones['A1']
                        crop = full_img_cv[box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
                        results['A1_value'] = detect_corner_markers(crop)
                        region_offsets['A1'] = (box['left'], box['top'])

                        # A2
                        box = st.session_state.zones['A2']
                        crop = full_img_cv[box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
                        results['A2_value'] = detect_bubbles(crop)
                        region_offsets['A2'] = (box['left'], box['top'])

                        # A3
                        box = st.session_state.zones['A3']
                        crop = full_img_cv[box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
                        results['A3_value'] = detect_bubbles(crop)
                        region_offsets['A3'] = (box['left'], box['top'])

                        # A4 (座標)
                        box = st.session_state.zones['A4']
                        results['A4_value'] = [
                            (box['left'], box['top']),
                            (box['left'] + box['width'], box['top'] + box['height'])
                        ]

                        st.session_state.recognition_results = results
                        st.session_state.result_image = draw_results_on_image(st.session_state.original_image, results, region_offsets)
                        st.session_state.cropping_mode = None # 結束標示模式
                        st.success("辨識完成！")
                        
                    except Exception as e:
                        st.error(f"錯誤: {e}")

        # 下載按鈕
        if st.session_state.recognition_results:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 簡單範例：匯出 A2 氣泡
                a2_data = [{'ID': i+1, 'X': c[0], 'Y': c[1], 'R': c[2]} for i, c in enumerate(st.session_state.recognition_results.get('A2_value', []))]
                pd.DataFrame(a2_data).to_excel(writer, sheet_name='A2_Data', index=False)
                # 這裡可以依需求加入 A1, A3, A4 的 sheet
            output.seek(0)
            st.download_button("下載 Excel", data=output, file_name="results.xlsx")

# --- 右側欄位：全版顯示與操作 ---
with col_right:
    if st.session_state.original_image is None:
        st.info("👈 請先從左側上傳圖片")
    else:
        current_mode = st.session_state.cropping_mode
        
        # 情況 1: 正在標示模式中 (顯示 Cropper)
        if current_mode in ['A1', 'A2', 'A3', 'A4']:
            st.warning(f"正在標示：{current_mode} (請在圖上畫框)")
            
            # 取得該區域目前的設定值 (如果有的話)
            default_box = st.session_state.zones.get(current_mode)
            
            # --- [重要修正] 安全地設定預設座標 ---
            default_coords = None
            if default_box and isinstance(default_box, dict) and 'left' in default_box:
                default_coords = (
                    default_box['left'],
                    default_box['top'],
                    default_box['width'],
                    default_box['height']
                )
            
            # 呼叫 Cropper，鎖定顯示原圖
            cropped_box = st_cropper(
                st.session_state.original_image, # 始終使用整張原圖
                realtime_update=True,
                box_color='#0000FF',
                aspect_ratio=None,
                default_coords=default_coords,
                key=f"cropper_{current_mode}" 
            )
            
            if cropped_box:
                st.session_state.zones[current_mode] = cropped_box

        # 情況 2: 已有辨識結果 (顯示紅框結果圖)
        elif st.session_state.result_image is not None:
            st.image(st.session_state.result_image, caption="辨識結果", use_container_width=True)
            
        # 情況 3: 預設狀態 (顯示整張原圖)
        else:
            st.image(st.session_state.original_image, caption="原始答案卡", use_container_width=True)
