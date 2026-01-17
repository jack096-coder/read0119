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
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    return open_cv_image, gray

def detect_corner_markers(img_crop_bgr):
    """
    在給定的裁切區域中辨識黑色方形定位點 (A1)
    回傳: 每個定位點的 4 個角座標列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    # 二值化處理，找黑色區域 (閾值可能需要調整)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_squares = []
    for cnt in contours:
        # 計算輪廓面積，過濾掉太小的雜訊
        area = cv2.contourArea(cnt)
        if area < 100: # 最小面積閾值，依實際圖片調整
            continue
            
        # 近似多邊形
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 如果近似結果有 4 個頂點，且接近正方形 (可加入長寬比判斷)
        if len(approx) == 4:
            # 取得這 4 個點的座標並轉為列表格式
            points = approx.reshape(4, 2).tolist()
            detected_squares.append(points)
            
    return detected_squares

def detect_bubbles(img_crop_bgr):
    """
    在給定的裁切區域中辨識圓形氣泡 (A2, A3)
    使用霍夫圓形變換 (Hough Circle Transform)
    回傳: 圓心與半徑列表 [(x, y, r), ...]
    """
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    # 高斯模糊以降噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # --- 關鍵參數調整區 ---
    # dp: 累加器解析度與影像解析度的反比。1 表示相同解析度。
    # minDist: 探測到的圓心之間的最小距離。太小會導致多個相鄰的圓被偵測到。
    # param1: Canny 邊緣檢測的高閾值。
    # param2: 累加器閾值。越小越容易偵測到圓，但也越多誤報。
    # minRadius/maxRadius: 預期的圓形半徑範圍。非常重要！
    rows = gray.shape[0]
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=rows / 20, # 依據氣泡密度調整
        param1=100,
        param2=30,   # 此值越低越敏感，需依實際圖片調整
        minRadius=10, # 最小半徑 (像素)
        maxRadius=35  # 最大半徑 (像素)
    )
    
    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # center_x, center_y, radius
            detected_circles.append((int(i[0]), int(i[1]), int(i[2])))
            
    return detected_circles

def draw_results_on_image(pil_image, results, region_offsets):
    """
    將辨識結果畫在原始圖片上 (用於顯示)
    results: 包含 A1_value, A2_value 等的字典
    region_offsets: 每個區域相對於原圖左上角的偏移量 (x, y)
    """
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() # 轉為 BGR 以供 OpenCV 繪圖

    # 繪製 A1 方塊 (紅色多邊形)
    if 'A1_value' in results:
        offset_x, offset_y = region_offsets.get('A1', (0, 0))
        for square in results['A1_value']:
            # 將相對座標加上區域偏移量，轉回絕對座標
            abs_points = np.array(square) + [offset_x, offset_y]
            pts = abs_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img_cv, [pts], True, (0, 0, 255), 2)

    # 繪製 A2, A3 圓圈的外切紅框
    for region_key in ['A2_value', 'A3_value']:
        if region_key in results:
            region_name = region_key.split('_')[0]
            offset_x, offset_y = region_offsets.get(region_name, (0, 0))
            for (cx, cy, r) in results[region_key]:
                # 計算絕對座標
                abs_cx = cx + offset_x
                abs_cy = cy + offset_y
                # 畫外切正方形 (紅框)
                top_left = (abs_cx - r, abs_cy - r)
                bottom_right = (abs_cx + r, abs_cy + r)
                cv2.rectangle(img_cv, top_left, bottom_right, (0, 0, 255), 2)
                # 選擇性：畫出圓心
                cv2.circle(img_cv, (abs_cx, abs_cy), 2, (0, 255, 0), 3)

    # A4 不需要繪圖，因為它本身就是一個框
    
    # 轉回 PIL 格式以在 Streamlit 顯示
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# --- Streamlit 主程式 ---

st.set_page_config(page_title="答案卡區域標記與辨識工具", layout="wide")

# 初始化 Session State
if 'img_file' not in st.session_state:
    st.session_state.img_file = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
# 用於儲存手動框選的區域座標 (box: left, top, width, height)
if 'zones' not in st.session_state:
    st.session_state.zones = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
# 當前正在進行框選的模式
if 'cropping_mode' not in st.session_state:
    st.session_state.cropping_mode = None
# 儲存辨識後的數值結果
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = {}
# 儲存帶有標記結果的最終圖片
if 'result_image' not in st.session_state:
    st.session_state.result_image = None

st.title("📝 答案卡區域標記與自動辨識")
st.write("請上傳空白答案卡，依序標記區域，最後執行辨識並匯出資料。")

# 建立左右分欄
col_left, col_right = st.columns([1, 2])

# --- 左側欄位：控制項 ---
with col_left:
    st.header("1. 上傳與操作")
    uploaded_file = st.file_uploader("請上傳空白答案卡 (jpg, png)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        if st.session_state.img_file != uploaded_file:
            st.session_state.img_file = uploaded_file
            st.session_state.original_image = Image.open(uploaded_file)
            # 重置狀態
            st.session_state.zones = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
            st.session_state.cropping_mode = None
            st.session_state.recognition_results = {}
            st.session_state.result_image = None
            
        st.success("圖片已載入")
        st.markdown("---")
        st.header("2. 手動標示區域")
        st.info("點擊下方按鈕，在右側圖上手動框選對應區域。")

        # 定義按鈕的回調函數，設定當前的框選模式
        def set_crop_mode(mode):
            st.session_state.cropping_mode = mode

        # 區域 A1
        col_a1_btn, col_a1_stat = st.columns([2, 1])
        col_a1_btn.button("(1) 標示定位點區域 A1", on_click=set_crop_mode, args=('A1',), use_container_width=True)
        if st.session_state.zones['A1']: col_a1_stat.success("已標示")
        
        # 區域 A2
        col_a2_btn, col_a2_stat = st.columns([2, 1])
        col_a2_btn.button("(2) 標示基本資料區 A2", on_click=set_crop_mode, args=('A2',), use_container_width=True)
        if st.session_state.zones['A2']: col_a2_stat.success("已標示")

        # 區域 A3
        col_a3_btn, col_a3_stat = st.columns([2, 1])
        col_a3_btn.button("(3) 標示選擇題區 A3", on_click=set_crop_mode, args=('A3',), use_container_width=True)
        if st.session_state.zones['A3']: col_a3_stat.success("已標示")
        
        # 區域 A4
        col_a4_btn, col_a4_stat = st.columns([2, 1])
        col_a4_btn.button("(4) 標示手寫區 A4", on_click=set_crop_mode, args=('A4',), use_container_width=True)
        if st.session_state.zones['A4']: col_a4_stat.success("已標示")

        st.markdown("---")
        st.header("3. 執行辨識與匯出")

        # 檢查是否所有區域都已標示
        all_zones_marked = all(st.session_state.zones.values())
        
        start_btn = st.button("開始辨識", disabled=not all_zones_marked, type="primary", use_container_width=True)
        
        if not all_zones_marked:
            st.warning("請先完成上方 (1)~(4) 的區域標示。")

        if start_btn and st.session_state.original_image:
            with st.spinner("正在進行影像分析與辨識，請稍候..."):
                try:
                    results = {}
                    region_offsets = {} # 紀錄每個區域相對於原圖的偏移量
                    full_img_cv = cv2.cvtColor(np.array(st.session_state.original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
                    
                    # --- 處理 A1 (定位點) ---
                    box = st.session_state.zones['A1']
                    # 根據框選座標裁切圖片 (注意 numpy slicing 是 y, then x)
                    crop_a1 = full_img_cv[box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
                    results['A1_value'] = detect_corner_markers(crop_a1)
                    region_offsets['A1'] = (box['left'], box['top'])

                    # --- 處理 A2 (基本資料圓圈) ---
                    box = st.session_state.zones['A2']
                    crop_a2 = full_img_cv[box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
                    results['A2_value'] = detect_bubbles(crop_a2)
                    region_offsets['A2'] = (box['left'], box['top'])

                    # --- 處理 A3 (選擇題圓圈) ---
                    box = st.session_state.zones['A3']
                    crop_a3 = full_img_cv[box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
                    results['A3_value'] = detect_bubbles(crop_a3)
                    region_offsets['A3'] = (box['left'], box['top'])

                    # --- 處理 A4 (手寫區座標) ---
                    box = st.session_state.zones['A4']
                    # 記錄 A4 的 4 個角座標 (左上, 右上, 右下, 左下)
                    results['A4_value'] = [
                        (box['left'], box['top']),
                        (box['left'] + box['width'], box['top']),
                        (box['left'] + box['width'], box['top'] + box['height']),
                        (box['left'], box['top'] + box['height'])
                    ]

                    st.session_state.recognition_results = results
                    
                    # 將結果繪製到圖片上
                    result_img_pil = draw_results_on_image(st.session_state.original_image, results, region_offsets)
                    st.session_state.result_image = result_img_pil
                    
                    # 辨識完成後，退出框選模式以顯示結果圖
                    st.session_state.cropping_mode = None 
                    st.success(f"辨識完成! 找到 A1定位點組: {len(results['A1_value'])}, A2氣泡: {len(results['A2_value'])}, A3氣泡: {len(results['A3_value'])}")

                except Exception as e:
                    st.error(f"辨識過程中發生錯誤: {e}")

        # 匯出 Excel 按鈕
        if st.session_state.recognition_results:
            # 準備 Excel 資料
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # A1 Sheet
                a1_data = []
                for i, square in enumerate(st.session_state.recognition_results.get('A1_value', [])):
                    row = {'Square_ID': i+1}
                    for j, pt in enumerate(square):
                        row[f'Corner_{j+1}_X'] = pt[0]
                        row[f'Corner_{j+1}_Y'] = pt[1]
                    a1_data.append(row)
                pd.DataFrame(a1_data).to_excel(writer, sheet_name='A1_Markers', index=False)
                
                # A2 Sheet
                a2_data = [{'Bubble_ID': i+1, 'Center_X': c[0], 'Center_Y': c[1], 'Radius': c[2]} 
                           for i, c in enumerate(st.session_state.recognition_results.get('A2_value', []))]
                pd.DataFrame(a2_data).to_excel(writer, sheet_name='A2_Bubbles', index=False)

                # A3 Sheet
                a3_data = [{'Bubble_ID': i+1, 'Center_X': c[0], 'Center_Y': c[1], 'Radius': c[2]} 
                           for i, c in enumerate(st.session_state.recognition_results.get('A3_value', []))]
                pd.DataFrame(a3_data).to_excel(writer, sheet_name='A3_Bubbles', index=False)

                # A4 Sheet
                a4_coords = st.session_state.recognition_results.get('A4_value', [])
                if a4_coords:
                    a4_data = [{
                        'Top_Left_X': a4_coords[0][0], 'Top_Left_Y': a4_coords[0][1],
                        'Bottom_Right_X': a4_coords[2][0], 'Bottom_Right_Y': a4_coords[2][1]
                    }]
                    pd.DataFrame(a4_data).to_excel(writer, sheet_name='A4_Handwriting', index=False)

            output.seek(0)
            st.download_button(
                label="📥 下載辨識結果 Excel",
                data=output,
                file_name="omr_recognition_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- 右側欄位：顯示區域 ---
with col_right:
    st.header("預覽與操作區")
    
    if st.session_state.original_image is None:
        st.info("請先在左側上傳圖片。")
    else:
        current_mode = st.session_state.cropping_mode
        
        # 如果處於任何一種框選模式
        if current_mode in ['A1', 'A2', 'A3', 'A4']:
            st.warning(f"正在標示區域: **{current_mode}**。請在下方圖片拖曳滑鼠框選，完成後請點擊「Apply」或雙擊滑鼠。")
            
            # 取得之前儲存的該區域的框 (如果有的話)，作為預設顯示
            default_box = st.session_state.zones[current_mode]
            box_color = '#0000FF' # 藍色框
            
            # 呼叫 streamlit-cropper 元件
            cropped_box = st_cropper(
                st.session_state.original_image,
                realtime_update=True,
                box_color=box_color,
                aspect_ratio=None, # 不固定比例
                default_coords=(default_box['left'], default_box['top'], default_box['width'], default_box['height']) if default_box else None,
                key=f"cropper_{current_mode}" # 使用不同的 key 強制重新渲染元件
            )
            
            # 當 cropper 回傳數值時 (使用者完成框選)
            if cropped_box:
                # 將框選座標存入 session state
                st.session_state.zones[current_mode] = cropped_box
                # 不自動退出模式，讓使用者可以微調，直到他們點擊下一個按鈕
                # st.session_state.cropping_mode = None 
                # st.rerun()

        # 如果有辨識結果圖，優先顯示結果圖
        elif st.session_state.result_image is not None:
            st.image(st.session_state.result_image, caption="辨識結果 (紅框為自動偵測項目)", use_container_width=True)
            st.info("藍色框選線已移除，圖上顯示的是 OpenCV 辨識出的紅框。")
            
        # 否則顯示原圖 (非框選模式，也無結果時)
        else:
            st.image(st.session_state.original_image, caption="原始圖片", use_container_width=True)
