import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io

# --- 影像處理函數 (保持不變) ---
def detect_corner_markers(img_crop_bgr):
    """辨識黑色方形定位點 (A1)"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: continue
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            points = approx.reshape(4, 2).tolist()
            detected_squares.append(points)
    return detected_squares

def detect_bubbles(img_crop_bgr):
    """辨識圓形氣泡 (A2, A3)"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=15,    
        param1=100,
        param2=25,     
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
    """繪製結果圖"""
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() 

    if 'A1_value' in results:
        offset_x, offset_y = region_offsets.get('A1', (0, 0))
        for square in results['A1_value']:
            abs_points = np.array(square) + [offset_x, offset_y]
            pts = abs_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img_cv, [pts], True, (0, 0, 255), 3)

    for region_key in ['A2_value', 'A3_value']:
        if region_key in results:
            region_name = region_key.split('_')[0]
            offset_x, offset_y = region_offsets.get(region_name, (0, 0))
            for (cx, cy, r) in results[region_key]:
                abs_cx = cx + offset_x
                abs_cy = cy + offset_y
                cv2.rectangle(img_cv, (abs_cx - r, abs_cy - r), (abs_cx + r, abs_cy + r), (0, 0, 255), 2)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# --- Streamlit 頁面設定 ---
st.set_page_config(page_title="答案卡辨識系統", layout="wide")

# Session State 初始化
if 'img_file' not in st.session_state:
    st.session_state.img_file = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'resized_image' not in st.session_state:
    st.session_state.resized_image = None
if 'scale_factor' not in st.session_state:
    st.session_state.scale_factor = 1.0
    
# zones: 儲存"已確認"的區域座標
if 'zones' not in st.session_state:
    st.session_state.zones = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
    
# cropping_mode: 當前正在操作哪個模式 (None, 'A1', 'A2', 'A3', 'A4')
if 'cropping_mode' not in st.session_state:
    st.session_state.cropping_mode = None

# temp_box: 儲存 cropper 即時回傳但"尚未確認"的座標
if 'temp_box' not in st.session_state:
    st.session_state.temp_box = None

if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = {}
if 'result_image' not in st.session_state:
    st.session_state.result_image = None


st.title("📝 答案卡標示與辨識 (互動優化版)")

col_left, col_right = st.columns([1, 2])

# ================= 左側：控制面板 =================
with col_left:
    st.header("1. 上傳與設定")
    uploaded_file = st.file_uploader("請上傳空白答案卡 (jpg, png)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # 處理檔案上傳
        if st.session_state.img_file != uploaded_file:
            st.session_state.img_file = uploaded_file
            
            # 讀圖與縮放預處理
            original_pil = Image.open(uploaded_file)
            st.session_state.original_image = original_pil
            
            display_width = 800
            w_percent = (display_width / float(original_pil.size[0]))
            h_size = int((float(original_pil.size[1]) * float(w_percent)))
            
            if original_pil.size[0] > display_width:
                st.session_state.resized_image = original_pil.resize((display_width, h_size), Image.Resampling.LANCZOS)
                st.session_state.scale_factor = 1 / w_percent 
            else:
                st.session_state.resized_image = original_pil
                st.session_state.scale_factor = 1.0

            # 重置所有狀態
            st.session_state.zones = {'A1': None, 'A2': None, 'A3': None, 'A4': None}
            st.session_state.cropping_mode = None
            st.session_state.temp_box = None
            st.session_state.recognition_results = {}
            st.session_state.result_image = None
            
        st.success(f"圖片已載入")
        
        # --- 切換模式的函數 ---
        def set_mode(mode):
            st.session_state.cropping_mode = mode
            # 切換模式時，清空暫存，確保 Cropper 重置
            st.session_state.temp_box = None 

        st.markdown("### 2. 標示區域")
        st.caption("點擊按鈕進入編輯模式，調整完畢後請按右側的「確定」鍵。")

        # 定義按鈕樣式：如果是當前模式，用 primary (紅色強調)，否則 secondary
        def get_btn_type(mode_name):
            return "primary" if st.session_state.cropping_mode == mode_name else "secondary"

        # A1 按鈕
        c1, c2 = st.columns([3, 1])
        c1.button("標示 A1 (定位點)", 
                  on_click=set_mode, args=('A1',), 
                  type=get_btn_type('A1'), 
                  use_container_width=True)
        if st.session_state.zones['A1']: c2.success("✔")

        # A2 按鈕
        c1, c2 = st.columns([3, 1])
        c1.button("標示 A2 (基本資料)", 
                  on_click=set_mode, args=('A2',), 
                  type=get_btn_type('A2'), 
                  use_container_width=True)
        if st.session_state.zones['A2']: c2.success("✔")

        # A3 按鈕
        c1, c2 = st.columns([3, 1])
        c1.button("標示 A3 (選擇題)", 
                  on_click=set_mode, args=('A3',), 
                  type=get_btn_type('A3'), 
                  use_container_width=True)
        if st.session_state.zones['A3']: c2.success("✔")

        # A4 按鈕
        c1, c2 = st.columns([3, 1])
        c1.button("標示 A4 (手寫區)", 
                  on_click=set_mode, args=('A4',), 
                  type=get_btn_type('A4'), 
                  use_container_width=True)
        if st.session_state.zones['A4']: c2.success("✔")

        st.markdown("---")
        
        # 辨識邏輯
        all_marked = all(st.session_state.zones.values())
        if st.button("開始辨識", disabled=not all_marked, type="primary", use_container_width=True):
            if st.session_state.original_image:
                with st.spinner("辨識中..."):
                    try:
                        results = {}
                        region_offsets = {}
                        scale = st.session_state.scale_factor
                        full_img_cv = cv2.cvtColor(np.array(st.session_state.original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
                        
                        for zone_key in ['A1', 'A2', 'A3']:
                            box = st.session_state.zones[zone_key]
                            real_left = int(box['left'] * scale)
                            real_top = int(box['top'] * scale)
                            real_width = int(box['width'] * scale)
                            real_height = int(box['height'] * scale)
                            
                            real_left = max(0, real_left)
                            real_top = max(0, real_top)
                            
                            crop = full_img_cv[real_top:real_top+real_height, real_left:real_left+real_width]
                            
                            if zone_key == 'A1':
                                results['A1_value'] = detect_corner_markers(crop)
                            else:
                                results[f'{zone_key}_value'] = detect_bubbles(crop)
                            region_offsets[zone_key] = (real_left, real_top)

                        # A4
                        box_a4 = st.session_state.zones['A4']
                        real_left = int(box_a4['left'] * scale)
                        real_top = int(box_a4['top'] * scale)
                        real_width = int(box_a4['width'] * scale)
                        real_height = int(box_a4['height'] * scale)
                        results['A4_value'] = [(real_left, real_top), (real_left + real_width, real_top + real_height)]

                        st.session_state.recognition_results = results
                        st.session_state.result_image = draw_results_on_image(st.session_state.original_image, results, region_offsets)
                        st.session_state.cropping_mode = None 
                        st.success("辨識完成！")
                        
                    except Exception as e:
                        st.error(f"錯誤: {e}")

        # 下載 Excel
        if st.session_state.recognition_results:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 簡化範例：只匯出A2
                a2_data = [{'ID': i+1, 'X': c[0], 'Y': c[1]} for i, c in enumerate(st.session_state.recognition_results.get('A2_value', []))]
                if a2_data: pd.DataFrame(a2_data).to_excel(writer, sheet_name='A2', index=False)
            output.seek(0)
            st.download_button("下載 Excel", data=output, file_name="results.xlsx")


# ================= 右側：工作區域 =================
with col_right:
    if st.session_state.original_image is None:
        st.info("👈 請先從左側上傳圖片")
    else:
        current_mode = st.session_state.cropping_mode
        
        # --- 情況 1: 編輯模式 (顯示 Cropper + 確認按鈕) ---
        if current_mode in ['A1', 'A2', 'A3', 'A4']:
            st.markdown(f"### 🔧 正在設定：**{current_mode}** 區域")
            st.info("請拖曳下方藍框至正確位置，完成後按「確定」。")
            
            # 設定初始位置：強制左上角 50x50
            # 只有當第一次進入該模式且尚未有暫存時，才使用 default_coords
            # 否則 Cropper 會維持使用者最後拖曳的狀態
            if st.session_state.temp_box is None:
                start_coords = (0, 0, 50, 50) 
            else:
                # 如果已經在拖曳中，這裡設為 None，讓 cropper 自己管理狀態
                start_coords = None

            # 呼叫 Cropper
            # key 設為 current_mode 確保切換按鈕時，藍框會重置
            box_data = st_cropper(
                st.session_state.resized_image, 
                realtime_update=True,
                box_color='#0000FF',
                aspect_ratio=None,
                default_coords=start_coords, 
                return_type='box',
                key=f"cropper_{current_mode}" 
            )
            
            # 將 cropper 的即時回傳值存入 temp_box
            if box_data:
                st.session_state.temp_box = box_data

            # --- 確認按鈕 ---
            # 只有當 temp_box 有值時才允許確認
            if st.button(f"✅ 確定儲存 {current_mode} 區域", type="primary", use_container_width=True):
                if st.session_state.temp_box:
                    # 1. 將暫存值寫入永久 zones
                    st.session_state.zones[current_mode] = st.session_state.temp_box
                    # 2. 清除模式與暫存
                    st.session_state.cropping_mode = None
                    st.session_state.temp_box = None
                    # 3. 強制刷新頁面，回到預覽狀態
                    st.rerun()
                else:
                    st.warning("請先調整框線")

        # --- 情況 2: 顯示辨識結果 ---
        elif st.session_state.result_image is not None:
            st.image(st.session_state.result_image, caption="辨識結果", use_container_width=True)
            
        # --- 情況 3: 預覽狀態 (顯示原圖 + 已標記的區域) ---
        else:
            st.image(st.session_state.resized_image, caption="原始預覽圖", use_container_width=True)
            
            # 可以在這裡畫出已經標記好的綠色框框給使用者看 (選用功能)
            # 這裡簡單列出狀態
            marked_zones = [k for k, v in st.session_state.zones.items() if v is not None]
            if marked_zones:
                st.caption(f"目前已標記區域: {', '.join(marked_zones)}")
