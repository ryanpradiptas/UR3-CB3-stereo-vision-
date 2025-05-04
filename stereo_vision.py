# Import library yang diperlukan
import numpy as np
import cv2

# ============================================================
# Bagian 1: Parameter Kalibrasi Kamera Stereo
# ============================================================
"""
Parameter kalibrasi ini didapatkan dari proses kalibrasi kamera stereo.
Setiap kamera (kiri dan kanan) memiliki parameter intrinsik dan ekstrinsik unik.
"""
# === Parameter Kalibrasi Kamera Kiri ===
# Koefisien distorsi radial dan tangensial
D_left = np.array([-0.4106382404282637, 0.207527214606134, 0.01125258307869868, 0.0003884850050468037, 0]) 
# Matriks kamera intrinsik (focal length, principal point)
K_left = np.array([
    [1521.996967411467, 0, 533.0299162482958], 
    [0, 1523.097959911586, 410.4658279208516], 
    [0, 0, 1]
]) 
# Matriks rotasi hasil rectifikasi
R_left = np.array([
    [0.9986846159616873, -0.004513876491244535, 0.05107506985290136], 
    [0.004083415971338171, 0.999955288152529, 0.00852920921291536], 
    [-0.05111128598912357, -0.008309429271282726, 0.9986583949628238]
])
# Matriks proyeksi 3x4
P_left = np.array([
    [1605.222318650465, 0, 393.7757530212402, 0], 
    [0, 1605.222318650465, 419.1833305358887, 0], 
    [0, 0, 1, 0]
])

# === Parameter Kalibrasi Kamera Kanan ===
D_right = np.array([-0.4598684251082059, 0.5445937307886485, 0.01345297199075331, 0.001649438644451924, 0])
K_right = np.array([
    [1514.118574353898, 0, 430.0525179559846], 
    [0, 1516.411192597767, 408.6720280486751], 
    [0, 0, 1]
])
R_right = np.array([
    [0.9985534479781977, -0.02836274560051881, 0.04567894693240061], 
    [0.02874659031786245, 0.9995565467236821, -0.007768104461481172], 
    [-0.04543836568308539, 0.009069981467883931, 0.998925968408181]
])
P_right = np.array([
    [1605.222318650465, 0, 393.7757530212402, -167.4926316361975], 
    [0, 1605.222318650465, 419.1833305358887, 0], 
    [0, 0, 1, 0]
])

# ============================================================
# Bagian 2: Persiapan Rectifikasi Gambar
# ============================================================
"""
Membuat peta transformasi untuk mengoreksi distorsi dan menyelaraskan 
epipolar line pada gambar stereo menggunakan parameter kalibrasi.
"""
# Resolusi gambar kamera
size = (1024, 768)

# Hitung mapping untuk rectifikasi kamera kiri
map_left_x, map_left_y = cv2.initUndistortRectifyMap(K_left, D_left, R_left, P_left[:3, :3], size, cv2.CV_32FC1)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(K_right, D_right, R_right, P_right[:3, :3], size, cv2.CV_32FC1)

# ============================================================
# Bagian 3: Deteksi Objek Berwarna Hijau
# ============================================================
def detect_green_cube(img):
    """
    Mendeteksi objek berwarna hijau dalam gambar menggunakan thresholding HSV.
    
    Parameter:
    img (numpy array): Gambar input dalam format BGR
    
    Return:
    tuple: Koordinat centroid (cx, cy) atau None jika tidak terdeteksi
    """
    # Rentang warna hijau dalam HSV (Hue, Saturation, Value)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    # Konversi warna ke ruang HSV dan aplikasikan thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Temukan kontur pada masking
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Ambil kontur terbesar berdasarkan area
        largest_contour = max(contours, key=cv2.contourArea)
        # Hitung momen untuk mencari centroid
        M = cv2.moments(largest_contour)
        if M['m00'] != 0: # Hindari division by zero
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None # Tidak ada objek terdeteksi