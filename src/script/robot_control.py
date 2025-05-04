# Import library yang diperlukan
import numpy as np
import math, cmath
import tf.transformations as tf
from math import cos, sin, atan2, acos, asin, sqrt, pi, radians
from scipy.spatial.transform import Rotation as R
from numpy import linalg


# ============================================================
# Bagian 1: Parameter Kalibrasi Tangan-Mata (Hand-Eye)
# ============================================================
"""
Transformasi antara sistem koordinat kamera dan robot.
Diperoleh dari proses hand-eye calibration menggunakan ROS easy_handeye.
"""
qw = 0.9799839430285541         # Komponen w quaternion
qx =  -0.19592425097874797      # Komponen x quaternion
qy = 0.033945497703366856       # Komponen y quaternion
qz = 0.009636517539832476       # Komponen z quaternion
t_vector = np.array(            # Vektor translasi [x, y, z]
    [-0.04067329032103688, -0.05307357604382259, 0.3465391101684713])

# Konversi quaternion ke matriks rotasi 3x3
R_cam_base = R.from_quat([qx, qy, qz, qw]).as_matrix()
T_cam_base = np.eye(4)
T_cam_base[:3, :3] = R_cam_base # Komponen rotasi
T_cam_base[:3, 3] = t_vector    # Komponen translasi

## ============================================================
# Bagian 2: Parameter Kinematika Robot UR3 (DH Parameters)
# ============================================================
"""
Parameter Denavit-Hartenberg untuk robot UR3.
Satuan dalam meter (translasi) dan radian (rotasi).
"""
# Parameter d (translasi sepanjang sumbu z)
d1 = 0.1519     # Jarak dari base ke joint 2  
d4 = 0.11235    # Jarak dari joint 4 ke joint 5
d5 = 0.08535    # Jarak dari joint 5 ke joint 6
d6 = 0.0819     # Jarak dari joint 6 ke end-effector
d = np.matrix([d1, 0, 0, d4, d5, d6])
# Parameter a (translasi sepanjang sumbu x)
a2 = -0.24365   # Panjang link 2
a3 = -0.21325   # Panjang link 3
a = np.matrix([0, a2, a3, 0, 0, 0])
# Parameter α (sudut twist antara sumbu z)
alph = np.matrix([pi/2, 0, 0, pi/2, -pi/2, 0])

# ============================================================
# Bagian 3: Fungsi Transformasi Koordinat
# ============================================================
def convert_camera_to_robot(obj_cam):
    """
    Mengonversi koordinat 3D dari frame kamera ke frame robot.
    
    Parameter:
    obj_cam (numpy array): Koordinat [x, y, z] dalam frame kamera
    
    Return:
    numpy array: Koordinat [x, y, z] dalam frame robot
    """
    # Koreksi sistem koordinat kamera (Y dan Z ditukar)
    obj_cam_corrected = obj_cam[[0, 2, 1]]
    # Konversi ke koordinat homogen [x, y, z, 1]
    obj_cam_corrected_hom = np.hstack((obj_cam_corrected, 1))
    # Transformasi menggunakan matriks kalibrasi
    obj_robot_hom = T_cam_base.dot(obj_cam_corrected_hom)
    return obj_robot_hom[:3] # Kembalikan koordinat 3D

# ============================================================
# Bagian 4: Konversi Pose ROS ke Matriks Transformasi
# ============================================================
def ros2htm(ros_pose):
    """
    Mengonversi ROS Pose ke matriks transformasi homogen 4x4.
    
    Parameter:
    ros_pose (geometry_msgs/Pose): Posisi dan orientasi ROS
    
    Return:
    numpy matrix: Matriks transformasi 4x4
    """
    # Ekstrak posisi
    px = ros_pose.position.x
    py = ros_pose.position.y
    pz = ros_pose.position.z

    # Ekstrak orientasi dalam quaternion
    q = [ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w]
    # Konversi quaternion ke Euler angles (ZYX convention)
    rx, ry, rz = tf.euler_from_quaternion(q)

    Rz = np.matrix([[cos(rz), -sin(rz), 0],     # Matriks rotasi sumbu Z
                     [sin(rz), cos(rz), 0],
                     [0, 0, 1]])
    Ry = np.matrix([[cos(ry), 0, sin(ry)],      # Matriks rotasi sumbu Y
                     [0, 1, 0],
                     [-sin(ry), 0, cos(ry)]])
    Rx = np.matrix([[1, 0, 0],                  # Matriks rotasi sumbu X
                     [0, cos(rx), -sin(rx)],
                     [0, sin(rx), cos(rx)]])
    T = np.matrix(np.identity(4))               # Gabungkan rotasi dan translasi
    T[:3, :3] = Rz * Ry * Rx                    # Komposisi rotasi Z -> Y -> X
    T[0, 3] = px                                # Posisi x
    T[1, 3] = py                                # Posisi y
    T[2, 3] = pz                                # Posisi z
    return T

# ============================================================
# Bagian 5: Forward Kinematic (DH Transformation)
# ============================================================
def AH(n, th, c):
    """
    Menghitung matriks transformasi untuk joint ke-n menggunakan parameter DH.
    
    Parameter:
    n (int): Nomor joint (1-6)
    th (matrix): Matriks sudut joint
    c (int): Kolom solusi
    
    Return:
    numpy matrix: Matriks transformasi 4x4
    """
    # Matriks translasi sepanjang x (parameter a)
    T_a = np.matrix(np.identity(4))
    T_a[0, 3] = a[0, n - 1]
    # Matriks translasi sepanjang z (parameter d)
    T_d = np.matrix(np.identity(4))
    T_d[2, 3] = d[0, n - 1]
    # Matriks rotasi sumbu z (sudut theta)
    Rzt = np.matrix([[cos(th[n - 1, c]), -sin(th[n - 1, c]), 0, 0],
                     [sin(th[n - 1, c]), cos(th[n - 1, c]), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # Matriks rotasi sumbu x (sudut alpha)
    Rxa = np.matrix([[1, 0, 0, 0],
                     [0, cos(alph[0, n - 1]), -sin(alph[0, n - 1]), 0],
                     [0, sin(alph[0, n - 1]), cos(alph[0, n - 1]), 0],
                     [0, 0, 0, 1]])
    A_i = T_d * Rzt * T_a * Rxa     # Transformasi gabungan
    return A_i

def HTrans(th, c):
    """
    Menghitung matriks transformasi dari base ke end-effector (T06).
    
    Parameter:
    th (matrix): Matriks sudut joint
    c (int): Kolom solusi
    
    Return:
    numpy matrix: Matriks transformasi 4x4
    """
    # Hitung matriks transformasi untuk setiap joint
    A1 = AH(1, th, c)
    A2 = AH(2, th, c)
    A3 = AH(3, th, c)
    A4 = AH(4, th, c)
    A5 = AH(5, th, c)
    A6 = AH(6, th, c)
    T06 = A1 * A2 * A3 * A4 * A5 * A6   # Kalikan semua matriks transformasi
    return T06

# ============================================================
# Bagian 6: Inverse Kinematic
# ============================================================
def invKine(desired_pos):
    """
    Menghitung inverse kinematics untuk posisi end-effector yang diinginkan.
    
    Parameter:
    desired_pos (numpy matrix): Matriks transformasi 4x4 target
    
    Return:
    numpy matrix: Matriks 6x8 berisi 8 solusi sudut joint (radian)
    """
    # Matriks penyimpanan solusi
    th = np.matrix(np.zeros((6, 8)))    

    # ------ Langkah 1: Hitung theta 1 ------
    # Posisi titik siku (P_05)
    P_05 = desired_pos * np.matrix([0, 0, -d6, 1]).T - np.matrix([0, 0, 0, 1]).T

    # Hitung dua kemungkinan sudut theta 1
    psi = atan2(P_05[1, 0], P_05[0, 0])                     # Sudut dasar
    phi = acos(d4 / sqrt(P_05[0, 0]**2 + P_05[1, 0]**2))    # Koreksi geometri
    th[0, 0:4] = pi / 2 + psi + phi                         # Solusi 1-4
    th[0, 4:8] = pi / 2 + psi - phi                         # Solusi 5-8
    th = th.real                                            # Pastikan nilai real

    # ------ Langkah 2: Hitung theta 5 ------
    # Konfigurasi yang valid
    cl = [0, 4]
    for i in range(len(cl)):
        # Hitung transformasi inverse
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_16 = T_10 * desired_pos

        # Hitung theta5 menggunakan geometri segitiga
        th[4, c:c+2] = acos((T_16[2, 3] - d4) / d6)
        th[4, c+2:c+4] = -acos((T_16[2, 3] - d4) / d6)
    th = th.real

    # ------ Langkah 3: Hitung theta 6 ------
    cl = [0, 2, 4, 6]
    for i in range(len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_16 = linalg.inv(T_10 * desired_pos)
        # Hitung theta6 dari komponen rotasi
        th[5, c:c+2] = atan2(-T_16[1, 2] / sin(th[4, c]), T_16[0, 2] / sin(th[4, c]))
    th = th.real

    # ------ Langkah 4: Hitung theta 3 ------
    cl = [0, 2, 4, 6]
    for i in range(len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_65 = AH(6, th, c)
        T_54 = AH(5, th, c)
        T_14 = T_10 * desired_pos * linalg.inv(T_54 * T_65)
        # Hitung posisi relatif untuk theta3
        P_13 = T_14 * np.matrix([0, -d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T
        # Hukum cosinus untuk segitiga
        t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2) / (2 * a2 * a3))
        th[2, c] = t3.real          # Solusi elbow-up
        th[2, c + 1] = -t3.real     # Solusi elbow-down

    # ------ Langkah 5: Hitung theta 2 dan 4 ------
    cl = range(8)
    for i in (cl):
        # Hitung transformasi antara joint
        T_10 = linalg.inv(AH(1, th, c))
        T_65 = linalg.inv(AH(6, th, c))
        T_54 = linalg.inv(AH(5, th, c))
        T_14 = T_10 * desired_pos * T_65 * T_54
        # Hitung theta2 dari proyeksi geometri
        P_13 = T_14 * np.matrix([0, -d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T
        th[1, c] = -atan2(P_13[1, 0], -P_13[0, 0]) + asin(a3 * sin(th[2, c]) / linalg.norm(P_13))
        # Hitung theta4 menggunakan transformasi inverse
        T_32 = linalg.inv(AH(3, th, c))
        T_21 = linalg.inv(AH(2, th, c))
        T_34 = T_32 * T_21 * T_14
        th[3, c] = atan2(T_34[1, 0], T_34[0, 0])
    th = th.real

    return th       # Kembalikan solusi real