# Import library yang diperlukan
import os
import time
import datetime
import csv
import numpy as np


def write_integration_csv(final_coord, koordinat_robot, ik_solution, joint_target):
    """
    Mencatat data integrasi sistem ke file CSV untuk keperluan logging dan analisis.
    
    Parameter:
    final_coord (np.array/iterable): Koordinat akhir objek yang stabil dari buffer
    koordinat_robot (np.array/iterable): Hasil konversi koordinat ke frame robot
    ik_solution (np.matrix): Matriks solusi inverse kinematics (6x8) dalam radian
    joint_target (np.array/iterable): Sudut joint target yang dipilih untuk eksekusi
    """

    # ============================================================
    # Bagian 1: Persiapan Direktori dan File
    # ============================================================
    log_folder = "log"                  # Nama folder penyimpanan log
    if not os.path.exists(log_folder):  # Cek keberadaan folder
        os.makedirs(log_folder)         # Buat folder jika belum ada
    
    # ============================================================
    # Bagian 2: Penulisan Data ke CSV
    # ============================================================
    file_path = os.path.join(log_folder, "log_integrasi.csv")   # Path lengkap file
    file_exists = os.path.exists(file_path)                     # Cek keberadaan file
    
    with open(file_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Tulis header jika file baru
        if not file_exists:
            writer.writerow(["Timestamp", "Final_Coordinate", "Converted_Coordinate", "IK_Solution", "Target_Joint"])
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_coord_str = str(final_coord.tolist() if isinstance(final_coord, np.ndarray) else final_coord)
        koordinat_robot_str = str(koordinat_robot.tolist() if isinstance(koordinat_robot, np.ndarray) else koordinat_robot) # Koordinat akhir objek (frame kamera)
        ik_solution_str = str(np.round(ik_solution, 6).tolist())                                                            # Hasil konversi ke frame robot
        joint_target_str = str(joint_target.tolist() if isinstance(joint_target, np.ndarray) else joint_target)             # Seluruh solusi inverse kinematics
        
        writer.writerow([timestamp, final_coord_str, koordinat_robot_str, ik_solution_str, joint_target_str])               # Rekam data dalam satu baris CSV
