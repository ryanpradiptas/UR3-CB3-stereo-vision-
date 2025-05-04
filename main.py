# ============================================================
# Bagian 1: Inisialisasi Library dan Parameter Global
# ============================================================
# Import library yang diperlukan
import cv2
import numpy as np
import rospy
import moveit_commander
import sys
import tkinter as tk
import subprocess
from geometry_msgs.msg import Pose
from tkinter import ttk
from PIL import Image, ImageTk
from stereo_vision import map_left_x, map_left_y, map_right_x, map_right_y, detect_green_cube, P_left, P_right
from robot_control import convert_camera_to_robot, ros2htm, invKine
from csv_log import write_integration_csv

# Buffer untuk menyimpan koordinat deteksi untuk mengecek kestabilan
coordinate_buffer = []
BUFFER_SIZE = 5        # jumlah frame untuk dicek kestabilannya
VARIANCE_THRESHOLD = 0.005  # threshold untuk varian koordinat

# ============================================================
# Bagian 2: Kode Utama (Stereo Vision, Robot Control, dan CSV Log)
# ============================================================
class StereoVisionUR3CB3:
    # Inisialisasi GUI utama
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)

        # Frame untuk tombol setup ROS
        self.setup_frame = ttk.Frame(window)
        self.setup_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Daftar perintah ROS sesuai tombol
        self.commands = {
            "ROS Master": "roscore",
            "Camera Launch": "roslaunch ur3_moveit 0_camera_calibrated.launch",
            "Camera Calibration": "rosrun camera_calibration cameracalibrator.py --approximate 0.1 --pattern acircles --size 4x11 --square 0.020 right:=/right_camera/image_raw left:=/left_camera/image_raw right_camera:=/right_camera left_camera:=/left_camera",
            "Handeye Calibration": "roslaunch easy_handeye ur3_calibrated.launch",
            "UR3 Calibration": "roslaunch ur_calibration calibration_correction.launch robot_ip:=10.29.202.167 target_filename:=\"$(rospack find ur_calibration)/etc/ex-ur3_calibration.yaml\"",
            "UR3 Connection": "roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.29.202.167 kinematics_config:=$(rospack find ur_calibration)/etc/ex-ur3_calibration.yaml",
            "MoveIt Execution": "roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true",
            "Rviz": "roslaunch ur3_moveit_config moveit_rviz.launch config:=true",
            "Manual Book": "evince integration_log.txt"
        }

        # Membuat tombol untuk setiap perintah ROS
        for idx, (name, cmd) in enumerate(self.commands.items()):
            btn = ttk.Button(
                self.setup_frame,
                text=name,
                command=lambda c=cmd: self.run_command(c),
                width=20
            )
            btn.grid(row=0, column=idx, padx=2, sticky=tk.W)

        # Frame utama untuk video dan data
        self.main_frame  = ttk.Frame(window); self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.video_frame = ttk.Frame(self.main_frame); self.video_frame.pack(side=tk.TOP, pady=8)
        self.video_label = ttk.Label(self.video_frame); self.video_label.pack()
        self.data_frame  = ttk.Frame(self.main_frame); self.data_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.data_grid   = ttk.Frame(self.data_frame); self.data_grid.pack(expand=True)

         # Variabel untuk menampilkan data secara real-time
        self.final_coord_var = tk.StringVar(value="Menunggu dimulai")
        self.robot_coord_var = tk.StringVar(value="Menunggu dimulai")
        self.status_var = tk.StringVar(value="Idle")

        # Label dan nilai untuk koordinat kamera dan robot serta status robot
        ttk.Label(self.data_grid, text="Koordinat Kamera:",  font=('Helvetica',10,'bold')).grid(row=0,column=0,sticky=tk.W)
        ttk.Label(self.data_grid, textvariable=self.final_coord_var).grid(row=0,column=1,sticky=tk.W)
        ttk.Label(self.data_grid, text="Koordinat Robot:",  font=('Helvetica',10,'bold')).grid(row=1,column=0,sticky=tk.W,pady=(6,0))
        ttk.Label(self.data_grid, textvariable=self.robot_coord_var).grid(row=1,column=1,sticky=tk.W)
        ttk.Label(self.data_grid, text="Status:", font=('Helvetica',10,'bold')).grid(row=6,column=0,sticky=tk.W,pady=(6,0))
        ttk.Label(self.data_grid, textvariable=self.status_var).grid(row=6,column=1,sticky=tk.W)

        # Tombol kontrol utama (Start, Stop, Reset)
        style = ttk.Style()
        style.configure("Green.TButton", foreground="green") 
        style.configure("Red.TButton", foreground="red")  
        style.configure("Blue.TButton", foreground="blue")
        style.configure('Bold.TButton', font=('Helvetica', 10, 'bold'))
        ctrl = ttk.Frame(self.data_frame); ctrl.pack(pady=8)
        ttk.Button(ctrl, text="Start", command=self.start,style="Green.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Stop", command=self.stop,style="Red.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Reset", command=self.reset,style="Blue.TButton").pack(side=tk.LEFT, padx=4)

        # Inisialisasi variabel untuk loop dan kontrol
        self.running = False
        self.buffer  = []
        self.delay   = 1
        self.window.after(self.delay, self.update_loop)
        
    # Menjalankan perintah ROS dalam terminal baru
    def run_command(self, command):
        subprocess.Popen(
            ["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"],
            start_new_session=True
        )

    # Memulai proses deteksi dan kontrol robot
    def start(self):
        self.running = True
        self.status_var.set("Status: deteksi aktif")

        # Inisialisasi MoveIt dan ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ur3_move_integration', anonymous=True)
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        # Membuka koneksi ke kamera stereo
        self.cap_left  = cv2.VideoCapture(0)
        self.cap_right = cv2.VideoCapture(2)
        # Set resolusi kamera
        for cam in (self.cap_left, self.cap_right):
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    # Menghentikan semua proses
    def stop(self):
        self.running = False
        self.status_var.set("Robot dihentikan")


    # Mereset robot ke posisi awal
    def reset(self):
        self.running = False
        self.status_var.set("Robot direset")
        # Posisi joint saat reset (posisi UP)
        UP_position = [0.0, -1.571, 0.0, -1.571, 0.0, 0.0]
        self.group.set_joint_value_target(UP_position)
        self.group.go(wait=True)
        self.buffer.clear()

    # Loop utama untuk update citra, deteksi, dan kontrol robot
    def update_loop(self):
        if self.running:
            # Membaca frame dari kamera kiri dan kanan
            okL, frameL = self.cap_left.read()
            okR, frameR = self.cap_right.read()

            if okL and okR:
                # Rektifikasi gambar menggunakan peta kalibrasi
                rectL = cv2.remap(frameL, map_left_x, map_left_y, cv2.INTER_LINEAR)
                rectR = cv2.remap(frameR, map_right_x, map_right_y, cv2.INTER_LINEAR)
                # Deteksi kubus hijau di kedua gambar
                pL = detect_green_cube(rectL)
                pR = detect_green_cube(rectR)
                if pL and pR:
                    # Menandai titik deteksi di gambar
                    cv2.circle(rectL, pL, 5, (0,0,255), -1)
                    cv2.circle(rectR, pR, 5, (0,0,255), -1)

                    # Triangulasi titik untuk dapatkan koordinat 3D
                    ptsL = np.array([[pL[0]],[pL[1]]], dtype=np.float32)
                    ptsR = np.array([[pR[0]],[pR[1]]], dtype=np.float32)
                    P4  = cv2.triangulatePoints(P_left, P_right, ptsL, ptsR)
                    P3  = (P4[:3]/P4[3]).flatten() # Konversi ke koordinat 3D

                    # Update koordinat di GUI
                    self.final_coord_var.set(f"X:{P3[0]:.3f} Y:{P3[1]:.3f} Z:{P3[2]:.3f}")
                    self.buffer.append(P3)

                    # Cek kestabilan koordinat dalam buffer
                    if len(self.buffer) >= BUFFER_SIZE:
                        var = np.var(self.buffer, axis=0) # Hitung variansi
                        if np.all(var < VARIANCE_THRESHOLD):
                            # Jika stabil, konversi ke koordinat robot
                            mean3 = np.mean(self.buffer, axis=0)
                            robotXYZ = convert_camera_to_robot(mean3)
                            # Update tampilan koordinat robot
                            self.robot_coord_var.set(f"X:{robotXYZ[0] + 0.025:.3f} Y:{-abs(robotXYZ[1]- 1.2):.3f} Z:{abs(robotXYZ[2]-0.35):.3f}")

                            # Buat pose target untuk robot
                            pose = Pose()
                            pose.position.x = robotXYZ[0] + 0.025
                            pose.position.y = -abs(robotXYZ[1]-1.2)
                            pose.position.z = abs(robotXYZ[2]-0.35)
                            pose.orientation.x = 1.0

                            # Hitung inverse kinematics dan gerakkan robot
                            HTM  = ros2htm(pose)
                            sol  = invKine(HTM)
                            joints = np.array(sol[:,0]).flatten()
                            self.group.set_joint_value_target(joints)
                            self.group.go(wait=True)

                            # Log data ke CSV
                            write_integration_csv(mean3, robotXYZ, sol, joints)
                            self.status_var.set("Robot Bergerak")
                            self.buffer.clear()
                        else:
                            # Hapus data terlama jika tidak stabil
                            self.buffer.pop(0)

                # Tampilkan gambar stereo yang sudah diproses
                combo = cv2.hconcat([rectL, rectR])
                rgb   = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        # Pengaturan loop selanjutnya
        self.window.after(self.delay, self.update_loop)

    # Destruktor untuk membersihkan sumber daya ( Menutup kamera, moveit ROS, dan opencv)
    def __del__(self):
        for cam in (self.cap_left, self.cap_right):
            if cam.isOpened(): cam.release()
        moveit_commander.roscpp_shutdown()
        cv2.destroyAllWindows()

# ============================================================
# Bagian 3: Menjalankan Aplikasi
# ============================================================
if __name__ == '__main__':
    root = tk.Tk() # Membuat window Tkinter
    app  = StereoVisionUR3CB3(root, "Stereo-Vision UR3 CB3") # Inisialisasi aplikasi
    root.protocol("WM_DELETE_WINDOW", root.quit) # Handle close window
    root.mainloop() # Mulai main loop GUI