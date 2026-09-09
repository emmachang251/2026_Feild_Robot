# 機器人底層感測器回傳數據 (Rx)

# ToF 飛時測距儀
uint16 tof_distance_mm      # VL53L0X 測距 (毫米)

# IMU 姿態與加速度
float32 pitch_angle         # 俯仰角 (度)
float32 roll_angle          # 翻滾角 (度)
float32 accel_z             # Z軸加速度 (m/s^2 或 g)

# 輪速編碼器回授
int32 encoder_speed_left    # 左輪轉速 (rpm 或 ticks/sec)
int32 encoder_speed_right   # 右輪轉速 (rpm 或 ticks/sec)

# 系統狀態
uint8 system_status_flag    # 狀態旗標 (如碰撞、電量等)