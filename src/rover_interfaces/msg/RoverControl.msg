# 機器人底盤與硬體控制命令

# 底盤控制 (方向與速度)
uint8 left_dir       # 0:停, 1:正轉, 2:反轉
uint16 left_speed    # PWM 速度
uint8 right_dir      # 0:停, 1:正轉, 2:反轉
uint16 right_speed   # PWM 速度

# 擴充硬體控制 (陣列長度不限，極度彈性)
uint8[] servo_angles # 伺服馬達角度陣列 (0~180)
bool[] led_states    # LED 開關陣列 (true/false)