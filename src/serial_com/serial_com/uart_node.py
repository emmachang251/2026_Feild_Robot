import rclpy
from rclpy.node import Node
import serial
import struct
import threading
import socket
import time
from rover_interfaces.msg import RoverSensor, RoverControl
# 依你的方向定義修改：
# 若 cmd_vel_to_rover_control.py 是 0=正轉、1=反轉，就維持 0
# 若實際是 1=正轉、0=反轉，就改成 1
FORWARD_DIR = 1

class UartNode(Node):
    def __init__(self):
        super().__init__('uart_node')
        
        # Teleplot UDP 設定
        self.teleplot_address = ("127.0.0.1", 47269)
        self.teleplot_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        # 紀錄最近一次收到的左右輪目標 RPM
        self.left_target_rpm = 0.0
        self.right_target_rpm = 0.0

        # 1. 初始化 Publisher (發布 STM32 傳來的感測數據)
        self.sensor_pub = self.create_publisher(RoverSensor, 'sensor_data', 10)
        
        # 2. 初始化 Subscriber (接收高階控制指令，準備發給 STM32)
        # 此處訂閱我們上次確認不需要修改的 RoverControl.msg
        self.control_sub = self.create_subscription(
            RoverControl, 
            'chassis_control',
            self.control_callback, 
            10
        )
        
        # 3. 初始化 UART 序列埠 (請根據 RPi 5 實際的 UART 節點修改，例如 /dev/ttyAMA0 或 /dev/ttyUSB0)
        self.serial_port = '/dev/ttyACM0'
        self.baudrate = 115200
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=0.1)
            self.get_logger().info(f"成功連接 UART: {self.serial_port} @ {self.baudrate}")
        except Exception as e:
            self.get_logger().error(f"UART 連接失敗: {e}")
            return
            
        # 4. 啟動背景接收執行緒
        self.receive_thread = threading.Thread(target=self.receive_data_loop, daemon=True)
        self.receive_thread.start()
    def send_rpm_to_teleplot(
        self,
        left_actual_rpm: float,
        right_actual_rpm: float
    ) -> None:
        """把左右輪 target 與 actual RPM 傳送給 Teleplot"""

        timestamp_ms = int(time.time() * 1000)

        # 同一個 UDP 封包傳送四條曲線，時間戳一致
        message = "\n".join([
            f"left_target_rpm:{timestamp_ms}:{self.left_target_rpm}",
            f"left_actual_rpm:{timestamp_ms}:{left_actual_rpm}",
            f"right_target_rpm:{timestamp_ms}:{self.right_target_rpm}",
            f"right_actual_rpm:{timestamp_ms}:{right_actual_rpm}",
        ])

        try:
            self.teleplot_socket.sendto(
                message.encode("utf-8"),
                self.teleplot_address
            )
        except OSError as e:
            self.get_logger().warn(
                f"Teleplot UDP 傳送失敗: {e}"
            )
    def receive_data_loop(self):
        """背景執行緒：持續讀取並解析 STM32 回傳的 18 Bytes Rx 封包"""
        buffer = bytearray()
        
        while rclpy.ok():
            if self.ser.in_waiting > 0:
                buffer += self.ser.read(self.ser.in_waiting)
                
                # 尋找封包 Header (0xBB) 與 Footer (0x66)
                while len(buffer) >= 18:
                    if buffer[0] == 0xBB and buffer[17] == 0x66:
                        # 驗證資料長度位元是否為 0x0E (14 Bytes)
                        if buffer[1] == 0x0E:
                            # 計算 Checksum (Byte 2 到 Byte 15 的總和取低 8 位)
                            calculated_checksum = sum(buffer[2:16]) & 0xFF
                            received_checksum = buffer[16]
                            
                            if calculated_checksum == received_checksum:
                                self.parse_and_publish(buffer[2:16])
                            else:
                                self.get_logger().warn(f"Checksum 錯誤! 收到: {received_checksum}, 計算: {calculated_checksum}")
                        
                        # 移除已處理的 18 bytes
                        buffer = buffer[18:]
                    else:
                        # 若標頭不是 0xBB，移除第一個 byte 並繼續尋找
                        buffer.pop(0)

    def parse_and_publish(self, data_bytes):
        """將 14 Bytes 的資料區段解包並發布 ROS 2 訊息"""
        try:
            # 依序解析: 
            # < : Little-Endian (適用於 STM32 等 ARM Cortex-M)
            # H : uint16_t (ToF, 2 Bytes)
            # h : int16_t  (Pitch, Roll, Accel, EncL, EncR, 共 5 個 = hhhhh, 10 Bytes)
            # B : uint8_t  (Status, Reserved, 共 2 個 = BB, 2 Bytes)
            unpacked = struct.unpack('<HhhhhhBB', data_bytes)
            
            msg = RoverSensor()
            msg.tof_distance_mm = unpacked[0]
            
            # 將 int16 轉回 float32 的實際數值 (除以 100 還原小數點)
            msg.pitch_angle = unpacked[1] / 100.0
            msg.roll_angle = unpacked[2] / 100.0
            
            # 加速度還原 (假設 STM32 端也是乘 1000 後傳送以保留 3 位小數)
            msg.accel_z = unpacked[3] / 1000.0 
            
            msg.encoder_speed_left = unpacked[4]
            msg.encoder_speed_right = unpacked[5]
            msg.system_status_flag = unpacked[6]
            # unpacked[7] 是 Reserved 保留位，暫不使用
            # 依目前通訊定義，STM32 回傳的是 RPM × 10
            left_actual_rpm = unpacked[4] / 10.0
            right_actual_rpm = unpacked[5] / 10.0

            self.send_rpm_to_teleplot(
                left_actual_rpm,
                right_actual_rpm
            )
            self.sensor_pub.publish(msg)
            
        except struct.error as e:
            self.get_logger().error(f"解包錯誤: {e}")

    def control_callback(self, msg):
        """接收高階指令，打包成 Tx 封包發送給 STM32"""
        # Tx 控制封包 (15 Bytes): [0xAA] + [0x0B] + [...] + [Checksum] + [0x55]
        # 這裡的組裝邏輯與你先前規劃的 v2.0 一致，你需要依據 RoverControl 陣列內容填入
        
                # 速度欄位是 RPM × 10，因此除以 10 還原 RPM
        left_target_magnitude = msg.left_speed / 10.0
        right_target_magnitude = msg.right_speed / 10.0

        # 根據方向欄位恢復正負號
        left_sign = 1.0 if msg.left_dir == FORWARD_DIR else -1.0
        right_sign = 1.0 if msg.right_dir == FORWARD_DIR else -1.0

        self.left_target_rpm = (
            0.0
            if msg.left_speed == 0
            else left_sign * left_target_magnitude
        )

        self.right_target_rpm = (
            0.0
            if msg.right_speed == 0
            else right_sign * right_target_magnitude
        )
        # 範例結構 (需依據你 STM32 端的接收狀態機調整 byte 順序)
        tx_data = bytearray([
            0xAA, 0x0B, 
            msg.left_dir, (msg.left_speed >> 8) & 0xFF, msg.left_speed & 0xFF,
            msg.right_dir, (msg.right_speed >> 8) & 0xFF, msg.right_speed & 0xFF,
        ])
        
        # 補齊馬達陣列 (確保陣列長度足夠，避免 index out of range)
        for i in range(4):
            tx_data.append(msg.servo_angles[i] if i < len(msg.servo_angles) else 0)
            
        # 處理 LED 遮罩 (將 bool 陣列壓成 1 個 Byte)
        led_mask = 0
        for i in range(min(4, len(msg.led_states))):
            if msg.led_states[i]:
                led_mask |= (1 << i)
        tx_data.append(led_mask)
        
        # 計算 Checksum 並加上 Footer
        checksum = sum(tx_data[2:]) & 0xFF
        tx_data.append(checksum)
        tx_data.append(0x55)
        packet_hex = ' '.join(
            f'{byte:02X}' for byte in tx_data
        )

        #self.get_logger().info(
         #   f'TX UART ({len(tx_data)} bytes): {packet_hex}'
        #)

        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.write(tx_data)

def main(args=None):
    rclpy.init(args=args)
    node = UartNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()

        if hasattr(node, 'teleplot_socket'):
            node.teleplot_socket.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()