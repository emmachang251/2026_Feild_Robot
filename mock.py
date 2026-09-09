import serial

# 填入步驟一看到的第二個路徑
PORT = '/dev/pts/2' 
ser = serial.Serial(PORT, 115200)
print(f"等待接收資料... (監聽 {PORT})")

while True:
    if ser.in_waiting >= 18:
        data = ser.read(18)
        print("收到 ROS 2 傳來的指令了！")
        print("內容(Hex):", data.hex())