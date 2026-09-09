import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # --- 車體尺寸與安全設定 ---
        self.half_width = 0.25      # 車寬 40cm，左右各抓 25cm
        self.safe_distance_x = 0.20  # 正前方 20cm 內有東西就煞車
        
        # ★ 新增：用來「記憶」轉彎方向的變數
        self.locked_turn_direction = None 
        
        self.get_logger().info("🧠 智慧尋路系統已啟動！(具備轉向鎖定防呆機制)")

    def scan_callback(self, msg):
        obstacle_detected = False
        min_front_dist = float('inf')

        left_distances = []
        right_distances = []

        for i, r in enumerate(msg.ranges):
            # 過濾無效雜訊
            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                continue
            
            theta = msg.angle_min + i * msg.angle_increment
            deg = math.degrees(theta)
            
            # 1. 防撞機制：檢查正前方矩形
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            
            if 0.0 < x < self.safe_distance_x and abs(y) <= self.half_width:
                obstacle_detected = True
                if x < min_front_dist:
                    min_front_dist = x

            # 2. 收集左右大視野數據 (供決策使用)
            if 15.0 <= deg <= 45.0:
                left_distances.append(r)
            elif -45.0 <= deg <= -15.0:
                right_distances.append(r)

        cmd = Twist()
        
        if obstacle_detected:
            cmd.linear.x = 0.05   # 先煞車
            
            # ★ 核心防呆：如果「還沒決定方向」，才進行空間計算與鎖定
            if self.locked_turn_direction is None:
                avg_left = sum(left_distances) / len(left_distances) if len(left_distances) > 0 else 0.0
                avg_right = sum(right_distances) / len(right_distances) if len(right_distances) > 0 else 0.0
                
                # 決定並鎖定方向
                if avg_left >= avg_right:
                    self.locked_turn_direction = "left"
                else:
                    self.locked_turn_direction = "right"
                    
                self.get_logger().info(f"💡 決定轉向: 左邊({avg_left:.2f}m) vs 右邊({avg_right:.2f}m)")

            # ★ 依照記憶的方向，堅持轉到底
            if self.locked_turn_direction == "left":
                cmd.angular.z = 0.4
            else:
                cmd.angular.z = -0.4
                
            self.get_logger().warning(f"🚧 避障中：堅持向【{self.locked_turn_direction}】旋轉...")

        else:
            # ✅ 前方暢通，解除轉向記憶，繼續直走
            self.locked_turn_direction = None
            cmd.linear.x = 0.1167 
            cmd.angular.z = 0.0

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()