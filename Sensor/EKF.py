import math
import numpy as np

class EKF:
    def __init__(self):
        # 상태 벡터: [x, y, v_x, v_y, yaw]
        self.x = np.zeros((5, 1))
        self.P = np.eye(5) * 1000.0
        self.Q = np.diag([0.05, 0.05, 0.5, 0.5, 0.01])
        self.R_gps = np.diag([3.0, 3.0])
        
        self.is_initialized = False

    def predict(self, dt, accel_x, accel_y, yaw_rate):
        px = self.x[0, 0]
        py = self.x[1, 0]
        vx = self.x[2, 0]
        vy = self.x[3, 0]
        yaw = self.x[4, 0]

        # 물리 운동학 모델 기반 상태 예측
        self.x[0, 0] = px + vx * dt
        self.x[1, 0] = py + vy * dt
        self.x[2, 0] = vx + accel_x * dt
        self.x[3, 0] = vy + accel_y * dt
        
        # Yaw 각도 정규화
        new_yaw = yaw + yaw_rate * dt
        while new_yaw > math.pi:
            new_yaw -= 2.0 * math.pi
        while new_yaw < -math.pi:
            new_yaw += 2.0 * math.pi
        self.x[4, 0] = new_yaw

        # 야코비안 F 행렬
        F = np.array([
            [1, 0, dt,  0, 0],
            [0, 1,  0, dt, 0],
            [0, 0,  1,  0, 0],
            [0, 0,  0,  1, 0],
            [0, 0,  0,  0, 1]
        ])
        
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, z_x, z_y):
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        z = np.array([[z_x], [z_y]])
        
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        y = z - H @ self.x
        self.x = self.x + K @ y
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
    
   
def latlon_to_xy(lat, lon, ref_lat=37.239, ref_lon=126.773):
    lat_to_m = 111000.0
    lon_to_m = 111000.0 * math.cos(math.radians(ref_lat))
    x = (lon - ref_lon) * lon_to_m
    y = (lat - ref_lat) * lat_to_m
    return x, y
    
print(""x": float(self.x[0, 0]), "y": float(self.x[1, 0]),
            "vx": float(self.x[2, 0]),
            "vy": float(self.x[3, 0]),
            "yaw_rad": float(self.x[4, 0]),
            "yaw_deg": float(math.degrees(self.x[4, 0])))


