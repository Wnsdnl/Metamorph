def is_aggressive_driving(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    # 급가속 또는 급감속 감지 (예: 전방 또는 후방으로의 급격한 속도 변화)
    if abs(acc_x) >= 4.8:  # 약 0.5g
        return True

    # 급회전 감지 (예: 급격한 방향 전환)
    if abs(gyro_z) >= 0.5:  # 약 28.6도/s
        return True

    # 급피치 또는 급롤 감지 (예: 차량의 앞뒤 또는 좌우로의 급격한 기울기 변화)
    if abs(gyro_x) >= 0.3 or abs(gyro_y) >= 0.3:  # 약 17.2도/s
        return True

    return False

print(is_aggressive_driving(0.6404054,0.049428344,0.22680283,-0.04642576,-0.11522446,-0.16508633))
print(is_aggressive_driving(1.8807062,-0.52159345,-0.025224686,0.12522738,0.069867715,-0.047800206))