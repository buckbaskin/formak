"""
Source page: https://data.nasa.gov/Aerospace/Deorbit-Descent-and-Landing-Flight-1-DDL-F1-/vicw-ivgd
Download URL: https://data.nasa.gov/download/vicw-ivgd/application%2Fx-zip-compressed
MD5Sum of data: be645485bfdbdc0edf184bd85d355e37
"""
from math import sqrt
from matplotlib import pyplot as plt
import pandas as pd

REFERENCE_TIME = 1602596210210000000

lidar_data = pd.read_csv("Data/commercial_lidar.csv")
lidar_data.columns = lidar_data.columns.str.strip()
lidar_data["TIME_NANOSECONDS_TAI"] = lidar_data["TIME_NANOSECONDS_TAI"].apply(
    lambda x: int(x) - REFERENCE_TIME
)

imu_data = pd.read_csv("Data/dlc.csv")
imu_data.columns = imu_data.columns.str.strip()
imu_data["TIME_NANOSECONDS_TAI"] = imu_data["TIME_NANOSECONDS_TAI"].apply(
    lambda x: int(x) - REFERENCE_TIME
)

imu_x = imu_data["DATA_DELTA_VEL[1]"]
imu_y = imu_data["DATA_DELTA_VEL[2]"]
imu_z = imu_data["DATA_DELTA_VEL[3]"]
total_imu = imu_x * imu_x + imu_y * imu_y + imu_z * imu_z
total_imu = total_imu.apply(sqrt)

print("Lidar")
print(lidar_data.keys())

print("IMU")
print(imu_data.keys())
print(imu_data)

UPDATE_RATE_HZ = 50

for i in range(0, 3):
    plt.plot(
        imu_data["TIME_NANOSECONDS_TAI"] * 1e-9,
        imu_data["DATA_DELTA_VEL[%d]" % (i + 1)] * UPDATE_RATE_HZ,
        label="%d" % (i + 1),
    )

plt.plot(
    imu_data["TIME_NANOSECONDS_TAI"] * 1e-9,
    total_imu * UPDATE_RATE_HZ,
    label="total magnitude delta v",
)
plt.xlabel("seconds")
plt.ylabel("meters / sec2 (approx, from delta v)")
plt.legend()
plt.show()
