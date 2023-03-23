"""
Source page: https://data.nasa.gov/Aerospace/Deorbit-Descent-and-Landing-Flight-1-DDL-F1-/vicw-ivgd
Download URL: https://data.nasa.gov/download/vicw-ivgd/application%2Fx-zip-compressed
MD5Sum of data: be645485bfdbdc0edf184bd85d355e37
"""
from matplotlib import pyplot as plt
import pandas as pd

lidar_data = pd.read_csv("Data/commercial_lidar.csv")
lidar_data.columns = lidar_data.columns.str.strip()

print("Original")
print(lidar_data.keys())
print(lidar_data)


lidar_data["TIME_NANOSECONDS_TAI"] = lidar_data["TIME_NANOSECONDS_TAI"].apply(
    lambda x: int(x)
)

print("Parsed")
print(lidar_data)

for i in range(0, 4):
    plt.plot(
        lidar_data["TIME_NANOSECONDS_TAI"] * 1e-9,
        lidar_data["OMPS_Range_M[%d]" % (i + 1)],
        label="Range[%d]" % (i + 1),
    )
plt.legend()
plt.show()
