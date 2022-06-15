import numpy as np
import matplotlib.pyplot as plt

matP = np.array([
    [2.0, 4.0, 2.0],
    [6.0, 3.0, 3.0],
    [1.0, 2.0, 0.5],
    [16.0, 8.0, 4.0]
], dtype=np.float64)

matL = np.array([8, -4, 0], dtype=np.float64)

D = np.dot(matP, matL)

print(f"P: {matP}\n")
print(f"L: {matL}\n")
print(f"P . L = {D}\n")

print("The following points are on line (8, -4, 0)")
for i in range(len(D)):
    if D[i] == 0:
        x = matP[i][0] / matP[i][2]
        y = matP[i][1] / matP[i][2]
        print(f"({x}, {y})")

X = [matP[i][0] / matP[i][2] for i in range(4)]
Y = [matP[i][1] / matP[i][2] for i in range(4)]
lineX = np.linspace(0, 4, 100)
lineY = (-matL[0] * lineX - matL[2]) / matL[1]

plt.plot(X, Y, "r*")
plt.plot(lineX, lineY)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("2D line (8, -4, 0) and Four Homogeneous 2D points")
plt.show()
