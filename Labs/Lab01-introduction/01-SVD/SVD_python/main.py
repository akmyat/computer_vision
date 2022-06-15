"""
Find the SVD of a matrix
"""

import numpy as np
import cv2

matA = np.array([
    [3.0, 2.0, 4.0],
    [8.0, 4.0, 2.0],
    [1.0, 3.0, 2.0]
])

w, u, vt = cv2.SVDecomp(matA)

print(f"A: \n{matA}\nU: \n{u}\nW: \n{w}\nVt: \n{vt}")