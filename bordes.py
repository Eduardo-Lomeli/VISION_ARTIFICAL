import cv2
import numpy as np
imagen=cv2.imread("pangolin.jpg",0)
m,n =imagen.shape
kernel_dx=np.array([[1,1,1],[-1,-1,-1],[0,0,0]])
kernel_dy=np.array([[0,-1,1],[0,-1,1],[0,-1,1]])
bordes_dx=np.zeros_like(imagen)
bordes_dy=np.zeros_like(imagen)

for i in range(m-2):
    for j in range(n-2):
        res_dx=np.sum(imagen[i:i+3,j:j+3]*kernel_dx)
        res_dy=np.sum(imagen[i:i+3,j:j+3]*kernel_dy)
        if res_dx>40:
            bordes_dx[i,j]=255
        if res_dy>40:
            bordes_dy[i,j]=255
bordes = np.abs(bordes_dx) + np.abs(bordes_dy)
cv2.imshow("bordes",bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("end")