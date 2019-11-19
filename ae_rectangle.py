import matplotlib.pyplot as plt 
import numpy as np 
import cv2
HEIGHT, WIDTH = 128, 128
img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
# points = np.array([[100, 100],[100,200],[200,200],[200,100]])
center = np.array([[300],[300]])
rec_size = (30,30)

def get_points(rec_size, center, theta):
    points = np.matrix([
                        [-rec_size[0]/2, -rec_size[1]/2],
                        [-rec_size[0]/2, +rec_size[1]/2],
                        [+rec_size[0]/2, +rec_size[1]/2],
                        [+rec_size[0]/2, -rec_size[1]/2]])
    points = np.vstack((points.transpose(), np.ones((1,4))))
    theta = theta*np.pi/180
    R = np.matrix([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta),  np.cos(theta)]])
    
    T = np.hstack((R, center))
    T = np.vstack((T, np.array([0, 0, 1])))
    points = T*points
    return points[:2].transpose().astype(np.int32)
for x, y, theta in zip(np.random.rand(10)*128, np.random.rand(10)*128, np.random.rand(10)*90):
    center = np.array([[x], [y]])
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.fillPoly(img, [get_points(rec_size, center, theta)], color=(255,255,255))
    cv2.imshow('img', img)
    # cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()