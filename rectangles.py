import numpy as np
import cv2

""" Generate Rectangles """
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

def get_rectangles(NUM_IMAGES, HEIGHT, WIDTH):  
    imgs_input = []
    imgs_output = []
    theta_list = []
    rec_size_output = (30,30)
    for x, y, theta in zip(np.random.rand(NUM_IMAGES)*WIDTH, np.random.rand(NUM_IMAGES)*HEIGHT, np.random.rand(NUM_IMAGES)*90):
        random_size = int(np.random.rand(1)*20 + 20)   # random integer number between 20 and 40 
        rec_size_input = (random_size, random_size)
        center_input = np.array([[x], [y]])
        center_output = np.array([[64], [64]])
        theta_list.append(theta*np.pi/180)
        img_input = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
        img_output = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
        cv2.fillPoly(img_input,  [get_points(rec_size_input,  center_input,  theta)], color=(255,255,255))
        cv2.fillPoly(img_output, [get_points(rec_size_output, center_output, theta)], color=(255,255,255))
        imgs_input.append(img_input)
        imgs_output.append(img_output)
    return np.array(imgs_input), np.array(imgs_output), np.array(theta_list, dtype=np.float32)

def generate_dataset(BATCH_SIZE):
    '''
    Input image: rectangle images of various {position, size, rotation}.
    Output image: rectangle images of one {position, size} but various rotation angles as input image.
    '''
    HEIGHT, WIDTH = 128, 128
    images_input, images_output, theta_list = get_rectangles(BATCH_SIZE, HEIGHT, WIDTH)
    images_input = images_input.astype(np.float32) / 255.
    images_output = images_output.astype(np.float32) / 255.
    images_input[images_input >= .5] = 1.
    images_input[images_input < .5] = 0.
    images_output[images_output >= .5] = 1.
    images_output[images_output < .5] = 0.
    # dataset = tf.data.Dataset.from_tensor_slices(images).batch(BATCH_SIZE)    
    
    # return images.reshape((-1,HEIGHT*WIDTH))
    return images_input, images_output, theta_list