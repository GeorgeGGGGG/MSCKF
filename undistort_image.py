import cv2 as cv
import glob
import numpy as np

cameraResolution = [240, 180]
# cameraResolution = [346, 260]
imgsize = tuple(cameraResolution)
cameraCalMat = np.array([[198.444, 0., 104.829], [0., 198.826, 92.838], [0., 0., 1.]])
distCoeffs = np.array([-0.394, 0.156, -0.000125, 0.001629])
# cameraCalMat = np.array([[199.092366542, 0., 132.192071378], [0., 198.82882047, 110.712660011], [0., 0., 1.]])
# distCoeffs = np.array([-0.368436311798, 0.150947243557, -0.000296130534385, -0.000759431726241])
# cameraCalMat = np.array([[172.98992850734132, 0., 163.33639726024606],
#                          [0., 172.98303181090185, 134.99537889030861],
#                          [0., 0., 1.]])
# distCoeffs = np.array([-0.027576733308582076, -0.006593578674675004, 0.0008566938165177085, -0.00030899587045247486])
mapx, mapy = cv.initUndistortRectifyMap(cameraCalMat, distCoeffs, None, cameraCalMat, imgsize, 5)
# mapx, mapy = cv.fisheye.initUndistortRectifyMap(cameraCalMat, distCoeffs, None, cameraCalMat, imgsize, 5)
images = glob.glob('/home/antuser/Documents/MSCKF/Event_Cam_Flight/mav0/cam0/data/*.png')
undistorted_images = '/home/antuser/Documents/MSCKF/Event_Cam_Flight/mav0/cam0/undistorted_data'
images_events = glob.glob('/home/antuser/Documents/MSCKF/Event_Cam_Flight/mav0/cam0/reconstruction/*.png')
undistorted_images_events = '/home/antuser/Documents/MSCKF/Event_Cam_Flight/mav0/cam0/undistorted_reconstruction'
# images = glob.glob('/home/antuser/Documents/MSCKF/boxes_6dof/mav0/cam0/data/*.png')
# undistorted_images = '/home/antuser/Documents/MSCKF/boxes_6dof/mav0/cam0/undistorted_data'
# images_events = glob.glob('/home/antuser/Documents/MSCKF/boxes_6dof/mav0/cam0/reconstruction/*.png')
# undistorted_images_events = '/home/antuser/Documents/MSCKF/boxes_6dof/mav0/cam0/undistorted_reconstruction'
# images = glob.glob('/home/antuser/Documents/MSCKF/drone_racing/mav0/cam0/data/*.png')
# undistorted_images = '/home/antuser/Documents/MSCKF/drone_racing/mav0/cam0/undistorted_data'
# images_events = glob.glob('/home/antuser/Documents/MSCKF/drone_racing/mav0/cam0/reconstruction/*.png')
# undistorted_images_events = '/home/antuser/Documents/MSCKF/drone_racing/mav0/cam0/undistorted_reconstruction'

for image in images:
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    undistorted_img = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    cv.imwrite(undistorted_images + '/' + image[-23:], undistorted_img)

for image in images_events:
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    undistorted_img = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    cv.imwrite(undistorted_images_events + '/' + image[-20:], undistorted_img)