import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from utils import *

"""
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()
"""
class args(object):
    def __init__(self, path):
        self.path = path

# args = args(path='/home/antuser/Documents/MSCKF/boxes_6dof/01')
args = args(path='/home/antuser/Documents/MSCKF/Event_Cam_Flight/01')
# args = args(path='/home/antuser/Documents/MSCKF/drone_racing/01')

position = np.load(args.path + '/position.npz'); position = position.f.arr_0[1:, :]
gt_position = np.load(args.path + '/gt_position.npz'); gt_position = gt_position.f.arr_0[1:, :]
orientation = np.load(args.path + '/orientation.npz'); orientation = orientation.f.arr_0[1:, :]
gt_orientation = np.load(args.path + '/gt_orientation.npz'); gt_orientation = gt_orientation.f.arr_0[1:, :]
velocity = np.load(args.path + '/velocity.npz'); velocity = velocity.f.arr_0[1:, :]
gt_velocity = np.load(args.path + '/gt_velocity.npz'); gt_velocity = gt_velocity.f.arr_0[1:, :]
state_cov = np.load(args.path + '/state_cov.npz'); state_cov = state_cov.f.arr_0[1:, :]
timestamps = np.load(args.path + '/timestamps.npz'); timestamps = timestamps.f.arr_0
msckf_timestamps = np.load(args.path + '/msckf_timestamps.npz'); msckf_timestamps = msckf_timestamps.f.arr_0
acc_bias = np.load(args.path + '/acc_bias.npz'); acc_bias = acc_bias.f.arr_0[1:, :]
gyro_bias = np.load(args.path + '/gyro_bias.npz'); gyro_bias = gyro_bias.f.arr_0[1:, :]

gt_p_x = scipy.interpolate.CubicSpline(timestamps, gt_position[:, 0])
gt_p_y = scipy.interpolate.CubicSpline(timestamps, gt_position[:, 1])
gt_p_z = scipy.interpolate.CubicSpline(timestamps, gt_position[:, 2])
gt_position = np.column_stack((gt_p_x(msckf_timestamps), gt_p_y(msckf_timestamps), gt_p_z(msckf_timestamps)))

if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
    for i in range(orientation.shape[0]):
        orientation[i, :3] = R_to_rpy(to_rotation(orientation[i]).T)
    orientation[:, 2] = np.unwrap(orientation[:, 2])
    for i in range(orientation.shape[0]):
        orientation[i, :3] = orientation[i, :3] - np.array([-2.1887496, -0.0504558063, -3.0143040851])
        orientation[i, 2] = -orientation[i, 2]
    gt_o_x = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 0])
    gt_o_y = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 1])
    gt_o_z = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 2])
    gt_o_w = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 3])
    gt_orientation = np.column_stack((gt_o_x(msckf_timestamps), gt_o_y(msckf_timestamps), gt_o_z(msckf_timestamps),
                                      gt_o_w(msckf_timestamps)))
    for i in range(gt_orientation.shape[0]):
        gt_orientation[i, :3] = R_to_rpy(to_rotation(gt_orientation[i]).T)
    gt_orientation[:, 2] = np.unwrap(gt_orientation[:, 2])
    for i in range(gt_orientation.shape[0]):
        gt_orientation[i, :3] = gt_orientation[i, :3] - np.array([-2.1887496, -0.0504558063, -3.0143040851])
        gt_orientation[i, 2] = -gt_orientation[i, 2]

    titles = [('X', 'Pitch', 'X Error', 'Pitch Error'),
              ('Y', 'Yaw', 'Y Error', 'Yaw Error'),
              ('Z', 'Roll', 'Z Error', 'Roll Error')]

    state_titles = ['Pitch Error', 'Yaw Error', 'Roll Error',
                    'X Gyro Bias Error', 'Y Gyro Bias Error', 'Z Gyro Bias Error',
                    'X Velocity Error', 'Y Velocity Error', 'Z Velocity Error',
                    'X Acc Bias Error', 'Y Acc Bias Error', 'Z Acc Bias Error',
                    'X Error', 'Y Error', 'Z Error']

    dist = 69.803

elif args.path[:46] == '/home/antuser/Documents/MSCKF/Event_Cam_Flight':
    for i in range(orientation.shape[0]):
        orientation[i, :3] = quaternion_to_rpy(orientation[i])
    orientation[:, 2] = np.unwrap(orientation[:, 2])
    gt_o_x = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 0])
    gt_o_y = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 1])
    gt_o_z = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 2])
    gt_o_w = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 3])
    gt_orientation = np.column_stack((gt_o_x(msckf_timestamps), gt_o_y(msckf_timestamps), gt_o_z(msckf_timestamps),
                                      gt_o_w(msckf_timestamps)))
    for i in range(gt_orientation.shape[0]):
        gt_orientation[i, :3] = quaternion_to_rpy(gt_orientation[i])

    titles = [('North', 'Roll', 'North Error', 'Roll Error'),
              ('East', 'Pitch', 'East Error', 'Pitch Error'),
              ('Down', 'Yaw', 'Down Error', 'Yaw Error')]

    state_titles = ['Roll Error', 'Pitch Error', 'Yaw Error',
                    'North Gyro Bias Error', 'East Gyro Bias Error', 'Down Gyro Bias Error',
                    'North Velocity Error', 'East Velocity Error', 'Down Velocity Error',
                    'North Acc Bias Error', 'East Acc Bias Error', 'Down Acc Bias Error',
                    'North Error', 'East Error', 'Down Error']

    dist = 1072.818

else:
    for i in range(orientation.shape[0]):
        orientation[i, :3] = quaternion_to_rpy(orientation[i])
    # orientation[:, 2] = np.unwrap(orientation[:, 2])
    for i in range(orientation.shape[0]):
        orientation[i, :3] = orientation[i, :3] - np.array([-1.5396149, 0.0028283, 2.5325203])
    gt_o_x = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 0])
    gt_o_y = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 1])
    gt_o_z = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 2])
    gt_o_w = scipy.interpolate.CubicSpline(timestamps, gt_orientation[:, 3])
    gt_orientation = np.column_stack((gt_o_x(msckf_timestamps), gt_o_y(msckf_timestamps), gt_o_z(msckf_timestamps),
                                      gt_o_w(msckf_timestamps)))
    for i in range(gt_orientation.shape[0]):
        gt_orientation[i, :3] = quaternion_to_rpy(gt_orientation[i])
    # gt_orientation[:, 2] = np.unwrap(gt_orientation[:, 2])
    for i in range(gt_orientation.shape[0]):
        gt_orientation[i, :3] = gt_orientation[i, :3] - np.array([-1.5396149, 0.0028283, 2.5325203])

    titles = [('X', 'Pitch', 'X Error', 'Pitch Error'),
              ('Y', 'Roll', 'Y Error', 'Roll Error'),
              ('Z', 'Yaw', 'Z Error', 'Yaw Error')]

    state_titles = ['Pitch Error', 'Roll Error', 'Yaw Error',
                    'X Gyro Bias Error', 'Y Gyro Bias Error', 'Z Gyro Bias Error',
                    'X Velocity Error', 'Y Velocity Error', 'Z Velocity Error',
                    'X Acc Bias Error', 'Y Acc Bias Error', 'Z Acc Bias Error',
                    'X Error', 'Y Error', 'Z Error']

    dist = 336.32

gt_v_x = scipy.interpolate.CubicSpline(timestamps, gt_velocity[:, 0])
gt_v_y = scipy.interpolate.CubicSpline(timestamps, gt_velocity[:, 1])
gt_v_z = scipy.interpolate.CubicSpline(timestamps, gt_velocity[:, 2])
gt_velocity = np.column_stack((gt_v_x(msckf_timestamps), gt_v_y(msckf_timestamps), gt_v_z(msckf_timestamps)))

RMSE = np.sqrt(np.mean(np.sum((position - gt_position)**2, axis=-1)))
final_error = np.sqrt(np.sum((position[-1, :] - gt_position[-1, :])**2))

for i, title in enumerate(titles):
    p_error = position[:, i] - gt_position[:, i]
    o_error = (orientation[:, i] - gt_orientation[:, i]) * 180 / np.pi
    v_error = velocity[:, i] - gt_velocity[:, i]

    plt.figure()
    plt.plot(msckf_timestamps, position[:, i], label='Estimate')
    plt.plot(msckf_timestamps, gt_position[:, i], 'r', label='Truth')
    plt.xticks(np.arange(min(msckf_timestamps), max(msckf_timestamps)+5, step=10), fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Position [m]', fontweight='bold', fontsize=12)
    plt.xlabel('Time [sec]', fontweight='bold', fontsize=12)
    plt.title(title[0], fontweight='bold', fontsize=14)
    plt.legend(prop={'weight': 'bold'})
    if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
        if i == 0:
            plt.ylim(-1.4, 1.1)
            plt.savefig(args.path + '/boxesXSIFT.png')
        elif i == 1:
            plt.ylim(-1.2, 0.5)
            plt.savefig(args.path + '/boxesYSIFT.png')
        else:
            plt.ylim(-0.4, 3.4)
            plt.savefig(args.path + '/boxesZSIFT.png')
    elif args.path[:46] == '/home/antuser/Documents/MSCKF/Event_Cam_Flight':
        if i == 0:
            plt.ylim(-25, 700)
            plt.savefig(args.path + '/UAVNorthSIFT.png')
        elif i == 1:
            plt.ylim(-550, 25)
            plt.savefig(args.path + '/UAVEastSIFT.png')
        else:
            plt.ylim(-8, 6)
            plt.savefig(args.path + '/UAVDownSIFT.png')
    else:
        plt.savefig(args.path + '/position{}.png'.format(i))
    plt.close()

    plt.figure()
    plt.plot(msckf_timestamps, orientation[:, i]*180/np.pi, label='Estimate')
    plt.plot(msckf_timestamps, gt_orientation[:, i]*180/np.pi, 'r', label='Truth')
    plt.xticks(np.arange(min(msckf_timestamps), max(msckf_timestamps)+5, step=10), fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Orientation [deg]', fontweight='bold', fontsize=12)
    plt.xlabel('Time [sec]', fontweight='bold', fontsize=12)
    plt.title(title[1], fontweight='bold', fontsize=14)
    plt.legend(prop={'weight': 'bold'})
    if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
        plt.savefig(args.path + '/orientation{}.png'.format(i))
        if i == 0:
            plt.savefig(args.path + '/boxesPitchSIFT.png')
        elif i == 1:
            plt.savefig(args.path + '/boxesYawSIFT.png')
        else:
            plt.savefig(args.path + '/boxesRollSIFT.png')
    elif args.path[:46] == '/home/antuser/Documents/MSCKF/Event_Cam_Flight':
        if i == 0:
            plt.savefig(args.path + '/UAVRollSIFT.png')
        elif i == 1:
            plt.savefig(args.path + '/UAVPitchSIFT.png')
        else:
            plt.savefig(args.path + '/UAVYawSIFT.png')
    else:
        plt.savefig(args.path + '/orientation{}.png'.format(i))
    plt.close()

    plt.figure()
    plt.plot(msckf_timestamps, velocity[:, i], label='Estimate')
    plt.plot(msckf_timestamps, gt_velocity[:, i], 'r', label='Truth')
    plt.xticks(np.arange(min(msckf_timestamps), max(msckf_timestamps)+5, step=10), fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Velocity [m/s]', fontweight='bold', fontsize=12)
    plt.xlabel('Time [sec]', fontweight='bold', fontsize=12)
    plt.title(title[0], fontweight='bold', fontsize=14)
    plt.legend(prop={'weight': 'bold'})
    plt.savefig(args.path + '/velocity{}.png'.format(i))
    plt.close()

    plt.figure()
    plt.plot(msckf_timestamps, p_error)
    plt.xticks(np.arange(min(msckf_timestamps), max(msckf_timestamps)+5, step=10), fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Error [m]', fontweight='bold', fontsize=12)
    plt.xlabel('Time [sec]', fontweight='bold', fontsize=12)
    plt.title(title[2] + ' (RMSE: {} m, Final Error: {} m)'.format(
        np.round(RMSE, 3), np.round(final_error, 3)), fontweight='bold', fontsize=14)
    plt.savefig(args.path + '/error{}.png'.format(i))
    plt.close()

    plt.figure()
    plt.plot(msckf_timestamps, o_error)
    plt.xticks(np.arange(min(msckf_timestamps), max(msckf_timestamps)+5, step=10), fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Error [deg]', fontweight='bold', fontsize=12)
    plt.xlabel('Time [sec]', fontweight='bold', fontsize=12)
    MAE = np.abs(np.mean(o_error))
    plt.title(title[3] + ' (MAE: {} deg)'.format(np.round(MAE, 3)), fontweight='bold', fontsize=14)
    plt.savefig(args.path + '/orientation_error{}.png'.format(i))
    plt.close()

for i, title in enumerate(state_titles):
    plt.figure()
    if i < 3:
        o_error = (orientation[:, i] - gt_orientation[:, i]) * 180 / np.pi
        plt.plot(msckf_timestamps, o_error, label='Error')
        plt.plot(msckf_timestamps, np.sqrt(state_cov[:, i]) * 3 * 180/np.pi, 'r', label='3-sigma bounds')
        plt.plot(msckf_timestamps, -np.sqrt(state_cov[:, i]) * 3 * 180/np.pi, 'r')
        plt.ylabel('Error [deg]', fontweight='bold', fontsize=12)
        if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
            plt.ylim(-40, 40)
        elif args.path[:46] == '/home/antuser/Documents/MSCKF/Event_Cam_Flight':
            plt.ylim(-200, 200)
        plt.title(title, fontweight='bold', fontsize=14)
    elif i > 2 and i < 6:
        plt.plot(msckf_timestamps, gyro_bias[:, i-3], label='Estimate')
        plt.plot(msckf_timestamps, gyro_bias[:, i-3] + np.sqrt(state_cov[:, i]) * 3, 'r', label='3-sigma bounds')
        plt.plot(msckf_timestamps, gyro_bias[:, i-3] - np.sqrt(state_cov[:, i]) * 3, 'r')
        plt.ylabel('Estimate [rad/s]', fontweight='bold', fontsize=12)
        if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
            plt.ylim(-0.1, 0.1)
        plt.title(title, fontweight='bold', fontsize=14)
    elif i > 5 and i < 9:
        v_error = velocity[:, i - 6] - gt_velocity[:, i - 6]
        plt.plot(msckf_timestamps, v_error, label='Error')
        plt.plot(msckf_timestamps, np.sqrt(state_cov[:, i]) * 3, 'r', label='3-sigma bounds')
        plt.plot(msckf_timestamps, -np.sqrt(state_cov[:, i]) * 3, 'r')
        plt.ylabel('Error [m/s]', fontweight='bold', fontsize=12)
        if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
            plt.ylim(-2.5, 2.5)
        elif args.path[:46] == '/home/antuser/Documents/MSCKF/Event_Cam_Flight':
            if i == 6 or i == 7:
                plt.ylim(-325, 325)
            else:
                plt.ylim(-50, 50)
        plt.title(title, fontweight='bold', fontsize=14)
        pass
    elif i > 8 and i < 12:
        plt.plot(msckf_timestamps, acc_bias[:, i-9], label='Estimate')
        plt.plot(msckf_timestamps, acc_bias[:, i-9] + np.sqrt(state_cov[:, i]) * 3, 'r', label='3-sigma bounds')
        plt.plot(msckf_timestamps, acc_bias[:, i-9] - np.sqrt(state_cov[:, i]) * 3, 'r')
        plt.ylabel('Estimate [m/s^2]', fontweight='bold', fontsize=12)
        plt.title(title, fontweight='bold', fontsize=14)
    elif i > 11:
        p_error = position[:, i - 12] - gt_position[:, i - 12]
        plt.plot(msckf_timestamps, p_error, label='Error')
        plt.plot(msckf_timestamps, np.sqrt(state_cov[:, i]) * 3, 'r', label='3-sigma bounds')
        plt.plot(msckf_timestamps, -np.sqrt(state_cov[:, i]) * 3, 'r')
        plt.ylabel('Error [m]', fontweight='bold', fontsize=12)
        if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
            plt.ylim(-8, 8)
        elif args.path[:46] == '/home/antuser/Documents/MSCKF/Event_Cam_Flight':
            if i == 12 or i == 13:
                plt.ylim(-4750, 4750)
            else:
                plt.ylim(-500, 500)
        plt.title(title, fontweight='bold', fontsize=14)

    plt.xticks(np.arange(min(msckf_timestamps), max(msckf_timestamps)+5, step=10), fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlabel('Time [sec]', fontweight='bold', fontsize=12)
    plt.legend(prop={'weight': 'bold'})
    if args.path[:40] == '/home/antuser/Documents/MSCKF/boxes_6dof':
        if i == 0:
            plt.savefig(args.path + '/boxesPitchErrorSIFT.png')
        elif i == 1:
            plt.savefig(args.path + '/boxesYawErrorSIFT.png')
        elif i == 2:
            plt.savefig(args.path + '/boxesRollErrorSIFT.png')
        elif i == 3:
            plt.savefig(args.path + '/boxesXGyroSIFT.png')
        elif i == 4:
            plt.savefig(args.path + '/boxesYGyroSIFT.png')
        elif i == 5:
            plt.savefig(args.path + '/boxesZGyroSIFT.png')
        elif i == 6:
            plt.savefig(args.path + '/boxesVXErrorSIFT.png')
        elif i == 7:
            plt.savefig(args.path + '/boxesVYErrorSIFT.png')
        elif i == 8:
            plt.savefig(args.path + '/boxesVZErrorSIFT.png')
        elif i == 9:
            plt.savefig(args.path + '/boxesXAccSIFT.png')
        elif i == 10:
            plt.savefig(args.path + '/boxesYAccSIFT.png')
        elif i == 11:
            plt.savefig(args.path + '/boxesZAccSIFT.png')
        elif i == 12:
            plt.savefig(args.path + '/boxesXErrorSIFT.png')
        elif i == 13:
            plt.savefig(args.path + '/boxesYErrorSIFT.png')
        elif i == 14:
            plt.savefig(args.path + '/boxesZErrorSIFT.png')
    elif args.path[:46] == '/home/antuser/Documents/MSCKF/Event_Cam_Flight':
        if i == 0:
            plt.savefig(args.path + '/UAVRollErrorSIFT.png')
        elif i == 1:
            plt.savefig(args.path + '/UAVPitchErrorSIFT.png')
        elif i == 2:
            plt.savefig(args.path + '/UAVYawErrorSIFT.png')
        elif i == 3:
            plt.savefig(args.path + '/UAVNorthGyroSIFT.png')
        elif i == 4:
            plt.savefig(args.path + '/UAVEastGyroSIFT.png')
        elif i == 5:
            plt.savefig(args.path + '/UAVDownGyroSIFT.png')
        elif i == 6:
            plt.savefig(args.path + '/UAVVNorthErrorSIFT.png')
        elif i == 7:
            plt.savefig(args.path + '/UAVVEastErrorSIFT.png')
        elif i == 8:
            plt.savefig(args.path + '/UAVVDownErrorSIFT.png')
        elif i == 9:
            plt.savefig(args.path + '/UAVNorthAccSIFT.png')
        elif i == 10:
            plt.savefig(args.path + '/UAVEastAccSIFT.png')
        elif i == 11:
            plt.savefig(args.path + '/UAVDownAccSIFT.png')
        elif i == 12:
            plt.savefig(args.path + '/UAVNorthErrorSIFT.png')
        elif i == 13:
            plt.savefig(args.path + '/UAVEastErrorSIFT.png')
        elif i == 14:
            plt.savefig(args.path + '/UAVDownErrorSIFT.png')
    else:
        plt.savefig(args.path + '/state{}.png'.format(i))
    plt.close()

perc_error = np.array([])
for i in range(position.shape[0]-1):
    perc_error = np.append(perc_error, np.linalg.norm(position[i+1, :] - gt_position[i+1, :]) / dist)
percent_error = np.mean(perc_error)*100
print('Percent Error: {}%'.format(np.round(percent_error, 3)))
print('RMSE: {}'.format(np.round(RMSE, 3)))
print('Final Error: {}'.format(np.round(final_error, 3)))