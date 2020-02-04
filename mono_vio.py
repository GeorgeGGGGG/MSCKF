from queue import Queue
from threading import Thread
from mono_config import Config
from mono_image import ImageProcessor
from mono_msckf import MSCKF
from utils import *
import numpy as np


class VIO(object):
    def __init__(self, config, img_queue, imu_queue, gt_queue, run, events, viewer=None):
        self.config = config
        self.viewer = viewer
        self.run = run
        self.events = events

        self.gt_position = np.zeros((1, 3))
        self.gt_orientation = np.zeros((1, 4))
        self.gt_velocity = np.zeros((1, 3))
        self.gt_gyro_bias = np.zeros((1, 3))
        self.gt_acc_bias = np.zeros((1, 3))
        self.timestamps = np.array([])
        self.initial_time = False
        self.first = False

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.gt_queue = gt_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config, self.run)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.mono_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            if not self.first:
                time.sleep(5)
                if self.events and self.run == 1:
                    gt = self.gt_queue.get()

            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return

            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)

            if not self.initial_time:
                self.initial_time = feature_msg.timestamp

            try:
                if self.run == 0:
                    if not self.first:
                        gt = self.gt_queue.get()
                        self.first = True
                    while gt.timestamp <= feature_msg.timestamp:
                        gt = self.gt_queue.get()
                        gt_position = self.msckf.initial_rot @ (gt.p - self.config.initial_position)
                        gt_orientation = gt.q
                        gt_pose = Isometry3d(self.msckf.initial_rot @ to_rotation(gt.q).T, gt_position)
                        gt_velocity = self.msckf.initial_rot @ gt.v
                        self.gt_position = np.append(self.gt_position, np.reshape(gt_position, (1, 3)), axis=0)
                        self.gt_orientation = np.append(self.gt_orientation, np.reshape(gt_orientation, (1, 4)), axis=0)
                        self.gt_velocity = np.append(self.gt_velocity, np.reshape(gt_velocity, (1, 3)), axis=0)
                        self.timestamps = np.append(self.timestamps, gt.timestamp - self.initial_time)

                elif self.run == 1:
                    if self.events:
                        while gt.timestamp <= feature_msg.timestamp:
                            if self.first:
                                gt = self.gt_queue.get()
                            gt_position = gt.p - self.config.initial_position
                            gt_orientation = gt.q
                            gt_pose = Isometry3d(to_rotation(gt.q).T, gt_position)
                            gt_velocity = gt.v
                            self.gt_position = np.append(self.gt_position, np.reshape(gt_position, (1, 3)), axis=0)
                            self.gt_orientation = np.append(self.gt_orientation, np.reshape(gt_orientation, (1, 4)),
                                                            axis=0)
                            self.gt_velocity = np.append(self.gt_velocity, np.reshape(gt_velocity, (1, 3)), axis=0)
                            self.timestamps = np.append(self.timestamps, gt.timestamp - self.initial_time)
                            self.first = True
                    else:
                        gt = self.gt_queue.get()
                        gt_position = gt.p - self.config.initial_position
                        gt_orientation = gt.q
                        gt_pose = Isometry3d(to_rotation(gt.q).T, gt_position)
                        gt_velocity = gt.v
                        self.gt_position = np.append(self.gt_position, np.reshape(gt_position, (1, 3)), axis=0)
                        self.gt_orientation = np.append(self.gt_orientation, np.reshape(gt_orientation, (1, 4)), axis=0)
                        self.gt_velocity = np.append(self.gt_velocity, np.reshape(gt_velocity, (1, 3)), axis=0)
                        self.timestamps = np.append(self.timestamps, gt.timestamp - self.initial_time)
                        self.first = True
                elif self.run == 2:
                    if not self.first:
                        gt = self.gt_queue.get()
                        self.first = True
                    while gt.timestamp <= feature_msg.timestamp:
                        gt = self.gt_queue.get()
                        gt_position = self.msckf.initial_rot @ (gt.p - self.config.initial_position)
                        gt_orientation = gt.q
                        gt_pose = Isometry3d(self.msckf.initial_rot @ to_rotation(gt.q).T, gt_position)
                        gt_velocity = self.msckf.initial_rot @ gt.v
                        self.gt_position = np.append(self.gt_position, np.reshape(gt_position, (1, 3)), axis=0)
                        self.gt_orientation = np.append(self.gt_orientation, np.reshape(gt_orientation, (1, 4)), axis=0)
                        self.gt_velocity = np.append(self.gt_velocity, np.reshape(gt_velocity, (1, 3)), axis=0)
                        self.timestamps = np.append(self.timestamps, gt.timestamp - self.initial_time)
            except AttributeError:
                if self.run == 0:
                    basefile = '/home/antuser/Documents/MSCKF/boxes_6dof'
                elif self.run == 1:
                    basefile = '/home/antuser/Documents/MSCKF/Event_Cam_Flight'
                elif self.run == 2:
                    basefile = '/home/antuser/Documents/MSCKF/drone_racing'

                np.savez_compressed(basefile + '/01' + '/position.npz', self.msckf.position)
                np.savez_compressed(basefile + '/01' + '/gt_position.npz', self.gt_position)
                np.savez_compressed(basefile + '/01' + '/orientation.npz', self.msckf.orientation)
                np.savez_compressed(basefile + '/01' + '/gt_orientation.npz', self.gt_orientation)
                np.savez_compressed(basefile + '/01' + '/velocity.npz', self.msckf.velocity)
                np.savez_compressed(basefile + '/01' + '/gt_velocity.npz', self.gt_velocity)
                np.savez_compressed(basefile + '/01' + '/state_cov.npz', self.msckf.state_cov_buffer)
                np.savez_compressed(basefile + '/01' + '/msckf_timestamps.npz', self.msckf.timestamps)
                np.savez_compressed(basefile + '/01' + '/timestamps.npz', self.timestamps)
                np.savez_compressed(basefile + '/01' + '/acc_bias.npz', self.msckf.acc_bias)
                np.savez_compressed(basefile + '/01' + '/gyro_bias.npz', self.msckf.gyro_bias)
                exit()

            if result is not None:
                error = np.sqrt(np.sum((self.msckf.initial_rot @ result.pose.t - gt_position) ** 2))
                print('Position Error: {} m'.format(np.round(error, 4)))
            if result is not None and self.viewer is not None:
                pose = Isometry3d(self.msckf.initial_rot @ result.pose.R, self.msckf.initial_rot @ result.pose.t)
                self.viewer.update_pose(pose, gt_pose)
        

if __name__ == '__main__':
    import time

    from mono_dataset import Dataset, DataPublisher
    from viewer import Viewer

    viewer = Viewer()
    run = 1
    events = False

    if run == 0:
        path = '/home/antuser/Documents/MSCKF/boxes_6dof'
        offset = 0.
    elif run == 1:
        path = '/home/antuser/Documents/MSCKF/Event_Cam_Flight'
        offset = 361.53
    elif run == 2:
        path = '/home/antuser/Documents/MSCKF/drone_racing'
        offset = 29.25

    dataset = Dataset(path)
    dataset.set_starttime(offset=offset)

    img_queue = Queue()
    imu_queue = Queue()
    gt_queue = Queue()

    config = Config(run, events)
    msckf_vio = VIO(config, img_queue, imu_queue, gt_queue, run, events, viewer=viewer)

    duration = float('inf')
    ratio = 0.4  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.mono, img_queue, duration, ratio)
    gt_publisher = DataPublisher(
        dataset.groundtruth, gt_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    gt_publisher.start(now)