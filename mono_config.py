import numpy as np
import cv2
from utils import *


class OptimizationConfig(object):
    """
    Configuration parameters for 3d feature position optimization.
    """
    def __init__(self):
        self.translation_threshold = -1.0   # 0.2
        self.huber_epsilon = 0.01
        self.estimation_precision = 5e-7
        self.initial_damping = 1e-3
        self.outer_loop_max_iteration = 5  # 10
        self.inner_loop_max_iteration = 5  # 10

class Config(object):
    def __init__(self, run, events):
        # feature position optimization
        self.optimization_config = OptimizationConfig()

        # image processor
        self.run = run
        self.events = events
        self.model_sp = False
        self.model = False
        self.model_mp = False
        self.grid_row = 4
        self.grid_col = 5
        self.grid_num = self.grid_row * self.grid_col
        self.grid_min_feature_num = 3
        self.grid_max_feature_num = 5
        self.fast_threshold = 15
        self.ransac_threshold = 3
        self.stereo_threshold = 5
        self.max_iteration = 30
        self.track_precision = 0.01
        self.pyramid_levels = 3
        self.patch_size = 15
        self.win_size = (self.patch_size, self.patch_size)

        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.pyramid_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                self.max_iteration, 
                self.track_precision),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

        # MSCKF VIO #
        # gravity
        self.gravity_acc = 9.81
        self.gravity = np.array([0.0, 0.0, -self.gravity_acc])

        # Framte rate of the images. This variable is only used to
        # determine the timing threshold of each iteration of the filter.
        self.frame_rate = 20

        # Maximum number of camera states to be stored
        self.max_cam_state_size = 20

        # The position uncertainty threshold is used to determine
        # when to reset the system online. Otherwise, the ever-increasing
        # uncertainty will make the estimation unstable.
        # Note this online reset will be some dead-reckoning.
        # Set this threshold to nonpositive to disable online reset.
        self.position_std_threshold = -1.  # 8.0

        # Threshold for determine keyframes
        self.rotation_threshold = 0.2618
        self.translation_threshold = 0.4
        self.tracking_rate_threshold = 0.5

        # Noise related parameters (Use variance instead of standard deviation)
        self.gyro_noise = 0.005 ** 2  # 0.05
        self.acc_noise = 0.05 ** 2  # 0.1
        self.gyro_bias_noise = 0.001 ** 2  # 0.00004
        self.acc_bias_noise = 0.01 ** 2  # 0.002
        self.observation_noise = 3. ** 2

        # initial state
        self.velocity = np.zeros(3)

        # The initial covariance of orientation and position can be
        # set to 0. But for velocity, bias and extrinsic parameters, 
        # there should be nontrivial uncertainty.
        self.velocity_cov = 0.25
        self.gyro_bias_cov = 0.01
        self.acc_bias_cov = 0.01
        self.extrinsic_rotation_cov = 3.0462e-4
        self.extrinsic_translation_cov = 2.5e-5

        # calibration parameters #
        # T_imu_cam: takes a vector from the IMU frame to the cam frame.
        # T_cn_cnm1: takes a vector from the cam0 frame to the cam1 frame.

        # Boxes 6DOF
        if self.run == 0:
            self.T_imu_cam0 = np.identity(4)
            self.cam0_camera_model = 'pinhole'
            self.cam0_distortion_model = 'radtan'
            self.cam0_distortion_coeffs = np.array([0., 0., 0., 0.])
                # np.array([-0.368436311798, 0.150947243557, -0.000296130534385, -0.000759431726241])
            self.cam0_intrinsics = np.array([199.092366542, 198.82882047, 132.192071378, 110.712660011])
            self.cam0_resolution = np.array([240, 180])
            self.initial_orientation = np.array([-0.0680457284243, 0.885824586918, -0.458953710403, 0.00678761831174])
            self.initial_position = np.array([0.39745276688, 1.46356729429, 1.2767266244])

        # Event Flight
        elif self.run == 1:
            self.T_imu_cam0 = np.array([
                [1., -0.005790, 0.007779, -0.004451],
                [0.005880, 1., -0.011512, 0.008024],
                [-0.007712, 0.011557, 1., 0.020438],
                [0., 0., 0., 1.]])
            self.cam0_camera_model = 'pinhole'
            self.cam0_distortion_model = 'radtan'
            self.cam0_distortion_coeffs = np.array([0., 0., 0., 0.])  # np.array([-0.394, 0.156, -0.000125, 0.001629])
            self.cam0_intrinsics = np.array([198.444, 198.826, 104.829, 92.838])
            self.cam0_resolution = np.array([240, 180])
            if not self.events:
                self.initial_orientation = np.array(
                    [-0.025622201371903, 0.016181224433693, -0.08440608216843, 0.995970523693177])
                self.initial_position = np.array([0.732395756700792, -0.126372904754554, -0.002873209679708])
                self.velocity = np.array([27.6274266778717, -4.73796917796225, -0.190296930214541])
            else:
                self.initial_orientation = np.array(
                    [-0.024830755669052, 0.012087722569238, -0.083425861657053, 0.996131239417162])
                self.initial_position = np.array([5.49267424896765, -0.942013829389926, -0.03681926690838])
                self.velocity = np.array([27.679393142282, -4.68366748294387, -0.194618450783459])

        elif self.run == 2:
            self.T_imu_cam0 = np.array([[0.99997115, 0.0013817, -0.00746962, 0.0001805],
                                        [-0.00140853, 0.99999257, -0.00358775, -0.00431635],
                                        [0.0074646, 0.00359816, 0.99996567, -0.02754739],
                                        [0., 0., 0., 1.]])
            self.cam0_camera_model = 'pinhole'
            self.cam0_distortion_model = 'equidistant'
            self.cam0_distortion_coeffs = np.array([0., 0., 0., 0.])
            # np.array([-0.027576733308582076, -0.006593578674675004, 0.0008566938165177085, -0.00030899587045247486])
            self.cam0_intrinsics = np.array([172.98992850734132, 172.98303181090185,
                                             163.33639726024606, 134.99537889030861])
            self.cam0_resolution = np.array([346, 260])
            if not self.events:
                self.initial_orientation = np.array(
                    [-0.20966369545534, -0.663666343042566, 0.685299130367224, 0.214367026005504])
                self.initial_position = np.array([7.37212827568338, 4.0273003467798, -1.15113085482056])
                self.velocity = np.array([-0.092953177411861, -0.155123685400437, 0.059384562564257])
            else:
                self.initial_orientation = np.array([])
                self.initial_position = np.array([])
                self.velocity = np.array([])

        self.T_imu_body = np.identity(4)