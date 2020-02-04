import numpy as np
import cv2
import time
from itertools import chain, compress
from collections import defaultdict, namedtuple
from utils import SuperPointFrontend, SuperPointFrontendv2, MagicPointFrontend


class FeatureMetaData(object):
    """
    Contain necessary information of a feature for easy access.
    """
    def __init__(self):
        self.id = None           # int
        self.response = None     # float
        self.lifetime = None     # int
        self.cam0_point = None   # vec2
        self.descriptor = None


class FeatureMeasurement(object):
    """
    Mono measurement of a feature.
    """
    def __init__(self):
        self.id = None
        self.u0 = None
        self.v0 = None


class ImageProcessor(object):
    """
    Detect and track features in image sequences.
    """
    def __init__(self, config):
        self.config = config

        # Indicate if this is the first image message.
        self.is_first_img = True

        # ID for the next new feature.
        self.next_feature_id = 0

        # Feature detector
        if self.config.model_sp:
            self.detector = SuperPointFrontend(weights_path='/home/antuser/Documents/Thesis/Data/superpoint_v1.pth',
                                               nms_dist=4, conf_thresh=0.001, cuda=True)
        elif self.config.model:
            self.detector = SuperPointFrontendv2(weights_path='/home/antuser/Documents/Thesis/Data/models/'
                                                              'SuperPoint_Coco/round_2_pretrained/'
                                                              'superpoint_coco_weights_15-12.h5',
                                                 nms_dist=4, conf_thresh=0.001)
        elif self.config.model_mp:
            self.detector = MagicPointFrontend(weights_path='/home/antuser/Documents/Thesis/Data/models/'
                                                            'MagicPoint_Shapes/magicpoint_ss_weights_50-47.h5',
                                               nms_dist=4, conf_thresh=0.001)
        else:
            self.detector = cv2.xfeatures2d.SIFT_create()  # cv2.FastFeatureDetector_create(self.config.fast_threshold)
            self.descriptor = cv2.xfeatures2d.SIFT_create()  # cv2.xfeatures2d.FREAK_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # IMU message buffer.
        self.imu_msg_buffer = []

        # Previous and current images
        self.cam0_prev_img_msg = None
        self.cam0_curr_img_msg = None

        # Pyramids for previous and current image
        self.prev_cam0_pyramid = None
        self.curr_cam0_pyramid = None

        # Features in the previous and current image.
        # list of lists of FeatureMetaData
        self.prev_features = [[] for _ in range(self.config.grid_num)]  # Don't use [[]] * N
        self.curr_features = [[] for _ in range(self.config.grid_num)]

        # Number of features after each outlier removal step.
        # keys: before_tracking, after_tracking, after_matching, after_ransac
        self.num_features = defaultdict(int)

        # load config
        # Camera calibration parameters
        self.cam0_resolution = config.cam0_resolution   # vec2
        self.cam0_intrinsics = config.cam0_intrinsics   # vec4
        self.cam0_distortion_model = config.cam0_distortion_model     # string
        self.cam0_distortion_coeffs = config.cam0_distortion_coeffs   # vec4

        # Take a vector from cam0 frame to the IMU frame.
        self.T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.R_cam0_imu = self.T_cam0_imu[:3, :3]
        self.t_cam0_imu = self.T_cam0_imu[:3, 3]

    def mono_callback(self, mono_msg):
        """
        Callback function for the images.
        """
        self.cam0_curr_img_msg = mono_msg.cam0_msg

        # Build the image pyramids once since they're used at multiple places.
        self.create_image_pyramids()

        # Detect features in the first frame.
        if self.is_first_img:
            self.initialize_first_frame()
            self.is_first_img = False
        else:
            # Track the feature in the previous image.
            self.track_features()

            # Add new features into the current image.
            self.add_new_features()

            self.prune_features()

        try:
            return self.publish()
        finally:
            self.cam0_prev_img_msg = self.cam0_curr_img_msg
            self.prev_features = self.curr_features
            self.prev_cam0_pyramid = self.curr_cam0_pyramid

            # Initialize the current features to empty vectors.
            self.curr_features = [[] for _ in range(self.config.grid_num)]

    def imu_callback(self, msg):
        """
        Callback function for the imu message.
        """
        self.imu_msg_buffer.append(msg)

    def create_image_pyramids(self):
        """
        Create image pyramids used for KLT tracking.
        """
        curr_cam0_img = self.cam0_curr_img_msg.image
        # self.curr_cam0_pyramid = cv2.buildOpticalFlowPyramid(
        #     curr_cam0_img, self.config.win_size, self.config.pyramid_levels,
        #     None, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, False)[1]
        self.curr_cam0_pyramid = curr_cam0_img

    def initialize_first_frame(self):
        """
        Initialize the image processing sequence, which is basically detect 
        new features on the first image.
        """
        img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(img)

        response_inliers = []

        # Detect new features on the first image.
        if self.config.model_sp or self.config.model or self.config.model_mp:
            new_features, _, _ = self.detector.run(img)
            cam0_points = [(kp[0], kp[1]) for kp in new_features]
            for i in range(len(cam0_points)):
                response_inliers.append(new_features[i, -1])
        else:
            new_features = self.detector.detect(img)
            # new_features, new_descriptors = self.descriptor.compute(img, new_features)
            cam0_points = [kp.pt for kp in new_features]
            for i in range(len(cam0_points)):
                response_inliers.append(new_features[i].response)

        # Group the features into grids
        grid_new_features = [[] for _ in range(self.config.grid_num)]

        for i in range(len(cam0_points)):
            cam0_point = cam0_points[i]
            response = response_inliers[i]
            # descriptor = new_descriptors[i]

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row*self.config.grid_col + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            # new_feature.descriptor = descriptor
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.
        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features, key=lambda x: x.response,
                                  reverse=True)[:self.config.grid_min_feature_num]:
                self.curr_features[i].append(feature)
                self.curr_features[i][-1].id = self.next_feature_id
                self.curr_features[i][-1].lifetime = 1
                self.next_feature_id += 1

    def track_features(self):
        """
        Tracker features on the newly received image.
        """
        img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(img)
        K = np.array([
            [self.cam0_intrinsics[0], 0.0, self.cam0_intrinsics[2]],
            [0.0, self.cam0_intrinsics[1], self.cam0_intrinsics[3]],
            [0.0, 0.0, 1.0]])

        # Compute a rough relative rotation which takes a vector 
        # from the previous frame to the current frame.
        cam0_R_p_c = self.integrate_imu_data()

        # Organize the features in the previous image.
        prev_ids = []
        prev_lifetime = []
        prev_cam0_points = []
        # prev_descriptors = []

        for feature in chain.from_iterable(self.prev_features):
            prev_ids.append(feature.id)
            prev_lifetime.append(feature.lifetime)
            prev_cam0_points.append(feature.cam0_point)
            # prev_descriptors.append(feature.descriptor)
        prev_cam0_points = np.array(prev_cam0_points, dtype=np.float32)
        # prev_descriptors = np.array(prev_descriptors)

        # Number of the features before tracking.
        self.num_features['before_tracking'] = len(prev_cam0_points)

        # Abort tracking if there is no features in the previous frame.
        if len(prev_cam0_points) == 0:
            return

        # Track features using LK optical flow method.

        curr_cam0_points = self.predict_feature_tracking(
            prev_cam0_points, cam0_R_p_c, self.cam0_intrinsics)
        
        curr_cam0_points, track_inliers, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_cam0_pyramid, self.curr_cam0_pyramid,
            prev_cam0_points.astype(np.float32), 
            curr_cam0_points.astype(np.float32), 
            **self.config.lk_params)
        """
        curr_cam0_points = self.detector.detect(img)
        curr_cam0_points, curr_descriptors = self.descriptor.compute(img, curr_cam0_points)

        matches = self.bf.match(prev_descriptors, curr_descriptors)
        matches_idx = np.array([m.queryIdx for m in matches])
        prev_ids = np.array([prev_ids[idx] for idx in matches_idx])
        prev_lifetime = np.array([prev_lifetime[idx] for idx in matches_idx])
        prev_cam0_points = np.array([prev_cam0_points[idx] for idx in matches_idx])
        curr_cam0_points = cv2.KeyPoint_convert(curr_cam0_points, [m.trainIdx for m in matches])
        train_idx = np.array([m.trainIdx for m in matches])
        curr_descriptors = curr_descriptors[train_idx]
        prev_undistorted = cv2.undistortPoints(np.reshape(prev_cam0_points, (-1, 1, 2)), cameraMatrix=K,
                                               distCoeffs=self.cam0_distortion_coeffs)
        prev_undistorted = prev_undistorted.reshape((-1, 2))
        curr_undistorted = cv2.undistortPoints(np.reshape(curr_cam0_points, (-1, 1, 2)), cameraMatrix=K,
                                               distCoeffs=self.cam0_distortion_coeffs)
        curr_undistorted = curr_undistorted.reshape((-1, 2))
        E, mask = cv2.findEssentialMat(prev_undistorted, curr_undistorted, cameraMatrix=K, method=cv2.RANSAC)
        track_inliers = np.array(mask.ravel().tolist())
        """

        # Mark those tracked points out of the image region as untracked.
        for i, point in enumerate(curr_cam0_points):
            if not track_inliers[i]:
                continue
            if (point[0] < 0 or point[0] > img.shape[1]-1 or 
                point[1] < 0 or point[1] > img.shape[0]-1):
                track_inliers[i] = 0

        # Collect the tracked points.
        prev_tracked_ids = select(prev_ids, track_inliers)
        prev_tracked_lifetime = select(prev_lifetime, track_inliers)
        prev_tracked_cam0_points = select(prev_cam0_points, track_inliers)
        curr_tracked_cam0_points = select(curr_cam0_points, track_inliers)
        # curr_tracked_descriptors = select(curr_descriptors, track_inliers)

        # Number of features left after tracking.
        self.num_features['after_tracking'] = len(curr_tracked_cam0_points)

        # Number of features left after stereo matching.
        self.num_features['after_matching'] = len(curr_tracked_cam0_points)

        # Step 2: RANSAC on temporal image pairs of cam0
        cam0_ransac_inliers = self.two_point_ransac(prev_tracked_cam0_points, curr_tracked_cam0_points,
                                                    cam0_R_p_c, self.cam0_intrinsics, self.cam0_distortion_model,
                                                    self.cam0_distortion_coeffs, self.config.ransac_threshold, 0.99)
        # cam0_ransac_inliers = [1] * len(prev_tracked_cam0_points)

        # Number of features after ransac.
        after_ransac = 0
        for i in range(len(cam0_ransac_inliers)):
            if not cam0_ransac_inliers[i]:
                continue
            row = int(curr_tracked_cam0_points[i][1] / grid_height)
            col = int(curr_tracked_cam0_points[i][0] / grid_width)
            code = row * self.config.grid_col + col

            grid_new_feature = FeatureMetaData()
            grid_new_feature.id = prev_tracked_ids[i]
            grid_new_feature.lifetime = prev_tracked_lifetime[i] + 1
            grid_new_feature.cam0_point = curr_tracked_cam0_points[i]
            # grid_new_feature.descriptor = curr_tracked_descriptors[i]
            prev_tracked_lifetime[i] += 1

            self.curr_features[code].append(grid_new_feature)
            after_ransac += 1
        self.num_features['after_ransac'] = after_ransac

        # Compute the tracking rate.
        # prev_feature_num = sum([len(x) for x in self.prev_features])
        # curr_feature_num = sum([len(x) for x in self.curr_features])


    def add_new_features(self):
        """
        Detect new features on the image to ensure that the features are 
        uniformly distributed on the image.
        """
        curr_img = self.cam0_curr_img_msg.image
        grid_height, grid_width = self.get_grid_size(curr_img)

        # Create a mask to avoid redetecting existing features.
        mask = np.ones(curr_img.shape[:2], dtype='uint8')
        for feature in chain.from_iterable(self.curr_features):
            x, y = map(int, feature.cam0_point)
            mask[y-3:y+4, x-3:x+4] = 0

        # Detect new features.
        if self.config.model_sp or self.config.model or self.config.model_mp:
            new_features, _, _ = self.detector.run(curr_img)
        else:
            new_features = self.detector.detect(curr_img, mask=mask)
            # new_features, new_descriptors = self.descriptor.compute(curr_img, new_features)

        # Collect the new detected features based on the grid.
        # Select the ones with top response within each grid afterwards.
        new_feature_sieve = [[] for _ in range(self.config.grid_num)]
        # new_descriptor_sieve = [[] for _ in range(self.config.grid_num)]
        if self.config.model_sp or self.config.model or self.config.model_mp:
            for feature in new_features:
            # for feature, descriptor in zip(new_features, new_descriptors):
                row = int(feature[1] / grid_height)
                col = int(feature[0] / grid_width)
                code = row * self.config.grid_col + col
                new_feature_sieve[code].append(feature)
                # new_descriptor_sieve[code].append(descriptor)
        else:
            for feature in new_features:
            # for feature, descriptor in zip(new_features, new_descriptors):
                row = int(feature.pt[1] / grid_height)
                col = int(feature.pt[0] / grid_width)
                code = row * self.config.grid_col + col
                new_feature_sieve[code].append(feature)
                #new_descriptor_sieve[code].append(descriptor)

        new_features = []
        # new_descriptors = []
        #for features, descriptors in zip(new_feature_sieve, new_descriptor_sieve):
        for features in new_feature_sieve:
            if len(features) > self.config.grid_max_feature_num:
                if self.config.model_sp or self.config.model or self.config.model_mp:
                    #feature_idx = sorted(range(len(features)), key=lambda k: features[k][2], reverse=True)
                    features = sorted(features, key=lambda x: x[2], reverse=True)[:self.config.grid_max_feature_num]
                    #descriptors = [descriptors[x] for x in feature_idx[:self.config.grid_max_feature_num]]
                else:
                    #feature_idx = sorted(range(len(features)), key=lambda k: features[k].response, reverse=True)
                    features = sorted(features, key=lambda x: x.response, reverse=True)[:self.config.grid_max_feature_num]
                    #descriptors = [descriptors[x] for x in feature_idx[:self.config.grid_max_feature_num]]
            new_features.append(features)
            #new_descriptors.append(descriptors)
        new_features = list(chain.from_iterable(new_features))
        #new_descriptors = list(chain.from_iterable(new_descriptors))

        # Find the stereo matched points for the newly detected features.
        response_inliers = []
        if self.config.model_sp or self.config.model or self.config.model_mp:
            cam0_points = [(kp[0], kp[1]) for kp in new_features]
            for i in range(len(cam0_points)):
                response_inliers.append(new_features[i][-1])
        else:
            cam0_points = [kp.pt for kp in new_features]
            for i in range(len(cam0_points)):
                response_inliers.append(new_features[i].response)

        # Group the features into grids
        grid_new_features = [[] for _ in range(self.config.grid_num)]
        for i in range(len(cam0_points)):
            cam0_point = cam0_points[i]
            response = response_inliers[i]
            #descriptor = new_descriptors[i]

            row = int(cam0_point[1] / grid_height)
            col = int(cam0_point[0] / grid_width)
            code = row*self.config.grid_col + col

            new_feature = FeatureMetaData()
            new_feature.response = response
            new_feature.cam0_point = cam0_point
            #new_feature.descriptor = descriptor
            grid_new_features[code].append(new_feature)

        # Sort the new features in each grid based on its response.
        # And collect new features within each grid with high response.

        for i, new_features in enumerate(grid_new_features):
            for feature in sorted(new_features, key=lambda x: x.response,
                                  reverse=True)[:self.config.grid_min_feature_num]:
                self.curr_features[i].append(feature)
                self.curr_features[i][-1].id = self.next_feature_id
                self.curr_features[i][-1].lifetime = 1
                self.next_feature_id += 1

    def prune_features(self):
        """
        Remove some of the features of a grid in case there are too many 
        features inside of that grid, which ensures the number of features 
        within each grid is bounded.
        """
        for i, features in enumerate(self.curr_features):
            # Continue if the number of features in this grid does
            # not exceed the upper bound.
            if len(features) <= self.config.grid_max_feature_num:
                continue
            self.curr_features[i] = sorted(features, key=lambda x: x.lifetime,
                reverse=True)[:self.config.grid_max_feature_num]

    def publish(self):
        """
        Publish the features on the current image including both the 
        tracked and newly detected ones.
        """
        curr_ids = []
        curr_cam0_points = []
        for feature in chain.from_iterable(self.curr_features):
            curr_ids.append(feature.id)
            curr_cam0_points.append(feature.cam0_point)

        curr_cam0_points_undistorted = self.undistort_points(curr_cam0_points, self.cam0_intrinsics,
                                                             self.cam0_distortion_model, self.cam0_distortion_coeffs)

        features = []
        for i in range(len(curr_ids)):
            fm = FeatureMeasurement()
            fm.id = curr_ids[i]
            fm.u0 = curr_cam0_points_undistorted[i][0]
            fm.v0 = curr_cam0_points_undistorted[i][1]
            features.append(fm)

        feature_msg = namedtuple('feature_msg', ['timestamp', 'features'])(self.cam0_curr_img_msg.timestamp, features)
        return feature_msg

    def integrate_imu_data(self):
        """
        Integrates the IMU gyro readings between the two consecutive images, 
        which is used for both tracking prediction and 2-point RANSAC.

        Returns:
            cam0_R_p_c: a rotation matrix which takes a vector from previous 
                cam0 frame to current cam0 frame.
        """
        # Find the start and the end limit within the imu msg buffer.
        idx_begin = None
        for i, msg in enumerate(self.imu_msg_buffer):
            if msg.timestamp >= self.cam0_prev_img_msg.timestamp - 0.01:
                idx_begin = i
                break

        idx_end = None
        for i, msg in enumerate(self.imu_msg_buffer):
            if msg.timestamp >= self.cam0_curr_img_msg.timestamp - 0.004:
                idx_end = i
                break

        if idx_begin is None or idx_end is None:
            return np.identity(3)

        # Compute the mean angular velocity in the IMU frame.
        mean_ang_vel = np.zeros(3)
        for i in range(idx_begin, idx_end):
            mean_ang_vel += self.imu_msg_buffer[i].angular_velocity

        if idx_end > idx_begin:
            mean_ang_vel /= (idx_end - idx_begin)

        # Transform the mean angular velocity from the IMU frame to the 
        # cam0 and cam1 frames.
        cam0_mean_ang_vel = self.R_cam0_imu.T @ mean_ang_vel

        # Compute the relative rotation.
        dt = self.cam0_curr_img_msg.timestamp - self.cam0_prev_img_msg.timestamp
        cam0_R_p_c = cv2.Rodrigues(cam0_mean_ang_vel * dt)[0].T

        # Delete the useless and used imu messages.
        self.imu_msg_buffer = self.imu_msg_buffer[idx_end:]
        return cam0_R_p_c

    def rescale_points(self, pts1, pts2):
        """
        Arguments:
            pts1: first set of points.
            pts2: second set of points.

        Returns:
            pts1: scaled first set of points.
            pts2: scaled second set of points.
            scaling_factor: scaling factor
        """
        scaling_factor = 0
        for pt1, pt2 in zip(pts1, pts2):
            scaling_factor += np.linalg.norm(pt1)
            scaling_factor += np.linalg.norm(pt2)

        scaling_factor = (len(pts1) + len(pts2)) / scaling_factor * np.sqrt(2)

        for i in range(len(pts1)):
            pts1[i] *= scaling_factor
            pts2[i] *= scaling_factor

        return pts1, pts2, scaling_factor

    def two_point_ransac(self, pts1, pts2, R_p_c, intrinsics, distortion_model, distortion_coeffs,
                         inlier_error, success_probability):
         """
         Applies two point ransac algorithm to mark the inliers in the input set.

         Arguments:
             pts1: first set of points.
             pts2: second set of points.
             R_p_c: a rotation matrix takes a vector in the previous camera frame
                 to the current camera frame.
             intrinsics: intrinsics of the camera.
             distortion_model: distortion model of the camera.
             distortion_coeffs: distortion coefficients.
             inlier_error: acceptable error to be considered as an inlier.
             success_probability: the required probability of success.

         Returns:
             inlier_flag: 1 for inliers and 0 for outliers.
         """
         # Check the size of input point size.
         assert len(pts1) == len(pts2), 'Sets of different size are used...'
         if len(pts1) == 0:
             return [1] * len(pts1)

         norm_pixel_unit = 2.0 / (intrinsics[0] + intrinsics[1])
         iter_num = int(np.ceil(np.log(1-success_probability) / np.log(1-0.7*0.7)))

         # Initially, mark all points as inliers.
         inlier_markers = [1] * len(pts1)

         # Undistort all the points.
         pts1_undistorted = self.undistort_points(pts1, intrinsics,
             distortion_model, distortion_coeffs)
         pts2_undistorted = self.undistort_points(pts2, intrinsics,
             distortion_model, distortion_coeffs)

         # Compenstate the points in the previous image with
         # the relative rotation.
         for i, pt in enumerate(pts1_undistorted):
             pt_h = np.array([*pt, 1.0])
             pt_hc = R_p_c @ pt_h
             pts1_undistorted[i] = pt_hc[:2]

         # Normalize the points to gain numerical stability.
         pts1_undistorted, pts2_undistorted, scaling_factor = self.rescale_points(
             pts1_undistorted, pts2_undistorted)

         # Compute the difference between previous and current points,
         # which will be used frequently later.
         pts_diff = []
         for pt1, pt2 in zip(pts1_undistorted, pts2_undistorted):
             pts_diff.append(pt1 - pt2)

         # Mark the point pairs with large difference directly.
         # BTW, the mean distance of the rest of the point pairs are computed.
         mean_pt_distance = 0.0
         raw_inlier_count = 1e-6
         for i, pt_diff in enumerate(pts_diff):
             distance = np.linalg.norm(pt_diff)
             # 25 pixel distance is a pretty large tolerance for normal motion.
             # However, to be used with aggressive motion, this tolerance should
             # be increased significantly to match the usage.
             if distance > 50.0 * norm_pixel_unit:
                 inlier_markers[i] = 0
             else:
                 mean_pt_distance += distance
                 raw_inlier_count += 1

         mean_pt_distance /= raw_inlier_count

         # If the current number of inliers is less than 3, just mark
         # all input as outliers. This case can happen with fast
         # rotation where very few features are tracked.
         if raw_inlier_count < 3:
             return [0] * len(inlier_markers)

         # Before doing 2-point RANSAC, we have to check if the motion
         # is degenerated, meaning that there is no translation between
         # the frames, in which case, the model of the RANSAC does not work.
         # If so, the distance between the matched points will be almost 0.
         if mean_pt_distance < norm_pixel_unit:
             for i, pt_diff in enumerate(pts_diff):
                 if inlier_markers[i] == 0:
                     continue
                 if np.linalg.norm(pt_diff) > inlier_error * norm_pixel_unit:
                     inlier_markers[i] = 0
             return inlier_markers

         # In the case of general motion, the RANSAC model can be applied.
         # The three column corresponds to tx, ty, and tz respectively.
         coeff_t = []
         for i, pt_diff in enumerate(pts_diff):
             coeff_t.append(np.array([
                 pt_diff[1],
                 -pt_diff[0],
                 pts1_undistorted[i, 0] * pts2_undistorted[i, 1] -
                 pts1_undistorted[i, 1] * pts2_undistorted[i, 0]]))
         coeff_t = np.array(coeff_t)

         raw_inlier_idx = np.where(inlier_markers)[0]
         best_inlier_set = []
         best_error = 1e10

         for i in range(iter_num):
             # Randomly select two point pairs.
             # Although this is a weird way of selecting two pairs, but it
             # is able to efficiently avoid selecting repetitive pairs.
             pair_idx1 = np.random.choice(raw_inlier_idx)
             idx_diff = np.random.randint(1, len(raw_inlier_idx))
             pair_idx2 = (pair_idx1+idx_diff) % len(raw_inlier_idx)

             # Construct the model.
             coeff_t_ = np.array([coeff_t[pair_idx1], coeff_t[pair_idx2]])
             coeff_tx = coeff_t_[:, 0]
             coeff_ty = coeff_t_[:, 1]
             coeff_tz = coeff_t_[:, 2]
             coeff_l1_norm = np.linalg.norm(coeff_t_, 1, axis=0)
             base_indicator = np.argmin(coeff_l1_norm)

             try:
                 if base_indicator == 0:
                     A = np.array([coeff_ty, coeff_tz]).T
                     solution = np.linalg.inv(A) @ (-coeff_tx)
                     model = [1.0, *solution]
                 elif base_indicator == 1:
                     A = np.array([coeff_tx, coeff_tz]).T
                     solution = np.linalg.inv(A) @ (-coeff_ty)
                     model = [solution[0], 1.0, solution[1]]
                 else:
                     A = np.array([coeff_tx, coeff_ty]).T
                     solution = np.linalg.inv(A) @ (-coeff_tz)
                     model = [*solution, 1.0]
             except:
                 continue

             # Find all the inliers among point pairs.
             error = coeff_t @ model

             inlier_set = []
             for i, e in enumerate(error):
                 if inlier_markers[i] == 0:
                     continue
                 if np.abs(e) < inlier_error * norm_pixel_unit:
                     inlier_set.append(i)

             # If the number of inliers is small, the current model is
             # probably wrong.
             if len(inlier_set) < 0.2 * len(pts1_undistorted):
                 continue

             # Refit the model using all of the possible inliers.
             coeff_t_ = coeff_t[inlier_set]
             coeff_tx_better = coeff_t_[:, 0]
             coeff_ty_better = coeff_t_[:, 1]
             coeff_tz_better = coeff_t_[:, 2]

             try:
                 if base_indicator == 0:
                     A = np.array([coeff_ty_better, coeff_tz_better]).T
                     solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_tx_better)
                     model_better = [1.0, *solution]
                 elif base_indicator == 1:
                     A = np.array([coeff_tx_better, coeff_tz_better]).T
                     solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_ty_better)
                     model_better = [solution[0], 1.0, solution[1]]
                 else:
                     A = np.array([coeff_tx_better, coeff_ty_better]).T
                     solution = np.linalg.inv(A.T @ A) @ A.T @ (-coeff_tz_better)
                     model_better = [*solution, 1.0]
             except:
                 continue

             # Compute the error and upate the best model if possible.
             new_error = coeff_t @ model_better
             this_error = np.mean([np.abs(new_error[i]) for i in inlier_set])

             if len(inlier_set) > len(best_inlier_set):
                 best_error = this_error
                 best_inlier_set = inlier_set

         # Fill in the markers.
         inlier_markers = [0] * len(pts1)
         for i in best_inlier_set:
             inlier_markers[i] = 1

         return inlier_markers

    def get_grid_size(self, img):
        """
        # Size of each grid.
        """
        grid_height = int(np.ceil(img.shape[0] / self.config.grid_row))
        grid_width  = int(np.ceil(img.shape[1] / self.config.grid_col))
        return grid_height, grid_width

    def predict_feature_tracking(self, input_pts, R_p_c, intrinsics):
        """
        predictFeatureTracking Compensates the rotation between consecutive 
        camera frames so that feature tracking would be more robust and fast.

        Arguments:
            input_pts: features in the previous image to be tracked.
            R_p_c: a rotation matrix takes a vector in the previous camera 
                frame to the current camera frame. (matrix33)
            intrinsics: intrinsic matrix of the camera. (vec3)

        Returns:
            compensated_pts: predicted locations of the features in the 
                current image based on the provided rotation.
        """
        # Return directly if there are no input features.
        if len(input_pts) == 0:
            return []

        # Intrinsic matrix.
        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        H = K @ R_p_c @ np.linalg.inv(K)

        compensated_pts = []
        for i in range(len(input_pts)):
            p1 = np.array([*input_pts[i], 1.0])
            p2 = H @ p1
            compensated_pts.append(p2[:2] / p2[2])
        return np.array(compensated_pts, dtype=np.float32)

    def undistort_points(self, pts_in, intrinsics, distortion_model, 
        distortion_coeffs, rectification_matrix=np.identity(3),
        new_intrinsics=np.array([1, 1, 0, 0])):
        """
        Arguments:
            pts_in: points to be undistorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.
            rectification_matrix:
            new_intrinsics:

        Returns:
            pts_out: undistorted points.
        """
        if len(pts_in) == 0:
            return []
        
        pts_in = np.reshape(pts_in, (-1, 1, 2))
        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])
        K_new = np.array([
            [new_intrinsics[0], 0.0, new_intrinsics[2]],
            [0.0, new_intrinsics[1], new_intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            pts_out = cv2.fisheye.undistortPoints(pts_in, K, distortion_coeffs,
                rectification_matrix, K_new)
        else:   # default: 'radtan'
            pts_out = cv2.undistortPoints(pts_in, K, distortion_coeffs, None,
                rectification_matrix, K_new)
        return pts_out.reshape((-1, 2))

    def distort_points(self, pts_in, intrinsics, distortion_model, 
            distortion_coeffs):
        """
        Arguments:
            pts_in: points to be distorted.
            intrinsics: intrinsics of the camera.
            distortion_model: distortion model of the camera.
            distortion_coeffs: distortion coefficients.

        Returns:
            pts_out: distorted points. (N, 2)
        """
        if len(pts_in) == 0:
            return []

        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0]])

        if distortion_model == 'equidistant':
            pts_out = cv2.fisheye.distortPoints(pts_in, K, distortion_coeffs)
        else:   # default: 'radtan'
            homogenous_pts = cv2.convertPointsToHomogeneous(pts_in)
            pts_out, _ = cv2.projectPoints(homogenous_pts, 
                np.zeros(3), np.zeros(3), K, distortion_coeffs)
        return pts_out.reshape((-1, 2))


def skew(vec):
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def select(data, selectors):
    return [d for d, s in zip(data, selectors) if s]