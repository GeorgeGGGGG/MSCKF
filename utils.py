import numpy as np
import math
import torch
import cv2
import tensorflow as tf
tf.enable_eager_execution()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


# quaternion representation: [x, y, z, w]
# JPL convention


def skew(vec):
    """
    Create a skew-symmetric matrix from a 3-element vector.
    """
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def to_rotation(q):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    q = q / np.linalg.norm(q)

    vec = q[:3]
    w = q[3]

    R = (2*w*w-1)*np.identity(3) - 2*w*skew(vec) + 2*vec[:, None]*vec
    return R

def to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0,0] - R[1,1] - R[2,2]
            q = [t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2], R[1, 2]-R[2, 1]]
        else:
            t = 1 - R[0,0] + R[1,1] - R[2,2]
            q = [R[0, 1]+R[1, 0], t, R[2, 1]+R[1, 2], R[2, 0]-R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0,0] - R[1,1] + R[2,2]
            q = [R[0, 2]+R[2, 0], R[2, 1]+R[1, 2], t, R[0, 1]-R[1, 0]]
        else:
            t = 1 + R[0,0] + R[1,1] + R[2,2]
            q = [R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0], t]

    q = np.array(q) # * 0.5 / np.sqrt(t)
    return q / np.linalg.norm(q)

def rpy_to_quaternion(rpy):
    q = np.zeros((4,))

    cr = np.cos(rpy[0] / 2)
    cp = np.cos(rpy[1] / 2)
    cy = np.cos(rpy[2] / 2)
    sr = np.sin(rpy[0] / 2)
    sp = np.sin(rpy[1] / 2)
    sy = np.sin(rpy[2] / 2)

    q[3] = cr * cp * cy + sr * sp * sy
    q[0] = sr * cp * cy - cr * sp * sy
    q[1] = cr * sp * cy + sr * cp * sy
    q[2] = cr * cp * sy - sr * sp * cy

    return q

def quaternion_to_rpy(q):
    rpy = np.zeros((3,))
    q = q / np.linalg.norm(q)

    test = q[3]*q[1] - q[2]*q[0]
    if (test > 0.499):
        rpy[0] = 0
        rpy[1] = np.pi / 2
        rpy[2] = 2 * math.atan2(q[0], q[3])
    if (test < -0.499):
        rpy[0] = 0
        rpy[1] = -np.pi / 2
        rpy[2] = -2 * math.atan2(q[0], q[3])

    sqx = q[0]**2
    sqy = q[1]**2
    sqz = q[2]**2
    rpy[0] = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(sqx + sqy))
    rpy[1] = math.asin(2 * test)
    rpy[2] = math.atan2(2 * (q[2]*q[3] + q[0]*q[1]), 1 - 2*(sqy + sqz))

    return rpy

def R_to_rpy(R):
    rpy = np.zeros((3,))

    rpy[2] = math.atan2(R[0, 1], R[0, 0])
    rpy[1] = -math.asin(R[0, 2])
    rpy[0] = math.atan2(R[1, 2], R[2, 2])

    return rpy

def quaternion_normalize(q):
    """
    Normalize the given quaternion to unit quaternion.
    """
    return q / np.linalg.norm(q)

def quaternion_conjugate(q):
    """
    Conjugate of a quaternion.
    """
    return np.array([*-q[:3], q[3]])

def quaternion_multiplication(q1, q2):
    """
    Perform q1 * q2
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    L = np.array([
        [ q1[3],  q1[2], -q1[1], q1[0]],
        [-q1[2],  q1[3],  q1[0], q1[1]],
        [ q1[1], -q1[0],  q1[3], q1[2]],
        [-q1[0], -q1[1], -q1[2], q1[3]]
    ])

    q = L @ q2
    return q / np.linalg.norm(q)


def small_angle_quaternion(dtheta):
    """
    Convert the vector part of a quaternion to a full quaternion.
    This function is useful to convert delta quaternion which is  
    usually a 3x1 vector to a full quaternion.
    For more details, check Equation (238) and (239) in "Indirect Kalman 
    Filter for 3D Attitude Estimation: A Tutorial for quaternion Algebra".
    """
    dq = dtheta / 2.
    dq_square_norm = dq @ dq

    if dq_square_norm <= 1:
        q = np.array([*dq, np.sqrt(1-dq_square_norm)])
    else:
        q = np.array([*dq, 1.])
        q /= np.sqrt(1+dq_square_norm)
    return q


def from_two_vectors(v0, v1):
    """
    Rotation quaternion from v0 to v1.
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    d = v0 @ v1

    # if dot == -1, vectors are nearly opposite
    if d < -0.999999:
        axis = np.cross([1,0,0], v0)
        if np.linalg.norm(axis) < 0.000001:
            axis = np.cross([0,1,0], v0)
        q = np.array([*axis, 0.])
    elif d > 0.999999:
        q = np.array([0., 0., 0., 1.])
    else:
        s = np.sqrt((1+d)*2)
        axis = np.cross(v0, v1)
        vec = axis / s
        w = 0.5 * s
        q = np.array([*vec, w])
        
    q = q / np.linalg.norm(q)
    return quaternion_conjugate(q)   # hamilton -> JPL


class Isometry3d(object):
    """
    3d rigid transform.
    """
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)



def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """
    Performs NMS on the heatmap (prob) by considering hypothetical
    bounding boxes centered at each pixel's location. Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap of shape [H, W]
        size: a scalar, the size of the bounding boxes
        iou: a scalar, the IoU overlap threshold
        min_prob: a threshold under which all probabilities are discarded before NMS
        keep_top_k: an integer, the number of top scores to keep
    Return:
        prob: probability heatmap after NMS

    """
    pts = tf.cast(tf.where(tf.greater_equal(prob, min_prob)), dtype=tf.float32)
    size = tf.constant(size / 2.)
    boxes = tf.concat([pts - size, pts + size], axis=1)
    scores = tf.gather_nd(prob, tf.cast(pts, dtype=tf.int32))

    with tf.device('/cpu:0'):
        indices = tf.image.non_max_suppression(boxes, scores, tf.shape(boxes)[0], iou)

    pts = tf.gather(pts, indices)
    scores = tf.gather(scores, indices)

    if keep_top_k:
        k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
        scores, indices = tf.nn.top_k(scores, k)
        pts = tf.gather(pts, indices)

    prob = tf.scatter_nd(tf.cast(pts, dtype=tf.int32), scores, tf.shape(prob))

    return prob



class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(
            1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(
            c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(
            c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(
            c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(
            c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(
            c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(
            c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(
            c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(
            c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(
            c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(
            c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh, cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1,
                     pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        # === convert image =========================
        if img.ndim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.dtype != np.float32:
            img = img.astype('float32')
        if np.max(img) > 1.0:
            img = img / 255.0
        # ===========================================

        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        # Apply NMS.
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts.T, desc, heatmap



class SuperPointNetv2():

    def __init__(self):
        super(SuperPointNetv2, self).__init__()
        self.params_conv = {'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}
        self.params_pool = {'pool_size': 2, 'strides': 2, 'padding': 'same'}
        self.params_d = {'kernel_size': 1, 'padding': 'same', 'activation': None}
        # Shared Encoder.
        self.conv1a = tf.keras.layers.Conv2D(filters=64, name='conv1a', **self.params_conv)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.conv1b = tf.keras.layers.Conv2D(filters=64, name='conv1b', **self.params_conv)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.pool1 = tf.keras.layers.MaxPool2D(name='pool1', **self.params_pool)
        self.conv2a = tf.keras.layers.Conv2D(filters=64, name='conv2a', **self.params_conv)
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.conv2b = tf.keras.layers.Conv2D(filters=64, name='conv2b', **self.params_conv)
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        self.pool2 = tf.keras.layers.MaxPool2D(name='pool2', **self.params_pool)
        self.conv3a = tf.keras.layers.Conv2D(filters=128, name='conv3a', **self.params_conv)
        self.bn5 = tf.keras.layers.BatchNormalization(name='bn5')
        self.conv3b = tf.keras.layers.Conv2D(filters=128, name='conv3b', **self.params_conv)
        self.bn6 = tf.keras.layers.BatchNormalization(name='bn6')
        self.pool3 = tf.keras.layers.MaxPool2D(name='pool3', **self.params_pool)
        self.conv4a = tf.keras.layers.Conv2D(filters=128, name='conv4a', **self.params_conv)
        self.bn7 = tf.keras.layers.BatchNormalization(name='bn7')
        self.conv4b = tf.keras.layers.Conv2D(filters=128, name='conv4b', **self.params_conv)
        self.bn8 = tf.keras.layers.BatchNormalization(name='bn8')
        # Detector Head.
        self.conv5 = tf.keras.layers.Conv2D(filters=256, name='conv5', **self.params_conv)
        self.bn9 = tf.keras.layers.BatchNormalization(name='bn9')
        self.det = tf.keras.layers.Conv2D(filters=65, name='det', **self.params_d)
        self.bn10 = tf.keras.layers.BatchNormalization(name='bn10')
        # Descriptor Head.
        self.conv6 = tf.keras.layers.Conv2D(filters=256, name='conv6', **self.params_conv)
        self.bn11 = tf.keras.layers.BatchNormalization(name='bn11')
        self.desc = tf.keras.layers.Conv2D(filters=256, name='desc', **self.params_d)
        self.bn12 = tf.keras.layers.BatchNormalization(name='bn12')

    def forward(self):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image tensor shaped N x H x W x 1.
        Output
          det: Output point tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor tensor shaped N x 256 x H/8 x W/8.
        """
        def convBlock(conv_layer, bn_layer, input):
            x = conv_layer(input)
            x = bn_layer(x)

            return x

        # Shared Encoder.
        img_in = tf.keras.Input(shape=(None, None, 1))
        x = convBlock(self.conv1a, self.bn1, img_in)
        x = convBlock(self.conv1b, self.bn2, x)
        x = self.pool1(x)
        x = convBlock(self.conv2a, self.bn3, x)
        x = convBlock(self.conv2b, self.bn4, x)
        x = self.pool2(x)
        x = convBlock(self.conv3a, self.bn5, x)
        x = convBlock(self.conv3b, self.bn6, x)
        x = self.pool3(x)
        x = convBlock(self.conv4a, self.bn7, x)
        x = convBlock(self.conv4b, self.bn8, x)
        # Detector Head.
        det = convBlock(self.conv5, self.bn9, x)
        det = convBlock(self.det, self.bn10, det)
        # Descriptor Head.
        desc = convBlock(self.conv6, self.bn11, x)
        desc = convBlock(self.desc, self.bn12, desc)
        model = tf.keras.Model(img_in, [det, desc])
        return model


class SuperPointFrontendv2():
    """ Wrapper around net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh):
        self.name = 'SuperPoint'
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network.
        self.net = SuperPointNetv2()
        self.model = self.net.forward()
        self.model.load_weights(weights_path)

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
        """
        H, W = tf.shape(img)[0], tf.shape(img)[1]
        if np.max(img) > 1.0:
            img = img / 255.
        inp = img[tf.newaxis, :, :, tf.newaxis]
        # Forward pass of network.
        det, desc = self.model.predict_on_batch(inp)
        # --- Process points.
        prob = tf.nn.softmax(det, axis=-1)
        prob = prob[:, :, :, :-1]
        prob = tf.depth_to_space(prob, block_size=self.cell)
        prob = tf.squeeze(prob, axis=-1)
        prob = tf.map_fn(lambda p: box_nms(p, self.nms_dist, min_prob=self.conf_thresh), prob)
        prob = tf.image.crop_to_bounding_box(prob[..., tf.newaxis], self.border_remove, self.border_remove,
                                             H - self.border_remove, W - self.border_remove)
        prob = tf.squeeze(tf.image.pad_to_bounding_box(prob, self.border_remove, self.border_remove, H, W), axis=-1)
        prob = tf.squeeze(prob, axis=0)

        pts = tf.cast(tf.where(prob), dtype=tf.int32)

        # --- Process descriptor.
        desc = tf.image.resize_bicubic(desc, tf.shape(img))
        desc = tf.nn.l2_normalize(desc, axis=-1)
        desc = tf.squeeze(desc, axis=0)
        desc = tf.gather_nd(desc, pts)
        desc = tf.transpose(desc)

        conf = tf.gather_nd(prob, pts)
        pts = tf.concat([pts[:, 1][..., tf.newaxis], pts[:, 0][..., tf.newaxis]], axis=-1)
        pts = tf.concat([tf.cast(pts, dtype=tf.float32), conf[..., tf.newaxis]], axis=-1)
        pts = tf.transpose(pts)

        return pts.numpy().T, desc.numpy(), prob.numpy()


class MagicPointNet():

    def __init__(self):
        super(MagicPointNet, self).__init__()
        self.params_conv = {'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}
        self.params_pool = {'pool_size': 2, 'strides': 2, 'padding': 'same'}
        self.params_d = {'kernel_size': 1, 'padding': 'same', 'activation': None}
        # Shared Encoder.
        self.conv1a = tf.keras.layers.Conv2D(filters=64, name='conv1a', **self.params_conv)
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.conv1b = tf.keras.layers.Conv2D(filters=64, name='conv1b', **self.params_conv)
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        self.pool1 = tf.keras.layers.MaxPool2D(name='pool1', **self.params_pool)
        self.conv2a = tf.keras.layers.Conv2D(filters=64, name='conv2a', **self.params_conv)
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        self.conv2b = tf.keras.layers.Conv2D(filters=64, name='conv2b', **self.params_conv)
        self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
        self.pool2 = tf.keras.layers.MaxPool2D(name='pool2', **self.params_pool)
        self.conv3a = tf.keras.layers.Conv2D(filters=128, name='conv3a', **self.params_conv)
        self.bn5 = tf.keras.layers.BatchNormalization(name='bn5')
        self.conv3b = tf.keras.layers.Conv2D(filters=128, name='conv3b', **self.params_conv)
        self.bn6 = tf.keras.layers.BatchNormalization(name='bn6')
        self.pool3 = tf.keras.layers.MaxPool2D(name='pool3', **self.params_pool)
        self.conv4a = tf.keras.layers.Conv2D(filters=128, name='conv4a', **self.params_conv)
        self.bn7 = tf.keras.layers.BatchNormalization(name='bn7')
        self.conv4b = tf.keras.layers.Conv2D(filters=128, name='conv4b', **self.params_conv)
        self.bn8 = tf.keras.layers.BatchNormalization(name='bn8')
        # Detector Head.
        self.conv5 = tf.keras.layers.Conv2D(filters=256, name='conv5', **self.params_conv)
        self.bn9 = tf.keras.layers.BatchNormalization(name='bn9')
        self.det = tf.keras.layers.Conv2D(filters=65, name='det', **self.params_d)
        self.bn10 = tf.keras.layers.BatchNormalization(name='bn10')

    def forward(self):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image tensor shaped N x H x W x 1.
        Output
          det: Output point tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor tensor shaped N x 256 x H/8 x W/8.
        """
        def convBlock(conv_layer, bn_layer, input):
            x = conv_layer(input)
            x = bn_layer(x)

            return x

        # Shared Encoder.
        img_in = tf.keras.Input(shape=(None, None, 1))
        x = convBlock(self.conv1a, self.bn1, img_in)
        x = convBlock(self.conv1b, self.bn2, x)
        x = self.pool1(x)
        x = convBlock(self.conv2a, self.bn3, x)
        x = convBlock(self.conv2b, self.bn4, x)
        x = self.pool2(x)
        x = convBlock(self.conv3a, self.bn5, x)
        x = convBlock(self.conv3b, self.bn6, x)
        x = self.pool3(x)
        x = convBlock(self.conv4a, self.bn7, x)
        x = convBlock(self.conv4b, self.bn8, x)
        # Detector Head.
        det = convBlock(self.conv5, self.bn9, x)
        det = convBlock(self.det, self.bn10, det)
        model = tf.keras.Model(img_in, det)
        return model


class MagicPointFrontend():
    """ Wrapper around net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh):
        self.name = 'MagicPoint'
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network.
        self.net = MagicPointNet()
        self.model = self.net.forward()
        self.model.load_weights(weights_path)

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
        """
        H, W = tf.shape(img)[0], tf.shape(img)[1]
        if np.max(img) > 1.0:
            img = img / 255.
        inp = img[tf.newaxis, :, :, tf.newaxis]
        # Forward pass of network.
        det = self.model.predict_on_batch(inp)
        # --- Process points.
        prob = tf.nn.softmax(det, axis=-1)
        prob = prob[:, :, :, :-1]
        prob = tf.depth_to_space(prob, block_size=self.cell)
        prob = tf.squeeze(prob, axis=-1)
        prob = tf.map_fn(lambda p: box_nms(p, self.nms_dist, min_prob=self.conf_thresh), prob)
        prob = tf.image.crop_to_bounding_box(prob[..., tf.newaxis], self.border_remove, self.border_remove,
                                             H - self.border_remove, W - self.border_remove)
        prob = tf.squeeze(tf.image.pad_to_bounding_box(prob, self.border_remove, self.border_remove, H, W), axis=-1)
        prob = tf.squeeze(prob, axis=0)

        pts = tf.cast(tf.where(prob), dtype=tf.int32)

        conf = tf.gather_nd(prob, pts)
        pts = tf.concat([pts[:, 1][..., tf.newaxis], pts[:, 0][..., tf.newaxis]], axis=-1)
        pts = tf.concat([tf.cast(pts, dtype=tf.float32), conf[..., tf.newaxis]], axis=-1)
        pts = tf.transpose(pts)

        return pts.numpy().T, conf.numpy(), prob.numpy()