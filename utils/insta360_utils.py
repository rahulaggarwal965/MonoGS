import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from gaussian_splatting.utils.graphics_utils import focal2fov
import cv2
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

#insta360_K = np.array([
#  [305.8511660909695, 0.0, 571.9976412068438], 
#  [0.0, 304.8739784192282, 578.9734717897188], 
#  [0.0, 0.0, 1.0]
#  ])

insta360_D = np.array([[0.0829955798117611], [-0.027906274475464777], [0.0076202648985968895], [-0.0010836351255689319]])


class InstaSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback)
        self.latest_image = None
        self.counter = 0



    def spin(self):
        rospy.spin()

def undistort_image(image, K, D, zoom_factor = 1):
    h, w = image.shape[:2]
    new_K = K.copy()
    #new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
    #rospy.loginfo(f"New camera matrix: {new_K}")
    #log new k
    new_K[:2, :2] *= zoom_factor
    #zoom in a bit
    #print("new k", new_K)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_img


class Insta360PinholeDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        if "zoom_factor" in calibration.keys():
            self.zoom_factor = calibration["zoom_factor"]
        else:
            self.zoom_factor = 1.0
        # Camera prameters
        self.fx = calibration["fx"] * self.zoom_factor
        self.fy = calibration["fy"] * self.zoom_factor
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)

        self.og_k = np.array([[calibration["fx"], 0.0, self.cx], [0.0, calibration["fy"], self.cy], [0.0, 0.0, 1.0]])
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

        self.counter = 0
        self.latest_image = None
        if 'path' in config["Dataset"].keys():
            self.ros_stream = False
            self.cap = cv2.VideoCapture(config["Dataset"]["path"])
        else:
            self.ros_stream = True
            rospy.init_node('image_subscriber', anonymous=True)
            self.image_sub = rospy.Subscriber('/front_camera_image/compressed', CompressedImage, self.image_callback)
    
    def image_callback(self, msg):
        self.counter += 1
        if self.counter % 3 == 0:
            self.counter = 0
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if latest_image is not None:
                    self.latest_image = latest_image
            except Exception as e:
                rospy.logerr(f"Error processing image: {e}")

    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        if self.ros_stream:
            while self.latest_image is None:
                rospy.loginfo("Waiting for image")
                rospy.sleep(0.1)

            image = self.latest_image
            image=cv2.cvtColor(front_frame_rect, cv2.COLOR_BGR2RGB)
        else:
            ret, rec_frame = self.cap.read()
            front_frame = np.array(rec_frame[:, :1152])
            front_frame_rect = undistort_image(front_frame, self.og_k, insta360_D, self.zoom_factor)
            image=cv2.cvtColor(front_frame_rect, cv2.COLOR_BGR2RGB)
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        return image, depth, pose