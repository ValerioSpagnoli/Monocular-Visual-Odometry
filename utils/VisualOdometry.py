from utils.Camera import Camera

class VisualOdometry:
    def __init__(self):
        self._camera = Camera()
        self._camera_matrix = self._camera.get_camera_matrix()
        self._intrinsic_matrix = self._camera.get_intrinsic_matrix()
        self._camera_transform = self._camera.get_camera_transform()
        self._extrinsic_matrix = self._camera.get_extrinsic_matrix()
        self._camera_range = self._camera.get_camera_range()
        self._camera_resolution = self._camera.get_camera_resolution()

        self.map = None

    def initialize(self):
        pass

    def triangulate(self):
        pass

    def track(self):
        pass