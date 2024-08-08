import numpy as np

class Camera:
    def __init__(self):
        self.__camera_matrix, self.__camera_transform, self.__camera_range, self.__camera_resolution = self.__load_camera_data()
        self.__c_T_w = np.eye(4)

    def __load_camera_data(self):
        camera_matrix = []
        camera_transform = []
        camera_range = []
        camera_resolution = []

        try:
            file = open("data/camera.dat", "r")
            lines = file.readlines()

            for i in range(len(lines)):
                line = lines[i]
                if line.startswith('camera matrix'): 
                    for j in range(1,4):
                        line = lines[i+j]
                        tokens = [int(x) for x in line.split()]
                        camera_matrix.append(tokens)
            
                elif line.startswith('cam_transform'):
                    for j in range(1,5):
                        line = lines[i+j]
                        tokens = [float(x) for x in line.split()]
                        camera_transform.append(tokens)
                
                elif line.startswith('z_near'):
                    tokens = line.split()
                    z_near = float(tokens[1])
                    camera_range.append(z_near)
                
                elif line.startswith('z_far'):
                    tokens = line.split()
                    z_far = float(tokens[1])
                    camera_range.append(z_far)

                elif line.startswith('width'):
                    tokens = line.split()
                    width = int(tokens[1])
                    camera_resolution.append(width)

                elif line.startswith('height'):
                    tokens = line.split()
                    height = int(tokens[1])
                    camera_resolution.append(height)

        except: print("Error: Could not find the camera.dat file")

        camera_matrix = np.array(camera_matrix)
        camera_transform = np.array(camera_transform)
        
        return camera_matrix, camera_transform, camera_range, camera_resolution
    
    def get_camera_matrix(self):
        return self.__camera_matrix 
    
    def get_intrinsic_camera_matrix(self):
        intrinsic_camera_matrix = np.zeros((3,4))
        intrinsic_camera_matrix[:3,:3] = self.__camera_matrix
        return intrinsic_camera_matrix
        
    def get_camera_transform(self):
        return self.__camera_transform
    
    def get_camera_range(self):
        return self.__camera_range
    
    def get_camera_resolution(self):
        return self.__camera_resolution
    
    def set_c_T_w(self, c_T_w):
        self.__c_T_w = c_T_w

    def get_c_T_w(self):
        return self.__c_T_w
    
    def project_point(self, world_point):
        """
        Projects a 3D point from world coordinates to image coordinates.

        Args:
            world_point (numpy.ndarray): 3D point in initial camera coordinates (i.e. the camera coordinates of the first frame).

        Returns:
            tuple: A tuple containing a boolean value indicating if the point is successfully projected and the projected image point coordinates.
                   - If the point is successfully projected, the boolean value is True and the image point coordinates are returned.
                   - If the point is not successfully projected (e.g. point is behind the camera, outside the camera range, or outside the camera plane),
                     the boolean value is False and None is returned.
        """

        #* c_T_w:        world in camera pose 
        #* world_point:  3D point in initial camera coordinates (i.e. the camera coordinates of the first frame)
        #* camera_point: 3D point in current camera coordinates
        #* image_point:  2D point in image coordinates

        #* Projection: camera_point = c_T_w * world_point
        #*             image_point  = camera_matrix * camera_point

        camera_point_hom = self.__c_T_w @ np.append(world_point, 1)
        camera_point = camera_point_hom[:3] / camera_point_hom[3]

        image_point_hom = self.__camera_matrix @ camera_point   
        image_point = image_point_hom[:2] / image_point_hom[2]

        #* point is behind the camera
        if camera_point[2] <= 0: return False, None

        #* point is outside the camera range
        # if camera_point[2] <= self.__camera_range[0] or camera_point[2] >= self.__camera_range[1]: return False, None

        #* point is outside the camera plane
        if image_point[0] < 0 or image_point[0] >= self.__camera_resolution[0] or \
           image_point[1] < 0 or image_point[1] >= self.__camera_resolution[1]: 
            return False, None

        return True, image_point
    
    def project_points(self, world_points):
        """
        Projects a list of world points onto the image plane.

        Args:
            world_points (list): A list of world points to be projected.

        Returns:
            list: A list of image points corresponding to the projected world points.
        """
        image_points = []
        for world_point in world_points:
            is_inside, image_point = self.project_point(world_point)
            if is_inside: image_points.append(image_point)
        return image_points