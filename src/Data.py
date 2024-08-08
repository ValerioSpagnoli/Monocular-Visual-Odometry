from src.geometry_utils import *

class Data:
    def __init__(self) :
    
        self.__trajectory_data = self.__load_trajectory_data()
        self.__world_data = self.__load_world_data()
        self.__measurements_data = self.__load_measurements_data()

    def __load_trajectory_data(self):
        trajectory = []

        try:
            file = open("data/trajectory.dat", "r")
            lines = file.readlines()
            
            for line in lines:
                tokens = line.split()
                position = [float(x) for x in tokens[4:]]
                trajectory.append(position)

        except: print("Error: Could not find the trajectory.dat file")

        return trajectory

    def __load_world_data(self):
        world_map = {'position': [], 'appearance': []}
        
        try:
            file = open("data/world.dat", "r")
            lines = file.readlines()
            
            for line in lines:
                tokens = line.split()
                id = int(tokens[0])
                position = [float(x) for x in tokens[1:4]]
                appearance = [float(x) for x in tokens[4:]]
                world_map['position'].append(position)
                world_map['appearance'].append(appearance)
                
        except: print("Error: Could not find the world.dat file")

        return world_map


    def __load_measurements_data(self):
        measurements = []

        for i in range(121):
            measurement_file_name = f'meas-{i:05d}.dat'

            try:
                file = open(f'data/{measurement_file_name}', "r")
                lines = file.readlines()

                image_points = {'current_point_id':[], 'actual_point_id':[], 'position':[], 'appearance':[]}
                
                for line in lines:

                    if line.startswith('point'):
                        tokens = line.split()
                        current_point_id = int(tokens[1])
                        actual_point_id = int(tokens[2])
                        position = [float(x) for x in tokens[3:5]]
                        appearance = [float(x) for x in tokens[5:]]
                        
                        image_points['current_point_id'].append(current_point_id)
                        image_points['actual_point_id'].append(actual_point_id)
                        image_points['position'].append(position)
                        image_points['appearance'].append(appearance)

                measurements.append(image_points)

            except: print(f"Error: Could not find the {measurement_file_name} file")
                
        return measurements
        
    def get_trajectory_data(self):
        """
        Returns the trajectory data.

        Returns:
            list: A list of ground truth trajectory data positions.
        """
        return self.__trajectory_data
    
    def get_trajectory_data_poses(self):
        """
        Returns the ground truth poses of the trajectory data.

        Returns:
            list: A list of ground truth trajectory data poses.
        """
        gt_trajectory = self.__trajectory_data  
        gt_poses = []
        for i in range(0, len(gt_trajectory)):
            gt_pose = v2T([gt_trajectory[i][0], gt_trajectory[i][1], 0, 0, 0, gt_trajectory[i][2]])
            gt_poses.append(gt_pose)
        return gt_poses
    
    def get_world_data(self):
        """
        Returns the world data.

        Returns:
        - dict: A dictionary containing the world data.
            - 'position': A list of world data positions.
            - 'appearance': A list of world data appearances.
        """
        return self.__world_data
    
    def get_measurements_data(self, index):
        """
        Returns the measurements data at the specified index.

        Parameters:
        - index (int): The index of the measurements data to retrieve.

        Returns:
        - dict: A dictionary containing the measurements data.
            - 'current_point_id': A list of current point IDs.
            - 'actual_point_id': A list of actual point IDs.
            - 'position': A list of positions.
            - 'appearance': A list of appearances.
        """
        return self.__measurements_data[index]