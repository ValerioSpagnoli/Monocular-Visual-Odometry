import os
import time
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(handler)

class Data:

    def __init__(self, folder_path='data/'):    
        self.__folder_path = folder_path  
        self.__trajectory = self.__load_trajectory_data()
        self.__world = self.__load_world_data()
        self.__measurements = self.__load_measurements_data()


    #* TRAJECTORY DATA
    #* ------------------------------------------------------------------------------------- #
    
    def __load_trajectory_data(self):
        logging.info(f'Loading trajectory data from {self.__folder_path}/trajectory.dat')
        start = time.time()

        try:
            with open(f'{self.__folder_path}/trajectory.dat', 'r') as trajectory_file:
                trajectory_data = trajectory_file.readlines()
        except FileNotFoundError:
            assert False, f'Trajectory file not found at {self.__folder_path}/trajectory.dat'
        
        trajectory = dict()
        for line in trajectory_data:
            tokens = line.split(' ')
            tokens[-1] = tokens[-1].split('\n')[0]
            
            data = []
            for i in range(len(tokens)):
                if tokens[i] == '': continue
                else: data.append(float(tokens[i]))

            pose_id = int(data[0])
            odometry_pose = data[1:4]
            ground_truth_pose = data[4:7]

            trajectories = dict()
            trajectories['odometry_pose'] = odometry_pose
            trajectories['ground_truth_pose'] = ground_truth_pose

            trajectory[pose_id] = trajectories
        
        logging.info(f'{(time.time()-start):.2f} [s] - Trajectory data loaded successfully!')
        return trajectory
    

    def get_trajectory(self, pose_id=None):

        '''
        Returns the trajectory data for a specific pose id. If no pose id is given, returns the entire trajectory data.
        '''

        if pose_id is not None:
            return self.trajectory[pose_id] 
        return self.__trajectory
    

    def print_trajectory(self, pose_id=None):    

        '''
        Prints the trajectory data for a specific pose id. If no pose id is given, prints the entire trajectory data.
        
        Parameters:
        - pose_id (int, optional): The pose id for which the trajectory data is to be printed.
        '''

        print('Trajectory data: ')

        if pose_id is not None:
            print(f'Pose id: {pose_id}')
            print(f'  Odometry pose:     {self.__trajectory[pose_id]["odometry_pose"]}')
            print(f'  Ground truth pose: {self.__trajectory[pose_id]["ground_truth_pose"]}')
            return

        for key, value in self.__trajectory.items():
            print(f'Pose id: {key}')
            print(f'  Odometry pose:     {value["odometry_pose"]}')
            print(f'  Ground truth pose: {value["ground_truth_pose"]}')   




    #* WORLD DATA
    #* ------------------------------------------------------------------------------------- #
            
    def __load_world_data(self):
        logging.info(f'Loading world data from {self.__folder_path}/world.dat')
        start = time.time()

        try:
            with open(f'{self.__folder_path}/world.dat', 'r') as world_file:
                world_data = world_file.readlines()
        except FileNotFoundError:
            assert False, f'World file not found at {self.__folder_path}/world.dat'
            
        world = dict()
        for line in world_data:
            tokens = line.split(' ')
            tokens[-1] = tokens[-1].split('\n')[0]
            
            data = []
            for i in range(len(tokens)):
                if tokens[i] == '': continue
                else: data.append(float(tokens[i]))

            landmark_id = int(data[0])
            landmark_position = data[1:4]
            landmark_appearance = data[4:]

            landmarks = dict()
            landmarks['landmark_position'] = landmark_position
            landmarks['landmark_appearance'] = landmark_appearance

            world[landmark_id] = landmarks
            
        logging.info(f'{(time.time()-start):.2f} [s] - World data loaded successfully!')
        return world


    def get_world(self, landmark_id=None):

        '''
        Returns the world data for a specific landmark id. If no landmark id is given, returns the entire world data.
        '''

        if landmark_id is not None:
            return self.__world[landmark_id]
        return self.__world


    def print_world(self, landmark_id=None):

        '''
        Prints the world data for a specific landmark id. If no landmark id is given, prints the entire world data.

        Parameters:
        - landmark_id (int, optional): The landmark id for which the world data is to be printed.
        '''

        print('World data: ')

        if landmark_id is not None:
            print(f'Landmark id: {landmark_id}')
            print(f'  Landmark position:   {self.__world[landmark_id]["landmark_position"]}')
            print(f'  Landmark appearance: {self.__world[landmark_id]["landmark_appearance"]}')
            return

        for key, value in self.__world.items():
            print(f'Landmark id: {key}')
            print(f'  Landmark position:   {value["landmark_position"]}')
            print(f'  Landmark appearance: {value["landmark_appearance"]}')




    #* MEASUREMENTS DATA
    #* ------------------------------------------------------------------------------------- #

    def __load_measurements_data(self):
        logging.info(f'Loading measurements data from {self.__folder_path}/meas-*.dat')
        start = time.time()

        files = os.listdir("data/")
        files = [f for f in files if f.startswith("meas-")]
        files.sort()
        
        measurements = dict()

        for f in files:
            try:
                with open(f'{self.__folder_path}/{f}', 'r') as measurements_file:
                    measurements_data = measurements_file.readlines()
            except FileNotFoundError:
                assert False, f'Measurements file not found at {self.__folder_path}/{f}'

            sequence_id = 0
            gt_pose = []
            odom_pose = []
            points = {}

            for data in measurements_data:
                if data.startswith('seq'):
                    tokens = data.split(':')
                    tokens[-1] = tokens[-1].split('\n')[0]
                    sequence_id = int(tokens[1])

                if data.startswith('gt_pose'):
                    tokens = data.split(':')
                    tokens[-1] = tokens[-1].split('\n')[0]
                    tokens = tokens[1].split(' ')
                    for token in tokens:
                        if token == '': continue
                        gt_pose.append(float(token))
                    
                if data.startswith('odom_pose'):
                    tokens = data.split(':')
                    tokens[-1] = tokens[-1].split('\n')[0]
                    tokens = tokens[1].split(' ')
                    for token in tokens:
                        if token == '': continue
                        odom_pose.append(float(token))

                if data.startswith('point'):
                    tokens = data.split(' ')
                    tokens[-1] = tokens[-1].split('\n')[0]
                    point_data = []
                    for token in tokens:
                        if token == '' or token == 'point': continue
                        point_data.append(float(token))

                    point_id_current_measurement = int(point_data[0])
                    actual_point_id = int(point_data[1])
                    image_point = [point_data[2], point_data[3]]
                    appearance = point_data[4:]

                    points[point_id_current_measurement] = {
                        'actual_point_id': actual_point_id,
                        'image_point': image_point,
                        'appearance': appearance
                    }

                measurements[sequence_id] = {
                    'gt_pose': gt_pose,
                    'odom_pose': odom_pose,
                    'points': points
                }

        logging.info(f'{(time.time()-start):.2f} [s] - Measurements data loaded successfully!')
        return measurements 


    def get_measurements(self, sequence_id=None):
        
        '''
        Returns the measurements data for a specific sequence_id. If no sequence_id is given, returns the entire measurements data.
        '''

        if sequence_id is not None:
            return self.__measurements[sequence_id]
        return self.__measurements
    

    def print_measurements(self, sequence_id=None):
        
        '''
        Prints the measurements data for a specific sequence_id. If no sequence_id is given, prints the entire measurements data.
        
        Parameters:
        - sequence_id (int, optional): The sequence_id for which the measurements data is to be printed.
        '''

        print('Measurements data: ')

        if sequence_id is not None:
            print(f'Sequence_id: {sequence_id}')
            print(f'  Ground truth pose: {self.__measurements[sequence_id]["gt_pose"]}')
            print(f'  Odometry pose:     {self.__measurements[sequence_id]["odom_pose"]}')
            print(f'  Points: ')
            for point_id, point_data in self.__measurements[sequence_id]["points"].items():
                print(f'    Point id: {point_id}')
                print(f'      Actual point id: {point_data["actual_point_id"]}')
                print(f'      Image point:     {point_data["image_point"]}')
                print(f'      Appearance:      {point_data["appearance"]}')
            return

        for key, value in self.__measurements.items():
            print(f'Sequence_id: {key}')
            print(f'  Ground truth pose: {value["gt_pose"]}')
            print(f'  Odometry pose:     {value["odom_pose"]}')
            print(f'  Points: ')
            for point_id, point_data in value["points"].items():
                print(f'    Point id: {point_id}')
                print(f'      Actual point id: {point_data["actual_point_id"]}')
                print(f'      Image point:     {point_data["image_point"]}')
                print(f'      Appearance:      {point_data["appearance"]}')
            print('-'*50)