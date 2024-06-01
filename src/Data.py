import os

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
        world_map = {'id': [], 'position': [], 'appearance': []}
        
        try:
            file = open("data/world.dat", "r")
            lines = file.readlines()
            
            for line in lines:
                tokens = line.split()
                id = int(tokens[0])
                position = [float(x) for x in tokens[1:4]]
                appearance = [float(x) for x in tokens[4:]]
                world_map['id'].append(id)
                world_map['position'].append(position)
                world_map['appearance'].append(appearance)
                
        except: print("Error: Could not find the world.dat file")

        return world_map


    def __load_measurements_data(self):
        measurements = {'gt_pose':[], 'image_points':[]}

        for i in range(121):
            measurement_file_name = f'meas-{i:05d}.dat'

            try:
                file = open(f'data/{measurement_file_name}', "r")
                lines = file.readlines()

                image_points = {'current_point_id':[], 'actual_point_id':[], 'position':[], 'appearance':[]}
                
                for line in lines:

                    if line.startswith('gt_pose'):
                        tokens = line.split()
                        gt_pose = [float(x) for x in tokens[1:]]
                        measurements['gt_pose'].append(gt_pose)
                        
                    elif line.startswith('point'):
                        tokens = line.split()
                        current_point_id = int(tokens[1])
                        actual_point_id = int(tokens[2])
                        position = [float(x) for x in tokens[3:5]]
                        appearance = [float(x) for x in tokens[5:]]
                        
                        image_points['current_point_id'].append(current_point_id)
                        image_points['actual_point_id'].append(actual_point_id)
                        image_points['position'].append(position)
                        image_points['appearance'].append(appearance)

                measurements['image_points'].append(image_points)

            except: print(f"Error: Could not find the {measurement_file_name} file")
                
        return measurements
        
    def get_trajectory_data(self):
        return self.__trajectory_data
    
    def get_world_data(self):
        return self.__world_data
    
    def get_measurements_data(self):
        return self.__measurements_data
    
    def get_measurements_data_points(self, index):
        return self.__measurements_data['image_points'][index]