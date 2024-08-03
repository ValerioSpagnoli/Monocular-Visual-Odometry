import numpy as np

def data_association_on_appearance(set_1, set_2):
    points_1 = set_1['position']
    appearance_1 = set_1['appearance']

    points_2 = set_2['position']
    appearance_2 = set_2['appearance']

    matches = {'points_1':[], 'points_2':[], 'appearance':[]}

    for i in range(len(points_1)):
        for j in range(len(points_2)):
            if appearance_1[i] == appearance_2[j]:
                matches['points_1'].append(points_1[i])
                matches['points_2'].append(points_2[j])
                matches['appearance'].append(appearance_1[i])

    return matches

def data_association_on_distance(set_1, set_2):
    points_1_position = set_1['position']    
    points_1_appearance = set_1['appearance']
    
    points_2_position = set_2['position']
    points_2_appearance = set_2['appearance']

    matches = {'points_1':[], 'points_2':[]}

    mean_distance = 0
    mean_appereance_distance = 0
    for i in range(len(points_1_position)):
        min_distance = np.inf
        min_index = -1

        for j in range(len(points_2_position)):
            distance = np.linalg.norm(np.array(points_1_position[i])-np.array(points_2_position[j]))
            if distance < min_distance:
                min_distance = distance
                min_index = j

        matches['points_1'].append(points_1_position[i])
        matches['points_2'].append(points_2_position[min_index])
        mean_distance += min_distance
        mean_appereance_distance += np.linalg.norm(np.array(points_1_appearance[i])-np.array(points_2_appearance[min_index]))

    mean_distance /= len(points_1_position)
    mean_appereance_distance /= len(points_1_position)

    return matches, mean_distance, mean_appereance_distance

def data_association_2Dto3D(image_points, world_points, camera):
    image_points_position = image_points['position']
    image_points_appereance = image_points['appearance']

    world_points_position = world_points['position']
    world_points_appereance = world_points['appearance']
    
    matches = {'points_1':[], 'appearance_1':[], 'points_2':[], 'appearance_2':[]}

    mean_distance = 0
    mean_appearance_distance = 0
    for i in range(len(image_points_position)):
        image_point = image_points_position[i]
        min_distance = np.inf
        min_index = -1
        
        for j in range(len(world_points_position)):
            world_point = world_points_position[j]
            is_inside, projected_image_point = camera.project_point(world_point)
            if not is_inside: continue
            distance = np.linalg.norm(image_point-projected_image_point)
            if distance < min_distance:
                min_distance = distance
                min_index = j
        
        matches['points_1'].append(image_point)
        matches['appearance_1'].append(image_points_appereance[i])
        matches['points_2'].append(world_points_position[min_index])
        matches['appearance_2'].append(world_points_appereance[min_index])
        mean_distance += min_distance
        mean_appearance_distance += np.linalg.norm(np.array(image_points_appereance[i])-np.array(world_points_appereance[min_index]))

    mean_distance /= len(image_points_position)
    mean_appearance_distance /= len(image_points_position)

    return matches, mean_distance, mean_appearance_distance

def data_association_compatible_2Dto3D(image_points, world_points, camera):    
    matches, mean_distance, mean_appearance_distance = data_association_2Dto3D(image_points, world_points, camera)

    image_points_position = []
    image_points_appearance = []
    world_points_position = []
    world_points_appearance = []

    for i in range(len(matches['points_1'])):
        image_point_position = matches['points_1'][i]
        image_point_appearance = matches['appearance_1'][i]
        
        projected_world_point_position = camera.project_point(matches['points_2'][i])[1]
        projected_world_point_appearance = matches['appearance_2'][i]

        distance = np.linalg.norm(image_point_position-projected_world_point_position)
        appearance_distance = np.linalg.norm(np.array(image_point_appearance)-np.array(projected_world_point_appearance))

        if distance < mean_distance and appearance_distance < mean_appearance_distance:
            image_points_position.append(image_point_position)
            image_points_appearance.append(image_point_appearance)
            world_points_position.append(matches['points_2'][i])
            world_points_appearance.append(matches['appearance_2'][i])
            
    matches = {'points_1':image_points_position, 'appearance_1':image_points_appearance, 'points_2':world_points_position, 'appearance_2':world_points_appearance}

    return matches