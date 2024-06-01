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
    points_1 = set_1['position']    
    points_2 = set_2['position']

    matches = {'points_1':[], 'points_2':[]}

    for i in range(len(points_1)):
        min_distance = np.inf
        min_index = -1
        for j in range(len(points_2)):
            distance = np.linalg.norm(np.array(points_1[i])-np.array(points_2[j]))
            if distance < min_distance:
                min_distance = distance
                min_index = j
        matches['points_1'].append(points_1[i])
        matches['points_2'].append(points_2[min_index])

    return matches

def data_association_2Dto3D(image_points, world_points, camera):
    image_points = image_points['position']
    world_points = world_points['position']
    
    matches = {'points_1':[], 'points_2':[]}
    
    for i in range(len(image_points)):
        image_point = image_points[i]
        min_distance = np.inf
        min_index = -1
        
        for j in range(len(world_points)):
            world_point = world_points[j]
            is_inside, projected_image_point = camera.project_point(world_point)
            if not is_inside: continue
            distance = np.linalg.norm(image_point-projected_image_point)
            if distance < min_distance:
                min_distance = distance
                min_index = j
        
        matches['points_1'].append(image_point)
        matches['points_2'].append(world_points[min_index])

    return matches