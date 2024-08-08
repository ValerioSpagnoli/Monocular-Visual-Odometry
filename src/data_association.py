import numpy as np

def data_association_on_appearance(set_1, set_2, projection=0, camera=None):
    """
    Performs data association based on appearance between two sets of data points.
    Args:
        set_1 (dict): First set of data points containing 'position' and 'appearance' information.
        set_2 (dict): Second set of data points containing 'position' and 'appearance' information.
        projection (int, optional): If project the points to the camera plane. 0: No projection, 1: Project set_1, 2: Project set_2. Defaults to 0.
        camera (object, optional): Camera object used for projection. Defaults to None.
    Returns:
        dict: Dictionary containing matched data points based on appearance.
            - If projection is 1: {'points_1': [], 'projected_points_1': [], 'points_2': [], 'appearance': []}
            - If projection is 2: {'points_1': [], 'points_2': [], 'projected_points_2': [], 'appearance': []}
            - If projection is not 1 or 2: {'points_1': [], 'points_2': [], 'appearance': []}
    """

    assert(projection in [0, 1, 2], "Projection should be 0, 1 or 2.")

    points_1 = set_1['position']
    appearance_1 = set_1['appearance']

    points_2 = set_2['position']
    appearance_2 = set_2['appearance']

    if projection == 1: matches = {'points_1':[], 'projected_points_1':[], 'points_2':[], 'appearance':[]}
    elif projection == 2: matches = {'points_1':[], 'points_2':[], 'projected_points_2':[], 'appearance':[]}
    else: matches = {'points_1':[], 'points_2':[], 'appearance':[]}

    for i in range(len(points_1)):
        for j in range(len(points_2)):
            if appearance_1[i] == appearance_2[j]:
                point_1 = points_1[i]
                point_2 = points_2[j]
                appearance = appearance_1[i]

                if projection == 1:
                    is_inside, projected_point_1 = camera.project_point(point_1)
                    if not is_inside: continue
                    matches['points_1'].append(point_1)
                    matches['projected_points_1'].append(projected_point_1)
                    matches['points_2'].append(point_2)
                    matches['appearance'].append(appearance)
                    
                elif projection == 2:
                    is_inside, projected_point_2 = camera.project_point(point_2)
                    if not is_inside: continue
                    matches['points_1'].append(point_1)
                    matches['points_2'].append(point_2)
                    matches['projected_points_2'].append(projected_point_2)
                    matches['appearance'].append(appearance)

                else:
                    matches['points_1'].append(point_1)
                    matches['points_2'].append(point_2)
                    matches['appearance'].append(appearance)

    return matches


def data_association_2Dto3D(image_points, world_points, camera):
    matches = data_association_on_appearance(image_points, world_points, projection=2, camera=camera)

    image_points_position = matches['points_1']
    world_points_position = matches['points_2']
    projected_world_points_position = matches['projected_points_2']
    appearance = matches['appearance']
    
    matches = {'points_1':[], 'points_2':[], 'projected_points_2':[], 'appearance':[]}
    matched_image_points = []

    for i in range(len(image_points_position)):
        if i in matched_image_points: continue

        image_point = image_points_position[i]
        min_distance = np.inf
        min_index = -1

        for j in range(len(projected_world_points_position)):
            projected_world_point = projected_world_points_position[j]
            distance = np.linalg.norm(image_point-projected_world_point)
            if distance < min_distance:
                min_distance = distance
                min_index = j
        
        if min_index != -1: matched_image_points.append((i, min_index))

    for i, j in matched_image_points:
        matches['points_1'].append(image_points_position[i])
        matches['points_2'].append(world_points_position[j])
        matches['projected_points_2'].append(projected_world_points_position[j])
        matches['appearance'].append(appearance[i])

    return matches