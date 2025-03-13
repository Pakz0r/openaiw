import numpy as np

_X_SIZE = 2304 #1920  
_Y_SIZE = 1296 #1080
_Q_FACTOR = 20 # quantization factor

def preprocess(current_skeletons, tolerance=0.15, mean_confidence_tolerance=0.35):
    '''
    This function preprocess the skeleton returned by openpose frame by frame.
    First the less informative keypoints are removed, because they bring no informations about skeleton position considering them just as noise.
    After this, we calculate in just one cicle, skeleton by skeleton, the center of mass on x and y and the mean confidence. 
    For each keypoint we set the corresponding confidence to zero if it is less than a certain tolerance.
    If the mean confidence of the skeleton is less than a certain threshold, the skeleton is removed because the probability
    of  it to really be in the scene is poor.

    INPUT: 
        - current_skeletons: 2D list of skeltons which on the rows has the single skeleton keypoints
        - tolerance: threshold for which if the confidence is lower than it, the confidence itself is set to zero.
        - mean_confidence_tolerance: threshold for which if the mean confidence is lower than it, the skeleton is discarded
    
    OUTPUT:
        - list of tuple ([center_of_mass_x, center_of_mass_y], person). In the first element there is a list of two elements
          representing the center of mass on the two axes. The second element is the list of the i-th person keypoints.
    '''


    # We remove  keypoints: 15 16 23 24 20 21 
    # Keypoints to use for center of mass: 1 2 5 8 9 12
    keypoints_to_remove = [24, 23, 21, 20, 16, 15]
    new_current_skeletons = []
    
    # check if there are any skeleton in the frame (otherwise openpose returns a None type)
    if current_skeletons is not None:
        for person in current_skeletons:
            keypoints_for_center_of_mass = [12, 9, 8, 5, 2, 1]
            for i in keypoints_to_remove:
                person = np.delete(person, i, axis=0)
            # il frame pezzotto era il 136!
            # We first list through all keypoints confidences, and if they are lower than a delta are set to zero
            mean_confidence = 0.
            for i in range(len(person)):    # cycle through rows (keypoints)
                # set the confidence to 0 if it's under a certain threshold
                if person[i][2] < tolerance:
                    person[i][2] = 0   
        
                mean_confidence += person[i][2]    # accumulate the average confidence
            
            # delete, from the center of mass calculus, keypoints where x and/or y coordinate is zero 
            for i, idx in enumerate(keypoints_for_center_of_mass):
                if(person[idx][0]==0. and person[idx][1]==0.):  # IF X OR Y OF THE KEYPOINT IDX IS ZERO
                    keypoints_for_center_of_mass[i] = None      # Set the index to None to be removed later
            
            keypoints_for_center_of_mass = [x for x in keypoints_for_center_of_mass if x!=None]

            # calculate center of mass based on the body bust (shoulders, neck, hip)
            center_of_mass_x = np.sum(np.take(person, keypoints_for_center_of_mass, axis=0)[:,0])/len(keypoints_for_center_of_mass)
            center_of_mass_y = np.sum(np.take(person, keypoints_for_center_of_mass, axis=0)[:,1])/len(keypoints_for_center_of_mass)
            
            # mean confidence
            mean_confidence = mean_confidence/len(person)
            
            # calculate radius
            torso_keypoints = np.take(person,keypoints_for_center_of_mass,axis=0)[:,:2] # taking coordinates x,y for keypoints 1 and 8
            
            torso_perimeter =  0
            for i  in range(len(torso_keypoints)-1):
                torso_perimeter += np.linalg.norm(torso_keypoints[i,:] - torso_keypoints[i+1,:])

            if torso_perimeter == 0:
                 # [[x1,y1],[x2,y2]]
                mean_confidence = 0
            else:
                radius = int((torso_perimeter/len(keypoints_for_center_of_mass)) * 1.10)
                # the radius is scaled by the number of the keypoints of center of mass detected in the scene
                # and a multiplicative factor

            # remove the scores that at this point are useless
            person = np.delete(person, 2, 1)

            if mean_confidence > mean_confidence_tolerance: # if the skeleton is valid, we add it to the output
                new_current_skeletons.append(([center_of_mass_x, center_of_mass_y, radius], person))

    return new_current_skeletons


def postprocess(skeleton_window):
    '''
    This function postprocessed the skeleton windows before they get putted in the final data structure that goes in input to the model.
    There are performed coordinates min-max normalization in [0,1] range and then a quantization to a specific range of values to avoid
    model problems due to a large set of possible values.

    INPUT:
        - skeleton_window: list of skeleton windows of 75 frames

    OUTPUT:
        - postprocessed skeleton_windows
    
    '''
    # it should be done for the whole windows of 75 frames of a single skeleton
    # The shape of the windows is actually (75,19,2 ), it need to be turned into a (75,38)
    skeleton_window = np.reshape(skeleton_window, (25,38))
    
    for skeleton in skeleton_window:
        # Isolating the x coordinates and the y coordinates
        keypoints_x = skeleton[::2]  # x  - start at the beginning at take every second item
        keypoints_y = skeleton[1::2] # y - start at second item and take every second item

        # MIN-MAX NORMALIZATION
        # 0 values must be removed for the min-max normalization
        x = np.array(keypoints_x)
        x = x[ x != 0]
        y = np.array(keypoints_y)
        y = y[ y != 0]

        # We also round float values holding just N decimal places
        for i in range(skeleton.shape[0]):
            if i%2==0: #if it's the x coord
                skeleton[i] = (keypoints_x[int(i/2)] - x.min())/(x.max()-x.min())
            else: # if it's the y coord
                skeleton[i] = (keypoints_y[int(i/2)] - y.min())/(y.max()-y.min())
    
    return _quantization(skeleton_window)


def _quantization(skeleton_window):

    new_skeletons = skeleton_window
    
    # Define bins of quantization
    x_bins = np.linspace(0,1,int(_X_SIZE/_Q_FACTOR))
    y_bins = np.linspace(0,1,int(_Y_SIZE/_Q_FACTOR))
    
    for skeleton in new_skeletons:
        for i in range(skeleton.shape[0]):
            if i%2==0:
                #if it's x coord
                skeleton[i] = _find_bin(skeleton[i], x_bins)
            else:
                #if it's y coord
                skeleton[i] = _find_bin(skeleton[i], y_bins)

    return new_skeletons


def _find_bin(value, bins):
    for i in range(len(bins)):
        if value <= bins[i]:
            return bins[i]
