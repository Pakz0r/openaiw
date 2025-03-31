import numpy as np

_X_SIZE = 2304 #1920  
_Y_SIZE = 1296 #1080
_Q_FACTOR = 20 # quantization factor

def preprocess(current_skeletons, tolerance=0.15, mean_confidence_tolerance=0.6):
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
    skeleton_window = np.reshape(skeleton_window, (75,38))
    
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



#             ################ Iterazione: 1519 ################
# previous_skeletons:4
# missing_frames_counters:[7, 0, 0, 0]
# number of windows:4
# 6
# [([575.63681640625, 556.18974609375, 27], array([[  0.     ,   0.     ],
#        [568.4832 , 506.66913],
#        [574.51764, 506.64648],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [577.495  , 586.27606],
#        [589.11395, 583.3148 ],
#        [597.91815, 647.95856],
#        [636.288  , 706.9002 ],
#        [568.57434, 598.0421 ],
#        [553.77924, 674.51776],
#        [530.26074, 739.23285],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [547.853  , 762.80194],
#        [624.4409 , 724.58496]], dtype=float32)), ([800.5958658854166, 534.6549886067709, 46], array([[809.74   , 459.53107],
#        [798.1689 , 492.0085 ],
#        [765.79626, 497.87375],
#        [756.8944 , 539.1222 ],
#        [742.3695 , 577.46155],
#        [833.34924, 486.23657],
#        [848.11194, 530.1821 ],
#        [859.8906 , 562.66394],
#        [804.0381 , 577.2549 ],
#        [783.4507 , 577.3008 ],
#        [789.2328 , 636.2014 ],
#        [786.25806, 695.12225],
#        [818.77185, 577.25543],
#        [809.97626, 636.25195],
#        [798.09265, 686.22894],
#        [786.32434, 456.78662],
#        [812.9746 , 456.54562],
#        [812.8811 , 698.09436],
#        [801.0351 , 715.5367 ]], dtype=float32)), ([1762.0436197916667, 870.2245279947916, 60], array([[1742.9574 ,  742.2513 ],
#        [1772.3557 ,  809.79895],
#        [1813.5668 ,  812.6986 ],
#        [1822.3263 ,  880.6413 ],
#        [1819.3959 ,  933.5127 ],
#        [1728.2113 ,  803.98   ],
#        [1695.8606 ,  877.5167 ],
#        [1675.2004 ,  933.45294],
#        [1751.8173 ,  930.6985 ],
#        [1778.124  ,  933.58075],
#        [1766.438  , 1024.7142 ],
#        [   0.     ,    0.     ],
#        [1728.1866 ,  930.5903 ],
#        [1728.1896 , 1021.8251 ],
#        [   0.     ,    0.     ],
#        [1798.696  ,  756.8796 ],
#        [1754.6687 ,  751.01965],
#        [   0.     ,    0.     ],
#        [   0.     ,    0.     ]], dtype=float32)), ([527.9275390625, 582.06728515625, 31], array([[577.52484, 483.1553 ],
#        [530.2685 , 512.61755],
#        [533.269  , 527.1848 ],
#        [547.91754, 600.8241 ],
#        [571.4222 , 648.05035],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [527.2989 , 624.45435],
#        [515.5324 , 624.5056 ],
#        [533.1721 , 706.8356 ],
#        [509.74615, 786.2225 ],
#        [533.2688 , 621.5739 ],
#        [571.5277 , 692.00934],
#        [524.3587 , 733.3219 ],
#        [559.7657 , 480.3661 ],
#        [  0.     ,   0.     ],
#        [544.99097, 762.8389 ],
#        [550.8518 , 792.32434]], dtype=float32)), ([404.3414306640625, 469.085205078125, 234], array([[  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [539.0796 , 624.52435], a
#        [530.33777, 627.42993], a
#        [530.3992 , 709.7666 ],
#        [512.69434, 786.3547 ], 
#        [547.94836, 624.3864 ], a
#        [577.3624 , 683.2916 ],
#        [530.3204 , 739.14954],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [550.87714, 765.7223 ],
#        [547.99036, 792.2822 ]], dtype=float32)), ([503.8452962239583, 544.0332845052084, 35], array([[486.25165, 480.35995],
#        [509.77188, 506.62677],
#        [486.21954, 503.79498],
#        [480.35934, 544.95215],
#        [474.41556, 580.28595],
#        [530.3611 , 506.7366 ],
#        [527.4048 , 550.89606],
#        [500.8338 , 577.3993 ],
#        [503.68625, 583.2812 ],
#        [483.33463, 580.4357 ],
#        [483.13647, 650.9757 ],
#        [483.29984, 718.5863 ],
#        [509.69836, 583.32446],
#        [509.8062 , 648.03735],
#        [  0.     ,   0.     ],
#        [  0.     ,   0.     ],
#        [509.72275, 480.12976],
#        [  0.     ,   0.     ],
#        [480.27658, 736.27216]], dtype=float32))]