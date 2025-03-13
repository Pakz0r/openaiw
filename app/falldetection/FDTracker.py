import math
import numpy as np
from .utils import postprocess, preprocess

class FDTracker:
    def __init__(self, counter_threshold=10):
        # SHAPE OF SKELETON, EACH ROW IS COMPOSED BY ITS CENTER OF MASS IN X AND Y AND THE ARRAY OF KEYPOINTS COORINATES
        #[[x,y,r],[keypoints]],    0 skeleton i-esimo frame   x
        #[[x,y,r],[keypoints]],    1 skeleton i-esimo frame   y
        #[[x,y,r],[keypoints]]     2 skeleton i-esimo frame   z

        self.previous_skeletons = []    
        self.current_skeletons = []
        self.counter_threshold = counter_threshold
        self.missing_frames_counters = []
        self.windows = []
        self.global_counter = 0 # contatore che conta il numero di frame processati in totale dal tracker

    def _track(self):
        '''
        This function keeps the track of each individual through many steps:
            1. Calculating the distance of centers of mass between each person in the new frame and each person in the previous frame.
            2. Based on this distances matrix each "new person" is correlated to the corresponding "old person".
            3. The new detected skeletons are added to the people list, while the ones that are disappeard for more than 10 frames are deleted.

        INPUT: no inputs, but using the class attributes such as
            - previous_skeletons: list of the skeletons detected in the previous frame
            - current_skeletons: list of the skeletons detected in the current frame that need to be correlated to the previous ones
            - counter_threshold: number of frames above which if the skeleton is not detected, is deleted
            - missing_frames_counter: list of missing frames counter, the i-th element is number of frames where the i-th skeleton is not detected
            - windows: list of windows of 75 frames for each skeleton

        OUTPUT: 
            - no outputs. At each iteration the function manage_windows takes in input the windows and particular actions are performed.
        '''

        # calculate distances
        distances = np.empty([len(self.previous_skeletons), len(self.current_skeletons)], dtype=np.float32)
        for i, cs in enumerate(self.current_skeletons):
            for j, ps in enumerate(self.previous_skeletons):
                # calculating euclidean distance of center of mass of each skeleton of the previous frame from each skeleton of current frame
                distances[j][i] = np.sqrt(math.pow(cs[0][0]-ps[0][0], 2) + math.pow(cs[0][1]-ps[0][1], 2))

        #print(f"distances:{distances}")
        print(f"previous_skeletons:{len(self.previous_skeletons)}")
        print(f"missing_frames_counters:{self.missing_frames_counters}")
        print(f"number of windows:{len(self.windows)}")
        #input()
        #for window in self.windows:
        #    print(f"-- window size:{len(window)}")
        
        assigned_skeletons = [] #[2,1,0] -> [C,B,A]
        # if there are no colums with inf values, the corresponding column indices needs to be added to the assigneds_keletons 
        # because this list indicates the new skeletons detected in the scene
        # se il minimo supera il threshold della distanza, ricopia tale e quale l'indice??????????
        
        if distances.size != 0:
            ''' CASE 0:
            if there are new skeletons in the scene
            '''
            # scanning over rows of the distances
            for row_index in range(distances.shape[0]):     # cycle in rows. Each row is associated to a previous skeleton
                row = distances[row_index,:]             
                min_index = np.argmin(row)                  # return the index of the min element (hence, the min distance)
                if row[min_index] > self.previous_skeletons[row_index][0][2]:
                    assigned_skeletons.append(None)         # people no more present in the current frame
                    self.missing_frames_counters[row_index] = self.missing_frames_counters[row_index] + 1
                else:                                       # matching between frames found
                    distances[:,min_index] = math.inf       # setting all the column to inf, thus the column index will not be choosen again as a minimum
                    assigned_skeletons.append(min_index)    # keeping track of already assigned minumins
                    self.missing_frames_counters[row_index] = 0
            
            # are there any new skeletons? we cycle over columns and the indeces of non-inf columns correspond to new skeletons
            
            new_found_skeletons = [ind for ind, x in enumerate(distances[0,:]) if x < math.inf]
            
            # the value v at the i-th position of the list "assigned_skeletons" indicates that the skeleton ate the v-th position in "current_skeletons"
            # is going to override the skeleton at the i-th position in "previous_skeletons".
            # openpose returns the skeletons in random value, the skeletons sorting belongs to our algorithm
            for ind, val in enumerate(assigned_skeletons): 
                if(val != None): 
                    self.previous_skeletons[ind] = self.current_skeletons[val]

            # For each skeleton found in the current frame, we add a counter for its missing frames, its place
            # into the windows structure
            for new_found_skeleton in new_found_skeletons: # es. [3,5,6]
                self.previous_skeletons.append(self.current_skeletons[new_found_skeleton])
                self.windows.append([self.current_skeletons[new_found_skeleton][1].tolist()])
                self.missing_frames_counters.append(0)

            # checking if people are still present in the scene after counter_threshold attempts, if not those get deleted
            missing_skeletons_indices = [idx for idx, x in enumerate(self.missing_frames_counters) if x>self.counter_threshold]
            
            # delete
            if len(missing_skeletons_indices) != 0:       # check if the data structures are empty, to prevent crashing from np.delete
                self._delete_skeletons(missing_skeletons_indices)
           
            # Finally add each tracked skeleton into its window
            for i, skeleton in enumerate(self.previous_skeletons):
                self.windows[i].append(skeleton[1])

        elif len(self.previous_skeletons)==0 and len(self.current_skeletons)!=0:
            ''' CASE 1:
            if there are no previous_skeletons, meaning that it's the first iteration or a middle one corresponding to an empty scene
            initialize previous_skeletons with the current ones
            '''
            self.previous_skeletons = self.current_skeletons
            for _,cs in enumerate(self.current_skeletons):
                self.missing_frames_counters.append(0)  
                self.windows.append([cs[1]])

        elif len(self.current_skeletons)==0 and len(self.previous_skeletons)!=0:
            ''' CASE 2:
            if there are no current skeletons, just increase the missing frames counter for all the previous skeletons
            '''
            for idx, m in enumerate(self.missing_frames_counters):
                self.missing_frames_counters[idx] = m+1


    def _delete_skeletons(self,missing_skeletons_indices):
        '''
        This function deletes people skeletons from the data structures used for the tracking.

        INPUT: 
            - list of indices of people no longer tracked, that needs to be deleted from the tracking list
        
        OUTPUT:
            - void; the data structures are attributes of the class
        '''

        # Filtra missing_frames_counters mantenendo solo gli elementi non presenti in missing_skeletons_indices
        self.missing_frames_counters = [value for i, value in enumerate(self.missing_frames_counters) if i not in missing_skeletons_indices]
        # Filtra previous_skeletons mantenendo solo gli elementi non presenti in missing_skeletons_indices
        self.previous_skeletons = [skeleton for i, skeleton in enumerate(self.previous_skeletons) if i not in missing_skeletons_indices]
        # Filtra windows mantenendo solo gli elementi non presenti in missing_skeletons_indices
        self.windows = [window for i, window in enumerate(self.windows) if i not in missing_skeletons_indices]
        

    def _manage_windows(self):
        '''
        This function creates the data structure that contains a window of 75 frames for each person tracked.
        Each window is post-processed (normalization and quantization) before being putted in the structure

        INPUT: 
            None
        
        OUTPUT:
            - ready_windows: each i-th element of this structure is a list of 75 frames of the i-th person tracked
        '''
        ready_windows = []
        for i, w in enumerate(self.windows):
            if(len(w)==25):
                ready_windows.append(postprocess(w))
                del self.windows[i][:15]
        return ready_windows
        
    def run(self, keypoints):
        self.current_skeletons = preprocess(keypoints)
        self._track()
        return self._manage_windows()
