import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def plot_pandora_angles(euler=None, quaternion=None, x=1, y=1, z=1):
    # Invert the pitch sign because pandora dataset has opposite sign pitch
    euler = [euler[0], euler[1], -euler[2]]
    # Put W at the end, swap Y & Z and invert the pitch sign
    quaternion = [-quaternion[1], quaternion[3], quaternion[2], quaternion[0]]
    # Create a 3D axis
    fig = plt.figure(figsize=(10, 5))
    
    if euler is not None:
        # Convert Euler angles to rotation matrix
        rotation_matrix = R.from_euler('yzx', euler, degrees=True).as_matrix()
                
        # Draw standard axes
        ax1 = fig.add_subplot(121, projection='3d')
        
        ax1.quiver(0, 0, 0, x, 0, 0, color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax1.text(x*1.2, 0, 0, 'X', color='gray')

        ax1.quiver(0, 0, 0, 0, z, 0, color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax1.text(0, z*1.2, 0, 'Z', color='gray')
        
        ax1.quiver(0, 0, 0, 0, 0, y, color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax1.text(0, 0, y*1.2, 'Y', color='gray')

        # Apply rotation and draw the rotated axes
        new_x, new_y, new_z = np.dot(rotation_matrix, np.array([x, 0, 0]))
        ax1.quiver(0, 0, 0, new_x, new_y, new_z, color='red', arrow_length_ratio=0.2, linewidth=2)
        ax1.text(new_x, new_y, new_z, 'X\'', color='magenta')
        
        new_x, new_y, new_z = np.dot(rotation_matrix, np.array([0, z, 0]))
        ax1.quiver(0, 0, 0, new_x, new_y, new_z, color='red', arrow_length_ratio=0.2, linewidth=2)
        ax1.text(new_x, new_y, new_z, 'Z\'', color='magenta')

        new_x, new_y, new_z = np.dot(rotation_matrix, np.array([0, 0, y]))
        ax1.quiver(0, 0, 0, new_x, new_y, new_z, color='red', arrow_length_ratio=0.2, linewidth=2)
        ax1.text(new_x, new_y, new_z, 'Y\'', color='magenta')

        # Set axis labels
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Z-axis')
        ax1.set_zlabel('Y-axis')
        
        # Set plot limits
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])

        # Use fewer ticks
        ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax1.set_zticks([-1, -0.5, 0, 0.5, 1])

        # Set plot title
        ax1.set_title('Rotated Vector Visualization (Euler Angles)')

    if quaternion is not None:
        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Draw standard axes
        ax2 = fig.add_subplot(122, projection='3d')

        ax2.quiver(0, 0, 0, x, 0, 0, color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax2.text(x*1.2, 0, 0, 'X', color='gray')

        ax2.quiver(0, 0, 0, 0, z, 0, color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax2.text(0, z*1.2, 0, 'Z', color='gray')
        
        ax2.quiver(0, 0, 0, 0, 0, y, color='blue', arrow_length_ratio=0.2, linewidth=2)
        ax2.text(0, 0, y*1.2, 'Y', color='gray')
        
        # Apply rotation and draw the rotated axes
        new_x, new_y, new_z = np.dot(rotation_matrix, np.array([x, 0, 0]))
        ax2.quiver(0, 0, 0, new_x, new_y, new_z, color='red', arrow_length_ratio=0.2, linewidth=2)
        ax2.text(new_x, new_y, new_z, 'X\'', color='magenta')
        
        new_x, new_y, new_z = np.dot(rotation_matrix, np.array([0, z, 0]))
        ax2.quiver(0, 0, 0, new_x, new_y, new_z, color='red', arrow_length_ratio=0.2, linewidth=2)
        ax2.text(new_x, new_y, new_z, 'Z\'', color='magenta')
        
        new_x, new_y, new_z = np.dot(rotation_matrix, np.array([0, 0, y]))
        ax2.quiver(0, 0, 0, new_x, new_y, new_z, color='red', arrow_length_ratio=0.2, linewidth=2)
        ax2.text(new_x, new_y, new_z, 'Y\'', color='magenta')

        # Set axis labels
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Z-axis')
        ax2.set_zlabel('Y-axis')

        # Set plot limits
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([-1, 1])

        # Use fewer ticks
        ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax2.set_zticks([-1, -0.5, 0, 0.5, 1])

        # Set plot title
        ax2.set_title('Rotated Vector Visualization (Quaternion)')

    # Add a legend
    plt.legend()
    plt.show()


# 145 - base_1_ID01
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [0.16885199294634762, -38.901939147748266, -8.61127466215593]
quaternion = [0.9401023829240907, -0.07128719869237338, -0.33219179251676817, -0.02362244074757497]

# 207 - base_1_ID01
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [-31.05436963929531, -0.04537573685532269, -1.6690771048820752]
quaternion = [0.963249671917894, -0.013929549617247461, 0.0035192794323212484, -0.26771521708173085]

# 321 - base_1_ID01
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [-9.91589607558282, 20.191172806841692, -43.54969529959873]
quaternion = [0.9050935501262832, -0.3779490074736962, 0.19375922635609213, -0.014219627728339346]

# 372 - base_1_ID01
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [0.9885452681474626, 30.089098209786805, 7.846554866048627]
quaternion = [0.9630806431701239, 0.0683178678861793, 0.25956084779953703, -0.009455476959619735]

# 685 - base_1_ID01
# # roll, yaw, pitch
# # # W, X, Y, Z
euler = [4.619336695043608, -1.1895336558535705, 32.39130957991377]
quaternion = [0.959372141239612, 0.27832300314190794, 0.0012843465003723187, 0.041600329984478546]

# 800 - base_1_ID01
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [6.771488799022079, 5.313058163779811, 0.6214404275078036]
quaternion = [0.9970259803407606, 0.008146707180809037, 0.04659284964258087, 0.05875006544711875]

# 2051 - free_1_ID22
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [-1.723429600773091, -11.758134623439197, -42.007001877639205]
quaternion = [0.9290264643811644, -0.35507612439049924, -0.0902538638880675, -0.05068153238759372]

# 864 - base_1_ID02
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [-33.72143486976619, 33.556705738840364, -29.63922222023251]
quaternion = [0.864238527739204, -0.3153457549368242, 0.3381421388725576, -0.19779970638337324]

# 864 - free_1_ID22
# # roll, yaw, pitch
# # W, X, Y, Z
euler = [16.988115247645148, 75.2513394122382, 13.648409572910301]
quaternion = [0.7668192307075841, 0.18265189706296137, 0.6134552587370061, 0.044388428018857506]

plot_pandora_angles(euler, quaternion)