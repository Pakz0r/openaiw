import torch
import torch.nn as nn
import numpy as np

def parse_data(file_path="alignment_labels.txt"):
    # Initialize lists to store the points
    groups = []

    # Open and read the file
    with open(file_path, "r") as file:
        group = {"rgb_points": [], "depth_points": []}

        for line in file:
            parts = line.split()
            if len(parts) == 3:
                group["rgb_points"].append([int(parts[1]), int(parts[2])])
            elif len(parts) == 4:
                group["depth_points"].append([int(parts[1]), int(parts[2]), int(parts[3])])
            elif len(parts) == 0:
                if group["rgb_points"] and group["depth_points"]:
                    groups.append(group)
                    group = {"rgb_points": [], "depth_points": []}
    return groups

# Define a neural network model
class ImageAlignmentModel(nn.Module):
    def __init__(self, num_keypoints=8):
        super(ImageAlignmentModel, self).__init__()
        # Variables
        self.num_keypoints = num_keypoints

        depth_camera_matrix = np.array([
            [1/388.198, 0, -253.270],
            [0, 1/389.033, -213.934],
            [0, 0, 1]])
        depth_intrinsic_matrix = torch.tensor(depth_camera_matrix, dtype=torch.float)
        self.inverse_depth_intrinsic_transformation = nn.Linear(3, 3, bias=False)
        self.inverse_depth_intrinsic_transformation.weight = nn.Parameter(depth_intrinsic_matrix)

        R = np.array([
            [0.99997, 0.00715, -0.00105],
            [-0.00715, 0.99995,  0.00662],
            [0.00110, -0.00661, 0.99998]])
        T = np.array([0.06015, -0.00221, -0.02714])
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        extrinsic_matrix = torch.tensor(Rt, dtype=torch.float)
        self.extrinsic_transformation = nn.Linear(3, 3, bias=True)
        self.extrinsic_transformation.weight = nn.Parameter(extrinsic_matrix[:3, :3])
        self.extrinsic_transformation.bias = nn.Parameter(extrinsic_matrix[:3, 3])

        rgb_camera_matrix = np.array([
            [1144.361, 0, 966.359],
            [0, 1147.337, 548.038],
            [0, 0, 1]])
        rgb_intrinsic_matrix = torch.tensor(rgb_camera_matrix, dtype=torch.float)
        self.rgb_intrinsic_transformation = nn.Linear(3, 3, bias=False)
        self.rgb_intrinsic_transformation.weight = nn.Parameter(rgb_intrinsic_matrix)


    def forward(self, depth_points):
        print(self.inverse_depth_intrinsic_transformation.weight)
        
        output_points = torch.nn.Parameter(torch.zeros(self.num_keypoints, 2), requires_grad = True)
        # For each of the points apply the following steps:
        for i in range(self.num_keypoints):
            # Multiply x and y by depth
            # depth_points[i][0] *= depth_points[i][2]
            # depth_points[i][1] *= depth_points[i][2]
            # Create the input tensor
            point = torch.nn.Parameter(torch.tensor([depth_points[i][0], depth_points[i][1], depth_points[i][2]]), requires_grad = True)
            # Apply inverse intrinsic depth matrix multiplication
            point = self.inverse_depth_intrinsic_transformation(point)
            point = torch.relu(point)
            # Apply extrinsic matrix multiplication
            point = self.extrinsic_transformation(point)
            point = torch.relu(point)
            # Apply intrinsic rgb matrix multiplication
            point = self.rgb_intrinsic_transformation(point)
            point = torch.relu(point)
            # Divide x and y by z and round to the nearest integer
            output_points[i][0] = torch.round(point[0]/point[2])
            output_points[i][1] = torch.round(point[1]/point[2])
        
        return output_points


data = parse_data()
# Instantiate the model
model = ImageAlignmentModel()

print(list(model.parameters()))
input()
# Training parameters
# Use MAE loss
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 100

# Training loop (You will need a dataset with labeled images and keypoints)
for epoch in range(num_epochs):
    for sample in data:

        # Forward pass
        depth_points = torch.tensor(sample["depth_points"], dtype=torch.float32, requires_grad = True)
        rgb_points = torch.tensor(sample["rgb_points"], dtype=torch.float32)
        predicted_keypoints = model(depth_points)
        print(predicted_keypoints)
        input()
        
        # Compute the loss
        loss = criterion(predicted_keypoints, rgb_points)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
