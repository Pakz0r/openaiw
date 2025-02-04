import torch
import torch.nn as nn
import numpy as np
import cv2
import json

def parse_data(file_path="alignment_labels.json"):
    # Load the json file
    points = []
    with open(file_path) as f:
        json_file = json.load(f)
        for key, value in json_file.items():
            # Convert the list of lists to a numpy array
            points.append({"depth": np.array(value["depth"]), "rgb": np.array(value["rgb"])})
    return points

# Define a neural network model
class ImageAlignmentModel(nn.Module):
    def __init__(self, num_keypoints=8):
        super(ImageAlignmentModel, self).__init__()
        # Variables
        self.num_keypoints = num_keypoints
       
        self.fc1 = nn.Linear(2,3, bias=True)
        self.fc2 = nn.Linear(3,3, bias=True)
        self.fc3 = nn.Linear(3,2, bias=True)

      
    def forward(self, depth_points):
        aligned_points = self.fc1(depth_points)
        aligned_points = self.fc2(aligned_points)
        aligned_points = self.fc3(aligned_points)
        return aligned_points


data = parse_data()
model = ImageAlignmentModel()

# Training parameters
#criterion = nn.L1Loss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

num_epochs = 300

for epoch in range(num_epochs):
    for sample in data:

        depth_points = torch.tensor(sample["depth"], dtype=torch.float32, requires_grad = True) # torch.Size([8, 2])
        rgb_points = torch.tensor(sample["rgb"], dtype=torch.float32)
        
        predicted_keypoints = model(depth_points)

        # Compute the loss
        loss = criterion(predicted_keypoints, rgb_points)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss if it's a multiple of 25
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item())) 
  
# Load test_RGB and test_DEPTH images using opencv
rgb_image = cv2.imread("test_RGB.png", cv2.IMREAD_COLOR)
depth_image = cv2.imread("test_DEPTH.png", cv2.IMREAD_GRAYSCALE)

print(depth_image.shape[0], depth_image.shape[1])

depth_points = []
for i in range(depth_image.shape[1]):
    for j in range(depth_image.shape[0]):
        depth_points.append([i/512, j/424, depth_image[j][i]])

# Convert to NumPy array
depth_array = np.array(depth_points, dtype=np.float32)

# convert to tensor
depth_tensor = torch.tensor(depth_array, dtype=torch.float32)

aligned_depth_tensor = model(depth_tensor[:, :2])
aligned_depth_tensor = torch.cat((aligned_depth_tensor, depth_tensor[:,2].view(-1,1)), dim=1)

print(aligned_depth_tensor.shape)
# Convert the aligned_depth_tensor to an image which has the same shape as test_RGB
aligned_depth_image = torch.zeros(rgb_image.shape[0], rgb_image.shape[1], 1)
for i in range(aligned_depth_tensor.shape[0]):
    # If the x,y coordinates are within the bounds of the image, set the pixel value to the z value
    if (int(aligned_depth_tensor[i][0]*1920) < aligned_depth_image.shape[1] and int(aligned_depth_tensor[i][1]*1080) < aligned_depth_image.shape[0]):
        aligned_depth_image[int(aligned_depth_tensor[i][1]*1080)][int(aligned_depth_tensor[i][0]*1920)] = aligned_depth_tensor[i][2]*255

# Write the image to a file
cv2.imwrite("aligned_depth_image.png", aligned_depth_image.detach().numpy())