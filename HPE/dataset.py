import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
from facenet_pytorch import MTCNN
from PIL import Image
from scipy.spatial.transform import Rotation as R
import json
from scipy.spatial.transform import Rotation as R
import scipy
import numpy as np
import utils


class PandoraDataset(Dataset):
    def __init__(self, cached=True, write_faces=False, calculate_mean_std=False):
        print("PandoraDataset")
        self.samples = []

        if not cached:
            if write_faces:
                self.write_faces()
            if calculate_mean_std:
                self.calculate_mean_std()
            self.cache_dataset()

        print("Loading cached dataset...")
        for sample in tqdm(os.listdir("serialized_dataset/pandora/faces")):
            self.samples.append(sample)
               
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face = torch.load("serialized_dataset/pandora/faces/" + self.samples[idx])
        rotation_matrix = torch.load("serialized_dataset/pandora/labels/rotation_matrix/" + self.samples[idx])
        class_label = torch.load("serialized_dataset/pandora/labels/classes/" + self.samples[idx])
        return face, rotation_matrix, class_label

    def write_faces(self, margin=50):
        print("Writing faces to disk...")
        for folder in tqdm(os.listdir("dataset/pandora")):
            for subfolder in os.listdir("dataset/pandora/" + folder):
                for file in os.listdir("dataset/pandora/" + folder + "/" + subfolder + "/RGB"):
                    # Create folder for faces
                    if not os.path.exists("dataset/pandora/" + folder + "/" + subfolder + "/faces"):
                        os.makedirs("dataset/pandora/" + folder + "/" + subfolder + "/faces")
                    # Load image
                    image = Image.open("dataset/pandora/" + folder + "/" + subfolder + "/RGB/" + file).convert('RGB')
                    # Detect face
                    mtcnn = MTCNN(select_largest=True, post_process=False, device='cuda:0')
                    boxes, _ = mtcnn.detect(image)

                    if boxes is None:
                        continue

                    # Add margin while safe checking
                    boxes[0][0] = max(0, boxes[0][0] - margin)
                    boxes[0][1] = max(0, boxes[0][1] - margin)
                    boxes[0][2] = min(image.width, boxes[0][2] + margin)
                    boxes[0][3] = min(image.height, boxes[0][3] + margin)

                    # Crop the face using boxes
                    face = image.crop(boxes[0])

                    # Save the face to disk
                    face.save("dataset/pandora/" + folder + "/" + subfolder + "/faces/" + file )
        
    def calculate_mean_std(self):
        print("Calculating mean and standard deviation of the dataset...")
        mean = torch.zeros(200, 200, 3, dtype=torch.float64)
        std = torch.zeros(200, 200, 3, dtype=torch.float64)
        N = 0
        # Calculate mean
        for folder in tqdm(os.listdir("dataset/pandora")):
            for subfolder in os.listdir("dataset/pandora/" + folder):
                for file in os.listdir("dataset/pandora/" + folder + "/" + subfolder + "/faces"):
                    file_path = "dataset/pandora/" + folder + "/" + subfolder + "/faces/" + file
                    image = Image.open(file_path).convert('RGB')
                    # Convert to tensor
                    transform = transforms.Compose([ 
                        transforms.PILToTensor(),
                        transforms.Resize((200, 200)),
                    ])
                    tensor = transform(image)
                    # Swap axes to 200x200x3
                    tensor = tensor.permute(1, 2, 0)
                    # Add to mean
                    mean += tensor
                    N += 1
        mean /= N

        # Calculate standard deviation
        for folder in tqdm(os.listdir("dataset/pandora")):
            for subfolder in os.listdir("dataset/pandora/" + folder):
                for file in os.listdir("dataset/pandora/" + folder + "/" + subfolder + "/faces"):
                    file_path = "dataset/pandora/" + folder + "/" + subfolder + "/faces/" + file
                    image = Image.open(file_path).convert('RGB')
                    # Convert to tensor
                    transform = transforms.Compose([ 
                        transforms.PILToTensor(),
                        transforms.Resize((200, 200)),
                    ])
                    tensor = transform(image)
                    # Swap axes to 200x200x3
                    tensor = tensor.permute(1, 2, 0)
                    # Add to mean
                    std += (tensor - mean)**2
        std /= N
        std = torch.sqrt(std)
        
        print("N: ", N)
        print("Mean: ", mean)
        print("Std: ", std)

        # Save mean and std to disk
        torch.save(mean, "serialized_dataset/pandora/mean.pt")
        torch.save(std, "serialized_dataset/pandora/std.pt")

    def assign_class_label(self, euler_angles):
        discrete_classes = list(range(-175,185,10))

        pitch = min(discrete_classes, key=lambda x: abs(x - euler_angles[0][0]))
        yaw = min(discrete_classes, key=lambda x: abs(x - euler_angles[1][0]))
        roll = min(discrete_classes, key=lambda x: abs(x - euler_angles[2][0]))

        label = str(pitch) + "," + str(yaw) + "," + str(roll)
        classes = []
        with open("serialized_dataset/classes.json", 'r') as json_file:
            classes = json.load(json_file)

        for i, value in enumerate(classes):
            if label == value:
                return i
        
        return None    
        
    def cache_dataset(self, starting_sample = 0):
        print("Caching dataset to disk...")

        serialized_cont = starting_sample

        # Load mean and std
        mean = torch.load("serialized_dataset/pandora/mean.pt")
        std = torch.load("serialized_dataset/pandora/std.pt")

        # Define a tqdm bar
        pbar = tqdm(os.listdir("dataset/pandora"))
        for folder in pbar:
            for subfolder in os.listdir("dataset/pandora/" + folder):
                with open("dataset/pandora/" + folder + "/" + subfolder + "/data.json") as f:
                    json_file = json.load(f)
                    for row in json_file:
                        frame_num = row["frame_num"]
                        euler = row["orientation"]["euler"]

                        image_name = "000000"
                        image_name = image_name[:-len(str(frame_num))] + str(frame_num) + ".png"
                        image_path = "dataset/pandora/" + folder + "/" + subfolder + "/faces/" + image_name

                        # If the image does not exist, skip it
                        if not os.path.exists(image_path):
                            continue

                        # Load the image
                        face = Image.open(image_path).convert('RGB')

                        # Preprocess the image
                        transform = transforms.Compose([ 
                            transforms.PILToTensor(),
                            transforms.Resize((200, 200)),
                        ])
                        face_tensor = transform(face)
                        face_tensor = face_tensor.permute(1, 2, 0)

                        # Standardize the tensor
                        face_tensor = (face_tensor - mean) / std
                        face_tensor = face_tensor.permute(2, 0, 1)
                        face_tensor = face_tensor.type(torch.float32)

                        corrected_euler = [euler["pitch"], euler["roll"], -euler["yaw"]]
                        # WARNING - the dataset labeling is wrong: In reality the pitch is actually the roll, the roll is the yaw and the yaw is the pitch
                        # so corrected_euler = [euler["roll"], euler["yaw"], -euler["pitch"]]
                        euler_tensor = torch.tensor(corrected_euler, dtype=torch.float32)
                        euler_tensor = euler_tensor.unsqueeze(-1)

                        
                        rotation_matrix = R.from_euler('yzx', corrected_euler, degrees=True).as_matrix()

                        # Rewrite euler tensor to disk
                        new_euler_tensor = R.from_matrix(rotation_matrix).as_euler('xzy', degrees=True)
                        new_euler_tensor = torch.tensor(new_euler_tensor, dtype=torch.float32)
                        new_euler_tensor = new_euler_tensor.unsqueeze(-1)

                        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
                        
                        classes = self.assign_class_label(new_euler_tensor.tolist())
                        if classes == None:
                            print("Error in class labels for " + image_path)
                            return

                        # Save the tensor
                        torch.save(face_tensor, "serialized_dataset/pandora/faces/" + str(serialized_cont) + ".pt")
                        torch.save(new_euler_tensor, "serialized_dataset/pandora/labels/euler/" + str(serialized_cont) + ".pt")
                        torch.save(rotation_matrix, "serialized_dataset/pandora/labels/rotation_matrix/" + str(serialized_cont) + ".pt")
                        torch.save(classes, "serialized_dataset/pandora/labels/classes/" + str(serialized_cont) + ".pt")

                        serialized_cont += 1
                        pbar.set_description("Cached: %d" % serialized_cont)


class AFLW2000Dataset(Dataset):
    def __init__(self, cached=True, write_faces=False, calculate_mean_std=False):
        print("AFLW2000Dataset")
        self.samples = []

        if not cached:
            if write_faces:
                self.write_faces()
            if calculate_mean_std:
                self.calculate_mean_std()
            self.cache_dataset()

        print("Loading cached dataset...")
        for sample in tqdm(os.listdir("serialized_dataset/AFLW2000/faces")):
            self.samples.append(sample)
               
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face = torch.load("serialized_dataset/AFLW2000/faces/" + self.samples[idx])
        rotation_matrix = torch.load("serialized_dataset/AFLW2000/labels/rotation_matrix/" + self.samples[idx])
        class_label = torch.load("serialized_dataset/AFLW2000/labels/classes/" + self.samples[idx])
        return face, rotation_matrix, class_label

    def write_faces(self, margin=50):
        print("Writing faces to disk...")
        dir_path = "dataset/AFLW2000/"
        # Create folder for faces
        if not os.path.exists(dir_path + "faces"):
            os.makedirs(dir_path + "faces")
        for file in tqdm(os.listdir(dir_path)):
            if file.endswith(".jpg"):
                # Load image
                image = Image.open(dir_path + file).convert('RGB')
                # Detect face
                mtcnn = MTCNN(select_largest=True, post_process=False, device='cuda:0')
                boxes, _ = mtcnn.detect(image)

                if boxes is None:
                    continue

                # Add margin while safe checking
                boxes[0][0] = max(0, boxes[0][0] - margin)
                boxes[0][1] = max(0, boxes[0][1] - margin)
                boxes[0][2] = min(image.width, boxes[0][2] + margin)
                boxes[0][3] = min(image.height, boxes[0][3] + margin)

                # Crop the face using boxes
                face = image.crop(boxes[0])

                # Save the face to disk
                face.save(dir_path + "faces/" + file)
        
    def calculate_mean_std(self):
        print("Calculating mean and standard deviation of the dataset...")
        mean = torch.zeros(200, 200, 3, dtype=torch.float64)
        std = torch.zeros(200, 200, 3, dtype=torch.float64)
        N = 0
        # Calculate mean
        for folder in tqdm(os.listdir("dataset/pandora")):
            for subfolder in os.listdir("dataset/pandora/" + folder):
                for file in os.listdir("dataset/pandora/" + folder + "/" + subfolder + "/faces"):
                    file_path = "dataset/pandora/" + folder + "/" + subfolder + "/faces/" + file
                    image = Image.open(file_path).convert('RGB')
                    # Convert to tensor
                    transform = transforms.Compose([ 
                        transforms.PILToTensor(),
                        transforms.Resize((200, 200)),
                    ])
                    tensor = transform(image)
                    # Swap axes to 200x200x3
                    tensor = tensor.permute(1, 2, 0)
                    # Add to mean
                    mean += tensor
                    N += 1
        mean /= N

        # Calculate standard deviation
        for folder in tqdm(os.listdir("dataset/pandora")):
            for subfolder in os.listdir("dataset/pandora/" + folder):
                for file in os.listdir("dataset/pandora/" + folder + "/" + subfolder + "/faces"):
                    file_path = "dataset/pandora/" + folder + "/" + subfolder + "/faces/" + file
                    image = Image.open(file_path).convert('RGB')
                    # Convert to tensor
                    transform = transforms.Compose([ 
                        transforms.PILToTensor(),
                        transforms.Resize((200, 200)),
                    ])
                    tensor = transform(image)
                    # Swap axes to 200x200x3
                    tensor = tensor.permute(1, 2, 0)
                    # Add to mean
                    std += (tensor - mean)**2
        std /= N
        std = torch.sqrt(std)
        
        print("N: ", N)
        print("Mean: ", mean)
        print("Std: ", std)

        # Save mean and std to disk
        torch.save(mean, "serialized_dataset/pandora/mean.pt")
        torch.save(std, "serialized_dataset/pandora/std.pt")

    def assign_class_label(self, euler_angles):
        discrete_classes = list(range(-175,185,10))

        pitch = min(discrete_classes, key=lambda x: abs(x - euler_angles[0][0]))
        yaw = min(discrete_classes, key=lambda x: abs(x - euler_angles[1][0]))
        roll = min(discrete_classes, key=lambda x: abs(x - euler_angles[2][0]))

        label = str(pitch) + "," + str(yaw) + "," + str(roll)
        classes = []
        with open("serialized_dataset/classes.json", 'r') as json_file:
            classes = json.load(json_file)

        for i, value in enumerate(classes):
            if label == value:
                return i
        
        return None

    def cache_dataset(self, starting_sample = 0):
        print("Caching dataset to disk...")

        serialized_cont = starting_sample

        # Load mean and std
        mean = torch.load("serialized_dataset/pandora/mean.pt")
        std = torch.load("serialized_dataset/pandora/std.pt")

        dir_path = "dataset/AFLW2000/"
        pbar = tqdm(os.listdir(dir_path))
        for file in pbar:
            if file.endswith(".mat"):
                mat = scipy.io.loadmat(dir_path + file)

                pitch, yaw, roll = mat['Pose_Para'][0][:3]
                yaw = -yaw

                pitch, yaw, roll = R.from_euler('xzy', [pitch, yaw, -roll], degrees=False).as_euler('xzy', degrees=True)
                rotation_matrix = R.from_euler('yzx', [roll, yaw, pitch], degrees=True).as_matrix()
                new_euler_tensor = R.from_matrix(rotation_matrix).as_euler('xzy', degrees=True)
                new_euler_tensor = torch.tensor(new_euler_tensor, dtype=torch.float32)
                new_euler_tensor = new_euler_tensor.unsqueeze(-1)

                rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)

                image_name = file.replace(".mat", ".jpg")
                image_path = dir_path + "faces/" + image_name

                # If the image does not exist, skip it
                if not os.path.exists(image_path):
                    continue

                # Load the image
                face = Image.open(image_path).convert('RGB')

                # Preprocess the image
                transform = transforms.Compose([ 
                    transforms.PILToTensor(),
                    transforms.Resize((200, 200)),
                ])
                face_tensor = transform(face)
                face_tensor = face_tensor.permute(1, 2, 0)

                # Standardize the tensor
                face_tensor = (face_tensor - mean) / std
                face_tensor = face_tensor.permute(2, 0, 1)
                face_tensor = face_tensor.type(torch.float32)

                classes = self.assign_class_label(new_euler_tensor.tolist())
                if classes == None:
                    print("Error in class labels for " + image_path)
                    return

                # Save the tensor
                torch.save(face_tensor, "serialized_dataset/AFLW2000/faces/" + str(serialized_cont) + ".pt")
                torch.save(rotation_matrix, "serialized_dataset/AFLW2000/labels/rotation_matrix/" + str(serialized_cont) + ".pt")
                torch.save(new_euler_tensor, "serialized_dataset/AFLW2000/labels/euler/" + str(serialized_cont) + ".pt")
                torch.save(classes, "serialized_dataset/AFLW2000/labels/classes/" + str(serialized_cont) + ".pt")

                serialized_cont += 1
                pbar.set_description("Cached: %d" % serialized_cont)


class AIWatchDataset(Dataset):
    def __init__(self, cached=True, write_faces=False, calculate_mean_std=False):
        print("AIWatchDataset")
        self.samples = []

        if not cached:
            if write_faces:
                self.write_faces()
            if calculate_mean_std:
                self.calculate_mean_std()
            self.cache_dataset()
        
        print("Loading cached dataset...")
        for sample in tqdm(os.listdir("serialized_dataset/AIWatch/faces")):
            self.samples.append(sample)
               
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face = torch.load("serialized_dataset/AIWatch/faces/" + self.samples[idx])
        rotation_matrix = torch.load("serialized_dataset/AIWatch/labels/rotation_matrix/" + self.samples[idx])
        class_label = torch.load("serialized_dataset/AIWatch/labels/classes/" + self.samples[idx])
        return face, rotation_matrix, class_label

    def write_faces(self, margin=50):
        print("Writing faces to disk...")
        dir_path = "dataset/AIWatch/Color_filtered/"
        # Create folder for faces
        if not os.path.exists(dir_path + "faces"):
            os.makedirs(dir_path + "faces")
        for file in tqdm(os.listdir(dir_path)):
            if file.endswith(".png"):
                # Load image
                image = Image.open(dir_path + file).convert('RGB')
                # Detect face
                mtcnn = MTCNN(select_largest=True, post_process=False, device='cuda:0')
                boxes, _ = mtcnn.detect(image)

                if boxes is None:
                    continue

                # Add margin while safe checking
                boxes[0][0] = max(0, boxes[0][0] - margin)
                boxes[0][1] = max(0, boxes[0][1] - margin)
                boxes[0][2] = min(image.width, boxes[0][2] + margin)
                boxes[0][3] = min(image.height, boxes[0][3] + margin)

                # Crop the face using boxes
                face = image.crop(boxes[0])

                # Save the face to disk
                face.save(dir_path + "faces/" + file)
        
    def calculate_mean_std(self):
        print("Calculating mean and standard deviation of the dataset...")
        mean = torch.zeros(200, 200, 3, dtype=torch.float64)
        std = torch.zeros(200, 200, 3, dtype=torch.float64)
        N = 0
        # Calculate mean
        for folder in tqdm(os.listdir("dataset/pandora")):
            for subfolder in os.listdir("dataset/pandora/" + folder):
                for file in os.listdir("dataset/pandora/" + folder + "/" + subfolder + "/faces"):
                    file_path = "dataset/pandora/" + folder + "/" + subfolder + "/faces/" + file
                    image = Image.open(file_path).convert('RGB')
                    # Convert to tensor
                    transform = transforms.Compose([ 
                        transforms.PILToTensor(),
                        transforms.Resize((200, 200)),
                    ])
                    tensor = transform(image)
                    # Swap axes to 200x200x3
                    tensor = tensor.permute(1, 2, 0)
                    # Add to mean
                    mean += tensor
                    N += 1
        mean /= N

        # Calculate standard deviation
        for folder in tqdm(os.listdir("dataset/pandora")):
            for subfolder in os.listdir("dataset/pandora/" + folder):
                for file in os.listdir("dataset/pandora/" + folder + "/" + subfolder + "/faces"):
                    file_path = "dataset/pandora/" + folder + "/" + subfolder + "/faces/" + file
                    image = Image.open(file_path).convert('RGB')
                    # Convert to tensor
                    transform = transforms.Compose([ 
                        transforms.PILToTensor(),
                        transforms.Resize((200, 200)),
                    ])
                    tensor = transform(image)
                    # Swap axes to 200x200x3
                    tensor = tensor.permute(1, 2, 0)
                    # Add to mean
                    std += (tensor - mean)**2
        std /= N
        std = torch.sqrt(std)
        
        print("N: ", N)
        print("Mean: ", mean)
        print("Std: ", std)

        # Save mean and std to disk
        torch.save(mean, "serialized_dataset/pandora/mean.pt")
        torch.save(std, "serialized_dataset/pandora/std.pt")

    def assign_class_label(self, euler_angles):
        discrete_classes = list(range(-175,185,10))

        pitch = min(discrete_classes, key=lambda x: abs(x - euler_angles[0][0]))
        yaw = min(discrete_classes, key=lambda x: abs(x - euler_angles[1][0]))
        roll = min(discrete_classes, key=lambda x: abs(x - euler_angles[2][0]))

        label = str(pitch) + "," + str(yaw) + "," + str(roll)
        classes = []
        with open("serialized_dataset/classes.json", 'r') as json_file:
            classes = json.load(json_file)

        for i, value in enumerate(classes):
            if label == value:
                return i     
        return None

    def cache_dataset(self, starting_sample = 0):
        print("Caching dataset to disk...")

        serialized_cont = starting_sample

        # Load mean and std
        mean = torch.load("serialized_dataset/pandora/mean.pt")
        std = torch.load("serialized_dataset/pandora/std.pt")

        dir_path = "dataset/AIWatch/"
        pbar = tqdm(os.listdir(dir_path + "Angles/"))
        for file in pbar:
            if file.endswith(".txt"):

                rotation_matrix = None
                with open(dir_path + "Angles/" + file) as f:
                    angles = f.read().split(" ")
                    angles = [float(angle) for angle in angles]
                    label = torch.tensor(utils.convert_angles_from_AIWatch(angles))
                    rotation_matrix = R.from_euler('xzy', label, degrees=True).as_matrix()

                new_euler_tensor = R.from_matrix(rotation_matrix).as_euler('xzy', degrees=True)
                new_euler_tensor = torch.tensor(new_euler_tensor, dtype=torch.float32)
                new_euler_tensor = new_euler_tensor.unsqueeze(-1)
                rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)

                image_name = file.replace(".txt", ".png")
                image_path = dir_path + "Color_filtered/faces/" + image_name

                # If the image does not exist, skip it
                if not os.path.exists(image_path):
                    continue

                # Load the image
                face = Image.open(image_path).convert('RGB')

                # Preprocess the image
                transform = transforms.Compose([ 
                    transforms.PILToTensor(),
                    transforms.Resize((200, 200)),
                ])
                face_tensor = transform(face)
                face_tensor = face_tensor.permute(1, 2, 0)

                # Standardize the tensor
                face_tensor = (face_tensor - mean) / std
                face_tensor = face_tensor.permute(2, 0, 1)
                face_tensor = face_tensor.type(torch.float32)

                classes = self.assign_class_label(new_euler_tensor.tolist())
                if classes == None:
                    print("Error in class labels for " + image_path)
                    return

                # Save the tensor
                torch.save(face_tensor, "serialized_dataset/AIWatch/faces/" + str(serialized_cont) + ".pt")
                torch.save(rotation_matrix, "serialized_dataset/AIWatch/labels/rotation_matrix/" + str(serialized_cont) + ".pt")
                torch.save(new_euler_tensor, "serialized_dataset/AIWatch/labels/euler/" + str(serialized_cont) + ".pt")
                torch.save(classes, "serialized_dataset/AIWatch/labels/classes/" + str(serialized_cont) + ".pt")

                serialized_cont += 1
                pbar.set_description("Cached: %d" % serialized_cont)



if "__main__" == __name__:
    dataset = PandoraDataset(cached=False)
    # dataset = AFLW2000Dataset(cached=False)
    # dataset = AIWatchDataset(cached=False)