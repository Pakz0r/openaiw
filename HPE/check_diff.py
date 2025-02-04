import subprocess

IDs = ["01/", "04/", "05/", "06/", "07/", "08/", "09/", "10/", "11/", "12/", "13/", "14/", "15/", "16/", "17/", "18/", "19/", "20/", "21/", "22/"]

for ID in IDs:
    center_list = ["base_1_ID"+ID, "base_2_ID"+ID, "free_1_ID"+ID, "free_2_ID"+ID, "free_3_ID"+ID]

    for center in center_list:
        postfix = ID + center + "data.json"
        file1 = "/home/spoleto/hpe/dataset/pandora/" + postfix
        file2 = "/home/projects/SE4I/AIwatch/dataset/dataset/" + postfix

        diff_output = subprocess.run(["diff", file1, file2], capture_output=True, text=True)
        # if diff_output is not empty, then print the output
        if diff_output.stdout:
            # Copy file2 to file1
            #subprocess.run(["cp", file2, file1])
            print(file1)
            print(file2)
            print("\n")
            input()
