import os
import shutil

# Bill: I made this to help organize the test train val split

base_path = "/data/sign_language_videos/popsign_v2/313_normalized_hands"
#output_path = "/storage/home/hcoda1/4/wneubauer3/scratch/popsign_v2_mediapipe_split" #The output where the split is contained
output_path = "/data/sign_language_videos/popsign_v2/563_normalized_hands"

#Train set can be any users but the specified ids *must* be in the train set since
#they were in the train set for Popsign v1
train_id = ['1002', '1004', '1005', '1006', '1007', '1008', '1009', '1011', '1013', '1014', '1017', '1020']
test_id = ['1010', '1017'] # Found in PopSign V1
test_id.extend(['1030', '1031', '1032', '1033', '1035', '1036']) # Arbitrarily selected in PopSign V2
val_id = ['1001', '1003', '1012', '1018', '1027'] # Found in PopSign V1
val_id.extend(['1050', '1049', '1048']) #Arbitrarily selected in PopSign V2

print(f"train: {train_id}")
print(f"test: {test_id}")
print(f"val: {val_id}")

for sign in os.listdir(base_path):
    sign_input_path = os.path.join(base_path, sign)

    for mediapipe_file in os.listdir(sign_input_path):

      #Filename parsing
      # filename format: account-sign-timestamp.h5 
      #     example: gtsignstudy.4a.2.1030-a-2023_11_18_13_29_19.173-1.h5
      # id format:
      #     prefix.4a.2.XXXX where XXXX is id
      #     Sometimes may be truncated to 4a.2.XXXX, which is messy data    
      split = ''
      try:
            #account, sign, timestamp = mediapipe_file.split('-')
            account = mediapipe_file.split('-')[0]
      except:
           print(f"Error in parsing: {mediapipe_file}")
           
      user_id = account[-4:] #Grab last 4 characters

      #Categorizing in split
      if (user_id in train_id): #Redudant but must not ref
            split = 'train'
      elif (user_id in test_id):
           split = 'test'
      elif (user_id in val_id):
           split = 'validation'
      else:
           split = 'train'

      sign_output_path = os.path.join(output_path, split, sign)
      if not os.path.exists(sign_output_path):
           os.makedirs(sign_output_path, exist_ok=True)
      
      #Link / copy file
      mediapipe_input_path = os.path.join(sign_input_path, mediapipe_file)
      mediapipe_output_path = os.path.join(sign_output_path, mediapipe_file)
      os.link(mediapipe_input_path, mediapipe_output_path)

      #shutil.copy(sign_input_path, sign_output_path)
    