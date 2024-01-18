
import os
import numpy as np
import pandas as pd
import csv
import ipdb

SYNC_FILE_NAME = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/sync_params.xlsx'
KEY_COLUMN = 'video name'
VALUE_COLUMNS = ['video 2m walk start time (seconds)', 'sensor 2m walk start time (seconds)','FPS']
SYNC_SHEET_NAME = 'Sheet1'

ACC_DATA_DIR = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/acc_data/right_wrist'
LABEL_DATA_DIR = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/labeled data'
TARGET_DIR = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/synced_labeled_data_walking_non_walking'

ACC_SAMPLE_RATE = 100 # Hz
#LABEL_SAMPLE_RATE = 59.94005994005994 # for movies from TC center 60 FPS
missing_labels = []
def main():
    sync_dict = create_dictionary_from_excel(SYNC_FILE_NAME, SYNC_SHEET_NAME, KEY_COLUMN, VALUE_COLUMNS)
    for patient, val  in sync_dict.items():
        sync_sec = val[0:2]
        label_sample_rate = val[2]
        acc_data = read_acc_data(patient=patient)
        label_data, chorea_labels = read_label_data(patient=patient, 
                                     source_sample_rate=label_sample_rate,
                                     target_sample_rate=ACC_SAMPLE_RATE)
        if label_data is None:
            continue
        acc_data_sync, label_data_sync, chorea_labels_sync, time_data_sync = sync_data(acc_data, 
                                                   label_data,
                                                   chorea_labels,
                                                   sync_sec, 
                                                   ACC_SAMPLE_RATE)
        np.savez(os.path.join(TARGET_DIR, patient + '.npz'), acc_data_sync, label_data_sync, chorea_labels_sync, time_data_sync)


def create_dictionary_from_excel(file_path, sheet_name, key_column, value_columns):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

    # Extract the relevant columns from the DataFrame
    key_values = df[key_column].str.split('_').str[0].tolist()
    value_tuples = df[value_columns].apply(tuple, axis=1).tolist()

    # Create the dictionary
    data_dict = dict(zip(key_values, value_tuples))

    return data_dict

def read_acc_data(patient, files_dir=ACC_DATA_DIR):
    file_names = os.listdir(files_dir)  # Get all file names in the folder

    # Find the file name that matches the given name
    matching_file_name = None
    for file_name in file_names:
        if file_name.startswith(patient+'_'):
            matching_file_name = file_name
            break

    if matching_file_name is None:
        print(f"No file found matching the name '{patient}'")
        return None

    file_path = os.path.join(files_dir, matching_file_name)

    # Read the CSV file and extract the required columns
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader((line.replace('\0', '') for line in file))
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            # Check if the row has enough columns
            if len(row) >= 4:
                values = [row[1], row[2], row[3]]
                data.append(values)

    if not data:
        print("No data found in the CSV file.")
        return None

    # Convert the data to a NumPy array
    data_array = np.array(data)

    return data_array

def read_label_data(patient, 
                    source_sample_rate,
                    target_sample_rate=ACC_SAMPLE_RATE,
                    files_dir=LABEL_DATA_DIR):
    '''
    gets patient and return a numpy array with the labels at the sample
    rate of the acc [1Xnum_samples]
    '''
    subfolder_path = None
    timeline_csv_path = None
    # Find the subfolder that starts with the given name
    for root, dirs, files in os.walk(files_dir):
        for dir_name in dirs:
            if dir_name.startswith(patient+'_'):
                subfolder_path = os.path.join(root, dir_name)
                break

        if subfolder_path:
            break

    if subfolder_path is None:
        for file in os.listdir(files_dir):
            if file.startswith(patient+'_'):
                timeline_csv_path = os.path.join(files_dir, file)
                break
    else:
        timeline_csv_path = os.path.join(subfolder_path, "timeline.csv")

    if timeline_csv_path is None:
        print(f'No matching labels file or folder exist for {patient}')
        return None, None

    labels_by_frames = []
    chorea_labels_by_frames = []
    sections_counter = 0
    with open(timeline_csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 0:
                continue
            if sections_counter<3:
                if row[0]=='T':
                    sections_counter+=1
                    continue
            if sections_counter==2:
                first_frame = int(row[2])
                last_frame = int(row[3])
                activity_name = row[4]
                chorea_labels_by_frames.append((first_frame, last_frame, activity_name))
            if sections_counter==3:
                if row[0] == 'T':
                    break
                first_frame = int(row[2])
                last_frame = int(row[3])
                activity_name = row[4]
                labels_by_frames.append((first_frame, last_frame, activity_name))
    if len(labels_by_frames) == 0:
        print(f'patient {patient} has no labels')
        return None, None

    activity__dict = {'walking': 1, 'turning': -9, 'turning ':-9,'stumbling':-9,'stepping to the side': 0, 'standing':0, 'sitting':0, 'siting down':0, 'standing clapping hands':0, 
                      'sitting down':0, 'sitting clapping hands':0,'sit to stand':0,'sitting and writing ':0,
                       'sitting and driniking water':0,'sitting and writing':0,'standing up':0, 'standing and clapping hands':0, 
                       'standing and putting arms crossed on the chest':0, 'standing with arms crossed on the chest':0,
                       'standing and putting the arms down':0, 'standing and putting the hands down':0,'':0,
                       'stepping on a foam':0,'standing and putting the arms crossed on the chest':0,'standing with arms crossed on the chesr':0,
                       'standing and putting hands down':0, 'stepping off the foam':0,'bending over':0, 'sitting and clapping hands':0, 
                       'stepping up and down a step':0, 'sitiing down':0, 'moving hands up':0,
                       'clapping hands': 0, 'moving hands down':0, 'staidning up':0, 'step ups':-9,
                       'stambling':0, 'stepping over a step':0, 'stending up':0, 'stending':0, 
                       'stepping off of step': 0, 'standing off a step':0, 'turning around':0, 
                       'putting the arms crossed on the chest\n':0, 'walking backwards': 0,
                       'putting the arms crossed on the chest':0, 'arms crossed on the chest\n':0, 
                       '-9': -9}
    try:
        last_labeled_frame = labels_by_frames[-1][1]
    except:
        ipdb.set_trace()
    last_labeled_frame_sample = int(np.round(last_labeled_frame*(target_sample_rate/source_sample_rate)))
    labels_array = np.zeros(last_labeled_frame_sample+1).astype(int)
    for label_set in labels_by_frames:
        # if label_set[2] not in missing_labels and label_set[2] not in activity__dict.keys():
        #     print(f'missing {label_set[2]}')
        #     missing_labels.append(label_set[2])
        assert label_set[2] in activity__dict.keys(), f'label {label_set[2]} not in set of patinet: {patient}'
        if label_set[2]=='walking':
            start_sample = int(np.round(label_set[0])*(target_sample_rate/source_sample_rate))
            end_sample = int(np.round(label_set[1])*(target_sample_rate/source_sample_rate))
            labels_array[start_sample:end_sample]=1
    chorea_labels = np.ones(last_labeled_frame_sample+1).astype(int) * -1
    for label_set in chorea_labels_by_frames:
        start_sample = int(np.round(label_set[0])*(target_sample_rate/source_sample_rate))
        end_sample = int(np.round(label_set[1])*(target_sample_rate/source_sample_rate))
        level =  -1 if label_set[2] in ['', 'hided'] else int(label_set[2])
        chorea_labels[start_sample:end_sample]=level     
    # TODO: get label FPS ???????
    return labels_array, chorea_labels

def sync_data(acc_data, label_data, chorea_labels, sync_sec, ACC_SAMPLE_RATE):
    '''
    gets 2 numpy array and the sync time in seconds.
    The function sync the array and trim the edges so the 
    array will be with the same size
    '''
    acc_time_before_sync_event = int(ACC_SAMPLE_RATE*sync_sec[1])
    label_time_before_sync_event = int(ACC_SAMPLE_RATE*sync_sec[0])
    acc_data = acc_data[np.maximum(0, acc_time_before_sync_event-label_time_before_sync_event):]
    video_first_index = np.maximum(0, label_time_before_sync_event-acc_time_before_sync_event)
    label_data = label_data[video_first_index:]
    chorea_labels = chorea_labels[video_first_index:]
    # trim the end
    acc_data = acc_data[:label_data.shape[0]]
    video_first_sec = video_first_index / ACC_SAMPLE_RATE
    time_data = np.array(range(label_data.shape[0])) / ACC_SAMPLE_RATE + video_first_sec
    return acc_data, label_data, chorea_labels, time_data

    

if __name__ == "__main__":
    main()
