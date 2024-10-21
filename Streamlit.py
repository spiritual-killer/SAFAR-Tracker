import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque, defaultdict
import logging
import re
from datetime import datetime
from sklearn.preprocessing import MaxAbsScaler
from scipy.spatial.distance import euclidean
import os
import base64

# Define the functions as provided

# Function to draw boxes and log bounding box information
className = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0, 0)):
    height, width, _ = frame.shape
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 += offset[0]
        y1 += offset[1]
        x2 += offset[0]
        y2 += offset[1]

        center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cat = int(categories[i]) if categories is not None else 0
        color = (0, 255, 0)  # You can modify this as per category
        id = int(identities[i]) if identities is not None else 0
        name = className[cat]  # Use className to get the object category name

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{id}:{name}"
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Log the bounding box information
        logging.info(f"Bounding Box: ID={id}, Category={name}, Coordinates=({x1}, {y1}),({x2}, {y2})")

    return frame

def extract_info(log_file_path):
    dates, times, ids, categories, top_left_x, top_left_y, bottom_right_x, bottom_right_y = [[] for _ in range(8)]

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "Bounding Box" in line:
            parts = line.strip().split(' ')
            date, time = parts[0], parts[1].split(',')[0]  # Split by ',' and take the first part
            info = parts[4:]

            box_id = info[0].split('=')[1][:-1]

            # Handle single or multiple word categories
            if ',' in info[1]:
                category = info[1].split('=')[1][:-1]
                category_info_index = 2
            else:
                category = info[1].split('=')[1]
                if ',' in info[2]:
                    category += ' ' + info[2].split(',')[0]
                    category_info_index = 3
                else:
                    category += ' ' + info[2]
                    category_info_index = 4

            # Extract coordinates
            tlx = info[category_info_index].split('=')[1][1:-1]
            tly = info[category_info_index + 1].split(',')[0][0:-1]
            brx = info[category_info_index + 1].split(',')[1][1:]
            bry = info[category_info_index + 2][0:-1]

            dates.append(date)
            times.append(time)
            ids.append(box_id)
            categories.append(category)
            top_left_x.append(tlx)
            top_left_y.append(tly)
            bottom_right_x.append(brx)
            bottom_right_y.append(bry)
        
    data = {
    'Date': dates,
    'Time': times,
    'ID': ids,
    'Category': categories,
    'TLX': top_left_x,
    'TLY': top_left_y,
    'BRX': bottom_right_x,
    'BRY': bottom_right_y
}

    df = pd.DataFrame(data)
    most_common_category = df.groupby('ID')['Category'].agg(lambda x: x.mode()[0])
    # Replace the Category with the most common category for each ID
    df['Category'] = df['ID'].map(most_common_category)
    df[['ID', 'TLX', 'TLY', 'BRX', 'BRY']] = df[['ID', 'TLX', 'TLY', 'BRX', 'BRY']].astype(int)

    return df

def create_trajectory(df):
    df['X'] = (df['TLX'] + df['BRX']) / 2
    df['Y'] = (df['TLY'] + df['BRY']) / 2

    df.sort_values(by=['ID', 'Date', 'Time'], inplace=True)

    trajectories = []
    for (ID, category), group in df.groupby(['ID', 'Category']):
        trajectory = {
            'ID': ID,
            'Category': category,
            'X': group['X'].tolist(),
            'Y': group['Y'].tolist(),
            'Time': group['Time'].tolist()
        }
        trajectories.append(trajectory)

    for trajectory in trajectories:
        distances = [euclidean((trajectory['X'][i], trajectory['Y'][i]), (trajectory['X'][i+1], trajectory['Y'][i+1]))
                    for i in range(len(trajectory['X'])-1)]

        times = [(pd.to_datetime(trajectory['Time'][i+1]) - pd.to_datetime(trajectory['Time'][i])).total_seconds()
                for i in range(len(trajectory['Time'])-1)]

        velocities = [dist / time if time != 0 else 0 for dist, time in zip(distances, times)]

        trajectory['Speed'] = velocities
    
    return trajectories    

def generate_trajectory_txt(log_file_path, output_file_path):
    trajectories = create_trajectory(extract_info(log_file_path))
    
    with open(output_file_path, "w") as file:
        for trajectory in trajectories:
            file.write(f"ID: {trajectory['ID']}, Category: {trajectory['Category']}\n")
            
            for time, position, speed in zip(trajectory['Time'], zip(trajectory['X'], trajectory['Y']), trajectory['Speed']):
                x, y = position
                file.write(f"Time: {time}, Position: ({x},{y}), Speed: {speed}\n")
            
            file.write("\n")

def parse_trajectory_file(file_path):
    # Regular expression patterns
    id_category_pattern = re.compile(r'ID:\s*(\d+),\s*Category:\s*(\w+)')
    time_pattern = re.compile(r'Time:\s*([\d:]+),\s*Position:\s*\(([\d.]+),([\d.]+)\),\s*Speed:\s*([\d.]+)')

    # Data structure to store parsed data
    data = defaultdict(lambda: {'category': '', 'entries': []})

    current_id = None

    with open(file_path, 'r') as file:
        for line in file:
            id_category_match = id_category_pattern.match(line)
            if id_category_match:
                current_id = int(id_category_match.group(1))
                category = id_category_match.group(2)
                data[current_id]['category'] = category
            else:
                time_match = time_pattern.match(line)
                if time_match:
                    time_str = time_match.group(1)
                    position = (float(time_match.group(2)), float(time_match.group(3)))
                    speed = float(time_match.group(4))
                    data[current_id]['entries'].append({'time': time_str, 'position': position, 'speed': speed})

    return data

def calculate_average_speed(entries):
    total_speed = sum(entry['speed'] for entry in entries)
    return total_speed / len(entries) if entries else 0

def calculate_total_distance(entries):
    total_distance = 0
    for i in range(1, len(entries)):
        x1, y1 = entries[i-1]['position']
        x2, y2 = entries[i]['position']
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        total_distance += distance
    return total_distance

def process_data(data):
    for object_id, object_data in data.items():
        entries = object_data['entries']
        average_speed = calculate_average_speed(entries)
        total_distance = calculate_total_distance(entries)
        print(f"ID: {object_id}, Category: {object_data['category']}")
        print(f"  Average Speed: {average_speed:.2f}")
        print(f"  Total Distance: {total_distance:.2f}")

def generate_statistics(data):
    # Create an empty list to store dictionaries
    data_list = []

    # Convert processed data to list of dictionaries
    for object_id, object_data in data.items():
        entries = object_data['entries']
        average_speed = calculate_average_speed(entries)
        total_distance = calculate_total_distance(entries)
        data_list.append({
            'ID': object_id, 
            'Category': object_data['category'], 
            'Average Speed': average_speed, 
            'Total Distance': total_distance
        })

    # Create DataFrame from the list of dictionaries
    statistics_df = pd.DataFrame(data_list)

    # Initialize MaxAbsScaler
    scaler = MaxAbsScaler()

    # Apply the scaler to 'Average Speed' and 'Total Distance'
    statistics_df[['Relative Speed', 'Relative Distance']] = scaler.fit_transform(statistics_df[['Average Speed', 'Total Distance']])

    # Rescale to the range of 0 to 100
    statistics_df['Relative Speed'] = statistics_df['Relative Speed'] * 10
    statistics_df['Relative Distance'] = statistics_df['Relative Distance'] * 10

    # Drop the original 'Average Speed' and 'Total Distance' columns
    statistics_df.drop(columns=['Average Speed', 'Total Distance'], inplace=True)

    return statistics_df

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1c1c1c;  /* Dark grey background */
        color: white;
    }
    .stProgress > div > div > div > div {
        height: 20px;
    }
    .stProgress > div > div > div > div > div {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 14px;
        color: black;
    }
    .stText {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1c1c1c;  /* Dark grey background */
        color: white;
    }
    .stProgress > div > div > div > div {
        height: 20px;
    }
    .stProgress > div > div > div > div > div {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 14px;
        color: black;
    }
    .stText {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title for the app
st.markdown("<h1 style='color: ;'>Video Object Tracking and Statistics</h1>", unsafe_allow_html=True)

# File uploader section
uploaded_file = st.file_uploader("", type=["mp4"], key="file_uploader")

if uploaded_file is not None:
    # Progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Uploading... 0%")

    # Save the uploaded file to a temporary location
    with open("temp_video.mp4", "wb") as f:
        file_bytes = uploaded_file.getvalue()
        f.write(file_bytes)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
            status_text.text(f"Uploading... {percent_complete + 1}%")
        status_text.text("Upload complete!")

    # Button to get statistics
    if st.button("Get Statistics"):
        # Set up logging
        log_file_path = "temp_log.log"
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')

        # Load YOLOv9 model
        model = YOLO('yolov9c.pt')

        # Show status while ByteTrack is running
        status_text.text("ByteTrack Running...")

        # Track objects using the model
        results = model.track("temp_video.mp4", save=True)

        # Process bounding boxes from the frames
        for frame_id, result in enumerate(results):
            frame = result.orig_img
            bbox_xyxy = result.boxes.xyxy
            identities = result.boxes.id
            categories = result.boxes.cls

            # Draw boxes and log bounding box info
            processed_frame = draw_boxes(frame, bbox_xyxy, draw_trails=True, identities=identities, categories=categories)

        status_text.text("ByteTrack complete!")

        # Generate the trajectory file
        generate_trajectory_txt(log_file_path, "temp_trajectory.txt")

        # Parse trajectory data
        data = parse_trajectory_file("temp_trajectory.txt")

        # Generate statistics
        statistics_df = generate_statistics(data)

        # Display category counts
        st.markdown("<h2 style='color: green; text-align: center; font-family: Arial, sans-serif; font-weight: bold;'>Category Counts</h2>", unsafe_allow_html=True)
        category_counts_html = "".join([
            f"<p style='font-size: 24px; color: deeporange; font-weight: bold; text-align: center; margin: 10px 0;'>{category.capitalize()}: <span style='color: darkgreen;'>{count}</span></p>"
            for category, count in statistics_df['Category'].value_counts().to_dict().items()
        ])
        st.markdown(category_counts_html, unsafe_allow_html=True)

        # Display the statistics table
        st.markdown("<h2 style='color: darkorange; text-align: center;'>Statistics Table</h2>", unsafe_allow_html=True)

        # Use Streamlit's dataframe for better alignment and styling
        st.dataframe(statistics_df.style.set_properties(**{
            'background-color': 'lightgreen',  # Restored light green background
            'color': 'black',
            'border-color': 'black',
            'width': '100%'
        }).set_table_styles([{
            'selector': 'thead th',
            'props': [('background-color', 'lightgreen'), ('font-weight', 'bold')]
        }]).hide(axis="index"), use_container_width=True)

        # Display the output video
        st.markdown("<h2 style='color: orange; text-align: center;'>Output Video</h2>", unsafe_allow_html=True)  # Changed to green
        st.video("/Users/ritwikghosh/runs/detect/track13/temp_video.mp4")

        # Function to create a green download button for the video
        def download_video_button(video_path):
            with open(video_path, "rb") as file:
                video_bytes = file.read()
            b64 = base64.b64encode(video_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="output_video.mp4"><button style="background-color: green; color: white; padding: 10px 20px; border: none; border-radius: 5px;">Download Output Video</button></a>'
            return href

        # Provide the download button
        video_path = "/Users/ritwikghosh/runs/detect/track13/temp_video.mp4"  # Ensure this path is correct
        st.markdown(download_video_button(video_path), unsafe_allow_html=True)
