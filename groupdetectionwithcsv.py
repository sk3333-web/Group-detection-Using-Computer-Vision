from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import time
import logging
import csv  # Import the csv module to handle CSV operations
from pathlib import Path  # Import Path from pathlib to handle file paths
import argparse  # Import argparse for command-line argument parsing

# Configure logging to output to console only
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OfficeGroupDetector:
    def __init__(self, min_distance=150, max_group_distance=300, confidence_threshold=0.4, window_size=5):
        # Added min_distance and max_group_distance as parameters
        self.MIN_DISTANCE_THRESHOLD = min_distance
        self.MAX_GROUP_DISTANCE = max_group_distance
        self.workspace_groups = defaultdict(set)

        # Parameters for gender detection
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.detected_persons = {}

    def calculate_centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_proximity(self, centroid1, centroid2):
        return np.sqrt(((centroid1[0] - centroid2[0]) ** 2) +
                      ((centroid1[1] - centroid2[1]) ** 2))

    def detect_workspace_groups(self, current_frame_objects):
        """Detect groups based on proximity and seating arrangement."""
        workspace_groups = defaultdict(set)
        processed = set()

        sorted_objects = sorted(current_frame_objects.items(),
                                key=lambda x: (x[1]['centroid'][1], x[1]['centroid'][0]))  # Sort by y, then x

        current_group_id = 1
        current_row = []
        last_y = None

        for obj_id, obj_info in sorted_objects:
            if obj_id in processed:
                continue

            current_y = obj_info['centroid'][1]

            # Start new row if significant y-difference
            if last_y is not None and abs(current_y - last_y) > self.MIN_DISTANCE_THRESHOLD:
                if len(current_row) >= 1:
                    for member_id in current_row:
                        workspace_groups[current_group_id].add(member_id)
                    current_group_id += 1
                current_row = []

            current_row.append(obj_id)
            last_y = current_y
            processed.add(obj_id)

            # Check horizontal proximity for adjacent people
            for other_id, other_info in sorted_objects:
                if other_id not in processed:
                    distance = self.calculate_proximity(obj_info['centroid'], other_info['centroid'])
                    if distance < self.MAX_GROUP_DISTANCE:
                        current_row.append(other_id)
                        processed.add(other_id)

        # Process the last row
        if len(current_row) >= 1:
            for member_id in current_row:
                workspace_groups[current_group_id].add(member_id)
            current_group_id += 1

        return workspace_groups

    def detect_gender(self, person_crop, id, gender_model):
        """Detect gender using YOLO gender detection model."""
        try:
            gender_results = gender_model(person_crop)
            if gender_results and len(gender_results[0].boxes) > 0:
                scores = gender_results[0].boxes.conf.cpu().numpy()
                classes = gender_results[0].boxes.cls.cpu().numpy()

                # Determine the gender with the highest confidence
                max_idx = scores.argmax()
                confidence = scores[max_idx]
                detected_class = int(classes[max_idx])

                gender = 'Male' if detected_class == 1 else 'Female'
            else:
                gender = 'Unknown'
                confidence = 0.0

        except Exception as e:
            logging.error(f"Error in gender detection for person {id}: {e}")
            gender = 'Unknown'
            confidence = 0.0

        if id not in self.detected_persons:
            self.detected_persons[id] = {
                'gender_history': deque(maxlen=self.window_size),
                'male_confidence': 0.6,
                'female_confidence': 0.5,
                'recorded': False
            }

        person = self.detected_persons[id]
        person['gender_history'].append((gender, confidence))

        # Update cumulative confidence scores if above the threshold
        if confidence > self.confidence_threshold:
            if gender == 'Male':
                person['male_confidence'] += confidence
            elif gender == 'Female':
                person['female_confidence'] += confidence

        # Determine the current gender based on cumulative confidence or majority in sliding window
        if person['male_confidence'] > person['female_confidence']:
            current_gender = 'Male'
        elif person['female_confidence'] > person['male_confidence']:
            current_gender = 'Female'
        else:
            # Use majority in the sliding window as a fallback
            recent_genders = [g for g, c in person['gender_history'] if c > self.confidence_threshold]
            current_gender = max(set(recent_genders), key=recent_genders.count) if recent_genders else 'Unknown'

        person['gender'] = current_gender
        logging.debug(f"Updated gender for person {id}: {current_gender}")
        return current_gender

    def update_groups(self, tracked_objects, gender_model, frame):
        current_frame_objects = {}

        for obj in tracked_objects:
            if hasattr(obj, 'id') and obj.id is not None:
                try:
                    x1, y1, x2, y2 = map(float, obj.xyxy[0])
                    obj_id = int(obj.id[0])
                    centroid = self.calculate_centroid((x1, y1, x2, y2))

                    # Detect gender for the person
                    person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    gender = self.detect_gender(person_crop, obj_id, gender_model)

                    # Store centroid and gender
                    current_frame_objects[obj_id] = {'centroid': centroid, 'gender': gender}
                except (IndexError, AttributeError) as e:
                    logging.error(f"Error processing object {obj_id}: {e}")
                    continue

        workspace_groups = self.detect_workspace_groups(current_frame_objects)

        group_types = {}
        group_type_counts = defaultdict(int)  # Dictionary to hold counts of each group type

        for group_id, members in workspace_groups.items():
            num_people = len(members)
            genders = [current_frame_objects[member]['gender'] for member in members]

            if num_people == 2:
                if 'Male' in genders and 'Female' in genders:
                    group_types[group_id] = 'Couple'
                else:
                    group_types[group_id] = 'Pair'
            elif num_people > 2:
                group_types[group_id] = 'Group'
            else:  # Single-member groups
                group_types[group_id] = 'Individual'

            # Update group type counts
            group_type_counts[group_types[group_id]] += 1

        return workspace_groups, group_types, current_frame_objects, group_type_counts


def process_rtsp_stream(rtsp_url, model_path, gender_model_path, tracker_path, headless=False, frame_skip=3, detection_interval=1 / 15, min_distance=150, max_group_distance=300):
    """
    Process RTSP stream with YOLO model for object detection and gender classification using ByteTrack.
    """
    model = YOLO(model_path)
    gender_model = YOLO(gender_model_path)

    # Instantiate the detector with the provided distance thresholds
    detector = OfficeGroupDetector(min_distance=min_distance, max_group_distance=max_group_distance)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"Failed to open RTSP stream: {rtsp_url}")
        return

    frame_count = 0
    last_detection_time = 0

    # Define the CSV directory and file path
    csv_dir = Path(r"Allcsv")
    csv_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    csv_file = csv_dir / 'group_counts.csv'

    # Initialize the CSV file before the loop
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Time', 'Individuals', 'Couples', 'Pairs', 'Groups', 'Group ID', 'Members', 'Status'])
        writer.writeheader()
        logging.info(f"CSV file initialized at {csv_file}")

    # Initialize variables for tracking groups across frames
    prev_groups = {}  # {group_id: set(member_ids)}
    group_id_map = {}  # {group_signature: group_id}
    group_id_counter = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from RTSP stream.")
            break

        frame_count += 1
        if frame_count % (frame_skip + 1) != 0:
            continue

        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            try:
                # Use ByteTrack as the tracker backend
                results = model.track(source=frame, conf=0.5, classes=[0], persist=True, tracker=tracker_path)

                if results and len(results) > 0:
                    tracked_objects = results[0].boxes
                    groups, group_types, current_frame_objects, group_type_counts = detector.update_groups(tracked_objects, gender_model, frame)

                    # Prepare data for CSV
                    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    current_groups = {}
                    group_data_list = []
                    unmatched_prev_groups = set(prev_groups.keys())

                    for group_key, members in groups.items():
                        members_set = set(members)
                        group_signature = tuple(sorted(members_set))

                        # Check if this group was seen before
                        if group_signature in group_id_map:
                            group_id = group_id_map[group_signature]
                            status = 'Unchanged'
                            unmatched_prev_groups.discard(group_id)
                        else:
                            group_id = f'G{group_id_counter}'
                            group_id_counter += 1
                            group_id_map[group_signature] = group_id
                            status = 'New'

                        current_groups[group_id] = members_set

                        group_data = {
                            'Time': current_time_str,
                            'Individuals': group_type_counts.get('Individual', 0),
                            'Couples': group_type_counts.get('Couple', 0),
                            'Pairs': group_type_counts.get('Pair', 0),
                            'Groups': group_type_counts.get('Group', 0),
                            'Group ID': group_id,
                            'Members': ', '.join([f'Id {member_id}' for member_id in members_set]),
                            'Status': status
                        }
                        group_data_list.append(group_data)

                    # For unmatched previous groups, mark them as 'Split' or 'Dissolved'
                    for group_id in unmatched_prev_groups:
                        group_data = {
                            'Time': current_time_str,
                            'Individuals': group_type_counts.get('Individual', 0),
                            'Couples': group_type_counts.get('Couple', 0),
                            'Pairs': group_type_counts.get('Pair', 0),
                            'Groups': group_type_counts.get('Group', 0),
                            'Group ID': group_id,
                            'Members': ', '.join([f'Id {member_id}' for member_id in prev_groups[group_id]]),
                            'Status': 'Split'
                        }
                        group_data_list.append(group_data)

                    # Update previous groups
                    prev_groups = current_groups.copy()

                    # Write to CSV
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=['Time', 'Individuals', 'Couples', 'Pairs', 'Groups', 'Group ID', 'Members', 'Status'])
                        for group_data in group_data_list:
                            writer.writerow(group_data)
                    logging.info(f"Written group data to CSV at {current_time_str}")

                    if not headless:
                        # Draw bounding boxes and annotations for each detected object
                        for obj in tracked_objects:
                            try:
                                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                                obj_id = int(obj.id[0]) if obj.id is not None else -1
                                gender = current_frame_objects.get(obj_id, {}).get('gender', 'Unknown')

                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Draw gender
                                cv2.putText(frame, f"Id {obj_id}: {gender}",
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                            except Exception as e:
                                logging.error(f"Error drawing box for object {obj_id}: {e}")

                        # Annotate groups on the frame
                        for group_key, members in groups.items():
                            group_type = group_types.get(group_key, 'Individual')
                            group_signature = tuple(sorted(set(members)))
                            group_id = group_id_map.get(group_signature, 'Unknown')
                            member_centroids = [current_frame_objects[member]['centroid'] for member in members]

                            # Draw group type and ID annotation
                            for centroid in member_centroids:
                                cx, cy = map(int, centroid)
                                cv2.putText(frame, f"{group_type} {group_id}", (cx, cy - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                            # Draw lines between group members if the group has more than one member
                            if len(members) > 1:
                                for i, c1 in enumerate(member_centroids):
                                    for c2 in member_centroids[i + 1:]:
                                        cv2.line(frame, tuple(map(int, c1)), tuple(map(int, c2)), (255, 0, 0), 1)

                        # Display the processed frame
                        cv2.imshow("Office Group Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logging.info("Termination signal received. Exiting...")
                            break

                    last_detection_time = current_time

            except Exception as e:
                logging.error(f"Error processing frame: {e}")

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    logging.info("RTSP stream processing terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Office Group Detection using YOLO and ByteTrack")
    parser.add_argument('--headless', action='store_true', help="Run the script without displaying GUI windows")
    parser.add_argument('--rtsp_url', type=str, default="RTSP-LINKS", help="RTSP stream URL")
    parser.add_argument('--model_path', type=str, default=r"yolov8x.pt", help="Path to YOLO model")
    parser.add_argument('--gender_model_path', type=str, default=r"Gendermodel_Yolov8.pt", help="Path to Gender detection model")
    parser.add_argument('--tracker_path', type=str, default=r"bytetrack.yaml", help="Path to ByteTrack YAML file")
    parser.add_argument('--min_distance', type=int, default=100, help="Minimum distance threshold")
    parser.add_argument('--max_group_distance', type=int, default=200, help="Maximum group distance threshold")
    parser.add_argument('--frame_skip', type=int, default=3, help="Number of frames to skip")
    parser.add_argument('--detection_interval', type=float, default=1/15, help="Interval between detections in seconds")

    args = parser.parse_args()

    process_rtsp_stream(
        rtsp_url=args.rtsp_url,
        model_path=args.model_path,
        gender_model_path=args.gender_model_path,
        tracker_path=args.tracker_path,
        headless=args.headless,
        frame_skip=args.frame_skip,
        detection_interval=args.detection_interval,
        min_distance=args.min_distance,
        max_group_distance=args.max_group_distance
    )
