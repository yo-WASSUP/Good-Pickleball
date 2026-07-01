import time

import cv2
import numpy as np

from ..detection.rtmpose import RTMPoseProcessor
from ..court.mapper import PICKLEBALL_COURT_LENGTH, PICKLEBALL_COURT_WIDTH


class PlayerPoseVisualizer:
    """Detect, filter, and draw player pose keypoints."""

    def __init__(
        self,
        rtmpose_processor=None,
        person_detector=None,
        player_detector="pose",
        show_skeletons=True,
        show_player_trajectories=True,
        show_performance_stats=False,
        court_filter_margin=(3.0, 5.0),
    ):
        self.person_detector = person_detector
        self.player_detector = player_detector
        self.rtmpose_processor = rtmpose_processor
        if self.player_detector != "yolo-person" and self.rtmpose_processor is None:
            self.rtmpose_processor = RTMPoseProcessor()
        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_performance_stats = show_performance_stats
        self.current_pose_data = None
        self.current_box_data = None
        self.court_mapper = None
        self.court_filter_margin = court_filter_margin

        self.skeleton_connections = [
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

    def detect_players(self, roi, x1, y1, court_mapper=None):
        if self.player_detector == "yolo-person":
            return self._detect_players_with_boxes(roi, x1, y1, court_mapper)
        return self._detect_players_with_pose(roi, x1, y1, court_mapper)

    def _detect_players_with_boxes(self, roi, x1, y1, court_mapper=None):
        centroids = []
        point_left_hands = {}
        point_right_hands = {}
        self.current_pose_data = None

        if self.person_detector is None:
            self.current_box_data = None
            return centroids, point_left_hands, point_right_hands

        t0 = time.time()
        boxes = self.person_detector.process_frame(roi)
        if self.show_performance_stats:
            inference_name = getattr(self.person_detector, "inference_name", "YOLO-Person")
            print(f"{inference_name} inference took {time.time() - t0:.2f} sec")

        active_court_mapper = court_mapper or self.court_mapper
        filtered_boxes = []
        for box in boxes:
            bx1, by1, bx2, by2 = box[:4]
            bottom_center = (float((bx1 + bx2) / 2 + x1), float(by2 + y1))
            if not self._is_on_court(bottom_center, active_court_mapper):
                continue
            centroids.append({
                "point": bottom_center,
                "track_id": box[5] if len(box) > 5 else None,
                "score": box[4] if len(box) > 4 else None,
            })
            filtered_boxes.append(box)

        if filtered_boxes:
            self.current_box_data = {
                "boxes": filtered_boxes,
                "offset_x": x1,
                "offset_y": y1,
            }
        else:
            self.current_box_data = None

        return centroids, point_left_hands, point_right_hands

    def _detect_players_with_pose(self, roi, x1, y1, court_mapper=None):
        centroids = []
        point_left_hands = {}
        point_right_hands = {}
        self.current_box_data = None

        t0 = time.time()
        keypoints_all, _confidence_scores = self.rtmpose_processor.process_frame(roi)
        if self.show_performance_stats:
            inference_name = getattr(self.rtmpose_processor, "inference_name", "Pose")
            print(f"{inference_name} inference took {time.time() - t0:.2f} sec")

        if keypoints_all is None:
            self.current_pose_data = None
            return centroids, point_left_hands, point_right_hands

        persons = self._normalize_people(keypoints_all)
        filtered_people = []
        active_court_mapper = court_mapper or self.court_mapper

        for kp in persons:
            kp_arr = np.asarray(kp)
            if kp_arr.ndim != 2 or kp_arr.shape[0] < 17 or kp_arr.shape[1] < 2:
                continue

            lf = kp_arr[15]
            rf = kp_arr[16]
            if lf[0] <= 1 or lf[1] <= 1 or rf[0] <= 1 or rf[1] <= 1:
                continue

            mid_point = (
                (float(lf[0] + x1) + float(rf[0] + x1)) / 2,
                (float(lf[1] + y1) + float(rf[1] + y1)) / 2 + 10,
            )
            if not self._is_on_court(mid_point, active_court_mapper):
                continue

            filtered_people.append(kp_arr)
            centroids.append(mid_point)

            lh = kp_arr[9]
            rh = kp_arr[10]
            if lh[0] > 1 and lh[1] > 1:
                point_left_hands[mid_point[1]] = (int(lh[0] + x1), int(lh[1] + y1))
            if rh[0] > 1 and rh[1] > 1:
                point_right_hands[mid_point[1]] = (int(rh[0] + x1), int(rh[1] + y1))

        if filtered_people:
            self.current_pose_data = {
                "keypoints": np.asarray(filtered_people),
                "offset_x": x1,
                "offset_y": y1,
            }
        else:
            self.current_pose_data = None

        return centroids, point_left_hands, point_right_hands

    def _normalize_people(self, keypoints):
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim == 2:
                return [keypoints]
            if keypoints.ndim == 3:
                return [keypoints[i] for i in range(keypoints.shape[0])]
            return []
        if isinstance(keypoints, (list, tuple)):
            return list(keypoints)
        return []

    def _is_on_court(self, image_point, court_mapper):
        if court_mapper is None:
            return True
        court_position = court_mapper.image_to_court(image_point)
        if court_position is None or len(court_position) < 2:
            return False
        x, y = float(court_position[0]), float(court_position[1])
        margin = self.court_filter_margin
        if isinstance(margin, (list, tuple)):
            margin_x, margin_y = float(margin[0]), float(margin[1])
        else:
            margin_x = margin_y = float(margin)
        return -margin_x <= x <= PICKLEBALL_COURT_WIDTH + margin_x and -margin_y <= y <= PICKLEBALL_COURT_LENGTH + margin_y

    def draw_players(self, frame, player_tracker, cached_movement_stats, stats_visualizer=None, rally_count=0):
        if self.current_box_data is not None:
            selected_players = {
                position: player_tracker.players[position]
                for position in ["upper", "lower"]
                if player_tracker.players[position] is not None
            }
            self._draw_boxes_on_frame(
                frame,
                self.current_box_data["boxes"],
                self.current_box_data["offset_x"],
                self.current_box_data["offset_y"],
                selected_players,
            )

        if self.show_skeletons and self.current_pose_data is not None:
            t0 = time.time()
            self._draw_skeleton_on_frame(
                frame,
                self.current_pose_data["keypoints"],
                self.current_pose_data["offset_x"],
                self.current_pose_data["offset_y"],
            )
            if self.show_performance_stats:
                print(f"Drawing skeleton took {time.time() - t0:.2f} sec")

        t0 = time.time()
        for position in ["upper", "lower"]:
            if player_tracker.players[position] is None:
                continue

            color = (0, 255, 255) if position == "upper" else (255, 0, 255)
            cv2.circle(frame, tuple(map(int, player_tracker.players[position])), 5, color, -1, cv2.LINE_AA)
            if self.current_box_data is None:
                self._draw_player_label(frame, position, player_tracker.players[position], color)

            if self.show_player_trajectories:
                history = list(player_tracker.history[position])
                for i, pos in enumerate(history):
                    if pos is None:
                        continue
                    radius = int(2 + (i / len(history)) * 3) if history else 2
                    cv2.circle(frame, tuple(map(int, pos)), radius, color, -1, cv2.LINE_AA)

        if self.show_performance_stats:
            print(f"Drawing players and trajectories took {time.time() - t0:.2f} sec")

        if stats_visualizer is not None:
            t0 = time.time()
            stats_visualizer.draw_player_stats(frame, cached_movement_stats, rally_count)
            if self.show_performance_stats:
                print(f"Drawing player stats took {time.time() - t0:.2f} sec")

    def _draw_skeleton_on_frame(self, frame, keypoints, offset_x, offset_y):
        for person in self._normalize_people(keypoints):
            person_arr = np.asarray(person)
            if person_arr.ndim != 2 or person_arr.shape[1] < 2:
                continue

            keypoint_count = person_arr.shape[0]
            for a, b in self.skeleton_connections:
                if a >= keypoint_count or b >= keypoint_count:
                    continue
                x1, y1 = float(person_arr[a, 0]), float(person_arr[a, 1])
                x2, y2 = float(person_arr[b, 0]), float(person_arr[b, 1])
                if x1 > 1 and y1 > 1 and x2 > 1 and y2 > 1:
                    pt1 = (int(x1 + offset_x), int(y1 + offset_y))
                    pt2 = (int(x2 + offset_x), int(y2 + offset_y))
                    cv2.line(frame, pt1, pt2, (255, 191, 0), 2, cv2.LINE_AA)

            for i in range(keypoint_count):
                x_raw, y_raw = float(person_arr[i, 0]), float(person_arr[i, 1])
                if x_raw > 1 and y_raw > 1:
                    cv2.circle(frame, (int(x_raw + offset_x), int(y_raw + offset_y)), 3, (255, 128, 0), -1, cv2.LINE_AA)

    def _draw_boxes_on_frame(self, frame, boxes, offset_x, offset_y, selected_players):
        selected_players = {
            region: np.array(pos, dtype=np.float32)
            for region, pos in selected_players.items()
        }
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            bottom_center_arr = np.array([float((x1 + x2) / 2 + offset_x), float(y2 + offset_y)], dtype=np.float32)
            region = self._matched_selected_region(bottom_center_arr, selected_players)
            if region is None:
                continue
            color = (0, 255, 255) if region == "upper" else (255, 0, 255)
            pt1 = (int(x1 + offset_x), int(y1 + offset_y))
            pt2 = (int(x2 + offset_x), int(y2 + offset_y))
            bottom_center = tuple(map(int, bottom_center_arr))
            cv2.rectangle(frame, pt1, pt2, color, 2, cv2.LINE_AA)
            cv2.circle(frame, bottom_center, 4, color, -1, cv2.LINE_AA)
            self._draw_player_label(frame, region, (pt1[0], pt1[1]), color, anchor="box")

    def _matched_selected_region(self, bottom_center, selected_players):
        best_region = None
        best_distance = None
        for region, selected in selected_players.items():
            distance = float(np.linalg.norm(bottom_center - selected))
            if distance <= 12 and (best_distance is None or distance < best_distance):
                best_region = region
                best_distance = distance
        return best_region

    def _draw_player_label(self, frame, region, point, color, anchor="point"):
        label = "Upper" if region == "upper" else "Lower"
        x, y = int(point[0]), int(point[1])
        if anchor == "point":
            origin = (x + 8, y - 8)
        else:
            origin = (x, max(18, y - 7))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.54
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        x1 = max(0, origin[0] - 3)
        y1 = max(0, origin[1] - text_height - baseline - 3)
        x2 = min(frame.shape[1] - 1, origin[0] + text_width + 3)
        y2 = min(frame.shape[0] - 1, origin[1] + baseline + 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.putText(frame, label, origin, font, scale, color, thickness, cv2.LINE_AA)

    def get_current_pose_data(self):
        return self.current_pose_data


