from collections import deque

import numpy as np

from ..court.mapper import PICKLEBALL_COURT_WIDTH, PICKLEBALL_COURT_LENGTH, CourtMapper


class PlayerTracker:
    """
    Player tracking system.

    Tracks player positions, court coordinates, movement statistics, and writes
    one structured detection record per processed court frame.
    """

    def __init__(self, corners, threshold=680, history_size=50, detection_writer=None, fps=30):
        self.threshold = threshold
        self.fps = fps
        self.detection_writer = detection_writer
        self.max_frame_distance = 8.0 / self.fps
        self.strict_sideline_margin = 1.8
        self.loose_sideline_margin = 3.0
        self.baseline_margin = 5.0
        self.net_margin = 1.2

        self.players = {
            "upper": None,
            "lower": None,
        }
        self.history = {
            "upper": deque(maxlen=history_size),
            "lower": deque(maxlen=history_size),
        }
        self.court_history = {
            "upper": deque(maxlen=history_size),
            "lower": deque(maxlen=history_size),
        }

        self.match_stats = {
            "upper": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
            "lower": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
        }
        self.rally_stats = {
            "upper": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
            "lower": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
        }
        self.current_speed = {
            "upper": 0,
            "lower": 0,
        }
        self.region_track_ids = {
            "upper": None,
            "lower": None,
        }
        self.last_update_frame = {
            "upper": None,
            "lower": None,
        }
        self.track_id_bonus = 0.75
        self.track_id_bonus_max_court_distance = 2.5
        self.max_court_jump_min = 2.2
        self.max_court_jump_speed = 12.0

        self.court_mapper = CourtMapper(corners)

    def _empty_player_record(self):
        return {
            "image": None,
            "court": None,
            "speed": None,
            "track_id": None,
            "hands": {
                "left": None,
                "right": None,
            },
        }

    def _initialize_player_record(self):
        return {
            "upper": self._empty_player_record(),
            "lower": self._empty_player_record(),
        }

    def _point_or_none(self, point, zero_is_none=False):
        if point is None:
            return None
        try:
            x, y = point[0], point[1]
        except (TypeError, IndexError):
            return None
        if x is None or y is None:
            return None
        if zero_is_none and float(x) == 0.0 and float(y) == 0.0:
            return None
        return [float(x), float(y)]

    def _candidate_point(self, candidate):
        if isinstance(candidate, dict):
            return candidate.get("point")
        return candidate

    def _candidate_track_id(self, candidate):
        if not isinstance(candidate, dict):
            return None
        track_id = candidate.get("track_id")
        if track_id is None:
            return None
        try:
            return int(track_id)
        except (TypeError, ValueError):
            return None

    def write_detection_record(self, frame_index, players_record, ball_image_position, detect_frame_count,
                               ball_court_position=None, bounce_event=None):
        if self.detection_writer is None:
            return

        record = {
            "schema_version": "1.0",
            "frame": int(frame_index),
            "time_sec": round(frame_index / self.fps, 6) if self.fps else None,
            "detect_frame": int(detect_frame_count),
            "players": players_record,
            "pickleball_ball": {
                "image": self._point_or_none(ball_image_position, zero_is_none=True),
                "court": self._point_or_none(ball_court_position),
            },
            "bounce": bounce_event,
        }
        self.detection_writer.write(record)

    def update(self, frame_index, centroids, ball_image_position, left_hand_positions, right_hand_positions,
               detect_frame_count, ball_court_position=None, bounce_event=None):
        players_record = self._initialize_player_record()

        for region in ["upper", "lower"]:
            if self.players[region] is not None:
                self.match_stats[region]["total_frames"] += 1
                self.rally_stats[region]["total_frames"] += 1

        upper_court_centroids = []
        lower_court_centroids = []
        for centroid in centroids:
            point = self._candidate_point(centroid)
            if point is None:
                continue
            if point[1] < self.threshold:
                upper_court_centroids.append(centroid)
            else:
                lower_court_centroids.append(centroid)

        upper_court_centroids = self._select_region_candidate("upper", upper_court_centroids, frame_index)
        lower_court_centroids = self._select_region_candidate("lower", lower_court_centroids, frame_index)

        filtered_centroids = upper_court_centroids + lower_court_centroids

        for centroid in filtered_centroids:
            try:
                point = self._candidate_point(centroid)
                if point is None:
                    continue
                region = "upper" if point[1] < self.threshold else "lower"
                left_hand = left_hand_positions.get(point[1])
                right_hand = right_hand_positions.get(point[1])
                self._update_player_position(region, centroid, left_hand, right_hand, players_record, frame_index)
            except Exception as exc:
                print(f"Error processing player position: {exc}")
                import traceback
                traceback.print_exc()

        self.write_detection_record(
            frame_index,
            players_record,
            ball_image_position,
            detect_frame_count,
            ball_court_position=ball_court_position,
            bounce_event=bounce_event,
        )
        return self.players

    def _select_region_candidate(self, region, candidates, frame_index):
        if not candidates:
            return []

        court_width, court_length = self.court_mapper.court_dimensions
        center_x = court_width / 2
        candidates_with_court = []
        for candidate in candidates:
            point = self._candidate_point(candidate)
            if point is None:
                continue
            court_position = self.court_mapper.image_to_court(point)
            if court_position is None or len(court_position) < 2:
                continue
            x, y = float(court_position[0]), float(court_position[1])
            candidates_with_court.append({
                "candidate": candidate,
                "point": point,
                "x": x,
                "y": y,
                "track_id": self._candidate_track_id(candidate),
            })

        if candidates_with_court:
            plausible_candidates = [
                item for item in candidates_with_court
                if self._is_plausible_court_movement(region, item["x"], item["y"], frame_index)
            ]
            if plausible_candidates:
                candidates_with_court = plausible_candidates
            elif self._last_court_position(region) is not None:
                return []

            strict_candidates = [
                item for item in candidates_with_court
                if self._is_valid_player_court_position(region, item["x"], item["y"], strict=True)
            ]
            scored_candidates = strict_candidates
            if not scored_candidates:
                scored_candidates = [
                    item for item in candidates_with_court
                    if self._is_valid_player_court_position(region, item["x"], item["y"], strict=False)
                    and self._is_close_to_previous_court_position(region, item["x"], item["y"])
                ]
            if scored_candidates:
                scored_candidates.sort(key=lambda item: self._candidate_score(region, item, center_x, court_length))
                return [scored_candidates[0]["candidate"]]
            return []

        if region == "upper":
            candidates = [candidate for candidate in candidates if self._candidate_point(candidate) is not None]
            candidates = sorted(candidates, key=lambda p: -self._candidate_point(p)[1])
        else:
            candidates = [candidate for candidate in candidates if self._candidate_point(candidate) is not None]
            candidates = sorted(candidates, key=lambda p: -self._candidate_point(p)[1])
        if not candidates:
            return []
        return [candidates[0]]

    def _is_valid_player_court_position(self, region, x, y, strict=True):
        sideline_margin = self.strict_sideline_margin if strict else self.loose_sideline_margin
        if x < -sideline_margin or x > PICKLEBALL_COURT_WIDTH + sideline_margin:
            return False

        net_y = PICKLEBALL_COURT_LENGTH / 2
        if region == "upper":
            return -self.baseline_margin <= y <= net_y + self.net_margin
        return net_y - self.net_margin <= y <= PICKLEBALL_COURT_LENGTH + self.baseline_margin

    def _candidate_score(self, region, item, center_x, court_length):
        point = item["point"]
        x = item["x"]
        y = item["y"]
        net_y = court_length / 2
        expected_y = 0.0 if region == "upper" else court_length
        score = 0.0
        score += abs(x - center_x) * 1.4
        score += abs(y - expected_y) * 0.7

        if region == "upper" and y > net_y:
            score += 40.0
        if region == "lower" and y < net_y:
            score += 40.0

        outside_x = max(0.0, -x, x - PICKLEBALL_COURT_WIDTH)
        score += outside_x * 8.0

        previous_court = self._last_court_position(region)
        previous_court_distance = None
        if previous_court is not None:
            previous_court_distance = float(np.linalg.norm(np.array([x, y]) - previous_court))
            score += previous_court_distance * 0.8

        previous_image = self.players.get(region)
        if previous_image is not None:
            score += float(np.linalg.norm(np.array(point, dtype=np.float32) - np.array(previous_image, dtype=np.float32))) * 0.01

        preferred_track_id = self.region_track_ids.get(region)
        if preferred_track_id is not None and item["track_id"] == preferred_track_id:
            if previous_court_distance is None or previous_court_distance <= self.track_id_bonus_max_court_distance:
                score -= self.track_id_bonus

        return score

    def _last_court_position(self, region):
        history = self.court_history.get(region, [])
        for point in reversed(history):
            if point is None or len(point) < 2:
                continue
            return np.array([float(point[0]), float(point[1])], dtype=np.float32)
        return None

    def _is_close_to_previous_court_position(self, region, x, y):
        previous_court = self._last_court_position(region)
        if previous_court is None:
            return False
        distance = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - previous_court))
        return distance <= 3.0

    def _is_plausible_court_movement(self, region, x, y, frame_index):
        previous_court = self._last_court_position(region)
        if previous_court is None:
            return True

        distance = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - previous_court))
        previous_frame = self.last_update_frame.get(region)
        if previous_frame is None or self.fps <= 0:
            max_allowed = self.max_court_jump_min
        else:
            elapsed_sec = max(1, int(frame_index) - int(previous_frame)) / self.fps
            max_allowed = max(self.max_court_jump_min, self.max_court_jump_speed * elapsed_sec + 0.8)
        return distance <= max_allowed

    def _update_player_position(self, region, centroid, left_hand_pos, right_hand_pos, players_record, frame_index):
        point = self._candidate_point(centroid)
        if point is None:
            return

        track_id = self._candidate_track_id(centroid)
        if track_id is not None:
            self.region_track_ids[region] = track_id

        self.players[region] = point
        self.history[region].append(point)

        court_position = self.court_mapper.image_to_court(point)
        self.court_history[region].append(court_position)
        self.last_update_frame[region] = int(frame_index)

        player_record = players_record[region]
        player_record["image"] = self._point_or_none(point)
        player_record["court"] = self._point_or_none(court_position)
        player_record["speed"] = float(self.current_speed[region])
        player_record["track_id"] = track_id
        if left_hand_pos:
            player_record["hands"]["left"] = self._point_or_none(left_hand_pos)
        if right_hand_pos:
            player_record["hands"]["right"] = self._point_or_none(right_hand_pos)

    def _update_rally_and_match_stats(self, region, distance, speed):
        capped_speed = round(min(speed, 8.0), 2)

        self.rally_stats[region]["total_distance"] += distance
        self.rally_stats[region]["max_speed"] = max(self.rally_stats[region]["max_speed"], capped_speed)
        self.current_speed[region] = capped_speed

        self.match_stats[region]["total_distance"] += distance
        self.match_stats[region]["max_speed"] = max(self.match_stats[region]["max_speed"], capped_speed)
        self.current_speed[region] = capped_speed

    def start_new_rally(self):
        for region in ["upper", "lower"]:
            self.rally_stats[region]["total_distance"] = 0
            self.rally_stats[region]["max_speed"] = 0
            self.rally_stats[region]["total_frames"] = 0

    def get_player_movement_stats(self):
        stats = {}
        for region in ["upper", "lower"]:
            history = [pos for pos in list(self.court_history[region]) if pos is not None]
            region_stats = {
                "current_speed": 0,
                "rally_avg_speed": 0,
                "rally_max_speed": 0,
                "rally_distance": 0,
                "match_avg_speed": 0,
                "match_max_speed": 0,
                "match_distance": 0,
                "position_count": len(history),
            }

            if len(history) < 2:
                stats[region] = region_stats
                continue

            current_time = len(history) - 1
            window_start = max(0, current_time - int(self.fps / 2))
            half_second_total_distance = 0
            valid_frames = 0
            actual_time_span = 0
            sample_interval = 5

            if current_time - window_start < sample_interval:
                sample_points = [window_start, current_time]
            else:
                sample_points = list(range(window_start, current_time + 1, sample_interval))
                if current_time not in sample_points:
                    sample_points.append(current_time)

            for i in range(len(sample_points) - 1):
                idx1 = sample_points[i]
                idx2 = sample_points[i + 1]
                p1 = np.array(history[idx1])
                p2 = np.array(history[idx2])
                distance = np.linalg.norm(p2 - p1)
                time_span = (idx2 - idx1) / self.fps
                max_possible_distance = self.max_frame_distance * (idx2 - idx1)

                if distance > 0.05 and distance < max_possible_distance:
                    half_second_total_distance += distance
                    valid_frames += 1
                    actual_time_span += time_span

            current_speed = 0
            if valid_frames > 0 and actual_time_span > 0:
                current_speed = half_second_total_distance / actual_time_span
                self._update_rally_and_match_stats(region, half_second_total_distance, current_speed)

            current_speed = min(current_speed, 8.0)

            rally_distance = self.rally_stats[region]["total_distance"]
            rally_max_speed = self.rally_stats[region]["max_speed"]
            rally_frames = self.rally_stats[region]["total_frames"]
            if rally_frames > 1 and self.fps > 0:
                rally_time = rally_frames / self.fps
                rally_avg_speed = rally_distance / rally_time if rally_time > 0 else 0
            else:
                rally_avg_speed = 0

            match_distance = self.match_stats[region]["total_distance"]
            match_max_speed = self.match_stats[region]["max_speed"]
            match_frames = self.match_stats[region]["total_frames"]
            if match_frames > 1 and self.fps > 0:
                match_time = match_frames / self.fps
                match_avg_speed = match_distance / match_time if match_time > 0 else 0
            else:
                match_avg_speed = 0

            region_stats["current_speed"] = round(current_speed, 2)
            region_stats["rally_avg_speed"] = round(rally_avg_speed, 2)
            region_stats["rally_max_speed"] = round(rally_max_speed, 2)
            region_stats["rally_distance"] = round(rally_distance, 2)
            region_stats["match_avg_speed"] = round(match_avg_speed, 2)
            region_stats["match_max_speed"] = round(match_max_speed, 2)
            region_stats["match_distance"] = round(match_distance, 2)
            stats[region] = region_stats

        return stats

    def get_player_trajectories(self):
        return {region: list(history) for region, history in self.history.items()}

    def close(self):
        if self.detection_writer is not None:
            self.detection_writer.close()

