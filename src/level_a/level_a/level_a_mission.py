#!/usr/bin/env python3

import math
import time

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import rclpy

from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class MissionState(Enum):
    NORMAL = 'NORMAL'
    FIRST_AVOIDING = 'FIRST_AVOIDING'
    WAIT_NEXT_PIG = 'WAIT_NEXT_PIG'
    NEXT_AVOIDING = 'NEXT_AVOIDING'


@dataclass
class PigObservation:
    confidence: float
    forward_x_m: float
    lateral_y_m: float
    vertical_z_m: float
    side: str
    optical_depth_m: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def bbox_width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def bbox_height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def bbox_area(self) -> float:
        return self.bbox_width * self.bbox_height

    @property
    def distance_valid(self) -> bool:
        return math.isfinite(self.forward_x_m) and self.forward_x_m > 0.0


class LevelAMission(Node):
    """Level A multi-pig avoidance state machine."""

    ALL_DETECTIONS_FORMAT_VERSION = 1
    ALL_DETECTIONS_RECORD_SIZE = 10

    def __init__(self) -> None:
        super().__init__('level_a_mission')

        # The first pig keeps the existing 40 cm avoidance condition.
        self.declare_parameter('avoid_start_distance', 0.40)

        # The second and all later pigs use the requested 30 cm condition.
        self.declare_parameter('next_pig_trigger_distance', 0.30)

        # Missing is counted per received detection frame, not at the 20 Hz
        # mission timer rate.
        self.declare_parameter('pig_missing_confirm_frames', 5)

        # Raw all-pig detections need consecutive confirmation before lock.
        self.declare_parameter('candidate_confirm_frames', 3)

        # A tentative first candidate is not replaced when YOLO misses it for
        # only one or two frames and a farther pig remains visible.
        self.declare_parameter('candidate_missing_tolerance_frames', 2)

        # Stop output if the all-pig stream disappears. A timeout never means
        # that the active pig has been cleared.
        self.declare_parameter('detection_timeout_sec', 1.50)

        # Small association gate used because the existing detector has no ID.
        self.declare_parameter('active_match_min_iou', 0.10)
        self.declare_parameter('active_match_max_center_ratio', 0.65)
        self.declare_parameter('active_match_min_area_ratio', 0.40)
        self.declare_parameter('active_match_max_area_ratio', 2.50)

        self.avoid_start_distance = float(
            self.get_parameter('avoid_start_distance').value
        )
        self.next_pig_trigger_distance = float(
            self.get_parameter('next_pig_trigger_distance').value
        )
        self.pig_missing_confirm_frames = int(
            self.get_parameter('pig_missing_confirm_frames').value
        )
        self.candidate_confirm_frames = int(
            self.get_parameter('candidate_confirm_frames').value
        )
        self.candidate_missing_tolerance_frames = int(
            self.get_parameter('candidate_missing_tolerance_frames').value
        )
        self.detection_timeout_sec = float(
            self.get_parameter('detection_timeout_sec').value
        )
        self.active_match_min_iou = float(
            self.get_parameter('active_match_min_iou').value
        )
        self.active_match_max_center_ratio = float(
            self.get_parameter('active_match_max_center_ratio').value
        )
        self.active_match_min_area_ratio = float(
            self.get_parameter('active_match_min_area_ratio').value
        )
        self.active_match_max_area_ratio = float(
            self.get_parameter('active_match_max_area_ratio').value
        )

        self.validate_parameters()

        self.state = MissionState.NORMAL
        self.target_side = 'CENTER'
        self.active_pig: Optional[PigObservation] = None
        self.active_pig_missing_frames = 0
        self.avoided_pig_count = 0

        self.pending_candidate: Optional[PigObservation] = None
        self.pending_candidate_frames = 0
        self.pending_candidate_missing_frames = 0

        self.lane_change_status = 'UNKNOWN'
        self.lane_change_target = 'UNKNOWN'
        self.lane_holding_logged = False

        self.last_detection_time: Optional[float] = None
        self.last_status_publish_time = 0.0
        self.last_timeout_log_time = 0.0
        self.last_candidate_log_time = 0.0
        self.last_ignored_pig_count = -1
        self.last_published_target_side: Optional[str] = None

        self.all_pig_detection_sub = self.create_subscription(
            Float32MultiArray,
            '/level_a/pig_detections',
            self.all_pig_detection_callback,
            10,
        )
        self.lane_status_sub = self.create_subscription(
            String,
            '/level_a/lane_change_status',
            self.lane_status_callback,
            10,
        )

        self.target_side_pub = self.create_publisher(
            String,
            '/level_a/target_side',
            10,
        )
        self.mission_status_pub = self.create_publisher(
            String,
            '/level_a/mission_status',
            10,
        )

        self.timer = self.create_timer(0.05, self.mission_loop)

        self.publish_target_side('STOP')
        self.get_logger().info(
            'Level A multi-pig mission started: '
            f'first_trigger={self.avoid_start_distance:.2f} m, '
            f'next_trigger={self.next_pig_trigger_distance:.2f} m, '
            f'missing_frames={self.pig_missing_confirm_frames}'
        )

    def validate_parameters(self) -> None:
        if self.avoid_start_distance <= 0.0:
            raise ValueError('avoid_start_distance must be greater than zero')
        if self.next_pig_trigger_distance <= 0.0:
            raise ValueError(
                'next_pig_trigger_distance must be greater than zero'
            )
        if self.pig_missing_confirm_frames < 1:
            raise ValueError('pig_missing_confirm_frames must be at least one')
        if self.candidate_confirm_frames < 1:
            raise ValueError('candidate_confirm_frames must be at least one')
        if self.candidate_missing_tolerance_frames < 0:
            raise ValueError(
                'candidate_missing_tolerance_frames cannot be negative'
            )
        if self.detection_timeout_sec <= 0.0:
            raise ValueError('detection_timeout_sec must be greater than zero')
        if not 0.0 <= self.active_match_min_iou <= 1.0:
            raise ValueError('active_match_min_iou must be in [0, 1]')
        if self.active_match_max_center_ratio <= 0.0:
            raise ValueError(
                'active_match_max_center_ratio must be greater than zero'
            )
        if not (
            0.0
            < self.active_match_min_area_ratio
            <= 1.0
            <= self.active_match_max_area_ratio
        ):
            raise ValueError('active pig area-ratio limits are invalid')

    def all_pig_detection_callback(
        self,
        message: Float32MultiArray,
    ) -> None:
        try:
            detections = self.parse_all_pig_detections(message.data)
        except ValueError as error:
            self.get_logger().error(f'Invalid pig_detections message: {error}')
            return

        self.last_detection_time = time.monotonic()

        if self.state == MissionState.NORMAL:
            self.handle_normal(detections)
        elif self.state in (
            MissionState.FIRST_AVOIDING,
            MissionState.NEXT_AVOIDING,
        ):
            self.handle_active_avoidance(detections)
        elif self.state == MissionState.WAIT_NEXT_PIG:
            self.handle_wait_next_pig(detections)

    def parse_all_pig_detections(
        self,
        data,
    ) -> List[PigObservation]:
        if len(data) < 3:
            raise ValueError('message requires a three-value header')

        version = int(round(float(data[0])))
        record_size = int(round(float(data[1])))
        detection_count = int(round(float(data[2])))

        if version != self.ALL_DETECTIONS_FORMAT_VERSION:
            raise ValueError(f'unsupported format version: {version}')
        if record_size != self.ALL_DETECTIONS_RECORD_SIZE:
            raise ValueError(f'unexpected record size: {record_size}')
        if detection_count < 0:
            raise ValueError('detection count cannot be negative')

        expected_length = 3 + detection_count * record_size
        if len(data) != expected_length:
            raise ValueError(
                f'length={len(data)}, expected={expected_length}'
            )

        detections: List[PigObservation] = []
        for index in range(detection_count):
            start = 3 + index * record_size
            record = data[start:start + record_size]
            side = self.side_from_code(float(record[4]))

            detection = PigObservation(
                confidence=float(record[0]),
                forward_x_m=float(record[1]),
                lateral_y_m=float(record[2]),
                vertical_z_m=float(record[3]),
                side=side,
                optical_depth_m=float(record[5]),
                x1=float(record[6]),
                y1=float(record[7]),
                x2=float(record[8]),
                y2=float(record[9]),
            )

            if detection.bbox_area <= 0.0:
                continue
            detections.append(detection)

        return detections

    @staticmethod
    def side_from_code(side_code: float) -> str:
        if side_code > 0.5:
            return 'LEFT'
        if side_code < -0.5:
            return 'RIGHT'
        return 'CENTER'

    def handle_normal(self, detections: List[PigObservation]) -> None:
        candidate = self.track_first_candidate(detections)
        if candidate is None:
            return

        if not candidate.distance_valid:
            self.log_candidate_wait(
                '[NORMAL] tentative first pig has invalid distance; waiting'
            )
            return

        if candidate.side not in ('LEFT', 'RIGHT'):
            self.log_candidate_wait(
                '[NORMAL] tentative first pig side is CENTER/unknown; waiting'
            )
            return

        if candidate.forward_x_m > self.avoid_start_distance:
            self.log_candidate_wait(
                '[NORMAL] tracking tentative first pig: '
                f'count={len(detections)}, '
                f'distance={candidate.forward_x_m:.3f} m, '
                f'area={candidate.bbox_area:.0f}; '
                f'first trigger={self.avoid_start_distance:.2f} m'
            )
            return

        if self.pending_candidate_frames < self.candidate_confirm_frames:
            self.log_candidate_wait(
                '[NORMAL] tentative first pig is inside trigger range; '
                f'confirmation={self.pending_candidate_frames}/'
                f'{self.candidate_confirm_frames}'
            )
            return

        self.lock_active_pig(candidate, MissionState.FIRST_AVOIDING)

    def track_first_candidate(
        self,
        detections: List[PigObservation],
    ) -> Optional[PigObservation]:
        if self.pending_candidate is None:
            selected = self.choose_nearest_pig(detections)
            if selected is None:
                return None
            self.pending_candidate = selected
            self.pending_candidate_frames = 1
            self.pending_candidate_missing_frames = 0
            distance_text = (
                f'{selected.forward_x_m:.3f} m'
                if selected.distance_valid
                else 'invalid'
            )
            self.get_logger().info(
                '[NORMAL] selected tentative first pig: '
                f'side={selected.side}, '
                f'distance={distance_text}, '
                f'area={selected.bbox_area:.0f}'
            )
            return selected

        matched = self.match_active_pig(self.pending_candidate, detections)
        if matched is not None:
            self.pending_candidate = matched
            self.pending_candidate_missing_frames = 0
            self.pending_candidate_frames += 1
            return matched

        self.pending_candidate_missing_frames += 1
        self.pending_candidate_frames = 0

        if (
            self.pending_candidate_missing_frames
            <= self.candidate_missing_tolerance_frames
        ):
            self.log_candidate_wait(
                '[NORMAL] tentative first pig temporarily missing: '
                f'{self.pending_candidate_missing_frames}/'
                f'{self.candidate_missing_tolerance_frames}; '
                'other pigs are not allowed to replace it'
            )
            return None

        self.get_logger().info(
            '[NORMAL] tentative first pig exceeded missing tolerance; '
            'allowing a new nearest-pig decision'
        )
        self.reset_pending_candidate()
        selected = self.choose_nearest_pig(detections)
        if selected is None:
            return None
        self.pending_candidate = selected
        self.pending_candidate_frames = 1
        self.pending_candidate_missing_frames = 0
        return selected

    def handle_wait_next_pig(
        self,
        detections: List[PigObservation],
    ) -> None:
        candidate = self.choose_nearest_pig(detections)
        if candidate is None:
            self.reset_pending_candidate()
            return

        # Build confidence while the next pig is still far away. Therefore a
        # stable candidate can trigger immediately when it crosses 30 cm,
        # instead of waiting several additional frames at close range.
        candidate_confirmed = self.confirm_candidate(candidate)

        if not candidate.distance_valid:
            self.log_candidate_wait(
                '[WAIT_NEXT_PIG] selected largest bbox as nearest fallback: '
                f'count={len(detections)}, area={candidate.bbox_area:.0f}, '
                'distance invalid -> waiting'
            )
            return

        if candidate.side not in ('LEFT', 'RIGHT'):
            self.log_candidate_wait(
                '[WAIT_NEXT_PIG] nearest candidate side is CENTER/unknown; '
                'waiting'
            )
            return

        if candidate.forward_x_m > self.next_pig_trigger_distance:
            self.log_candidate_wait(
                '[WAIT_NEXT_PIG] nearest candidate: '
                f'count={len(detections)}, '
                f'distance={candidate.forward_x_m:.3f} m, '
                f'area={candidate.bbox_area:.0f} -> waiting'
            )
            return

        if not candidate_confirmed:
            return

        self.lock_active_pig(candidate, MissionState.NEXT_AVOIDING)

    def handle_active_avoidance(
        self,
        detections: List[PigObservation],
    ) -> None:
        if self.active_pig is None:
            self.get_logger().error(
                f'[{self.state.value}] active pig is unexpectedly empty'
            )
            self.transition_to_wait_next_pig()
            return

        matched_pig = self.match_active_pig(self.active_pig, detections)
        other_pig_count = len(detections)

        if matched_pig is not None:
            self.active_pig = matched_pig
            self.active_pig_missing_frames = 0
            other_pig_count = max(0, len(detections) - 1)

            if other_pig_count != self.last_ignored_pig_count:
                if other_pig_count > 0:
                    self.get_logger().info(
                        f'[{self.state.value}] active pig still visible; '
                        f'ignoring {other_pig_count} other pig(s) while locked'
                    )
                self.last_ignored_pig_count = other_pig_count
            return

        self.active_pig_missing_frames += 1
        missing = self.active_pig_missing_frames

        if missing == 1 or missing == self.pig_missing_confirm_frames:
            self.get_logger().info(
                f'[{self.state.value}] active pig missing '
                f'{missing}/{self.pig_missing_confirm_frames} frame(s); '
                f'{other_pig_count} unmatched pig(s) ignored'
            )

        if missing < self.pig_missing_confirm_frames:
            return

        if not self.lane_is_holding_current_target():
            if missing == self.pig_missing_confirm_frames:
                self.get_logger().warning(
                    f'[{self.state.value}] active pig is visually cleared, '
                    f'but lane controller has not reached {self.target_side}; '
                    'keeping the same target and delaying active-pig release'
                )
            return

        cleared_state = self.state
        self.avoided_pig_count += 1
        self.get_logger().info(
            f'[{cleared_state.value}] active pig cleared after '
            f'{missing} consecutive missing frames; '
            f'avoided_count={self.avoided_pig_count}'
        )
        self.transition_to_wait_next_pig()

    def lock_active_pig(
        self,
        candidate: PigObservation,
        next_state: MissionState,
    ) -> None:
        self.active_pig = candidate
        self.active_pig_missing_frames = 0
        self.target_side = self.opposite_side(candidate.side)
        self.state = next_state
        self.lane_holding_logged = False
        self.last_ignored_pig_count = -1
        self.reset_pending_candidate()

        trigger_name = (
            'first-pig existing trigger'
            if next_state == MissionState.FIRST_AVOIDING
            else 'next-pig 30 cm trigger'
        )
        self.get_logger().info(
            f'[{next_state.value}] locked active pig: '
            f'side={candidate.side}, target={self.target_side}, '
            f'distance={candidate.forward_x_m:.3f} m, '
            f'area={candidate.bbox_area:.0f}, rule={trigger_name}'
        )

    def transition_to_wait_next_pig(self) -> None:
        held_safe_side = self.target_side
        self.state = MissionState.WAIT_NEXT_PIG
        self.active_pig = None
        self.active_pig_missing_frames = 0
        self.lane_holding_logged = False
        self.last_ignored_pig_count = -1
        self.reset_pending_candidate()
        self.get_logger().info(
            '[WAIT_NEXT_PIG] active pig released; '
            f'holding safe side {held_safe_side} until the next pig reaches '
            'its trigger distance'
        )

    def confirm_candidate(self, candidate: PigObservation) -> bool:
        if self.pending_candidate is None:
            self.pending_candidate = candidate
            self.pending_candidate_frames = 1
            self.pending_candidate_missing_frames = 0
        else:
            matched = self.match_active_pig(
                self.pending_candidate,
                [candidate],
            )
            if matched is None:
                self.pending_candidate = candidate
                self.pending_candidate_frames = 1
                self.pending_candidate_missing_frames = 0
            else:
                self.pending_candidate = candidate
                self.pending_candidate_frames += 1
                self.pending_candidate_missing_frames = 0

        if self.pending_candidate_frames < self.candidate_confirm_frames:
            self.log_candidate_wait(
                f'[{self.state.value}] candidate confirmation '
                f'{self.pending_candidate_frames}/'
                f'{self.candidate_confirm_frames}'
            )
            return False

        return True

    def reset_pending_candidate(self) -> None:
        self.pending_candidate = None
        self.pending_candidate_frames = 0
        self.pending_candidate_missing_frames = 0

    @staticmethod
    def choose_nearest_pig(
        detections: List[PigObservation],
    ) -> Optional[PigObservation]:
        if not detections:
            return None

        distance_candidates = [
            item for item in detections if item.distance_valid
        ]
        if distance_candidates:
            return min(
                distance_candidates,
                key=lambda item: (
                    item.forward_x_m,
                    -item.bbox_area,
                    -item.confidence,
                ),
            )

        # 尚未鎖定豬時，距離無效的框不能成為candidate。這可避免
        # 高信心誤判框長時間占用pending candidate。距離暫時無效的
        # bbox仍保留在detections中，供已鎖定的active pig做短暫追蹤。
        return None

    def match_active_pig(
        self,
        active: PigObservation,
        detections: List[PigObservation],
    ) -> Optional[PigObservation]:
        matches = []

        for detection in detections:
            iou = self.bbox_iou(active.bbox, detection.bbox)
            center_ratio = self.normalized_center_distance(active, detection)
            area_ratio = self.bbox_area_ratio(active, detection)
            area_continuous = (
                self.active_match_min_area_ratio
                <= area_ratio
                <= self.active_match_max_area_ratio
            )

            accepted = (
                iou >= self.active_match_min_iou
                or (
                    center_ratio <= self.active_match_max_center_ratio
                    and area_continuous
                )
            )
            if not accepted:
                continue

            area_penalty = abs(math.log(max(area_ratio, 1.0e-6)))
            score = (-iou, center_ratio, area_penalty)
            matches.append((score, detection))

        if not matches:
            return None

        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    @staticmethod
    def bbox_iou(
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> float:
        first_x1, first_y1, first_x2, first_y2 = first
        second_x1, second_y1, second_x2, second_y2 = second

        intersection_width = max(
            0.0,
            min(first_x2, second_x2) - max(first_x1, second_x1),
        )
        intersection_height = max(
            0.0,
            min(first_y2, second_y2) - max(first_y1, second_y1),
        )
        intersection = intersection_width * intersection_height

        first_area = max(0.0, first_x2 - first_x1) * max(
            0.0,
            first_y2 - first_y1,
        )
        second_area = max(0.0, second_x2 - second_x1) * max(
            0.0,
            second_y2 - second_y1,
        )
        union = first_area + second_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    @staticmethod
    def normalized_center_distance(
        first: PigObservation,
        second: PigObservation,
    ) -> float:
        first_center_x = (first.x1 + first.x2) / 2.0
        first_center_y = (first.y1 + first.y2) / 2.0
        second_center_x = (second.x1 + second.x2) / 2.0
        second_center_y = (second.y1 + second.y2) / 2.0
        center_distance = math.hypot(
            first_center_x - second_center_x,
            first_center_y - second_center_y,
        )
        reference_diagonal = max(
            math.hypot(first.bbox_width, first.bbox_height),
            1.0,
        )
        return center_distance / reference_diagonal

    @staticmethod
    def bbox_area_ratio(
        first: PigObservation,
        second: PigObservation,
    ) -> float:
        if first.bbox_area <= 0.0:
            return math.inf
        return second.bbox_area / first.bbox_area

    @staticmethod
    def opposite_side(pig_side: str) -> str:
        if pig_side == 'LEFT':
            return 'RIGHT'
        if pig_side == 'RIGHT':
            return 'LEFT'
        return 'STOP'

    def lane_status_callback(self, message: String) -> None:
        # The existing lane controller publishes a descriptive string such as
        # "target=RIGHT state=HOLDING ...", not only "HOLDING".
        parsed_state = 'UNKNOWN'
        parsed_target = 'UNKNOWN'
        for token in message.data.split():
            if token.startswith('state='):
                parsed_state = token.split('=', 1)[1]
            elif token.startswith('target='):
                parsed_target = token.split('=', 1)[1]

        if message.data in ('MOVING', 'HOLDING'):
            parsed_state = message.data

        self.lane_change_status = parsed_state
        self.lane_change_target = parsed_target

        if (
            self.lane_is_holding_current_target()
            and self.state in (
                MissionState.FIRST_AVOIDING,
                MissionState.NEXT_AVOIDING,
            )
            and not self.lane_holding_logged
        ):
            self.lane_holding_logged = True
            self.get_logger().info(
                f'[{self.state.value}] lane controller is HOLDING '
                f'{self.target_side}; continuing until active pig clears'
            )

    def lane_is_holding_current_target(self) -> bool:
        target_matches = (
            self.lane_change_target == self.target_side
            or self.lane_change_target == 'UNKNOWN'
        )
        return self.lane_change_status == 'HOLDING' and target_matches

    def mission_loop(self) -> None:
        now = time.monotonic()
        detection_stream_fresh = (
            self.last_detection_time is not None
            and now - self.last_detection_time <= self.detection_timeout_sec
        )

        if not detection_stream_fresh:
            # 持續發布heartbeat，讓較晚啟動或重新連線的controller
            # 仍能收到目前的fail-safe STOP。
            self.publish_target_side('STOP', force=True)
            if now - self.last_timeout_log_time >= 2.0:
                self.last_timeout_log_time = now
                self.get_logger().warning(
                    '[MISSION] pig_detections stream unavailable; '
                    'target_side forced to STOP. Active pig is not cleared.'
                )
        else:
            # controller會忽略相同target，因此重發不會清除其
            # settle_count，卻能避免啟動時DDS尚未完成配對而漏接。
            self.publish_target_side(self.target_side, force=True)

        if now - self.last_status_publish_time >= 0.5:
            self.last_status_publish_time = now
            self.publish_mission_status(detection_stream_fresh)

    def publish_target_side(self, side: str, force: bool = False) -> None:
        # 非force呼叫仍可去重；mission_loop使用force=True作為heartbeat。
        if not force and side == self.last_published_target_side:
            return

        message = String()
        message.data = side
        self.target_side_pub.publish(message)
        self.last_published_target_side = side

    def publish_mission_status(self, detection_stream_fresh: bool) -> None:
        active_text = 'none'
        if self.active_pig is not None:
            distance_text = (
                f'{self.active_pig.forward_x_m:.3f}'
                if self.active_pig.distance_valid
                else 'invalid'
            )
            active_text = (
                f'side={self.active_pig.side},distance={distance_text},'
                f'area={self.active_pig.bbox_area:.0f},'
                f'missing={self.active_pig_missing_frames}/'
                f'{self.pig_missing_confirm_frames}'
            )

        message = String()
        message.data = (
            f'state={self.state.value} '
            f'target={self.target_side} '
            f'lane_state={self.lane_change_status} '
            f'lane_target={self.lane_change_target} '
            f'detection_fresh={detection_stream_fresh} '
            f'avoided_count={self.avoided_pig_count} '
            f'active={active_text}'
        )
        self.mission_status_pub.publish(message)

    def log_candidate_wait(self, text: str) -> None:
        now = time.monotonic()
        if now - self.last_candidate_log_time < 1.5:
            return
        self.last_candidate_log_time = now
        self.get_logger().info(text)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LevelAMission()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_target_side('STOP', force=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
