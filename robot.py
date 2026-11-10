
import pygame
import numpy as np
import math

from config import (
    TRACK_WIDTH,
    HEIGHT,
    SENSOR_COUNT,
    SENSOR_LENGTH,
    SENSOR_SPREAD,
    SENSOR_SAMPLE_STEP,

    ROBOT_RADIUS,
    MAX_TURN_RATE,
    MIN_SPEED,
    MAX_SPEED,

    MAX_LOST_LINE_FRAMES,
    MAX_IDLE_FRAMES,
    MIN_IDLE_DISTANCE,

    FITNESS_PROGRESS_WEIGHT,
    FITNESS_SURVIVAL_WEIGHT,
    FITNESS_LINE_WEIGHT,
    FITNESS_STUCK_PENALTY,

    GREEN,
    BLUE,
    RED,
    GRAY
)

from neural_network import NeuralNetwork


class Robot:

    # ==========================================
    # 1. INITIALIZATION
    # ==========================================

    def __init__(self, x, y, angle, brain=None, track_path=None):

        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)

        self.alive = True
        self.fitness = 0.0

        # Sensor data
        self.sensors = [0.0] * SENSOR_COUNT
        self.sensor_coords = []

        # Lost line detection
        self.lost_line_frames = 0

        # Anti-stuck detection
        self.idle_frames = 0
        self.last_check_pos = (self.x, self.y)

        # Movement telemetry
        self.current_speed = 0.0
        self.current_turn = 0.0
        self.throttle = 0.0

        # Neural network
        if (
            brain is not None
            and brain.input_size == SENSOR_COUNT
        ):
            self.brain = brain
        else:
            self.brain = NeuralNetwork(
                SENSOR_COUNT,
                8,
                2
            )

        # Track progress
        self.track_path = (
            list(track_path)
            if track_path is not None
            else []
        )

        self.progress_index = 0
        self.lap_count = 0
        self.progress_distance = 0.0

        self.path_cumulative = []
        self.path_total_length = 0.0

        self._initialize_track_path()


    # ==========================================
    # 2. TRACK PATH INITIALIZATION
    # ==========================================

    def _initialize_track_path(self):

        if len(self.track_path) < 2:
            return

        self.path_cumulative = [0.0]

        for i in range(1, len(self.track_path)):

            x1, y1 = self.track_path[i - 1]
            x2, y2 = self.track_path[i]

            distance = math.hypot(
                x2 - x1,
                y2 - y1
            )

            self.path_cumulative.append(
                self.path_cumulative[-1] + distance
            )

        # Closing segment for closed tracks
        first = self.track_path[0]
        last = self.track_path[-1]

        closing_distance = math.hypot(
            first[0] - last[0],
            first[1] - last[1]
        )

        self.path_total_length = (
            self.path_cumulative[-1] + closing_distance
        )


    # ==========================================
    # 3. COLOR DETECTION
    # ==========================================

    def _is_white(self, surface, x, y):

        if not (
            0 <= x < TRACK_WIDTH
            and 0 <= y < HEIGHT
        ):
            return False

        color = surface.get_at((int(x), int(y)))

        return (
            color[0] > 200
            and color[1] > 200
            and color[2] > 200
        )


    # ==========================================
    # 4. ROBOT COLLISION
    # ==========================================

    def _is_on_track(self, surface):

        # Check center
        if self._is_white(surface, self.x, self.y):
            return True

        # Check surrounding body area
        check_radius = ROBOT_RADIUS * 0.65

        for i in range(8):

            angle = math.radians(i * 45)

            px = self.x + math.cos(angle) * check_radius
            py = self.y + math.sin(angle) * check_radius

            if self._is_white(surface, px, py):
                return True

        return False


    # ==========================================
    # 5. SENSOR SYSTEM
    # ==========================================

    def read_sensors(self, track_surface):

        self.sensor_coords = []

        start_angle = (
            self.angle - SENSOR_SPREAD / 2
        )

        step_angle = (
            SENSOR_SPREAD / (SENSOR_COUNT - 1)
        )

        for i in range(SENSOR_COUNT):

            sensor_angle = (
                start_angle + i * step_angle
            )

            dx = math.cos(sensor_angle)
            dy = math.sin(sensor_angle)

            detected = False
            sensor_value = 0.0

            endpoint = (
                int(self.x + dx * SENSOR_LENGTH),
                int(self.y + dy * SENSOR_LENGTH)
            )

            # Scan from robot body toward sensor endpoint
            for distance in range(
                ROBOT_RADIUS,
                SENSOR_LENGTH + 1,
                SENSOR_SAMPLE_STEP
            ):

                sx = int(self.x + dx * distance)
                sy = int(self.y + dy * distance)

                if self._is_white(track_surface, sx, sy):

                    # Closer detection produces higher activation
                    sensor_value = 1.0 - (
                        (distance - ROBOT_RADIUS)
                        /
                        (
                            SENSOR_LENGTH
                            - ROBOT_RADIUS
                            + SENSOR_SAMPLE_STEP
                        )
                    )

                    sensor_value = max(
                        0.0,
                        min(1.0, sensor_value)
                    )

                    self.sensor_coords.append((sx, sy))
                    detected = True
                    break

            if not detected:
                self.sensor_coords.append(endpoint)

            self.sensors[i] = sensor_value


    # ==========================================
    # 6. TRACK PROGRESS
    # ==========================================

    def _update_progress(self, movement_distance):

        if not self.track_path:
            return 0.0

        path_size = len(self.track_path)

        # Search around previous progress position.
        # This avoids jumping to unrelated nearby segments.
        candidates = [
            (self.progress_index + offset) % path_size
            for offset in range(-2, 31)
        ]

        best_index = self.progress_index
        best_distance = float("inf")

        for index in candidates:

            px, py = self.track_path[index]

            distance = math.hypot(
                self.x - px,
                self.y - py
            )

            if distance < best_distance:
                best_distance = distance
                best_index = index

        previous_index = self.progress_index

        # Detect crossing from the end to the beginning
        if (
            previous_index > path_size * 0.8
            and best_index < path_size * 0.2
        ):
            self.lap_count += 1

        self.progress_index = best_index

        raw_progress = (
            self.lap_count * self.path_total_length
            + self.path_cumulative[best_index]
        )

        # Never reward backward movement.
        progress_delta = max(
            0.0,
            raw_progress - self.progress_distance
        )

        # Prevent unrealistic progress jumps.
        max_progress = movement_distance * 1.5 + 2.0

        progress_delta = min(
            progress_delta,
            max_progress
        )

        self.progress_distance += progress_delta

        return progress_delta


    # ==========================================
    # 7. MAIN ROBOT UPDATE
    # ==========================================

    def update(self, track_surface):

        if not self.alive:
            return

        # Check current position
        if not self._is_on_track(track_surface):
            self.alive = False
            return

        # Read sensor information
        self.read_sensors(track_surface)

        # Lost line tolerance
        if max(self.sensors) < 0.01:

            self.lost_line_frames += 1

            if self.lost_line_frames > MAX_LOST_LINE_FRAMES:
                self.alive = False
                return

        else:
            self.lost_line_frames = 0

        # ======================================
        # Neural network inference
        # ======================================

        inputs = np.asarray(
            self.sensors,
            dtype=np.float64
        )

        output = self.brain.forward(inputs)

        if not np.all(np.isfinite(output)):
            self.alive = False
            return

        # Output 0: Steering
        # Output 1: Throttle

        self.current_turn = (
            float(output[0]) * MAX_TURN_RATE
        )

        self.throttle = (
            float(output[1]) + 1.0
        ) / 2.0

        self.throttle = max(
            0.0,
            min(1.0, self.throttle)
        )

        # ======================================
        # Movement physics
        # ======================================

        self.angle += self.current_turn

        self.current_speed = (
            MIN_SPEED
            + self.throttle * (MAX_SPEED - MIN_SPEED)
        )

        old_x = self.x
        old_y = self.y

        self.x += (
            math.cos(self.angle) * self.current_speed
        )

        self.y += (
            math.sin(self.angle) * self.current_speed
        )

        movement_distance = math.hypot(
            self.x - old_x,
            self.y - old_y
        )

        # ======================================
        # Track progress fitness
        # ======================================

        progress_delta = self._update_progress(
            movement_distance
        )

        self.fitness += (
            progress_delta * FITNESS_PROGRESS_WEIGHT
        )

        # ======================================
        # Survival fitness
        # ======================================

        self.fitness += FITNESS_SURVIVAL_WEIGHT

        # ======================================
        # Line following fitness
        # ======================================

        center_index = SENSOR_COUNT // 2

        if self.sensors[center_index] > 0.25:

            self.fitness += FITNESS_LINE_WEIGHT

        # ======================================
        # Anti-stuck detection
        # ======================================

        self.idle_frames += 1

        if self.idle_frames >= MAX_IDLE_FRAMES:

            distance = math.hypot(
                self.x - self.last_check_pos[0],
                self.y - self.last_check_pos[1]
            )

            if distance < MIN_IDLE_DISTANCE:

                self.fitness -= FITNESS_STUCK_PENALTY
                self.alive = False
                return

            self.last_check_pos = (
                self.x,
                self.y
            )

            self.idle_frames = 0


    # ==========================================
    # 8. ROBOT VISUALIZATION
    # ==========================================

    def draw(self, screen, is_best=False):

        if not self.alive:
            return

        color = GREEN if is_best else BLUE

        center = (
            int(self.x),
            int(self.y)
        )

        pygame.draw.circle(
            screen,
            color,
            center,
            ROBOT_RADIUS
        )

        # Draw heading direction
        heading_end = (
            int(
                self.x
                + math.cos(self.angle) * ROBOT_RADIUS * 1.8
            ),
            int(
                self.y
                + math.sin(self.angle) * ROBOT_RADIUS * 1.8
            )
        )

        pygame.draw.line(
            screen,
            (255, 255, 255),
            center,
            heading_end,
            2
        )

        # Show sensor rays for best robot
        if is_best:

            for i, (sx, sy) in enumerate(self.sensor_coords):

                sensor_color = (
                    GREEN
                    if self.sensors[i] > 0.25
                    else RED
                )

                pygame.draw.line(
                    screen,
                    GRAY,
                    center,
                    (sx, sy),
                    1
                )

                pygame.draw.circle(
                    screen,
                    sensor_color,
                    (sx, sy),
                    3
                )
