import math

# ==========================================
# 1. SIMULATION WINDOW
# ==========================================

TRACK_WIDTH = 800
PANEL_WIDTH = 300

WIDTH = TRACK_WIDTH + PANEL_WIDTH
HEIGHT = 600

FPS = 60


# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================

SENSOR_COUNT = 11
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2

# Output 0 = Steering
# Output 1 = Throttle


# ==========================================
# 3. ROBOT SENSOR CONFIGURATION
# ==========================================

SENSOR_LENGTH = 20
SENSOR_SPREAD = math.radians(140)

ROBOT_RADIUS = 7

# Toleransi ketika robot kehilangan garis
MAX_LOST_LINE_FRAMES = 10

# Anti-stuck detection
MAX_IDLE_FRAMES = 60
MIN_IDLE_DISTANCE = 25



# Robot movement
MAX_TURN_RATE = math.radians(12)

MIN_SPEED = 2.2
MAX_SPEED = 4.7

# Sensor sampling
SENSOR_SAMPLE_STEP = 2



# ==========================================
# 4. EVOLUTION CONFIGURATION
# ==========================================

POPULATION_SIZE = 30

ELITE_COUNT = 5
RANDOM_IMMIGRANTS = 4

MUTATION_RATE = 0.25
MUTATION_SCALE = 0.5


# ==========================================
# 5. FITNESS CONFIGURATION
# ==========================================

FITNESS_PROGRESS_WEIGHT = 5.0
FITNESS_SURVIVAL_WEIGHT = 0.2
FITNESS_LINE_WEIGHT = 2.0

FITNESS_STUCK_PENALTY = 5.0


# ==========================================
# 6. SIMULATION SPEED
# ==========================================

SIMULATION_SPEEDS = [1, 2, 5]


# ==========================================
# 7. FILE CONFIGURATION
# ==========================================

BEST_BRAIN_FILE = "best_brain.npz"


# ==========================================
# 8. COLORS (RGB)
# ==========================================

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

GREEN = (50, 220, 100)
RED = (255, 70, 70)
BLUE = (60, 160, 255)
YELLOW = (255, 200, 50)

GRAY = (140, 145, 155)

PANEL_BG = (32, 34, 40)
CARD_BG = (45, 48, 56)
