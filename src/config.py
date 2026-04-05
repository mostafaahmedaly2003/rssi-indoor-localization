# All project constants in one place
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Floor map settings
FLOOR_WIDTH = 6   # meters
FLOOR_HEIGHT = 6  # meters

# 9 reference locations on a 3x3 grid
LOCATIONS = {
    0: (1, 1), 1: (3, 1), 2: (5, 1),
    3: (1, 3), 4: (3, 3), 5: (5, 3),
    6: (1, 5), 7: (3, 5), 8: (5, 5),
}

LOCATION_NAMES = {
    0: "Entrance", 1: "Corridor A", 2: "Corridor B",
    3: "Lab West",  4: "Center Hub", 5: "Lab East",
    6: "Office A",  7: "Meeting Room", 8: "Office B",
}

# 6 Access Point positions (walls/ceiling)
AP_POSITIONS = [(0,0), (6,0), (0,6), (6,6), (3,0), (0,3)]

# Dataset
SAMPLES_PER_LOCATION = 100
NOISE_STD = 3.0       # dBm noise (realistic WiFi)
TX_POWER = -30        # dBm
PATH_LOSS_EXP = 2.5

# Model output paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
