# saving path
FOLDER_NAME = 'pred7'
model_path = f'checkpoints/{FOLDER_NAME}'
train_path = f'{model_path}/train'
gifs_path = f'{model_path}/gifs'

# predictor settings
generator_path = f'checkpoints/wgan_3000'
N_GEN_SAMPLE = 4  # how many samples do you want to generate
N_AGENTS = 3

# save training data
SUMMARY_WINDOW = 32  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 500  # how many episodes before saving a gif

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 16  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 1  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# training parameters
MAX_EPISODE_STEP = 28
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 64
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 18  # how many threads does your CPU have

# network parameters
NODE_INPUT_DIM = 9
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = 960  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = True  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 3  # 0 unless you want to collect data using GPUs

USE_WANDB = False

COMMS_RANGE = 32.0    
MAX_TRAVEL_COEF = 0
TOTAL_TRAVEL_COEF = 0

INTENT_HORIZON = 3 

MEET_MODE = 'pred'          
MEET_SYNC_TOL_M = 18.0
MEET_RADIUS_FRAC = 0.45
MEET_BUFFER_ALPHA = 0.4
MEET_BUFFER_BETA  = 6.0     
MEET_LATE_TOL     = 12      
R_MEET_SUCCESS    = 0    
R_MEET_LATE       = 0    

# --- Rendezvous Protocol Parameters ---
RENDEZVOUS_INFO_RADIUS_M = 8.0 

MEET_BUFFER_ALPHA = 0.2     
MEET_BUFFER_BETA  = 4.0     

MEET_RADIUS_FRAC = 0.45     
MEET_LATE_TOL     = 12     

R_MEET_SUCCESS    = 0     
R_MEET_LATE       = 0    