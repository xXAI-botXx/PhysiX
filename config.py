import os
import yaml

# Load project configuration from YAML file, path can be overridden via PROJECT_CONFIG env var
DEFAULT_CONFIG_FILE = os.environ.get('PROJECT_CONFIG', os.path.join(os.path.dirname(__file__), 'project_config.yaml'))

with open(DEFAULT_CONFIG_FILE, 'r') as f:
    _CONFIG = yaml.safe_load(f)

# Environment variable overrides for each config key
RAW_DATA_PATH = os.environ.get('RAW_DATA_PATH', _CONFIG.get('raw_data_path'))
CLEANED_DATA_PATH = os.environ.get('CLEANED_DATA_PATH', _CONFIG.get('cleaned_data_path'))
NORMALIZED_DATA_PATH = os.environ.get('NORMALIZED_DATA_PATH', _CONFIG.get('normalized_data_path'))
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', _CONFIG.get('checkpoint_dir'))
EMBEDDINGS_DIR = os.environ.get('EMBEDDINGS_DIR', _CONFIG.get('embeddings_dir'))
RESULTS_DIR = os.environ.get('RESULTS_DIR', _CONFIG.get('results_dir'))
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', _CONFIG.get('tokenizer_path'))
CACHE_DIR = os.environ.get('CACHE_DIR', _CONFIG.get('cache_dir'))

# Export all config as a dict
CONFIG = {
    'raw_data_path': RAW_DATA_PATH,
    'cleaned_data_path': CLEANED_DATA_PATH,
    'normalized_data_path': NORMALIZED_DATA_PATH,
    'checkpoint_dir': CHECKPOINT_DIR,
    'embeddings_dir': EMBEDDINGS_DIR,
    'results_dir': RESULTS_DIR,
    'tokenizer_path': TOKENIZER_PATH,
    'cache_dir': CACHE_DIR,
} 