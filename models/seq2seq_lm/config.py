from transformers import AutoConfig
from .argparser import get_args

_args = get_args()

#
GPUS = -1
ACCELERATOR = "dp"

#
HL_TOKEN = "[HL]"
MODEL_CONFIG = AutoConfig.from_pretrained(_args.model_name_or_path)

# MAX_INPUT_LENGTH = 1024  # max 1024
# MAX_OUTPUT_LENGTH = 128
# MAX_INPUT_LENGTH = 512  # max 1024
# MAX_QUESTION_LENGTH = 32
