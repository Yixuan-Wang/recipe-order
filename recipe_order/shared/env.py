from __future__ import annotations

from dotenv import load_dotenv
from os import environ

load_dotenv()

PRETRAINED_MODEL_PERM = "bert-base-uncased"
PRETRAINED_MODEL_GRAPH_ISOMORPHISM = "bert-base-uncased"
PRETRAINED_MODEL_POINTER = "bert-base-uncased"
PATH_DATA_MMRES = environ["PATH_DATA_MMRES"]
INSTRUCTION_IS_ORDERED = True
CUDA_DEVICE = environ["CUDA_DEVICE"]
CUDA_DEVICE_SUBSIDARY = environ["CUDA_DEVICE_SUBSIDARY"]
# GET_PRETRAINED_MODEL_POINTER_HIDDEN_SIZE = lambda model: model.config.