import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the model
# Import the custom layer - adjust the import path as needed
# from utils.model import LatentProj, SelfAttentionBlock, CrossAttentionBlock, PeptideProj  # Update this import based on where your LatentProj is defined
# with tf.keras.utils.custom_object_scope({'LatentProj': LatentProj,'PeptideProj': PeptideProj, 'SelfAttentionBlock': SelfAttentionBlock, 'CrossAttentionBlock': CrossAttentionBlock}):
#     model = load_model('runs/run_20250603-111633/best_weights.h5')

from utils.model_archive import AttentionLayer, PositionalEncoding, AnchorPositionExtractor
# Import any additional classes/functions used by Lambda layers
from utils.model_archive import *  # Import all potential dependencies


import uuid

def wrap_layer(layer_class):
    def fn(**config):
        config.pop('trainable', None)
        config.pop('dtype', None)
        # assign a unique name to avoid duplicates
        config['name'] = f"{layer_class.__name__.lower()}_{uuid.uuid4().hex[:8]}"
        return layer_class.from_config(config)
    return fn


# Load the model with wrapped custom objects
model = load_model(
    'model_output/peptide_mhc_cross_attention_model.h5',
    custom_objects={
        'AttentionLayer': wrap_layer(AttentionLayer),
        'RotaryPositionalEncoding': wrap_layer(RotaryPositionalEncoding),
        'AnchorPositionExtractor': wrap_layer(AnchorPositionExtractor),
    }
)
# Display and save the model's architecture

# Display model summary
model.summary()

# Create better visualization as SVG with cleaner layout
tf.keras.utils.plot_model(
    model,
    to_file='model_output/model_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # Top to bottom layout
    dpi=200,       # Higher resolution
    expand_nested=True,  # Expand nested models to show all layers
    show_layer_activations=True  # Show activation functions
)