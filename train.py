import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback

# Enable mixed precision training
mixed_precision.set_global_policy('mixed_float16')

# Memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configuration
CONFIG = {
    'IMG_SIZE': 224,  # Reduced from original size
    'BATCH_SIZE': 16,  # Adjusted for memory
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-4,
    'NUM_CLASSES': 4,
    'GRAD_ACCUMULATION_STEPS': 4  # Gradient accumulation for effective larger batch size
}

def create_model():
    # Using B7 instead of B2 - much heavier but still powerful
    base_model = EfficientNetB7(
        weights='imagenet',
        include_top=False,
        input_shape=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(CONFIG['NUM_CLASSES'], activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def create_data_pipeline(data_dir, is_training=True):
    def preprocess(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Create dataset from directory
    dataset = tf.data.Dataset.from_tensor_slices((
        tf.data.Dataset.list_files(os.path.join(data_dir, '*/*.jpg'), shuffle=is_training),
        [0 if 'glioma' in x.decode() else 1 if 'meningioma' in x.decode() else 2 if 'pituitary' in x.decode() else 3 
         for x in tf.data.Dataset.list_files(os.path.join(data_dir, '*/*.jpg'), shuffle=is_training)]
    ))

    # Apply preprocessing
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(
            lambda x, y: (tf.image.random_brightness(x, 0.2), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(CONFIG['BATCH_SIZE'])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

class GradientAccumulationModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_accumulation_steps = CONFIG['GRAD_ACCUMULATION_STEPS']
        
    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        
        # Split batch into smaller chunks
        x_chunks = tf.split(x, self.grad_accumulation_steps)
        y_chunks = tf.split(y, self.grad_accumulation_steps)
        
        accumulated_gradients = None
        accumulated_loss = 0
        accumulated_metrics = {}
        
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            with tf.GradientTape() as tape:
                y_pred = self(x_chunk, training=True)
                loss = self.compiled_loss(y_chunk, y_pred)
                scaled_loss = loss / self.grad_accumulation_steps
            
            gradients = tape.gradient(scaled_loss, self.trainable_variables)
            
            if accumulated_gradients is None:
                accumulated_gradients = gradients
            else:
                accumulated_gradients = [accu_grad + grad for accu_grad, grad in zip(accumulated_gradients, gradients)]
            
            accumulated_loss += loss
            
            # Update metrics
            self.compiled_metrics.update_state(y_chunk, y_pred)
            for metric in self.metrics:
                if metric.name not in accumulated_metrics:
                    accumulated_metrics[metric.name] = 0
                accumulated_metrics[metric.name] += metric.result()
        
        # Apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))
        
        # Average the metrics
        accumulated_loss /= self.grad_accumulation_steps
        for metric_name in accumulated_metrics:
            accumulated_metrics[metric_name] /= self.grad_accumulation_steps
        
        return {'loss': accumulated_loss, **accumulated_metrics}

def train():
    # Initialize wandb
    wandb.init(project='brain-tumor-detection-optimized', config=CONFIG)
    
    # Create model with gradient accumulation
    model = GradientAccumulationModel(create_model())
    
    # Compile with mixed precision optimizer
    optimizer = Adam(learning_rate=CONFIG['LEARNING_RATE'])
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create datasets
    train_dataset = create_data_pipeline('data/train', is_training=True)
    val_dataset = create_data_pipeline('data/val', is_training=False)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        WandbCallback()
    ]
    
    # Train
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['EPOCHS'],
        callbacks=callbacks
    )
    
    # Fine-tune
    print("Fine-tuning the model...")
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE'] / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks
    )
    
    wandb.finish()

if __name__ == '__main__':
    train()    