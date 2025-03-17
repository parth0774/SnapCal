import os
import logging
from datetime import datetime
import tensorflow as tf
from keras.src import regularizers
from keras.src import MobileNetV2
from keras.src import Model
from keras.src import Dense, Dropout, GlobalAveragePooling2D
from keras.src import ImageDataGenerator
from keras.src import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.src import SGD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU devices: {len(gpus)}")
    except RuntimeError as e:
        logger.error(f"Error setting GPU memory growth: {str(e)}")

def create_model(img_width=299, img_height=299, n_classes=101):
    """
    Create and compile the food classification model based on MobileNetV2.
    
    Returns:
        Compiled Keras model
    """
    # Create base model
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_height, img_width, 3)
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(n_classes, kernel_regularizer=regularizers.l2(0.001), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    return model

def create_data_generators(img_width=299, img_height=299, batch_size=32):
    """
    Create training and validation data generators with augmentation.
    
    Returns:
        train_generator, validation_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        'food-101/train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = test_datagen.flow_from_directory(
        'food-101/test',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

def train_model(epochs=15, batch_size=32, fine_tune=True):
    """
    Train the model with the option to fine-tune.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        fine_tune: Whether to fine-tune the base model after initial training
        
    Returns:
        Trained model and training history
    """
    img_width, img_height = 299, 299
    n_classes = 101
    
    # Create folders for logs and models
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create model and data generators
    model = create_model(img_width, img_height, n_classes)
    train_generator, validation_generator = create_data_generators(img_width, img_height, batch_size)
    
    # Calculate steps per epoch
    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)
    steps_per_epoch = nb_train_samples // batch_size
    validation_steps = nb_validation_samples // batch_size
    
    logger.info(f"Training samples: {nb_train_samples}, Validation samples: {nb_validation_samples}")
    logger.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f'models/best_model_{timestamp}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(f'logs/training_log_{timestamp}.csv'),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        ),
        TensorBoard(log_dir=f'logs/tensorboard_{timestamp}')
    ]
    
    # Initial training phase
    logger.info("Starting initial training phase...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs // 2 if fine_tune else epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning phase (optional)
    if fine_tune:
        logger.info("Starting fine-tuning phase...")
        # Unfreeze some layers of the base model
        for layer in model.layers[:]:
            layer.trainable = True
            
        # Recompile with lower learning rate
        model.compile(
            optimizer=SGD(learning_rate=0.0001, momentum=0.9),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
        
        # Continue training
        fine_tune_history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs // 2,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for k in fine_tune_history.history:
            history.history[k].extend(fine_tune_history.history[k])
    
    # Save final model
    model.save(f'models/final_model_{timestamp}.h5')
    logger.info(f"Model saved as models/final_model_{timestamp}.h5")
    
    return model, history

def main():
    """Main function to execute model training"""
    logger.info("Starting model training...")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Train model with fine-tuning
    train_model(epochs=15, batch_size=32, fine_tune=True)
    
    logger.info("Model training completed!")

if __name__ == "__main__":
    main()
