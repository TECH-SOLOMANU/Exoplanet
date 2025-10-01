import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LightCurveCNN:
    """CNN model for exoplanet detection from light curve data"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.input_length = 2048  # Standard light curve length
        self.classes = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build 1D CNN architecture for light curve classification"""
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First convolutional block
            layers.Conv1D(32, 7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Fourth convolutional block
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(len(self.classes), activation='softmax')
        ])
        
        return model
    
    def preprocess_light_curve(self, flux_data: np.ndarray) -> np.ndarray:
        """Preprocess light curve data for CNN input"""
        
        # Normalize flux
        flux_normalized = (flux_data - np.median(flux_data)) / np.std(flux_data)
        
        # Remove outliers (simple sigma clipping)
        std = np.std(flux_normalized)
        flux_clipped = np.clip(flux_normalized, -5*std, 5*std)
        
        # Resize to standard length
        if len(flux_clipped) > self.input_length:
            # Downsample
            indices = np.linspace(0, len(flux_clipped)-1, self.input_length, dtype=int)
            flux_resized = flux_clipped[indices]
        elif len(flux_clipped) < self.input_length:
            # Pad with median value
            pad_value = np.median(flux_clipped)
            pad_length = self.input_length - len(flux_clipped)
            flux_resized = np.pad(flux_clipped, (0, pad_length), 
                                constant_values=pad_value)
        else:
            flux_resized = flux_clipped
        
        return flux_resized.reshape(-1, 1)  # Add channel dimension
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the CNN model"""
        try:
            logger.info("Starting CNN model training...")
            
            # Build model
            self.model = self.build_model((self.input_length, 1))
            
            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=1e-6
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=f"{self.model_path}_best.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(
                X_val, y_val, verbose=0
            )
            
            # Save final model
            if self.model_path:
                self.save_model()
            
            logger.info(f"CNN training completed. Val Accuracy: {val_accuracy:.4f}")
            
            return {
                "val_accuracy": float(val_accuracy),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "val_loss": float(val_loss),
                "epochs_trained": len(history.history['loss']),
                "history": {
                    "accuracy": [float(x) for x in history.history['accuracy']],
                    "val_accuracy": [float(x) for x in history.history['val_accuracy']],
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to train CNN model: {e}")
            raise
    
    def predict(self, light_curves: List[np.ndarray]) -> List[Dict]:
        """Make predictions on light curve data with saliency maps"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Preprocess light curves
            X = np.array([
                self.preprocess_light_curve(lc) for lc in light_curves
            ])
            
            # Make predictions
            predictions = self.model.predict(X)
            
            results = []
            for i, pred in enumerate(predictions):
                class_idx = np.argmax(pred)
                confidence = float(np.max(pred))
                
                # Generate saliency map
                saliency_map = self.generate_saliency_map(X[i:i+1])
                
                result = {
                    "prediction": self.classes[class_idx],
                    "confidence": confidence,
                    "probabilities": {
                        self.classes[j]: float(pred[j]) 
                        for j in range(len(self.classes))
                    },
                    "saliency_map": saliency_map.flatten().tolist()
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to make CNN predictions: {e}")
            raise
    
    def generate_saliency_map(self, input_data: np.ndarray) -> np.ndarray:
        """Generate saliency map for model explanation"""
        try:
            with tf.GradientTape() as tape:
                tape.watch(input_data)
                predictions = self.model(input_data)
                loss = tf.reduce_max(predictions)
            
            # Calculate gradients
            gradients = tape.gradient(loss, input_data)
            
            # Take absolute values and reduce to 1D
            saliency = tf.reduce_mean(tf.abs(gradients), axis=-1)
            
            return saliency.numpy()
            
        except Exception as e:
            logger.error(f"Failed to generate saliency map: {e}")
            return np.zeros((input_data.shape[0], input_data.shape[1]))
    
    def save_model(self):
        """Save trained model"""
        try:
            model_dir = Path(self.model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self.model.save(f"{self.model_path}_cnn.h5")
            logger.info(f"CNN model saved to {self.model_path}_cnn.h5")
            
        except Exception as e:
            logger.error(f"Failed to save CNN model: {e}")
            raise
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(f"{self.model_path}_cnn.h5")
            logger.info(f"CNN model loaded from {self.model_path}_cnn.h5")
            
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            raise
    
    def plot_training_history(self, history: Dict):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training')
        axes[0, 0].plot(history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training')
        axes[0, 1].plot(history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        plt.tight_layout()
        return fig