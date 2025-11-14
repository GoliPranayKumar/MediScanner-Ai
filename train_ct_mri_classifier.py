"""
CT/MRI Brain Scan Binary Classifier Training Script

This script:
1. Organizes raw dataset into train/test structure
2. Loads and preprocesses images
3. Builds and trains a CNN model
4. Evaluates on test set
5. Tests predictions on unseen demo images
"""

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
from datetime import datetime

class CTMRIClassifier:
    """Binary classifier for CT vs MRI brain scans"""
    
    def __init__(self, base_data_path=".", output_dir="models"):
        """
        Initialize classifier
        
        Args:
            base_data_path: Path to folder containing Demo images/ and Unseen demo images/
            output_dir: Directory to save models and results
        """
        self.base_path = Path(base_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Organized data paths
        self.data_dir = self.base_path / "data"
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.unseen_dir = self.base_path / "Unseen demo images"
        
        self.model = None
        self.history = None
        self.class_indices = {"CT": 0, "MRI": 1}
        
    def organize_dataset(self):
        """
        Reorganize dataset from:
        Demo images/
          ├── train_a/
          ├── train_b/
          ├── test_a/
          └── test_b/
        
        To:
        data/
          ├── train/
          │   ├── CT/
          │   └── MRI/
          └── test/
              ├── CT/
              └── MRI/
        """
        print("="*60)
        print("ORGANIZING DATASET")
        print("="*60)
        
        # Remove existing data directory if it exists
        if self.data_dir.exists():
            print(f"Removing existing {self.data_dir}...")
            shutil.rmtree(self.data_dir)
        
        # Create directory structure
        self.data_dir.mkdir(exist_ok=True)
        self.train_dir.mkdir(exist_ok=True)
        (self.train_dir / "CT").mkdir(exist_ok=True)
        (self.train_dir / "MRI").mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        (self.test_dir / "CT").mkdir(exist_ok=True)
        (self.test_dir / "MRI").mkdir(exist_ok=True)
        
        # Source directories
        demo_images_dir = self.base_path / "Demo images"
        
        if not demo_images_dir.exists():
            print(f"❌ ERROR: {demo_images_dir} not found!")
            print("Please ensure 'Demo images' folder exists with train_a, train_b, test_a, test_b subfolders")
            return False
        
        # Mapping: source -> destination
        mappings = [
            (demo_images_dir / "train_a", self.train_dir / "CT", "train_a → train/CT"),
            (demo_images_dir / "train_b", self.train_dir / "MRI", "train_b → train/MRI"),
            (demo_images_dir / "test_a", self.test_dir / "CT", "test_a → test/CT"),
            (demo_images_dir / "test_b", self.test_dir / "MRI", "test/MRI"),
        ]
        
        total_images = 0
        for src, dst, label in mappings:
            if not src.exists():
                print(f"⚠️  WARNING: {src} not found, skipping...")
                continue
            
            # Copy all images from source to destination
            image_count = 0
            for img_file in src.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img_file, dst / img_file.name)
                    image_count += 1
            
            print(f"✓ {label}: {image_count} images")
            total_images += image_count
        
        print(f"\n✓ Dataset organized successfully! Total images: {total_images}")
        return True
    
    def load_dataset(self, img_size=224, batch_size=32):
        """
        Load dataset using tf.keras.utils.image_dataset_from_directory
        
        Args:
            img_size: Image size (will be resized to img_size x img_size)
            batch_size: Batch size for training
        
        Returns:
            train_dataset, test_dataset, class_names
        """
        print("\n" + "="*60)
        print("LOADING DATASET")
        print("="*60)
        
        # Load training dataset
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            seed=42,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode='binary'  # Binary classification: CT=0, MRI=1
        )
        
        # Load test dataset
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            self.test_dir,
            seed=42,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode='binary',
            shuffle=False
        )
        
        class_names = train_dataset.class_names
        print(f"✓ Classes: {class_names}")
        print(f"✓ Training batches: {len(train_dataset)}")
        print(f"✓ Test batches: {len(test_dataset)}")
        
        return train_dataset, test_dataset, class_names
    
    def normalize_and_augment(self, train_dataset, test_dataset):
        """
        Normalize pixel values and apply data augmentation
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
        
        Returns:
            Augmented train_dataset and normalized test_dataset
        """
        print("\n" + "="*60)
        print("NORMALIZING AND AUGMENTING DATA")
        print("="*60)
        
        # Normalization layer
        normalization_layer = layers.Rescaling(1./255)
        
        # Data augmentation for training
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        
        # Apply normalization to both datasets
        train_dataset = train_dataset.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        test_dataset = test_dataset.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation only to training data
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Optimize performance
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        
        print("✓ Data normalized and augmented")
        print("✓ Prefetching enabled for optimal performance")
        
        return train_dataset, test_dataset
    
    def build_model(self, input_shape=(224, 224, 3)):
        """
        Build a simple CNN model for binary classification
        
        Args:
            input_shape: Input image shape
        """
        print("\n" + "="*60)
        print("BUILDING CNN MODEL")
        print("="*60)
        
        self.model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print("✓ Model built successfully")
        print(f"  Total parameters: {self.model.count_params():,}")
        self.model.summary()
        
    def train(self, train_dataset, test_dataset, epochs=20):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            test_dataset: Validation dataset
            epochs: Number of epochs
        """
        print("\n" + "="*60)
        print(f"TRAINING MODEL FOR {epochs} EPOCHS")
        print("="*60)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training complete")
        
    def evaluate(self, test_dataset):
        """
        Evaluate model on test set
        
        Args:
            test_dataset: Test dataset
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        results = self.model.evaluate(test_dataset, verbose=0)
        
        print(f"✓ Test Loss: {results[0]:.4f}")
        print(f"✓ Test Accuracy: {results[1]:.4f}")
        print(f"✓ Test AUC: {results[2]:.4f}")
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history.png", dpi=100, bbox_inches='tight')
        print(f"✓ Training history saved to {self.output_dir / 'training_history.png'}")
        plt.close()
    
    def save_model(self):
        """Save trained model"""
        model_path = self.output_dir / "ct_mri_classifier.h5"
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save model info
        info = {
            "model_type": "CT/MRI Binary Classifier",
            "input_shape": [224, 224, 3],
            "classes": list(self.class_indices.keys()),
            "total_parameters": int(self.model.count_params()),
            "training_date": datetime.now().isoformat()
        }
        
        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Model info saved to {info_path}")
    
    def predict_on_unseen_images(self):
        """
        Test predictions on unseen demo images
        """
        print("\n" + "="*60)
        print("TESTING ON UNSEEN DEMO IMAGES")
        print("="*60)
        
        if not self.unseen_dir.exists():
            print(f"⚠️  WARNING: {self.unseen_dir} not found")
            print("Skipping unseen image predictions")
            return
        
        # Get all image files
        image_files = list(self.unseen_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not image_files:
            print(f"⚠️  No images found in {self.unseen_dir}")
            return
        
        print(f"Found {len(image_files)} unseen images")
        
        predictions = []
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize
                
                # Predict
                prediction = self.model.predict(img_array, verbose=0)[0][0]
                
                # Interpret prediction
                class_name = "MRI" if prediction > 0.5 else "CT"
                confidence = prediction if prediction > 0.5 else (1 - prediction)
                
                predictions.append({
                    "filename": img_path.name,
                    "predicted_class": class_name,
                    "confidence": float(confidence),
                    "raw_prediction": float(prediction)
                })
                
                print(f"  {img_path.name}: {class_name} ({confidence:.2%})")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")
        
        # Save predictions
        predictions_path = self.output_dir / "unseen_predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\n✓ Predictions saved to {predictions_path}")
        
        return predictions
    
    def run_full_pipeline(self, epochs=20):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print("CT/MRI BRAIN SCAN CLASSIFIER - FULL TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Organize dataset
        if not self.organize_dataset():
            return False
        
        # Step 2: Load dataset
        train_dataset, test_dataset, class_names = self.load_dataset()
        
        # Step 3: Normalize and augment
        train_dataset, test_dataset = self.normalize_and_augment(train_dataset, test_dataset)
        
        # Step 4: Build model
        self.build_model()
        
        # Step 5: Train
        self.train(train_dataset, test_dataset, epochs=epochs)
        
        # Step 6: Evaluate
        self.evaluate(test_dataset)
        
        # Step 7: Plot training history
        self.plot_training_history()
        
        # Step 8: Save model
        self.save_model()
        
        # Step 9: Test on unseen images
        self.predict_on_unseen_images()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"Models and results saved to: {self.output_dir}")
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train CT/MRI Binary Classifier"
    )
    parser.add_argument(
        "--base-path",
        default=".",
        help="Base path containing 'Demo images' and 'Unseen demo images' folders"
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = CTMRIClassifier(
        base_data_path=args.base_path,
        output_dir=args.output
    )
    
    # Run pipeline
    classifier.run_full_pipeline(epochs=args.epochs)


if __name__ == "__main__":
    main()
