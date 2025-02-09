# matcher.py

from ultralytics import YOLO
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from .visualization import VisualizationHandler
import traceback


class VehicleModelMatcher:
    def __init__(self, model_path='best.pt', save_dir='output'):
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not exists: {model_path}")
        self.model = YOLO(model_path)
        self.save_dir = save_dir
        self.feature_cache_path = os.path.join(save_dir, 'feature_cache.json')
        self.comparison_dir = os.path.join(save_dir, 'comparisons')
        os.makedirs(self.comparison_dir, exist_ok=True)
        self.feature_cache = self.load_feature_cache()
        os.makedirs(save_dir, exist_ok=True)
        self.visualization_handler = VisualizationHandler()
        self.processed_count = 0
        self.file_count = 0


    def load_feature_cache(self) -> dict:
        """Load feature cache"""
        if os.path.exists(self.feature_cache_path):
            with open(self.feature_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

            # Convert old format to new format if necessary
            converted_cache = {}
            for vehicle_model, data in cache_data.items():
                if isinstance(data, dict) and 'features' in data and 'images' in data:
                    # Already in new format
                    converted_cache[vehicle_model] = data
                else:
                    # Convert old format (single feature) to new format (list of features)
                    converted_cache[vehicle_model] = {
                        'features': [data] if data is not None else [],  # Convert single feature to list
                        'images': []  # Initialize empty images list for old entries
                    }
            return converted_cache

        return {}

    def save_feature_cache(self):
        """Save feature cache"""
        with open(self.feature_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_cache, f, ensure_ascii=False, indent=2)

    def extract_features(self, results) -> List[dict]:
        """
        Extract features from detection results
        Return: category, location, confidence and other information of each detected target
        """
        features = []
        if not results or len(results) == 0:
            return features

        # Get image size
        img_height, img_width = results[0].orig_img.shape[:2]

        # Process each detection box
        for box in results[0].boxes:
            # Get the coordinates of the box (normalized to between 0-1)
            x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()

            # Get category and confidence
            class_id = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])

            # Calculate the center point and dimensions of the box (normalized)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            features.append({
                'class_id': class_id,
                'center': [float(center_x), float(center_y)],
                'size': [float(width), float(height)],
                'confidence': float(confidence)
            })

        return features

    def calculate_similarity(self, features1: List[dict], features2: List[dict]) -> float:
        """
        Calculate the similarity between two sets of features
        First check if they have the same number of features for each class
        Then calculate comprehensive similarity using multiple indicators
        """
        if not features1 or not features2:
            return 0.0

        # Group features by category
        def group_by_class(features):
            grouped = {}
            for f in features:
                class_id = f['class_id']
                if class_id not in grouped:
                    grouped[class_id] = []
                grouped[class_id].append(f)
            return grouped

        grouped1 = group_by_class(features1)
        grouped2 = group_by_class(features2)

        # Check if they have the same classes
        if set(grouped1.keys()) != set(grouped2.keys()):
            return 0.0

        # Check if they have same number of features for each class
        for class_id in grouped1.keys():
            if len(grouped1[class_id]) != len(grouped2[class_id]):
                return 0.0

        # Calculate similarity score only if feature counts match
        total_score = 0
        for class_id in grouped1.keys():
            objs1 = grouped1[class_id]
            objs2 = grouped2[class_id]

            # Calculate the positional similarity between all pairs of objects in this category
            class_score = 0
            for obj1 in objs1:
                obj_scores = []
                for obj2 in objs2:
                    # Calculate center point distance
                    center_dist = math.sqrt(
                        (obj1['center'][0] - obj2['center'][0]) ** 2 +
                        (obj1['center'][1] - obj2['center'][1]) ** 2
                    )
                    # Calculate size difference
                    size_diff = math.sqrt(
                        (obj1['size'][0] - obj2['size'][0]) ** 2 +
                        (obj1['size'][1] - obj2['size'][1]) ** 2
                    )

                    # Comprehensive score
                    position_score = 1 / (1 + center_dist)
                    size_score = 1 / (1 + size_diff)
                    conf_score = (obj1['confidence'] + obj2['confidence']) / 2

                    obj_scores.append(position_score * 0.4 + size_score * 0.3 + conf_score * 0.3)

                if obj_scores:
                    class_score += max(obj_scores)

            # Normalize the score for this category
            if len(objs1) > 0:
                class_score /= len(objs1)
            total_score += class_score

        # Normalized total score
        final_score = total_score / len(grouped1)
        return final_score

    def find_matching_model(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the vehicle model that best matches the input image
        Args:
            image_path: Enter image path
            top_k: The best number of matches returned
        Returns:
            List of (vehicle_model, similarity_score) tuples
        """
        # Detect input images
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        results = self.model(img)
        input_features = self.extract_features(results)

        # Calculate similarity to all known vehicle models
        similarities = []
        for vehicle_model, images in self.feature_cache.items():
            for img_path, data in images.items():
                stored_features = data['features']
                similarity = self.calculate_similarity(input_features, stored_features)
                similarities.append((vehicle_model, similarity))
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Visualize the matches
        if similarities:
            self.visualization_handler.create_comparison_visualization(
                self.comparison_dir,
                image_path,
                similarities[:top_k],
                self._find_vehicle_image  # Function to find the vehicle image
            )

        return similarities[:top_k]

    def process_image(self, image_path: str, vehicle_model: str) -> dict:
        """Process and save images with vehicle models"""
        self.processed_count += 1
        vehicle_model = vehicle_model.split(".")[0]
        result = {
            'success': False,
            'vehicle_model': vehicle_model,
            'message': '',
            'save_path': None
        }

        try:
            img = cv2.imread(image_path)
            if img is None:
                result['message'] = f"Unable to read image: {image_path}"
                return result

            # Run the model
            results = self.model(img)

            # Extract features
            features = self.extract_features(results)
            if not features:
                result['message'] = "No features detected in image"
                return result

            # Initialize model entry if it doesn't exist
            if vehicle_model not in self.feature_cache:
                self.feature_cache[vehicle_model] = {}

            # Save the annotated image
            plotted_img = results[0].plot()
            save_path = self.create_storage_path(vehicle_model, len(self.feature_cache[vehicle_model]), change=True)
            cv2.imwrite(save_path, plotted_img)

            # Store features under the specific save path
            self.feature_cache[vehicle_model][save_path] = {
                'features': features
            }

            self.save_feature_cache()

            result.update({
                'success': True,
                'message': f"Processed successfully: {image_path} -> {save_path}",
                'save_path': save_path,
                'features_count': len(features)
            })

        except Exception as e:
            result['message'] = f"Error processing image: {str(e)}"

        return result, self.processed_count

    def create_storage_path(self, vehicle_model: str, index: int, change: False) -> str:
        """Create storage path with index for multiple images"""
        if change:
            base_dir = os.path.join(self.save_dir, 'images', vehicle_model)
        else:
            base_dir = os.path.join(self.save_dir, vehicle_model)
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{vehicle_model}_{index}.jpg")

    def normalize_vehicle_model(self, model: str) -> str:
        """Standardized vehicle model"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            model = model.replace(char, '_')
        return model

    def delete_vehicle(self, vehicle_model: str) -> bool:
        """Delete stored vehicle information"""
        if vehicle_model not in self.feature_cache:
            print(f"No vehicle information found with vehicle model {vehicle_model}")
            return False

        try:
            # Delete feature cache
            del self.feature_cache[vehicle_model]
            self.save_feature_cache()

            # Delete stored pictures
            img_path = self._find_vehicle_image(vehicle_model)
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
                print(f"Picture deleted：{img_path}")

            print(f"Successfully deleted vehicle information with vehicle model {vehicle_model}")
            return True

        except Exception as e:
            print(f"An error occurred while deleting vehicle information: {str(e)}")
            return False

    def batch_delete_vehicles(self, vehicle_models: List[str]) -> Dict[str, bool]:
        """Delete vehicle information in batches

        Args:
            vehicle_models: List of vehicle models to be deleted
        Returns:
            Dict[str, bool]: Deletion results for each vehicle model
        """
        results = {}
        for model in vehicle_models:
            results[model] = self.delete_vehicle(model)
        return results

    def modify_vehicle(self, old_vehicle_model: str, new_vehicle_model: str, new_image_path: str = None) -> bool:
        """Modify vehicle information"""
        print("\nDEBUG INFORMATION:")
        print(f"1. Attempting to modify vehicle:")
        print(f"   - Old model: {old_vehicle_model}")
        print(f"   - New model: {new_vehicle_model}")
        print(f"   - New image path: {new_image_path}")

        # Debug feature cache content
        print("\n2. Feature cache status:")
        print(f"   - Available models: {list(self.feature_cache.keys())}")
        print(f"   - Is old model in cache?: {old_vehicle_model in self.feature_cache}")

        if old_vehicle_model not in self.feature_cache:
            print(f"Error: No vehicle information found with vehicle model {old_vehicle_model}")
            return False

        try:
            print("\n3. Starting modification process:")
            if new_image_path:
                print("   - Processing new image")
                if not os.path.exists(new_image_path):
                    print(f"   - Error: New picture does not exist: {new_image_path}")
                    return False

                # Process new images
                img = cv2.imread(new_image_path)
                print("   - Image read successfully")
                results = self.model(img)
                print("   - Model inference completed")
                new_features = self.extract_features(results)
                print("   - Features extracted")

                # save new images
                plotted_img = results[0].plot()
                new_save_path = self.create_storage_path(new_vehicle_model)
                cv2.imwrite(new_save_path, plotted_img)
                print(f"   - New image saved to: {new_save_path}")

                # delete old images
                old_img_path = self._find_vehicle_image(old_vehicle_model)
                if old_img_path and os.path.exists(old_img_path):
                    os.remove(old_img_path)
                    print(f"   - Old image removed: {old_img_path}")
            else:
                print("   - No new image provided")
                print("   - Using existing features")
                new_features = self.feature_cache[old_vehicle_model]
                # Rename old pictures
                old_img_path = self._find_vehicle_image(old_vehicle_model)
                if old_img_path and os.path.exists(old_img_path):
                    new_save_path = self.create_storage_path(new_vehicle_model)
                    shutil.move(old_img_path, new_save_path)
                    print(f"   - Image renamed from {old_img_path} to {new_save_path}")

            print("\n4. Updating feature cache:")
            if old_vehicle_model != new_vehicle_model:
                print(f"   - Removing old model entry: {old_vehicle_model}")
                del self.feature_cache[old_vehicle_model]
            print(f"   - Adding new model entry: {new_vehicle_model}")
            self.feature_cache[new_vehicle_model] = new_features
            print("   - Saving feature cache")
            self.save_feature_cache()

            print("\n5. Modification completed successfully")
            return True

        except Exception as e:
            print(f"\nERROR: An error occurred while modifying vehicle information:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return False

    def list_stored_vehicles(self) -> List[dict]:
        """Get information about all stored vehicles"""
        vehicles = []
        for vehicle_model, images in self.feature_cache.items():
            for img_path, data in images.items():
                features = data['features']  # Extract features for the specific image
                # Find the corresponding image file
                # img_path = self._find_vehicle_image(vehicle_model)
                vehicles.append({
                    'vehicle_model': vehicle_model,
                    'features_count': len(features),
                    'image_path': img_path,
                    'storage_date': self._get_file_date(img_path) if img_path else 'Unknown'
                })
        return vehicles

    def _find_vehicle_image(self, vehicle_model: str) -> str:
        """Find the storage path of vehicle pictures"""
        for root, _, files in os.walk(self.save_dir):
            normalized_model = self.normalize_vehicle_model(vehicle_model)
            for file in files:
                if file.startswith(normalized_model) and file.lower().endswith(('.jpg', '.png')):
                    return os.path.join(root, file)
        return None

    def _get_file_date(self, file_path: str) -> str:
        """Get file creation date"""
        if file_path and os.path.exists(file_path):
            timestamp = os.path.getctime(file_path)
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return 'Unknown'

    def batch_process_images(self, folder_path: str) -> Dict[str, dict]:
        """Batch process images in a folder"""
        results = {}
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        try:
            # Get all pictures in a folder
            image_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith(valid_extensions)]

            if not image_files:
                return {'error': {'success': False, 'message': f"No valid image files found in {folder_path}"}}

            # Process each image
            for img_file in image_files:
                # Extract vehicle model from filename
                vehicle_model = img_file.split('_')[0]
                image_path = os.path.join(folder_path, img_file)
                self.file_count = self.file_count + 1

                # Process the image and store results
                result = self.process_image(image_path, vehicle_model)
                results[vehicle_model] = result

        except Exception as e:
            results['error'] = {
                'success': False,
                'message': f"Batch processing error: {str(e)}"
            }

        return results, self.file_count

