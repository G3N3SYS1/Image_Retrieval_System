# visualization.py

import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import platform
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional
from matplotlib.gridspec import GridSpec
import numpy as np
import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk


class VisualizationHandler:
    @staticmethod
    def create_comparison_visualization(
        comparison_dir: str,
        query_image: str,
        matches: List[Tuple[str, float]],
        find_vehicle_image_func
    ) -> Optional[str]:
        if not matches:
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_folder = os.path.join(comparison_dir, f'comparison_{timestamp}')
        os.makedirs(comparison_folder, exist_ok=True)

        query_img = cv2.imread(query_image)
        if query_img is None:
            raise ValueError(f"Unable to read query image: {query_image}")

        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # Create the combined visualization with two columns
        num_matches = len(matches)
        num_rows = (num_matches + 2) // 3  # +2 to account for the query imag
        fig = plt.figure(figsize=(16,9))
        gs = GridSpec(2, 3, figure=fig)

        plt.subplot(gs[0])
        plt.imshow(query_img)
        plt.title('Input Image')
        plt.axis('off')

        # Process the matches and plot them in two columns
        for idx, (plate_number, similarity) in enumerate(matches):
            match_img_path = find_vehicle_image_func(plate_number)
            match_img = cv2.imread(match_img_path) if match_img_path and os.path.exists(match_img_path) else None
            plt.subplot(gs[(idx + 1) // 3, (idx + 1) % 3])  # Adjust subplot indexing
            plt.imshow(match_img)
            # plt.title(f'Car Plate: {plate_number}\nSimilarity: {similarity:.2%}')
            plt.title(f'\nCar Plate: {plate_number} --> {similarity:.2%}')
            plt.axis('off')

        plt.tight_layout()
        comparison_img_path = os.path.join(comparison_folder, 'comparison_visualization.jpg')
        plt.savefig(comparison_img_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Open the saved image using the system's default photo app
        if platform.system() == 'Windows':
            os.startfile(comparison_img_path)

        return comparison_img_path

    @staticmethod
    def save_similarity_data(matches: List[Tuple[str, float]], feature_cache: dict, comparison_folder: str) -> str:
        all_similarities = []
        for plate_number, stored_features in feature_cache.items():
            similarity = next((score for p, score in matches if p == plate_number), 0.0)
            if similarity > 0:
                all_similarities.append({
                    'License_plate_number': plate_number,
                    'Similarity': f'{similarity:.2%}'
                })

        csv_path = os.path.join(comparison_folder, 'similarities.csv')
        if all_similarities:
            df = pd.DataFrame(all_similarities)
            df = df.sort_values('Similarity', ascending=False)
        else:
            df = pd.DataFrame(columns=['License_plate_number', 'Similarity'])

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return csv_path

    @staticmethod
    def create_no_results_visualization(comparison_folder: str, query_img_with_boxes: np.ndarray) -> str:
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        query_img_rgb = cv2.cvtColor(query_img_with_boxes, cv2.COLOR_BGR2RGB)
        plt.imshow(query_img_rgb)
        plt.title('Query Image (with detections)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, 'No matches found',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 fontsize=14)
        plt.axis('off')

        comparison_img_path = os.path.join(comparison_folder, 'no_results.jpg')
        plt.tight_layout()
        plt.savefig(comparison_img_path, dpi=300, bbox_inches='tight')
        plt.close()

        return comparison_img_path

    @staticmethod
    def plot_detection_results(image: np.ndarray, boxes: List[Tuple[float, float, float, float]], classes: List[int],
                               confidences: List[float]) -> np.ndarray:
        img_copy = image.copy()

        for box, class_id, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box

            height, width = image.shape[:2]
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)

            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

            label = f'Class {class_id}: {conf:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1

            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            cv2.rectangle(img_copy,
                          (x1, y1 - label_height - baseline - 5),
                          (x1 + label_width, y1),
                          color,
                          -1)  # Filled rectangle

            cv2.putText(img_copy, label, (x1, y1 - baseline - 5), font, font_scale, (0, 0, 0), font_thickness)

        return img_copy


class ImageViewer:
    def __init__(self, root, image_folder):
        self.root = root
        self.root.title("Image Viewer")

        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.index = 0

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.prev_button = tk.Button(root, text="<", command=self.show_prev_image)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(root, text=">", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.show_image()

    def show_image(self):
        img_path = os.path.join(self.image_folder, self.image_files[self.index])
        img = Image.open(img_path)
        img = img.resize((800, 600), Image.LANCZOS)  # Resize for better display
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo  # Keep a reference to avoid garbage collection
        self.root.title(self.image_files[self.index])  # Set window title to image filename

    def show_prev_image(self):
        self.index = (self.index - 1) % len(self.image_files)
        self.show_image()

    def show_next_image(self):
        self.index = (self.index + 1) % len(self.image_files)
        self.show_image()