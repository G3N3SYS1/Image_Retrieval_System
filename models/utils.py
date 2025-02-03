from tkinter import Tk, filedialog
import os

def select_folder() -> str:
    """Open the folder selection dialog"""
    # root = Tk()
    # root.withdraw()  # Hide main window
    folder_path = filedialog.askdirectory(title="Select picture folder")
    return folder_path if folder_path else None

def validate_image_path(path: str) -> bool:
    """Verify that the image path is valid"""
    if not path:
        print("Error: path can't be empty")
        return False
        
    if not os.path.exists(path):
        print(f"Error: file not exists - {path}")
        return False
            
    if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        print("Error：Format not supported，Please use image in .png, .jpg, .jpeg, .bmp 或 .gif format")
        return False
            
    return True

def get_valid_image_path(prompt: str) -> str:
    """Get the valid image path and support returning to the main menu"""
    while True:
        print("\nenter 'q' or 'back' return to main menu")
        path = input(prompt).strip()
        
        if path.lower() in ['q', 'back']:
            return None
            
        if validate_image_path(path):
            return path
        
        print("Please reenter your choice or enter 'q' return to main menu.")