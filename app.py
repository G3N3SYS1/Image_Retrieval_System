# app.py
import tkinter as tk
from tkinter import ttk, messagebox, Tk, Canvas, Entry, Button, PhotoImage
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog
import threading
import sys
import os
from models.matcher import CarBottomMatcher
from models.visualization import ImageViewer

MODEL_PATH = "C:\\Users\\MaxwellLee\\PycharmProjects\\CarPlateRecognition\\segment_748_epoch_100.pt"
OUTPUT_DIR = 'output'
matcher = CarBottomMatcher(MODEL_PATH, OUTPUT_DIR)

path = getattr(sys, '_MEIPASS', os.getcwd())
os.chdir(path)

class RegisterPage:
    def __init__(self, root):
        self.root = root
        self.root.resizable(False, False)
        self.output_path = None
        self.is_processing = False
        self.path_list = []
        self.current_img_y = 242
        self.current_text_y = 240
        self.current_img_x = 160
        self.current_text_x = 191
        # Add flag to track first button press
        self.first_add = True

        self.valid_extensions = (".jpg", ".jpeg", ".png")
        self.setup_ui()

    def is_valid_file(self, file_path):
        """Check if the file has a valid image extension"""
        return file_path.lower().endswith(self.valid_extensions)

    def setup_ui(self):
        # Create UI elements
        self.create_canvas()
        self.load_images()
        self.create_title()
        self.create_path_entry()
        self.create_processing_area()
        self.create_end_page()
        self.add_drag_and_drop_support()

    def create_canvas(self):
        self.canvas = Canvas(self.root, width=659, height=464)
        self.canvas.pack(fill="both", expand=True)

    def load_images(self):
        # Load all image resources
        self.bg = PhotoImage(file=r"assets/main_page/background_image.png")
        self.title_1 = PhotoImage(file=r"assets/main_page/Image_Retrieval.png")
        self.title_2 = PhotoImage(file=r"assets/main_page/System.png")
        self.imgpth_bg = PhotoImage(file=r"assets/main_page/image_path_background_box.png")
        self.imgpth_txtbox = PhotoImage(file=r"assets/main_page/image_path.png")
        self.add_button_img = PhotoImage(file=r"assets/main_page/add_button.png")
        self.folder_button_img = PhotoImage(file=r"assets/main_page/folder_button.png")
        self.drag_drop_img = PhotoImage(file=r"assets/main_page/drag_and_drop_box_initial.png")
        self.drag_drop_process_img = PhotoImage(file=r"assets/main_page/drag_and_drop_box_final.png")
        self.folder_icon_img = PhotoImage(file=r"assets/main_page/folder_icon.png")
        self.file_icon_img = PhotoImage(file=r"assets/main_page/file_icon.png")
        self.ok_button_img = PhotoImage(file=r"assets/main_page/ok_button.png")

        # Set background
        self.canvas.create_image(0, 0, image=self.bg, anchor="nw")

    def create_title(self):
        # self.canvas.create_image(241, 66, image=self.title_1)
        # self.canvas.create_image(411, 96, image=self.title_2)
        self.canvas.create_image(288, 76, image=self.title_1)
        self.canvas.create_image(458, 106, image=self.title_2)

    def create_path_entry(self):
        # Image Path UI elements
        self.canvas.create_image(300, 166, image=self.imgpth_bg)
        self.canvas.create_image(263, 165, image=self.imgpth_txtbox)

        self.path_entry = Entry(self.root, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
        self.path_entry.place(x=142, y=155, width=242.0, height=23)
        self.path_entry.insert(0, "Enter File or Directory Path Here (.JPG)")

        # Change self.load_file to self.add_image
        add_button = Button(self.root, image=self.add_button_img,
                            borderwidth=0, highlightthickness=0,
                            command=self.add_image_from_entry, relief="flat")
        self.canvas.create_window(430, 165, window=add_button)

        folder_button = Button(self.root, image=self.folder_button_img,
                               borderwidth=0, highlightthickness=0,
                               command=self.select_path, relief="flat")
        self.canvas.create_window(500, 165, window=folder_button)

    def create_processing_area(self):
        # Initial drag and drop image
        self.drag_drop_img_id = self.canvas.create_image(328, 314,
                                                         image=self.drag_drop_img)

    def create_end_page(self):
        ok_button = Button(self.root, image=self.ok_button_img,
                           borderwidth=0, highlightthickness=0,
                           command=self.switch_to_loading_page, relief="flat")
        self.canvas.create_window(483, 438, window=ok_button)

    def select_path(self):
        """Handle folder selection"""
        path = filedialog.askdirectory()
        if not path:
            return

        try:
            files = os.listdir(path)
            if not files:
                raise ValueError("The selected folder is empty.")

            # Validate file extensions
            valid_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not valid_files:
                raise ValueError("No supported image files found.\n"
                                 "Accepted formats: .jpg, .jpeg, .png")

            self.output_path = path
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def add_image_from_entry(self):
        """Process selected images"""
        folder_or_file_path = self.path_entry.get()

        if not folder_or_file_path:
            messagebox.showerror("Error", "Path must not be empty.")
            return

        try:
            if not os.path.exists(folder_or_file_path):
                raise FileNotFoundError("The specified path does not exist.")

            # Validate path before proceeding
            if os.path.isfile(folder_or_file_path):
                if not self.is_valid_file(folder_or_file_path):
                    raise ValueError(
                        f"Unsupported file format. Accepted formats: {', '.join(self.valid_extensions)}")
            elif os.path.isdir(folder_or_file_path):
                # Check if directory contains any valid images
                valid_files = [f for f in os.listdir(folder_or_file_path)
                               if f.lower().endswith(self.valid_extensions)]
                if not valid_files:
                    raise ValueError(
                        f"No supported image files found.\nAccepted formats: {', '.join(self.valid_extensions)}")
            else:
                raise ValueError("Invalid path type")

            # Only call update_ui_for_processing on first add
            if self.first_add:
                self.update_ui_for_processing()
                self.first_add = False

            # If we get here, validation passed
            self.load_image(folder_or_file_path)

            # Process based on path type
            if os.path.isfile(folder_or_file_path):
                self.process_single_file(folder_or_file_path)
            elif os.path.isdir(folder_or_file_path):
                self.process_directory(folder_or_file_path)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_image(self, file_path):
        """
        Display icon and path name, storing paths for future use
        Args:
            file_path: Path to file or directory
        """
        # Get just the name from the path
        name = os.path.basename(file_path)

        # Add path to list if not already present
        if file_path not in self.path_list:
            self.path_list.append(file_path)

            # Create icon based on path type
            if os.path.isfile(file_path):
                icon_id = self.canvas.create_image(self.current_img_x, self.current_img_y,
                                                   image=self.file_icon_img)
            else:
                icon_id = self.canvas.create_image(self.current_img_x, self.current_img_y,
                                                   image=self.folder_icon_img)

            # Create text label for the path name
            text_id = self.canvas.create_text(self.current_text_x, self.current_text_y,
                                              text=name,
                                              fill="#000000",
                                              font=("Baskervville SC", 8),
                                              anchor="w")  # west/left alignment

            # Update y positions for next item
            self.current_img_y += 41
            self.current_text_y += 40

            # Store the IDs if you need to modify/delete them later
            if not hasattr(self, 'path_items'):
                self.path_items = []
            self.path_items.append((icon_id, text_id))

    def process_single_file(self, file_path):
        """Process a single image file"""
        if not file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            raise ValueError("Unsupported file format")

        self.is_processing = True

    def process_directory(self, dir_path):
        """Process all images in a directory"""
        image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not image_files:
            raise ValueError("No supported image files found in directory")

        self.is_processing = True

    def update_ui_for_processing(self):
        """Update UI elements when processing starts"""
        if hasattr(self, 'drag_drop_img_id'):
            self.canvas.delete(self.drag_drop_img_id)
        self.canvas.create_image(328, 314, image=self.drag_drop_process_img)

    def add_drag_and_drop_support(self):
        """Add drag-and-drop support to the canvas."""
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, event):
        """Handle files dropped via drag-and-drop."""
        dropped_path = event.data

        # Remove curly braces if present (common in Windows drag-drop)
        if dropped_path.startswith('{') and dropped_path.endswith('}'):
            dropped_path = dropped_path[1:-1]

        try:
            if not os.path.exists(dropped_path):
                raise FileNotFoundError("The specified path does not exist.")

            # Validate path before proceeding
            if os.path.isfile(dropped_path):
                if not self.is_valid_file(dropped_path):
                    raise ValueError(
                        f"Unsupported file format. Accepted formats: {', '.join(self.valid_extensions)}")
            elif os.path.isdir(dropped_path):
                # Check if directory contains any valid images
                valid_files = [f for f in os.listdir(dropped_path)
                               if f.lower().endswith(self.valid_extensions)]
                if not valid_files:
                    raise ValueError(
                        f"No supported image files found.\nAccepted formats: {', '.join(self.valid_extensions)}")
            else:
                raise ValueError("Invalid path type")

            # Only call update_ui_for_processing on first add
            if self.first_add:
                self.update_ui_for_processing()
                self.first_add = False

            # If we get here, validation passed
            self.load_image(dropped_path)

            # Process based on path type
            if os.path.isfile(dropped_path):
                self.process_single_file(dropped_path)
            elif os.path.isdir(dropped_path):
                self.process_directory(dropped_path)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def switch_to_loading_page(self):
        """Switch to RegisterPage."""
        self.root.destroy()
        loading_root = TkinterDnD.Tk()
        loading_root.title("Recognition System")
        loading_root.geometry("659x464")
        LoadingPage(loading_root, self.path_list)
        loading_root.mainloop()
pass

class LoadingPage:
    def __init__(self, root, path_list=None):
        self.root = root
        self.root.resizable(False, False)
        self.is_processing = False
        self.path_list = path_list if path_list else []
        self.total_files = len(self.path_list)
        self.processed_files = 0
        self.setup_ui()
        if self.path_list:
            self.start_processing()

    def setup_ui(self):
        # Create UI elements
        self.create_canvas()
        self.load_images()
        self.create_title()
        self.create_loading_area()


    def create_canvas(self):
        self.canvas = Canvas(self.root, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)

    def load_images(self):
        # Load all image resources
        self.bg = PhotoImage(file=r"assets/loading_page/background_image.png")
        self.title_1 = PhotoImage(file=r"assets/loading_page/Image_Retrieval.png")
        self.title_2 = PhotoImage(file=r"assets/loading_page/System.png")
        self.loading_box_img = PhotoImage(file=r"assets/loading_page/loading_box.png")
        self.loading_font = PhotoImage(file=r"assets/loading_page/loading_font.png")
        self.complete_font = PhotoImage(file=r"assets/loading_page/complete_font.png")
        self.status_font = PhotoImage(file=r"assets/loading_page/status_font.png")
        self.done_button_img = PhotoImage(file=r"assets/loading_page/done_button.png")
        self.initial_progress_bar = PhotoImage(file=r"assets/loading_page/initial_bar.png")
        self.tenth_progress_bar = PhotoImage(file=r"assets/loading_page/10_bar.png")
        self.twentieth_progress_bar = PhotoImage(file=r"assets/loading_page/20_bar.png")
        self.thirtieth_progress_bar = PhotoImage(file=r"assets/loading_page/30_bar.png")
        self.fourtieth_progress_bar = PhotoImage(file=r"assets/loading_page/40_bar.png")
        self.fiftieth_progress_bar = PhotoImage(file=r"assets/loading_page/50_bar.png")
        self.sixtieth_progress_bar = PhotoImage(file=r"assets/loading_page/60_bar.png")
        self.seventieth_progress_bar = PhotoImage(file=r"assets/loading_page/70_bar.png")
        self.eightieth_progress_bar = PhotoImage(file=r"assets/loading_page/80_bar.png")
        self.ninetieth_progress_bar = PhotoImage(file=r"assets/loading_page/90_bar.png")
        self.hundredth_progress_bar = PhotoImage(file=r"assets/loading_page/100_bar.png")

        # Set background
        self.canvas.create_image(0, 0, image=self.bg, anchor="nw")

    def create_title(self):
        self.canvas.create_image(288, 66, image=self.title_1)
        self.canvas.create_image(458, 96, image=self.title_2)

    def create_loading_area(self):
        self.canvas.create_image(326, 239, image=self.loading_box_img)
        self.create_loading_text = self.canvas.create_image(329.25, 199.25, image=self.loading_font)
        self.canvas.create_image(325, 233, image=self.initial_progress_bar)
        self.progress_text = self.canvas.create_text(325, 265, text=f"0 of {self.total_files} files",
                                                   fill="#000000", font=("Baskervville SC", 10))

    def update_progress(self):
        self.processed_files += 1
        progress = (self.processed_files / self.total_files) * 100
        progress_level = (progress // 10) * 10  # Round to nearest 10

        progress_bars = {
            0: self.initial_progress_bar,
            10: self.tenth_progress_bar,
            20: self.twentieth_progress_bar,
            30: self.thirtieth_progress_bar,
            40: self.fourtieth_progress_bar,
            50: self.fiftieth_progress_bar,
            60: self.sixtieth_progress_bar,
            70: self.seventieth_progress_bar,
            80: self.eightieth_progress_bar,
            90: self.ninetieth_progress_bar,
            100: self.hundredth_progress_bar
        }

        self.canvas.create_image(325, 233, image=progress_bars[progress_level])
        self.canvas.itemconfig(self.progress_text,
                               text=f"{self.processed_files} of {self.total_files} files, {progress:.2f}%")

    def start_processing(self):
        """Start image processing in background"""
        if not self.path_list:
            print("No paths to process!")
            return

        def process_thread():
            for folder_path in self.path_list:
                try:
                    self.process_images(folder_path)
                except Exception as e:
                    print(f"Error processing {folder_path}: {str(e)}")
        # Start processing thread
        self.is_processing = True
        print("Starting processing thread...")
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    def process_images(self, folder_path):
        try:
            if os.path.isfile(folder_path):
                result = matcher.process_image(folder_path)
            else:
                results = matcher.batch_process_images(folder_path)
        except Exception as e:
            error_msg = f"Error processing {folder_path}: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
        finally:
            self.root.after(0, self.update_progress)
            self.root.after(0, self.check_completion)

    def check_completion(self):
        if self.processed_files >= self.total_files:
            self.finish_processing()

    def finish_processing(self):
        """Called when processing is complete to do final update and cleanup"""
        # Do one final update
        self.is_processing = False
        self.canvas.delete(self.create_loading_text)
        self.canvas.create_image(329.25, 199.25, image=self.complete_font)
        messagebox.showinfo("Success!", "Processing Complete!")
        self.switch_to_main_page()

    def switch_to_main_page(self):
        """Switch to RegisterPage."""
        self.root.destroy()
        main_root = TkinterDnD.Tk()
        main_root.title("Recognition System")
        main_root.geometry("659x464")
        VehicleListView(main_root)
        main_root.mainloop()
    pass

class VehicleListView:
    def __init__(self, root):
        self.root = root
        self.root.resizable(False, False)
        self.root.title("Vehicle List")
        self.root.geometry("659x464")
        self.select_all_state = False
        self.select_all_button = None

        self.checkbox_vars = {}

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.create_list_frame()
        self.create_buttons()
        self.refresh_list()
        self.image_windows = {}
        self.search_active = False
        self.search_entry = None
        self.highlighted_labels = []  # Track highlighted labels
        # Global shortcuts
        self.root.bind_all('<F5>', lambda e: self.refresh_list())
        self.root.bind_all('<Control-d>', lambda e: self.delete_image())
        self.root.bind_all('<Control-f>', self.start_search)
        # Root-level shortcuts that shouldn't work in search entry
        self.root.bind_all('<Control-a>', lambda e: self.select_all())



    def create_list_frame(self):
        self.list_frame = ttk.Frame(self.main_frame, relief="solid", borderwidth=1)  # Added border
        self.list_frame.pack(fill='both', expand=True)

        # Configure column weight
        self.list_frame.grid_columnconfigure(0, weight=1)
        self.list_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.list_frame, highlightthickness=0)  # Removed canvas border
        self.scrollbar = ttk.Scrollbar(self.list_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Bind canvas resize to frame
        self.list_frame.bind('<Configure>', self.on_frame_configure)
        self.scrollable_frame.bind('<Configure>', self.on_frame_change)

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Grid layout instead of pack
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=(10, 0))
        self.scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10))

    def on_frame_configure(self, event=None):
        # Set canvas width to match list_frame
        self.canvas.configure(width=self.list_frame.winfo_width() - 40)  # Account for scrollbar and padding

    def on_frame_change(self, event=None):
        # Update scroll region when frame changes
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def create_buttons(self):
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill='x', pady=20)

        ttk.Button(button_frame, text='Refresh', command=self.refresh_list).pack(side='right', padx=10)
        self.select_all_button = ttk.Button(button_frame, text='Select All', command=self.select_all)
        self.select_all_button.pack(side='left', padx=10)
        ttk.Button(button_frame, text='Add', command=self.switch_to_register_page).pack(side='left', padx=10)
        ttk.Button(button_frame, text='Delete', command=self.delete_image).pack(side='left', padx=10)
        # ttk.Button(button_frame, text='Modify', command=self.get_selected).pack(side='left', padx=10)
        ttk.Button(button_frame, text='Compare', command=self.compare_image).pack(side='left', padx=10)
        ttk.Button(button_frame, text='Modify', command=None).pack(side='left', padx=10)

    def refresh_list(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.checkbox_vars.clear()

        header_frame = ttk.Frame(self.scrollable_frame)
        header_frame.pack(fill='x', padx=15, pady=10)

        header_text = ttk.Label(header_frame,
                                text=f"{'Select':<10}{'License plate number':<25}{'Number of features':<22}{'           Storage date':<35}",
                                font=('TkDefaultFont', 10, 'bold'))
        header_text.pack(side='left', anchor='w')

        self.search_button = tk.Button(header_frame,
                                       text="🔍",
                                       font=("Arial", 12),
                                       command=self.start_search)
        self.search_button.pack(side='right', padx=5)

        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', padx=15, pady=10)

        vehicles = matcher.list_stored_vehicles()
        if vehicles:
            for v in vehicles:
                frame = ttk.Frame(self.scrollable_frame)
                frame.pack(fill='x', padx=15, pady=5)

                var = tk.BooleanVar(value=False)
                self.checkbox_vars[v['plate_number']] = var

                ttk.Checkbutton(frame, variable=var, onvalue=True, offvalue=False).pack(side='left', padx=(10, 30))

                # Create clickable label for plate number
                plate_label = ttk.Label(frame, text=f"{v['plate_number']}", cursor="hand2")
                plate_label.pack(side='left', padx=(30, 130))
                plate_label.bind('<Button-1>', lambda e, plate=v['plate_number']: self.view_images(plate))

                ttk.Label(frame, text=f"{v['features_count']}").pack(side='left', padx=(0, 120))
                ttk.Label(frame, text=f"{v['storage_date']}").pack(side='left')

                ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', padx=15, pady=5)
        else:
            messagebox.showerror(title="Error!", message="There is currently no stored vehicle information.")

    def get_selected(self):
        selected = [plate for plate, var in self.checkbox_vars.items() if var.get()]
        if selected:
            messagebox.showinfo("Selected Vehicles", f"Selected plates:\n{', '.join(selected)}")
        else:
            messagebox.showinfo("Selected Vehicles", "No vehicles selected")

    def select_all(self):
        self.select_all_state = not self.select_all_state
        for var in self.checkbox_vars.values():
            var.set(self.select_all_state)
        self.select_all_button.configure(text='Deselect All' if self.select_all_state else 'Select All')

    def delete_image(self):
        selected = [plate for plate, var in self.checkbox_vars.items() if var.get()]
        if not selected:
            messagebox.showerror(title="Error!", message="No vehicles selected.")
            return
        results = matcher.batch_delete_vehicles(selected)
        self.refresh_list()  # Refresh the list after deletion

    def compare_image(self):
        try:
            image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            matches = matcher.find_matching_plate(image_path)
            if not matches:
                messagebox.showerror(
                    title="Error!", message="No matching result found.")
                return
        except Exception as e:
            error_message = f"Recognize Error: {str(e)}"
            messagebox.showerror(
                title="Error!",
                message=error_message
            )
            return

    def view_images(self, plate_number):
        # Check if window already open for this plate
        if plate_number in self.image_windows:
            self.image_windows[plate_number].root.lift()  # Bring existing window to front
            return

        image_folder = os.path.join(OUTPUT_DIR, "images", plate_number)
        if not os.path.exists(image_folder):
            messagebox.showerror("Error", f"No images found for plate {plate_number}")
            return

        # Create new window for image viewer
        viewer_window = tk.Toplevel(self.root)
        viewer_window.protocol("WM_DELETE_WINDOW",
                               lambda p=plate_number: self.close_image_window(p))

        image_viewer = ImageViewer(viewer_window, image_folder)
        self.image_windows[plate_number] = image_viewer

    def close_image_window(self, plate_number):
        if plate_number in self.image_windows:
            self.image_windows[plate_number].root.destroy()
            del self.image_windows[plate_number]

    def start_search(self, event=None):
        if not self.search_active:
            self.search_frame = ttk.Frame(self.main_frame)
            self.search_frame.pack(side='top', fill='x', pady=(0, 10))

            self.search_entry = ttk.Entry(self.search_frame)
            self.search_entry.pack(side='left', fill='x', expand=True, padx=5)
            self.search_entry.bind('<Return>', self.do_search)
            self.search_entry.bind('<Escape>', self.close_search)
            self.search_entry.bind('<Control-a>', lambda e: self.search_entry.select_range(0, 'end'))
            self.search_entry.focus_set()
            self.search_active = True
            self.search_entry.bind('<KeyRelease>', self.do_search)

    def close_search(self, event=None):
        self.clear_highlights()
        if self.search_active:
            self.search_frame.destroy()
            self.search_active = False
            self.search_entry = None

    def do_search(self, event=None):
        # Clear previous highlights
        for old_label in self.highlighted_labels:
            old_label.configure(style='')
        self.highlighted_labels.clear()

        search_text = self.search_entry.get().strip()
        if not search_text:
            return

        found = False
        found_widget = None

        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        text = child.cget('text').strip()
                        if text == search_text:
                            style = ttk.Style()
                            style.configure('Highlight.TLabel', background='yellow', foreground='black')
                            child.configure(style='Highlight.TLabel')
                            self.highlighted_labels.append(child)
                            found = True
                            found_widget = widget

                            # Check corresponding checkbox when Enter is pressed
                            if event and event.keysym == 'Return':
                                for checkbox_child in widget.winfo_children():
                                    if isinstance(checkbox_child, ttk.Checkbutton):
                                        # Get checkbox variable and set it
                                        self.checkbox_vars[text].set(True)
                                        break

        if found and found_widget:
            widget_y = found_widget.winfo_y()
            frame_height = self.scrollable_frame.winfo_height()
            relative_pos = widget_y / frame_height
            self.canvas.yview_moveto(relative_pos)
            self.canvas.update_idletasks()

        elif not found and event and event.keysym == 'Return':
            messagebox.showinfo("Search Result", "No matching license plate found.")

    def clear_highlights(self):
        for label in self.highlighted_labels:
            label.configure(background='', foreground='')
        self.highlighted_labels.clear()

    def switch_to_register_page(self):
        """Switch to RegisterPage."""
        self.root.destroy()
        register_root = TkinterDnD.Tk()
        RegisterPage(register_root)  # Switch to the RegisterPage now
        register_root.mainloop()

if __name__ == '__main__':
    root = TkinterDnD.Tk()
    root.title('Recognition System')
    root.geometry("659x464")
    app = RegisterPage(root)  # Start with VehicleListView
    root.mainloop()