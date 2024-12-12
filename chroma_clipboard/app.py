import asyncio
import io
import signal
import sys

import cv2
import keyboard
import numpy as np
import qasync
import win32clipboard  # For clipboard management
from PIL import Image, ImageGrab
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QCoreApplication, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QMenu,
                               QPushButton, QSlider, QSpinBox, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

from chroma_clipboard.resources_rc import qt_resource_data

QRC = qt_resource_data  # To prevent import from being optimized away

# Globals for configuration
CHROMA_KEY_LOWER_BOUND = (50, 200, 100)  # Lower HSV bounds for green screen
CHROMA_KEY_UPPER_BOUND = (90, 255, 255)  # Upper HSV bounds for green screen
BLUR_KERNEL_SIZE = 15  # Gaussian blur kernel size
MORPH_KERNEL_SIZE = 5  # Morphology kernel size

def advanced_chroma_key(image: Image.Image, lower_bound, upper_bound, blur_kernel, morph_kernel, alpha_scale=1.0) -> Image.Image:
    """
    Advanced chroma keying using OpenCV with adjustable intensity.

    Args:
        image: Input PIL Image.
        lower_bound: Lower HSV bounds for chroma key.
        upper_bound: Upper HSV bounds for chroma key.
        blur_kernel: Gaussian blur kernel size.
        morph_kernel: Morphology kernel size.
        alpha_scale: Scaling factor for blending transparency.

    Returns:
        A PIL Image with the background partially removed based on the mask.
    """
    # Convert PIL Image to OpenCV format (RGBA -> BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    # Create a smooth mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound).astype(np.float32)
    mask = mask / 255.0  # Normalize to [0, 1]

    # Apply Gaussian blur for smooth transitions
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

    # Spill suppression and edge refinement
    morph_kernel = (morph_kernel, morph_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(morph_kernel, np.uint8))

    # Scale mask for alpha blending
    mask_scaled = (mask * alpha_scale).astype(np.float32)
    alpha_channel = (1 - mask_scaled) * 255

    # Perform blending using NumPy
    result = image_cv * (1 - mask_scaled[:, :, None])  # Preserve non-masked areas
    result = result.astype(np.uint8)

    # Add alpha channel from the mask
    result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_rgba[:, :, 3] = alpha_channel.astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA))


class ChromaKeyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Chroma Key Adjuster")
        self.setGeometry(100, 100, 800, 600)

        # Layouts
        self.main_layout = QVBoxLayout(self)
        self.controls_layout = QVBoxLayout()
        self.preview_layout = QVBoxLayout()

        # Clipboard Image
        self.original_image = self.get_clipboard_image()
        self.preview_image = self.original_image.copy()

        # Controls
        self.add_controls()

        # Preview Display
        self.image_label = QLabel()
        self.update_preview()
        self.preview_layout.addWidget(self.image_label)

        # Copy to Clipboard Button
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.controls_layout.addWidget(self.copy_button)

        # Combine Layouts
        self.main_layout.addLayout(self.preview_layout)
        self.main_layout.addLayout(self.controls_layout)

    def add_controls(self):
        # Chroma Key Color
        self.color_input = QLineEdit("#22FF00")
        self.color_input.setPlaceholderText("Chroma Key Color (#RRGGBB)")
        self.color_input.textChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Chroma Key Color"))
        self.controls_layout.addWidget(self.color_input)

        # Tolerance
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(100)
        self.tolerance_slider.setValue(50)
        self.tolerance_slider.valueChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Tolerance"))
        self.controls_layout.addWidget(self.tolerance_slider)

        # Blur Kernel Size
        self.blur_spinbox = QSpinBox()
        self.blur_spinbox.setMinimum(1)
        self.blur_spinbox.setMaximum(50)
        self.blur_spinbox.setValue(BLUR_KERNEL_SIZE)
        self.blur_spinbox.valueChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Blur Kernel Size"))
        self.controls_layout.addWidget(self.blur_spinbox)

        # Morph Kernel Size
        self.morph_spinbox = QSpinBox()
        self.morph_spinbox.setMinimum(1)
        self.morph_spinbox.setMaximum(50)
        self.morph_spinbox.setValue(MORPH_KERNEL_SIZE)
        self.morph_spinbox.valueChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Morph Kernel Size"))
        self.controls_layout.addWidget(self.morph_spinbox)

    def get_clipboard_image(self) -> Image.Image:
        """Grabs the image from the clipboard."""
        image = ImageGrab.grabclipboard()
        if image is None:
            raise RuntimeError("No image found in clipboard.")
        return image.convert("RGBA")

    def update_preview(self):
        """Updates the preview based on current settings."""
        try:
            # Parse Chroma Key Color
            color_code = self.color_input.text()
            if color_code.startswith("#"):
                color_code = color_code[1:]
            chroma_key_color = tuple(int(color_code[i:i+2], 16) for i in (0, 2, 4))

            # Convert RGB to HSV
            color_rgb = np.uint8([[chroma_key_color]])
            color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0][0]

            # Tolerance and HSV Bounds
            tolerance = self.tolerance_slider.value()
            if tolerance == 0:
                # No replacement: Show the original image
                self.preview_image = self.original_image.copy()
                print("Tolerance is 0: Showing original image")
            else:
                # Dynamically adjust the bounds
                lower_bound = np.array([
                    max(0, int(color_hsv[0]) - tolerance % 180),  # Hue
                    max(0, int(color_hsv[1]) - (tolerance * 4)),  # Saturation
                    max(0, int(color_hsv[2]) - (tolerance * 4)),  # Value
                ])
                upper_bound = np.array([
                    min(180, int(color_hsv[0]) + tolerance % 180),  # Hue
                    min(255, int(color_hsv[1]) + (tolerance * 4)),  # Saturation
                    min(255, int(color_hsv[2]) + (tolerance * 4)),  # Value
                ])
                print("Lower Bound (HSV):", lower_bound)
                print("Upper Bound (HSV):", upper_bound)

                # Apply Chroma Key with smoother transitions
                processed_image = advanced_chroma_key(
                    self.original_image, lower_bound, upper_bound,
                    self.blur_spinbox.value(), self.morph_spinbox.value(), alpha_scale=1.0 - (tolerance / 100)
                )
                self.preview_image = processed_image

            # Update Preview Image
            qt_image = ImageQt(self.preview_image).convertToFormat(QImage.Format_ARGB32)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)
            self.image_label.repaint()

        except Exception as e:
            print(f"Error updating preview: {e}")



    def copy_to_clipboard(self):
        """Copies the processed image to the clipboard."""
        output = io.BytesIO()
        self.preview_image.save(output, format="PNG")
        png_data = output.getvalue()
        output.close()

        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            png_format = win32clipboard.RegisterClipboardFormat("PNG")
            win32clipboard.SetClipboardData(png_format, png_data)
        finally:
            win32clipboard.CloseClipboard()
        print("Copied to clipboard!")


class TrayApp(QApplication):
    open_window_signal = Signal()

    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)
        self.chroma_window: ChromaKeyWindow | None = None

        # Create the tray icon
        self.tray_icon = QSystemTrayIcon(QIcon(":/icon.png"))

        # Create the tray menu
        self.tray_menu = QMenu()

        # Add "Quit" action
        self.quit_action = QAction("Quit")
        self.quit_action.triggered.connect(QCoreApplication.quit)
        self.tray_menu.addAction(self.quit_action)

        # Associate the menu with the tray icon
        self.tray_icon.setContextMenu(self.tray_menu)

        # Show the tray icon
        self.tray_icon.show()

        # Connect the signal to the slot
        self.open_window_signal.connect(self.show_chroma_key_window)

        # Register global shortcut
        keyboard.add_hotkey('ctrl+alt+k', self.open_window_signal.emit)
        print("Shortcut registered: Ctrl+Alt+K for opening Chroma Key Adjuster")

    def show_chroma_key_window(self):
        self.chroma_window = ChromaKeyWindow()
        self.chroma_window.show()
        self.chroma_window.activateWindow()


def signal_handler(sig, frame):
    print("Exiting...")
    QApplication.quit()

app_close_event = asyncio.Event()

def main():
    app = TrayApp(sys.argv)

    # Handle CTRL-C signal
    signal.signal(signal.SIGINT, signal_handler)
    timer = QTimer()
    timer.start(250)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    app.aboutToQuit.connect(slot=app_close_event.set)
    event_loop = qasync.QEventLoop(app)  # type: ignore
    asyncio.set_event_loop(event_loop)  # type: ignore
    with event_loop:  # type: ignore
        event_loop.run_until_complete(app_close_event.wait())  # type: ignore


if __name__ == "__main__":
    main()
