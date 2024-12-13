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
                               QPushButton, QSlider, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

from chroma_clipboard.resources_rc import qt_resource_data

QRC = qt_resource_data  # To prevent import from being optimized away

# Globals for configuration
CHROMA_KEY_LOWER_BOUND = (50, 200, 100)  # Lower HSV bounds for green screen
CHROMA_KEY_UPPER_BOUND = (90, 255, 255)  # Upper HSV bounds for green screen


def advanced_chroma_key(image: Image.Image, lower_bound, upper_bound, blur_kernel, morph_kernel, spill_suppression=0.5) -> Image.Image:
    """
    Advanced chroma keying with adjustable parameters.
    This version does not forcefully treat green as spill by default.
    It only removes colors within the specified HSV range.
    """

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    # Create a mask based on the given HSV bounds
    mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))

    # Edge refinement
    blur_size = (blur_kernel, blur_kernel)
    mask_blurred = cv2.GaussianBlur(mask, blur_size, 0)
    morph_size = (morph_kernel, morph_kernel)
    mask_refined = cv2.morphologyEx(mask_blurred, cv2.MORPH_CLOSE, np.ones(morph_size, np.uint8))

    # Preserve original foreground detail
    foreground_mask = cv2.bitwise_not(mask_refined)
    foreground = cv2.bitwise_and(image_cv, image_cv, mask=foreground_mask)

    # If spill_suppression is used, one could implement a more general approach
    # that blends keyed-out areas with neighboring colors. For now, this code
    # does not forcefully remove green unless it falls within the keyed range.

    # Create alpha channel
    alpha = cv2.bitwise_not(mask_refined)

    # Merge channels with adjusted alpha
    b, g, r = cv2.split(foreground)
    result_rgba = cv2.merge((r, g, b, alpha))

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
        self.color_input = QLineEdit("00ff22")
        self.color_input.setPlaceholderText("Chroma Key Color(s) (#RRGGBB)")
        self.color_input.textChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Chroma Key Color"))
        self.controls_layout.addWidget(self.color_input)

        # Transparency Color
        self.transparency_color_input = QLineEdit("ff0000")
        self.transparency_color_input.setPlaceholderText("Transparency Overlay Color (#RRGGBB)")
        self.transparency_color_input.textChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Transparency Overlay Color"))
        self.controls_layout.addWidget(self.transparency_color_input)

        # Tolerance
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(100)
        self.tolerance_slider.setValue(20)
        self.tolerance_slider.valueChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Tolerance"))
        self.controls_layout.addWidget(self.tolerance_slider)

        # Blur Kernel Size
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(1)
        self.blur_slider.setMaximum(100)
        self.blur_slider.setValue(15)
        self.blur_slider.valueChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Blur Kernel Size"))
        self.controls_layout.addWidget(self.blur_slider)

        # Morph Kernel Size
        self.morph_slider = QSlider(Qt.Horizontal)
        self.morph_slider.setMinimum(1)
        self.morph_slider.setMaximum(100)
        self.morph_slider.setValue(5)
        self.morph_slider.valueChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Morph Kernel Size"))
        self.controls_layout.addWidget(self.morph_slider)

        # Spill Suppression
        self.spill_slider = QSlider(Qt.Horizontal)
        self.spill_slider.setMinimum(0)
        self.spill_slider.setMaximum(100)
        self.spill_slider.setValue(50)
        self.spill_slider.valueChanged.connect(self.update_preview)
        self.controls_layout.addWidget(QLabel("Spill Suppression"))
        self.controls_layout.addWidget(self.spill_slider)

    def get_clipboard_image(self) -> Image.Image:
        """Grabs the image from the clipboard."""
        image = ImageGrab.grabclipboard()
        if image is None:
            raise RuntimeError("No image found in clipboard.")
        return image.convert("RGBA")

    def update_preview(self):
        """Updates the preview based on current settings."""
        try:
            self.preview_image = self.original_image.copy()

            color_codes = [code.strip().lstrip('#') for code in self.color_input.text().split(',')]

            for color_code in color_codes:
                chroma_key_color = tuple(int(color_code[i:i+2], 16) for i in (0, 2, 4))

                # Convert RGB to HSV
                color_rgb = np.uint8([[chroma_key_color]])
                color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0][0]

                # Tolerance and HSV Bounds
                tolerance = self.tolerance_slider.value()
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

                # Apply Chroma Key
                spill_suppression = self.spill_slider.value() / 100.0
                self.preview_image = advanced_chroma_key(
                    self.preview_image, lower_bound, upper_bound,
                    self.blur_slider.value(), self.morph_slider.value(), spill_suppression
                )

            # Replace fully transparent pixels with user-specified color
            transparency_color = tuple(int(self.transparency_color_input.text()[i:i+2], 16) for i in (0, 2, 4))
            preview_array = np.array(self.preview_image)
            transparent_mask = preview_array[:, :, 3] == 0  # Alpha channel is 0
            preview_array[transparent_mask] = [*transparency_color, 255]  # Apply transparency color

            # Convert the modified array back to a PIL Image
            self.preview_image = Image.fromarray(preview_array, "RGBA")

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
        self.chroma_window.raise_()

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
