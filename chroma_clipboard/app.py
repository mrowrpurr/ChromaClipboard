import sys
import io
import keyboard
import win32clipboard  # For clipboard management
from PIL import Image, ImageGrab
import numpy as np
import cv2
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from chroma_clipboard.resources_rc import qt_resource_data

QRC = qt_resource_data # To prevent import from being optimized away

# Globals for configuration
CHROMA_KEY_LOWER_BOUND = (50, 100, 100)  # Lower HSV bounds for green screen
CHROMA_KEY_UPPER_BOUND = (90, 255, 255)  # Upper HSV bounds for green screen
BLUR_KERNEL_SIZE = (15, 15)  # Gaussian blur kernel size
MORPH_KERNEL_SIZE = (5, 5)  # Morphology kernel size


def advanced_chroma_key(image: Image.Image) -> Image.Image:
    """
    Optimized chroma keying with minimal foreground erosion and refined edge handling.

    Args:
        image: Input PIL Image.

    Returns:
        A PIL Image with the background removed.
    """
    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    # Create mask for green
    mask = cv2.inRange(hsv, np.array(CHROMA_KEY_LOWER_BOUND), np.array(CHROMA_KEY_UPPER_BOUND))

    # Edge refinement (less aggressive)
    mask_blurred = cv2.GaussianBlur(mask, BLUR_KERNEL_SIZE, 0)
    mask_refined = cv2.morphologyEx(mask_blurred, cv2.MORPH_CLOSE, np.ones(MORPH_KERNEL_SIZE, np.uint8))

    # Preserve original foreground detail
    foreground_mask = cv2.bitwise_not(mask_refined)  # Invert mask for foreground
    foreground = cv2.bitwise_and(image_cv, image_cv, mask=foreground_mask)

    # Spill suppression: Adjust green spill without affecting edges
    r, g, b = cv2.split(foreground)
    green_spill_mask = (g > r) & (g > b) & (g > 100)
    g[green_spill_mask] = (r[green_spill_mask] + b[green_spill_mask]) // 2  # Suppress green

    # Create alpha channel
    alpha = cv2.bitwise_not(mask_refined)

    # Merge channels with adjusted alpha
    result_rgba = cv2.merge((r, g, b, alpha))

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA))


def replace_clipboard_with_image(image: Image.Image):
    """
    Replaces the clipboard content with a PNG image to preserve transparency.
    """
    print("Replacing clipboard with PNG image...")

    # Ensure the image is in RGBA mode
    image = image.convert("RGBA")

    # Save the image as PNG in memory
    output = io.BytesIO()
    image.save(output, format="PNG")
    png_data = output.getvalue()
    output.close()

    # Set PNG data in clipboard
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        png_format = win32clipboard.RegisterClipboardFormat("PNG")
        win32clipboard.SetClipboardData(png_format, png_data)
    finally:
        win32clipboard.CloseClipboard()

    print("Clipboard updated successfully with PNG format.")


def process_screenshot():
    """
    Processes the clipboard screenshot using OpenCV-based chroma keying.
    """
    print("Processing screenshot with OpenCV chroma key...")
    image = ImageGrab.grabclipboard()
    if image is not None:
        print("Image found in clipboard")
        processed_image = advanced_chroma_key(image)
        replace_clipboard_with_image(processed_image)
    else:
        print("No image found in clipboard.")


class TrayApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)

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

        # Register global shortcut
        keyboard.add_hotkey("ctrl+alt+k", process_screenshot)

        print("Shortcut registered: Ctrl+Alt+K for processing screenshots")


def main():
    app = TrayApp(sys.argv)
    app.exec()


if __name__ == "__main__":
    main()
