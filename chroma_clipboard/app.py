import io
import os
import sys

import cv2
import keyboard
import numpy as np
import win32clipboard  # For clipboard management
from PIL import Image, ImageGrab
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from chroma_clipboard.resources_rc import qt_resource_data

QRC = qt_resource_data # To prevent import from being optimized away

# Globals for configuration
BLUR_KERNEL_SIZE = (15, 15)  # Gaussian blur kernel size

MORPH_KERNEL_SIZE = 3  # Smaller morphology kernel for finer detail preservation

# Updated Globals for chroma key
CHROMA_KEY_LOWER_BOUND = (50, 180, 110)  # Lower HSV bounds for green screen
CHROMA_KEY_UPPER_BOUND = (78, 255, 255)  # Upper HSV bounds for green screen

def refined_chroma_key(image: Image.Image) -> Image.Image:
    """
    Refined chroma keying to remove green backgrounds without spill suppression.

    Args:
        image: Input PIL Image.

    Returns:
        A PIL Image with the background fully removed for target green.
    """
    try:
        # Ensure the image is in RGBA mode
        image = image.convert("RGBA")

        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

        # Create mask for target chroma key greens
        main_mask = cv2.inRange(hsv, np.array(CHROMA_KEY_LOWER_BOUND), np.array(CHROMA_KEY_UPPER_BOUND))

        # Preserve foreground
        foreground_mask = cv2.bitwise_not(main_mask)
        foreground = cv2.bitwise_and(image_cv, image_cv, mask=foreground_mask)

        # Create alpha channel: Full transparency for green areas
        alpha = cv2.bitwise_not(main_mask)

        # Merge channels with alpha channel
        r, g, b = cv2.split(foreground)
        result_rgba = cv2.merge((r, g, b, alpha))

        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA))

    except Exception as e:
        print(f"Error processing image: {e}")
        raise


def process_screenshot():
    """
    Processes the clipboard screenshot using OpenCV-based chroma keying.
    """
    print("Processing screenshot with OpenCV chroma key...")
    try:
        image = get_image_from_clipboard()
        print("Image found in clipboard")
        processed_image = refined_chroma_key(image)
        replace_clipboard_with_image(processed_image)
    except ValueError as e:
        print(f"Failed to process screenshot: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def get_image_from_clipboard() -> Image.Image:
    """
    Retrieves the image from the clipboard, ensuring it is valid and in a compatible format.

    Returns:
        A PIL Image in RGBA mode.

    Raises:
        ValueError: If the clipboard does not contain an image.
    """
    clipboard_data = ImageGrab.grabclipboard()

    if isinstance(clipboard_data, list):
        # If the clipboard contains file paths, try to open the first one
        file_path = clipboard_data[0] if clipboard_data else None
        if file_path and os.path.isfile(file_path):
            print(f"Clipboard contains file path: {file_path}")
            return Image.open(file_path).convert("RGBA")

    if clipboard_data is None:
        raise ValueError("No image found in clipboard.")

    if not isinstance(clipboard_data, Image.Image):
        raise ValueError("Clipboard data is not a valid image.")

    # Ensure the image is in RGBA mode
    return clipboard_data.convert("RGBA")


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
