import time
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = r"C:\Users\qiaos\VLM\detect_downscaled.jpg"
INTERVAL_SECONDS = 5

plt.ion()
fig, ax = plt.subplots()
im_artist = ax.imshow(Image.open(IMAGE_PATH))
ax.axis("off")
fig.canvas.manager.set_window_title(IMAGE_PATH)

try:
    while plt.fignum_exists(fig.number):
        im_artist.set_data(Image.open(IMAGE_PATH))
        fig.canvas.draw_idle()
        plt.pause(INTERVAL_SECONDS)
except KeyboardInterrupt:
    pass
