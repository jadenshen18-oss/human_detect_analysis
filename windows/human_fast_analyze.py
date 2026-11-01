import base64
import time
from openai import OpenAI
from PIL import Image              # For loading the image
import matplotlib.pyplot as plt    # For rendering the image window


# 1. Connected to FastFlowLM's default port
client = OpenAI(
    base_url="http://localhost:52625/v1", 
    api_key="not-needed-for-local"
)

# Helper function to convert your image into a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 2. Your target image path
image_path = r"C:\Users\qiaos\VLM\senior_pain.jpeg" 
base64_image = encode_image(image_path)

start_time = time.perf_counter()

# 3. Requesting analysis from your local NPU model
response = client.chat.completions.create(
    model="qwen3vl-flash:4b",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Analyze the image step-by-step according to these strict rules. "
                        "Output ONLY the final matching string. Do not include any explanation, punctuation, or extra words.\n\n"
                        "Rules:\n"
                        "1. If the person's upper body is not upright, output 'Fall' and stop.\n"
                        "2. If the person is on the ground, output 'Fall' and stop.\n"
                        "3. If the person in the image is in pain, output 'In pain' and stop.\n"
                        "4. Otherwise, output 'All Good'."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    stream=False, # Set to False so your script gets the clean, single-word string instantly
    temperature=0.0 # Force maximum determinism for classification tasks
)

end_time = time.perf_counter()

# 4. Calculate total latency
latency_seconds = end_time - start_time

# Extract and clean up the result
result = response.choices[0].message.content.strip()
print(f"Status: {result}")
print(f"Latency: {latency_seconds:.3f} seconds")

try:
    # Load image from disk
    img = Image.open(image_path)
    
    # Create a figure and strip out the standard pixel coordinates/axes
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off') 
    
    # Set the window title and plot title to show the status and latency
    plt.title(f"Classification: {result}\nLatency: {latency_seconds:.2f}s", fontsize=14, fontweight='bold')
    
    print("\nDisplaying image on screen... (Close the image window to finish the script)")
    plt.show()

except Exception as e:
    print(f"\nCould not display image: {e}")