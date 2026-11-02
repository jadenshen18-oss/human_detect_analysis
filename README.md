# human_detect_analysis
Collect camera snapshot from Arduino Uno Q when detecting human being and send to Windows Laptop. Run Qwen vision language model locally on the laptop to analyze the snapshot to find if the person is in pain or fall. If the person is in pain or fall, use human voice to alarm on the event.

The second version of this project uses qwen3vl-flash:4b. The model was downloaded through "flm pull qwen3vl-flash:4b" and run locally in AMD Ryzen AI 340 laptop. human_fast_analyze.py was created to use the AMD's NPU - neural processor to speed up Vision Language Model processing.

Compared with using ollama to run qwen3-vl:4b on AMD CPU, the NPU runs at least 10x faster, saving per snapshot processing latency from 40+ s to under 5s. It also allows Arduino to send snapshot in a faster pace, e.g. every 3s because inference on AMD laptop now can handle it in around 3s.

Before running human_fast_analyze.py. Use another CMD shell to run "flm serve". It starts an local OPENAI sever that receives image and prompt from my script and download them to NPU and run the VLM inference.

Open Task Manager -> Performance tab, you can see NPU shows workload when running human_fast_analyze.py.
