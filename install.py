import launch
import os
import platform

system = platform.system()

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if lib == "dlib-bin" and system == "Darwin":
            lib = "dlib"  # replace dlib-bin as dlib
        if lib == "onnxruntime-gpu==1.15.0" and system == "Darwin":
            continue  # skip onnxruntime-gpu
        if not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"wav2lip_uhq requirement: {lib}")
