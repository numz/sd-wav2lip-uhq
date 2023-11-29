import launch
import os
import platform
import locale

system = platform.system()

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if lib == "dlib-bin" and system == "Darwin":
            lib = "dlib"  # replace dlib-bin as dlib
        if lib == "git+https://github.com/suno-ai/bark.git":
            if locale.setlocale(locale.LC_ALL, '') == "zh_CN.UTF-8" or "Chinese (Simplified)_China.utf8":
                lib = "git+https://mirror.ghproxy.com/https://github.com/suno-ai/bark.git" # China Github mirrors
        if lib == "onnxruntime-gpu==1.15.0" and system == "Darwin":
            continue  # skip onnxruntime-gpu
        if not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"wav2lip_uhq requirement: {lib}")
