import os
import cv2
import numpy as np
from PIL import Image
import subprocess
import insightface
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple
from pkg_resources import resource_filename
from modules.shared import state, opts
import modules.face_restoration
from modules.upscaler import Upscaler, UpscalerData
from modules.face_restoration import FaceRestoration, restore_faces
import scripts.wav2lip.audio as audio
import tempfile
from ifnude import detect
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@dataclass
class ImageResult:
    path: Union[str, None] = None
    similarity: Union[Dict[int, float], None] = None  # face, 0..1

    def image(self) -> Union[Image.Image, None]:
        if self.path:
            return Image.open(self.path)
        return None


@dataclass
class UpscaleOptions:
    scale: int = 1
    upscaler: UpscalerData = None
    upscale_visibility: float = 0.5
    face_restorer: FaceRestoration = None
    restorer_visibility: float = 0.5


class FaceSwap:
    def __init__(self, face=None, audio=None, face_index=None, source=None, resize_factor=None, face_restore_model=None, code_former_weight=None):
        self.faceswap_folder = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
        self.wav2lip_folder = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
        self.faceswap_output_folder = os.path.join(self.wav2lip_folder, 'wav2lip', 'output', 'faceswap')
        self.face = face
        self.audio = audio
        self.source = source
        self.resize_factor = resize_factor
        self.code_former_weight = code_former_weight
        self.face_restore_model = face_restore_model
        self.model = self.faceswap_folder + "/model/inswapper_128.onnx"
        self.faces_index = {face_index}
        self.ffmpeg_binary = self.find_ffmpeg_binary()
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), self.model)
        self.face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        self.face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        self.mel_step_size = 16
        if audio is not None:
            self.nb_frame = self.calc_frame()

    def calc_frame(self):

        video_stream = cv2.VideoCapture(self.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        wav = audio.load_wav(self.audio, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        return len(mel_chunks)

    def convert_to_sd(self, img):
        shapes = []
        chunks = detect(img)
        for chunk in chunks:
            shapes.append(chunk["score"] > 0.7)
        return [any(shapes), tempfile.NamedTemporaryFile(delete=False, suffix=".png")]

    def find_ffmpeg_binary(self):
        for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
            try:
                package_path = resource_filename(package, 'binaries')
                files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0] if files else 'ffmpeg'
            except:
                return 'ffmpeg'

    def get_framerate(self, video_file):
        video = cv2.VideoCapture(video_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps

    def create_video_from_images(self, nb_frames):
        fps = str(self.get_framerate(self.face))
        command = [self.ffmpeg_binary, "-y", "-framerate", fps, "-start_number", "0", "-i",
                   self.faceswap_output_folder + "/face_swap_%05d.png", "-vframes",
                   str(nb_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "8000k",
                   self.faceswap_output_folder + "/video.mp4"]

        self.execute_command(command)

    def execute_command(self, command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr)

    def get_face_single(self, img_data: np.ndarray, face_index=0, det_size=(640, 640)):
        face = self.face_analyser.get(img_data)
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = (det_size[0] // 2, det_size[1] // 2)
            self.face_analyser.prepare(ctx_id=0, det_size=det_size_half)
            face = self.face_analyser.get(img_data)
            self.face_analyser.prepare(ctx_id=0, det_size=det_size)
            try:
                return sorted(face, key=lambda x: x.bbox[0])[face_index]
            except IndexError:
                return None
        try:
            return sorted(face, key=lambda x: x.bbox[0])[face_index]
        except IndexError:
            return None

    def swap_face(self,
                  source_img: Image.Image,
                  target_img: Image.Image,
                  model: Union[str, None] = None,
                  faces_index: Set[int] = {0},
                  upscale_options: Union[UpscaleOptions, None] = None,
                  ) -> ImageResult:
        result_image = target_img
        converted = self.convert_to_sd(target_img)
        scale, fn = converted[0], converted[1]
        if model is not None and not scale:
            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
            source_face = self.get_face_single(source_img, face_index=0)
            if source_face is not None:
                result = target_img
                for face_num in faces_index:
                    target_face = self.get_face_single(target_img, face_index=face_num)
                    if target_face is not None:
                        result = self.face_swapper.get(result, target_face, source_face)
                    else:
                        print(f"No target face found for {face_num}")
                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                print("No source face found")
        result_image.save(fn.name)
        return ImageResult(path=fn.name)

    def resume(self):
        return self.faceswap_output_folder + "/video.mp4"

    def generate(self):
        original_codeformer_weight = opts.code_former_weight
        original_face_restoration_model = opts.face_restoration_model

        opts.code_former_weight = self.code_former_weight
        opts.face_restoration_model = self.face_restore_model
        video_stream = cv2.VideoCapture(self.face)

        print('Reading video frames for face swap...')
        frame_number = 0

        while frame_number != self.nb_frame+1:
            f_number = str(frame_number).rjust(5, '0')
            print("[INFO] Processing frame: " + str(frame_number) + " of " + str(self.nb_frame) + " - ", end="\r")
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if self.resize_factor > 1:
                frame = cv2.resize(frame,
                                   (frame.shape[1] // self.resize_factor, frame.shape[0] // self.resize_factor))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.swap_face(
                self.source,
                frame,
                faces_index=self.faces_index,
                model=self.model,
                upscale_options=None
            )
            # copy image to output folder
            face_swapped = cv2.imread(result.path)
            face_swapped = cv2.cvtColor(face_swapped, cv2.COLOR_RGB2BGR)
            image_restored = modules.face_restoration.restore_faces(face_swapped)
            image_restored2 = cv2.cvtColor(image_restored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.faceswap_output_folder + "/face_swap_" + f_number + ".png", image_restored2)

            frame_number += 1

        self.create_video_from_images(frame_number - 1)
        opts.code_former_weight = original_codeformer_weight
        opts.face_restoration_model = original_face_restoration_model
        return self.faceswap_output_folder + "/video.mp4"
