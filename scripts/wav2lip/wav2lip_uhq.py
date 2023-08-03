import os
import requests
import base64
import io
import numpy as np
from PIL import Image
import cv2
import dlib
import torch
import scripts.wav2lip.face_detection as face_detection
from imutils import face_utils
import subprocess

from modules import processing


class Wav2LipUHQ:
    def __init__(self, face, audio):
        self.wav2lip_folder = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
        self.original_video = face
        self.audio = audio
        self.w2l_video = self.wav2lip_folder + '/results/result_voice.mp4'
        self.original_is_image = self.original_video.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def assure_path_exists(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    def get_framerate(self, video_file):
        video = cv2.VideoCapture(video_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps

    def execute_command(self, command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr)

    def create_video_from_images(self, nb_frames):
        fps = str(self.get_framerate(self.w2l_video))
        command = ["ffmpeg", "-y", "-framerate", fps, "-start_number", "0", "-i",
                   self.wav2lip_folder + "/output/final/output_%05d.png", "-vframes",
                   str(nb_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "8000k",
                   self.wav2lip_folder + "/output/video.mp4"]

        self.execute_command(command)

    def extract_audio_from_video(self):
        command = ["ffmpeg", "-y", "-i", self.w2l_video, "-vn", "-acodec", "copy",
                   self.wav2lip_folder + "/output/output_audio.aac"]
        self.execute_command(command)

    def add_audio_to_video(self):
        command = ["ffmpeg", "-y", "-i", self.wav2lip_folder + "/output/video.mp4", "-i",
                   self.wav2lip_folder + "/output/output_audio.aac", "-c:v", "copy", "-c:a", "aac", "-strict",
                   "experimental", self.wav2lip_folder + "/output/output_video.mp4"]
        self.execute_command(command)

    def create_image(self, image, mask, payload, shape, img_count):
        output_dir = self.wav2lip_folder + '/output/final/'
        image = open(image, "rb").read()
        image_mask = open(mask, "rb").read()
        url = payload["url"]
        payload = payload["payload"]
        payload["init_images"] = ["data:image/png;base64," + base64.b64encode(image).decode('UTF-8')]
        payload["mask"] = "data:image/png;base64," + base64.b64encode(image_mask).decode('UTF-8')

        path = output_dir
        response = requests.post(url=f'{url}', json=payload)
        r = response.json()
        for idx in range(len(r['images'])):
            i = r['images'][idx]
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
            image_name = path + "output_" + str(img_count).rjust(5, '0') + ".png"
            image.save(image_name)

    def create_img(self, image, mask, shape, img_count):
        image = Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('RGB')
        output_dir = self.wav2lip_folder + '/output/final/'
        p = processing.StableDiffusionProcessingImg2Img(
            outpath_samples=output_dir,
        )  # we'll set up the rest later

        p.c = None
        p.extra_network_data = None
        p.image_conditioning = None
        p.init_latent = None
        p.mask_for_overlay = None
        p.negative_prompts = None
        p.nmask = None
        p.overlay_images = None
        p.paste_to = None
        p.prompts = None
        p.width, p.height = shape[0], shape[1]
        p.steps = 150
        p.seed = 65541238
        p.seed_resize_from_h = 0
        p.seed_resize_from_w = 0
        p.seeds = None
        p.subseeds = None
        p.uc = None
        p.sampler = None
        p.sampler_name = "Euler a"
        p.tiling = False
        p.restore_faces = True
        p.do_not_save_samples = True
        p.mask_blur = 4
        p.extra_generation_params["Mask blur"] = 4
        p.denoising_strength = 0
        p.cfg_scale = 1
        p.inpainting_mask_invert = 0
        p.inpainting_fill = 1
        p.inpaint_full_res = 0
        p.inpaint_full_res_padding = 32

        p.init_images = [image]
        p.image_mask = mask

        processed = processing.process_images(p)
        results = processed.images[0]
        image_name = output_dir + "output_" + str(img_count).rjust(5, '0') + ".png"
        results.save(image_name)

    def initialize_dlib_predictor(self):
        print("[INFO] Loading the predictor...")
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=self.device)
        predictor = dlib.shape_predictor(self.wav2lip_folder + "/predicator/shape_predictor_68_face_landmarks.dat")
        return detector, predictor

    def initialize_video_streams(self):
        print("[INFO] Loading File...")
        vs = cv2.VideoCapture(self.w2l_video)
        if self.original_is_image:
            vi = cv2.imread(self.original_video)
        else:
            vi = cv2.VideoCapture(self.original_video)
        return vs, vi

    def dilate_mouth(self, mouth, w, h):
        mask = np.zeros((w, h), dtype=np.uint8)
        cv2.fillPoly(mask, [mouth], 255)
        kernel = np.ones((10, 10), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_points = contours[0].squeeze()
        return dilated_points

    def execute(self):
        output_dir = self.wav2lip_folder + '/output/'
        image_path = output_dir + "images/"
        mask_path = output_dir + "masks/"
        debug_path = output_dir + "debug/"

        detector, predictor = self.initialize_dlib_predictor()
        vs, vi = self.initialize_video_streams()
        (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (jstart, jend) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (nstart, nend) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        max_frame = str(int(vs.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_number = 0

        while True:
            print("Processing frame: " + str(frame_number) + " of " + max_frame)

            ret, w2l_frame = vs.read()
            if not ret:
                break

            if self.original_is_image:
                original_frame = vi
            else:
                ret, original_frame = vi.read()
                if not ret:
                    break

            w2l_gray = cv2.cvtColor(w2l_frame, cv2.COLOR_RGB2GRAY)
            original_gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
            if w2l_gray.shape != original_gray.shape:
                w2l_gray = cv2.resize(w2l_gray, (original_gray.shape[1], original_gray.shape[0]))
                w2l_frame = cv2.resize(w2l_frame, (original_gray.shape[1], original_gray.shape[0]))

            diff = np.abs(original_gray - w2l_gray)
            diff[diff > 10] = 255
            diff[diff <= 10] = 0
            cv2.imwrite(debug_path + "diff_" + str(frame_number) + ".png", diff)

            rects = detector.get_detections_for_batch(np.array([np.array(w2l_frame)]))
            mask = np.zeros_like(diff)

            # Process each detected face
            for (i, rect) in enumerate(rects):
                # copy pixel from diff to mask where pixel is in rects
                shape = predictor(original_gray, dlib.rectangle(*rect))
                shape = face_utils.shape_to_np(shape)

                jaw = shape[jstart:jend][1:-1]
                nose = shape[nstart:nend][2]

                shape = predictor(w2l_gray, dlib.rectangle(*rect))
                shape = face_utils.shape_to_np(shape)

                mouth = shape[mstart:mend][:-8]
                mouth = np.delete(mouth, [3], axis=0)
                mouth = self.dilate_mouth(mouth, original_gray.shape[0], original_gray.shape[1])
                # affiche les points sur un clone de l'image
                clone = w2l_frame.copy()
                for (x, y) in np.concatenate((jaw, mouth, [nose])):
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    cv2.imwrite(debug_path + "points_" + str(frame_number) + ".png", clone)

                external_shape = np.append(jaw, [nose], axis=0)
                kernel = np.ones((3, 3), np.uint8)
                external_shape_pts = external_shape.reshape((-1, 1, 2))
                mask = cv2.fillPoly(mask, [external_shape_pts], 255)
                mask = cv2.erode(mask, kernel, iterations=5)
                masked_diff = cv2.bitwise_and(diff, diff, mask=mask)
                cv2.fillConvexPoly(masked_diff, mouth, 255)
                masked_save = cv2.GaussianBlur(masked_diff, (15, 15), 0)

                cv2.imwrite(mask_path + 'image_' + str(frame_number).rjust(5, '0') + '.png', masked_save)
                masked_diff = np.uint8(masked_diff / 255)
                masked_diff = cv2.cvtColor(masked_diff, cv2.COLOR_GRAY2BGR)
                dst = w2l_frame * masked_diff
                cv2.imwrite(debug_path + "dst_" + str(frame_number) + ".png", dst)
                original_frame = original_frame * (1 - masked_diff) + dst

                height, width, _ = original_frame.shape
                image_name = image_path + 'image_' + str(frame_number).rjust(5, '0') + '.png'
                mask_name = mask_path + 'image_' + str(frame_number).rjust(5, '0') + '.png'
                cv2.imwrite(image_name, original_frame)
                self.create_img(image_name, mask_name, (width, height), frame_number)

            frame_number += 1

        cv2.destroyAllWindows()
        vs.release()
        if not self.original_is_image:
            vi.release()

        print("[INFO] Create Video output!")
        self.create_video_from_images(frame_number - 1)
        print("[INFO] Extract Audio from input!")
        self.extract_audio_from_video()
        print("[INFO] Add Audio to Video!")
        self.add_audio_to_video()

        print("[INFO] Done! file save in output/video_output.mp4")
