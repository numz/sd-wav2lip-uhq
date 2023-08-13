import os
import numpy as np
import cv2
import dlib
import torch
import scripts.wav2lip.face_detection as face_detection
from imutils import face_utils
import subprocess
from modules.shared import state
from pkg_resources import resource_filename
import modules.face_restoration


class Wav2LipUHQ:
    def __init__(self, face, audio,mouth_mask_dilatation,erode_face_mask, mask_blur, only_mouth, resize_factor, debug=False):
        self.wav2lip_folder = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
        self.original_video = face
        self.audio = audio
        self.mouth_mask_dilatation = mouth_mask_dilatation
        self.erode_face_mask = erode_face_mask
        self.mask_blur = mask_blur
        self.only_mouth = only_mouth
        self.w2l_video = self.wav2lip_folder + '/results/result_voice.mp4'
        self.original_is_image = self.original_video.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ffmpeg_binary = self.find_ffmpeg_binary()
        self.resize_factor = resize_factor
        self.debug = debug

    def find_ffmpeg_binary(self):
        for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
            try:
                package_path = resource_filename(package, 'binaries')
                files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0] if files else 'ffmpeg'
            except:
                return 'ffmpeg'

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
        command = [self.ffmpeg_binary, "-y", "-framerate", fps, "-start_number", "0", "-i",
                   self.wav2lip_folder + "/output/final/output_%05d.png", "-vframes",
                   str(nb_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "8000k",
                   self.wav2lip_folder + "/output/video.mp4"]

        self.execute_command(command)

    def extract_audio_from_video(self):
        command = [self.ffmpeg_binary, "-y", "-i", self.w2l_video, "-vn", "-acodec", "copy",
                   self.wav2lip_folder + "/output/output_audio.aac"]
        self.execute_command(command)

    def add_audio_to_video(self):
        command = [self.ffmpeg_binary, "-y", "-i", self.wav2lip_folder + "/output/video.mp4", "-i",
                   self.wav2lip_folder + "/output/output_audio.aac", "-c:v", "copy", "-c:a", "aac", "-strict",
                   "experimental", self.wav2lip_folder + "/output/output_video.mp4"]
        self.execute_command(command)

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
        kernel = np.ones((self.mouth_mask_dilatation, self.mouth_mask_dilatation), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_points = contours[0].squeeze()
        return dilated_points

    def execute(self):
        output_dir = self.wav2lip_folder + '/output/'
        debug_path = output_dir + "debug/"
        final_path = self.wav2lip_folder + '/output/final/'
        detector, predictor = self.initialize_dlib_predictor()
        vs, vi = self.initialize_video_streams()
        (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (jstart, jend) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (nstart, nend) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        max_frame = str(int(vs.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_number = 0

        while True:
            print("[INFO] Processing frame: " + str(frame_number) + " of " + max_frame + " - ", end="\r")
            if state.interrupted:
                break

            # Read frame
            ret, w2l_frame = vs.read()
            if not ret:
                break

            if self.original_is_image:
                original_frame = vi
            else:
                vi.set(cv2.CAP_PROP_POS_FRAMES, (frame_number % int(vi.get(cv2.CAP_PROP_FRAME_COUNT))) - 1)
                ret, original_frame = vi.read()

            # Convert to gray
            w2l_gray = cv2.cvtColor(w2l_frame, cv2.COLOR_RGB2GRAY)
            original_gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
            if w2l_gray.shape != original_gray.shape:
                if self.resize_factor > 1:
                    original_gray = cv2.resize(original_gray, (w2l_gray.shape[1], w2l_gray.shape[0]))
                    original_frame = cv2.resize(original_frame, (w2l_gray.shape[1], w2l_gray.shape[0]))
                else:
                    w2l_gray = cv2.resize(w2l_gray, (original_gray.shape[1], original_gray.shape[0]))
                    w2l_frame = cv2.resize(w2l_frame, (original_gray.shape[1], original_gray.shape[0]))

            # Calculate diff between frames and apply threshold
            diff = np.abs(original_gray - w2l_gray)
            diff[diff > 10] = 255
            diff[diff <= 10] = 0

            # Detect faces
            rects = detector.get_detections_for_batch(np.array([np.array(w2l_frame)]))

            # Initialize mask
            mask = np.zeros_like(diff)

            # Process each detected face
            for (i, rect) in enumerate(rects):
                # Get face coordinates
                if not self.only_mouth:
                    shape = predictor(original_gray, dlib.rectangle(*rect))
                    shape = face_utils.shape_to_np(shape)
                    jaw = shape[jstart:jend][1:-1]
                    nose = shape[nstart:nend][2]

                # Get mouth coordinates
                shape = predictor(w2l_gray, dlib.rectangle(*rect))
                shape = face_utils.shape_to_np(shape)

                mouth = shape[mstart:mend][:-8]
                mouth = np.delete(mouth, [3], axis=0)
                if self.mouth_mask_dilatation > 0:
                    mouth = self.dilate_mouth(mouth, original_gray.shape[0], original_gray.shape[1])

                # Create mask for face
                if not self.only_mouth:
                    external_shape = np.append(jaw, [nose], axis=0)
                    external_shape_pts = external_shape.reshape((-1, 1, 2))
                    mask = cv2.fillPoly(mask, [external_shape_pts], 255)
                    if self.erode_face_mask > 0:
                        kernel = np.ones((self.erode_face_mask, self.erode_face_mask), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                    masked_diff = cv2.bitwise_and(diff, diff, mask=mask)
                else:
                    masked_diff = mask

                # Create mask for mouth
                cv2.fillConvexPoly(masked_diff, mouth, 255)

                # Save mask
                if self.mask_blur > 0:
                    blur = self.mask_blur if self.mask_blur % 2 == 1 else self.mask_blur - 1
                    masked_save = cv2.GaussianBlur(masked_diff, (blur, blur), 0)
                else:
                    masked_save = masked_diff

                # Prepare for restoration
                masked_diff = np.uint8(masked_diff / 255)
                masked_diff = cv2.cvtColor(masked_diff, cv2.COLOR_GRAY2BGR)
                dst = w2l_frame * masked_diff
                original = original_frame.copy()
                original_frame = original_frame * (1 - masked_diff) + dst

                # Restore face
                image_restored = modules.face_restoration.restore_faces(original_frame)

                # Apply restored face to original image with mask attention
                extended_mask = np.stack([masked_save]*3, axis=-1)
                normalized_mask = extended_mask / 255.0
                dst2 = image_restored * normalized_mask
                original = original * (1 - normalized_mask) + dst2
                original = original.astype(np.uint8)

                # Save final image
                cv2.imwrite(final_path + "output_" + str(frame_number).rjust(5, '0') + ".png", original)

                if self.debug:
                    clone = w2l_frame.copy()
                    if not self.only_mouth:
                        for (x, y) in np.concatenate((jaw, mouth, [nose])):
                            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    else:
                        for (x, y) in mouth:
                            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                    f_number = str(frame_number).rjust(5, '0')
                    cv2.imwrite(debug_path + "diff_" + f_number+ ".png", diff)
                    cv2.imwrite(debug_path + "points_" + f_number + ".png", clone)
                    cv2.imwrite(debug_path + 'mask_' + f_number + '.png', masked_save)
                    cv2.imwrite(debug_path + "dst_" + f_number + ".png", dst)
                    cv2.imwrite(debug_path + 'image_' + f_number + '.png', original_frame)
                    cv2.imwrite(debug_path + "face_restore_" + f_number + ".png", image_restored)
                    cv2.imwrite(debug_path + "dst2_" + f_number + ".png", dst2)

            frame_number += 1

        if not state.interrupted:
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
            return self.wav2lip_folder + "/output/output_video.mp4"
        else:
            print("[INFO] Interrupted!")
            return None
