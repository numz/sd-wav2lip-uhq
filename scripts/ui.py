import json
from scripts.wav2lip_uhq_extend_paths import wav2lip_uhq_sys_extend
import gradio as gr
from scripts.wav2lip.w2l import W2l
from scripts.wav2lip.wav2lip_uhq import Wav2LipUHQ
from modules.shared import state
from scripts.bark.tts import TTS
from scripts.faceswap.swap import FaceSwap

speaker_id = "v2/en_speaker_0"


def on_ui_tabs():
    wav2lip_uhq_sys_extend()
    speaker_json = json.load(open("extensions/sd-wav2lip-uhq/scripts/bark/speakers.json", "r"))
    speaker_list = [speaker["name"] for speaker in speaker_json if
                    speaker["language"] == "English" and speaker["gender"] == "Male"]
    speaker_language = list(set([speaker["language"] for speaker in speaker_json]))
    speaker_gender = list(set([speaker["gender"] for speaker in speaker_json]))

    def update_speaker_list(new_language, new_gender):
        # Mettez à jour la liste des speakers basée sur la langue et le genre sélectionnés
        global speaker_id
        new_speaker_list = [speaker["name"] for speaker in speaker_json if
                            speaker["language"] == new_language and speaker["gender"] == new_gender]
        audio_mp3 = [speaker["prompt_audio"] for speaker in speaker_json if speaker["name"] in new_speaker_list[0]][0]
        speaker_id = [speaker["id"] for speaker in speaker_json if speaker["name"] in new_speaker_list[0]][0]
        return [gr.Dropdown.update(choices=new_speaker_list, value=new_speaker_list[0]),
                gr.Audio.update(value=audio_mp3), gr.Dropdown.update(value=new_language)]

    def select_speaker(speaker):
        # Mettez à jour l'audio basé sur le speaker sélectionné
        global speaker_id
        audio_mp3 = [sp["prompt_audio"] for sp in speaker_json if sp["name"] == speaker][0]
        speaker_id = [sp["id"] for sp in speaker_json if sp["name"] == speaker][0]
        return gr.Audio.update(value=audio_mp3)

    with gr.Blocks(analytics_enabled=False) as wav2lip_uhq_interface:
        gr.Markdown(
            "<div align='center'> <h3><a href='https://github.com/numz/sd-wav2lip-uhq'> Follow installation instructions here </a> </h3> </div>")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        video = gr.Video(label="Video", format="mp4",
                                         info="Filepath of video/image that contains faces to use",
                                         file_types=["mp4", "png", "jpg", "jpeg", "avi"])
                        face_swap_img = gr.Image(label="Face Swap", type="pil")
                        face_index_slider = gr.Slider(minimum=0, maximum=20, step=1, value=0, label="Face index",
                                                    info="index of face to swap, left face in image is 0")

                    with gr.Column():
                        with gr.Row():
                            language = gr.Dropdown(
                                speaker_language, label="Language", info="Select the language to use",
                                value="English"
                            )
                            gender = gr.Dropdown(
                                speaker_gender, label="Gender", info="Select gender", value="Male"
                            )
                        with gr.Row():
                            speaker = gr.Dropdown(
                                speaker_list, label="Speaker", info="Select the speaker to use",
                                value=speaker_list[0]
                            )
                            low_vram = gr.Radio(["False", "True"], value="True", label="Low VRAM",
                                                info="Less than 16GB of VRAM, set True")
                        with gr.Row():
                            audio_example = gr.Audio(label="Audio example",
                                                     value="https://dl.suno-models.io/bark/prompts/prompt_audio/en_speaker_0.mp3")
                        with gr.Column():
                            suno_prompt = gr.Textbox(label="Prompt", placeholder="Prompt", lines=5, type="text",info="Don't forget that bark can only generate 14 seconds of audio at a time, so for long text, you need to use [split] to split the text into multiple prompts")
                            temperature = gr.Slider(label="Generation temperature", minimum=0.01, maximum=1, step=0.01, value=0.7,
                                                  info="1.0 more diverse, 0.0 more conservative")
                            silence = gr.Slider(label="Silence", minimum=0, maximum=1, step=0.01, value=0.25, info="Silence after [split] in seconde")
                            generate_audio = gr.Button("Generate")
                            audio = gr.Audio(label="Speech", type="filepath")

                        # if language changed, update speaker list
                        language.change(update_speaker_list, [language, gender], [speaker, audio_example])
                        gender.change(update_speaker_list, [language, gender], [speaker, audio_example])
                        speaker.change(select_speaker, speaker, audio_example)

                with gr.Row():
                    checkpoint = gr.Radio(["wav2lip", "wav2lip_gan"], value="wav2lip_gan", label="Checkpoint",
                                          info="Wav2lip model to use")
                    face_restore_model = gr.Radio(["CodeFormer", "GFPGAN"], value="GFPGAN",
                                                  label="Face Restoration Model",
                                                  info="Model to use")

                with gr.Row():
                    no_smooth = gr.Checkbox(label="No Smooth", info="Prevent smoothing face detections")
                    only_mouth = gr.Checkbox(label="Only Mouth", info="Only track the mouth")
                    active_debug = gr.Checkbox(label="Active Debug", info="Active Debug")
                with gr.Row():
                    with gr.Column():
                        resize_factor = gr.Slider(minimum=1, maximum=4, step=1, label="Resize Factor",
                                                  info="Reduce the resolution by this factor.")
                        mouth_mask_dilatation = gr.Slider(minimum=0, maximum=128, step=1, value=15,
                                                          label="Mouth Mask Dilate",
                                                          info="Dilatation of the mask around the mouth (in pixels)")
                        erode_face_mask = gr.Slider(minimum=0, maximum=128, step=1, value=15, label="Face Mask Erode",
                                                    info="Erode the mask around the face (in pixels)")
                        mask_blur = gr.Slider(minimum=0, maximum=128, step=1, value=15, label="Mask Blur",
                                              info="Kernel size of Gaussian blur for masking")
                        code_former_weight = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.75,
                                                       label="Code Former Fidelity",
                                                       info="0 for better quality, 1 for better identity (Effect only if codeformer is selected)")
                    with gr.Column():
                        pad_top = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Top",
                                            info="Padding above lips")
                        pad_bottom = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Bottom",
                                               info="Padding below lips")
                        pad_left = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Left",
                                             info="Padding to the left of lips")
                        pad_right = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Right",
                                              info="Padding to the right of lips")

            with gr.Column():
                with gr.Tabs(elem_id="wav2lip_generated"):
                    with gr.Row():
                        faceswap_video = gr.Video(label="faceSwap video", format="mp4")
                        wav2lip_video = gr.Video(label="Wav2Lip video", format="mp4")
                        restore_video = gr.Video(label="Restored face video", format="mp4")
                        result = gr.Video(label="Generated video", format="mp4")
                generate_btn = gr.Button("Generate")
                interrupt_btn = gr.Button('Interrupt', elem_id=f"interrupt", visible=True)
                resume_btn = gr.Button('Resume', elem_id=f"resume", visible=True)

        def on_interrupt():
            state.interrupt()
            return "Interrupted"

        def gen_audio(suno_prompt, temperature, silence, low_vram):
            global speaker_id
            if suno_prompt is None or speaker_id is None:
                return
            tts = TTS(suno_prompt, speaker_id, temperature, silence,None, low_vram)
            wav = tts.generate()
            # delete tts object to free memory
            del tts

            return wav

        def generate(video, face_swap_img, face_index, audio, checkpoint, face_restore_model, no_smooth, only_mouth, resize_factor,
                     mouth_mask_dilatation, erode_face_mask, mask_blur, pad_top, pad_bottom, pad_left, pad_right,
                     active_debug, code_former_weight):
            state.begin()

            if video is None or audio is None:
                print("[ERROR] Please select a video and an audio file")
                return

            if face_swap_img is not None:
                face_swap = FaceSwap(video, audio, face_index, face_swap_img, resize_factor, face_restore_model, code_former_weight)
                video = face_swap.generate()

            w2l = W2l(video, audio, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left,
                      pad_right, face_swap_img)
            w2l.execute()

            w2luhq = Wav2LipUHQ(video, face_restore_model, mouth_mask_dilatation, erode_face_mask, mask_blur,
                                only_mouth, face_swap_img, resize_factor, code_former_weight, active_debug)

            return w2luhq.execute()

        def resume(video,face_swap_img, face_restore_model, only_mouth, resize_factor, mouth_mask_dilatation, erode_face_mask,
                   mask_blur, active_debug, code_former_weight):
            state.begin()
            if face_swap_img is not None:
                face_swap = FaceSwap()
                video = face_swap.resume()
            w2luhq = Wav2LipUHQ(video, face_restore_model, mouth_mask_dilatation, erode_face_mask, mask_blur,
                                only_mouth, face_swap_img, resize_factor, code_former_weight, active_debug)

            return w2luhq.execute(True)

        generate_audio.click(
            gen_audio,
            [suno_prompt, temperature, silence, low_vram],
            audio)

        generate_btn.click(
            generate,
            [video, face_swap_img, face_index_slider, audio, checkpoint, face_restore_model, no_smooth, only_mouth, resize_factor, mouth_mask_dilatation,
             erode_face_mask, mask_blur, pad_top, pad_bottom, pad_left, pad_right, active_debug, code_former_weight],
            [faceswap_video, wav2lip_video, restore_video, result])

        resume_btn.click(
            resume,
            [video,face_swap_img, face_restore_model, only_mouth, resize_factor, mouth_mask_dilatation, erode_face_mask,
             mask_blur, active_debug, code_former_weight],
            [faceswap_video, wav2lip_video, restore_video, result])

        interrupt_btn.click(on_interrupt)

    return [(wav2lip_uhq_interface, "Wav2lip Studio", "wav2lip_uhq_interface")]
