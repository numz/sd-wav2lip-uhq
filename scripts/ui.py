from scripts.wav2lip_uhq_extend_paths import wav2lip_uhq_sys_extend
import gradio as gr
from scripts.wav2lip.w2l import W2l
from scripts.wav2lip.wav2lip_uhq import Wav2LipUHQ
from modules.shared import state


def on_ui_tabs():
    wav2lip_uhq_sys_extend()

    with gr.Blocks(analytics_enabled=False) as wav2lip_uhq_interface:
        gr.Markdown(
            "<div align='center'> <h3><a href='https://github.com/numz/sd-wav2lip-uhq'> Follow installation instructions here </a> </h3> </div>")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    video = gr.Video(label="Video", format="mp4",
                                     info="Filepath of video/image that contains faces to use",
                                     file_types=["mp4", "png", "jpg", "jpeg", "avi"])
                    audio = gr.Audio(label="Speech",
                                     info="Filepath of audio file to use as raw audio source",
                                     type="filepath")
                with gr.Row():
                    checkpoint = gr.Radio(["wav2lip", "wav2lip_gan"], value="wav2lip_gan", label="Checkpoint",
                                          info="Name of saved checkpoint to load weights fvrom")
                    no_smooth = gr.Checkbox(label="No Smooth", info="Prevent smoothing face detections")
                    only_mouth = gr.Checkbox(label="Only Mouth", info="Only track the mouth")
                    active_debug = gr.Checkbox(label="Active Debug", info="Active Debug")
                with gr.Row():
                    with gr.Column():
                        resize_factor = gr.Slider(minimum=1, maximum=4, step=1, label="Resize Factor",
                                                  info="Reduce the resolution by this factor.")
                        mouth_mask_dilatation = gr.Slider(minimum=0, maximum=64, step=1, value=15,
                                                          label="Mouth Mask Dilate",
                                                          info="Dilatation of the mask around the mouth (in pixels)")
                        erode_face_mask = gr.Slider(minimum=0, maximum=64, step=1, value=15, label="Face Mask Erode",
                                                    info="Erode the mask around the face (in pixels)")
                        mask_blur = gr.Slider(minimum=0, maximum=64, step=1, value=15, label="Mask Blur",
                                              info="Kernel size of Gaussian blur for masking")
                        code_former_weight = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.75,
                                                       label="Code Former Fidelity",
                                                       info="0 for better quality, 1 for better identity")
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
                        wav2lip_video = gr.Video(label="Wav2Lip video", format="mp4")
                        restore_video = gr.Video(label="Restored face video", format="mp4")
                        result = gr.Video(label="Generated video", format="mp4")
                generate_btn = gr.Button("Generate")
                interrupt_btn = gr.Button('Interrupt', elem_id=f"interrupt", visible=True)

        def on_interrupt():
            state.interrupt()
            return "Interrupted"

        def generate(video, audio, checkpoint, no_smooth, only_mouth, resize_factor, mouth_mask_dilatation,
                     erode_face_mask, mask_blur, pad_top, pad_bottom, pad_left, pad_right, active_debug,code_former_weight):
            state.begin()
            if video is None or audio is None or checkpoint is None:
                return
            w2l = W2l(video, audio, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left,
                      pad_right)
            w2l.execute()

            w2luhq = Wav2LipUHQ(video, audio, mouth_mask_dilatation, erode_face_mask, mask_blur, only_mouth,
                                resize_factor, code_former_weight, active_debug)

            return w2luhq.execute()

        generate_btn.click(
            generate,
            [video, audio, checkpoint, no_smooth, only_mouth, resize_factor, mouth_mask_dilatation,
             erode_face_mask, mask_blur, pad_top, pad_bottom, pad_left, pad_right, active_debug, code_former_weight],
            [wav2lip_video, restore_video, result])

        interrupt_btn.click(on_interrupt)

    return [(wav2lip_uhq_interface, "Wav2lip Uhq", "wav2lip_uhq_interface")]
