from scripts.wav2lip_uhq_extend_paths import wav2lip_uhq_sys_extend
import gradio as gr
from scripts.wav2lip.w2l import W2l
from scripts.wav2lip.wav2lip_uhq import Wav2LipUHQ
from modules.shared import opts, state

def on_ui_tabs():
    wav2lip_uhq_sys_extend()

    with gr.Blocks(analytics_enabled=False) as wav2lip_uhq_interface:
        gr.Markdown("<div align='center'> <h3> Follow installation instructions <a href='https://github.com/numz/sd-wav2lip-uhq'> here </a> </h3> </div>")
        with gr.Row():
            video = gr.File(label="Video or Image", info="Filepath of video/image that contains faces to use")
            audio = gr.File(label="Audio", info="Filepath of video/audio file to use as raw audio source")
            with gr.Column():
                checkpoint = gr.Radio(["wav2lip", "wav2lip_gan"], value="wav2lip_gan",  label="Checkpoint", info="Name of saved checkpoint to load weights from")
                no_smooth = gr.Checkbox(label="No Smooth", info="Prevent smoothing face detections over a short temporal window")
                resize_factor = gr.Slider(minimum=1, maximum=4, step=1, label="Resize Factor", info="Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p")
                generate_btn = gr.Button("Generate")
                interrupt_btn = gr.Button('Interrupt', elem_id=f"interrupt", visible=True)

        with gr.Row():
            with gr.Column():
                pad_top = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Top", info="Padding above lips")
                pad_bottom = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Bottom", info="Padding below lips")
                pad_left = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Left", info="Padding to the left of lips")
                pad_right = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Right", info="Padding to the right of lips")

            with gr.Column():
                with gr.Tabs(elem_id="wav2lip_generated"):
                    result = gr.Video(label="Generated video", format="mp4").style(width=256)


        def on_interrupt():
            state.interrupt()
            return "Interrupted"

        def generate(video, audio, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left, pad_right):
            state.begin()
            if video is None or audio is None or checkpoint is None:
                return
            w2l = W2l(video.name, audio.name, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left, pad_right)
            w2l.execute()

            w2luhq = Wav2LipUHQ(video.name, audio.name)
            w2luhq.execute()
            return "extensions\\sd-wav2lip-uhq\\scripts\\wav2lip\\output\\output_video.mp4"

        generate_btn.click(
            generate, 
            [video, audio, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left, pad_right], 
            result)

        interrupt_btn.click(on_interrupt)
        
    return [(wav2lip_uhq_interface, "Wav2lip Uhq", "wav2lip_uhq_interface")]