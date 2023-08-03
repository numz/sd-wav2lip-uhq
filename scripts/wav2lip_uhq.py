import os
from modules import script_callbacks
import modules.paths as ph
from scripts.wav2lip_uhq_extend_paths import wav2lip_uhq_sys_extend

def init_wav2lip_uhq():
    wav2lip_uhq_sys_extend()
    # import our on_ui_tabs and on_ui_settings functions from the respected files
    from ui import on_ui_tabs

    # trigger webui's extensions mechanism using our imported main functions -
    # first to create the actual deforum gui, then to make the deforum tab in webui's settings section
    script_callbacks.on_ui_tabs(on_ui_tabs)

init_wav2lip_uhq()
