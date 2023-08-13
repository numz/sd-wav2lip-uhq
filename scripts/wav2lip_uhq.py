from modules import script_callbacks
from scripts.wav2lip_uhq_extend_paths import wav2lip_uhq_sys_extend


def init_wav2lip_uhq():
    wav2lip_uhq_sys_extend()
    from ui import on_ui_tabs
    script_callbacks.on_ui_tabs(on_ui_tabs)


init_wav2lip_uhq()
