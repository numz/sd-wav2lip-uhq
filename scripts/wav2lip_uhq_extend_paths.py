import os
import sys


def wav2lip_uhq_sys_extend():
    wav2lip_uhq_folder_name = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

    basedirs = [os.getcwd()]
    for _ in basedirs:
        wav2lip_uhq_paths_to_ensure = [os.path.join(wav2lip_uhq_folder_name, 'scripts')]
        for wav2lip_uhq_scripts_path_fix in wav2lip_uhq_paths_to_ensure:
            if wav2lip_uhq_scripts_path_fix not in sys.path:
                sys.path.extend([wav2lip_uhq_scripts_path_fix])
