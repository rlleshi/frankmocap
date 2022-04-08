import schedule
import subprocess


def pose_extract():
    script_path = 'mocap_utils/generate_3D_dataset.py'
    subargs = [
        'python',
        script_path,
        '--src-dir',
        '../data/vortanz/kinesphere_5s/',
        '--out-dir',
        'mocap_output/vortanz/kinesphere_5s/810x456/',
        '--level',
        '3'
    ]
    subprocess.run(subargs)
    return schedule.CancelJob

schedule.every().tuesday.at('20:00').do(pose_extract)

while True:
    schedule.run_pending()
