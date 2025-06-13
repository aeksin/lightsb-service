import os.path

from generator import generate_ALAE, generate_sd
from head_detection import head_detector
from ultralytics import YOLO

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def head_detection_cleanup():
    detected__heads_files = os.scandir(f"{SCRIPT_PATH}/head_detection_results/")
    for file in detected__heads_files:

        if file.is_file() and os.path.exists(file):
            os.remove(file)


def aging_pipeline(
    filenames,
    output_path=f"{SCRIPT_PATH}/result",
    generation_model="ALAE",
    result_count=5,
    delta=-4,
):
    if type(filenames) == str:
        filenames = [filenames]

    filenames = [os.path.abspath(filename) for filename in filenames]
    output_path = os.path.abspath(output_path)
    checkpoint_path = f"{SCRIPT_PATH}/yolo11s_weights.pt"

    model = YOLO(checkpoint_path)
    hd_model = head_detector(model)
    detected_heads = hd_model.predict(filenames)
    if "ALAE" in generation_model:
        aged_people = generate_ALAE(
            detected_heads, output_path, generation_model, result_count, delta
        )
    elif "Stable Diffusion" in generation_model:
        aged_people = generate_sd(detected_heads, output_path)
    head_detection_cleanup()

    return aged_people
