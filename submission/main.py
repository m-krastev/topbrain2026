"""
You should NOT need to edit this file. Instead, edit inference.py to implement
your algorithm's infer_ct() and infer_mr() functions.

It is meant to run within a container.
main.py serves as the entrypoint for the docker container.

To run the container locally, you can call the following bash script:

  bash ./test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  bash ./save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/
"""

import json
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK
from inference import do_you_use_pytorch_cuda, infer_ct, infer_mr

print(f"user algo use torch cuda = {do_you_use_pytorch_cuda}")

# check if main.py is run in a docker env
cgroup = Path("/proc/self/cgroup")
exec_in_docker = (
    Path("/.dockerenv").is_file() or cgroup.is_file() and "docker" in cgroup.read_text()
)
print(f"exec_in_docker? {exec_in_docker}")
if exec_in_docker:
    # Setting correct paths for input, output and resources
    # depending on whether the algorithm is run in a docker container or locally
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
else:
    # for local non-docker env debugging
    INPUT_PATH = Path("./test/input/interf0")  # or interf1
    OUTPUT_PATH = Path("./test/output_local")


def run():
    # Check whether torch CUDA is available

    # NOTE: This relies on torch being installed in the environment
    if do_you_use_pytorch_cuda:
        from torch_utilities import _show_torch_cuda_info

        _show_torch_cuda_info()

    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("head-ct-angiography",): interf_ct_handler,
        ("head-mr-angiography",): interf_mr_handler,
    }[interface_key]

    # Call the handler
    # Run your prediction algorithm
    print("Running prediction algorithm...")
    return handler()


def interf_ct_handler():
    # Read the input

    input_head_ct_angiography = load_image_file(
        location=INPUT_PATH / "images/head-ct-angio",
    )

    output_seg_sitkimg = infer_ct(input_head_ct_angiography)

    check_image_mask_geometry(input_head_ct_angiography, output_seg_sitkimg)

    # Save your output

    write_image_file(
        location=OUTPUT_PATH / "images/extended-head-angio-segmentation",
        image=output_seg_sitkimg,
    )

    return 0


def interf_mr_handler():
    # Read the input

    input_head_mr_angiography = load_image_file(
        location=INPUT_PATH / "images/head-mr-angio",
    )

    output_seg_sitkimg = infer_mr(input_head_mr_angiography)

    check_image_mask_geometry(input_head_mr_angiography, output_seg_sitkimg)

    # Save your output

    write_image_file(
        location=OUTPUT_PATH / "images/extended-head-angio-segmentation",
        image=output_seg_sitkimg,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_file = (
        glob(str(location / "*.mha"))
        + glob(str(location / "*.nii.gz"))
        + glob(str(location / "*.nii"))
    )[0]  # There is just one file in the input folder!
    print(f"input_file = {input_file}")
    return SimpleITK.ReadImage(input_file)


def write_image_file(*, location, image):
    # Save your output
    print("Saving output...")

    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )
    print("Done!")


def check_image_mask_geometry(image, mask, atol=1e-5):
    checks = {
        "size": image.GetSize() == mask.GetSize(),
        "spacing": np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=atol),
        "direction": np.allclose(image.GetDirection(), mask.GetDirection(), atol=atol),
        # "origin": np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=atol),
    }

    for name, result in checks.items():
        print(f"{name:10}: {'OK' if result else 'MISMATCH'}")

    print("\nImage:")
    print("  size     :", image.GetSize())
    print("  spacing  :", image.GetSpacing())
    print("  origin   :", image.GetOrigin())
    print("  direction:", image.GetDirection())

    print("\nMask:")
    print("  size     :", mask.GetSize())
    print("  spacing  :", mask.GetSpacing())
    print("  origin   :", mask.GetOrigin())
    print("  direction:", mask.GetDirection())

    return all(checks.values())


if __name__ == "__main__":
    raise SystemExit(run())
