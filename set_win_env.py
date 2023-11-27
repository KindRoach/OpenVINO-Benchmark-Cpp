import shutil
import winreg
import os
from pathlib import Path


def add_directory_to_path(dir_to_add):
    # Path to str
    dir_to_add = dir_to_add.__str__()

    # Get current environment variable
    key_path = "Environment"
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE)

    try:
        current_paths = winreg.QueryValueEx(key, "Path")[0]
        current_paths_set = set(current_paths.split(os.pathsep))

        # check dir exist
        if dir_to_add not in current_paths_set:
            # add if not exist
            new_path = f"{current_paths}{os.pathsep}{dir_to_add}"
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            print(f"\"{dir_to_add}\" added to Path.")
        else:
            print(f"\"{dir_to_add}\" already exist in Path")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        winreg.CloseKey(key)


target_dir = Path.home().joinpath("dev")
target_dir.mkdir(exist_ok=True)
for source_item in [
    Path("lib/opencv"),
    Path("lib/openvino"),
]:
    target_item = target_dir.joinpath(Path(*source_item.parts[1:]))
    if not target_item.exists():
        shutil.copytree(source_item, target_item)
        print(f"\"{source_item}\" copied to \"{target_item}\".")
    else:
        print(f"\"{target_item}\" already exist.")

for bin_dir in [
    "opencv/build/x64/vc16/bin",
    "openvino/runtime/bin/intel64/Debug",
    "openvino/runtime/3rdparty/tbb/bin",
]:
    add_directory_to_path(target_dir.joinpath(bin_dir))
