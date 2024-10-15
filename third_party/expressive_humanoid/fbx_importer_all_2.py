"""This script converts CMU Motion Capture data from FBX format to NPY format. """
import sys
import os
from tqdm import tqdm
from poselib.skeleton.skeleton3d import SkeletonMotion
import multiprocessing

sys.path.append("/home/user/fbx_setup/fbx_python_bindings/build/Distrib/site-packages/fbx/")

def process_file(i, fbx_file, all_fbx_path):
    if fbx_file.endswith(".fbx"):
        print(i, fbx_file)
        motion = SkeletonMotion.from_fbx(
            fbx_file_path=os.path.join(all_fbx_path, fbx_file),
            root_joint="Hips",
            fps=60
        )
        motion.to_file(f"data/npy/{fbx_file[:-4]}.npy")

def main():
    all_fbx_path = "data/cmu_fbx_all/"
    all_fbx_files = sorted(os.listdir(all_fbx_path))

    all_fbx_filtered = []
    for fbx in all_fbx_files:
        npy = fbx.split(".")[0] + ".npy"
        target_motion_file = os.path.join(all_fbx_path, "../npy/", npy)
        if os.path.exists(target_motion_file):
            print("Already exists, skip: ", fbx)
            continue
        all_fbx_filtered.append(fbx)

    print(len(all_fbx_filtered))

    for fbx_file in tqdm(all_fbx_filtered):
        process_file(None, fbx_file, all_fbx_path)

    n_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm(pool.starmap(process_file, [(i, fbx_file, all_fbx_path) for i, fbx_file in enumerate(all_fbx_filtered)]), total=len(all_fbx_filtered)))

if __name__ == "__main__":
    main()