import numpy as np
import fast

def load_patch(x_start_val_lvl3, y_start_val_lvl3, filename, level, patch_size):
    if not isinstance(filename, str):
        filename = filename.numpy().decode("utf-8")
    # convert coordinates to numpy (necessary due to slicing in tf that is unstable)
    x_start_val_lvl3 = np.asarray(x_start_val_lvl3)
    y_start_val_lvl3 = np.asarray(y_start_val_lvl3)  
    importer = fast.WholeSlideImageImporter.create(filename)
    wsi = importer.runAndGetOutputData()
    patch_access = wsi.getAccess(fast.ACCESS_READ)
    patch_small = patch_access.getPatchAsImage(level, int(x_start_val_lvl3[0]), int(y_start_val_lvl3[0]), patch_size, patch_size, False)
    patch_small = np.asarray(patch_small)

    return np.asarray(patch_small)
