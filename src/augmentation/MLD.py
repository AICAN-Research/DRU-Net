# Multi-lens Distortion


def multi_lens_distortion(image, num_lenses, radius_range, strength_range):
    """
    Apply a smooth lens distortion effect with multiple lenses to an image.

    Parameters:
        image (np.array): Input image of shape (H, W, C).
        num_lenses (int): Number of lenses to apply.
        radius_range (tuple): A tuple of (min_radius, max_radius) for the lens effect.
        strength_range (tuple): A tuple of (min_strength, max_strength) for the lens effect.

    Returns:
        np.array: Image with multiple lens effects applied.
    """
    H, W, C = image.shape
    # Randomly generate lens centers within the image boundaries.
    cx = np.random.randint(0, W, size=num_lenses)
    cy = np.random.randint(0, H, size=num_lenses)

    # Initialize distorted_image to be the original image.
    # It will be updated as each lens is applied.
    distorted_image = np.copy(image)
    yidx, xidx = np.indices((H, W))

    # Apply each lens.
    for i in range(num_lenses):
        # Randomly select radius and strength for the current lens within the provided ranges.
        radius = np.random.randint(radius_range[0], radius_range[1])
        strength = np.random.uniform(strength_range[0], strength_range[1])

        # Calculate the Euclidean distance to the center of the lens for each point in the image.
        dx = xidx - cx[i]
        dy = yidx - cy[i]
        r = np.sqrt(dx**2 + dy**2)

        # Normalized distance within the lens.
        normalized_r = r / radius

        # Calculate a smooth scaling factor that goes from 1 at the lens center (r=0)
        # to 0 at the lens perimeter (r=radius).
        scaling_factor = np.maximum(1 - normalized_r, 0)

        # Compute the distortion for each point in the image, scaled by the
        # distance to the lens center.
        distorted_y = dy * (1 - strength * scaling_factor) + cy[i]
        distorted_x = dx * (1 - strength * scaling_factor) + cx[i]

        # Ensure the new indices are not out of bounds.
        distorted_y = np.clip(distorted_y, 0, H - 1).astype(int)
        distorted_x = np.clip(distorted_x, 0, W - 1).astype(int)

        # Create the distorted image by mixing original and distorted coordinates.
        distorted_image = distorted_image[distorted_y, distorted_x]

    return distorted_image
