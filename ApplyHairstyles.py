# Tim Horrell
# helper functions to apply various hairstyles
# consider resize, transparency, color correction

import cv2
import matplotlib.pyplot as plt
import numpy as np
from anchorPoints import *

def apply_bald_hairstyle(path, color_correction = 0):
    '''
    applies the bald hairstyle to the image
    '''
    anchor_points = findingAnchorPoints(path)[1:6]      #left forehead, right forehead, left ear, right ear
    hair_anchor = [(38, 105), (145, 105), (20, 161), (167, 161)]
    apply_hair_color(path, 'Hairstyles/bald.png', anchor_points, hair_anchor)

def apply_curly_hairstyle(path, color_correction = 0):
    '''
    applies the curly hairstyle to the image
    '''
    pass
    # TODO

def apply_doo_hairstyle(path, color_correction = 0):
    '''
    applies the doo-rag hairstyle to the image
    '''
    pass
    # TODO

def apply_mullet_hairstyle(path, color_correction = 0):
    '''
    applies the mullet hairstyle to the image
    '''
    anchor_points = findingAnchorPoints(path)[2:6]      #left forehead, right forehead, left ear, right ear
    hair_anchor = [(70, 175), (281, 175), (55, 243), (313, 243)]
    apply_hair(path, 'Hairstyles/mullet.png', anchor_points, hair_anchor)

def apply_waves_hairstyle(path, color_correction = 0):
    '''
    applies the waves hairstyle to the image
    '''
    anchor_points = findingAnchorPoints(path)[2:6]      #left forehead, right forehead, left ear, right ear
    hair_anchor = [(50, 140), (185, 140), (30, 195), (205, 195)]
    apply_hair(path, 'Hairstyles/waves.png', anchor_points, hair_anchor)


def apply_hair(background_path, overlay_path, face_anchor, hair_anchor):
    '''
    Applies the cut hair with a rescale factor to the background_path image,
    with softened edges for a more natural look.
    '''

    # Load the background image
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    bg_height, bg_width = background.shape[:2]

    # Load the overlay image with an alpha channel (transparency)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # Warp perspective of the hairstyle
    perspective_matrix = cv2.getPerspectiveTransform(np.float32(hair_anchor), np.float32(face_anchor))
    overlay = cv2.warpPerspective(overlay, perspective_matrix, (bg_width, bg_height))

    # Check if the overlay has an alpha channel
    if overlay.shape[2] != 4:
        raise ValueError("Overlay image must have an alpha channel (4 channels).")

    # Extract the BGR channels and the alpha channel (mask)
    overlay_bgr = overlay[:, :, :3]
    mask = overlay[:, :, 3]

    # Erode the alpha mask to remove any white border
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Blur the alpha mask to soften the edges
    blurred_mask = cv2.GaussianBlur(eroded_mask, (9, 9), 0)

    # Normalize the mask to keep it in the 0-255 range
    blurred_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)

    # Create a binary mask for blending
    alpha = blurred_mask.astype(float) / 255.0

    # Expand dimensions of the alpha mask to match the BGR channels (H, W, 3)
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    # Blend the overlay with the background using the alpha mask
    background = background * (1 - alpha_3d) + overlay_bgr * alpha_3d

    # Convert the result back to an 8-bit unsigned integer type
    background = background.astype(np.uint8)

    # Save the final image
    cv2.imwrite(background_path, background)


def apply_hair_color(background_path, overlay_path, face_anchor, hair_anchor):
    '''
    Applies the cut hair with a rescale factor to the background_path image and recolors the overlay 
    to match a target color while preserving shadows and details.
    '''

    # Load the background image
    background = cv2.imread(background_path, cv2.IMREAD_COLOR)
    bg_height, bg_width = background.shape[:2]

    # Load the overlay image with an alpha channel (transparency)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # Warp perspective of the hairstyle
    perspective_matrix = cv2.getPerspectiveTransform(np.float32(hair_anchor), np.float32(face_anchor[1:]))
    overlay = cv2.warpPerspective(overlay, perspective_matrix, (bg_width, bg_height))

    # Check if the overlay has an alpha channel
    if overlay.shape[2] != 4:
        raise ValueError("Overlay image must have an alpha channel (4 channels).")

    # Extract the BGR channels and the alpha channel (mask)
    overlay_bgr = overlay[:, :, :3]
    mask = overlay[:, :, 3]

    # Convert the overlay to the Lab color space
    overlay_lab = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2Lab)

    # set the target recolor to the first forehead coordinate
    target_color_coords = face_anchor[0]

    # Sample the target color from the background at the specified coordinates
    target_color_bgr = background[target_color_coords[1], target_color_coords[0]]
    target_color_lab = cv2.cvtColor(np.uint8([[target_color_bgr]]), cv2.COLOR_BGR2Lab)[0][0]

    # Split the Lab channels of the overlay
    l_channel, a_channel, b_channel = cv2.split(overlay_lab)

    # Replace the 'a' and 'b' channels of the overlay with the target color's 'a' and 'b' channels
    a_channel[:] = target_color_lab[1]
    b_channel[:] = target_color_lab[2]

    # Merge the channels back to form the recolored overlay
    recolored_overlay_lab = cv2.merge([l_channel, a_channel, b_channel])

    # Convert the recolored overlay back to the BGR color space
    recolored_overlay_bgr = cv2.cvtColor(recolored_overlay_lab, cv2.COLOR_Lab2BGR)

    # Blur the alpha mask to soften the edges
    blurred_mask = cv2.GaussianBlur(mask, (11, 11), 0)

    # Normalize the mask to keep it in the 0-255 range
    blurred_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)

    # Create a binary mask for blending
    alpha = blurred_mask.astype(float) / 255.0

    # Expand dimensions of the alpha mask to match the BGR channels (H, W, 3)
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    # Blend the recolored overlay with the background using the alpha mask
    background = background * (1 - alpha_3d) + recolored_overlay_bgr * alpha_3d

    # Convert the result back to an 8-bit unsigned integer type
    background = background.astype(np.uint8)

    # Save the final image
    cv2.imwrite(background_path, background)