import pixellib
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
import time

# Load model
segmentation_model = instance_segmentation()
segmentation_model.load_model("mask_rcnn_coco.h5")

cap = cv2.VideoCapture(0)

background_mode = 0
style_index = 0
style_names = [
    "JET",
    "cartoon",
    "pencil",
    "oil",
    "gray",
    "cartoon2",
    "black"
]

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    result = segmentation_model.segmentFrame(frame, show_bboxes=False)[0]

    masks = result['masks']
    class_ids = result['class_ids']
                                                                    # Create blank mask for person(s)
    person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    '''
    max_area = 0
    largest_person_mask = None
   
    for i in range(masks.shape[-1]):
        if class_ids[i] == 1:                                        # COCO class ID for "person"
            current_mask = masks[:, :, i].astype("uint8") * 255
            area = cv2.countNonZero(current_mask)
            if area > max_area:
                max_area = area
                largest_person_mask = current_mask

    if largest_person_mask is not None:
        person_mask = largest_person_mask
    '''
    
    for i in range(masks.shape[-1]):
        if class_ids[i] == 1:
            current_mask = masks[:, :, i].astype("uint8") * 255
            person_mask = cv2.bitwise_or(person_mask, current_mask)

    if np.count_nonzero(person_mask) > 0:

                                                                        # Get only person part
        person_only = cv2.bitwise_and(frame, frame, mask=person_mask)

                                                                        # Style the person area
        gray = cv2.cvtColor(person_only, cv2.COLOR_BGR2GRAY)

                                                                        # Edges
        edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 7), 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        JET = cv2.applyColorMap(gray, cv2.COLORMAP_JET )
        styled_person_JET = cv2.bitwise_and(JET, edges)
                                                                        #cartoon person
        cartoon = cv2.stylization(person_only, sigma_s=100, sigma_r=0.07)
        styled_person_cartoon = cv2.bitwise_and(cartoon, cartoon, mask=person_mask)

                                                                        #pencil person
        gray2, sketch = cv2.pencilSketch(person_only, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        gray_bgr = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        styled_person_pencil = cv2.bitwise_and(gray_bgr, gray_bgr, mask=person_mask)

                                                                        #oil paint
        oil_paint = cv2.xphoto.oilPainting(person_only, 9, 3)
        styled_person_oil = cv2.bitwise_and(oil_paint, oil_paint, mask=person_mask)

                                                                        #black person
        black = np.zeros_like(frame)
        styled_person_black = cv2.bitwise_and(black, black, mask=person_mask)

                                                                        #gray person
        styled_person_gray =  cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                                                                        # cartoon 2
        smoothed = cv2.bilateralFilter(person_only, d=9, sigmaColor=250, sigmaSpace=250)
        styled_person_cartoon2 = cv2.bitwise_and(smoothed, edges)

                                                                         # Get the background
        background_mask = cv2.bitwise_not(person_mask)

        if background_mode == 0:
            # Normal background
            background = cv2.bitwise_and(frame, frame, mask=background_mask)
        elif background_mode == 1:
            # Black background
            black = np.zeros_like(frame)
            background = cv2.bitwise_and(black, black, mask=background_mask)
        elif background_mode == 2:
            # Blurred background
            blurred = cv2.GaussianBlur(frame, (35, 35), 0)
            background = cv2.bitwise_and(blurred, blurred, mask=background_mask)
                                                                    
        #final_frame = cv2.add(background, styled_person_JET)         # Combine styled person with normal background
        #final_frame = cv2.bitwise_and(styled_person_cartoon, styled_person_cartoon, mask=person_mask)
        if style_names[style_index] == "JET":
            styled = styled_person_JET
        elif style_names[style_index] == "cartoon":
            styled = styled_person_cartoon
        elif style_names[style_index] == "pencil":
            styled = styled_person_pencil
        elif style_names[style_index] == "oil":
            styled = styled_person_oil
        elif style_names[style_index] == "gray":
            styled = styled_person_gray
        elif style_names[style_index] == "cartoon2":
            styled = styled_person_cartoon2
        elif style_names[style_index] == "black":
            styled = styled_person_black
        else:
            styled = person_only  # fallback

        final_frame = cv2.add(background, styled)

    else:
        final_frame = frame.copy()

    escaped_time = time.time() - start_time
    fps = 1/escaped_time
    cv2.putText(final_frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(final_frame, f'Style: {style_names[style_index]}', (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    bg_mode_text = ["Normal BG", "Black BG", "Blurred BG"][background_mode]
    cv2.putText(final_frame, f'BG: {bg_mode_text}', (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Styled Person", final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        style_index = (style_index + 1) % len(style_names)
    elif key == ord('b'):
        background_mode = (background_mode + 1) % 3

cap.release()
cv2.destroyAllWindows()

#cv2.COLORMAP_HOT       # 🔥 Red/yellow hotmap
#cv2.COLORMAP_OCEAN     # 🌊 Blue-green ocean effect
#cv2.COLORMAP_PINK      # 🌸 Soft pinkish tone
#cv2.COLORMAP_TWILIGHT  # 🌆 Purples and blues