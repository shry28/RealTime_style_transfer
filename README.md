# 🎨 Real-Time Person Segmentation & Style Transfer

## 👤 Author:
Shaurya Choudhary 

---

## 📝 Project Objective
Build a real-time vision system that:
- Segments **only the person** in a webcam feed.
- Applies **various visual styles** like cartoon, pencil sketch, etc.
- Allows **keyboard control** to switch styles and background modes.

---

## 🧠 Tech Stack
- Python 3.10  
- OpenCV  
- PixelLib (with TensorFlow backend)  
- NumPy  
- Pre-trained Mask R-CNN (`mask_rcnn_coco.h5`)
   ### 🔗 Download Pre-trained Model

Download the `mask_rcnn_coco.h5` file (245 MB) from the official release:

👉 [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_coco.h5)

Place it in the same directory as `segmentation.py` before running the code.

---

## ⚙️ How It Works
1. Access webcam using OpenCV.
2. Use PixelLib to segment the "person" class.
3. Apply styles like cartoon, grayscale, pencil sketch **only on the person**.
4. Combine styled person with:
   - Original background  
   - Blacked-out background  
   - Blurred background  
5. Interactive keyboard controls to switch **styles** and **backgrounds**.

---

## 🎨 Supported Styles
| Style # | Style Description |
|--------|--------------------|
| 1 | `styled_person_JET` - Jet colormap |
| 2 | `styled_person_cartoon` - Cartoon |
| 3 | `styled_person_pencil` - Pencil sketch |
| 4 | `styled_person_oil` - Oil painting |
| 5 | `styled_person_gray` - Grayscale |
| 6 | `styled_person_cartoon2` - Smooth Cartoon |
| 7 | `styled_person_black` - Full black silhouette |

---

## 🌌 Background Modes
Toggle between:
- `0`: Original background  
- `1`: Blacked out  
- `2`: Blurred background  

## 🎮 Keyboard Controls
Key Action:
  - n	 Switch to next style (cycles through 7 styles)  
  - b	Switch to next background mode (original → black → blurred → original...)
  - q	Quit the program

```python
if background_mode == 0:
    background = cv2.bitwise_and(frame, frame, mask=background_mask)
elif background_mode == 1:
    black = np.zeros_like(frame)
    background = cv2.bitwise_and(black, black, mask=background_mask)
elif background_mode == 2:
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    background = cv2.bitwise_and(blurred, blurred, mask=background_mask)
