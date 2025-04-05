# MaskRNN_Surveillance
Below is a README for your GitHub project based on the Mask R-CNN surveillance system we adapted. It’s structured to be clear, concise, and professional, tailored to your assignment’s context while providing enough detail for others to understand, use, and contribute to the project. Feel free to tweak it based on your specific needs or repository details!

---

# MaskRCNN-Surveillance

**Automating Surveillance Image Analysis with Mask R-CNN**

This project leverages a pre-trained Mask R-CNN model to analyze still images from surveillance cameras. It detects and classifies people into categories like "thieves/unwanted," "workers/garbage men," or "friendly people," and distinguishes between "weapons" and "tools" based on detected objects. Designed as a prototype for a business workflow automation assignment, it aims to streamline manual security monitoring processes.

---

## Project Overview

### Use Case
Manual review of surveillance footage is time-consuming and prone to human error. This solution automates the process by:
- Identifying individuals and their roles based on contextual clues (e.g., objects they carry).
- Flagging potential threats (e.g., weapons) versus benign items (e.g., tools).
- Generating annotated images for quick security review.

### Features
- **Person Classification:** Categorizes people as "thief/unwanted," "worker/garbage," "friendly," or "unknown" based on nearby objects.
- **Object Detection:** Differentiates weapons (e.g., knife) from tools (e.g., bottle) using COCO labels.
- **Visualization:** Outputs images with colored masks, bounding boxes, and labels for easy interpretation.

---

## Requirements

- **Python 3.8+**
- **Libraries:**
  - `torch` (PyTorch)
  - `torchvision`
  - `Pillow` (PIL)
  - `opencv-python` (cv2)
  - `numpy`
- **Hardware:** GPU recommended (CUDA support), but CPU works too.

Install dependencies:
```bash
pip install torch torchvision pillow opencv-python numpy
```

---

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/[your-username]/MaskRCNN-Surveillance.git
   cd MaskRCNN-Surveillance
   ```

2. **Prepare Images:**
   - Place surveillance images (`.jpg`, `.jpeg`, `.png`) in a directory (e.g., `images/`).
   - Update the `image_dir` path in `main()` to point to your image folder:
     ```python
     image_dir = "path/to/your/images"
     ```

3. **Run the Script:**
   ```bash
   python surveillance.py
   ```

---

## Usage

- **Input:** Still images from surveillance cameras.
- **Output:** Annotated images saved in the same directory (e.g., `surveillance_image1.jpg`) with:
  - People highlighted in colors (red = thief, green = worker, blue = friendly, gray = unknown).
  - Objects labeled as "weapon" (magenta) or "tool" (yellow).
  - Console logs detailing detected people and objects.

- **Example:**
  - Input: Image of a person holding a knife.
  - Output: Red mask around person labeled "thief/unwanted," magenta box around "knife (weapon)."

---

## Code Structure

- **`surveillance.py`:**
  - `get_prediction()`: Runs Mask R-CNN inference on images.
  - `classify_person()`: Assigns person categories based on nearby objects.
  - `classify_object()`: Differentiates weapons from tools.
  - `analyze_surveillance_image()`: Processes and visualizes results.
  - `main()`: Loops through image directory and executes analysis.

---

## Limitations

- **COCO Constraints:** Relies on pre-trained COCO labels, missing some items (e.g., "gun," "broom"). Custom training needed for broader detection.
- **Contextual Rules:** Simple proximity-based classification may misinterpret unrelated objects.
- **Static Images Only:** Doesn’t support real-time video (yet!).

---

## Future Improvements

- **Custom Training:** Fine-tune Mask R-CNN on a surveillance dataset with labels like "gun," "uniform," or "mask."
- **Real-Time Processing:** Extend to video streams using OpenCV’s video capture.
- **Web Interface:** Integrate with a tool like CodePen or Flask for a user-friendly UI.
- **Accuracy Boost:** Use NLP or additional AI (e.g., GPT) to refine context analysis.

---

## Prototype Results

- **Test Case 1:** Person with a knife → Correctly flagged as "thief/unwanted" with "weapon."
- **Test Case 2:** Person with a bottle → Identified as "worker/garbage" with "tool."
- **Challenges:** Misclassified a person with a backpack as "thief" when it was a worker—needs better rules.

*See `images/output/` for sample annotated outputs (add your own after running).*

---

## Contributing

1. Fork the repo.
2. Create a branch (`git checkout -b feature/your-idea`).
3. Commit changes (`git commit -m "Add feature X"`).
4. Push (`git push origin feature/your-idea`).
5. Open a pull request.

---

## Credits

- Built with [PyTorch](https://pytorch.org/) and [TorchVision](https://pytorch.org/vision/stable/index.html).
- Developed as part of an xAI-assisted assignment by [your name] and Grok (xAI).
- Inspired by Mask R-CNN for instance segmentation.

---

## License

MIT License - feel free to use, modify, and distribute!

---

Let me know if you’d like to adjust anything—add your name, tweak the tone, or include specific results from your tests! Ready to push this to GitHub?
