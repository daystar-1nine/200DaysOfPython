# 🚀 DAY 98 / 200 — IMAGE SEGMENTATION

## 🧠 Masterclass: Decoding the Pixel Matrix

### The Concept:
Object Detection gives us `[x1, y1, x2, y2]`. It draws a rigid box. 
Segmentation forces the neural network to output an entire matrix matching the exact dimensions of the image. 
If the image is `(640x640)`, the output is `(640x640)`.
For every single pixel `(i, j)`, the network outputs a probability `(0.0 - 1.0)` that the pixel belongs to the foreground object. 
We then evaluate this using **IoU** (Intersection of true pixel matrices vs predicted pixel matrices) and the **Dice Coefficient**. 
Because producing a `640x640` matrix requires preserving spatial resolution, we use the **U-Net** architecture. The Encoder crushes the image into deep features, and the Decoder up-samples those features back into a `640x640` grid, using *Skip Connections* to inject original boundary data that was lost during down-sampling.
