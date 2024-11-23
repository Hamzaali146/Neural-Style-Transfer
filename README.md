# 🎨 Neural Style Transfer (NST) with TensorFlow

![NST Banner](https://via.placeholder.com/800x200?text=Neural+Style+Transfer+with+TensorFlow)  
*Create stunning artwork by blending content and style with the power of AI!*

---

## 📖 **Overview**

Neural Style Transfer (NST) is a deep learning technique that combines the **content** of one image with the **style** of another to generate artistic outputs. This project uses TensorFlow and the pre-trained **VGG19 model** to create visually stunning images, leveraging content and style loss optimization.

Inspired by Gatys et al.'s 2015 paper, this project allows you to experiment with blending your favorite content and style images to generate creative outputs.

---

## 🖼️ **Sample Outputs**

| **Content Image**              | **Style Image**              | **Generated Image**          |
|--------------------------------|------------------------------|------------------------------|
| ![Content Image](https://via.placeholder.com/300x300?text=Content+Image) | ![Style Image](https://via.placeholder.com/300x300?text=Style+Image) | ![Output Image](https://via.placeholder.com/300x300?text=Generated+Image) |

---

## 🚀 **Features**

- **Blend Any Images:** Merge a content image with any style image.
- **Adjustable Weights:** Customize content and style importance using weights.
- **State-of-the-Art Model:** Powered by the pre-trained VGG19 architecture.
- **Efficient Optimization:** Fast convergence with TensorFlow’s `GradientTape`.

---

## ⚙️ **How It Works**

1. Extract features from **content** and **style images** using VGG19.
2. Calculate:
   - **Content Loss**: Retain structure and details from the content image.
   - **Style Loss**: Replicate textures and patterns from the style image using Gram matrices.
3. Minimize the **total loss** to generate the stylized image through gradient descent.

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nst-tensorflow.git
   cd nst-tensorflow
