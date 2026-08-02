# 🖼️ Deep Learning Image Captioning System using PyTorch

A complete **Deep Learning Image Captioning Pipeline** built with **PyTorch**, featuring custom dataset engineering, GPU-accelerated data loading, CNN-based visual feature extraction, and sequence generation using RNN/Transformer architectures.

---

# 📌 Project Overview

Image Captioning is a **Multimodal Deep Learning** task that combines **Computer Vision** and **Natural Language Processing (NLP)** to automatically generate descriptive captions for images.

This project builds a complete end-to-end pipeline starting from raw image ingestion to caption generation. It provides a scalable, GPU-ready framework for training image captioning models using the **Flickr8k dataset**.

The system includes:

- 🖼️ Image preprocessing and augmentation
- 📂 Custom PyTorch Dataset & DataLoader
- 🧠 CNN / Vision Transformer Encoder
- 📝 RNN / LSTM / Transformer Decoder
- 🎯 Attention Mechanism
- 🚀 Efficient GPU training
- 📊 Caption generation and evaluation

---

# ✨ Key Features

- 📂 **Custom PyTorch Dataset**
  - Efficient on-demand image loading
  - Memory-safe implementation
  - RGB image conversion

- ⚡ **GPU-Accelerated Data Pipeline**
  - CUDA-compatible tensors
  - Multi-worker DataLoader
  - Pinned memory support
  - Batch shuffling

- 🖼️ **Image Preprocessing**
  - Resize and center crop
  - ImageNet normalization
  - Data augmentation
  - Tensor conversion

- 🧠 **Feature Extraction**
  - ResNet-50
  - EfficientNet
  - Vision Transformer (ViT)
  - Transfer Learning

- 📝 **Caption Generation**
  - Vocabulary construction
  - Tokenization
  - Word embeddings
  - Attention-based Decoder

- 📈 **Evaluation Metrics**
  - BLEU
  - METEOR
  - ROUGE-L
  - CIDEr

---

# 🏗️ End-to-End System Architecture

```text
                     Flickr8k Dataset
                   (8,091 JPG Images)
                           │
                           ▼
              Image Preprocessing Pipeline
                           │
                           ▼
                 Custom PyTorch Dataset
                           │
                           ▼
                PyTorch DataLoader
                           │
                           ▼
              Tensor Batch [32,3,224,224]
                           │
                           ▼
                  CNN / ViT Encoder
                           │
                           ▼
                  Visual Feature Vector
                           │
                           ▼
              Attention-based Decoder
                   (LSTM/Transformer)
                           │
                           ▼
                  Generated Caption
```

---

# 🧮 Methodology

## 1. Environment Setup

The project begins by initializing the deep learning environment using Python scientific libraries.

### Libraries Used

- PyTorch
- Torchvision
- Pillow (PIL)
- NumPy
- Matplotlib
- Seaborn
- Glob
- OS

GPU availability is automatically detected using CUDA.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 2. Dataset Discovery

The Flickr8k dataset contains **8,091 real-world images**.

Python's `glob` module scans the image directory and indexes every `.jpg` file.

```text
Flickr8k/
│
├── Images/
│     ├── image1.jpg
│     ├── image2.jpg
│     ├── ...
│
└── Captions.txt
```

---

## 3. Custom PyTorch Dataset

A custom `torch.utils.data.Dataset` is implemented for memory-efficient image loading.

### Responsibilities

- Collect image paths
- Load images on demand
- Convert images to RGB
- Apply transforms
- Return image-caption pairs

This avoids loading the complete dataset into RAM.

---

## 4. DataLoader Pipeline

The custom dataset is wrapped inside a PyTorch `DataLoader`.

### Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Shuffle | True |
| Pin Memory | True |
| Num Workers | Multiple CPU Workers |

Benefits include:

- Parallel image loading
- Faster GPU transfer
- Randomized batches
- Better training efficiency

---

## 5. Image Preprocessing

Neural networks require fixed-size inputs.

Each image undergoes the following transformations:

```text
Original Image
      │
      ▼
Resize (256 × 256)
      │
      ▼
Center Crop (224 × 224)
      │
      ▼
Tensor Conversion
      │
      ▼
Normalization
```

Normalization uses ImageNet statistics:

```text
Mean:
[0.485, 0.456, 0.406]

Std:
[0.229, 0.224, 0.225]
```

Final tensor shape:

```text
[3,224,224]
```

---

# 📝 Text Processing Pipeline

Natural language captions are converted into numerical representations.

```text
"A brown dog runs across the grass"

        │

        ▼

Tokenization

        │

        ▼

['<START>', 'a', 'brown', 'dog', 'runs', 'across', 'the', 'grass', '<END>']

        │

        ▼

Vocabulary Lookup

        │

        ▼

[1, 14, 203, 89, 412, 55, 12, 607, 2]
```

---

## Vocabulary Construction

Special tokens include:

| Token | Purpose |
|--------|----------|
| `<PAD>` | Sequence Padding |
| `<START>` | Start of Caption |
| `<END>` | End of Caption |
| `<UNK>` | Unknown Word |

Rare words are replaced by `<UNK>` using a frequency threshold.

---

# 🧠 Encoder–Decoder Architecture

The caption generation model follows a **Sequence-to-Sequence (Seq2Seq)** architecture.

```text
Image
   │
   ▼
CNN / Vision Transformer
   │
   ▼
Image Features
   │
   ▼
Attention Layer
   │
   ▼
LSTM / Transformer Decoder
   │
   ▼
Generated Caption
```

---

## CNN Encoder

The encoder extracts high-level semantic features.

Supported backbones:

- ResNet-50
- EfficientNet
- Vision Transformer (ViT)

Transfer learning is used by removing the final classification layer.

Example feature vector:

```text
2048-Dimensional Feature Vector
```

---

## Attention-Based Decoder

The decoder generates captions word-by-word.

Supported models:

- LSTM
- GRU
- Transformer Decoder

Attention enables the decoder to focus on important image regions while generating each word.

---

# 🎯 Training Strategy

## Teacher Forcing

During training, the decoder receives the **ground truth previous word** instead of its own prediction.

```text
Ground Truth

<START>

↓

A

↓

Dog

↓

Running

↓

<END>
```

This significantly improves convergence and learning stability.

---

## Loss Function

Cross-Entropy Loss is used while ignoring padded tokens.

Training objective:

- Minimize caption prediction error
- Learn image-text relationships
- Improve sequence generation

---

# 🚀 Inference

Two decoding strategies are supported.

## Greedy Search

```text
Highest Probability

↓

Next Word

↓

Repeat
```

Advantages:

- Very Fast
- Low Memory

Disadvantages:

- May generate repetitive captions

---

## Beam Search

Instead of keeping only one prediction, Beam Search maintains the **Top-K candidate captions**.

Advantages:

- Better caption quality
- More natural sentences

Disadvantages:

- Slightly slower

---

# 📊 Evaluation Metrics

Generated captions are compared with ground-truth captions using standard NLP metrics.

| Metric | Description |
|---------|-------------|
| BLEU-1 | Unigram Precision |
| BLEU-2 | Bigram Precision |
| BLEU-3 | Trigram Precision |
| BLEU-4 | 4-gram Precision |
| METEOR | Semantic Matching |
| ROUGE-L | Longest Common Subsequence |
| CIDEr | Consensus-based Caption Evaluation |

---

# 📈 Project Highlights

- ✅ End-to-End Image Captioning Pipeline
- ✅ Custom PyTorch Dataset Engineering
- ✅ Efficient DataLoader Implementation
- ✅ GPU-Ready Training Pipeline
- ✅ CNN Encoder + Attention Decoder
- ✅ Sequence-to-Sequence Learning
- ✅ Transfer Learning Support
- ✅ Scalable to Flickr30k & MS-COCO
- ✅ Modular and Reusable Codebase

---

# 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3 |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision |
| Image Processing | Pillow (PIL) |
| Numerical Computing | NumPy |
| Visualization | Matplotlib, Seaborn |
| Dataset | Flickr8k |
| Hardware | CUDA GPU |

---

# 📂 Repository Structure

```text
.
├── dataset.py                 # Custom Flickr8k Dataset
├── dataloader.py              # DataLoader configuration
├── preprocess.py              # Image preprocessing pipeline
├── vocabulary.py              # Vocabulary creation & tokenization
├── encoder.py                 # CNN / ViT Encoder
├── decoder.py                 # LSTM / Transformer Decoder
├── attention.py               # Attention Mechanism
├── train.py                   # Model training
├── inference.py               # Caption generation
├── evaluate.py                # BLEU, METEOR, ROUGE, CIDEr evaluation
├── utils.py                   # Helper functions
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Getting Started

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/image-captioning-pytorch.git

cd image-captioning-pytorch
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Prepare Dataset

Download the **Flickr8k dataset** and organize it as:

```text
dataset/
│
├── Images/
├── captions.txt
```

---

## 4. Train the Model

```bash
python train.py
```

---

## 5. Generate Captions

```bash
python inference.py
```

---

# 🌍 Applications

- 🖼️ Automatic Image Captioning
- ♿ Assistive Technology for Visually Impaired
- 🔍 Image Search & Retrieval
- 🤖 Vision-Language AI Systems
- 📱 Smart Photo Organization
- 🛒 E-commerce Image Understanding
- 🚗 Autonomous Driving Scene Description
- 🛰️ Remote Sensing & Satellite Image Analysis

---

# 🔮 Future Improvements

- 🔥 Vision Transformer (ViT) Encoder
- 🤖 BLIP / BLIP-2 Integration
- 🧠 CLIP-based Image Embeddings
- ⚡ Mixed Precision (FP16) Training
- 🚀 Distributed Multi-GPU Training
- 📱 ONNX / TensorRT Deployment
- 🌐 Hugging Face Transformers Integration
- 🎯 Interactive Web App using Streamlit

---

# 📄 License

This project is licensed under the **MIT License**.

See the **LICENSE** file for more details.

---

# 👨‍💻 Author

**Arjun Singh Gangwar**

- 🎓 M.Tech in Data Science
- 💻 AI • Deep Learning • Computer Vision • Natural Language Processing
- 🚀 Research Interests: Vision-Language Models, Image Captioning, Explainable AI, Data Engineering

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Your support encourages continued development and open-source contributions.
