# 🖼️ Image Captioning using VGG16 + LSTM (Flickr8k Dataset)

This project demonstrates an end-to-end image captioning pipeline using the Flickr8k dataset. It leverages a pre-trained VGG16 CNN to extract image features and an LSTM-based decoder to generate natural language captions. The entire implementation is done in a Jupyter Notebook using TensorFlow/Keras.

---

## 📦 Dataset

This project uses the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which contains:

- 8,000 real-world images
- 5 human-annotated captions per image

### 📥 How to Use

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Create a folder structure like this:

image-caption/
├── Flickr8k/
│ ├── Images/
│ └── captions.txt


3. Run the notebook to:
   - Extract features using VGG16
   - Process captions
   - Train the model
   - Generate captions

---

## 📂 Project Files

| File                 | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `image-caption.ipynb`| Main Jupyter Notebook with all training and inference steps |
| `image_caption.h5`   | Trained LSTM model weights (Keras format)                   |
| `features.pkl`       | (Optional) Precomputed image features from VGG16            |
| `mapping.pkl`        | (Optional) Word-index and index-word dictionary             |
| `assets/sample.jpg`  | Example test image for demo                                 |

---

## 🧠 Model Architecture

- **Encoder**: VGG16 (pretrained, no top layer) for feature extraction
- **Decoder**: Embedding → LSTM → Dense
- Combines image context and sequence context to predict the next word
- Evaluation: BLEU score from `nltk.translate`

---

## 🔧 Key Functions

- `clean(text)`: Cleans and tokenizes caption strings
- `data_generator(...)`: Yields batches for training
- `predict_caption(...)`: Greedy decoding for caption generation
- `idx_to_word(...)`: Converts model predictions to text

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/shravandeshpandee/image-caption.git
   cd image-caption
