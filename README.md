# FeelScope 🎭 | Emotion Classification using Deep Learning

**FeelScope** is a Deep Learning project I built to classify emotions expressed in text. I experimented with multiple recurrent neural network architectures, compared their performance, and finally selected **Bidirectional GRU (BiGRU)** as my final model.

I built an interactive web application using Streamlit that allows users to enter text, predict the underlying emotion, and explore the model's confidence and probability distribution across different emotion categories.

The project combines text preprocessing, word embeddings, recurrent neural networks and model deployment into an end-to-end Deep Learning application.

## ✨ Features

- **Emotion Prediction:** Predicts the emotion expressed in user-provided text.
- **Expressive Results:** Displays an emotion-specific emoji alongside the predicted emotion.
- **Confidence Score:** Shows the model's confidence in its prediction.
- **Probability Distribution:** Visualises the prediction probabilities across all six emotions.
- **Batch Analysis:** Allows multiple sentences to be analysed together.
- **Prediction History:** Keeps track of recent predictions during a session.
- **CSV Export:** Allows batch analysis results to be downloaded.
- **Interactive Interface:** A custom-designed Streamlit interface for exploring the model.

## 🧠 About the Model

I used the [Emotion Dataset by dair-ai](https://huggingface.co/datasets/dair-ai/emotion), available on Hugging Face.

The task involves classifying text into six emotion categories:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

Instead of directly choosing an architecture, I experimented with different recurrent neural networks to understand their performance on this classification task.

My experiments included:

- SimpleRNN
- LSTM
- Bidirectional LSTM (BiLSTM)
- GRU
- Bidirectional GRU (BiGRU)

After comparing the results, I selected BiGRU as my final architecture.

### Final BiGRU Architecture

My final model consists of the following layers:

1. **Embedding Layer:** Converts integer-encoded words into dense vector representations.
2. **Bidirectional GRU (128 units):** Processes the input sequence in both forward and backward directions.
3. **Dropout (0.5):** Helps reduce overfitting.
4. **Bidirectional GRU (64 units):** Extracts further sequential representations.
5. **Dropout (0.5):** Adds another regularization layer.
6. **Dense Output Layer:** Uses Softmax activation to predict probabilities for the six emotion categories.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | Bidirectional GRU |
| Vocabulary Size | 10,000 |
| Embedding Dimension | 300 |
| Maximum Sequence Length | 50 |
| Padding | Post |
| Truncation | Post |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Output Activation | Softmax |
| Number of Classes | 6 |

## 📊 Model Experimentation & Comparison

One of the important parts of this project was experimenting with different architectures and embedding dimensions rather than directly settling on a single model.

I compared LSTM and BiLSTM configurations and subsequently evaluated BiGRU as my final architecture.

### Experimental Results

| Model | Embedding Dimension | Accuracy | Loss |
|---|---:|---:|---:|
| LSTM | 128 | 91.25% | 0.29856 |
| BiLSTM | 300 | 88.20% | 0.41711 |
| BiLSTM | 128 | 90.80% | 0.24165 |
| **BiGRU** | **300** | **91.70%** | **0.22475** |

### What I Observed

**1. LSTM**

My LSTM model achieved 91.25% accuracy with an embedding dimension of 128.

**2. Bidirectional LSTM (300)**

The initial BiLSTM configuration achieved 88.20% accuracy, which was lower than my unidirectional LSTM.

**3. Bidirectional LSTM (128)**

I experimented by reducing the embedding dimension from 300 to 128. This resulted in a considerable improvement:

- Accuracy increased from 88.20% to 90.80%.
- Loss decreased from 0.41711 to 0.24165.

This experiment showed me that increasing the embedding dimension does not necessarily guarantee better model performance.

**4. Bidirectional GRU (300)**

My final BiGRU model achieved the best recorded performance among the configurations I experimented with:

- **Accuracy: 91.70%**
- **Loss: 0.22475**

### Final Model Selection

After these experiments, I selected **BiGRU with an embedding dimension of 300** as my final architecture.

It achieved the highest recorded accuracy and the lowest loss among the compared configurations.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| TensorFlow / Keras | Deep Learning model development |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Scikit-learn | Model evaluation and preprocessing |
| Hugging Face Datasets | Dataset loading |
| Hugging Face Hub | Model hosting |
| Streamlit | Web application development |

## 📁 Project Structure

```text
FeelScope/
│
├── app.py
├── requirements.txt
├── README.md
├── NOTEBOOK_EXPORT.md
├── MODEL_CARD.md
│
├── models/
│   ├── bigru_model.keras
│   ├── tokenizer.json
│   ├── labels.json
│   └── config.json
│
├── notebooks/
│   └── emotion_predictor.ipynb
│
├── scripts/
│   ├── upload_model.py
│   └── download_model.py
│
├── src/
│   ├── model_manager.py
│   └── prediction.py
│
└── .streamlit/
    └── config.toml
```

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd FeelScope
```

### 2. Create a virtual environment

For Windows:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

For Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the trained model

I host the trained model artifacts on Hugging Face to keep the GitHub repository lightweight.

The model repository contains the following files:

- `bigru_model.keras`
- `tokenizer.json`
- `labels.json`
- `config.json`

Configure the Hugging Face repository in `.streamlit/secrets.toml`:

```toml
HF_MODEL_REPO = "YOUR_USERNAME/feelscope-bigru"
```

For a private Hugging Face repository, also configure a read-access token:

```toml
HF_TOKEN = "hf_your_read_token"
```

The application downloads and loads the required artifacts automatically.

Alternatively, the four files can be placed directly inside the local `models/` directory.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deployment

I use **Streamlit Community Cloud** to deploy FeelScope.

My deployment workflow:

1. Push the project to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select `app.py` as the main application file.
4. Configure the Hugging Face model repository in Streamlit Secrets.
5. Deploy the application.

The trained model is downloaded automatically when the application starts. Streamlit caching prevents unnecessary repeated model loading during the session.

## 📦 Model Artifacts

The application requires four important artifacts:

| File | Purpose |
|---|---|
| `bigru_model.keras` | Contains my trained BiGRU model and weights |
| `tokenizer.json` | Preserves the fitted tokenizer and vocabulary |
| `labels.json` | Stores the emotion labels in their correct output order |
| `config.json` | Stores the text preprocessing configuration |

All four artifacts must correspond to the same trained model.

For instructions on exporting the trained model from my notebook and uploading it to Hugging Face, refer to `NOTEBOOK_EXPORT.md`.

## 💡 What I Learned

Working on FeelScope helped me explore several important Deep Learning concepts through practical experimentation.

Some of the major concepts I worked with include:

- Text preprocessing and tokenization.
- Word embeddings and their dimensionality.
- Recurrent Neural Networks.
- LSTM and GRU architectures.
- Bidirectional sequence processing.
- Model architecture experimentation and comparison.
- Class imbalance and class weighting.
- Overfitting and early stopping.
- Model evaluation and performance analysis.
- Model serialization and deployment.
- Integrating a trained Deep Learning model into a web application.

Most importantly, this project helped me understand how architectural choices and hyperparameter experimentation can influence the performance of a Deep Learning model.

---

**Built with curiosity, experimentation and a lot of learning.** 🚀