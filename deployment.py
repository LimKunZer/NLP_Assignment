import stopwordsiso as stopwords
import malaya
import re
import emoji
import contractions
import torch.nn as nn
import torch
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# define english and malay stopwords
stop_words = set(stopwords.stopwords(["en", "ms"]))

# load normalisation model
lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
corrector = malaya.spelling_correction.probability.load(language_model = lm)
stemmer = malaya.stem.huggingface()
normalizer = malaya.normalizer.rules.load(corrector, stemmer)

# method to remove non-alphanumeric characters, superscripts, punctuation with non-standard spacing, and newlines, as well as expand contractions
def remove_characters(text):
  return re.sub(r'[^\w\s]', '', re.sub('[²³¹⁰ⁱ⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ]', ' ', re.sub('[.,!?]', ' ', contractions.fix(emoji.demojize(text.lower()).replace('_', ' ').replace(':', ' ')))))

# method to add spacing between joined numberword and consecutive emojis
def add_spacing(text):
  i = 0
  while i < len(text) - 1:
    if emoji.is_emoji(text[i]) and emoji.is_emoji(text[i + 1]):
      text = text[:i + 1] + ' ' + text[i + 1:]
    i += 1
  i = 0
  while i < len(text) - 1:
    if text[i].isdigit() and text[i + 1].isalpha():
      text = text[:i + 1] + ' ' + text[i + 1:]
    i += 1
  return text

# normalise cleaned_review into new column fixed_review
def normalise_words(text):
  return normalizer.normalize(text)['normalize']

# method to remove stopwords
def remove_stopwords(text):
  return " ".join([word for word in text.split() if word not in stop_words])

# Create CNN class
class SentimentCNN(nn.Module):
  def __init__(self, vocabSize, embeddingDim, numFilters, filterSizes, outputDim):
    super(SentimentCNN, self).__init__()
    self.embedding = nn.Embedding(vocabSize, embeddingDim)
    self.convs = nn.ModuleList([
      nn.Conv2d(in_channels=1, out_channels=numFilters, kernel_size=(fs, embeddingDim))
      for fs in filterSizes
    ])
    self.fc = nn.Linear(len(filterSizes) * numFilters, outputDim)
    self.dropout = nn.Dropout(0.5)
    
  def forward(self, x):
    x = self.embedding(x)  # x: [batch_size, seq_len, embedding_dim]
    x = x.unsqueeze(1)     # x: [batch_size, 1, seq_len, embedding_dim]
    conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # List of [batch_size, num_filters, seq_len - filter_size[n] + 1]
    pooled = [torch.max(c, dim=2)[0] for c in conved]  # List of [batch_size, num_filters]
    cat = torch.cat(pooled, dim=1)  # cat: [batch_size, len(filter_sizes) * num_filters]
    cat = self.dropout(cat)
    return self.fc(cat)
    
  def predict(self, text):
    self.eval()
    input_tensor = preprocess_text(text).squeeze(2).unsqueeze(1)
    with torch.no_grad():
      predictions = self(input_tensor)
    return predictions.argmax(dim=1).item()

with open('cnn_tokenizer.pkl', 'rb') as handle:
  cnn_tokenizer = pickle.load(handle)

# Preprocess text
def preprocess_text(text):
    cleaned_text = remove_characters(add_spacing(text))
    normalised_text = normalise_words(cleaned_text)
    preprocessed_text = remove_stopwords(normalised_text)
    tokenised_text = cnn_tokenizer.texts_to_sequences([preprocessed_text])
    padded_text = pad_sequences(tokenised_text, maxlen=50, padding='post')
    return torch.tensor(padded_text, dtype=torch.long).unsqueeze(0)  # Add batch dimension

cnn_model = SentimentCNN(5000, 128, 100, [2, 3, 4], 3)
cnn_model.load_state_dict(torch.load('cnn_sentiment.pth'))

st.title("Sentiment Prediction of Online Store Review (Malay/English)")

review = st.text_input("Enter a store review in Malay and/or English: ")

if st.button("Get Sentiment"):
    if review.strip() != "":
      st.write('Sentiment prediction:', cnn_model(preprocess_text(review)))
    else:
      st.write("Please enter a review.")

if st.button("Clear"):
  st.text_input("Enter a store review in Malay and/or English: ", value="", key="new")
