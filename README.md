# VizioText - AI-Powered Image Captioning

## Description

VizioText is an AI-powered image captioning web application that generates high-quality captions for images using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) models. It integrates **VGG19** for feature extraction and **LLM-based language models** for improved fluency and accuracy, providing users with precise and contextually appropriate image descriptions.

## Features

- **AI-driven image captioning** powered by CNN + LSTM models.
- **Feature extraction** using the **VGG19** model.
- **Enhanced language fluency** with **microsoft/git-base** integration.
- **User-friendly web interface** developed with Streamlit.
- **Supports multiple image formats** for caption generation.

## Tech Stack

- **Machine Learning Models**: CNN, LSTM, VGG19
- **Frameworks & Libraries**: TensorFlow, Streamlit
- **Large Language Model**: microsoft/git-base
- **Programming Language**: Python

## Installation & Execution

### 1. Clone the Repository

Clone the project repository and navigate to the project folder:

```sh
git clone https://github.com/your-repository.git
cd viziotext
```

### 2. Install Dependencies

Install the required dependencies:

```sh
pip install -r requirements.txt
```

### 3. Run the Web Application

Start the Streamlit application:

```sh
streamlit run app.py
```
