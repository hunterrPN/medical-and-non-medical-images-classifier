"""
Medical Image Classifier - Streamlit App
========================================
A web application for classifying medical vs non-medical images from PDFs using CNN.
"""

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import fitz  # PyMuPDF
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time

# Configure TensorFlow to avoid warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Page configuration
st.set_page_config(
    page_title="🏥 Medical Image Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-card {
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class ImageExtractor:
    """Extract images from PDF files"""
    
    def extract_from_pdf(self, pdf_bytes):
        """Extract images from PDF bytes"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_pages = min(len(doc), 100)  # Limit to 100 pages
            
            for page_num in range(total_pages):
                status_text.text(f"Processing page {page_num + 1}/{total_pages}...")
                progress_bar.progress((page_num + 1) / total_pages)
                
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                            image = Image.open(io.BytesIO(img_data))
                            
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            # Filter out very small images
                            if image.size[0] > 64 and image.size[1] > 64:
                                images.append({
                                    'image': image,
                                    'page': page_num + 1,
                                    'index': img_index + 1,
                                    'size': image.size
                                })
                        
                        pix = None
                    except Exception as e:
                        st.warning(f"Skipped image {img_index + 1} on page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            progress_bar.empty()
            status_text.empty()
            
            return images
            
        except Exception as e:
            st.error(f"Error extracting images from PDF: {str(e)}")
            return []

class MedicalImageClassifier:
    """CNN model for medical image classification"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build CNN architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the model with progress tracking"""
        
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        
        # Custom callback for Streamlit
        class StreamlitCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                
                # Update metrics
                with metrics_container.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Loss", f"{logs.get('loss', 0):.4f}")
                    col2.metric("Accuracy", f"{logs.get('accuracy', 0):.4f}")
                    col3.metric("Val Loss", f"{logs.get('val_loss', 0):.4f}")
                    col4.metric("Val Accuracy", f"{logs.get('val_accuracy', 0):.4f}")
        
        # Callbacks
        callbacks = [
            StreamlitCallback(),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Clear progress elements
        progress_bar.empty()
        status_text.empty()
        metrics_container.empty()
        
        return self.history
    
    def predict(self, image):
        """Predict single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label = "medical" if prediction > 0.5 else "non-medical"
        
        return prediction, label, confidence

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    return img_array

def create_dataset(image_data, img_size=224):
    """Create dataset from extracted images"""
    X, y = [], []
    split_point = len(image_data) // 2
    
    progress_bar = st.progress(0)
    
    for i, img_info in enumerate(image_data):
        progress_bar.progress((i + 1) / len(image_data))
        
        # Preprocess image
        processed_img = preprocess_image(img_info['image'], (img_size, img_size))
        X.append(processed_img)
        
        # Label: first half = medical (1), second half = non-medical (0)
        y.append(1 if i < split_point else 0)
    
    progress_bar.empty()
    return np.array(X), np.array(y)

def display_training_plots(history):
    """Display training history plots"""
    if not history or not history.history:
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Loss', 'Precision', 'Recall'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['accuracy'], 
                            name='Training', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_accuracy'], 
                            name='Validation', line=dict(color='red')), row=1, col=1)
    
    # Loss
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['loss'], 
                            name='Training', line=dict(color='blue'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_loss'], 
                            name='Validation', line=dict(color='red'), showlegend=False), row=1, col=2)
    
    # Precision
    if 'precision' in history.history:
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['precision'], 
                                name='Training', line=dict(color='blue'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_precision'], 
                                name='Validation', line=dict(color='red'), showlegend=False), row=2, col=1)
    
    # Recall
    if 'recall' in history.history:
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['recall'], 
                                name='Training', line=dict(color='blue'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_recall'], 
                                name='Validation', line=dict(color='red'), showlegend=False), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Training Progress", showlegend=True)
    fig.update_xaxes(title_text="Epoch")
    
    st.plotly_chart(fig, use_container_width=True)

def display_confusion_matrix(y_true, y_pred):
    """Display confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Non-Medical', 'Medical'],
        y=['Non-Medical', 'Medical'],
        color_continuous_scale='Blues',
        text_auto=True,
        title="Confusion Matrix"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Image Classifier</h1>
        <p>AI-powered classification of medical vs non-medical images from PDFs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Model Configuration")
        epochs = st.slider("Training Epochs", 5, 50, 20, help="Number of training epochs")
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1, help="Training batch size")
        img_size = st.selectbox("Image Size", [128, 224, 256], index=1, help="Input image resolution")
        
        st.markdown("## 📊 Model Status")
        if st.session_state.model:
            st.success("✅ Model trained and ready")
        else:
            st.info("🔄 No model trained yet")
        
        st.markdown("## ℹ️ How to Use")
        st.markdown("""
        1. **Training**: Upload PDF with medical images in first half, non-medical in second half
        2. **Prediction**: Upload any PDF to classify its images
        3. **Results**: View classifications and download CSV
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏋️ Training", "🔍 Prediction", "📊 Analysis"])
    
    with tab1:
        st.markdown("## 🏋️ Model Training")
        
        st.markdown("""
        <div class="info-box">
        <strong>📋 Training Data Format:</strong>
        <ul>
        <li>Upload a PDF with medical and non-medical images</li>
        <li>First half of images should be medical (X-rays, MRIs, CT scans, etc.)</li>
        <li>Second half should be non-medical (landscapes, objects, people, etc.)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            training_pdf = st.file_uploader(
                "📄 Upload Training PDF",
                type=['pdf'],
                help="PDF containing medical images in first half, non-medical in second half"
            )
        
        with col2:
            train_button = st.button("🚀 Start Training", type="primary", use_container_width=True)
        
        if training_pdf and train_button:
            with st.spinner("🔄 Training model..."):
                # Extract images
                st.info("📥 Extracting images from PDF...")
                extractor = ImageExtractor()
                image_data = extractor.extract_from_pdf(training_pdf.getvalue())
                
                if len(image_data) < 6:
                    st.error("❌ Need at least 6 images for training (3 medical + 3 non-medical)")
                    return
                
                st.success(f"✅ Extracted {len(image_data)} images from PDF")
                
                # Display statistics
                split_point = len(image_data) // 2
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Images", len(image_data))
