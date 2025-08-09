🎯 Features

📄 PDF Processing: Extract and classify images from PDF documents
🧠 CNN Architecture: Custom-built convolutional neural network
🌐 Web Interface: Interactive Streamlit web application
📊 Real-time Training: Live progress tracking and metrics
🎨 Visualization: Training plots and confusion matrices
📥 Export Results: Download classification results as CSV
☁️ Google Colab Support: Run directly in Colab with ngrok

📊 Model Architecture
CNN Architecture:
├── Conv2D(32) + MaxPool + Dropout(0.25)
├── Conv2D(64) + MaxPool + Dropout(0.25)  
├── Conv2D(128) + MaxPool + Dropout(0.25)
├── Conv2D(256) + MaxPool + Dropout(0.25)
├── Flatten + Dense(512) + Dropout(0.5)
├── Dense(256) + Dropout(0.5)
└── Dense(1, sigmoid) # Binary classification

📁 Project Structure
├── notebooks/              # Jupyter notebooks for Colab
├── streamlit_app/          # Web application code
├── src/                    # Core Python modules
├── sample_data/            # Sample datasets and generators
├── docs/                   # Documentation
└── assets/                 # Images and demos

📚 Documentation

📖 Setup Guide
🎯 Usage Instructions
🏗️ Model Architecture
📊 Performance Analysis

🔬 Technical Details

Framework: TensorFlow/Keras
Architecture: Convolutional Neural Network (CNN)
Input: 224x224 RGB images
Output: Binary classification (Medical/Non-Medical)
Loss: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy, Precision, Recall

📈 Performance Metrics
MetricScoreAccuracy89.3%Precision87.6%Recall91.2%F1-Score89.4%
🎯 Use Cases

Medical Document Processing: Automatically sort medical PDFs
Research Data Organization: Classify research images
Healthcare Workflow: Streamline medical image handling
Educational Tools: Teach medical image recognition

⚠️ Important Notes

Educational Purpose: This tool is for educational and research purposes only
Not for Medical Diagnosis: Should not be used for actual medical diagnosis
Data Privacy: Ensure compliance with healthcare data regulations
Model Limitations: Performance depends on training data quality

🙏 Acknowledgments

TensorFlow Team for the deep learning framework
Streamlit for the amazing web app framework
PyMuPDF for PDF processing capabilities
Open Source Community for the tools and libraries

📞 Contact

Author: Pratik Nainwal
Email: pratiknainwal2@gmail.com
GitHub: hunterrPN
