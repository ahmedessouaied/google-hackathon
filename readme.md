# AI Doctor: Multimodal Medical Assistant

> Where healthcare meets automation: Transforming unstructured medical data into actionable insights with voice, vision, and text capabilities.

## 🏆 1st Place: AI Anatomy Hackathon Project
GDG SUP'COM x Carthage - April 19, 2025

## 📋 Project Overview

This repository contains our smart Medical Chatbot (with Vision and Voice) developed for the AI Anatomy hackathon challenge. We've created a personalized AI Doctor voice assistant that combines advanced Natural Language Processing (NLP), computer vision, and speech recognition to transform healthcare delivery.

Our system leverages:
- **Meta Llama3 Vision 90B** multimodal LLM for exceptional image and text understanding
- **OpenAI Whisper** for precise speech-to-text conversion
- **Graph-based knowledge retrieval** for medical information access

We're also participating in the associated Kaggle challenge: **Clinical NLP Challenge: Medical Answer Generation**, which focuses on generating accurate answers from clinical case descriptions.

## 🎯 Key Components

### 🎤 Multimodal Medical Assistant
- Voice interface for natural interaction using OpenAI Whisper
- Image analysis capabilities via Meta Llama3 Vision 90B
- Text-based clinical reasoning and diagnosis assistance
- Seamless integration between all input modalities

### 📊 Graph RAG (Retrieval-Augmented Generation)
- Knowledge graph representation of medical entities and relationships
- Graph-based retrieval for contextual information
- Enhanced reasoning by traversing medical knowledge connections
- Neo4J integration for efficient querying and visualization

### 📝 Medical Report Generator
- Automatic generation of structured clinical reports from unstructured notes
- Medical terminology standardization and formatting
- Patient-friendly summary creation
- Integration with standard healthcare documentation formats

### ⚠️ Diabetic Risk Analyzer
- Early identification of diabetes risk factors
- Pattern recognition across patient history
- Predictive modeling for diabetes progression
- Personalized intervention recommendations

### 📅 Automatic Scheduling System
- Priority-based appointment scheduling
- Resource allocation optimization
- Patient follow-up management
- Integration with clinical workflows

### 📓 Kaggle Challenge Notebook
- Specialized solution for medical answer generation
- Fine-tuned NLP model for clinical question answering
- Performance optimization for Rouge-L score
- Comprehensive analysis of clinical reasoning

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/ihebbettaibe/ai-doctor.git
cd ai-doctor

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Neo4J for Graph RAG (requires Docker)
docker-compose up -d neo4j

# Download model weights
python download_models.py
```

## 🚀 Usage

```bash
# Run the main application with voice and vision capabilities
python app.py --mode full

# Run with text-only mode
python app.py --mode text

# Run the Kaggle notebook
jupyter notebook notebooks/kaggle_medical_qa.ipynb

# For API-only mode
python api.py
```

## 📊 Kaggle Challenge: Medical Answer Generation

Our system includes a specialized module for the Kaggle challenge that:
- Takes medical case questions as input
- Leverages our Graph RAG for knowledge retrieval
- Generates relevant, medically accurate answers
- Achieves competitive Rouge-L scores on the leaderboard

### Sample Input/Output:

**Input Question:**
```
A 23-year-old pregnant woman at 22 weeks of gestation presents with dysuria, urinary frequency, and lower abdominal pain for 2 days. She has no allergies. What is the most appropriate antibiotic treatment?
```

**Generated Answer:**
```
Nitrofurantoin
```

## 🔬 Datasets

We used multiple data sources to train and evaluate our models:

- Hackathon-provided synthetic medical text
- Public datasets (modified MIMIC-III, i2b2)
- Kaggle challenge dataset (clinical cases with expert answers)
- Voice recordings for multimodal analysis training
- Medical image datasets for visual component training

## 🛠️ Project Structure

```
ai-doctor/
├── app.py                    # Main application
├── api.py                    # FastAPI backend
├── models/
│   ├── llama3/               # Meta Llama3 Vision 90B integration
│   ├── whisper/              # OpenAI Whisper integration
│   └── graph_rag/            # Knowledge graph components
├── modules/
│   ├── voice_processor/      # Speech recognition and synthesis
│   ├── vision_processor/     # Medical image analysis
│   ├── report_generator/     # Clinical report creation
│   ├── diabetic_risk/        # Diabetes risk analysis
│   └── scheduler/            # Appointment management
├── data/                     # Training and test datasets
├── kaggle_solution/          # Medical answer generation module
├── notebooks/                # Development notebooks
│   └── kaggle_medical_qa.ipynb  # Kaggle challenge solution
├── utils/                    # Helper functions
├── tests/                    # Unit and integration tests
├── download_models.py        # Script to download model weights
└── requirements.txt          # Dependencies
```

## 👥 Team

- Iheb Bettaieb
- Ahmed Essouaied
- Wassim Derwich
- Nidhal Sanaa

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GDG SUP'COM x Carthage for organizing the AI Anatomy hackathon
