# 🎧 **StressBuster AI**  
*A Voice-Powered Stress Detection and Management App*

---

## 🌟 **Overview**  
StressBuster AI is an innovative voice-based application designed to assess stress levels from **students' voice recordings**. By leveraging cutting-edge AI models like **OpenAI Whisper**, the app provides **personalized stress management tips**, helping users tackle stress effectively.

Say goodbye to stress with a simple voice recording and let StressBuster AI guide you towards a healthier mind!

---

## 🚀 **Features**
- 🎙️ **Voice Analysis**: Detects stress levels using advanced speech emotion recognition.
- 🧠 **Emotion Detection**: Identifies emotions like Happy, Sad, Angry, Neutral, and more!
- 📊 **Accurate Results**: Achieves over **91% accuracy** in emotion recognition.
- 🎯 **Stress Management Tips**: Offers tailored advice to reduce stress and improve well-being.
- 🔗 **Pre-trained Model**: Ready-to-use AI model for speech emotion recognition.

---

## 🛠️ **Tech Stack**
- **Languages**: 
  - HTML (73.6%)
  - Python (26.4%)

- **Tools**:
  - Wandb for experiment tracking
  - Google Drive for model hosting

---

## 📂 **Dataset**
- Combines multiple datasets for rich emotional diversity:
  - [RAVDESS](https://zenodo.org/records/1188976#.XsAXemgzaUk)
  - [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee/data)
  - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
- **Emotion Distribution**:
  | **Emotion** | **Count** |
  |-------------|-----------|
  | Sad         | 752       |
  | Happy       | 752       |
  | Angry       | 752       |
  | Neutral     | 716       |
  | Disgust     | 652       |
  | Fearful     | 652       |
  | Surprised   | 652       |
  | Calm        | 192       |
- *Note*: "Calm" samples excluded due to low representation.

---

## ⚡ **How It Works**
1. 🎤 **Record Voice**: Upload a voice recording of the user.
2. 🔍 **Analyze Emotions**: The AI model processes the audio to detect emotions.
3. 💡 **Provide Insights**: Displays stress levels and offers tailored stress management tips.

---

## 📥 **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Ritik650/StressBuster-AI.git
   cd StressBuster-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

---

#

---

## 📊 **Results**
| **Epoch** | **Loss** | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-----------|----------|--------------|---------------|------------|--------------|
| 25        | 0.5008   | 91.99%       | 92.30%        | 91.99%     | 91.98%       |

---

## 🧪 **Experiment Tracking**
Model training and evaluation are tracked using **Wandb**. Check out the experiment logs and visualizations [here](https://wandb.ai/firdhoworking-sepuluh-nopember-institute-of-technology/speech-emotion-recognition).

---

## 🤝 **Contributing**
We welcome contributions to enhance StressBuster AI!  
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add feature-name"`.
4. Push the branch: `git push origin feature-name`.
5. Open a Pull Request.

---

## 📜 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---



**Transform Stress into Strength with StressBuster AI!** 🙌  

---

Would you like me to assist you with committing this new README file or making additional edits?

