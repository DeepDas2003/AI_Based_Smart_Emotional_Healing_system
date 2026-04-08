***Environment Overview and Motivation***

The Emotional Healing Environment is a custom AI simulation designed to recognize, interpret, and respond to human emotional expressions. It serves as a controlled framework in which images of faces—captured from diverse datasets—are processed to detect emotions such as happiness, sadness, anger, or neutrality. The environment leverages advanced computer vision techniques, including YOLOv8-based face detection, to extract face regions from color images, which are then converted to grayscale for consistent emotion analysis.

Within this environment, each step involves presenting an input (a face image) to the system, which then predicts the corresponding emotional state. The system assigns a reward score based on the accuracy or relevance of the detected emotion, simulating a feedback loop similar to human emotional response evaluation, Even generates advise so that humans by reading and acting  that way can overcome that emotion too. This iterative process allows for the testing of reinforcement learning techniques and AI decision-making strategies in a well-defined, reproducible setting.

The environment is modular and can be integrated with external AI models, such as Hugging Face-compatible language models, to provide additional guidance, supportive messages, or suggestions for emotional healing interventions. This hybrid setup enables both perceptual analysis (emotion detection) and interactive AI communication (friendly or supportive responses).


Mental health and emotional well-being are increasingly recognized as critical components of overall health. However, access to personalized support is often limited due to social, economic, or logistical barriers. The motivation behind developing this Emotional Healing Environment stems from the need to create AI-driven systems capable of:

Understanding human emotions: By analyzing facial cues and expressions, the system can detect subtle emotional states, facilitating more empathetic interactions.
Providing personalized feedback: Through reinforcement learning and reward-based training, the system can generate responses tailored to an individual’s emotional state, promoting positive mental reinforcement.
Enabling research and experimentation: The controlled environment allows researchers and developers to experiment with AI models for emotional analysis, testing different architectures, reward structures, and response strategies without ethical concerns related to real users.
Bridging the gap between AI and mental health support: By integrating perceptual AI and generative AI models, the environment demonstrates how technology can assist in emotional healing and well-being at scale, potentially augmenting traditional mental health resources.

Overall, this environment represents a step towards empathetic AI, where technology is not only capable of recognizing human emotions but also of interacting in ways that support mental and emotional health, providing both insights and guidance in a safe, experimental setup.




***Definitions of action and observation spaces***

| **Component**       | **Type**              | **Description**                                                                     |
| ------------------- | --------------------- | ----------------------------------------------------------------------------------- |
| Action              | Image                 | Preprocessed face image; optionally include AI-guided textual messages.             |
| Observation         | Dictionary            | Contains predicted emotion, confidence score, image metadata, and step info.        |
| Reward              | float                 | Numerical feedback based on action effectiveness (e.g., correct emotion detection). |
| Episode Termination | Boolean               | Signals the end of a session after MAX_STEPS or achieving success criteria.         |
<img width="1551" height="773" alt="image" src="https://github.com/user-attachments/assets/15f99ef1-4323-4b91-9a47-b09033718dbd" />


<img width="1532" height="744" alt="image" src="https://github.com/user-attachments/assets/20c6a8bb-9bfa-4ad7-a971-17789f432a83" />

***Task descriptions with expected difficulty levels***

This task describes the classificication facial emotions by recognising and scanning faces . Based on this there are Task categories on thich if it takes less no of steps considers easier task and if takes more no of steps then hard
<img width="769" height="100" alt="Screenshot 2026-04-08 203620" src="https://github.com/user-attachments/assets/fe6387ce-43ec-4971-9672-4e89e3b51575" />

if no of steps < = 4  then task 1 gets competed and considered easy task
if no of steps < = 8 then task 2 gets competed and considered moderate task
if no of steps < = 12  then task 3 gets completed and considered hard task
if no of steps goes above 12 then  environment gets reset automatically instead of getting in trap of infinte loop

Another  most difficulty is that due to my cnn model accuracy 70% sometimes model misdetects emotions althoug I used yolov8n-face-lindevs.pt for caturing and extracting faces

My cnn model's confusion matrix
<img width="858" height="656" alt="Screenshot 2026-04-02 231922" src="https://github.com/user-attachments/assets/160aecd3-5451-430c-a985-87d209aa42b7" />


***Setup and usage instructions***


For cloning git ling in terminal

git clone https://github.com/DeepDas2003/AI_Based_Smart_Emotional_Healing_system
cd AI_Based_Smart_Emotional_Healing_system

*Setup*

pip install -r requirements.txt

*Environment Variable*

Set the following environment variables:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

running

python app.py

***Baseline Performance***

Baseline metrics will be updated after evaluation and testing.
For example 
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4.1-mini
set HF_TOKEN=your_huggingface_token



