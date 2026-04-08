import torch
from torchvision import transforms
from PIL import Image
from model import EmotionResNet

EMOTION_ADVICE = {
    "angry": "Take deep breaths and count to 10.",
    "disgust": "Focus on something pleasant or neutral.",
    "fear": "Breathe slowly and calmly.",
    "happy": "Share your joy but stay balanced.",
    "neutral": "Maintain calm and focus.",
    "sad": "Talk to someone or do a small activity.",
    "surprise": "Stay calm and think clearly."
}

MODEL_PATH = "./third_best_model.pth"

class EmotionEnv:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        self.label_map = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
        self.max_steps = 12
        self.reset()

    def _load_model(self):
        model = EmotionResNet(num_classes=7).to(self.device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.eval()
        return model

    def reset(self):
        self.steps = 0
        self.emotion = "neutral"
        self.confidence = 0.0
        self.history = []
        self.total_reward = 0.0
        return {"emotion": self.emotion, "steps": self.steps}

    def step(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        prev_emotion = self.emotion
        self.steps += 1

        img_gray = img.convert("L")
        tensor = self.transform(img_gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        self.emotion = self.label_map[pred.item()]
        self.confidence = confidence.item()
        reward = self._compute_reward(prev_emotion, self.emotion)
        self.total_reward += reward
        self.history.append({"prev": prev_emotion, "new": self.emotion, "reward": reward})

        done = self.steps >= self.max_steps
        task_status = self._get_task_status()

        result = {
            "obs": {
                "emotion": self.emotion,
                "confidence": round(self.confidence,3),
                "steps": self.steps,
                "advice": EMOTION_ADVICE.get(self.emotion,"")
            },
            "reward": round(reward,3),
            "total_reward": round(self.total_reward,3),
            "done": done,
            "task_status": task_status
        }

        if done:
            self.reset()

        return result

    def _compute_reward(self, prev, curr):
        negative = ["angry","disgust","fear","sad"]
        positive = ["happy","surprise"]
        if curr in positive: return 0.5
        if prev in negative and curr=="neutral": return 0.8
        if prev in negative and curr in positive: return 1.0
        if prev in positive and curr=="neutral": return 0.5
        if curr==prev: return 0.2
        if curr in negative and prev not in negative: return 0.0
        if prev==positive[1] and curr==positive[0]: return 0.5
        return 0.0

    def _get_task_status(self):
        if self.steps <=4 and self.emotion in ["happy","neutral"]:
            return "Task 1 (Easy) Completed"
        elif self.steps <=8 and self.emotion in ["happy","neutral"]:
            return "Task 2 (Medium) Completed"
        elif self.steps <=12 and self.emotion in ["happy","neutral"]:
            return "Task 3 (Hard) Completed"
        elif self.steps>=self.max_steps:
            return "Max Steps Reached"
        return ""
