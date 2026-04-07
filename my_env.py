import torch
from torchvision import transforms
from PIL import Image
from model import EmotionResNet

MODEL_PATH = "./third_best_model.pth"

EMOTION_ADVICE = {
    "angry": "Take deep breaths and count to 10.",
    "disgust": "Focus on something pleasant.",
    "fear": "Breathe slowly and calmly.",
    "happy": "Stay balanced and positive.",
    "neutral": "Maintain calm focus.",
    "sad": "Talk to someone.",
    "surprise": "Stay calm and think clearly."
}

class EmotionEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = self._load_model()

        self.transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

        self.label_map = {
            0:'angry',1:'disgust',2:'fear',
            3:'happy',4:'neutral',5:'sad',6:'surprise'
        }

        self.max_steps = 12
        self.reset()

    def _load_model(self):
        model = EmotionResNet(num_classes=7).to(self.device)
        try:
            model.load_state_dict(
               torch.load(MODEL_PATH, map_location=self.device),
               strict=False
            )
        except Exception as e:
            print("Model load failed:", e)
        model.eval()
        torch.set_grad_enabled(False)
        return model

    def reset(self):
        self.steps = 0
        self.emotion = "neutral"
        self.confidence = 0.0
        self.history = []
        self.total_reward = 0.0

    def step(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
            
        prev = self.emotion
        self.steps += 1

        img_gray = img.convert("L")
        tensor = self.transform(img_gray).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        self.emotion = self.label_map[pred.item()]
        self.confidence = confidence.item()

        reward = self._compute_reward(prev, self.emotion)
        self.total_reward += reward

        self.history.append({"prev": prev, "new": self.emotion})

        done = (
            self.steps >= self.max_steps or
            self.emotion == "happy" or
            self._no_improvement()
        )

        return {
            "obs": {
                "emotion": self.emotion,
                "confidence": round(self.confidence,3),
                "steps": self.steps,
                "advice": EMOTION_ADVICE.get(self.emotion,"")
            },
            "reward": round(reward,3),
            "total_reward": round(self.total_reward,3),
            "done": done
        }

    def _compute_reward(self, prev, curr):
        negative = ["angry","disgust","fear","sad"]
        positive = ["happy","surprise"]

        if curr in positive: return 0.5
        if prev in negative and curr == "neutral": return 0.8
        if prev in negative and curr in positive: return 1.0
        if curr == prev: return 0.2
        return 0.0

    def _no_improvement(self):
        if len(self.history) < 3:
            return False
        negative = ["angry","disgust","fear","sad"]
        return all(h["new"] in negative for h in self.history[-3:])
