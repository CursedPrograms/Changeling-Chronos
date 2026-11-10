"""Face classifiers (FairFace): gender (ViT, int8 ONNX) and the 4 FairFace groups (ResNet-34, ONNX)."""
import cv2
import numpy as np
import onnxruntime as ort


class GenderClassifier:
    # FairFace images are tight face crops; trim the looser FFHQ framing to match.
    MARGIN = 0.16

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, aligned):
        """Return ('male' | 'female', confidence) for an FFHQ-aligned RGB crop."""
        m = int(aligned.shape[0] * self.MARGIN)
        face = aligned[m:aligned.shape[0] - m, m:aligned.shape[1] - m]
        x = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1
        logits = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None]})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        # Labels: 0 = Female, 1 = Male
        return ('male', float(p[1])) if p[1] > p[0] else ('female', float(p[0]))


class RaceClassifier:
    """FairFace's 4-group model (ResNet-34, ONNX): White, Black, Asian, Indian. Chronos uses it only to notice a
    face drifting away from its own group as it ages, not to label anyone."""
    GROUPS = ('white', 'black', 'asian', 'indian')
    MARGIN = 0.16
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, aligned):
        """The four groups' probabilities (a numpy array, in GROUPS order) for an FFHQ-aligned RGB crop."""
        m = int(aligned.shape[0] * self.MARGIN)
        face = aligned[m:aligned.shape[0] - m, m:aligned.shape[1] - m]
        x = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - self.MEAN) / self.STD
        # 18 outputs: 7 race slots (this model fills the first 4), 2 gender, 9 age
        logits = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None]})[0][0][:4]
        p = np.exp(logits - logits.max())
        return p / p.sum()
