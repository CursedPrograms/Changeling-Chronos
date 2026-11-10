"""Age a face forwards or backwards: align -> e4e invert -> (optional) pivotal tuning -> latent edit -> paste back."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .align import FaceAligner, paste_back
from .classify import GenderClassifier, RaceClassifier
from .encoder import Encoder4Editing
from .stylegan2 import Generator
from . import weights

# Age lives in the coarse and middle W+ layers (face shape, skin, hair line, wrinkles); the finest
# layers (colour, lighting) stay those of the input photo.
EDIT_LAYERS = slice(0, 10)
# Distance moved along the unit age direction at an amount of 1.0 (the slider's end).
BASE_STEP = 6.0


@dataclass
class AgeResult:
    image: np.ndarray           # full photo with the aged face pasted back
    aligned: np.ndarray         # aligned 1024x1024 input crop
    reconstruction: np.ndarray  # generator's reconstruction of the crop (before editing)
    edited: np.ndarray          # aligned 1024x1024 aged crop
    amount: float               # -1 (much younger) .. +1 (much older)


def to_tensor(image, size, device):
    """RGB uint8 HxWx3 -> 1x3xSxS float in [-1, 1]."""
    t = torch.from_numpy(image).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode='area')
    return t


def to_image(tensor):
    """1x3xHxW in [-1, 1] -> RGB uint8 HxWx3."""
    t = tensor[0].detach().clamp(-1, 1).add(1).mul(127.5).round()
    return t.permute(1, 2, 0).byte().cpu().numpy()


class AgeShifter:
    def __init__(self, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        paths = weights.ensure_all()

        self.aligner = FaceAligner(str(paths['yunet']))

        ckpt = torch.load(paths['e4e'], map_location='cpu', weights_only=True)
        state = ckpt['state_dict']
        self.encoder = Encoder4Editing(1024)
        self.encoder.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')})
        self.generator = Generator(1024)
        self.generator.load_state_dict({k[len('decoder.'):]: v for k, v in state.items() if k.startswith('decoder.')})
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        del ckpt, state

        # The encoder is ~1 GB of weights and runs once per photo, so it stays on the CPU;
        # the GPU is left for the generator and pivotal tuning (fits in 4 GB).
        self.encoder.eval()
        self.generator.eval().to(self.device).requires_grad_(False)

        # age.npy: one 512-d vector per layer (all the same), kept as a unit vector. Positive along it
        # is YOUNGER (checked on test faces), so SIGN = -1 makes +amount older.
        d = torch.from_numpy(np.load(paths['age'])[0]).float().to(self.device)
        self.direction = d / d.norm()
        # In this generator age and gender are tangled inside the network (ageing pushes old women towards
        # men), though the two direction vectors are nearly at right angles (taking one out of the other
        # does nothing). So the gender is kept by looking: FairFace reads the face before and after, and
        # a drifted result is nudged back along the gender direction (positive = more male).
        g = torch.from_numpy(np.load(paths['gender'])[0]).float().to(self.device)
        self.gender_direction = g / g.norm()
        self.classifier = GenderClassifier(paths['gender_cls'])
        # Getting younger also drifts a face towards East Asian (the age direction leans along Natio's regional
        # directions: about +0.27 towards East Asian and -0.28 away from European; FFHQ's children are a different
        # crowd from its adults), so a blonde woman became an East Asian girl. Kept the same way as the gender: the
        # age step goes without its share along the three regional directions, and FairFace's four groups read
        # the face before and after; a drifted result is nudged back towards its group and away from the new one.
        self.race_directions = {}
        for group, key in (('white', 'race_white'), ('black', 'race_black'), ('asian', 'race_yellow')):
            r = torch.from_numpy(np.load(paths[key])[0]).float().to(self.device)
            self.race_directions[group] = r / r.norm()
        basis = torch.linalg.qr(torch.stack(list(self.race_directions.values()), 1))[0]   # 512 x 3, orthonormal
        kept = self.direction - basis @ (basis.T @ self.direction)
        self.direction_keep_race = kept / kept.norm()
        self.race_classifier = RaceClassifier(paths['race_cls'])
        self._lpips = None

    SURE = 0.8                  # Auto trusts FairFace's reading only this sure
    KEEP_GENDER_STEP = 1.2      # how far each nudge back goes (the swap itself uses 5)
    KEEP_GENDER_TRIES = 6
    AGE_GENDER_HOLD = 3.0       # the push back towards the kept gender, per unit of ageing (see age())
    KEEP_RACE_STEP = 1.0        # each nudge back towards the face's own group (Natio's whole change is about 2)
    KEEP_RACE_TRIES = 6

    def _p(self, image, gender):
        """FairFace's probability that `image` (an aligned face) is `gender`."""
        seen, confidence = self.classifier(image)
        return confidence if seen == gender else 1 - confidence

    @torch.no_grad()
    def _keep_gender(self, w_edit, gender, p_before):
        """Nudge an aged latent back along the gender direction until FairFace is about as sure of the
        original gender as it was for the photo (not just past 50 %: an old woman read 55 % female already
        looks like a man)."""
        sign = -1.0 if gender == 'female' else 1.0
        want = min(0.95, p_before) - 0.05
        for _ in range(self.KEEP_GENDER_TRIES):
            if self._p(to_image(self.generator(w_edit)), gender) >= want:
                break
            w_edit = w_edit.clone()
            w_edit[:, EDIT_LAYERS] += sign * self.KEEP_GENDER_STEP * self.gender_direction
        return w_edit

    @torch.no_grad()
    def _keep_race(self, w_edit, group, p_before):
        """Nudge an aged latent back until FairFace is about as sure of the face's group (white, black, asian,
        indian) as it was for the photo: towards the group's own direction (Indian has none) and away from the
        group the face drifted to."""
        own = RaceClassifier.GROUPS.index(group)
        want = min(0.95, p_before) - 0.05
        for _ in range(self.KEEP_RACE_TRIES):
            p = self.race_classifier(to_image(self.generator(w_edit)))
            if p[own] >= want:
                break
            p[own] = -1
            drifted = RaceClassifier.GROUPS[int(p.argmax())]
            step = torch.zeros_like(self.direction)
            if group in self.race_directions:
                step += self.race_directions[group]
            if drifted in self.race_directions:
                step -= self.race_directions[drifted]
            w_edit = w_edit.clone()
            w_edit[:, EDIT_LAYERS] += self.KEEP_RACE_STEP * step
        return w_edit

    SIGN = -1.0

    @torch.no_grad()
    def invert(self, aligned):
        w = self.encoder(to_tensor(aligned, 256, 'cpu'))
        return w.to(self.device) + self.latent_avg

    def edit(self, w, amount, keep_race=False):
        """amount -1 (much younger) .. +1 (much older). keep_race: the age step without its share along the
        regional directions."""
        w = w.clone()
        direction = self.direction_keep_race if keep_race else self.direction
        w[:, EDIT_LAYERS] += self.SIGN * amount * BASE_STEP * direction
        return w

    def tune(self, w, aligned, steps, lr=3e-4, progress=None):
        """Pivotal Tuning (Roich et al. 2021): fine-tune the generator so that the e4e latent
        reproduces this exact face, so edits keep the person's identity. Modifies
        self.generator in place; age() restores the pretrained weights afterwards."""
        if self._lpips is None:
            import warnings
            import lpips
            with warnings.catch_warnings():  # lpips uses deprecated torchvision/torch.load arguments
                warnings.simplefilter('ignore')
                self._lpips = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)

        gen = self.generator.requires_grad_(True)
        opt = torch.optim.Adam(gen.parameters(), lr=lr)
        target = to_tensor(aligned, 1024, self.device)
        target_256 = F.interpolate(target, size=(256, 256), mode='area')

        for i in range(steps):
            out = gen(w)
            out_256 = F.interpolate(out, size=(256, 256), mode='area')
            loss = self._lpips(out_256, target_256).mean() + F.mse_loss(out_256, target_256)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if progress:
                progress(i + 1, steps, loss.item())
            if loss.item() < 0.02:
                break

        gen.requires_grad_(False)

    def _align(self, image):
        landmarks = self.aligner.detect(image)
        if landmarks is None:
            raise ValueError('No face found in the image.')
        return self.aligner.align(image, landmarks, 1024)

    def age(self, image, amount, tune_steps=0, progress=None, amounts=None, keep_hair=False, output='frame',
            keep_gender='auto', keep_race='auto'):
        """Age the largest face in `image` (RGB uint8) by `amount` (-1 much younger .. +1 much older).
        With `amounts` (a list), returns one result per amount from a single tuning (for a timeline).
        output: 'frame' (the default) makes .image the aged face frame itself; 'photo' pastes it back into the
        whole photo (with keep_hair, only the face: the photo keeps its own hair).
        keep_gender: 'female' or 'male' keeps that gender as the face ages; 'auto' reads it with FairFace and
        keeps it only when FairFace is sure (80 %+; it often isn't for old faces); None doesn't.
        keep_race: True holds the face's FairFace group (white, black, asian, indian) as it ages; 'auto' only when
        FairFace is sure of it (80 %+); None doesn't."""
        aligned, matrix = self._align(image)
        w = self.invert(aligned)
        gender, p_gender = self.classifier(aligned)
        if keep_gender in ('female', 'male'):
            if keep_gender != gender:
                gender, p_gender = keep_gender, 0.9
        elif keep_gender == 'auto' and p_gender < self.SURE:
            keep_gender = None
        self.last_gender = (gender, p_gender, keep_gender)
        race_p = self.race_classifier(aligned)
        group, p_group = RaceClassifier.GROUPS[int(race_p.argmax())], float(race_p.max())
        if keep_race == 'auto':
            keep_race = p_group >= self.SURE
        keep_race = bool(keep_race)
        self.last_race = (group, p_group, keep_race)

        pretrained = None
        if tune_steps > 0:
            pretrained = {k: v.cpu().clone() for k, v in self.generator.state_dict().items()}
        try:
            if pretrained is not None:
                self.tune(w, aligned, tune_steps, progress=progress)
            with torch.no_grad():
                recon = to_image(self.generator(w))
                results = []
                for a in (amounts if amounts is not None else [amount]):
                    w_edit = self.edit(w, a, keep_race)
                    if keep_gender and a > 0.05:
                        # ageing drifts a face towards male (heavier brows and jaw, StyleGAN learned that from
                        # FFHQ's older faces), and FairFace doesn't see it on a woman with long hair: so a
                        # push back along the gender direction, as far as the face is aged, before the check
                        sign = -1.0 if gender == 'female' else 1.0
                        w_edit[:, EDIT_LAYERS] += sign * self.AGE_GENDER_HOLD * a * self.gender_direction
                    if keep_gender and abs(a) > 0.05:
                        w_edit = self._keep_gender(w_edit, gender, p_gender)
                    if keep_race and abs(a) > 0.05:
                        w_edit = self._keep_race(w_edit, group, p_group)
                    edited = to_image(self.generator(w_edit))
                    results.append(AgeResult(image=edited if output == 'frame' else paste_back(image, edited, matrix, face_only=keep_hair), aligned=aligned,
                                             reconstruction=recon, edited=edited, amount=a))
        finally:
            if pretrained is not None:
                self.generator.load_state_dict(pretrained)
                torch.cuda.empty_cache()
        return results if amounts is not None else results[0]
