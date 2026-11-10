"""Changeling Chronos: age a face forwards or backwards in a photo.

    python main.py                          # the window
    python main.py photo.jpg --age 0.6      # 60% older, writes photo_older_60.jpg
    python main.py photo.jpg --age -0.5     # 50% younger
    python main.py photo.jpg --timeline     # five ages side by side, writes photo_timeline.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

QUALITY_STEPS = {'fast': 0, 'balanced': 100, 'best': 250}
KEEP_RACE = {'auto': 'auto', 'on': True, 'off': None}


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    ok, buf = cv2.imencode(Path(path).suffix or '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit(f'Could not write image: {path}')
    buf.tofile(str(path))


def run_cli(args):
    from chronos import AgeShifter
    from window import TIMELINE, timeline_strip

    shifter = AgeShifter(args.device)
    steps = QUALITY_STEPS[args.quality] if args.steps is None else args.steps

    def progress(i, total, loss):
        print(f'\r  learning the face {i}/{total}  loss {loss:.4f}', end='', flush=True)

    for path in args.images:
        src = Path(path)
        print(f'{src.name}:')
        if args.timeline:
            results = shifter.age(read_rgb(src), 0, steps, progress, amounts=TIMELINE, keep_hair=args.keep_hair,
                                   keep_gender=None if args.keep_gender == 'off' else args.keep_gender,
                                   keep_race=KEEP_RACE[args.keep_race])
            image, tag = timeline_strip(results), 'timeline'
        else:
            output = 'photo' if (args.whole_photo or args.keep_hair) else 'frame'
            result = shifter.age(read_rgb(src), args.age, steps, progress, keep_hair=args.keep_hair, output=output,
                                 keep_gender=None if args.keep_gender == 'off' else args.keep_gender,
                                 keep_race=KEEP_RACE[args.keep_race])
            image = result.image
            tag = ('older' if args.age >= 0 else 'younger') + f'_{abs(args.age) * 100:.0f}'
        if steps:
            print()
        out = Path(args.output) if args.output and len(args.images) == 1 else src.with_name(f'{src.stem}_{tag}{src.suffix}')
        write_rgb(out, image)
        print(f'  saved {out}')


def run_ui(args):
    from chronos import AgeShifter
    from window import Window

    Window(AgeShifter, args.device).run()


def main():
    parser = argparse.ArgumentParser(description='Changeling Chronos: age a face forwards or backwards (StyleGAN2 + e4e + PTI).')
    parser.add_argument('images', nargs='*', help='Input photo(s). Omit to open the window.')
    parser.add_argument('-o', '--output', help='Output path (single input only).')
    parser.add_argument('--age', type=float, default=0.5, help='-1 (much younger) .. +1 (much older). Default 0.5.')
    parser.add_argument('--timeline', action='store_true', help='Five ages side by side instead of one.')
    parser.add_argument('--keep-gender', choices=['auto', 'female', 'male', 'off'], default='auto',
                        help='Hold the face\'s gender while ageing (auto: only when FairFace is sure).')
    parser.add_argument('--keep-race', choices=list(KEEP_RACE), default='auto',
                        help='Hold the face\'s ethnicity while ageing: younger faces drift towards East Asian '
                             'without it (auto: only when FairFace is sure).')
    parser.add_argument('--whole-photo', action='store_true',
                        help='Paste the result back into the whole photo (default: just the aged face frame).')
    parser.add_argument('--keep-hair', action='store_true',
                        help='Whole photo, but only the face changes; the photo keeps its own hair.')
    parser.add_argument('--quality', choices=list(QUALITY_STEPS), default='balanced',
                        help='fast = encoder only; balanced/best = also learn the face first (keeps identity).')
    parser.add_argument('--steps', type=int, help='Override the number of face-learning steps.')
    parser.add_argument('--device', help='cuda or cpu (default: cuda if available).')
    parser.add_argument('--download-models', action='store_true', help='Download any missing models and exit.')
    args = parser.parse_args()

    if args.download_models:
        from chronos import weights
        weights.ensure_all()
    elif args.images:
        run_cli(args)
    else:
        run_ui(args)


if __name__ == '__main__':
    main()
