"""window.py - Changeling Chronos's window, in Changeling's look (customtkinter, the Atelier blue).

Open a photo, slide Age towards younger or older, press Age. The work runs on its own thread, with a
completion bar while it learns your face (Balanced and Best fine-tune the generator on it, so the result
still looks like you). Timeline makes five ages at once (much younger to much older) from one learning
run. Save writes the result next to the photo, or wherever you choose.
"""
import os
import threading
import time
import tkinter as tk
import webbrowser
from gallery import remember, show as show_gallery
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
TAGLINE = "a face, forwards or backwards in time"
GITHUB = "https://github.com/CursedPrograms/Changeling-Chronos"
KOFI = "https://ko-fi.com/cursedentertainment"
PINK = "#007bb7"

BLUE, BLUE_HOVER = "#007ab7", "#005184"     # the Cursed site's blue: the important buttons, and "male"


def gender_tint(seg):
    """Colours a gender selector by what's picked: female pink, male blue, anything else the usual pink."""
    def tint(value=None):
        v = (value or seg.get() or "").lower()
        colour = PINK if ("female" in v or "woman" in v) else BLUE if ("male" in v or v == "man") else PINK
        seg.configure(selected_color=colour, selected_hover_color=colour)
    old = seg.cget("command")
    seg.configure(command=lambda v: (tint(v), old(v) if old else None))
    tint()

QUALITY_STEPS = {"Fast": 0, "Balanced": 100, "Best": 250}
TIMELINE = [-1.0, -0.5, 0.0, 0.5, 1.0]
# what the result is: the aged face frame on its own (default), or pasted back into the photo
KEEP_GENDER = {"Auto": "auto", "Woman": "female", "Man": "male", "Off": None}
RESULTS = {"Face frame": ("frame", False), "Whole photo": ("photo", False), "Photo, own hair": ("photo", True)}
IMAGE_TYPES = [("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp *.jfif"), ("All files", "*.*")]


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if ok:
        buf.tofile(path)
    return ok


def age_words(amount):
    if abs(amount) < 0.05:
        return "as you are"
    word = "older" if amount > 0 else "younger"
    return f"{abs(amount):.0%} {word}"


def timeline_strip(results, height=512):
    """The aligned faces side by side, each labelled."""
    tiles = []
    for r in results:
        face = cv2.resize(r.edited, (height, height), interpolation=cv2.INTER_AREA)
        label = "you" if abs(r.amount) < 0.05 else ("younger" if r.amount < 0 else "older") + f" {abs(r.amount):.0%}"
        cv2.rectangle(face, (0, 0), (len(label) * 14 + 16, 32), (0, 0, 0), -1)
        cv2.putText(face, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(face)
    return np.concatenate(tiles, axis=1)



def set_logo(root, here):
    """The Changeling logo as the window's icon (title bar and taskbar). customtkinter puts its own icon in
    about 200 ms after the window opens, so this goes in just after it."""
    ico, png = os.path.join(here, "logo.ico"), os.path.join(here, "logo.png")

    def apply():
        try:
            if os.name == "nt" and os.path.exists(ico):
                root.iconbitmap(ico)
            elif os.path.exists(png):
                root._logo_icon = tk.PhotoImage(file=png)
                root.iconphoto(True, root._logo_icon)
        except tk.TclError:
            pass
    root.after(300, apply)


def logo_image(here, size=56):
    """The logo for the sidebar, or None."""
    png = os.path.join(here, "logo.png")
    if not os.path.exists(png):
        return None
    return ctk.CTkImage(Image.open(png), size=(size, size))

class Window:
    def __init__(self, make_shifter, device=None):
        self.make_shifter = make_shifter
        self.device = device
        self.shifter = None
        self.photo = None            # RGB
        self.photo_path = None
        self.result = None           # what the After side shows (RGB) and what Save writes
        self.closeup = None
        self.job = None
        self.state = {"busy": False, "stage": "", "done": 0, "total": 0, "t0": 0.0, "message": ""}
        self.lock = threading.Lock()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(os.path.join(HERE, "theme.json"))
        self.root = ctk.CTk()
        set_logo(self.root, HERE)
        self.root.title("Changeling Chronos")
        self.root.geometry("1280x780")
        self.root.minsize(980, 620)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build()
        self.root.bind("<Control-o>", lambda e: self.open())
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Return>", lambda e: self.go())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.after(100, self.watch)
        self._run(self._load_models, "Loading the models")

    # ---- the window
    def _build(self):
        stage = ctk.CTkFrame(self.root, corner_radius=12)
        stage.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        stage.grid_columnconfigure((0, 1), weight=1, uniform="half")
        stage.grid_rowconfigure(1, weight=1)
        for col, text in enumerate(("Before", "After")):
            ctk.CTkLabel(stage, text=text.upper(), text_color="gray50",
                         font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=col, pady=(8, 0))
        self.before = tk.Label(stage, bg="#16161a", bd=0, text="Open a photo  (Ctrl+O)", fg="gray60",
                               font=("Segoe UI", 12), cursor="hand2")
        self.before.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.before.bind("<Button-1>", lambda e: self.open())
        self.after_view = tk.Label(stage, bg="#16161a", bd=0, fg="gray60", font=("Segoe UI", 12))
        self.after_view.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.before.bind("<Configure>", lambda e: self.root.after_idle(self._refresh))

        side = ctk.CTkFrame(self.root, width=310, corner_radius=12)
        side.grid(row=0, column=1, sticky="ns", padx=(7, 14), pady=14)
        side.grid_propagate(False)
        logo = logo_image(HERE)
        if logo is not None:
            ctk.CTkLabel(side, image=logo, text="").pack(anchor="w", padx=14, pady=(12, 0))
        ctk.CTkLabel(side, text="CHANGELING", text_color="gray55",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=14, pady=(12, 0))
        ctk.CTkLabel(side, text="Chronos", font=ctk.CTkFont(size=24, weight="bold"), text_color=PINK).pack(anchor="w", padx=14)
        ctk.CTkLabel(side, text=TAGLINE, text_color="gray60", wraplength=280, justify="left").pack(anchor="w", padx=14)

        ctk.CTkButton(side, text="Open photo  (Ctrl+O)", command=self.open).pack(fill="x", padx=14, pady=(16, 4))

        self._section(side, "Age")
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=14)
        ctk.CTkLabel(row, text="younger", text_color="gray60").pack(side="left")
        ctk.CTkLabel(row, text="older", text_color="gray60").pack(side="right")
        self.amount_text = ctk.CTkLabel(row, text="50% older")
        self.amount_text.pack()
        self.amount = ctk.CTkSlider(side, from_=-100, to=100, number_of_steps=40,
                                    command=lambda v: self.amount_text.configure(text=age_words(v / 100)))
        self.amount.set(50)
        self.amount.pack(fill="x", padx=14, pady=(0, 4))
        self.timeline = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(side, text="Timeline: five ages at once", variable=self.timeline).pack(anchor="w", padx=14, pady=6)
        self._section(side, "Keep gender")
        self.keep_gender = ctk.CTkSegmentedButton(side, values=list(KEEP_GENDER))
        self.keep_gender.set("Auto")
        self.keep_gender.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(side, text="Ageing can drift a face towards the other gender; this holds it. Auto reads the "
                     "face and only holds it when sure.", text_color="gray55", wraplength=280,
                     justify="left").pack(anchor="w", padx=14)

        self._section(side, "Result")
        self.result_mode = ctk.CTkSegmentedButton(side, values=list(RESULTS))
        self.result_mode.set("Face frame")
        self.result_mode.pack(fill="x", padx=14, pady=4)

        self._section(side, "Quality")
        self.quality = ctk.CTkSegmentedButton(side, values=list(QUALITY_STEPS))
        self.quality.set("Balanced")
        self.quality.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(side, text="Balanced and Best learn your face first, so the result still looks like you "
                     "(about 1 and 3 minutes). Fast takes seconds.", text_color="gray55",
                     wraplength=280, justify="left").pack(anchor="w", padx=14)

        self.go_btn = ctk.CTkButton(side, fg_color=BLUE, hover_color=BLUE_HOVER, text="Age  (Enter)", height=42,
                                    font=ctk.CTkFont(size=15, weight="bold"), command=self.go)
        self.go_btn.pack(fill="x", padx=14, pady=(18, 6))
        self.bar = ctk.CTkProgressBar(side, mode="determinate")
        self.bar.set(0)
        self.bar.pack(fill="x", padx=14, pady=(4, 2))
        self.status = ctk.CTkLabel(side, text="", text_color="gray65", wraplength=280, justify="left")
        self.status.pack(anchor="w", padx=14)
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(10, 0))
        self.save_btn = ctk.CTkButton(row, text="Save  (Ctrl+S)", command=self.save, state="disabled")
        self.save_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(row, text="Gallery", width=80, command=self.gallery).pack(side="left", padx=(4, 4))
        self.face_btn = ctk.CTkButton(row, text="Close-up", width=90, command=self.show_closeup, state="disabled")
        self.face_btn.pack(side="left")

        foot = ctk.CTkFrame(side, fg_color="transparent")
        foot.pack(side="bottom", fill="x", padx=14, pady=(0, 10))
        links = ctk.CTkFrame(foot, fg_color="transparent")
        links.pack(anchor="w")
        for text, url in (("GitHub", GITHUB), ("Ko-fi", KOFI)):
            l = ctk.CTkLabel(links, text=text, text_color=("#005184", "#34a9e3"), cursor="hand2")
            l.pack(side="left", padx=(0, 14))
            l.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
        ctk.CTkLabel(foot, text="© Cursed Entertainment 2026", text_color="gray45").pack(anchor="w")

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color="gray50",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=14, pady=(14, 1))

    def _show(self, label, image):
        if image is None:
            label.configure(image="")
            label.image = None
            return
        w, h = max(label.winfo_width(), 64), max(label.winfo_height(), 64)
        ih, iw = image.shape[:2]
        k = min(w / iw, h / ih)
        small = cv2.resize(image, (max(int(iw * k), 1), max(int(ih * k), 1)), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(small))
        label.configure(image=photo, text="")
        label.image = photo

    def _refresh(self):
        self._show(self.before, self.photo)
        self._show(self.after_view, self.result)

    # ---- the jobs, on their own thread; the window follows self.state
    def _run(self, work, stage):
        with self.lock:
            self.state.update(busy=True, stage=stage, done=0, total=0, t0=time.monotonic(), message="")
        self.go_btn.configure(state="disabled")
        self.job = threading.Thread(target=self._guard, args=(work,), daemon=True)
        self.job.start()

    def _guard(self, work):
        try:
            message = work()
        except Exception as e:
            message = f"Couldn't: {e}"
        with self.lock:
            self.state.update(busy=False, message=message or "")

    def _step(self, done, total, _loss=None):
        with self.lock:
            self.state.update(stage="Learning your face", done=done, total=total)

    def watch(self):
        with self.lock:
            s = dict(self.state)
        if s["busy"]:
            if s["total"]:
                if self.bar.cget("mode") != "determinate":
                    self.bar.stop()
                    self.bar.configure(mode="determinate")
                frac = s["done"] / s["total"]
                self.bar.set(frac)
                text = f"{s['stage']}  {s['done']} / {s['total']}  {frac:.0%}"
                elapsed = time.monotonic() - s["t0"]
                if s["done"] and elapsed > 1:
                    left = int(elapsed / s["done"] * (s["total"] - s["done"]))
                    text += f"  {left // 60}:{left % 60:02d} left"
            else:
                if self.bar.cget("mode") != "indeterminate":
                    self.bar.configure(mode="indeterminate")
                    self.bar.start()
                text = s["stage"] + " ..."
            self.status.configure(text=text)
        elif self.job is not None and not self.job.is_alive():
            self.job = None
            self.bar.stop()
            self.bar.configure(mode="determinate")
            self.bar.set(1 if self.result is not None else 0)
            self.status.configure(text=s["message"])
            self.go_btn.configure(state="normal" if self.shifter else "disabled")
            self.save_btn.configure(state="normal" if self.result is not None else "disabled")
            self.face_btn.configure(state="normal" if self.closeup is not None else "disabled")
            self._refresh()
        self.root.after(100, self.watch)

    def _load_models(self):
        self.shifter = self.make_shifter(self.device)
        return "Ready: open a photo." if self.photo is None else "Ready: press Age."

    # ---- what the buttons do
    def open(self, path=None):
        path = path or filedialog.askopenfilename(title="A photo with a face", filetypes=IMAGE_TYPES)
        if not path:
            return
        image = read_rgb(path)
        if image is None:
            self.status.configure(text=f"Can't read {os.path.basename(path)}")
            return
        self.photo, self.photo_path, self.result, self.closeup = image, path, None, None
        self.save_btn.configure(state="disabled")
        self.face_btn.configure(state="disabled")
        self._refresh()

    def go(self):
        if self.shifter is None or self.photo is None or self.job is not None:
            if self.photo is None:
                self.open()
            return
        amount = self.amount.get() / 100
        steps = QUALITY_STEPS[self.quality.get()]
        image, timeline = self.photo, self.timeline.get()
        output, keep_hair = RESULTS[self.result_mode.get()]
        keep_gender = KEEP_GENDER[self.keep_gender.get()]

        def work():
            with self.lock:
                self.state["stage"] = "Ageing"
            if timeline:
                results = self.shifter.age(image, 0, steps, self._step, amounts=TIMELINE, keep_hair=keep_hair,
                                           keep_gender=keep_gender)
                self.result = timeline_strip(results)
                self.closeup = np.concatenate([results[0].aligned, results[0].reconstruction], axis=1)
                return "Done: five ages, much younger to much older."
            r = self.shifter.age(image, amount, steps, self._step, keep_hair=keep_hair, output=output,
                                 keep_gender=keep_gender)
            self.result = r.image
            self.closeup = np.concatenate([r.aligned, r.reconstruction, r.edited], axis=1)
            return f"Done: {age_words(amount)}."
        self.result, self.closeup = None, None
        self._run(work, "Reading the face")

    def gallery(self):
        """Everything made with the Changelings (saving still asks where; the gallery lists it too)."""
        show_gallery(self.root, "Chronos")

    def save(self):
        if self.result is None:
            return
        stem, _ = os.path.splitext(os.path.basename(self.photo_path or "photo.png"))
        tag = "timeline" if self.timeline.get() else age_words(self.amount.get() / 100).replace("%", "").replace(" ", "_")
        path = filedialog.asksaveasfilename(title="Save the result", defaultextension=".png",
                                            initialdir=os.path.dirname(self.photo_path or HERE),
                                            initialfile=f"{stem}_{tag}.png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path and write_rgb(path, self.result):
            self.status.configure(text=f"Saved {os.path.basename(path)}")
            remember(path, "Chronos")

    def show_closeup(self):
        if self.closeup is None:
            return
        top = ctk.CTkToplevel(self.root)
        top.title("Close-up: the photo, the reconstruction, the result")
        h, w = self.closeup.shape[:2]
        k = 400 / h
        strip = cv2.resize(self.closeup, (int(w * k), 400), interpolation=cv2.INTER_AREA)
        img = ctk.CTkImage(Image.fromarray(strip), size=(strip.shape[1], 400))
        ctk.CTkLabel(top, image=img, text="").pack(padx=10, pady=10)
        top.after(100, top.lift)

    def run(self, path=None):
        if path:
            self.root.after(300, lambda: self.open(path))
        self.root.mainloop()
