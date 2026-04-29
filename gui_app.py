import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
"""
Zmienna EXE jest używana aby wstawić własną ścieżkę pythona.
"""
MODEL_PATH = "runs/detect/train2/weights/best.pt"
OUTPUT_DIR = "./inference_output"
EXE = "C:/Users/mateu/Desktop/Systemy_rozpoznawania/.venv/Scripts/python.exe"
class App:
    def __init__(self, root):
        self.root = root
        root.title("Infer Crop – GUI")
        root.geometry("1000x800")

        self.image_path = None

        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)

        self.listbox = tk.Listbox(self.left_frame, width=30, height=30, font=("Arial", 12))
        self.listbox.pack()

        self.btn_select = tk.Button(self.right_frame, text="Wybierz obraz", command=self.select_image, font=("Arial", 14))
        self.btn_select.pack(pady=10)

        self.original_label = tk.Label(self.right_frame, text="(Obraz oryginalny)")
        self.original_label.pack(pady=10)

        self.btn_run = tk.Button(self.right_frame, text="Rozpoznaj znak drogowy", command=self.run_inference, font=("Arial", 14))
        self.btn_run.pack(pady=10)

        self.result_label = tk.Label(self.right_frame, text="(Obraz wynikowy)")
        self.result_label.pack(pady=10)

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.image_path = path
            self.show_image(path, self.original_label, size=(400, 300))

    def run_inference(self):
        if not self.image_path:
            messagebox.showerror("Błąd", "Najpierw wybierz obraz!")
            return

        cmd = [
            EXE,
            "Infer_Crop.py",
            "--model", MODEL_PATH,
            "--image", self.image_path,
            "--out", OUTPUT_DIR
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            messagebox.showerror("Błąd", "Wystąpił problem przy uruchamianiu skryptu.")
            return

        base = os.path.basename(self.image_path)
        name, ext = os.path.splitext(base)

        result_path = os.path.join(OUTPUT_DIR, f"{name}_result{ext}")
        txt_path = os.path.join(OUTPUT_DIR, f"{name}_labels.txt")

        if not os.path.exists(result_path):
            messagebox.showerror("Błąd", f"Nie znaleziono pliku wynikowego:\n{result_path}")
            return

        self.show_image(result_path, self.result_label, size=(400, 300))

        self.listbox.delete(0, tk.END)

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.listbox.insert(tk.END, line.strip())
        else:
            self.listbox.insert(tk.END, "Brak wykrytych znaków")

    def show_image(self, path, label, size=(400, 300)):
        img = Image.open(path)
        img = img.resize(size)
        img = ImageTk.PhotoImage(img)

        label.config(image=img)
        label.image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()