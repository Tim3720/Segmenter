import tkinter as tk
import os
import threading
import time

from tkinter.filedialog import askdirectory

from mean_segmenter import Segmenter


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()

        self.img_dir = "C:/Users/timka/Documents/Arbeit/Testprofil-M181-CTD-035-JPG"
        self.save_dir = "Results/Stack"
        self.info = {
            "n_files": 0,
            "object_counter": 0,
            "total_time": 0,
            "current_file": 0,
            "segmented_files": 0,
            "avg_time": 0,
            "time": 0,
        }

        self.segmenter = None

        # segmenter settings:
        self.stack_size = tk.IntVar(value=100)
        self.n_threads = tk.IntVar(value=8)
        self.min_area = tk.IntVar(value=400)
        self.threshold = tk.IntVar(value=0)
        self.min_var = tk.IntVar(value=30)
        self.save_full_img = tk.BooleanVar(value=False)

        self.btn_width = 20

        self.settings()
        self.btns()
        self.displays()

        self.configuring = False

    def get_img_dir(self):
        self.img_dir = askdirectory(initialdir="Documents")

    def get_save_dir(self):
        self.save_dir = askdirectory(initialdir="Documents")

    def ask_info(self, label):
        label_text = "".join(
            [name + ": " + str(self.info[name]) + "\n" for name in self.info]
        )
        label.config(text=label_text)
        label.after(100, self.ask_info, label)

    def start_segmentation(self):
        for f in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir, f))

        self.segmenter = Segmenter(
            stack_size=self.stack_size.get(),
            n_threads=self.n_threads.get(),
            threshold=self.threshold.get(),
            min_area=self.min_area.get(),
            min_var=self.min_var.get(),
            queue_max_size=10,
            save_full_img=self.save_full_img.get(),
            master=self,
        )

        threading.Thread(
            target=self.segmenter.segment, args=(self.img_dir, self.save_dir)
        ).start()

    def main(self):
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        while self.configuring:
            print("Waiting...")
            time.sleep(0.1)
        # if self.segmenter:
        #     self.segmenter.running = False
        self.running = False
        # print("s", self.segmenter.running)
        print(list(threading.enumerate()))
        time.sleep(0.5)
        print(list(threading.enumerate()))
        while threading.active_count() > 1:
            time.sleep(0.1)
        self.root.destroy()

    def settings(self):
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(side="left", padx=10)
        # stack_size
        tk.Scale(
            settings_frame,
            from_=10,
            to=200,
            resolution=10,
            label="Stack size:",
            orient="horizontal",
            variable=self.stack_size,
        ).grid(row=0, column=0)

        # n_threads
        tk.Scale(
            settings_frame,
            from_=1,
            to=16,
            resolution=1,
            label="Threads:",
            orient="horizontal",
            variable=self.n_threads,
        ).grid(row=0, column=1)

        # min_area
        tk.Scale(
            settings_frame,
            from_=0,
            to=1000,
            resolution=50,
            label="Minimal area:",
            orient="horizontal",
            variable=self.min_area,
        ).grid(row=0, column=2)

        # threshold
        tk.Scale(
            settings_frame,
            from_=-1,
            to=255,
            resolution=1,
            label="Threshold:",
            orient="horizontal",
            variable=self.threshold,
        ).grid(row=1, column=0)

        # min_var
        tk.Scale(
            settings_frame,
            from_=0,
            to=200,
            resolution=10,
            label="Minmimal variance:",
            orient="horizontal",
            variable=self.min_var,
        ).grid(row=1, column=1)

        # save_full_imgs
        tk.Checkbutton(
            settings_frame, variable=self.save_full_img, text="Save full imgs"
        ).grid(row=1, column=2)

    def btns(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side="left")
        # load img dir
        tk.Button(
            btn_frame,
            text="Choose image directory",
            command=self.get_img_dir,
            width=self.btn_width,
        ).pack()
        # load save_dir
        tk.Button(
            btn_frame,
            text="Choose save directory",
            command=self.get_save_dir,
            width=self.btn_width,
        ).pack()
        # start segmentation
        tk.Button(
            btn_frame,
            text="Start segmentation",
            command=self.start_segmentation,
            width=self.btn_width,
        ).pack()

    def displays(self):
        disp_frame = tk.Frame(self.root)
        disp_frame.pack(side="left", padx=10)
        label_text = "".join(
            [name + ": " + str(self.info[name]) + "\n" for name in self.info]
        )
        label = tk.Label(disp_frame, text=label_text)
        label.pack()
        label.after(100, self.ask_info, label)


if __name__ == "__main__":
    a = App()
    a.main()
