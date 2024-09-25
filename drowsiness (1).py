import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import cvhelpers as cvHelp
import mediapipe as mp
import time
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from threading import Thread
import  pygame
import sys
def get_mediapipe_app(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh
fps = cvHelp.TrackFPS(0.5)
def play_alert_sound(sound):
    pygame.mixer.init()
    pygame.mixer.music.load(sound)
    pygame.mixer.music.play()


def distance(point_1, point_2):
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):

    try:

        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0

    return ear

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):

    left_ear = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR
def get_mar(landmarks , mouth_idx , w, h):
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in mouth_idx:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, w, h)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P1_P0 = distance(coords_points[1], coords_points[0])
        P2_P3 = distance(coords_points[2], coords_points[3])
        P5_P4 = distance(coords_points[5], coords_points[4])
        P6_P7 = distance(coords_points[6], coords_points[7])

        mar = (P2_P3 + P5_P4 + P6_P7 ) / (3.0 * P1_P0)

    except:
        mar = 0.0
    return mar


class MainPage(ttk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.controller = controller

        # Background image
        self.background_image = Image.open("ca.png")
        self.background_image = self.background_image.resize((640,480))
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        background_label = ttk.Label(self, image=self.background_photo)
        background_label.image = self.background_photo
        background_label.place(relwidth=1, relheight=1)

        # Start button
        start_button = ttk.Button(self, text="Start", command=self.start_application)
        Quit_button = ttk.Button(self, text="Quit", command=self.Quit)
        start_button.place(relx=0.5, rely=0.86, anchor="center")
        Quit_button.place(relx=0.5, rely=0.94, anchor="center")
    def start_application(self):
        self.controller.show_frame(Page2)
    def Quit(self) :
        sys.exit()
fps = cvHelp.TrackFPS(0.5)
class Page2(ttk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.controller = controller

        # code for Page2
        self.cv_frame = ttk.Frame(self)  # Create a frame to hold the OpenCV window
        self.cv_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Back button
        back_button = ttk.Button(self, text="Quit", command=self.quit)
        back_button.place(relx=0.45, rely=0)
        self.state= {
            "start": time.perf_counter(),

        }
        # Start OpenCV loop
        self.start_opencv_loop()

    def go_to_page1(self):
        self.controller.show_frame(MainPage)

    def start_opencv_loop(self):
        # Function to run OpenCV loop
        fps = cvHelp.TrackFPS(.05)
        gui = cvHelp.cvGUI(self.cv_frame, 640, 480)  # Pass cv_frame instead of root
        gui.camStart()
        face = mp.solutions.face_mesh.FaceMesh()
        face = get_mediapipe_app()
        def myloop():

            ignore, frame = gui.cam.read()
            chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
            chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
            mouth_idxs = [78, 308, 82, 84,13 , 17 , 312 ,314]
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face.process(imgRGB)
            imgH, imgW, _ = frame.shape
            if results:
                for id, lm in enumerate(results.multi_face_landmarks[0].landmark):
                        ih, iw, ic = frame.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        if id in chosen_left_eye_idxs or id in chosen_right_eye_idxs:
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
                        elif id in mouth_idxs:
                            cv2.circle(frame , (x, y), 1, (233, 255, 0), 1)
                ear = calculate_avg_ear(results.multi_face_landmarks[0].landmark,
                                               chosen_left_eye_idxs,
                                               chosen_right_eye_idxs,
                                               imgH,
                                               imgW
                                               )
                mar = get_mar(results.multi_face_landmarks[0].landmark,
                                               mouth_idxs,
                                               imgH,
                                               imgW
                                               )
                end = time.perf_counter()
                if ear <0.25 and mar < 1.2:
                    if end - self.state['start'] > 2 and (end - self.state['start'] - 0.5) % 2 > 0 and (end - self.state['start'] - 0.5) % 2 <0.5 :
                        alert_thread = Thread(target=play_alert_sound("wake_up.mp3"))
                        alert_thread.start()
                elif ear < 0.25 and mar > 1.2:
                    if end - self.state['start'] > 2 and (end - self.state['start'] - 0.5) % 2.5 > 0 and (end - self.state['start'] - 0.5) % 2.5 <0.5 :
                        alert_thread = Thread(target=play_alert_sound("focus.mp3"))
                        alert_thread.start()
                else :
                    self.state['start'] = time.perf_counter()

                cv2.putText(frame,
                            f"MAR: {round(mar, 2)}", (30, 80),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 250), 2
                            )
                cv2.putText(frame,
                            f"EAR: {round(ear, 2)}", (30, 110),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 250), 2
                            )
                cv2.putText(frame, str(int(fps.getFPS())).rjust(3) + str(' FPS'), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1,
                            (255, 0, 0),
                            3)
            gui.displayImg(frame)
            self.controller.after(10, myloop)
        myloop()
class MainApplication(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Driver Sleep Alert")
        self.geometry("640x480")
        self.resizable(False, False)

        # Center the window on the screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_coordinate = (screen_width - 640) // 2
        y_coordinate = (screen_height - 480) // 2
        self.geometry(f"+{x_coordinate}+{y_coordinate}")

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MainPage, Page2):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
