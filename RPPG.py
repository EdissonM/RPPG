# #!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import spo
import heart


def main():
    graph_size = (600, 450)
    sg.theme('DarkGrey6')
    heart_rate = '-MLINE-' + sg.WRITE_ONLY_KEY
    spo_rate = '-MLINE-' + sg.WRITE_ONLY_KEY
    layout = [[sg.Text("Click Roi Area", size=(40, 1), key=spo_rate),
               sg.Text("", size=(40, 1), key=heart_rate)],
               [sg.ProgressBar(1, orientation='h', size=(20, 20), key='progress')],
              [sg.Graph(graph_size, (0, 450), (600, 0), key='-GRAPH-', enable_events=True, drag_submits=True)], ]

    window = sg.Window('RPPG', layout)
    progress_bar = window['progress']
    graph_elem = window['-GRAPH-']
    a_id = None
    cap = cv2.VideoCapture(0)
    new_roi = False
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    y1, y2, x1, x2 = 0, 0, 0, 0
    heart_process = heart.Process()
    spo_process = spo.Process()
    index = 0
    select_roi = False
    while True:
        event, values = window.read(timeout=0)
        if event in ('Exit', None):
            break
        ret, frame = cap.read()
        img_bytes = cv2.imencode('.png', frame)[1].tobytes()
        if a_id:
            graph_elem.delete_figure(a_id)
        a_id = graph_elem.draw_image(data=img_bytes, location=(0, 0))
        graph_elem.TKCanvas.tag_lower(a_id)
        if event == "-GRAPH-":
            graph_elem.Erase()
            x1 = values['-GRAPH-'][0] - 10 if values['-GRAPH-'][0] - 10 > 0 else 10
            x2 = values['-GRAPH-'][0] + 10 if values['-GRAPH-'][0] + 10 < width else int(width)
            y1 = values['-GRAPH-'][1] - 10 if values['-GRAPH-'][1] - 10 > 0 else 10
            y2 = values['-GRAPH-'][1] + 10 if values['-GRAPH-'][1] + 10 < height else int(height)
            graph_elem.DrawRectangle((x1, y1), (x2, y2), line_color='red')
            index = 0
            heart_process.reset()
            window[heart_rate].update("Recording")
            select_roi = True
        if select_roi:
            progress_bar.update_bar(index, 500)
            cropped = frame[y1:y2, x1:x2]
            heart_process.update(cropped)
            spo_process.update(cropped)
        if heart_process.get_bmp() != 0:
            window[heart_rate].update("Heart rate Estimation {:.2f}".format(heart_process.get_bmp()))
        if spo_process.get_spo() != 0:
            window[spo_rate].update("SpO2 Estimation {:.2f}".format(spo_process.get_spo()))
        index += 1

    window.close()


if __name__ == "__main__":
    main()
