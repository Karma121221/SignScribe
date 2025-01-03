import os
import cv2

cap = cv2.VideoCapture(0)
directory = 'Image/'

while True:
    _, frame = cap.read()
    count = {
        'a': len(os.listdir(directory + "A")),
        'b': len(os.listdir(directory + "B")),
        'c': len(os.listdir(directory + "C")),
        'd': len(os.listdir(directory + "D")),
        'e': len(os.listdir(directory + "E")),
        'f': len(os.listdir(directory + "F")),
        'g': len(os.listdir(directory + "G")),
        'h': len(os.listdir(directory + "H")),
        'i': len(os.listdir(directory + "I")),
        'j': len(os.listdir(directory + "J")),
        'k': len(os.listdir(directory + "K")),
        'l': len(os.listdir(directory + "L")),
        'm': len(os.listdir(directory + "M")),
        'n': len(os.listdir(directory + "N")),
        'o': len(os.listdir(directory + "O")),
        'p': len(os.listdir(directory + "P")),
        'q': len(os.listdir(directory + "Q")),
        'r': len(os.listdir(directory + "R")),
        's': len(os.listdir(directory + "S")),
        't': len(os.listdir(directory + "T")),
        'u': len(os.listdir(directory + "U")),
        'v': len(os.listdir(directory + "V")),
        'w': len(os.listdir(directory + "W")),
        'x': len(os.listdir(directory + "X")),
        'y': len(os.listdir(directory + "Y")),
        'z': len(os.listdir(directory + "Z"))
    }

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 600), (255, 255, 255), 2)
    cv2.imshow("Capture Images", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])
    frame = frame[40:400, 0:300]
    interrupt = cv2.waitKey(10)

    # Uncomment this section to display counts on the frame
    for letter, letter_count in count.items():
        cv2.putText(frame, f"{letter}: {letter_count}", (10, 100 + ord(letter) - ord('a') * 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        if interrupt & 0xFF == ord(letter):
            cv2.imwrite(f"{directory}{letter.upper()}/{count[letter]}.png", frame)

    if interrupt & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()