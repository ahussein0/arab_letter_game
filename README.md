# Arabic Letter Learning Game

This game is designed to help users learn Arabic letters through an interactive and engaging experience. The game uses a webcam to track hand movements, allowing players to drag and drop letters into a bin based on audio prompts.

## How to Run the Game

1. **Ensure you have the necessary dependencies installed**:
   - Python 3
   - OpenCV
   - Pygame
   - gTTS
   - imutils
   - cvzone

2. **Launch the game**:
   ```bash
   python3 arabic_letter_game.py
   ```

3. **Gameplay Instructions**:
   - The game will start with an intro screen displaying the title and instructions.
   - Press the `SPACEBAR` to begin the game.
   - Use your index finger to drag the correct letter to the bin based on the sound prompt.
   - If you pick up the wrong letter, point your index finger down to release it.

4. **Exit the Game**:
   - Press the `ESC` key to exit the game at any time.

## Features

- Interactive gameplay using hand tracking.
- Audio prompts for letter recognition.
- Real-time feedback on correct and incorrect answers.
- Review screen at the end of the game to see your performance.

## Notes

- Ensure your webcam is properly connected and accessible by your system.
- The game is best played in a well-lit environment for optimal hand tracking.

## Customization

You can adjust the game parameters in the code:
- `game_speed`: Controls how fast letters fall
- `letter_size`: Controls the size of the letters
- `ARABIC_LETTERS`: You can modify this list to include different Arabic letters or characters

Sources:
caffemodel: https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
deploy.prototxt: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt

Project Link: [https://github.com/ahussein0/face_detector](https://https://github.com/ahussein0/arabic_letter_game)
