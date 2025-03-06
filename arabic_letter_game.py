from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import os
import collections
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS
import pygame
import tempfile


ARABIC_LETTERS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 
    'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'
]



# Initialize game variables
score = 0
game_speed = 1
letter_size = 60
target_letter_idx = random.randint(0, len(ARABIC_LETTERS) - 1)
new_letter_countdown = 0
letters = []
currently_grabbed_letter = None  # Track the currently grabbed letter
target_area = None
last_announcement_time = 0
announcement_delay = 5
last_target_letters = []
score_display_time = 0  # For showing score animation
score_message = ""  # For showing score messages
bin_columns = []  # Track occupied columns in the bin
game_over = False  # Track if game is over
answers_history = []  # Track all answers for review
scroll_position = 0  # Track scroll position for review screen

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize pygame mixer for audio
pygame.mixer.init()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alternative", action="store_true",
    help="use alternative English letters instead of Arabic")
args = vars(ap.parse_args())

# Function to announce letter
def announce_letter(letter):
    try:
        # Create temporary file for audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts = gTTS(text=letter, lang='ar')
        tts.save(temp_file.name)
        
        # Play the audio
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        
        # Clean up
        temp_file.close()
        os.unlink(temp_file.name)
    except Exception as e:
        print(f"Error announcing letter: {e}")

# Function to draw Arabic text on image using PIL
def draw_arabic_text(img, text, position, font_size=60, color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf"
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
                
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Set up window (normal mode, resizable)
cv2.namedWindow("Arabic Letter Game", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aralizer", 1280, 720)  # Default HD resolution

# Get initial frame to determine dimensions
frame_check = vs.read()
if frame_check is None:
    print("[ERROR] Could not access the camera. Please check camera permissions.")
    print("On macOS, you may need to grant camera access in System Preferences > Security & Privacy > Camera")
    vs.stop()
    exit(1)

# Set initial dimensions
screen_width = 1280
screen_height = 720

def show_intro_screen(display, w, h):
    # fill the screen with a solid color
    display[:] = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
    
    # draw the game title
    title_text = "Arabic Letter Learning Game"
    title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.putText(display, title_text, 
               ((w - title_size[0]) // 2, h//2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # draw the instructions
    instructions = "Drag the correct letter according to the sound to the bin.\nUse your index finger, and if you picked up a wrong letter, point your index finger down to let go."
    y_offset = h//2 - 20
    for line in instructions.split('\n'):
        line_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(display, line, 
                   ((w - line_size[0]) // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 40
    
    # fraw the start prompt
    start_text = "Press SPACEBAR to begin"
    start_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.putText(display, start_text, 
               ((w - start_size[0]) // 2, h//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # intro screen
    cv2.imshow("Arabic Letter Game", display)
    
    # spacebar begins the game
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break

# Show intro screen before starting the game
show_intro_screen(np.zeros((screen_height, screen_width, 3), dtype=np.uint8), screen_width, screen_height)

def get_new_target_letter():
    global last_target_letters
    available_indices = [i for i in range(len(ARABIC_LETTERS)) if i not in last_target_letters]
    if not available_indices:  # If we've used all letters
        last_target_letters = []
        available_indices = list(range(len(ARABIC_LETTERS)))
    new_idx = random.choice(available_indices)
    last_target_letters.append(new_idx)
    if len(last_target_letters) > 5:  # keep track of last 5 letters
        last_target_letters.pop(0)
    return new_idx

def spawn_new_letters(w):
    global letters, game_over
    
    # If score is 50 or more, trigger game over
    if score >= 50:
        game_over = True
        return
        
    # clear existing letters
    letters = []
    
    # Always include the target letter
    positions = []
    # Add target letter at random position
    x = random.randint(100, w - 200)
    positions.append(x)
    letters.append({
        'letter': ARABIC_LETTERS[target_letter_idx] if not args.get("alternative", False) else ARABIC_LETTERS[target_letter_idx],
        'letter_idx': target_letter_idx,
        'x': x,
        'y': 50,
        'grabbed': False,
        'velocity_y': 0,
        'in_bin': False
    })
    
    # 2-3 random distractor letters
    num_distractors = random.randint(2, 3)
    available_indices = [i for i in range(len(ARABIC_LETTERS)) 
                        if i != target_letter_idx and i not in last_target_letters]
    
    for _ in range(num_distractors):
        letter_idx = random.choice(available_indices)
        available_indices.remove(letter_idx)  # Avoid repeating the same letter
        
        # Find a position that doesn't overlap with existing letters
        while True:
            x = random.randint(100, w - 200)
            if all(abs(x - pos) > 100 for pos in positions):  # Ensure 100px gap between letters
                positions.append(x)
                break
        
        letters.append({
            'letter': ARABIC_LETTERS[letter_idx] if not args.get("alternative", False) else ARABIC_LETTERS[letter_idx],
            'letter_idx': letter_idx,
            'x': x,
            'y': 50,
            'grabbed': False,
            'velocity_y': 0,
            'in_bin': False
        })

def get_available_bin_column(target_area):
    global bin_columns
    column_width = 90  # Width of letter box + spacing
    num_columns = 3  # Maximum number of columns in the bin
    bin_width = target_area['x2'] - target_area['x1']
    
    # Initialize columns if empty
    if not bin_columns:
        bin_columns = [False] * num_columns
    
    # Find first available column
    for i in range(num_columns):
        if not bin_columns[i]:
            bin_columns[i] = True
            # Calculate x position for this column
            x = target_area['x1'] + (i * column_width) + (bin_width - num_columns * column_width) // 2
            return i, x
    
    # If all columns are occupied, clear the first one
    bin_columns[0] = True
    x = target_area['x1'] + (bin_width - num_columns * column_width) // 2
    return 0, x

# Main game loop
while True:
    # Read frame
    frame = vs.read()
    
    # Flip the frame horizontally to correct the mirrored display
    frame = cv2.flip(frame, 1)
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate scaling to fit window while maintaining aspect ratio
    scale = min(screen_width/frame_width, screen_height/frame_height)
    target_width = int(frame_width * scale)
    target_height = int(frame_height * scale)
    
    # Resize frame
    frame = cv2.resize(frame, (target_width, target_height))
    
    # Create black canvas
    display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Calculate position to center the frame
    x_offset = (screen_width - target_width) // 2
    y_offset = (screen_height - target_height) // 2
    
    # Place the frame in the center of the display
    display[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = frame
    
    # Get dimensions for further calculations
    (h, w) = screen_height, screen_width
    
    if game_over:
        # Create review screen with gradient background
        display = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            alpha = i / h
            display[i] = [int(40 + 20 * alpha), int(45 + 25 * alpha), int(60 + 30 * alpha)]
            
        # Draw congratulations message
        cv2.putText(display, "Congratulations! Game Complete!", 
                   (w//2 - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Final Score: {score}", 
                   (w//2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display answer history
        y_pos = 150
        cv2.putText(display, "Review of Your Answers:", 
                   (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 40
        
        # Show all answers with scrolling
        for answer in answers_history:
            # Draw answer box with glow effect
            box_color = (0, 100, 0) if answer['correct'] else (100, 0, 0)  # Darker green/red
            for i in range(3, 0, -1):
                alpha = 0.2 * (1 - i/3)  
                cv2.rectangle(display, 
                            (45-i*2, y_pos-30-i*2),
                            (w-45+i*2, y_pos+10+i*2),
                            box_color, -1)
            
            # Draw main box with darker background
            cv2.rectangle(display, (45, y_pos-30), (w-45, y_pos+10), 
                         (30, 30, 30), -1)  # Darker background
            
            # raw target letter with darker background
            target_bg_color = (40, 40, 40)  # Darker background for letters
            cv2.rectangle(display, (150, y_pos-25), (250, y_pos+5), target_bg_color, -1)
            display = draw_arabic_text(display, answer['target'], 
                                     (170, y_pos-30), font_size=40, color=(255, 255, 255))  # Further adjusted y-position
            
            # arrow
            cv2.putText(display, "->", (270, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # selected letter with darker background
            selected_bg_color = (40, 40, 40)  
            cv2.rectangle(display, (320, y_pos-25), (420, y_pos+5), selected_bg_color, -1)
            display = draw_arabic_text(display, answer['selected'],
                                     (340, y_pos-30), font_size=40, color=(255, 255, 255))  # Further adjusted y-position
            
            # Add result indicator with adjusted colors
            result_text = "CORRECT" if answer['correct'] else "WRONG"
            text_color = (100, 255, 100) if answer['correct'] else (255, 100, 100)  # Brighter text
            cv2.putText(display, result_text, (w-200, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            y_pos += 50
        
        # Draw exit instruction
        cv2.putText(display, "Press 'ESC' to exit", 
                   (w//2 - 100, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Arabic Letter Game", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            break
        continue
    
    # gradient background
    background = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        alpha = i / h
        background[i] = [int(40 + 20 * alpha), int(45 + 25 * alpha), int(60 + 30 * alpha)]  # Dark blue gradient
    
    # blend background with camera feed
    alpha = 0.85  # adjust transparency
    display = cv2.addWeighted(display, alpha, background, 1-alpha, 0)
    
    # spawn new letters if none exist
    if len(letters) == 0:
        spawn_new_letters(w)
        announce_letter(ARABIC_LETTERS[target_letter_idx])
        last_announcement_time = time.time()
    
    # Define target area (bin) on the right with modern design
    target_area = {
        'x1': w - 150,
        'y1': h // 2 - 100,
        'x2': w - 50,
        'y2': h // 2 + 100
    }
    
    # Draw modern bin with gradient and glow effect
    # Draw glow
    for i in range(5, 0, -1):
        alpha = 0.2 * (1 - i/5)
        pt1 = (int(target_area['x1'])-i*2, int(target_area['y1'])-i*2)
        pt2 = (int(target_area['x2'])+i*2, int(target_area['y2'])+i*2)
        cv2.rectangle(display, pt1, pt2, (0, 255, 0), 2)
    
    # Main bin
    pt1 = (int(target_area['x1']), int(target_area['y1']))
    pt2 = (int(target_area['x2']), int(target_area['y2']))
    # Draw semi-transparent fill
    overlay = display.copy()
    cv2.rectangle(overlay, pt1, pt2, (0, 100, 0), -1)
    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
    # Draw border
    cv2.rectangle(display, pt1, pt2, (0, 255, 0), 2)

    # Draw modern bin opening
    pts = np.array([
        [target_area['x1'] - 20, target_area['y1']],
        [target_area['x2'] + 20, target_area['y1']],
        [target_area['x2'], target_area['y1'] + 40],
        [target_area['x1'], target_area['y1'] + 40]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # Draw fill
    overlay = display.copy()
    cv2.fillPoly(overlay, [pts], (0, 100, 0))
    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
    # Draw border
    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
    
    # Find hands
    hands, frame = detector.findHands(frame, draw=False)
    current_finger_pos = None
    
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        if fingers[1]:  # Index finger up
            index_finger = hand["lmList"][8]
            current_finger_pos = (index_finger[0], index_finger[1])
            
            # Draw finger position with glow effect
            for i in range(4, 0, -1):
                alpha = 0.25 * (1 - i/4)
                cv2.circle(display, current_finger_pos, 12+i*2, (50, 50, 255), -1, cv2.LINE_AA)
            cv2.circle(display, current_finger_pos, 12, (100, 100, 255), -1, cv2.LINE_AA)
            cv2.circle(display, current_finger_pos, 6, (0, 0, 255), -1)
    
    # Process letters
    for letter_obj in letters[:]:
        if letter_obj['in_bin']:
            # Handle letters in the bin
            letter_obj['velocity_y'] = min(letter_obj['velocity_y'] + 0.5, 8)
            letter_obj['y'] += letter_obj['velocity_y']
            
            # Draw falling letter in bin with glow effect
            if letter_obj['y'] <= h:
                pt1 = (int(letter_obj['x'] - 10), int(letter_obj['y'] - 10))
                pt2 = (int(letter_obj['x'] + 80), int(letter_obj['y'] + 80))
                # Add glow effect
                for i in range(3, 0, -1):
                    alpha = 0.3 * (1 - i/3)
                    glow_color = (100, 255, 100) if letter_obj['letter_idx'] == target_letter_idx else (100, 100, 255)
                    cv2.rectangle(display, 
                                (pt1[0]-i*2, pt1[1]-i*2),
                                (pt2[0]+i*2, pt2[1]+i*2),
                                glow_color, -1)
                # Main letter background
                bg_color = (100, 255, 100) if letter_obj['letter_idx'] == target_letter_idx else (100, 100, 255)
                cv2.rectangle(display, pt1, pt2, bg_color, -1)
                display = draw_arabic_text(display, letter_obj['letter'],
                                         (int(letter_obj['x']), int(letter_obj['y'])),
                                         font_size=letter_size)
            
            # Remove when fully fallen
            if letter_obj['y'] > h:
                if hasattr(letter_obj, 'bin_column'):
                    bin_columns[letter_obj['bin_column']] = False  # Free up the column
                letters.remove(letter_obj)
            
        elif not letter_obj['grabbed']:
            # Apply gravity to falling letters
            letter_obj['velocity_y'] = min(letter_obj['velocity_y'] + 0.1, 3)
            letter_obj['y'] += letter_obj['velocity_y']
            
            # Draw letter background
            pt1 = (int(letter_obj['x'] - 10), int(letter_obj['y'] - 10))
            pt2 = (int(letter_obj['x'] + 80), int(letter_obj['y'] + 80))
            # Use neutral background for all falling letters
            bg_color = (255, 255, 200)
            cv2.rectangle(display, pt1, pt2, bg_color, -1)
            
            # Draw letter
            display = draw_arabic_text(display, letter_obj['letter'],
                                     (int(letter_obj['x']), int(letter_obj['y'])),
                                     font_size=letter_size)
            
            # Check if letter can be grabbed (only if no letter is currently grabbed)
            if current_finger_pos and currently_grabbed_letter is None:
                if (letter_obj['x'] - 10 <= current_finger_pos[0] <= letter_obj['x'] + 80 and
                    letter_obj['y'] - 10 <= current_finger_pos[1] <= letter_obj['y'] + 80):
                    letter_obj['grabbed'] = True
                    letter_obj['velocity_y'] = 0
                    currently_grabbed_letter = letter_obj
            
            # Remove if off screen
            if letter_obj['y'] > h:
                letters.remove(letter_obj)
        else:
            # Move grabbed letter with finger
            if current_finger_pos:
                letter_obj['x'] = current_finger_pos[0] - 40
                letter_obj['y'] = current_finger_pos[1] - 40
                
                # Draw grabbed letter
                pt1 = (int(letter_obj['x'] - 10), int(letter_obj['y'] - 10))
                pt2 = (int(letter_obj['x'] + 80), int(letter_obj['y'] + 80))
                # Highlight the grabbed letter based on whether it's the target
                bg_color = (200, 255, 200) if letter_obj['letter_idx'] == target_letter_idx else (255, 200, 200)
                cv2.rectangle(display, pt1, pt2, bg_color, -1)
                
                display = draw_arabic_text(display, letter_obj['letter'],
                                         (int(letter_obj['x']), int(letter_obj['y'])),
                                         font_size=letter_size)
                
                # Check if letter is in bin area
                if (target_area['x1'] <= letter_obj['x'] <= target_area['x2'] and
                    target_area['y1'] - 50 <= letter_obj['y'] <= target_area['y2']):
                    # Get available column in bin
                    column_idx, bin_x = get_available_bin_column(target_area)
                    letter_obj['bin_column'] = column_idx
                    letter_obj['x'] = bin_x  # Set x position to column position
                    
                    # Mark letter as in bin and check if correct
                    letter_obj['in_bin'] = True
                    letter_obj['grabbed'] = False
                    currently_grabbed_letter = None  # Reset currently grabbed letter
                    if letter_obj['letter_idx'] == target_letter_idx:
                        score += 10
                        score_message = "+10!"
                        score_display_time = time.time()
                        # Record correct answer
                        answers_history.append({
                            'target': ARABIC_LETTERS[target_letter_idx],
                            'selected': letter_obj['letter'],
                            'correct': True
                        })
                        # Immediately get new target letter and announce it
                        target_letter_idx = get_new_target_letter()
                        announce_letter(ARABIC_LETTERS[target_letter_idx])
                        last_announcement_time = time.time()
                        # Spawn new letters immediately
                        letters = [l for l in letters if l['in_bin']]  # keep only falling letters
                        spawn_new_letters(w)
                    else:
                        score = max(0, score - 5)
                        score_message = "-5 (Wrong Letter)"
                        score_display_time = time.time()
                        # Record wrong answer
                        answers_history.append({
                            'target': ARABIC_LETTERS[target_letter_idx],
                            'selected': letter_obj['letter'],
                            'correct': False
                        })
                        # Repeat the target letter sound for wrong answers
                        announce_letter(ARABIC_LETTERS[target_letter_idx])
                        last_announcement_time = time.time()
                        # Spawn new letters immediately after wrong answer too
                        letters = [l for l in letters if l['in_bin']]  # keep only falling letters
                        spawn_new_letters(w)
            else:
                # Release letter if finger is up
                letter_obj['grabbed'] = False
                currently_grabbed_letter = None  # Reset currently grabbed letter
                letter_obj['velocity_y'] = 0
    
    # Display score with modern design
    score_bg = display.copy()
    score_text = f"Score: {score}"
    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(score_bg, (10, 10), (text_size[0] + 30, 45), (0, 0, 0), -1)
    cv2.addWeighted(score_bg, 0.3, display, 0.7, 0, display)
    cv2.putText(display, score_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show score change animation with modern effect
    if time.time() - score_display_time < 1.5:  # Show for 1.5 seconds
        alpha = 1.0 - (time.time() - score_display_time) / 1.5  # Fade out
        font_scale = 1.2 + alpha * 0.3  # Text starts larger and shrinks
        
        # Calculate centered position for score message
        text_size = cv2.getTextSize(score_message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = w//2 - text_size[0]//2
        text_y = h//2
        
        # Draw subtle glow (just one layer)
        glow_color = (0, 0, 255) if "-5" in score_message else (0, 255, 0)
        cv2.putText(display, score_message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.1,
                   glow_color, 3, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(display, score_message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                   (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show frame
    cv2.imshow("Arabic Letter Game", display)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC key
        break

# Cleanup
pygame.mixer.quit()
cv2.destroyAllWindows()
vs.stop()