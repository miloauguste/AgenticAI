import pygame
import random
import sys
import numpy as np
import pyttsx3
import speech_recognition as sr
import json
import os
import threading
import math

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Constants
FPS = 60

# Get scrUpdaeen dimensions for fullscreen
pygame.init()
info = pygame.display.Info()plub
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h

# Colors
BLUE = (64, 164, 223)
DARK_BLUE = (41, 128, 185)
GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)
RED = (231, 76, 60)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
OCEAN_BLUE = (20, 105, 180)
OCEAN_DARK = (15, 85, 145)
OCEAN_LIGHT = (100, 180, 255)
PURPLE = (128, 0, 128)
LIGHT_GREEN = (144, 238, 144)

class MathProblemGenerator:
    """Generates grade-appropriate math problems with difficulty scaling"""

    def __init__(self, level=1):
        self.level = level

    def set_level(self, level):
        """Update difficulty level"""
        self.level = level

    def get_max_sum(self):
        """Get maximum sum based on level (10 at level 1, up to 20 at level 10+)"""
        return min(10 + (self.level - 1), 20)

    def generate_addition(self):
        """Generate an addition problem appropriate for first grade"""
        max_sum = self.get_max_sum()
        # Ensure both numbers are single digit for early levels
        if self.level <= 3:
            a = random.randint(1, min(9, max_sum - 1))
            b = random.randint(0, min(9, max_sum - a))
        else:
            a = random.randint(1, max_sum - 1)
            b = random.randint(0, max_sum - a)
        answer = a + b
        problem_text = f"{a} + {b} = ?"
        return {'text': problem_text, 'answer': answer, 'type': 'addition', 'a': a, 'b': b}

    def generate_subtraction(self):
        """Generate a subtraction problem with no negative results"""
        max_minuend = self.get_max_sum()
        # Minuend is the larger number, subtrahend is smaller
        if self.level <= 3:
            a = random.randint(1, min(10, max_minuend))
        else:
            a = random.randint(1, max_minuend)
        b = random.randint(0, a)  # Ensure no negative results
        answer = a - b
        problem_text = f"{a} - {b} = ?"
        return {'text': problem_text, 'answer': answer, 'type': 'subtraction', 'a': a, 'b': b}

    def generate_comparison(self):
        """Generate a comparison problem (bigger/smaller)"""
        max_num = self.get_max_sum()
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        # Ensure they're different
        while b == a:
            b = random.randint(1, max_num)

        # Randomly decide if asking for bigger or smaller
        ask_bigger = random.choice([True, False])
        if ask_bigger:
            answer = max(a, b)
            problem_text = f"Which is BIGGER: {a} or {b}?"
        else:
            answer = min(a, b)
            problem_text = f"Which is SMALLER: {a} or {b}?"

        return {
            'text': problem_text,
            'answer': answer,
            'type': 'comparison',
            'a': a,
            'b': b,
            'ask_bigger': ask_bigger,
            'options': [a, b]
        }

    def generate_random_problem(self, operation="addition"):
        """Generate a problem based on selected operation focus

        Args:
            operation: "addition", "subtraction", or "both"
        """
        if operation == "addition":
            return self.generate_addition()
        elif operation == "subtraction":
            return self.generate_subtraction()
        else:  # "both"
            problem_type = random.choice(['addition', 'subtraction'])
            if problem_type == 'addition':
                return self.generate_addition()
            else:
                return self.generate_subtraction()

    def generate_shop_problem(self, current_shells, cost):
        """Generate a problem for shop purchases: how many more shells needed?"""
        if current_shells >= cost:
            # Can afford - simple subtraction for change
            change = current_shells - cost
            problem_text = f"You have {current_shells} shells. It costs {cost}. How much change?"
            return {'text': problem_text, 'answer': change, 'type': 'shop_change'}
        else:
            # Can't afford - how many more needed
            needed = cost - current_shells
            problem_text = f"You have {current_shells} shells. It costs {cost}. How many more do you need?"
            return {'text': problem_text, 'answer': needed, 'type': 'shop_needed'}


class MathRiddleOverlay:
    """Overlay for displaying math riddles when caught by shark"""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.active = False
        self.current_problem = None
        self.user_input = ""
        self.timer = 15 * 60  # 15 seconds at 60 FPS
        self.result_message = ""
        self.result_timer = 0
        self.lives = 3
        self.math_generator = MathProblemGenerator()

        # Overlay dimensions
        self.overlay_width = 500
        self.overlay_height = 350
        self.overlay_x = (screen_width - self.overlay_width) // 2
        self.overlay_y = (screen_height - self.overlay_height) // 2

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.problem_font = pygame.font.Font(None, 64)
        self.button_font = pygame.font.Font(None, 36)
        self.info_font = pygame.font.Font(None, 32)

        # Button layout (0-9)
        self.buttons = []
        self.create_buttons()

    def create_buttons(self):
        """Create number buttons 0-9"""
        self.buttons = []
        button_size = 50
        gap = 10
        start_x = self.overlay_x + (self.overlay_width - (5 * button_size + 4 * gap)) // 2
        start_y = self.overlay_y + 180

        # First row: 1-5
        for i in range(5):
            x = start_x + i * (button_size + gap)
            y = start_y
            self.buttons.append({'num': i + 1, 'rect': pygame.Rect(x, y, button_size, button_size)})

        # Second row: 6-9, 0
        for i in range(5):
            x = start_x + i * (button_size + gap)
            y = start_y + button_size + gap
            num = (i + 6) if i < 4 else 0
            self.buttons.append({'num': num, 'rect': pygame.Rect(x, y, button_size, button_size)})

    def resize(self, screen_width, screen_height):
        """Handle screen resize"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.overlay_x = (screen_width - self.overlay_width) // 2
        self.overlay_y = (screen_height - self.overlay_height) // 2
        self.create_buttons()

    def start_riddle(self, level, lives, operation="addition", timer_enabled=True):
        """Start a new math riddle"""
        self.active = True
        self.math_generator.set_level(level)
        self.current_problem = self.math_generator.generate_random_problem(operation)
        self.user_input = ""
        self.timer = 15 * 60  # 15 seconds
        self.timer_enabled = timer_enabled
        self.result_message = ""
        self.result_timer = 0
        self.lives = lives

    def handle_event(self, event):
        """Handle input events, returns (solved, correct, input_value)"""
        if not self.active or self.result_timer > 0:
            return None, None, None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                return self.check_answer()
            elif event.key == pygame.K_BACKSPACE:
                self.user_input = self.user_input[:-1]
            elif event.unicode.isdigit() and len(self.user_input) < 3:
                self.user_input += event.unicode

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                for button in self.buttons:
                    if button['rect'].collidepoint(event.pos):
                        if len(self.user_input) < 3:
                            self.user_input += str(button['num'])
                        return None, None, None

        return None, None, None

    def check_answer(self):
        """Check if the answer is correct"""
        if not self.user_input:
            return None, None, None

        try:
            user_answer = int(self.user_input)
            correct = user_answer == self.current_problem['answer']

            if correct:
                self.result_message = "CORRECT! You escaped!"
                self.result_timer = 90  # 1.5 seconds
            else:
                self.result_message = f"Wrong! The answer was {self.current_problem['answer']}"
                self.result_timer = 90

            return True, correct, user_answer
        except ValueError:
            return None, None, None

    def update(self):
        """Update timer and result display"""
        if not self.active:
            return None

        if self.result_timer > 0:
            self.result_timer -= 1
            if self.result_timer <= 0:
                self.active = False
                return 'done'
            return None

        # Only count down timer if enabled
        if self.timer_enabled:
            self.timer -= 1
            if self.timer <= 0:
                # Time's up - count as wrong answer
                self.result_message = f"Time's up! The answer was {self.current_problem['answer']}"
                self.result_timer = 90
                return 'timeout'

        return None

    def draw(self, surface):
        """Draw the riddle overlay"""
        if not self.active:
            return

        # Semi-transparent background
        overlay_bg = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay_bg.fill((0, 0, 0, 180))
        surface.blit(overlay_bg, (0, 0))

        # Main overlay box
        box_color = (40, 80, 120)
        border_color = (100, 150, 200)
        pygame.draw.rect(surface, box_color,
                        (self.overlay_x, self.overlay_y, self.overlay_width, self.overlay_height),
                        border_radius=15)
        pygame.draw.rect(surface, border_color,
                        (self.overlay_x, self.overlay_y, self.overlay_width, self.overlay_height),
                        3, border_radius=15)

        # Title
        title_text = "SHARK RIDDLE!"
        title_surface = self.title_font.render(title_text, True, (255, 200, 50))
        title_rect = title_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                     self.overlay_y + 40))
        surface.blit(title_surface, title_rect)

        # Problem text
        if self.result_timer > 0:
            # Show result message
            result_color = LIGHT_GREEN if "CORRECT" in self.result_message else RED
            result_surface = self.info_font.render(self.result_message, True, result_color)
            result_rect = result_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                           self.overlay_y + 100))
            surface.blit(result_surface, result_rect)
        else:
            # Show problem
            problem_surface = self.problem_font.render(self.current_problem['text'], True, WHITE)
            problem_rect = problem_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                             self.overlay_y + 100))
            surface.blit(problem_surface, problem_rect)

            # User input display
            input_display = self.user_input if self.user_input else "_"
            input_surface = self.problem_font.render(input_display, True, YELLOW)
            input_rect = input_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                         self.overlay_y + 150))
            surface.blit(input_surface, input_rect)

        # Draw number buttons
        for button in self.buttons:
            # Button background
            pygame.draw.rect(surface, (60, 100, 140), button['rect'], border_radius=8)
            pygame.draw.rect(surface, (100, 150, 200), button['rect'], 2, border_radius=8)

            # Button text
            btn_text = self.button_font.render(str(button['num']), True, WHITE)
            btn_rect = btn_text.get_rect(center=button['rect'].center)
            surface.blit(btn_text, btn_rect)

        # Lives display
        lives_y = self.overlay_y + self.overlay_height - 50
        lives_text = "Lives: " + "❤️" * self.lives + "🖤" * (3 - self.lives)
        # Fallback for systems without emoji support
        lives_text_fallback = f"Lives: {self.lives}/3"
        lives_surface = self.info_font.render(lives_text_fallback, True, RED)
        surface.blit(lives_surface, (self.overlay_x + 30, lives_y))

        # Timer display
        if self.timer_enabled:
            time_left = max(0, self.timer // 60)
            timer_color = RED if time_left <= 5 else WHITE
            timer_text = f"Time: {time_left}s"
        else:
            timer_color = LIGHT_GREEN
            timer_text = "No Timer"
        timer_surface = self.info_font.render(timer_text, True, timer_color)
        surface.blit(timer_surface, (self.overlay_x + self.overlay_width - 120, lives_y))

        # Instructions
        if self.result_timer <= 0:
            instr_text = "Type answer and press ENTER"
            instr_surface = pygame.font.Font(None, 24).render(instr_text, True, (180, 180, 180))
            instr_rect = instr_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                         self.overlay_y + self.overlay_height - 20))
            surface.blit(instr_surface, instr_rect)


class AudioManager:
    def __init__(self):
        # Initialize text-to-speech only
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level
    
    def speak(self, text):
        """Convert text to speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

class UserManager:
    def __init__(self):
        self.users_file = "users.json"
        self.settings_file = "game_settings.json"
        self.current_user = None
        self.user_data = self.load_users()
        self.settings = self.load_settings()
    
    def load_users(self):
        """Load user data from file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        """Save user data to file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.user_data, f, indent=2)
        except Exception as e:
            print(f"Error saving user data: {e}")
    
    def register_user(self, username):
        """Register a new user or load existing user"""
        self.current_user = username

        if username not in self.user_data:
            self.user_data[username] = {
                "games_played": 0,
                "highest_level": 1,
                "total_score": 0,
                "shells": 0,
                "math_stats": {
                    "problems_solved": 0,
                    "correct_answers": 0,
                    "riddles_escaped": 0
                }
            }
            print(f"New user {username} registered!")
        else:
            # Ensure existing users have the new fields
            user = self.user_data[username]
            if "shells" not in user:
                user["shells"] = 0
            if "math_stats" not in user:
                user["math_stats"] = {
                    "problems_solved": 0,
                    "correct_answers": 0,
                    "riddles_escaped": 0
                }
            print(f"Welcome back, {username}!")

        # Save as last user
        self.set_last_user(username)
        self.save_users()
    
    def update_stats(self, level_reached):
        """Update user statistics"""
        if self.current_user:
            user_stats = self.user_data[self.current_user]
            user_stats["games_played"] += 1
            user_stats["highest_level"] = max(user_stats["highest_level"], level_reached)
            user_stats["total_score"] += level_reached * 100
            self.save_users()
    
    def get_user_stats(self):
        """Get current user statistics"""
        if self.current_user and self.current_user in self.user_data:
            return self.user_data[self.current_user]
        return None
    
    def load_settings(self):
        """Load game settings from file"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except:
                return {"last_user": None, "fullscreen": False}
        return {"last_user": None, "fullscreen": False}
    
    def save_settings(self):
        """Save game settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def get_last_user(self):
        """Get the last user who played"""
        return self.settings.get("last_user", None)
    
    def set_last_user(self, username):
        """Set the last user who played"""
        self.settings["last_user"] = username
        self.save_settings()
    
    def get_preferred_fullscreen(self):
        """Get user's fullscreen preference"""
        return self.settings.get("fullscreen", False)
    
    def set_preferred_fullscreen(self, fullscreen):
        """Set user's fullscreen preference"""
        self.settings["fullscreen"] = fullscreen
        self.save_settings()
    
    def get_music_enabled(self):
        """Get music enabled preference"""
        return self.settings.get("music_enabled", True)
    
    def set_music_enabled(self, enabled):
        """Set music enabled preference"""
        self.settings["music_enabled"] = enabled
        self.save_settings()
    
    def get_music_volume(self):
        """Get music volume preference"""
        return self.settings.get("music_volume", 0.7)
    
    def set_music_volume(self, volume):
        """Set music volume preference"""
        self.settings["music_volume"] = max(0.0, min(1.0, volume))
        self.save_settings()
    
    def get_music_folder(self):
        """Get custom music folder path"""
        return self.settings.get("music_folder", "music")
    
    def set_music_folder(self, folder_path):
        """Set custom music folder path"""
        self.settings["music_folder"] = folder_path
        self.save_settings()
    
    def get_max_playtime(self):
        """Get maximum playtime in minutes"""
        return self.settings.get("max_playtime_minutes", 30)  # Default 30 minutes
    
    def set_max_playtime(self, minutes):
        """Set maximum playtime in minutes"""
        self.settings["max_playtime_minutes"] = max(1, minutes)  # Minimum 1 minute
        self.save_settings()
    
    def get_session_start_time(self):
        """Get current session start time"""
        return self.settings.get("session_start_time", None)
    
    def set_session_start_time(self, timestamp):
        """Set current session start time"""
        self.settings["session_start_time"] = timestamp
        self.save_settings()
    
    def get_total_playtime_today(self):
        """Get total playtime for today in minutes"""
        import datetime
        today = datetime.date.today().isoformat()
        daily_data = self.settings.get("daily_playtime", {})
        return daily_data.get(today, 0)
    
    def add_playtime_today(self, minutes):
        """Add playtime to today's total"""
        import datetime
        today = datetime.date.today().isoformat()
        daily_data = self.settings.get("daily_playtime", {})
        daily_data[today] = daily_data.get(today, 0) + minutes
        self.settings["daily_playtime"] = daily_data
        self.save_settings()
    
    def reset_session(self):
        """Reset current session"""
        self.settings["session_start_time"] = None
        self.save_settings()
    
    def get_parental_pin(self):
        """Get parental control PIN"""
        return self.settings.get("parental_pin", "1234")  # Default PIN
    
    def set_parental_pin(self, new_pin):
        """Set new parental control PIN"""
        self.settings["parental_pin"] = new_pin
        self.save_settings()
    
    def verify_parental_access(self, entered_pin):
        """Verify parental access with PIN"""
        return entered_pin == self.get_parental_pin()
    
    def get_parental_override_active(self):
        """Check if parental override is currently active"""
        override_data = self.settings.get("parental_override", {})
        import time
        current_time = time.time()
        
        # Check if override is active and not expired
        if override_data.get("active", False):
            expiry_time = override_data.get("expiry_time", 0)
            if current_time < expiry_time:
                return True
            else:
                # Override expired, deactivate it
                self.deactivate_parental_override()
        
        return False
    
    def activate_parental_override(self, duration_hours=1):
        """Activate parental override for specified duration"""
        import time
        expiry_time = time.time() + (duration_hours * 3600)
        
        self.settings["parental_override"] = {
            "active": True,
            "expiry_time": expiry_time,
            "duration_hours": duration_hours
        }
        self.save_settings()
        print(f"Parental override activated for {duration_hours} hour(s)")
    
    def deactivate_parental_override(self):
        """Deactivate parental override"""
        self.settings["parental_override"] = {
            "active": False,
            "expiry_time": 0,
            "duration_hours": 0
        }
        self.save_settings()
        print("Parental override deactivated")
    
    def reset_daily_playtime(self):
        """Reset today's playtime (parental function)"""
        import datetime
        today = datetime.date.today().isoformat()
        daily_data = self.settings.get("daily_playtime", {})
        old_time = daily_data.get(today, 0)
        daily_data[today] = 0
        self.settings["daily_playtime"] = daily_data
        self.save_settings()
        print(f"Daily playtime reset from {old_time:.1f} minutes to 0 minutes")

    # Shell currency methods
    def get_user_shells(self):
        """Get current user's shell count"""
        if self.current_user and self.current_user in self.user_data:
            return self.user_data[self.current_user].get("shells", 0)
        return 0

    def add_shells(self, amount):
        """Add shells to current user"""
        if self.current_user and self.current_user in self.user_data:
            current = self.user_data[self.current_user].get("shells", 0)
            self.user_data[self.current_user]["shells"] = current + amount
            self.save_users()
            return True
        return False

    def spend_shells(self, amount):
        """Spend shells (returns True if successful, False if not enough)"""
        if self.current_user and self.current_user in self.user_data:
            current = self.user_data[self.current_user].get("shells", 0)
            if current >= amount:
                self.user_data[self.current_user]["shells"] = current - amount
                self.save_users()
                return True
        return False

    # Math stats methods
    def get_math_stats(self):
        """Get current user's math statistics"""
        if self.current_user and self.current_user in self.user_data:
            return self.user_data[self.current_user].get("math_stats", {
                "problems_solved": 0,
                "correct_answers": 0,
                "riddles_escaped": 0
            })
        return {"problems_solved": 0, "correct_answers": 0, "riddles_escaped": 0}

    def record_math_problem(self, correct):
        """Record a math problem attempt"""
        if self.current_user and self.current_user in self.user_data:
            stats = self.user_data[self.current_user].get("math_stats", {
                "problems_solved": 0,
                "correct_answers": 0,
                "riddles_escaped": 0
            })
            stats["problems_solved"] = stats.get("problems_solved", 0) + 1
            if correct:
                stats["correct_answers"] = stats.get("correct_answers", 0) + 1
            self.user_data[self.current_user]["math_stats"] = stats
            self.save_users()

    def record_riddle_escape(self):
        """Record a successful riddle escape"""
        if self.current_user and self.current_user in self.user_data:
            stats = self.user_data[self.current_user].get("math_stats", {
                "problems_solved": 0,
                "correct_answers": 0,
                "riddles_escaped": 0
            })
            stats["riddles_escaped"] = stats.get("riddles_escaped", 0) + 1
            self.user_data[self.current_user]["math_stats"] = stats
            self.save_users()

    # Math features settings
    def get_math_features(self):
        """Get math features settings"""
        return self.settings.get("math_features", {
            "shark_riddles_enabled": True,
            "math_gates_enabled": True,
            "shells_enabled": True,
            "comparison_zones_enabled": True
        })

    def set_math_feature(self, feature, enabled):
        """Set a specific math feature on/off"""
        if "math_features" not in self.settings:
            self.settings["math_features"] = {
                "shark_riddles_enabled": True,
                "math_gates_enabled": True,
                "shells_enabled": True,
                "comparison_zones_enabled": True
            }
        self.settings["math_features"][feature] = enabled
        self.save_settings()

    # Math operation preference
    MATH_OPERATIONS = ["addition", "subtraction", "both"]

    def get_math_operation(self):
        """Get the selected math operation focus (addition, subtraction, or both)"""
        return self.settings.get("math_operation", "addition")

    def set_math_operation(self, operation):
        """Set the math operation focus"""
        if operation in self.MATH_OPERATIONS:
            self.settings["math_operation"] = operation
            self.save_settings()
            return True
        return False

    def cycle_math_operation(self):
        """Cycle through math operations and return the new setting"""
        current = self.get_math_operation()
        current_index = self.MATH_OPERATIONS.index(current) if current in self.MATH_OPERATIONS else 0
        new_index = (current_index + 1) % len(self.MATH_OPERATIONS)
        new_operation = self.MATH_OPERATIONS[new_index]
        self.set_math_operation(new_operation)
        return new_operation

    # Math timer setting
    def get_math_timer_enabled(self):
        """Get whether the math riddle timer is enabled"""
        return self.settings.get("math_timer_enabled", True)

    def set_math_timer_enabled(self, enabled):
        """Set whether the math riddle timer is enabled"""
        self.settings["math_timer_enabled"] = enabled
        self.save_settings()

    def toggle_math_timer(self):
        """Toggle the math timer on/off and return new state"""
        current = self.get_math_timer_enabled()
        self.set_math_timer_enabled(not current)
        return not current

class PlaytimeManager:
    def __init__(self, user_manager):
        self.user_manager = user_manager
        self.session_start_time = None
        self.is_time_limit_reached = False
        self.time_warning_shown = False
        
    def start_session(self):
        """Start a new play session"""
        import time
        
        # Check if parental override is active
        if self.user_manager.get_parental_override_active():
            print("PARENTAL OVERRIDE ACTIVE - Time limits temporarily disabled")
            current_time = time.time()
            self.session_start_time = current_time
            self.user_manager.set_session_start_time(current_time)
            self.is_time_limit_reached = False
            self.time_warning_shown = False
            print("Play session started with parental override")
            return True
        
        # Check if daily playtime limit has already been reached
        max_time = self.user_manager.get_max_playtime()
        today_playtime = self.user_manager.get_total_playtime_today()
        
        if today_playtime >= max_time:
            self.is_time_limit_reached = True
            print("DAILY PLAYTIME LIMIT ALREADY REACHED!")
            print(f"You have already played for {today_playtime:.1f} minutes today.")
            print(f"Daily limit: {max_time} minutes.")
            print("Please come back tomorrow or ask a parent to use parental controls.")
            print("Parental controls: Run game with --parental flag")
            return False
        
        current_time = time.time()
        self.session_start_time = current_time
        self.user_manager.set_session_start_time(current_time)
        self.is_time_limit_reached = False
        self.time_warning_shown = False
        
        remaining_time = max_time - today_playtime
        print(f"Play session started. Remaining playtime today: {remaining_time:.1f} minutes")
        return True
        
    def get_session_time_minutes(self):
        """Get current session time in minutes"""
        if self.session_start_time is None:
            return 0
        import time
        return (time.time() - self.session_start_time) / 60
    
    def get_remaining_time_minutes(self):
        """Get remaining playtime in minutes"""
        max_time = self.user_manager.get_max_playtime()
        # Include both today's previous playtime AND current session time
        total_used_time = self.user_manager.get_total_playtime_today() + self.get_session_time_minutes()
        return max(0, max_time - total_used_time)
    
    def check_time_limit(self):
        """Check if time limit has been reached"""
        # Skip time limit checks if parental override is active
        if self.user_manager.get_parental_override_active():
            return False
        
        remaining_time = self.get_remaining_time_minutes()
        
        # Show warning at 5 minutes remaining
        if remaining_time <= 5 and remaining_time > 0 and not self.time_warning_shown:
            self.time_warning_shown = True
            print(f"WARNING: Only {remaining_time:.1f} minutes of playtime remaining!")
            print("Ask a parent to activate parental override if more time is needed.")
            return False
        
        # Check if time limit reached
        if remaining_time <= 0:
            if not self.is_time_limit_reached:
                self.is_time_limit_reached = True
                self.end_session()
                print("PLAYTIME LIMIT REACHED!")
                print(f"You have played for {self.user_manager.get_max_playtime()} minutes.")
                print("Please take a break. The game will now close.")
                print("Parents can use --parental flag to access override controls.")
            return True
        
        return False
    
    def end_session(self):
        """End the current play session"""
        if self.session_start_time:
            session_minutes = self.get_session_time_minutes()
            self.user_manager.add_playtime_today(session_minutes)
            total_today = self.user_manager.get_total_playtime_today()
            print(f"Session ended. Played for {session_minutes:.1f} minutes this session.")
            print(f"Total playtime today: {total_today:.1f} minutes.")
        
        self.user_manager.reset_session()
        self.session_start_time = None
    
    def format_time(self, minutes):
        """Format time in minutes to readable string"""
        if minutes < 1:
            return f"{int(minutes * 60)}s"
        elif minutes < 60:
            return f"{int(minutes)}m {int((minutes % 1) * 60)}s"
        else:
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours}h {mins}m"

class MusicManager:
    def __init__(self, user_manager):
        self.user_manager = user_manager
        self.current_music = None
        self.music_files = []
        self.current_track_index = 0
        self.is_playing = False
        
        # Initialize music system
        pygame.mixer.music.set_volume(self.user_manager.get_music_volume())
        
        # Load music files
        self.load_music_files()
        
        # Start playing if enabled and files available
        if self.user_manager.get_music_enabled() and self.music_files:
            self.play_next_track()
    
    def load_music_files(self):
        """Load music files from the music folder"""
        music_folder = self.user_manager.get_music_folder()
        supported_formats = ['.mp3', '.wav', '.ogg', '.m4a']
        
        try:
            # Create music folder if it doesn't exist
            if not os.path.exists(music_folder):
                os.makedirs(music_folder)
                print(f"Created music folder: {music_folder}")
                print(f"Add your music files (.mp3, .wav, .ogg, .m4a) to the '{music_folder}' folder")
                return
            
            # Scan for music files
            self.music_files = []
            for file in os.listdir(music_folder):
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    full_path = os.path.join(music_folder, file)
                    self.music_files.append(full_path)
            
            if self.music_files:
                random.shuffle(self.music_files)  # Randomize playlist
                print(f"Found {len(self.music_files)} music files in '{music_folder}'")
            else:
                print(f"No music files found in '{music_folder}'. Supported formats: {', '.join(supported_formats)}")
                
        except Exception as e:
            print(f"Error loading music files: {e}")
    
    def play_next_track(self):
        """Play the next track in the playlist"""
        if not self.music_files or not self.user_manager.get_music_enabled():
            return
        
        try:
            track = self.music_files[self.current_track_index]
            pygame.mixer.music.load(track)
            pygame.mixer.music.play()
            self.is_playing = True
            
            track_name = os.path.basename(track)
            print(f"Now playing: {track_name}")
            
            # Move to next track for next time
            self.current_track_index = (self.current_track_index + 1) % len(self.music_files)
            
        except pygame.error as e:
            print(f"Error playing music file '{track}': {e}")
            # Try next track
            self.current_track_index = (self.current_track_index + 1) % len(self.music_files)
            if self.current_track_index != 0:  # Avoid infinite loop
                self.play_next_track()
    
    def update(self):
        """Check if current track finished and play next"""
        if self.is_playing and not pygame.mixer.music.get_busy():
            self.play_next_track()
    
    def toggle_music(self):
        """Toggle music on/off"""
        if self.user_manager.get_music_enabled():
            self.stop_music()
            self.user_manager.set_music_enabled(False)
            print("Music disabled")
        else:
            self.user_manager.set_music_enabled(True)
            print("Music enabled")
            if self.music_files:
                self.play_next_track()
    
    def stop_music(self):
        """Stop current music"""
        pygame.mixer.music.stop()
        self.is_playing = False
    
    def adjust_volume(self, change):
        """Adjust music volume by change amount"""
        current_volume = self.user_manager.get_music_volume()
        new_volume = max(0.0, min(1.0, current_volume + change))
        self.user_manager.set_music_volume(new_volume)
        pygame.mixer.music.set_volume(new_volume)
        print(f"Music volume: {int(new_volume * 100)}%")
    
    def skip_track(self):
        """Skip to next track"""
        if self.music_files and self.user_manager.get_music_enabled():
            self.play_next_track()

class OceanBackground:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.background = self.create_ocean_background()
        self.decorative_fish = self.create_decorative_fish()
        
        self.fire_particles = [] 
        # Level-based volcano system
        self.all_volcanoes = self.create_all_volcanoes()  # Create all possible volcanoes
        self.current_level = 1
        self.active_volcanoes = []  # Only volcanoes active at current level
        
        # Lava effects
        self.lava_particles = []
        self.lava_flows = []  # Tracks lava flowing down volcano sides
        self.steam_particles = []  # Steam rising from volcanoes
        self.animation_timer = 0
        
        # Ocean floor life
        self.sea_crabs = self.create_sea_crabs()
        self.kelp_plants = self.create_kelp_forest()
        self.algae_patches = self.create_algae_patches()
        
        # Earthquake system
        self.active_earthquakes = []
        self.screen_shake_offset = {'x': 0, 'y': 0}
        
        # Sound manager reference (will be set by game)
        self.sound_manager = None

    def create_fire_particle(self, volcano):
        """Create a fire particle shooting from the volcano crater"""
        crater_x = volcano['x'] + volcano['width'] // 2
        crater_y = volcano['base_y'] - volcano['height'] + 20
        angle = random.uniform(-0.5, 0.5)  # Slight spread
        speed = random.uniform(4, 8) if volcano['activity_level'] == 'high' else random.uniform(2, 5)
        return {
            'x': crater_x,
            'y': crater_y,
            'vel_x': speed * math.sin(angle),
            'vel_y': -speed * math.cos(angle) - random.uniform(1, 3),
            'size': random.randint(4, 8),
            'life': random.randint(18, 30),
            'max_life': 30,
            'color': random.choice([(255, 180, 0), (255, 100, 0), (255, 255, 100)])
        }
    def create_ocean_background(self):
        """Create an ocean background with gradient and waves"""
        background = pygame.Surface((self.width, self.height))
        
        # Create gradient from light blue at top to dark blue at bottom
        for y in range(self.height):
            ratio = y / self.height
            # Interpolate between light and dark ocean colors
            r = int(OCEAN_LIGHT[0] * (1 - ratio) + OCEAN_DARK[0] * ratio)
            g = int(OCEAN_LIGHT[1] * (1 - ratio) + OCEAN_DARK[1] * ratio)
            b = int(OCEAN_LIGHT[2] * (1 - ratio) + OCEAN_DARK[2] * ratio)
            pygame.draw.line(background, (r, g, b), (0, y), (self.width, y))
        
        # Add wave patterns
        import math
        for wave_y in range(0, self.height, 60):
            for x in range(0, self.width, 5):
                wave_offset = int(15 * math.sin((x + wave_y) * 0.02))
                wave_color = (min(255, OCEAN_BLUE[0] + 20), 
                             min(255, OCEAN_BLUE[1] + 20), 
                             min(255, OCEAN_BLUE[2] + 20))
                if wave_y + wave_offset < self.height:
                    pygame.draw.circle(background, wave_color, 
                                     (x, wave_y + wave_offset), 2)
        
        # Add some coral/seaweed
        for i in range(20):
            x = random.randint(0, self.width)
            bottom_y = self.height
            height = random.randint(50, 150)
            coral_color = random.choice([(0, 100, 0), (100, 50, 0), (150, 0, 150)])
            
            # Draw seaweed/coral strands
            for segment in range(0, height, 10):
                y = bottom_y - segment
                sway = int(10 * math.sin(segment * 0.1))
                if y > 0:
                    pygame.draw.circle(background, coral_color, (x + sway, y), 3)
        
        return background
    
    def create_all_volcanoes(self):
        """Create all possible volcanoes for level-based activation"""
        volcanoes = []
        
        # Create 5 volcanoes with progressive unlock levels
        # Level 2: First volcano (high activity)
        # Level 4: Second volcano (medium activity) - 2 volcanoes total  
        # Level 4: Third volcano (high activity) - 3 volcanoes total
        # Level 4: Fourth volcano (medium activity) - 4 volcanoes total  
        # Level 4: Fifth volcano (medium activity) - 5 volcanoes total (max reached)
        volcano_configs = [
            {'x': self.width - 180, 'base_y': self.height - 120, 'height': 180, 'width': 120, 'activity': 'high', 'unlock_level': 2},
            {'x': self.width - 350, 'base_y': self.height - 80, 'height': 140, 'width': 100, 'activity': 'medium', 'unlock_level': 4},
            {'x': self.width - 500, 'base_y': self.height - 160, 'height': 200, 'width': 140, 'activity': 'high', 'unlock_level': 4},
            {'x': self.width - 650, 'base_y': self.height - 100, 'height': 160, 'width': 110, 'activity': 'medium', 'unlock_level': 4},
            {'x': self.width - 800, 'base_y': self.height - 140, 'height': 170, 'width': 125, 'activity': 'medium', 'unlock_level': 4}
        ]
        
        for config in volcano_configs:
            # Generate rugged surface points for realistic appearance
            rugged_points = []
            base_points = 12  # Number of points around the volcano
            
            for i in range(base_points):
                angle = (i / base_points) * 2 * math.pi
                
                # Base radius varies for ruggedness
                base_radius = config['width'] / 2 * (0.8 + 0.4 * random.random())
                top_radius = config['width'] / 4 * (0.6 + 0.8 * random.random())
                
                # Create rugged height variations
                height_variation = config['height'] * (0.85 + 0.3 * random.random())
                
                # Calculate points for base and top
                base_x = config['x'] + config['width']/2 + base_radius * math.cos(angle)
                base_y = config['base_y']
                
                top_x = config['x'] + config['width']/2 + top_radius * math.cos(angle)
                top_y = config['base_y'] - height_variation
                
                rugged_points.append({
                    'base': (base_x, base_y),
                    'top': (top_x, top_y),
                    'angle': angle
                })
            
            volcano = {
                'x': config['x'],
                'base_y': config['base_y'],
                'height': config['height'],
                'width': config['width'],
                'activity_level': config['activity'],
                'unlock_level': config['unlock_level'],
                'rugged_points': rugged_points,
                'eruption_timer': random.randint(0, 300),
                'last_eruption': 0,
                'crater_glow_phase': random.uniform(0, math.pi * 2),
                'next_eruption_time': random.randint(180, 600),  # 3-10 seconds
                'earthquake_timer': 0,
                'earthquake_started': False,
                'lava_flow_points': [],  # Will store active lava flow paths
                'flow_timer': 0,
                'is_active': False  # Will be activated based on level
            }
            volcanoes.append(volcano)
        
        return volcanoes
    
    def update_level(self, new_level):
        """Update current level and activate appropriate volcanoes"""
        if new_level != self.current_level:
            self.current_level = new_level
            
            # Update active volcanoes based on level
            self.active_volcanoes = []
            for volcano in self.all_volcanoes:
                if volcano['unlock_level'] <= self.current_level:
                    volcano['is_active'] = True
                    self.active_volcanoes.append(volcano)
                else:
                    volcano['is_active'] = False
            
            # Ensure all active volcanoes have full effects (earthquake, eruption, lava flow)
            for volcano in self.active_volcanoes:
                # Reset timers for smooth integration
                volcano['eruption_timer'] = random.randint(0, 60)
                volcano['earthquake_started'] = False
                # Ensure ALL volcanoes have medium+ activity for full effects
                if volcano['activity_level'] == 'low':
                    volcano['activity_level'] = 'medium'  # Upgrade low activity to ensure lava flows
            
            print(f"Level {new_level}: {len(self.active_volcanoes)} volcanoes active")
    
    def get_active_volcano_count(self):
        """Get number of currently active volcanoes"""
        return len(self.active_volcanoes)
    
    def create_decorative_fish(self):
        """Create small decorative fish that swim in the background"""
        fish_list = []
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (255, 255, 100),  # Yellow
            (255, 150, 100),  # Orange
            (150, 100, 255),  # Purple
            (100, 255, 255),  # Cyan
        ]
        
        for i in range(15):
            fish = {
                'x': random.randint(0, self.width),
                'y': random.randint(100, self.height - 100),
                'color': random.choice(colors),
                'speed': random.uniform(0.5, 2.0),
                'direction': random.choice([-1, 1]),
                'size': random.randint(8, 15),
                'swim_timer': random.randint(0, 100)
            }
            fish_list.append(fish)
        
        return fish_list
    
    def update_decorative_fish(self):
        """Update positions of decorative fish"""
        for fish in self.decorative_fish:
            fish['x'] += fish['speed'] * fish['direction']
            fish['swim_timer'] += 1
            
            # Add swimming motion
            fish['y'] += math.sin(fish['swim_timer'] * 0.1) * 0.5
            
            # Wrap around screen
            if fish['direction'] > 0 and fish['x'] > self.width + 20:
                fish['x'] = -20
            elif fish['direction'] < 0 and fish['x'] < -20:
                fish['x'] = self.width + 20
            
            # Keep fish in bounds vertically
            fish['y'] = max(50, min(self.height - 50, fish['y']))
    
    def draw_decorative_fish(self, surface):
        """Draw decorative fish on the surface"""
        for fish in self.decorative_fish:
            x, y = int(fish['x']), int(fish['y'])
            color = fish['color']
            size = fish['size']
            
            # Fish body (ellipse)
            pygame.draw.ellipse(surface, color, (x, y, size, size//2))
            
            # Fish tail
            if fish['direction'] > 0:
                tail_points = [(x, y + size//4), (x - size//3, y), (x - size//3, y + size//2)]
            else:
                tail_points = [(x + size, y + size//4), (x + size + size//3, y), (x + size + size//3, y + size//2)]
            pygame.draw.polygon(surface, color, tail_points)
            
            # Fish eye
            eye_x = x + size//3 if fish['direction'] > 0 else x + 2*size//3
            pygame.draw.circle(surface, BLACK, (eye_x, y + size//4), 2)
    
    def create_sea_crabs(self):
        """Create sea crabs on the ocean floor"""
        crabs = []
        
        # Create 8-12 crabs scattered on the ocean floor
        for i in range(random.randint(8, 12)):
            crab = {
                'x': random.randint(50, self.width - 50),
                'y': self.height - random.randint(5, 25),  # Near ocean floor
                'size': random.randint(8, 15),
                'leg_phase': random.uniform(0, math.pi * 2),
                'move_timer': random.randint(0, 300),
                'move_direction': random.choice([-1, 0, 1]),
                'claw_phase': random.uniform(0, math.pi * 2),
                'color': random.choice([
                    (150, 75, 50),   # Brown
                    (180, 90, 60),   # Light brown
                    (120, 60, 40),   # Dark brown
                    (160, 80, 45)    # Reddish brown
                ])
            }
            crabs.append(crab)
        
        return crabs
    
    def create_kelp_forest(self):
        """Create kelp plants swaying in the current"""
        kelp_plants = []
        
        # Create 15-20 kelp plants
        for i in range(random.randint(15, 20)):
            # Kelp segments for realistic swaying
            segments = random.randint(5, 10)
            base_x = random.randint(20, self.width - 20)
            base_y = self.height - 5
            
            kelp_plant = {
                'base_x': base_x,
                'base_y': base_y,
                'segments': segments,
                'height': random.randint(80, 150),
                'sway_phase': random.uniform(0, math.pi * 2),
                'sway_intensity': random.uniform(15, 30),
                'color': random.choice([
                    (34, 139, 34),   # Forest green
                    (46, 125, 50),   # Sea green
                    (85, 107, 47),   # Dark olive green
                    (60, 179, 113)   # Medium sea green
                ]),
                'width': random.randint(3, 6)
            }
            kelp_plants.append(kelp_plant)
        
        return kelp_plants
    
    def create_algae_patches(self):
        """Create algae patches on rocks and surfaces"""
        algae_patches = []
        
        # Create 20-30 algae patches
        for i in range(random.randint(20, 30)):
            patch = {
                'x': random.randint(0, self.width),
                'y': random.randint(self.height - 100, self.height),
                'radius': random.randint(5, 20),
                'density': random.uniform(0.3, 0.8),
                'color': random.choice([
                    (107, 142, 35),  # Olive drab
                    (154, 205, 50),  # Yellow green
                    (124, 252, 0),   # Lawn green
                    (50, 205, 50),   # Lime green
                    (32, 178, 170)   # Light sea green
                ]),
                'growth_phase': random.uniform(0, math.pi * 2)
            }
            algae_patches.append(patch)
        
        return algae_patches
    
    def create_steam_particle(self, volcano):
        """Create steam particle rising from volcano crater"""
        crater_x = volcano['x'] + volcano['width'] // 2
        crater_y = volcano['base_y'] - volcano['height'] + 20
        
        particle = {
            'x': crater_x + random.randint(-10, 10),
            'y': crater_y,
            'vel_x': random.uniform(-0.5, 0.5),
            'vel_y': random.uniform(-1.5, -0.8),  # Rising motion
            'size': random.randint(2, 6),
            'life': random.randint(120, 200),  # Longer life than lava
            'max_life': 200,
            'opacity': random.uniform(0.3, 0.7),
            'expansion_rate': random.uniform(0.02, 0.05)
        }
        return particle
    
    def create_lava_particle(self, volcano):
        """Create a new lava particle for eruption effect"""
        import math
        
        # Create particle at volcano crater
        crater_x = volcano['x'] + volcano['width'] // 2
        crater_y = volcano['base_y'] - volcano['height'] + 20
        
        # Vary particle properties based on volcano activity
        if volcano['activity_level'] == 'high':
            vel_range = (-4, 4)
            vel_y_range = (-10, -5)
            size_range = (4, 10)
            life_range = (80, 150)
        elif volcano['activity_level'] == 'medium':
            vel_range = (-3, 3)
            vel_y_range = (-8, -4)
            size_range = (3, 8)
            life_range = (60, 120)
        else:  # low activity
            vel_range = (-2, 2)
            vel_y_range = (-6, -3)
            size_range = (2, 6)
            life_range = (40, 100)
        
        particle = {
            'x': crater_x + random.randint(-20, 20),
            'y': crater_y,
            'vel_x': random.uniform(*vel_range),
            'vel_y': random.uniform(*vel_y_range),
            'size': random.randint(*size_range),
            'life': random.randint(*life_range),
            'max_life': life_range[1],
            'color_phase': random.uniform(0, math.pi * 2),
            'volcano_id': id(volcano)  # Track which volcano created this particle
        }
        return particle
    
    def create_lava_flow(self, volcano):
        """Create lava flow down the side of a volcano"""
        import math
        
        # Choose a random side of the volcano to flow down
        flow_side = random.choice(['left', 'right', 'front'])
        
        # Start point near crater
        crater_x = volcano['x'] + volcano['width'] // 2
        crater_y = volcano['base_y'] - volcano['height'] + 20
        
        if flow_side == 'left':
            start_x = crater_x - random.randint(10, 25)
        elif flow_side == 'right':
            start_x = crater_x + random.randint(10, 25)
        else:  # front
            start_x = crater_x + random.randint(-15, 15)
        
        # Create flow path points
        flow_points = []
        current_x = start_x
        current_y = crater_y
        
        # Generate flowing path down the volcano
        segments = random.randint(8, 15)
        for i in range(segments):
            # Move down and slightly sideways
            current_y += random.randint(8, 15)
            current_x += random.randint(-8, 8)
            
            # Stop if we reach the base
            if current_y >= volcano['base_y'] - 10:
                current_y = volcano['base_y'] - 5
                flow_points.append((current_x, current_y))
                break
            
            flow_points.append((current_x, current_y))
        
        # Create lava flow object
        lava_flow = {
            'volcano_id': id(volcano),
            'points': flow_points,
            'progress': 0,  # How far the flow has progressed (0-1)
            'width': random.randint(3, 8),
            'life': random.randint(300, 600),  # How long flow lasts
            'max_life': 600,
            'heat': 1.0,  # Heat level (1.0 = hottest, 0 = cooled)
            'flow_speed': random.uniform(0.02, 0.05)  # How fast it flows
        }
        
        return lava_flow
    
    def update_earthquake_effects(self):
        """Update active earthquakes and screen shake"""
        # Update existing earthquakes
        for earthquake in self.active_earthquakes[:]:
            earthquake['timer'] += 1
            earthquake['intensity'] *= 0.98  # Gradually fade
            
            # Remove finished earthquakes
            if earthquake['timer'] > earthquake['duration']:
                self.active_earthquakes.remove(earthquake)
        
        # Calculate screen shake based on active earthquakes
        total_shake_x = 0
        total_shake_y = 0
        
        for earthquake in self.active_earthquakes:
            shake_intensity = earthquake['intensity'] * earthquake['base_intensity']
            shake_x = random.uniform(-shake_intensity, shake_intensity)
            shake_y = random.uniform(-shake_intensity, shake_intensity)
            total_shake_x += shake_x
            total_shake_y += shake_y
        
        # Smooth the shake for less jarring effect
        self.screen_shake_offset['x'] = total_shake_x * 0.7 + self.screen_shake_offset['x'] * 0.3
        self.screen_shake_offset['y'] = total_shake_y * 0.7 + self.screen_shake_offset['y'] * 0.3
    
    def trigger_earthquake(self, volcano):
        """Trigger an earthquake before volcanic eruption"""
        # Create earthquake based on volcano activity
        intensity_map = {'low': 2, 'medium': 4, 'high': 6}
        base_intensity = intensity_map[volcano['activity_level']]
        
        earthquake = {
            'volcano_id': id(volcano),
            'timer': 0,
            'duration': random.randint(60, 120),  # 1-2 seconds
            'base_intensity': base_intensity,
            'intensity': 1.0
        }
        
        self.active_earthquakes.append(earthquake)
        
        # Play earthquake sound
        if self.sound_manager:
            try:
                earthquake_sound = SoundGenerator.generate_earthquake(volcano['activity_level'])
                pygame.mixer.Sound.play(earthquake_sound)
            except:
                pass
    
    def update_volcano_effects(self):
        """Update volcano animation and lava effects for all volcanoes"""
        self.animation_timer += 1
        
        # Update earthquake effects
        self.update_earthquake_effects()
        
        # Update ocean floor life
        self.update_ocean_floor_life()
        
        # Update each active volcano individually
        for volcano in self.active_volcanoes:
            volcano['eruption_timer'] += 1
            volcano['earthquake_timer'] += 1
            volcano['flow_timer'] += 1
            volcano['crater_glow_phase'] += 0.05
            
            # Generate steam continuously from active volcanoes
            if random.random() < 0.3:  # 30% chance each frame
                steam_particle = self.create_steam_particle(volcano)
                self.steam_particles.append(steam_particle)
            
            # --- FIRE PARTICLES: Generate during eruption ---
            # Mark volcano as erupting for a short time after eruption
            if volcano['eruption_timer'] - volcano['last_eruption'] < 20:
                volcano['is_erupting'] = True
            else:
                volcano['is_erupting'] = False

            # Generate fire particles if erupting
            if volcano.get('is_erupting', False):
                for _ in range(random.randint(1, 3)):
                    self.fire_particles.append(self.create_fire_particle(volcano))

            # Check if earthquake should start (30 seconds before eruption)
            time_to_eruption = volcano['next_eruption_time'] - (volcano['eruption_timer'] - volcano['last_eruption'])
            if time_to_eruption <= 180 and not volcano['earthquake_started']:  # 3 seconds before
                self.trigger_earthquake(volcano)
                volcano['earthquake_started'] = True
            
            # Check if this volcano should erupt
            if volcano['eruption_timer'] - volcano['last_eruption'] > volcano['next_eruption_time']:
                # Trigger eruption
                self.trigger_volcano_eruption(volcano)
                
                # Create lava flows during eruption (ALL active volcanoes get flows)
                flow_count = 2 if volcano['activity_level'] == 'high' else 1
                for _ in range(flow_count):
                    lava_flow = self.create_lava_flow(volcano)
                    self.lava_flows.append(lava_flow)
                
                # Reset for next eruption
                if volcano['activity_level'] == 'high':
                    volcano['next_eruption_time'] = random.randint(120, 300)  # 2-5 seconds
                elif volcano['activity_level'] == 'medium':
                    volcano['next_eruption_time'] = random.randint(240, 480)  # 4-8 seconds
                else:  # low activity
                    volcano['next_eruption_time'] = random.randint(480, 900)  # 8-15 seconds
                
                volcano['last_eruption'] = volcano['eruption_timer']
                volcano['earthquake_started'] = False
        
        # Update existing lava particles
        for particle in self.lava_particles[:]:  # Copy list to allow removal during iteration
            # Apply physics
            particle['x'] += particle['vel_x']
            particle['y'] += particle['vel_y']
            particle['vel_y'] += 0.2  # Gravity
            particle['vel_x'] *= 0.98  # Air resistance
            particle['life'] -= 1
            
            # Remove dead particles
            if particle['life'] <= 0 or particle['y'] > self.height:
                self.lava_particles.remove(particle)
        
        # Update lava flows
        for flow in self.lava_flows[:]:
            # Advance flow progress
            flow['progress'] = min(1.0, flow['progress'] + flow['flow_speed'])
            flow['life'] -= 1
            flow['heat'] = max(0, flow['heat'] - 0.002)  # Cool down over time
            
            # Remove old flows
            if flow['life'] <= 0 or flow['heat'] <= 0:
                self.lava_flows.remove(flow)
        for fire in self.fire_particles[:]:
            fire['x'] += fire['vel_x']
            fire['y'] += fire['vel_y']
            fire['vel_x'] *= 0.97
            fire['vel_y'] += 0.2  # gravity
            fire['life'] -= 1
            fire['size'] *= 0.97  # shrink as it fades
            if fire['life'] <= 0 or fire['y'] < 0:
                self.fire_particles.remove(fire)

        # Update steam particles
        for steam in self.steam_particles[:]:
            # Apply physics - steam rises and disperses
            steam['x'] += steam['vel_x']
            steam['y'] += steam['vel_y']
            steam['vel_x'] *= 0.99  # Slight horizontal dampening
            steam['vel_y'] *= 0.98  # Slight vertical dampening
            steam['size'] += steam['expansion_rate']  # Steam expands as it rises
            steam['life'] -= 1
            
            # Fade opacity as steam disperses
            life_ratio = steam['life'] / steam['max_life']
            steam['opacity'] = steam['opacity'] * life_ratio
            
            # Remove old steam
            if steam['life'] <= 0 or steam['y'] < -50:
                self.steam_particles.remove(steam)
    
    def trigger_volcano_eruption(self, volcano):
        """Trigger an eruption for a specific volcano"""
        # Create burst of lava particles based on activity level
        if volcano['activity_level'] == 'high':
            particle_count = random.randint(15, 25)
        elif volcano['activity_level'] == 'medium':
            particle_count = random.randint(10, 18)
        else:  # low activity
            particle_count = random.randint(5, 12)
        
        for _ in range(particle_count):
            self.lava_particles.append(self.create_lava_particle(volcano))
        
        # Play eruption sound if sound manager is available
        if self.sound_manager:
            try:
                eruption_sound = SoundGenerator.generate_volcano_eruption(volcano['activity_level'])
                pygame.mixer.Sound.play(eruption_sound)
            except:
                pass  # Silently handle sound generation errors
    
    def update_ocean_floor_life(self):
        """Update animations for ocean floor creatures and plants"""
        import math
        
        # Update crab movements and animations
        for crab in self.sea_crabs:
            crab['leg_phase'] += 0.3
            crab['claw_phase'] += 0.2
            crab['move_timer'] -= 1
            
            # Occasional movement
            if crab['move_timer'] <= 0:
                crab['move_direction'] = random.choice([-1, 0, 0, 1])  # More likely to stay still
                crab['move_timer'] = random.randint(60, 300)  # 1-5 seconds
            
            # Move crab if it has a direction
            if crab['move_direction'] != 0:
                crab['x'] += crab['move_direction'] * 0.5
                # Keep crabs on screen
                crab['x'] = max(20, min(self.width - 20, crab['x']))
        
        # Update kelp swaying
        for kelp in self.kelp_plants:
            kelp['sway_phase'] += 0.02
        
        # Update algae growth animation
        for algae in self.algae_patches:
            algae['growth_phase'] += 0.05
            
    def draw_fire_effects(self, surface):
        """Draw fire particles shooting from volcanoes"""
        for fire in self.fire_particles:
            life_ratio = fire['life'] / fire['max_life']
            color = (
                min(255, int(fire['color'][0] * life_ratio + 100 * (1 - life_ratio))),
                min(255, int(fire['color'][1] * life_ratio)),
                int(fire['color'][2] * life_ratio)
            )
            alpha = int(180 * life_ratio)
            fire_surf = pygame.Surface((int(fire['size']*2), int(fire['size']*2)), pygame.SRCALPHA)
            pygame.draw.circle(fire_surf, (*color, alpha), (int(fire['size']), int(fire['size'])), int(fire['size']))
            surface.blit(fire_surf, (int(fire['x'] - fire['size']), int(fire['y'] - fire['size'])))
    def draw_ocean_floor_life(self, surface):
        """Draw sea crabs, kelp, and algae"""
        import math
        
        # Draw algae patches first (background)
        for algae in self.algae_patches:
            # Animated algae with slight size variation
            growth_factor = 0.8 + 0.2 * (1 + math.sin(algae['growth_phase']))
            current_radius = int(algae['radius'] * growth_factor)
            
            # Create algae patch with multiple circles for organic look
            for i in range(int(algae['density'] * 15)):
                offset_x = random.randint(-current_radius, current_radius)
                offset_y = random.randint(-current_radius//2, current_radius//2)
                patch_x = algae['x'] + offset_x
                patch_y = algae['y'] + offset_y
                
                if 0 <= patch_x < self.width and 0 <= patch_y < self.height:
                    circle_size = random.randint(2, 5)
                    pygame.draw.circle(surface, algae['color'], (patch_x, patch_y), circle_size)
        
        # Draw kelp plants
        for kelp in self.kelp_plants:
            # Calculate swaying motion
            sway_offset = kelp['sway_intensity'] * math.sin(kelp['sway_phase'])
            
            # Draw kelp segments
            for segment in range(kelp['segments']):
                segment_ratio = segment / kelp['segments']
                segment_height = kelp['height'] * segment_ratio
                
                # Current segment position
                current_x = kelp['base_x'] + sway_offset * segment_ratio
                current_y = kelp['base_y'] - segment_height
                
                # Next segment position
                if segment < kelp['segments'] - 1:
                    next_ratio = (segment + 1) / kelp['segments']
                    next_height = kelp['height'] * next_ratio
                    next_x = kelp['base_x'] + sway_offset * next_ratio
                    next_y = kelp['base_y'] - next_height
                    
                    # Draw kelp segment
                    pygame.draw.line(surface, kelp['color'], 
                                   (int(current_x), int(current_y)), 
                                   (int(next_x), int(next_y)), 
                                   kelp['width'])
                
                # Add kelp leaves occasionally
                if segment % 2 == 0 and segment > 0:
                    leaf_x = current_x + random.randint(-8, 8)
                    leaf_y = current_y + random.randint(-5, 5)
                    leaf_size = random.randint(3, 6)
                    pygame.draw.circle(surface, kelp['color'], (int(leaf_x), int(leaf_y)), leaf_size)
        
        # Draw sea crabs
        for crab in self.sea_crabs:
            # Crab body (ellipse)
            body_x = int(crab['x'])
            body_y = int(crab['y'])
            body_size = crab['size']
            
            pygame.draw.ellipse(surface, crab['color'], 
                              (body_x - body_size//2, body_y - body_size//3, 
                               body_size, body_size//2))
            
            # Animated crab legs
            leg_offset = 2 * math.sin(crab['leg_phase'])
            for i in range(6):  # 6 legs
                angle = (i / 6) * math.pi
                leg_x = body_x + (body_size//2 + 5) * math.cos(angle)
                leg_y = body_y + leg_offset + 3 * math.sin(angle)
                
                # Draw leg
                pygame.draw.line(surface, crab['color'], 
                               (body_x, body_y), 
                               (int(leg_x), int(leg_y)), 2)
            
            # Animated claws
            claw_offset = 3 * math.sin(crab['claw_phase'])
            
            # Left claw
            left_claw_x = body_x - body_size//2 - 5
            left_claw_y = body_y - 2 + claw_offset
            pygame.draw.circle(surface, crab['color'], (int(left_claw_x), int(left_claw_y)), 3)
            
            # Right claw
            right_claw_x = body_x + body_size//2 + 5
            right_claw_y = body_y - 2 - claw_offset
            pygame.draw.circle(surface, crab['color'], (int(right_claw_x), int(right_claw_y)), 3)
            
            # Crab eyes
            pygame.draw.circle(surface, BLACK, (body_x - 3, body_y - 4), 2)
            pygame.draw.circle(surface, BLACK, (body_x + 3, body_y - 4), 2)
    
    def draw_steam_effects(self, surface):
        """Draw steam rising from volcanoes"""
        for steam in self.steam_particles:
            # Create semi-transparent steam effect
            steam_color = (200, 200, 200)  # Light gray
            alpha = int(steam['opacity'] * 255)
            
            if alpha > 0:
                # Create surface for steam particle with transparency
                steam_surf = pygame.Surface((steam['size'] * 2, steam['size'] * 2), pygame.SRCALPHA)
                steam_surf.set_alpha(alpha)
                pygame.draw.circle(steam_surf, steam_color, 
                                 (int(steam['size']), int(steam['size'])), 
                                 int(steam['size']))
                
                # Blit steam particle to main surface
                surface.blit(steam_surf, (int(steam['x'] - steam['size']), int(steam['y'] - steam['size'])))
    
    def draw_volcanoes(self, surface):
        """Draw all 3D volcanoes with animated lava"""
        import math
        
        # Volcano color palette for realistic appearance
        VOLCANO_COLORS = {
            'base_dark': (60, 30, 15),
            'base_mid': (90, 45, 25),
            'base_light': (120, 60, 35),
            'rock_dark': (70, 35, 20),
            'rock_light': (140, 70, 40),
            'crater_hot': (200, 50, 0),
            'crater_bright': (255, 80, 0)
        }
        
        # Draw active volcanoes from back to front for proper depth
        for volcano in sorted(self.active_volcanoes, key=lambda v: v['x']):
            self.draw_single_volcano(surface, volcano, VOLCANO_COLORS)
        
        # Draw all lava particles on top
        self.draw_lava_particles(surface)
    
    def draw_single_volcano(self, surface, volcano, colors):
        """Draw a single volcano with rugged, realistic 3D appearance"""
        import math
        
        # Draw rugged volcano outline using the pre-generated points
        rugged_points = volcano['rugged_points']
        
        # Create base polygon points
        base_polygon = [point['base'] for point in rugged_points]
        top_polygon = [point['top'] for point in rugged_points]
        
        # Draw base (darkest)
        pygame.draw.polygon(surface, colors['base_dark'], base_polygon)
        
        # Draw individual faces for 3D effect
        num_points = len(rugged_points)
        for i in range(num_points):
            next_i = (i + 1) % num_points
            
            # Create face polygon (4 points: two base, two top)
            face_points = [
                rugged_points[i]['base'],
                rugged_points[next_i]['base'],
                rugged_points[next_i]['top'],
                rugged_points[i]['top']
            ]
            
            # Calculate face color based on angle (simple lighting)
            angle = rugged_points[i]['angle']
            light_factor = 0.5 + 0.5 * math.cos(angle - math.pi/4)  # Light from upper left
            
            face_color = (
                int(colors['base_mid'][0] * light_factor),
                int(colors['base_mid'][1] * light_factor),
                int(colors['base_mid'][2] * light_factor)
            )
            
            pygame.draw.polygon(surface, face_color, face_points)
            
            # Add rock texture details
            if random.random() < 0.3:  # 30% chance for rock details
                detail_x = (rugged_points[i]['base'][0] + rugged_points[i]['top'][0]) / 2
                detail_y = (rugged_points[i]['base'][1] + rugged_points[i]['top'][1]) / 2
                detail_size = random.randint(2, 5)
                pygame.draw.circle(surface, colors['rock_dark'], 
                                 (int(detail_x), int(detail_y)), detail_size)
        
        # Draw top surface
        pygame.draw.polygon(surface, colors['base_light'], top_polygon)
        
        # Draw crater with dynamic glow
        crater_center_x = volcano['x'] + volcano['width'] // 2
        crater_center_y = volcano['base_y'] - volcano['height'] + 20
        
        # Animated glow based on activity level and volcano's individual phase
        activity_multiplier = {'low': 0.6, 'medium': 0.8, 'high': 1.0}[volcano['activity_level']]
        glow_intensity = (0.4 + 0.6 * math.sin(volcano['crater_glow_phase'])) * activity_multiplier
        
        glow_color = (
            int(255 * glow_intensity),
            int(100 * glow_intensity * 0.8),
            int(20 * glow_intensity * 0.6)
        )
        
        # Draw layered glow effect
        max_radius = int(15 + 10 * activity_multiplier)
        for radius in range(max_radius, 3, -2):
            alpha = int(40 * glow_intensity * (1 - radius / max_radius))
            if alpha > 0:
                glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*glow_color, alpha), (radius, radius), radius)
                surface.blit(glow_surf, (crater_center_x - radius, crater_center_y - radius))
        
        # Draw crater opening
        crater_radius = int(8 + 4 * activity_multiplier)
        pygame.draw.circle(surface, colors['crater_hot'], (crater_center_x, crater_center_y), crater_radius)
        inner_radius = max(2, crater_radius - 3)
        pygame.draw.circle(surface, colors['crater_bright'], (crater_center_x, crater_center_y), inner_radius)
    
    def draw_lava_particles(self, surface):
        """Draw all lava particles with realistic effects"""
        import math
        
        for particle in self.lava_particles:
            # Calculate color based on particle life and animation
            life_ratio = particle['life'] / particle['max_life']
            heat = 0.5 + 0.5 * math.sin(particle['color_phase'] + self.animation_timer * 0.2)
            
            # More realistic color transitions
            if life_ratio > 0.8:
                # White hot
                color = (255, int(255 * heat), int(220 * heat))
            elif life_ratio > 0.6:
                # Yellow hot
                color = (255, int(200 * heat), int(100 * heat))
            elif life_ratio > 0.4:
                # Orange
                color = (255, int(120 * heat), int(30 * heat))
            elif life_ratio > 0.2:
                # Red
                color = (int(200 * life_ratio), int(50 * heat), 0)
            else:
                # Dark red/black
                color = (int(100 * life_ratio), int(20 * life_ratio), 0)
            
            # Draw particle with size based on life
            particle_size = max(1, int(particle['size'] * life_ratio))
            particle_pos = (int(particle['x']), int(particle['y']))
            
            # Draw main particle
            pygame.draw.circle(surface, color, particle_pos, particle_size)
            
            # Add glow effect for hot particles
            if life_ratio > 0.5 and particle_size > 2:
                glow_radius = particle_size + 2
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                glow_alpha = int(60 * life_ratio)
                glow_color = (*color[:3], glow_alpha)
                pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
                surface.blit(glow_surf, (particle_pos[0] - glow_radius, particle_pos[1] - glow_radius))
    
    def draw_lava_flows(self, surface):
        """Draw flowing lava down volcano sides"""
        import math
        
        for flow in self.lava_flows:
            if flow['progress'] <= 0:
                continue
            
            points = flow['points']
            if len(points) < 2:
                continue
            
            # Calculate how many points to draw based on progress
            total_points = len(points)
            visible_points = max(1, int(total_points * flow['progress']))
            
            # Draw the flow
            for i in range(visible_points - 1):
                start_point = points[i]
                end_point = points[i + 1]
                
                # Calculate color based on heat
                heat_ratio = flow['heat']
                if heat_ratio > 0.8:
                    # Very hot - bright yellow/white
                    color = (255, int(255 * heat_ratio), int(200 * heat_ratio))
                elif heat_ratio > 0.6:
                    # Hot - orange/yellow
                    color = (255, int(180 * heat_ratio), int(100 * heat_ratio))
                elif heat_ratio > 0.4:
                    # Warm - orange/red
                    color = (255, int(120 * heat_ratio), int(50 * heat_ratio))
                elif heat_ratio > 0.2:
                    # Cooling - red
                    color = (int(200 * heat_ratio), int(80 * heat_ratio), 0)
                else:
                    # Cool - dark red/black
                    color = (int(100 * heat_ratio), int(40 * heat_ratio), 0)
                
                # Calculate width based on position and heat
                position_factor = 1.0 - (i / max(1, total_points - 1))  # Thicker at top
                current_width = max(1, int(flow['width'] * position_factor * heat_ratio))
                
                # Draw flow segment
                if current_width > 1:
                    # Draw thick line as a series of circles
                    steps = max(1, int(math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) / 2))
                    for step in range(steps + 1):
                        t = step / max(1, steps)
                        x = int(start_point[0] + t * (end_point[0] - start_point[0]))
                        y = int(start_point[1] + t * (end_point[1] - start_point[1]))
                        pygame.draw.circle(surface, color, (x, y), current_width)
                        
                        # Add glow for hot lava
                        if heat_ratio > 0.6:
                            glow_radius = current_width + 2
                            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                            glow_alpha = int(40 * heat_ratio)
                            glow_color = (*color[:3], glow_alpha)
                            pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
                            surface.blit(glow_surf, (x - glow_radius, y - glow_radius))
                else:
                    # Draw simple line
                    pygame.draw.line(surface, color, start_point, end_point, 1)
    
    def get_screen_shake_offset(self):
        """Get current screen shake offset for earthquake effects"""
        return self.screen_shake_offset
    
    def set_sound_manager(self, sound_manager):
        """Set the sound manager for volcano eruption sounds"""
        self.sound_manager = sound_manager
    
    def draw(self, surface):
        """Draw the complete ocean background"""
        surface.blit(self.background, (0, 0))
        self.update_decorative_fish()
        self.draw_decorative_fish(surface)
        self.update_volcano_effects()
        self.draw_volcanoes(surface)
        self.draw_fire_effects(surface)   # <-- Draw fire above volcanoes
        self.draw_steam_effects(surface)  # <-- Draw steam above volcanoes
        self.draw_lava_flows(surface)

class SoundGenerator:
    @staticmethod
    def generate_roar():
        """Generate a dinosaur-like roar sound"""
        duration = 1.5
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # Create a complex roar sound with multiple frequencies
        arr = np.zeros(frames)
        
        # Low frequency rumble (30-80 Hz)
        for freq in [35, 45, 65]:
            t = np.linspace(0, duration, frames)
            wave = np.sin(2 * np.pi * freq * t)
            # Add some modulation for growl effect
            modulation = 1 + 0.3 * np.sin(2 * np.pi * 8 * t)
            arr += wave * modulation * 0.3
        
        # Mid frequency growl (100-300 Hz)
        for freq in [120, 180, 250]:
            t = np.linspace(0, duration, frames)
            wave = np.sin(2 * np.pi * freq * t)
            # Envelope to make it fade in and out
            envelope = np.exp(-t * 2) + 0.3 * np.exp(-t * 0.5)
            arr += wave * envelope * 0.2
        
        # Add some noise for texture
        noise = np.random.normal(0, 0.1, frames)
        arr += noise * np.exp(-t * 1.5) * 0.1
        
        # Normalize and convert to integer
        arr = arr / np.max(np.abs(arr)) * 0.7
        arr = (arr * 32767).astype(np.int16)
        
        # Convert to stereo and ensure C-contiguous
        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)
    
    @staticmethod
    def generate_volcano_eruption(intensity='medium'):
        """Generate realistic volcano eruption sound"""
        # Vary duration and characteristics based on intensity
        if intensity == 'low':
            duration = 2.0
            base_volume = 0.3
            rumble_freq_range = (20, 60)
        elif intensity == 'medium':
            duration = 3.5
            base_volume = 0.5
            rumble_freq_range = (25, 80)
        else:  # high intensity
            duration = 5.0
            base_volume = 0.7
            rumble_freq_range = (30, 100)
        
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        t = np.linspace(0, duration, frames)
        
        # Deep rumble - the core of the eruption
        for freq in np.arange(rumble_freq_range[0], rumble_freq_range[1], 8):
            wave = np.sin(2 * np.pi * freq * t)
            # Slowly building intensity
            envelope = np.minimum(t / 0.5, 1.0) * np.exp(-(t - duration/2)**2 / (duration/3)**2)
            arr += wave * envelope * base_volume * 0.4
        
        # Mid-frequency explosion sounds (steam and gas release)
        for freq in [200, 350, 500, 750]:
            wave = np.sin(2 * np.pi * freq * t)
            # Quick burst pattern
            burst_envelope = np.exp(-t * 2) + 0.5 * np.exp(-t * 0.8)
            # Add some randomness for realistic variation
            phase_shift = np.random.uniform(0, 2 * np.pi)
            modulation = 1 + 0.4 * np.sin(2 * np.pi * 12 * t + phase_shift)
            arr += wave * burst_envelope * modulation * base_volume * 0.2
        
        # High-frequency hissing (steam and debris)
        for freq in [1000, 1500, 2000, 3000]:
            wave = np.sin(2 * np.pi * freq * t)
            hiss_envelope = np.exp(-t * 1.5) * np.maximum(0, 1 - t / duration)
            # Rapid modulation for hissing effect
            hiss_mod = 1 + 0.6 * np.sin(2 * np.pi * 25 * t)
            arr += wave * hiss_envelope * hiss_mod * base_volume * 0.15
        
        # Add realistic noise for debris and rock sounds
        noise_intensity = base_volume * 0.3
        rock_noise = np.random.normal(0, noise_intensity, frames)
        # Shape noise with envelope
        noise_envelope = np.exp(-t * 1.2) * (1 + 0.5 * np.sin(2 * np.pi * 0.8 * t))
        arr += rock_noise * noise_envelope
        
        # Add subsonic rumble that you "feel" more than hear
        subsonic_freq = 15
        subsonic = np.sin(2 * np.pi * subsonic_freq * t)
        subsonic_envelope = np.minimum(t / 1.0, 1.0) * np.exp(-t / duration * 2)
        arr += subsonic * subsonic_envelope * base_volume * 0.5
        
        # Dynamic compression and normalization
        arr = arr / (np.max(np.abs(arr)) + 0.001) * base_volume
        
        # Add slight distortion for realism
        arr = np.tanh(arr * 1.2) * 0.9
        
        # Convert to integer format
        arr = (arr * 32767).astype(np.int16)
        
        # Convert to stereo and ensure C-contiguous
        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)
    
    @staticmethod
    def generate_earthquake(intensity='medium'):
        """Generate realistic earthquake rumble sound"""
        # Vary duration and characteristics based on intensity
        if intensity == 'low':
            duration = 3.0
            base_volume = 0.4
            rumble_range = (8, 25)
        elif intensity == 'medium':
            duration = 5.0
            base_volume = 0.6
            rumble_range = (5, 30)
        else:  # high intensity
            duration = 7.0
            base_volume = 0.8
            rumble_range = (3, 35)
        
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        t = np.linspace(0, duration, frames)
        
        # Very low frequency rumble - the signature of earthquakes
        for freq in np.arange(rumble_range[0], rumble_range[1], 3):
            wave = np.sin(2 * np.pi * freq * t)
            # Build up slowly, peak in middle, fade out
            envelope = np.sin(np.pi * t / duration) ** 2
            # Add irregular modulation for realism
            modulation = 1 + 0.3 * np.sin(2 * np.pi * (freq * 0.1) * t + np.random.uniform(0, 2*np.pi))
            arr += wave * envelope * modulation * base_volume * 0.6
        
        # Add subsonic components that you feel more than hear
        for freq in [2, 4, 6, 8]:
            wave = np.sin(2 * np.pi * freq * t)
            subsonic_envelope = np.minimum(t / 1.0, 1.0) * np.maximum(0, 1 - (t - duration/2)**2 / (duration/3)**2)
            arr += wave * subsonic_envelope * base_volume * 0.8
        
        # Add rock grinding/cracking sounds
        mid_frequencies = [40, 60, 80, 120, 180]
        for freq in mid_frequencies:
            wave = np.sin(2 * np.pi * freq * t)
            # Irregular bursts
            burst_pattern = np.maximum(0, np.sin(2 * np.pi * 0.8 * t + np.random.uniform(0, 2*np.pi)))
            crack_envelope = burst_pattern * np.exp(-t / duration * 1.5)
            arr += wave * crack_envelope * base_volume * 0.2
        
        # Add realistic noise for shifting earth
        noise_intensity = base_volume * 0.4
        earth_noise = np.random.normal(0, noise_intensity, frames)
        # Filter noise to lower frequencies
        noise_envelope = np.exp(-t / duration * 0.8) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
        arr += earth_noise * noise_envelope
        
        # Dynamic compression and normalization
        arr = arr / (np.max(np.abs(arr)) + 0.001) * base_volume
        
        # Add slight compression for realism
        arr = np.tanh(arr * 1.1) * 0.95
        
        # Convert to integer format
        arr = (arr * 32767).astype(np.int16)
        
        # Convert to stereo and ensure C-contiguous
        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)
    
    @staticmethod
    def generate_success():
        """Generate a success/level complete sound"""
        duration = 0.8
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # Rising tone sequence
        frequencies = [261, 329, 392, 523]  # C, E, G, C (major chord)
        arr = np.zeros(frames)
        
        for i, freq in enumerate(frequencies):
            start = i * frames // 4
            end = (i + 1) * frames // 4
            t = np.linspace(0, duration/4, end - start)
            wave = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 3)
            arr[start:end] += wave * envelope * 0.5
        
        # Normalize
        arr = arr / np.max(np.abs(arr)) * 0.6
        arr = (arr * 32767).astype(np.int16)
        
        # Convert to stereo and ensure C-contiguous
        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)
    
    @staticmethod
    def generate_swimming():
        """Generate subtle swimming/water sound"""
        duration = 0.3
        sample_rate = 22050
        frames = int(duration * sample_rate)

        # Soft water splash sound
        t = np.linspace(0, duration, frames)
        noise = np.random.normal(0, 1, frames)

        # Filter for water-like sound (high frequency filtered noise)
        arr = noise * np.exp(-t * 8) * 0.1

        # Normalize
        arr = arr / np.max(np.abs(arr)) * 0.3
        arr = (arr * 32767).astype(np.int16)

        # Convert to stereo and ensure C-contiguous
        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)

    @staticmethod
    def generate_correct_answer():
        """Generate ascending tone for correct answer"""
        duration = 0.4
        sample_rate = 22050
        frames = int(duration * sample_rate)

        # Ascending major chord arpeggio (C, E, G, C)
        frequencies = [523, 659, 784, 1047]  # C5, E5, G5, C6
        arr = np.zeros(frames)

        for i, freq in enumerate(frequencies):
            start = i * frames // 4
            end = (i + 1) * frames // 4 + frames // 8  # Overlap for smoother sound
            end = min(end, frames)
            segment_len = end - start
            t = np.linspace(0, (end - start) / sample_rate, segment_len)
            wave = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 4) * 0.6
            arr[start:end] += wave * envelope

        # Add a final shimmer
        t_full = np.linspace(0, duration, frames)
        shimmer = np.sin(2 * np.pi * 1047 * t_full) * np.exp(-t_full * 2) * 0.3
        arr += shimmer

        # Normalize
        arr = arr / (np.max(np.abs(arr)) + 0.001) * 0.7
        arr = (arr * 32767).astype(np.int16)

        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)

    @staticmethod
    def generate_wrong_answer():
        """Generate descending tone for wrong answer"""
        duration = 0.5
        sample_rate = 22050
        frames = int(duration * sample_rate)

        # Descending minor tones
        frequencies = [400, 350, 300, 250]
        arr = np.zeros(frames)

        for i, freq in enumerate(frequencies):
            start = i * frames // 4
            end = (i + 1) * frames // 4 + frames // 8
            end = min(end, frames)
            segment_len = end - start
            t = np.linspace(0, (end - start) / sample_rate, segment_len)
            wave = np.sin(2 * np.pi * freq * t)
            # Add slight dissonance
            wave += np.sin(2 * np.pi * (freq * 1.06) * t) * 0.3
            envelope = np.exp(-t * 3) * 0.5
            arr[start:end] += wave * envelope

        # Normalize
        arr = arr / (np.max(np.abs(arr)) + 0.001) * 0.6
        arr = (arr * 32767).astype(np.int16)

        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)

    @staticmethod
    def generate_shell_collect():
        """Generate sparkle/chime sound for shell collection"""
        duration = 0.3
        sample_rate = 22050
        frames = int(duration * sample_rate)

        t = np.linspace(0, duration, frames)
        arr = np.zeros(frames)

        # High-pitched sparkle tones
        sparkle_freqs = [1200, 1600, 2000, 2400]
        for i, freq in enumerate(sparkle_freqs):
            delay = i * 0.03  # Slight delay between each tone
            delay_samples = int(delay * sample_rate)
            remaining = frames - delay_samples
            if remaining > 0:
                t_segment = np.linspace(0, remaining / sample_rate, remaining)
                wave = np.sin(2 * np.pi * freq * t_segment)
                envelope = np.exp(-t_segment * 12) * 0.4
                arr[delay_samples:] += wave * envelope

        # Add a soft bell undertone
        bell_freq = 800
        bell = np.sin(2 * np.pi * bell_freq * t) * np.exp(-t * 6) * 0.3
        arr += bell

        # Normalize
        arr = arr / (np.max(np.abs(arr)) + 0.001) * 0.5
        arr = (arr * 32767).astype(np.int16)

        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)

    @staticmethod
    def generate_gate_open():
        """Generate sound for math gate opening"""
        duration = 0.6
        sample_rate = 22050
        frames = int(duration * sample_rate)

        t = np.linspace(0, duration, frames)
        arr = np.zeros(frames)

        # Rising whoosh sound
        start_freq = 200
        end_freq = 800
        freq_sweep = start_freq + (end_freq - start_freq) * (t / duration)
        sweep = np.sin(2 * np.pi * freq_sweep * t / 2) * np.exp(-t * 2) * 0.5
        arr += sweep

        # Mechanical click sounds
        for click_time in [0.1, 0.2]:
            click_start = int(click_time * sample_rate)
            click_len = int(0.05 * sample_rate)
            if click_start + click_len < frames:
                click_t = np.linspace(0, 0.05, click_len)
                click = np.sin(2 * np.pi * 600 * click_t) * np.exp(-click_t * 40)
                arr[click_start:click_start + click_len] += click * 0.4

        # Final success chime
        chime_start = int(0.3 * sample_rate)
        chime_t = t[chime_start:] - 0.3
        chime = np.sin(2 * np.pi * 1000 * chime_t) * np.exp(-chime_t * 4) * 0.4
        arr[chime_start:] += chime

        # Normalize
        arr = arr / (np.max(np.abs(arr)) + 0.001) * 0.6
        arr = (arr * 32767).astype(np.int16)

        stereo_arr = np.array([arr, arr]).T
        stereo_arr = np.ascontiguousarray(stereo_arr)
        return pygame.sndarray.make_sound(stereo_arr)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.base_size = (30, 20)
        self.image = pygame.Surface(self.base_size, pygame.SRCALPHA)
        self.draw_fish()
        self.rect = self.image.get_rect()
        self.rect.x = 50
        self.rect.y = SCREEN_HEIGHT // 2
        self.base_speed = 5
        self.speed = 5

        # Animation for fin wiggling
        self.animation_timer = 0
        self.facing_right = True  # Fish faces right by default

        # Lives and invincibility for math riddles
        self.lives = 3
        self.max_lives = 3
        self.invincibility_timer = 0  # Frames of invincibility after escape
        self.invincibility_duration = 180  # 3 seconds at 60 FPS
        self.flash_timer = 0

        # Power-up effects
        self.speed_boost_timer = 0
        self.invincibility_powerup_timer = 0
    
    def draw_fish(self):
        """Static fish drawing for initialization"""
        # This is just for initial setup - animated version will replace it
        pass

    def update(self, keys):
        # Update animation timer
        self.animation_timer += 1

        # Update invincibility timer
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
            self.flash_timer += 1

        # Update power-up timers
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= 1
            self.speed = int(self.base_speed * 1.5)
        else:
            self.speed = self.base_speed

        if self.invincibility_powerup_timer > 0:
            self.invincibility_powerup_timer -= 1

        # Track if player is moving for sound effects and animation
        is_moving = False

        # Determine facing direction based on movement
        if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and self.rect.right < SCREEN_WIDTH:
            self.facing_right = True
            self.rect.x += self.speed
            is_moving = True
        if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and self.rect.left > 0:
            self.facing_right = False
            self.rect.x -= self.speed
            is_moving = True

        # Vertical movement (doesn't change facing direction)
        if (keys[pygame.K_UP] or keys[pygame.K_w]) and self.rect.top > 0:
            self.rect.y -= self.speed
            is_moving = True
        if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and self.rect.bottom < SCREEN_HEIGHT:
            self.rect.y += self.speed
            is_moving = True

        # Adjust animation speed based on movement
        if is_moving:
            # Fish wiggle faster when swimming
            self.animation_timer += 0.5  # Increase animation speed when moving

        # Redraw fish with animated fins
        self.draw_animated_fish()

        return is_moving

    def is_invulnerable(self):
        """Check if player is currently invulnerable"""
        return self.invincibility_timer > 0 or self.invincibility_powerup_timer > 0

    def start_invincibility(self):
        """Start post-escape invincibility"""
        self.invincibility_timer = self.invincibility_duration
        self.flash_timer = 0

    def lose_life(self):
        """Lose a life, returns True if still alive"""
        self.lives -= 1
        return self.lives > 0

    def reset_lives(self):
        """Reset lives to max"""
        self.lives = self.max_lives

    def apply_speed_boost(self, duration_frames=600):
        """Apply speed boost power-up (default 10 seconds)"""
        self.speed_boost_timer = duration_frames

    def apply_invincibility_powerup(self, duration_frames=300):
        """Apply invincibility power-up (default 5 seconds)"""
        self.invincibility_powerup_timer = duration_frames

    def draw_animated_fish(self):
        """Draw the fish with animated fins and invincibility effect"""
        # Clear the surface first
        self.image.fill((0, 0, 0, 0))  # Transparent

        # Fish colors - flash when invincible
        if self.is_invulnerable() and (self.flash_timer // 5) % 2 == 0:
            FISH_ORANGE = (255, 255, 255)  # White flash
            FISH_YELLOW = (200, 200, 255)
            FISH_LIGHT_ORANGE = (255, 255, 200)
        else:
            FISH_ORANGE = (255, 165, 0)
            FISH_YELLOW = (255, 215, 0)
            FISH_LIGHT_ORANGE = (255, 200, 100)

        # Calculate wiggle offsets based on animation timer
        base_intensity = 2.5
        time_factor = self.animation_timer * 0.1

        tail_wiggle = int(base_intensity * math.sin(time_factor * 2.5))
        top_fin_wiggle = int(base_intensity * 0.7 * math.sin(time_factor * 1.8 + math.pi/4))
        bottom_fin_wiggle = int(base_intensity * 0.6 * math.sin(time_factor * 2.1 + math.pi/2))
        body_sway = int(0.8 * math.sin(time_factor * 1.5))

        if self.facing_right:
            body_y = 4 + body_sway
            pygame.draw.ellipse(self.image, FISH_ORANGE, (8, body_y, 20, 12))
            pygame.draw.ellipse(self.image, FISH_LIGHT_ORANGE, (10, body_y + 2, 16, 8))

            tail_base_y = 10 + body_sway
            tail_points = [
                (2, tail_base_y + tail_wiggle),
                (8, tail_base_y - 4 + tail_wiggle//2),
                (8, tail_base_y + 4 - tail_wiggle//2)
            ]
            pygame.draw.polygon(self.image, FISH_YELLOW, tail_points)

            top_fin_base_y = body_y
            top_fin_points = [
                (14, top_fin_base_y + top_fin_wiggle),
                (18, top_fin_base_y - 3 + top_fin_wiggle),
                (22, top_fin_base_y + top_fin_wiggle)
            ]
            pygame.draw.polygon(self.image, FISH_YELLOW, top_fin_points)

            bottom_fin_base_y = body_y + 12
            bottom_fin_points = [
                (14, bottom_fin_base_y + bottom_fin_wiggle),
                (18, bottom_fin_base_y + 3 + bottom_fin_wiggle),
                (22, bottom_fin_base_y + bottom_fin_wiggle)
            ]
            pygame.draw.polygon(self.image, FISH_YELLOW, bottom_fin_points)

            eye_y = body_y + 4
            pygame.draw.circle(self.image, BLACK, (22, eye_y), 2)
            pygame.draw.circle(self.image, WHITE, (23, eye_y - 1), 1)

        else:
            body_y = 4 + body_sway
            pygame.draw.ellipse(self.image, FISH_ORANGE, (2, body_y, 20, 12))
            pygame.draw.ellipse(self.image, FISH_LIGHT_ORANGE, (4, body_y + 2, 16, 8))

            tail_base_y = 10 + body_sway
            tail_points = [
                (28, tail_base_y + tail_wiggle),
                (22, tail_base_y - 4 + tail_wiggle//2),
                (22, tail_base_y + 4 - tail_wiggle//2)
            ]
            pygame.draw.polygon(self.image, FISH_YELLOW, tail_points)

            top_fin_base_y = body_y
            top_fin_points = [
                (16, top_fin_base_y + top_fin_wiggle),
                (12, top_fin_base_y - 3 + top_fin_wiggle),
                (8, top_fin_base_y + top_fin_wiggle)
            ]
            pygame.draw.polygon(self.image, FISH_YELLOW, top_fin_points)

            bottom_fin_base_y = body_y + 12
            bottom_fin_points = [
                (16, bottom_fin_base_y + bottom_fin_wiggle),
                (12, bottom_fin_base_y + 3 + bottom_fin_wiggle),
                (8, bottom_fin_base_y + bottom_fin_wiggle)
            ]
            pygame.draw.polygon(self.image, FISH_YELLOW, bottom_fin_points)

            eye_y = body_y + 4
            pygame.draw.circle(self.image, BLACK, (8, eye_y), 2)
            pygame.draw.circle(self.image, WHITE, (7, eye_y - 1), 1)

# ...existing code...

class Shark(pygame.sprite.Sprite):
    def __init__(self, x, y, speed):
        super().__init__()
        self.image = pygame.Surface((50, 30), pygame.SRCALPHA)
        self.draw_shark()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed

        # For more natural movement
        self.target_pos = (random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))
        self.oscillation_phase = random.uniform(0, 2 * math.pi)
        self.oscillation_speed = random.uniform(0.01, 0.03)
        self.oscillation_amplitude = random.randint(8, 18)
        self.smooth_direction = [random.uniform(-1, 1), random.uniform(-1, 1)]

        self.direction_x = random.choice([-1, 0, 1])
        self.direction_y = random.choice([-1, 1])
        self.direction_z = 0
        self.base_size = (50, 30)
        self.current_scale = 1.0
        self.min_scale = 0.5
        self.max_scale = 1.5
        self.direction_change_timer = random.randint(60, 180)
        self.animation_timer = 0
        self.fin_wiggle_offset = 0
        self.facing_right = True
        self.current_level = 1

    def set_level(self, level):
        self.current_level = level
        if level >= 10:
            self.direction_z = random.choice([-1, 0, 1])
            self.current_scale = random.uniform(self.min_scale, self.max_scale)

    def draw_shark(self):
        SHARK_GRAY = (70, 70, 70)
        pygame.draw.ellipse(self.image, SHARK_GRAY, (5, 8, 35, 14))
        pygame.draw.circle(self.image, BLACK, (12, 12), 2)
        mouth_points = [(8, 18), (12, 20), (16, 18)]
        pygame.draw.polygon(self.image, BLACK, mouth_points)

    def draw_animated_shark(self, surface):
        temp_surface = pygame.Surface(self.base_size, pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))
        SHARK_GRAY = (70, 70, 70)
        SHARK_DARK = (50, 50, 50)
        movement_speed = abs(self.smooth_direction[0]) + abs(self.smooth_direction[1])
        speed_multiplier = 1 + (movement_speed * 0.5)
        wiggle_intensity = 3
        tail_wiggle = int(wiggle_intensity * math.sin(self.animation_timer * 0.3 * speed_multiplier))
        dorsal_wiggle = int(wiggle_intensity * 0.5 * math.sin(self.animation_timer * 0.25 * speed_multiplier))
        side_wiggle = int(wiggle_intensity * 0.7 * math.sin(self.animation_timer * 0.35 * speed_multiplier))

        if self.facing_right:
            pygame.draw.ellipse(temp_surface, SHARK_GRAY, (5, 8, 35, 14))
            tail_points = [
                (2, 15 + tail_wiggle),
                (8, 10 + tail_wiggle//2),
                (8, 20 - tail_wiggle//2)
            ]
            pygame.draw.polygon(temp_surface, SHARK_GRAY, tail_points)
            dorsal_points = [
                (15, 8 + dorsal_wiggle),
                (20, 2 + dorsal_wiggle),
                (25, 8 + dorsal_wiggle)
            ]
            pygame.draw.polygon(temp_surface, SHARK_DARK, dorsal_points)
            side_fin_points = [
                (10, 22 + side_wiggle),
                (15, 25 + side_wiggle),
                (20, 22 + side_wiggle)
            ]
            pygame.draw.polygon(temp_surface, SHARK_DARK, side_fin_points)
            pygame.draw.circle(temp_surface, BLACK, (32, 12), 2)
            mouth_points = [(35, 18), (40, 20), (44, 18)]
            pygame.draw.polygon(temp_surface, BLACK, mouth_points)
        else:
            pygame.draw.ellipse(temp_surface, SHARK_GRAY, (10, 8, 35, 14))
            tail_points = [
                (47, 15 + tail_wiggle),
                (42, 10 + tail_wiggle//2),
                (42, 20 - tail_wiggle//2)
            ]
            pygame.draw.polygon(temp_surface, SHARK_GRAY, tail_points)
            dorsal_points = [
                (25, 8 + dorsal_wiggle),
                (30, 2 + dorsal_wiggle),
                (35, 8 + dorsal_wiggle)
            ]
            pygame.draw.polygon(temp_surface, SHARK_DARK, dorsal_points)
            side_fin_points = [
                (30, 22 + side_wiggle),
                (35, 25 + side_wiggle),
                (40, 22 + side_wiggle)
            ]
            pygame.draw.polygon(temp_surface, SHARK_DARK, side_fin_points)
            pygame.draw.circle(temp_surface, BLACK, (18, 12), 2)
            mouth_points = [(6, 18), (10, 20), (15, 18)]
            pygame.draw.polygon(temp_surface, BLACK, mouth_points)

        self.image.fill((0, 0, 0, 0))
        if self.current_level >= 10:
            scaled_width = int(self.base_size[0] * self.current_scale)
            scaled_height = int(self.base_size[1] * self.current_scale)
            scaled_surface = pygame.transform.scale(temp_surface, (scaled_width, scaled_height))
            offset_x = (self.base_size[0] - scaled_width) // 2
            offset_y = (self.base_size[1] - scaled_height) // 2
            self.image.blit(scaled_surface, (offset_x, offset_y))
        else:
            self.image.blit(temp_surface, (0, 0))

    def update(self, player_pos=None):
        self.animation_timer += 1

        # Smoothly steer toward the player, but with randomness for nature
        if player_pos:
            px, py = player_pos
            sx, sy = self.rect.center
            dx = px - sx
            dy = py - sy
            dist = math.hypot(dx, dy)
            if dist > 1:
                # Normalize and add some randomness
                steer_x = dx / dist + random.uniform(-0.2, 0.2)
                steer_y = dy / dist + random.uniform(-0.2, 0.2)
                # Blend with previous direction for smoothness
                self.smooth_direction[0] = self.smooth_direction[0] * 0.92 + steer_x * 0.08
                self.smooth_direction[1] = self.smooth_direction[1] * 0.92 + steer_y * 0.08
                # Clamp to [-1, 1]
                norm = math.hypot(self.smooth_direction[0], self.smooth_direction[1])
                if norm > 1:
                    self.smooth_direction[0] /= norm
                    self.smooth_direction[1] /= norm
            # Face toward the player
            self.facing_right = px > sx

        # Oscillate movement for more natural swimming
        self.oscillation_phase += self.oscillation_speed
        osc = math.sin(self.oscillation_phase) * self.oscillation_amplitude
        move_x = self.smooth_direction[0] * self.speed + osc * 0.1
        move_y = self.smooth_direction[1] * self.speed + osc * 0.05

        self.rect.x += int(move_x)
        self.rect.y += int(move_y)

        # Multi-dimensional movement for level 10+
        if self.current_level >= 10:
            depth_speed = 0.01
            self.current_scale += depth_speed * self.direction_z
            if self.current_scale <= self.min_scale:
                self.current_scale = self.min_scale
                self.direction_z = 1
            elif self.current_scale >= self.max_scale:
                self.current_scale = self.max_scale
                self.direction_z = -1

        # Bounce off screen boundaries
        if self.rect.left <= 0 or self.rect.right >= SCREEN_WIDTH:
            self.smooth_direction[0] *= -1
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.smooth_direction[1] *= -1

        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

        # Occasionally pick a new random oscillation
        if random.randint(1, 180) == 1:
            self.oscillation_speed = random.uniform(0.01, 0.03)
            self.oscillation_amplitude = random.randint(8, 18)

        # Redraw shark with animated fins
        self.draw_animated_shark(None)

# ...existing code..    
    def change_direction(self):
        """Randomly change the shark's movement direction"""
        # 70% chance to change direction, 30% chance to keep current direction
        if random.random() < 0.7:
            self.direction_x = random.choice([-1, 0, 1])
            self.direction_y = random.choice([-1, 1])
            
            # For level 10+, also change Z direction (depth)
            if self.current_level >= 10:
                self.direction_z = random.choice([-1, 0, 1])
            
            # Prevent sharks from being completely stationary
            if self.direction_x == 0 and self.direction_y == 0:
                self.direction_y = random.choice([-1, 1])


class Shell(pygame.sprite.Sprite):
    """Collectible shell currency sprite"""

    def __init__(self, x, y):
        super().__init__()
        self.base_size = (20, 18)
        self.image = pygame.Surface(self.base_size, pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        # Animation
        self.animation_timer = random.randint(0, 60)
        self.bob_offset = 0
        self.sparkle_timer = 0

        self.draw_shell()

    def draw_shell(self):
        """Draw an animated shell sprite"""
        self.image.fill((0, 0, 0, 0))

        # Shell colors
        SHELL_MAIN = (255, 220, 180)
        SHELL_DARK = (210, 170, 130)
        SHELL_LIGHT = (255, 240, 220)
        SHELL_PINK = (255, 200, 200)

        # Main shell body (spiral)
        center_x, center_y = 10, 10

        # Outer shell curve
        pygame.draw.ellipse(self.image, SHELL_MAIN, (2, 4, 16, 12))
        pygame.draw.ellipse(self.image, SHELL_DARK, (2, 4, 16, 12), 1)

        # Shell ridges (darker lines)
        for i in range(4):
            start_x = 4 + i * 3
            pygame.draw.line(self.image, SHELL_DARK, (start_x, 6), (start_x + 2, 14), 1)

        # Highlight
        pygame.draw.arc(self.image, SHELL_LIGHT, (3, 5, 8, 6), 0.5, 2.5, 2)

        # Inner spiral hint
        pygame.draw.arc(self.image, SHELL_PINK, (6, 7, 6, 5), 0, 3.14, 1)

        # Add sparkle effect
        if self.sparkle_timer > 0:
            sparkle_alpha = int(255 * (self.sparkle_timer / 30))
            sparkle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(sparkle_surf, (255, 255, 255, sparkle_alpha), (2, 2), 2)
            self.image.blit(sparkle_surf, (14, 2))

    def update(self):
        """Update shell animation"""
        self.animation_timer += 1

        # Bobbing motion
        self.bob_offset = int(2 * math.sin(self.animation_timer * 0.08))
        self.rect.y += self.bob_offset - int(2 * math.sin((self.animation_timer - 1) * 0.08))

        # Sparkle effect
        if self.sparkle_timer > 0:
            self.sparkle_timer -= 1
        elif random.random() < 0.02:  # 2% chance to sparkle
            self.sparkle_timer = 30

        self.draw_shell()


class ShellManager:
    """Manages shell spawning and collection"""

    def __init__(self, shells_group, screen_width, screen_height):
        self.shells = shells_group
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.spawn_timer = 0
        self.spawn_interval = random.randint(300, 600)  # 5-10 seconds at 60 FPS
        self.max_shells = 5

    def update(self, player, sharks, gates):
        """Update shell spawning and check for collection"""
        self.spawn_timer += 1

        # Spawn new shells periodically
        if self.spawn_timer >= self.spawn_interval and len(self.shells) < self.max_shells:
            self.spawn_shell(sharks, gates)
            self.spawn_timer = 0
            self.spawn_interval = random.randint(300, 600)

        # Update existing shells
        for shell in self.shells:
            shell.update()

        # Check for collection
        collected = pygame.sprite.spritecollide(player, self.shells, True)
        return len(collected)

    def spawn_shell(self, sharks, gates):
        """Spawn a new shell at a safe position"""
        max_attempts = 20

        for _ in range(max_attempts):
            # Random position avoiding edges
            x = random.randint(150, self.screen_width - 200)
            y = random.randint(50, self.screen_height - 50)

            # Create temporary rect to check collisions
            test_rect = pygame.Rect(x, y, 20, 18)

            # Check if position is safe (not overlapping sharks or gates)
            safe = True
            for shark in sharks:
                if test_rect.inflate(30, 30).colliderect(shark.rect):
                    safe = False
                    break

            if safe and gates:
                for gate in gates:
                    if test_rect.inflate(30, 30).colliderect(gate.rect):
                        safe = False
                        break

            if safe:
                shell = Shell(x, y)
                self.shells.add(shell)
                return


class PowerUpManager:
    """Tracks active power-ups and their effects"""

    POWER_UPS = {
        'speed_boost': {'name': 'Speed Boost', 'cost': 5, 'duration': 600, 'icon': '⚡'},
        'invincibility': {'name': 'Invincibility', 'cost': 10, 'duration': 300, 'icon': '🛡️'},
        'shark_skip': {'name': 'Shark Skip', 'cost': 8, 'duration': 0, 'icon': '🦈'}  # Instant effect
    }

    def __init__(self):
        self.active_powerups = {}
        self.shark_skip_available = False

    def purchase_powerup(self, powerup_type, player):
        """Apply a purchased power-up"""
        if powerup_type not in self.POWER_UPS:
            return False

        powerup = self.POWER_UPS[powerup_type]

        if powerup_type == 'speed_boost':
            player.apply_speed_boost(powerup['duration'])
            self.active_powerups['speed_boost'] = powerup['duration']
        elif powerup_type == 'invincibility':
            player.apply_invincibility_powerup(powerup['duration'])
            self.active_powerups['invincibility'] = powerup['duration']
        elif powerup_type == 'shark_skip':
            self.shark_skip_available = True

        return True

    def update(self):
        """Update power-up timers"""
        expired = []
        for powerup, timer in self.active_powerups.items():
            self.active_powerups[powerup] = timer - 1
            if self.active_powerups[powerup] <= 0:
                expired.append(powerup)

        for powerup in expired:
            del self.active_powerups[powerup]

    def use_shark_skip(self):
        """Use shark skip if available"""
        if self.shark_skip_available:
            self.shark_skip_available = False
            return True
        return False

    def get_active_powerups(self):
        """Get list of active power-ups for HUD display"""
        return list(self.active_powerups.keys())


class ShopOverlay:
    """Between-level shop interface"""

    def __init__(self, screen_width, screen_height, user_manager):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.user_manager = user_manager
        self.active = False
        self.selected_item = None
        self.purchase_problem = None
        self.user_input = ""
        self.result_message = ""
        self.result_timer = 0

        # Shop items
        self.items = [
            {'key': '1', 'type': 'speed_boost', 'name': 'Speed Boost', 'cost': 5, 'desc': 'Swim 50% faster for 10 seconds'},
            {'key': '2', 'type': 'invincibility', 'name': 'Invincibility', 'cost': 10, 'desc': 'Cannot be caught for 5 seconds'},
            {'key': '3', 'type': 'shark_skip', 'name': 'Shark Skip', 'cost': 8, 'desc': 'Remove one shark next level'}
        ]

        # Overlay dimensions
        self.overlay_width = 550
        self.overlay_height = 400
        self.overlay_x = (screen_width - self.overlay_width) // 2
        self.overlay_y = (screen_height - self.overlay_height) // 2

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.item_font = pygame.font.Font(None, 32)
        self.info_font = pygame.font.Font(None, 28)

    def resize(self, screen_width, screen_height):
        """Handle screen resize"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.overlay_x = (screen_width - self.overlay_width) // 2
        self.overlay_y = (screen_height - self.overlay_height) // 2

    def show(self):
        """Show the shop overlay"""
        self.active = True
        self.selected_item = None
        self.purchase_problem = None
        self.user_input = ""
        self.result_message = ""
        self.result_timer = 0

    def hide(self):
        """Hide the shop overlay"""
        self.active = False

    def handle_event(self, event, math_generator):
        """Handle shop input, returns purchased power-up type or None"""
        if not self.active:
            return None, False  # (powerup_type, skip_shop)

        if self.result_timer > 0:
            return None, False

        if self.purchase_problem:
            # In purchase confirmation mode
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    return self.check_purchase_answer()
                elif event.key == pygame.K_BACKSPACE:
                    self.user_input = self.user_input[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.purchase_problem = None
                    self.selected_item = None
                    self.user_input = ""
                elif event.unicode.isdigit() and len(self.user_input) < 3:
                    self.user_input += event.unicode
            return None, False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE or event.key == pygame.K_ESCAPE:
                # Skip shop
                return None, True

            # Select item by number
            for item in self.items:
                if event.unicode == item['key']:
                    shells = self.user_manager.get_user_shells()
                    if shells >= item['cost']:
                        # Can afford - create purchase problem
                        self.selected_item = item
                        self.purchase_problem = math_generator.generate_shop_problem(shells, item['cost'])
                        self.user_input = ""
                    else:
                        self.result_message = f"Not enough shells! Need {item['cost'] - shells} more."
                        self.result_timer = 90
                    return None, False

        return None, False

    def check_purchase_answer(self):
        """Check if purchase math answer is correct"""
        if not self.user_input or not self.purchase_problem:
            return None, False

        try:
            user_answer = int(self.user_input)
            correct = user_answer == self.purchase_problem['answer']

            if correct:
                # Complete purchase
                self.user_manager.spend_shells(self.selected_item['cost'])
                self.user_manager.record_math_problem(True)
                self.result_message = f"Purchased {self.selected_item['name']}!"
                self.result_timer = 90
                powerup_type = self.selected_item['type']
                self.purchase_problem = None
                self.selected_item = None
                self.user_input = ""
                return powerup_type, False
            else:
                self.user_manager.record_math_problem(False)
                self.result_message = f"Wrong! The answer was {self.purchase_problem['answer']}"
                self.result_timer = 90
                self.purchase_problem = None
                self.selected_item = None
                self.user_input = ""
                return None, False
        except ValueError:
            return None, False

    def update(self):
        """Update shop state"""
        if self.result_timer > 0:
            self.result_timer -= 1

    def draw(self, surface):
        """Draw the shop overlay"""
        if not self.active:
            return

        # Semi-transparent background
        overlay_bg = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay_bg.fill((0, 0, 0, 180))
        surface.blit(overlay_bg, (0, 0))

        # Main shop box
        box_color = (60, 40, 80)
        border_color = (150, 100, 180)
        pygame.draw.rect(surface, box_color,
                        (self.overlay_x, self.overlay_y, self.overlay_width, self.overlay_height),
                        border_radius=15)
        pygame.draw.rect(surface, border_color,
                        (self.overlay_x, self.overlay_y, self.overlay_width, self.overlay_height),
                        3, border_radius=15)

        # Title
        title_text = "OCEAN SHOP"
        title_surface = self.title_font.render(title_text, True, (255, 220, 100))
        title_rect = title_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                     self.overlay_y + 40))
        surface.blit(title_surface, title_rect)

        # Shell count
        shells = self.user_manager.get_user_shells()
        shells_text = f"You have: {shells} shells"
        shells_surface = self.item_font.render(shells_text, True, YELLOW)
        shells_rect = shells_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                       self.overlay_y + 80))
        surface.blit(shells_surface, shells_rect)

        if self.result_timer > 0:
            # Show result message
            result_color = LIGHT_GREEN if "Purchased" in self.result_message else RED
            result_surface = self.item_font.render(self.result_message, True, result_color)
            result_rect = result_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                           self.overlay_y + 200))
            surface.blit(result_surface, result_rect)
        elif self.purchase_problem:
            # Show purchase problem
            problem_surface = self.item_font.render(self.purchase_problem['text'], True, WHITE)
            problem_rect = problem_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                             self.overlay_y + 150))
            surface.blit(problem_surface, problem_rect)

            # User input
            input_display = self.user_input if self.user_input else "_"
            input_surface = self.title_font.render(input_display, True, YELLOW)
            input_rect = input_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                         self.overlay_y + 200))
            surface.blit(input_surface, input_rect)

            # Instructions
            instr_text = "Type answer and press ENTER (ESC to cancel)"
            instr_surface = self.info_font.render(instr_text, True, (180, 180, 180))
            instr_rect = instr_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                         self.overlay_y + 250))
            surface.blit(instr_surface, instr_rect)
        else:
            # Show items
            y_offset = 120
            for item in self.items:
                item_y = self.overlay_y + y_offset

                # Item box
                item_rect = pygame.Rect(self.overlay_x + 30, item_y, self.overlay_width - 60, 60)
                can_afford = shells >= item['cost']
                item_color = (50, 70, 90) if can_afford else (40, 40, 50)
                pygame.draw.rect(surface, item_color, item_rect, border_radius=8)
                pygame.draw.rect(surface, (100, 130, 160) if can_afford else (60, 60, 70),
                               item_rect, 2, border_radius=8)

                # Key indicator
                key_text = f"[{item['key']}]"
                key_color = YELLOW if can_afford else (100, 100, 100)
                key_surface = self.item_font.render(key_text, True, key_color)
                surface.blit(key_surface, (item_rect.x + 10, item_y + 5))

                # Item name and cost
                name_text = f"{item['name']} - {item['cost']} shells"
                name_color = WHITE if can_afford else (100, 100, 100)
                name_surface = self.item_font.render(name_text, True, name_color)
                surface.blit(name_surface, (item_rect.x + 60, item_y + 5))

                # Description
                desc_surface = self.info_font.render(item['desc'], True,
                                                     (180, 180, 180) if can_afford else (80, 80, 80))
                surface.blit(desc_surface, (item_rect.x + 60, item_y + 32))

                y_offset += 70

        # Skip instruction
        skip_text = "Press SPACE to skip shop"
        skip_surface = self.info_font.render(skip_text, True, (150, 150, 150))
        skip_rect = skip_surface.get_rect(center=(self.overlay_x + self.overlay_width // 2,
                                                   self.overlay_y + self.overlay_height - 25))
        surface.blit(skip_surface, skip_rect)


class MathGate(pygame.sprite.Sprite):
    """Vertical barrier requiring math answer to pass"""

    def __init__(self, x, screen_height, level, operation="addition"):
        super().__init__()
        self.gate_width = 20
        self.gate_height = screen_height
        self.image = pygame.Surface((self.gate_width, self.gate_height), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = 0

        self.locked = True
        self.problem = None
        self.level = level
        self.operation = operation
        self.animation_timer = 0
        self.unlock_animation = 0

        # Generate problem
        self.math_generator = MathProblemGenerator(level)
        self.generate_problem()

        self.draw_gate()

    def generate_problem(self):
        """Generate a new math problem for this gate"""
        self.problem = self.math_generator.generate_random_problem(self.operation)

    def draw_gate(self):
        """Draw the gate sprite"""
        self.image.fill((0, 0, 0, 0))

        if self.locked:
            # Locked gate - solid barrier with glow
            gate_color = (100, 60, 140)
            glow_intensity = int(30 * (1 + math.sin(self.animation_timer * 0.1)))

            # Glow effect
            for i in range(3):
                glow_alpha = 50 - i * 15 + glow_intensity // 3
                glow_rect = pygame.Rect(i * 2, 0, self.gate_width - i * 4, self.gate_height)
                glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                glow_surf.fill((*gate_color, glow_alpha))
                self.image.blit(glow_surf, glow_rect.topleft)

            # Main gate
            pygame.draw.rect(self.image, gate_color, (4, 0, self.gate_width - 8, self.gate_height))

            # Gate bars
            bar_color = (130, 90, 170)
            for y in range(0, self.gate_height, 30):
                pygame.draw.rect(self.image, bar_color, (2, y, self.gate_width - 4, 3))

            # Lock symbols
            for y in range(50, self.gate_height - 50, 100):
                pygame.draw.circle(self.image, (200, 150, 220), (self.gate_width // 2, y), 6)
                pygame.draw.circle(self.image, gate_color, (self.gate_width // 2, y), 4)

        else:
            # Unlocked gate - fading out
            alpha = max(0, 200 - self.unlock_animation * 10)
            if alpha > 0:
                fade_surf = pygame.Surface((self.gate_width, self.gate_height), pygame.SRCALPHA)
                fade_color = (100, 200, 100, alpha)
                pygame.draw.rect(fade_surf, fade_color, (4, 0, self.gate_width - 8, self.gate_height))
                self.image.blit(fade_surf, (0, 0))

    def update(self):
        """Update gate animation"""
        self.animation_timer += 1

        if not self.locked:
            self.unlock_animation += 1
            if self.unlock_animation > 30:
                self.kill()  # Remove gate after unlock animation

        self.draw_gate()

    def unlock(self):
        """Unlock the gate"""
        self.locked = False
        self.unlock_animation = 0

    def get_problem_text(self):
        """Get the problem text for display"""
        if self.problem:
            return self.problem['text']
        return ""

    def check_answer(self, answer):
        """Check if answer is correct"""
        if self.problem:
            return answer == self.problem['answer']
        return False


class NumberComparisonZone(pygame.sprite.Sprite):
    """Dual-path zone with number comparison challenge"""

    def __init__(self, x, screen_height, level):
        super().__init__()
        self.zone_width = 150
        self.zone_height = screen_height
        self.image = pygame.Surface((self.zone_width, self.zone_height), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = 0

        self.level = level
        self.animation_timer = 0

        # Generate comparison
        self.math_generator = MathProblemGenerator(level)
        self.generate_comparison()

        # Path rects (top half and bottom half)
        path_height = screen_height // 2 - 30
        self.top_path_rect = pygame.Rect(x, 30, self.zone_width, path_height)
        self.bottom_path_rect = pygame.Rect(x, screen_height // 2 + 30, self.zone_width, path_height)

        self.draw_zone()

    def generate_comparison(self):
        """Generate a new comparison problem"""
        self.problem = self.math_generator.generate_comparison()
        self.number_top = self.problem['a']
        self.number_bottom = self.problem['b']
        self.ask_bigger = self.problem['ask_bigger']
        self.correct_answer = self.problem['answer']

        # Determine which path is correct
        if self.ask_bigger:
            self.correct_path = 'top' if self.number_top > self.number_bottom else 'bottom'
        else:
            self.correct_path = 'top' if self.number_top < self.number_bottom else 'bottom'

    def draw_zone(self):
        """Draw the comparison zone"""
        self.image.fill((0, 0, 0, 0))

        # Zone background with transparency
        zone_surf = pygame.Surface((self.zone_width, self.zone_height), pygame.SRCALPHA)
        zone_surf.fill((80, 120, 160, 100))
        self.image.blit(zone_surf, (0, 0))

        # Divider line in middle
        pygame.draw.line(self.image, (200, 200, 200),
                        (0, self.zone_height // 2), (self.zone_width, self.zone_height // 2), 3)

        # Top path with number
        top_color = (100, 150, 200)
        pygame.draw.rect(self.image, top_color, (10, 50, self.zone_width - 20, 80), border_radius=10)

        # Bottom path with number
        bottom_color = (150, 100, 200)
        pygame.draw.rect(self.image, bottom_color,
                        (10, self.zone_height // 2 + 50, self.zone_width - 20, 80), border_radius=10)

        # Numbers
        number_font = pygame.font.Font(None, 64)
        top_num_surf = number_font.render(str(self.number_top), True, WHITE)
        top_num_rect = top_num_surf.get_rect(center=(self.zone_width // 2, 90))
        self.image.blit(top_num_surf, top_num_rect)

        bottom_num_surf = number_font.render(str(self.number_bottom), True, WHITE)
        bottom_num_rect = bottom_num_surf.get_rect(center=(self.zone_width // 2, self.zone_height // 2 + 90))
        self.image.blit(bottom_num_surf, bottom_num_rect)

        # Prompt at top
        prompt_font = pygame.font.Font(None, 24)
        prompt_text = "BIGGER" if self.ask_bigger else "SMALLER"
        prompt_surf = prompt_font.render(f"Swim through {prompt_text}!", True, YELLOW)
        prompt_rect = prompt_surf.get_rect(center=(self.zone_width // 2, 20))
        self.image.blit(prompt_surf, prompt_rect)

    def update(self):
        """Update zone animation"""
        self.animation_timer += 1
        # Could add animation effects here

    def check_player_path(self, player_rect):
        """Check which path player is taking, returns 'correct', 'wrong', or None"""
        # Check if player is in the zone horizontally
        if not (player_rect.right > self.rect.x and player_rect.left < self.rect.right):
            return None

        player_center_y = player_rect.centery

        # Determine which path player is in
        if player_center_y < self.zone_height // 2:
            chosen_path = 'top'
        else:
            chosen_path = 'bottom'

        if chosen_path == self.correct_path:
            return 'correct'
        else:
            return 'wrong'


class ParentalControlConsole:
    def __init__(self, user_manager):
        self.user_manager = user_manager
    
    def run_parental_console(self):
        """Run the parental control console interface"""
        print("\n" + "="*50)
        print("      PARENTAL CONTROL CONSOLE")
        print("="*50)
        print()
        
        # PIN verification
        attempts = 3
        while attempts > 0:
            pin = input(f"Enter parental PIN ({attempts} attempts remaining): ").strip()
            
            if self.user_manager.verify_parental_access(pin):
                print("Access granted")
                break
            else:
                attempts -= 1
                print("Incorrect PIN")
                if attempts == 0:
                    print("Access denied. Exiting parental console.")
                    return
        
        # Main parental menu
        while True:
            self.show_parental_menu()
            choice = input("Select option (1-8): ").strip()
            
            if choice == "1":
                self.view_playtime_status()
            elif choice == "2":
                self.modify_time_limits()
            elif choice == "3":
                self.activate_override()
            elif choice == "4":
                self.deactivate_override()
            elif choice == "5":
                self.reset_daily_time()
            elif choice == "6":
                self.change_pin()
            elif choice == "7":
                self.view_user_stats()
            elif choice == "8":
                print("Exiting parental console...")
                break
            else:
                print("Invalid option. Please try again.")
    
    def show_parental_menu(self):
        """Display the parental control menu"""
        print("\n" + "-"*40)
        print("PARENTAL CONTROL OPTIONS:")
        print("-"*40)
        print("1. View Playtime Status")
        print("2. Modify Time Limits")
        print("3. Activate Override (Allow Unlimited Play)")
        print("4. Deactivate Override")
        print("5. Reset Daily Playtime")
        print("6. Change Parental PIN")
        print("7. View User Statistics")
        print("8. Exit")
        print("-"*40)
    
    def view_playtime_status(self):
        """Show current playtime status"""
        today_time = self.user_manager.get_total_playtime_today()
        max_time = self.user_manager.get_max_playtime()
        remaining = max(0, max_time - today_time)
        override_active = self.user_manager.get_parental_override_active()
        
        print(f"\nPLAYTIME STATUS:")
        print(f"Today's playtime: {today_time:.1f} minutes")
        print(f"Daily limit: {max_time} minutes")
        print(f"Remaining time: {remaining:.1f} minutes")
        print(f"Parental override: {'ACTIVE' if override_active else 'INACTIVE'}")
        
        if override_active:
            override_data = self.user_manager.settings.get("parental_override", {})
            import time
            remaining_override = (override_data.get("expiry_time", 0) - time.time()) / 3600
            print(f"Override expires in: {remaining_override:.1f} hours")
    
    def modify_time_limits(self):
        """Modify daily time limits"""
        current_limit = self.user_manager.get_max_playtime()
        print(f"\nCurrent daily limit: {current_limit} minutes")
        
        try:
            new_limit = int(input("Enter new daily limit (5-480 minutes): "))
            if 5 <= new_limit <= 480:
                self.user_manager.set_max_playtime(new_limit)
                print(f"Daily limit updated to {new_limit} minutes")
            else:
                print("Invalid limit. Must be between 5 and 480 minutes.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    def activate_override(self):
        """Activate parental override"""
        if self.user_manager.get_parental_override_active():
            print("Override is already active.")
            return
        
        print("\nActivate Parental Override:")
        print("1. 1 hour")
        print("2. 2 hours") 
        print("3. 4 hours")
        print("4. Custom duration")
        
        choice = input("Select duration (1-4): ").strip()
        
        if choice == "1":
            self.user_manager.activate_parental_override(1)
        elif choice == "2":
            self.user_manager.activate_parental_override(2)
        elif choice == "3":
            self.user_manager.activate_parental_override(4)
        elif choice == "4":
            try:
                hours = float(input("Enter hours (0.1-24): "))
                if 0.1 <= hours <= 24:
                    self.user_manager.activate_parental_override(hours)
                else:
                    print("Invalid duration. Must be between 0.1 and 24 hours.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        else:
            print("Invalid option.")
    
    def deactivate_override(self):
        """Deactivate parental override"""
        if not self.user_manager.get_parental_override_active():
            print("No override is currently active.")
            return
        
        confirm = input("Deactivate override? (y/N): ").strip().lower()
        if confirm == 'y':
            self.user_manager.deactivate_parental_override()
        else:
            print("Override remains active.")
    
    def reset_daily_time(self):
        """Reset today's playtime"""
        confirm = input("Reset today's playtime to 0? (y/N): ").strip().lower()
        if confirm == 'y':
            self.user_manager.reset_daily_playtime()
        else:
            print("Daily playtime not reset.")
    
    def change_pin(self):
        """Change parental PIN"""
        current_pin = input("Enter current PIN: ").strip()
        if not self.user_manager.verify_parental_access(current_pin):
            print("Incorrect current PIN.")
            return
        
        new_pin = input("Enter new PIN (4-8 digits): ").strip()
        if len(new_pin) >= 4 and len(new_pin) <= 8 and new_pin.isdigit():
            confirm_pin = input("Confirm new PIN: ").strip()
            if new_pin == confirm_pin:
                self.user_manager.set_parental_pin(new_pin)
                print("PIN updated successfully.")
            else:
                print("PINs do not match.")
        else:
            print("PIN must be 4-8 digits.")
    
    def view_user_stats(self):
        """View user statistics"""
        current_user = self.user_manager.settings.get("last_user", "Unknown")
        print(f"\nUSER STATISTICS for {current_user}:")
        
        # Get user data
        users_data = {}
        try:
            with open('users.json', 'r') as f:
                users_data = json.load(f)
        except:
            pass
        
        user_data = users_data.get(current_user, {})
        print(f"Games played: {user_data.get('games_played', 0)}")
        print(f"Highest level: {user_data.get('highest_level', 1)}")
        print(f"Total score: {user_data.get('total_score', 0)}")
        
        # Show recent playtime
        daily_data = self.user_manager.settings.get("daily_playtime", {})
        print(f"\nRECENT PLAYTIME:")
        for date, minutes in sorted(daily_data.items())[-7:]:  # Last 7 days
            print(f"{date}: {minutes:.1f} minutes")

class Game:
    def __init__(self):
        global SCREEN_WIDTH, SCREEN_HEIGHT
        
        # Initialize display mode
        self.fullscreen = False  # Start in windowed mode by default
        self.windowed_width = 1024
        self.windowed_height = 768
        self.fullscreen_width = SCREEN_WIDTH
        self.fullscreen_height = SCREEN_HEIGHT
        
        # Start in windowed mode
        SCREEN_WIDTH = self.windowed_width
        SCREEN_HEIGHT = self.windowed_height
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Sharks and Minnows")
        self.clock = pygame.time.Clock()
        
        # Scale fonts based on screen size
        base_font_size = max(24, SCREEN_HEIGHT // 25)
        small_font_size = max(18, SCREEN_HEIGHT // 35)
        
        self.font = pygame.font.Font(None, base_font_size)
        self.small_font = pygame.font.Font(None, small_font_size)
        
        # Initialize audio and user management
        print("Initializing audio system...")
        self.audio_manager = AudioManager()
        self.user_manager = UserManager()
        
        # Initialize music manager
        print("Initializing music system...")
        self.music_manager = MusicManager(self.user_manager)
        
        # Initialize playtime manager
        print("Initializing playtime manager...")
        self.playtime_manager = PlaytimeManager(self.user_manager)
        
        # Check for previous user
        last_user = self.user_manager.get_last_user()
        preferred_fullscreen = self.user_manager.get_preferred_fullscreen()
        
        if last_user:
            # Welcome back returning user automatically
            print(f"Welcome back, {last_user}!")
            self.user_manager.register_user(last_user)
            self.game_state = "user_prompt"  # Ask if new user wants to play
            
            # Apply their fullscreen preference
            if preferred_fullscreen != self.fullscreen:
                self.toggle_fullscreen()
        else:
            # No previous user, go directly to name input
            self.game_state = "name_input"
            self.username_input = ""
            self.username_prompt = "Enter your name and press ENTER:"
        
        # Generate sound effects
        print("Generating sound effects...")
        self.roar_sound = SoundGenerator.generate_roar()
        self.success_sound = SoundGenerator.generate_success()
        self.swimming_sound = SoundGenerator.generate_swimming()
        self.correct_sound = SoundGenerator.generate_correct_answer()
        self.wrong_sound = SoundGenerator.generate_wrong_answer()
        self.shell_collect_sound = SoundGenerator.generate_shell_collect()
        self.gate_open_sound = SoundGenerator.generate_gate_open()
        print("Sound effects ready!")

        # Create ocean background
        print("Creating ocean background...")
        self.ocean_background = OceanBackground(SCREEN_WIDTH, SCREEN_HEIGHT)
        # Connect sound manager for volcano eruptions
        self.ocean_background.set_sound_manager(self)
        print("Ocean background ready!")

        # Initialize math systems
        print("Initializing math systems...")
        self.math_generator = MathProblemGenerator()
        self.math_riddle_overlay = MathRiddleOverlay(SCREEN_WIDTH, SCREEN_HEIGHT)
        print("Math systems ready!")

        # Don't start game until we have username
        self.player = None
        self.all_sprites = None
        self.sharks = None
        self.shells = None
        self.shell_manager = None
        self.math_gates = None
        self.comparison_zones = None
        self.power_up_manager = None
        self.shop_overlay = None
        self.level = 1

        # Parental PIN input for time limit override
        self.pin_input = ""
        self.pin_error_message = ""
        self.pin_attempts = 3
        
    def reset_game(self):
        self.player = Player()
        self.player.reset_lives()
        self.all_sprites = pygame.sprite.Group()
        self.sharks = pygame.sprite.Group()
        self.shells = pygame.sprite.Group()
        self.math_gates = pygame.sprite.Group()
        self.comparison_zones = pygame.sprite.Group()
        self.all_sprites.add(self.player)

        self.level = 1
        self.game_state = "playing"  # playing, caught, won, riddle, gate_puzzle, shop
        self.create_sharks()

        # Initialize math systems
        self.math_generator.set_level(self.level)
        self.shell_manager = ShellManager(self.shells, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.power_up_manager = PowerUpManager()
        self.shop_overlay = ShopOverlay(SCREEN_WIDTH, SCREEN_HEIGHT, self.user_manager)

        # Create math gates and comparison zones if enabled
        math_features = self.user_manager.get_math_features()
        if math_features.get("math_gates_enabled", True):
            self.create_math_gates()
        if math_features.get("comparison_zones_enabled", True):
            self.create_comparison_zones()

        # Initialize volcano system for level 1 (no volcanoes initially)
        self.ocean_background.update_level(self.level)
        
    def create_sharks(self):
        # Clear existing sharks
        for shark in self.sharks:
            shark.kill()

        # Create sharks based on level
        num_sharks = 2 + (self.level // 2)

        # Apply shark skip power-up if available
        if self.power_up_manager and self.power_up_manager.use_shark_skip():
            num_sharks = max(1, num_sharks - 1)

        for i in range(num_sharks):
            x = random.randint(200, 600)
            y = random.randint(50, SCREEN_HEIGHT - 50)
            speed = 2 + (self.level * 0.3)
            shark = Shark(x, y, speed)
            shark.set_level(self.level)  # Set the current level for movement complexity
            self.sharks.add(shark)
            self.all_sprites.add(shark)

    def create_math_gates(self):
        """Create math gates for the current level"""
        # Clear existing gates
        for gate in self.math_gates:
            gate.kill()

        # Number of gates based on level (1-2 gates)
        num_gates = min(2, 1 + self.level // 3)

        # Get operation preference
        operation = self.user_manager.get_math_operation()

        # Position gates between start and safe zone
        start_zone_width = max(100, SCREEN_WIDTH // 8)
        safe_zone_width = max(100, SCREEN_WIDTH // 8)
        available_width = SCREEN_WIDTH - start_zone_width - safe_zone_width - 100

        for i in range(num_gates):
            # Distribute gates evenly
            gate_x = start_zone_width + 100 + (i + 1) * (available_width // (num_gates + 1))
            gate = MathGate(gate_x, SCREEN_HEIGHT, self.level, operation)
            self.math_gates.add(gate)
            self.all_sprites.add(gate)

    def create_comparison_zones(self):
        """Create number comparison zones for the current level"""
        # Clear existing zones
        for zone in self.comparison_zones:
            zone.kill()

        # Only add comparison zones starting at level 2
        if self.level < 2:
            return

        # One comparison zone per level (max 1)
        start_zone_width = max(100, SCREEN_WIDTH // 8)
        safe_zone_width = max(100, SCREEN_WIDTH // 8)

        # Place zone roughly in the middle-right of the play area
        zone_x = SCREEN_WIDTH - safe_zone_width - 200
        zone = NumberComparisonZone(zone_x, SCREEN_HEIGHT, self.level)
        self.comparison_zones.add(zone)
        self.all_sprites.add(zone)

    def next_level(self):
        self.level += 1
        self.player.rect.x = 50
        self.player.rect.y = SCREEN_HEIGHT // 2
        self.math_generator.set_level(self.level)

        # Clear and recreate level elements
        self.shells.empty()
        self.math_gates.empty()
        self.comparison_zones.empty()

        self.create_sharks()

        # Create math features if enabled
        math_features = self.user_manager.get_math_features()
        if math_features.get("math_gates_enabled", True):
            self.create_math_gates()
        if math_features.get("comparison_zones_enabled", True):
            self.create_comparison_zones()

        # Update volcano system for new level
        self.ocean_background.update_level(self.level)

        self.game_state = "playing"
        
    def draw_text(self, text, font, color, x, y, center=False):
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surface, rect)
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        global SCREEN_WIDTH, SCREEN_HEIGHT
        
        self.fullscreen = not self.fullscreen
        print(f"Toggling to {'fullscreen' if self.fullscreen else 'windowed'} mode")
        
        try:
            if self.fullscreen:
                # Switch to fullscreen
                SCREEN_WIDTH = self.fullscreen_width
                SCREEN_HEIGHT = self.fullscreen_height
                print(f"Setting fullscreen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
            else:
                # Switch to windowed mode
                SCREEN_WIDTH = self.windowed_width
                SCREEN_HEIGHT = self.windowed_height
                print(f"Setting windowed: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # Rescale fonts for new screen size
            base_font_size = max(24, SCREEN_HEIGHT // 25)
            small_font_size = max(18, SCREEN_HEIGHT // 35)
            self.font = pygame.font.Font(None, base_font_size)
            self.small_font = pygame.font.Font(None, small_font_size)
            
            # Recreate ocean background for new screen size
            self.ocean_background = OceanBackground(SCREEN_WIDTH, SCREEN_HEIGHT)
            # Connect sound manager for volcano eruptions
            self.ocean_background.set_sound_manager(self)

            # Resize math overlays
            if self.math_riddle_overlay:
                self.math_riddle_overlay.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
            if self.shop_overlay:
                self.shop_overlay.resize(SCREEN_WIDTH, SCREEN_HEIGHT)

            # Save fullscreen preference
            self.user_manager.set_preferred_fullscreen(self.fullscreen)
            print("Fullscreen toggle completed successfully")
            
        except Exception as e:
            print(f"Error toggling fullscreen: {e}")
            # Revert the toggle if it failed
            self.fullscreen = not self.fullscreen
    
    def draw_username_input_screen(self):
        """Draw the username input screen"""
        # Title
        self.draw_text("SHARKS AND MINNOWS", self.font, WHITE, SCREEN_WIDTH // 2, 180, True)

        # Prompt
        self.draw_text(self.username_prompt, self.small_font, WHITE, SCREEN_WIDTH // 2, 260, True)

        # Input box
        input_box_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, 290, 300, 40)
        pygame.draw.rect(self.screen, WHITE, input_box_rect)
        pygame.draw.rect(self.screen, BLACK, input_box_rect, 2)

        # Username text
        if self.username_input:
            self.draw_text(self.username_input, self.font, BLACK, SCREEN_WIDTH // 2, 310, True)

        # Cursor (blinking effect)
        cursor_visible = (pygame.time.get_ticks() // 500) % 2
        if cursor_visible:
            cursor_x = SCREEN_WIDTH // 2 + len(self.username_input) * 10
            pygame.draw.line(self.screen, BLACK, (cursor_x, 295), (cursor_x, 325), 2)

        # Instructions
        self.draw_text("Type your name and press ENTER", self.small_font, WHITE, SCREEN_WIDTH // 2, 360, True)
        self.draw_text("Use BACKSPACE to delete", self.small_font, WHITE, SCREEN_WIDTH // 2, 385, True)

        # Math settings
        operation = self.user_manager.get_math_operation()
        op_display = {"addition": "Addition (+)", "subtraction": "Subtraction (-)", "both": "Both (+/-)"}
        timer_enabled = self.user_manager.get_math_timer_enabled()
        timer_status = "ON (15s)" if timer_enabled else "OFF"

        self.draw_text(f"Math: {op_display.get(operation, operation)} | Timer: {timer_status}", self.small_font, LIGHT_GREEN, SCREEN_WIDTH // 2, 430, True)
        self.draw_text("Press O: change operation | T: toggle timer", self.small_font, (180, 180, 180), SCREEN_WIDTH // 2, 455, True)

        self.draw_text("F11: Fullscreen | M: Music | O: Math Op | T: Timer | ESC: Quit", self.small_font, WHITE, SCREEN_WIDTH // 2, 500, True)
        self.draw_text("(Game starts in windowed mode)", self.small_font, WHITE, SCREEN_WIDTH // 2, 525, True)
    
    def draw_user_prompt_screen(self):
        """Draw the screen asking if new user wants to play"""
        # Title
        self.draw_text("SHARKS AND MINNOWS", self.font, WHITE, SCREEN_WIDTH // 2, 150, True)

        # Welcome back message
        last_user = self.user_manager.get_last_user()
        if last_user:
            self.draw_text(f"Welcome back, {last_user}!", self.font, WHITE, SCREEN_WIDTH // 2, 220, True)

            # User stats
            stats = self.user_manager.get_user_stats()
            if stats:
                self.draw_text(f"Your Best Level: {stats['highest_level']}", self.small_font, WHITE, SCREEN_WIDTH // 2, 270, True)
                self.draw_text(f"Games Played: {stats['games_played']}", self.small_font, WHITE, SCREEN_WIDTH // 2, 300, True)

        # Math settings
        operation = self.user_manager.get_math_operation()
        op_display = {"addition": "Addition (+)", "subtraction": "Subtraction (-)", "both": "Both (+/-)"}
        timer_enabled = self.user_manager.get_math_timer_enabled()
        timer_status = "ON (15s)" if timer_enabled else "OFF"

        self.draw_text(f"Math: {op_display.get(operation, operation)} | Timer: {timer_status}", self.small_font, LIGHT_GREEN, SCREEN_WIDTH // 2, 340, True)
        self.draw_text("Press O: change operation | T: toggle timer", self.small_font, (180, 180, 180), SCREEN_WIDTH // 2, 360, True)

        # Options
        self.draw_text("Press SPACE to continue as " + last_user, self.font, GREEN, SCREEN_WIDTH // 2, 410, True)
        self.draw_text("Press N for New User", self.font, YELLOW, SCREEN_WIDTH // 2, 450, True)

        # Instructions
        self.draw_text("F11: Fullscreen | M: Music | O: Math Op | T: Timer | ESC: Quit", self.small_font, WHITE, SCREEN_WIDTH // 2, 510, True)

    def draw_time_limit_screen(self):
        """Draw the time limit reached screen with parental PIN entry"""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((20, 20, 60))
        self.screen.blit(overlay, (0, 0))

        # Title
        self.draw_text("PLAYTIME LIMIT REACHED", self.font, RED, SCREEN_WIDTH // 2, 120, True)

        # Player info
        current_user = self.user_manager.current_user or "Player"
        max_time = self.user_manager.get_max_playtime()
        today_playtime = self.user_manager.get_total_playtime_today()

        self.draw_text(f"{current_user}, you've played {today_playtime:.0f} minutes today!", self.small_font, WHITE, SCREEN_WIDTH // 2, 180, True)
        self.draw_text(f"Daily limit: {max_time} minutes", self.small_font, WHITE, SCREEN_WIDTH // 2, 210, True)

        # Message
        self.draw_text("Time for a break! Come back tomorrow to play again.", self.small_font, YELLOW, SCREEN_WIDTH // 2, 260, True)

        # Parental override section
        self.draw_text("--- PARENT OVERRIDE ---", self.small_font, (150, 150, 150), SCREEN_WIDTH // 2, 320, True)
        self.draw_text("Enter parental PIN to unlock more playtime:", self.small_font, WHITE, SCREEN_WIDTH // 2, 350, True)

        # PIN input box
        pin_box_rect = pygame.Rect(SCREEN_WIDTH // 2 - 100, 380, 200, 40)
        pygame.draw.rect(self.screen, WHITE, pin_box_rect)
        pygame.draw.rect(self.screen, BLACK, pin_box_rect, 2)

        # PIN display (hidden as asterisks)
        pin_display = "*" * len(self.pin_input)
        if pin_display:
            self.draw_text(pin_display, self.font, BLACK, SCREEN_WIDTH // 2, 400, True)

        # Cursor (blinking effect)
        cursor_visible = (pygame.time.get_ticks() // 500) % 2
        if cursor_visible:
            cursor_x = SCREEN_WIDTH // 2 + len(self.pin_input) * 12
            pygame.draw.line(self.screen, BLACK, (cursor_x, 385), (cursor_x, 415), 2)

        # Error message
        if self.pin_error_message:
            self.draw_text(self.pin_error_message, self.small_font, RED, SCREEN_WIDTH // 2, 440, True)

        # Attempts remaining
        self.draw_text(f"Attempts remaining: {self.pin_attempts}", self.small_font, (180, 180, 180), SCREEN_WIDTH // 2, 470, True)

        # Instructions
        self.draw_text("Type PIN and press ENTER | BACKSPACE to delete", self.small_font, WHITE, SCREEN_WIDTH // 2, 510, True)
        self.draw_text("Press ESC to quit game", self.small_font, (150, 150, 150), SCREEN_WIDTH // 2, 540, True)

        # Default PIN hint for parents
        self.draw_text("(Default PIN: 1234 | Change via --parental flag)", self.small_font, (100, 100, 100), SCREEN_WIDTH // 2, 580, True)

    def game_over_audio(self):
        """Provide audio feedback when game ends"""
        import time
        time.sleep(2)  # Wait for roar to finish
        stats = self.user_manager.get_user_stats()
        if stats:
            if self.level == stats['highest_level']:
                self.audio_manager.speak(f"New personal best! You reached level {self.level}!")
            else:
                self.audio_manager.speak(f"Game over! You reached level {self.level}. Your best is level {stats['highest_level']}.")
    
    def level_complete_audio(self):
        """Provide audio feedback when level completes"""
        import time
        time.sleep(1)  # Wait for success sound
        self.audio_manager.speak(f"Excellent! Level {self.level} complete! Getting ready for level {self.level + 1}.")
        
    def run(self):
        running = True
        
        while running:
            self.clock.tick(FPS)
            
            # Update music
            self.music_manager.update()
            
            # Check playtime limit (only when not already at time limit screen)
            if self.game_state != "time_limit_reached" and self.playtime_manager.check_time_limit():
                self.game_state = "time_limit_reached"
                self.pin_input = ""
                self.pin_error_message = ""
                self.pin_attempts = 3
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.KEYDOWN:
                    # F11 key to toggle fullscreen
                    if event.key == pygame.K_F11:
                        print("F11 pressed - toggling fullscreen")
                        self.toggle_fullscreen()
                    # ESC key to quit game
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    # F12 key to change user
                    elif event.key == pygame.K_F12:
                        self.game_state = "name_input"
                        self.username_input = ""
                        self.username_prompt = "Enter your name and press ENTER:"
                    # M key to toggle music
                    elif event.key == pygame.K_m:
                        self.music_manager.toggle_music()
                    # O key to cycle math operation
                    elif event.key == pygame.K_o:
                        new_op = self.user_manager.cycle_math_operation()
                        print(f"Math operation set to: {new_op}")
                    # T key to toggle math timer (without Ctrl)
                    elif event.key == pygame.K_t and not pygame.key.get_pressed()[pygame.K_LCTRL]:
                        timer_on = self.user_manager.toggle_math_timer()
                        print(f"Math timer: {'ON' if timer_on else 'OFF'}")
                    # Plus/Minus keys for volume control (1% intervals)
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.music_manager.adjust_volume(0.01)  # 1% increase
                    elif event.key == pygame.K_MINUS:
                        self.music_manager.adjust_volume(-0.01)  # 1% decrease
                    # S key to skip track
                    elif event.key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
                        self.music_manager.skip_track()
                    # T key to adjust playtime limit
                    elif event.key == pygame.K_t and pygame.key.get_pressed()[pygame.K_LCTRL]:
                        current_limit = self.user_manager.get_max_playtime()
                        print(f"Current playtime limit: {current_limit} minutes")
                        print("Use Ctrl+Shift+T to increase limit by 5 minutes")
                        print("Use Ctrl+Alt+T to decrease limit by 5 minutes")
                    elif event.key == pygame.K_t and pygame.key.get_pressed()[pygame.K_LCTRL] and pygame.key.get_pressed()[pygame.K_LSHIFT]:
                        # Increase playtime limit
                        current_limit = self.user_manager.get_max_playtime()
                        new_limit = current_limit + 5
                        self.user_manager.set_max_playtime(new_limit)
                        print(f"Playtime limit increased to {new_limit} minutes")
                    elif event.key == pygame.K_t and pygame.key.get_pressed()[pygame.K_LCTRL] and pygame.key.get_pressed()[pygame.K_LALT]:
                        # Decrease playtime limit
                        current_limit = self.user_manager.get_max_playtime()
                        new_limit = max(5, current_limit - 5)  # Minimum 5 minutes
                        self.user_manager.set_max_playtime(new_limit)
                        print(f"Playtime limit decreased to {new_limit} minutes")
                    elif self.game_state == "user_prompt":
                        if event.key == pygame.K_SPACE:
                            # Continue with previous user
                            if self.playtime_manager.start_session():  # Start timing
                                self.reset_game()
                                self.game_state = "playing"
                            else:
                                # Playtime limit reached, show override screen
                                self.game_state = "time_limit_reached"
                                self.pin_input = ""
                                self.pin_error_message = ""
                                self.pin_attempts = 3
                        elif event.key == pygame.K_n:
                            # New user wants to play
                            self.game_state = "name_input"
                            self.username_input = ""
                            self.username_prompt = "Enter your name and press ENTER:"
                    elif self.game_state == "name_input":
                        if event.key == pygame.K_RETURN:
                            if self.username_input.strip():
                                # Register user and start game
                                self.user_manager.register_user(self.username_input.strip())
                                if self.playtime_manager.start_session():  # Start timing
                                    self.reset_game()
                                    self.game_state = "playing"
                                else:
                                    # Playtime limit reached, show override screen
                                    self.game_state = "time_limit_reached"
                                    self.pin_input = ""
                                    self.pin_error_message = ""
                                    self.pin_attempts = 3
                            else:
                                self.username_prompt = "Name cannot be empty! Enter your name:"
                        elif event.key == pygame.K_BACKSPACE:
                            self.username_input = self.username_input[:-1]
                        else:
                            # Add character to username (only printable characters)
                            if event.unicode.isprintable() and len(self.username_input) < 20:
                                self.username_input += event.unicode

                    elif self.game_state == "time_limit_reached":
                        if event.key == pygame.K_RETURN:
                            if self.pin_input:
                                # Verify PIN
                                if self.user_manager.verify_parental_access(self.pin_input):
                                    # PIN correct - activate parental override
                                    self.user_manager.activate_parental_override(1)  # 1 hour override
                                    self.pin_input = ""
                                    self.pin_error_message = ""
                                    self.pin_attempts = 3
                                    # Start a new session and continue
                                    self.playtime_manager.start_session()
                                    if self.player is None:
                                        self.reset_game()
                                    self.game_state = "playing"
                                    print("PARENTAL OVERRIDE ACTIVATED - 1 hour of additional playtime granted")
                                else:
                                    # PIN incorrect
                                    self.pin_attempts -= 1
                                    self.pin_input = ""
                                    if self.pin_attempts > 0:
                                        self.pin_error_message = f"Incorrect PIN! {self.pin_attempts} attempts remaining."
                                    else:
                                        # Out of attempts - exit game
                                        self.pin_error_message = "Too many incorrect attempts. Goodbye!"
                                        pygame.time.wait(2000)
                                        running = False
                        elif event.key == pygame.K_BACKSPACE:
                            self.pin_input = self.pin_input[:-1]
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                        else:
                            # Add digit to PIN (only digits, max 10 chars)
                            if event.unicode.isdigit() and len(self.pin_input) < 10:
                                self.pin_input += event.unicode

                    elif event.key == pygame.K_SPACE:
                        if self.game_state == "caught":
                            self.reset_game()
                        elif self.game_state == "won":
                            # Show shop before next level if shells enabled
                            math_features = self.user_manager.get_math_features()
                            if math_features.get("shells_enabled", True) and self.shop_overlay:
                                self.shop_overlay.show()
                                self.game_state = "shop"
                            else:
                                self.next_level()

                # Handle riddle overlay input
                if self.game_state == "riddle" and self.math_riddle_overlay.active:
                    solved, correct, answer = self.math_riddle_overlay.handle_event(event)
                    if solved is not None:
                        self.user_manager.record_math_problem(correct)
                        if correct:
                            self.correct_sound.play()
                            self.player.start_invincibility()
                            self.user_manager.record_riddle_escape()
                        else:
                            self.wrong_sound.play()
                            if not self.player.lose_life():
                                # Out of lives - game over
                                self.game_state = "caught"
                                self.roar_sound.play()
                                self.user_manager.update_stats(self.level)
                                threading.Thread(target=self.game_over_audio, daemon=True).start()

                # Handle gate puzzle input
                if self.game_state == "gate_puzzle" and hasattr(self, 'current_gate_problem'):
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            if self.gate_user_input:
                                try:
                                    answer = int(self.gate_user_input)
                                    if self.current_gate.check_answer(answer):
                                        self.correct_sound.play()
                                        self.gate_open_sound.play()
                                        self.current_gate.unlock()
                                        self.user_manager.record_math_problem(True)
                                        self.game_state = "playing"
                                    else:
                                        self.wrong_sound.play()
                                        self.user_manager.record_math_problem(False)
                                        # Bounce player back
                                        self.player.rect.x -= 50
                                        self.game_state = "playing"
                                    self.gate_user_input = ""
                                except ValueError:
                                    pass
                        elif event.key == pygame.K_BACKSPACE:
                            self.gate_user_input = self.gate_user_input[:-1]
                        elif event.unicode.isdigit() and len(self.gate_user_input) < 3:
                            self.gate_user_input += event.unicode

                # Handle shop input
                if self.game_state == "shop" and self.shop_overlay and self.shop_overlay.active:
                    powerup, skip = self.shop_overlay.handle_event(event, self.math_generator)
                    if powerup:
                        self.power_up_manager.purchase_powerup(powerup, self.player)
                        self.correct_sound.play()
                    if skip:
                        self.shop_overlay.hide()
                        self.next_level()

            # Update game states
            if self.game_state == "riddle":
                result = self.math_riddle_overlay.update()
                if result == 'done':
                    if self.player.lives > 0:
                        self.game_state = "playing"
                elif result == 'timeout':
                    self.wrong_sound.play()
                    if not self.player.lose_life():
                        self.game_state = "caught"
                        self.roar_sound.play()
                        self.user_manager.update_stats(self.level)

            if self.game_state == "shop" and self.shop_overlay:
                self.shop_overlay.update()

            if self.power_up_manager:
                self.power_up_manager.update()

            # Update
            if self.game_state == "playing":
                keys = pygame.key.get_pressed()
                is_moving = self.player.update(keys)

                # Update sharks with player position so they face the minnow
                player_pos = (self.player.rect.centerx, self.player.rect.centery)
                for shark in self.sharks:
                    shark.update(player_pos)

                # Update shells if enabled
                math_features = self.user_manager.get_math_features()
                if math_features.get("shells_enabled", True) and self.shell_manager:
                    collected = self.shell_manager.update(self.player, self.sharks, self.math_gates)
                    if collected > 0:
                        self.user_manager.add_shells(collected)
                        self.shell_collect_sound.play()

                # Update math gates
                for gate in self.math_gates:
                    gate.update()

                # Update comparison zones
                for zone in self.comparison_zones:
                    zone.update()

                # Play swimming sound occasionally when moving
                if is_moving and random.randint(1, 30) == 1:  # 1 in 30 chance per frame
                    self.swimming_sound.play()

                # Check if player reached the safe zone (proportional to screen width)
                safe_zone_width = max(100, SCREEN_WIDTH // 8)
                if self.player.rect.right >= SCREEN_WIDTH - safe_zone_width:
                    self.game_state = "won"
                    self.success_sound.play()  # Play success sound

                    # Audio feedback for level completion
                    threading.Thread(target=self.level_complete_audio, daemon=True).start()

                # Check collision with math gates
                if math_features.get("math_gates_enabled", True):
                    for gate in self.math_gates:
                        if gate.locked and self.player.rect.colliderect(gate.rect):
                            self.game_state = "gate_puzzle"
                            self.current_gate = gate
                            self.current_gate_problem = gate.problem
                            self.gate_user_input = ""
                            break

                # Check comparison zones
                if math_features.get("comparison_zones_enabled", True):
                    for zone in self.comparison_zones:
                        result = zone.check_player_path(self.player.rect)
                        if result == 'correct':
                            self.correct_sound.play()
                            self.user_manager.record_math_problem(True)
                            zone.kill()
                        elif result == 'wrong':
                            self.wrong_sound.play()
                            self.user_manager.record_math_problem(False)
                            # Spawn an extra shark as penalty
                            x = random.randint(200, 600)
                            y = random.randint(50, SCREEN_HEIGHT - 50)
                            speed = 2 + (self.level * 0.3)
                            new_shark = Shark(x, y, speed)
                            new_shark.set_level(self.level)
                            self.sharks.add(new_shark)
                            self.all_sprites.add(new_shark)
                            zone.kill()

                # Check collision with sharks
                if pygame.sprite.spritecollide(self.player, self.sharks, False):
                    if not self.player.is_invulnerable():
                        # Check if riddles are enabled
                        if math_features.get("shark_riddles_enabled", True):
                            self.game_state = "riddle"
                            operation = self.user_manager.get_math_operation()
                            timer_enabled = self.user_manager.get_math_timer_enabled()
                            self.math_riddle_overlay.start_riddle(self.level, self.player.lives, operation, timer_enabled)
                        else:
                            self.game_state = "caught"
                            self.roar_sound.play()
                            self.user_manager.update_stats(self.level)
                            threading.Thread(target=self.game_over_audio, daemon=True).start()
            
            # Draw
            if self.game_state == "name_input":
                # Draw simple background for username input
                self.screen.fill(OCEAN_BLUE)
                self.draw_username_input_screen()
            elif self.game_state == "user_prompt":
                # Draw simple background for user prompt
                self.screen.fill(OCEAN_BLUE)
                self.draw_user_prompt_screen()
            elif self.game_state == "time_limit_reached":
                # Draw time limit screen with parental override
                self.screen.fill(OCEAN_BLUE)
                self.draw_time_limit_screen()
            else:
                # Get earthquake screen shake offset
                shake_offset = self.ocean_background.get_screen_shake_offset()
                
                # Create a temporary surface for screen shake effect
                if abs(shake_offset['x']) > 0.1 or abs(shake_offset['y']) > 0.1:
                    # Apply screen shake
                    temp_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                    self.ocean_background.draw(temp_surface)
                    
                    # Blit the shaken surface to screen with offset
                    shake_x = int(shake_offset['x'])
                    shake_y = int(shake_offset['y'])
                    self.screen.blit(temp_surface, (shake_x, shake_y))
                else:
                    # No earthquake, draw normally
                    self.ocean_background.draw(self.screen)
                
                # Draw zones (proportional to screen size) with transparency
                start_zone_width = max(100, SCREEN_WIDTH // 8)
                safe_zone_width = max(100, SCREEN_WIDTH // 8)
                
                # Create semi-transparent surfaces for zones
                start_zone = pygame.Surface((start_zone_width, SCREEN_HEIGHT))
                start_zone.set_alpha(100)  # Semi-transparent
                start_zone.fill(GREEN)
                self.screen.blit(start_zone, (0, 0))
                
                safe_zone = pygame.Surface((safe_zone_width, SCREEN_HEIGHT))
                safe_zone.set_alpha(100)  # Semi-transparent
                safe_zone.fill(YELLOW)
                self.screen.blit(safe_zone, (SCREEN_WIDTH - safe_zone_width, 0))
                
                # Draw shells
                if self.shells:
                    self.shells.draw(self.screen)

                # Draw sprites
                if self.all_sprites:
                    self.all_sprites.draw(self.screen)

                # Draw UI (during gameplay or riddle/gate states)
                if self.game_state in ["playing", "riddle", "gate_puzzle"]:
                    self.draw_text(f"Player: {self.user_manager.current_user}", self.small_font, WHITE, 10, 10)
                    self.draw_text(f"Level: {self.level}", self.font, WHITE, 10, 35)

                    # Show lives (hearts)
                    lives_text = f"Lives: {'<3 ' * self.player.lives}{'X ' * (3 - self.player.lives)}"
                    self.draw_text(lives_text, self.small_font, RED, 10, 65)

                    # Show shells
                    math_features = self.user_manager.get_math_features()
                    if math_features.get("shells_enabled", True):
                        shells = self.user_manager.get_user_shells()
                        self.draw_text(f"Shells: {shells}", self.small_font, YELLOW, 10, 85)

                    # Show active power-ups
                    if self.power_up_manager:
                        active = self.power_up_manager.get_active_powerups()
                        if active:
                            powerup_text = "Active: " + ", ".join(active)
                            self.draw_text(powerup_text, self.small_font, LIGHT_GREEN, 10, 105)
                        if self.power_up_manager.shark_skip_available:
                            self.draw_text("Shark Skip Ready!", self.small_font, PURPLE, 10, 125)

                    # Show user stats
                    stats = self.user_manager.get_user_stats()
                    if stats:
                        y_offset = 145 if self.power_up_manager and (self.power_up_manager.get_active_powerups() or self.power_up_manager.shark_skip_available) else 105
                        if not math_features.get("shells_enabled", True):
                            y_offset = 85
                        self.draw_text(f"Best Level: {stats['highest_level']}", self.small_font, WHITE, 10, y_offset)

                    self.draw_text("Arrow keys or WASD to move", self.small_font, WHITE, 10, SCREEN_HEIGHT - 70)
                    self.draw_text("F11: Fullscreen | F12: Change User | ESC: Quit", self.small_font, WHITE, 10, SCREEN_HEIGHT - 50)
                    self.draw_text("M: Music | O: Math Op | T: Timer | +/-: Volume", self.small_font, WHITE, 10, SCREEN_HEIGHT - 30)

                    # Show music status
                    if self.user_manager.get_music_enabled():
                        volume = int(self.user_manager.get_music_volume() * 100)
                        music_status = f"Music: ON ({volume}%)"
                    else:
                        music_status = "Music: OFF"
                    self.draw_text(music_status, self.small_font, WHITE, 10, SCREEN_HEIGHT - 10)

                    # Show remaining playtime
                    remaining_time = self.playtime_manager.get_remaining_time_minutes()
                    time_str = self.playtime_manager.format_time(remaining_time)
                    time_color = RED if remaining_time <= 5 else WHITE
                    self.draw_text(f"Time: {time_str}", self.small_font, time_color, SCREEN_WIDTH - 150, 10)

                    # Show current math operation focus
                    operation = self.user_manager.get_math_operation()
                    op_display = {"addition": "Addition (+)", "subtraction": "Subtraction (-)", "both": "Both (+/-)"}
                    self.draw_text(f"Math: {op_display.get(operation, operation)}", self.small_font, LIGHT_GREEN, SCREEN_WIDTH - 200, 35)

                    # Show timer status
                    timer_enabled = self.user_manager.get_math_timer_enabled()
                    timer_status = "Timer: ON" if timer_enabled else "Timer: OFF"
                    timer_color = WHITE if timer_enabled else LIGHT_GREEN
                    self.draw_text(timer_status, self.small_font, timer_color, SCREEN_WIDTH - 150, 55)

                # Draw gate puzzle overlay
                if self.game_state == "gate_puzzle" and hasattr(self, 'current_gate_problem'):
                    # Semi-transparent background
                    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 150))
                    self.screen.blit(overlay, (0, 0))

                    # Gate puzzle box
                    box_width, box_height = 400, 200
                    box_x = (SCREEN_WIDTH - box_width) // 2
                    box_y = (SCREEN_HEIGHT - box_height) // 2
                    pygame.draw.rect(self.screen, (60, 40, 100), (box_x, box_y, box_width, box_height), border_radius=15)
                    pygame.draw.rect(self.screen, (120, 80, 160), (box_x, box_y, box_width, box_height), 3, border_radius=15)

                    self.draw_text("MATH GATE", self.font, YELLOW, SCREEN_WIDTH // 2, box_y + 30, True)
                    self.draw_text(self.current_gate_problem['text'], self.font, WHITE, SCREEN_WIDTH // 2, box_y + 80, True)

                    input_display = self.gate_user_input if self.gate_user_input else "_"
                    self.draw_text(input_display, self.font, YELLOW, SCREEN_WIDTH // 2, box_y + 130, True)
                    self.draw_text("Type answer and press ENTER", self.small_font, (180, 180, 180), SCREEN_WIDTH // 2, box_y + 170, True)

            # Game state messages
            if self.game_state == "caught":
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                overlay.set_alpha(128)
                overlay.fill(BLACK)
                self.screen.blit(overlay, (0, 0))

                self.draw_text("CAUGHT BY SHARKS!", self.font, RED, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, True)
                self.draw_text(f"Level Reached: {self.level}", self.font, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, True)
                self.draw_text("Press SPACE to restart", self.small_font, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, True)

            elif self.game_state == "won":
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                overlay.set_alpha(128)
                overlay.fill(BLACK)
                self.screen.blit(overlay, (0, 0))

                self.draw_text("SAFE!", self.font, GREEN, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, True)
                self.draw_text(f"Level {self.level} Complete!", self.font, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, True)
                shells_msg = "Press SPACE to visit shop" if self.user_manager.get_math_features().get("shells_enabled", True) else "Press SPACE for next level"
                self.draw_text(shells_msg, self.small_font, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, True)

            elif self.game_state == "riddle":
                self.math_riddle_overlay.draw(self.screen)

            elif self.game_state == "shop" and self.shop_overlay:
                self.shop_overlay.draw(self.screen)
            
            pygame.display.flip()
        
        # End playtime session
        self.playtime_manager.end_session()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Sharks and Minnows Game with Parental Controls')
    parser.add_argument('--parental', action='store_true', 
                       help='Open parental control console')
    args = parser.parse_args()
    
    if args.parental:
        # Run parental control console
        print("Starting Parental Control Console...")
        
        # Initialize minimal components needed for parental controls
        user_manager = UserManager()
        console = ParentalControlConsole(user_manager)
        console.run_parental_console()
    else:
        # Run normal game
        game = Game()
        game.run()