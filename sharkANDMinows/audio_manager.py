import pygame
import pyttsx3
import os
import random
import numpy as np
import threading
import math

class AudioManager:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
    
    def speak(self, text):
        """Convert text to speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

class MusicManager:
    def __init__(self, user_manager):
        self.user_manager = user_manager
        self.current_music = None
        self.music_files = []
        self.current_track_index = 0
        self.is_playing = False
        
        pygame.mixer.music.set_volume(self.user_manager.get_music_volume())
        self.load_music_files()
        
        if self.user_manager.get_music_enabled() and self.music_files:
            self.play_next_track()
    
    def load_music_files(self):
        """Load music files from the music folder"""
        music_folder = self.user_manager.get_music_folder()
        supported_formats = ['.mp3', '.wav', '.ogg', '.m4a']
        
        try:
            if not os.path.exists(music_folder):
                os.makedirs(music_folder)
                print(f"Created music folder: {music_folder}")
                print(f"Add your music files (.mp3, .wav, .ogg, .m4a) to the '{music_folder}' folder")
                return
            
            self.music_files = []
            for file in os.listdir(music_folder):
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    full_path = os.path.join(music_folder, file)
                    self.music_files.append(full_path)
            
            if self.music_files:
                random.shuffle(self.music_files)
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
            
            self.current_track_index = (self.current_track_index + 1) % len(self.music_files)
            
        except pygame.error as e:
            print(f"Error playing music file '{track}': {e}")
            self.current_track_index = (self.current_track_index + 1) % len(self.music_files)
            if self.current_track_index != 0:
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

class SoundGenerator:
    def __init__(self):
        self.sounds = {}
        print("Generating sound effects...")
        self.generate_all_sounds()
        print("Sound effects ready!")
    
    def generate_sine_wave(self, frequency, duration, sample_rate=22050, amplitude=0.1):
        """Generate a sine wave for given frequency and duration"""
        frames = int(duration * sample_rate)
        arr = amplitude * np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
        arr = (arr * 32767).astype(np.int16)
        stereo_arr = np.column_stack((arr, arr))
        return pygame.sndarray.make_sound(stereo_arr)
    
    def generate_rumble(self, duration=2.0, sample_rate=22050):
        """Generate earthquake rumble sound"""
        frames = int(duration * sample_rate)
        low_freq = 30 + 10 * np.random.random(frames)
        rumble = 0.2 * np.sin(2 * np.pi * low_freq * np.linspace(0, duration, frames))
        
        noise = 0.05 * np.random.random(frames)
        rumble = rumble + noise
        
        envelope = np.exp(-2 * np.linspace(0, duration, frames))
        rumble = rumble * envelope
        
        rumble = (rumble * 32767).astype(np.int16)
        stereo_rumble = np.column_stack((rumble, rumble))
        return pygame.sndarray.make_sound(stereo_rumble)
    
    def generate_volcanic_explosion(self, duration=1.5, sample_rate=22050):
        """Generate volcanic explosion sound"""
        frames = int(duration * sample_rate)
        t = np.linspace(0, duration, frames)
        
        low_boom = 0.3 * np.sin(2 * np.pi * 40 * t)
        mid_crash = 0.2 * np.sin(2 * np.pi * 200 * t)
        high_crack = 0.15 * np.sin(2 * np.pi * 800 * t) * np.exp(-5 * t)
        noise = 0.1 * np.random.random(frames)
        
        explosion = low_boom + mid_crash + high_crack + noise
        explosion = explosion * np.exp(-1.5 * t)
        
        explosion = np.clip(explosion, -1, 1)
        explosion = (explosion * 32767).astype(np.int16)
        stereo_explosion = np.column_stack((explosion, explosion))
        return pygame.sndarray.make_sound(stereo_explosion)
    
    def generate_bubbling_lava(self, duration=3.0, sample_rate=22050):
        """Generate bubbling lava sound"""
        frames = int(duration * sample_rate)
        t = np.linspace(0, duration, frames)
        
        base_freq = 100
        bubble_freq = base_freq + 50 * np.sin(10 * t) * np.random.random(frames)
        bubbling = 0.15 * np.sin(2 * np.pi * bubble_freq * t)
        
        gurgle = 0.1 * np.sin(2 * np.pi * 300 * t) * (1 + 0.5 * np.sin(20 * t))
        
        noise = 0.05 * np.random.random(frames)
        lava_sound = bubbling + gurgle + noise
        
        lava_sound = np.clip(lava_sound, -1, 1)
        lava_sound = (lava_sound * 32767).astype(np.int16)
        stereo_lava = np.column_stack((lava_sound, lava_sound))
        return pygame.sndarray.make_sound(stereo_lava)
    
    def generate_all_sounds(self):
        """Generate all game sound effects"""
        self.sounds['eat'] = self.generate_sine_wave(600, 0.1)
        self.sounds['level_up'] = self.generate_sine_wave(800, 0.3)
        self.sounds['game_over'] = self.generate_sine_wave(200, 1.0)
        self.sounds['earthquake'] = self.generate_rumble(2.0)
        self.sounds['volcano_explosion'] = self.generate_volcanic_explosion(1.5)
        self.sounds['lava_bubbling'] = self.generate_bubbling_lava(3.0)
    
    def play_sound(self, sound_name):
        """Play a specific sound effect"""
        if sound_name in self.sounds:
            try:
                self.sounds[sound_name].play()
            except pygame.error:
                pass
    
    def play_earthquake(self):
        """Play earthquake sound effect"""
        self.play_sound('earthquake')
    
    def play_volcano_explosion(self):
        """Play volcano explosion sound effect"""
        self.play_sound('volcano_explosion')
    
    def play_lava_bubbling(self):
        """Play lava bubbling sound effect"""
        self.play_sound('lava_bubbling')