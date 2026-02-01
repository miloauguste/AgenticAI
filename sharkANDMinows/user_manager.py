import json
import os
import time
from datetime import datetime

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
                "total_score": 0
            }
            print(f"New user {username} registered!")
        else:
            print(f"Welcome back, {username}!")
        
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
                return self._default_settings()
        return self._default_settings()
    
    def _default_settings(self):
        """Return default settings"""
        return {
            "last_user": None,
            "fullscreen": False,
            "music_enabled": True,
            "music_volume": 0.1,
            "session_start_time": None,
            "daily_playtime": {},
            "max_playtime_minutes": 15,
            "parental_override": {
                "active": False,
                "expiry_time": 0,
                "duration_hours": 0
            }
        }
    
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
    
    def set_fullscreen(self, fullscreen):
        """Set fullscreen preference"""
        self.settings["fullscreen"] = fullscreen
        self.save_settings()
    
    def get_music_enabled(self):
        """Get music enabled setting"""
        return self.settings.get("music_enabled", True)
    
    def set_music_enabled(self, enabled):
        """Set music enabled setting"""
        self.settings["music_enabled"] = enabled
        self.save_settings()
    
    def get_music_volume(self):
        """Get music volume setting"""
        return self.settings.get("music_volume", 0.1)
    
    def set_music_volume(self, volume):
        """Set music volume setting"""
        self.settings["music_volume"] = volume
        self.save_settings()
    
    def get_music_folder(self):
        """Get music folder path"""
        return "music"
    
    def get_today_string(self):
        """Get today's date as string"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def get_total_playtime_today(self):
        """Get total playtime for today in minutes"""
        today = self.get_today_string()
        daily_playtime = self.settings.get("daily_playtime", {})
        return daily_playtime.get(today, 0)
    
    def add_playtime_today(self, minutes):
        """Add playtime minutes to today's total"""
        today = self.get_today_string()
        daily_playtime = self.settings.get("daily_playtime", {})
        current_time = daily_playtime.get(today, 0)
        daily_playtime[today] = current_time + minutes
        self.settings["daily_playtime"] = daily_playtime
        self.save_settings()
    
    def get_max_playtime(self):
        """Get maximum playtime allowed per day in minutes"""
        return self.settings.get("max_playtime_minutes", 15)
    
    def set_max_playtime(self, minutes):
        """Set maximum playtime allowed per day in minutes"""
        self.settings["max_playtime_minutes"] = minutes
        self.save_settings()
    
    def set_session_start_time(self, timestamp):
        """Set session start time"""
        self.settings["session_start_time"] = timestamp
        self.save_settings()
    
    def reset_session(self):
        """Reset session data"""
        self.settings["session_start_time"] = None
        self.save_settings()
    
    def get_parental_override_active(self):
        """Check if parental override is currently active"""
        override_data = self.settings.get("parental_override", {})
        if not override_data.get("active", False):
            return False
        
        current_time = time.time()
        expiry_time = override_data.get("expiry_time", 0)
        
        if current_time >= expiry_time:
            self.deactivate_parental_override()
            return False
        
        return True
    
    def activate_parental_override(self, duration_hours):
        """Activate parental override for specified duration"""
        current_time = time.time()
        expiry_time = current_time + (duration_hours * 3600)
        
        self.settings["parental_override"] = {
            "active": True,
            "expiry_time": expiry_time,
            "duration_hours": duration_hours
        }
        self.save_settings()
        print(f"Parental override activated for {duration_hours} hours")
    
    def deactivate_parental_override(self):
        """Deactivate parental override"""
        self.settings["parental_override"] = {
            "active": False,
            "expiry_time": 0,
            "duration_hours": 0
        }
        self.save_settings()

class PlaytimeManager:
    def __init__(self, user_manager):
        self.user_manager = user_manager
        self.session_start_time = None
        self.is_time_limit_reached = False
        self.time_warning_shown = False
        
    def start_session(self):
        """Start a new play session"""
        if self.user_manager.get_parental_override_active():
            print("PARENTAL OVERRIDE ACTIVE - Time limits temporarily disabled")
            current_time = time.time()
            self.session_start_time = current_time
            self.user_manager.set_session_start_time(current_time)
            self.is_time_limit_reached = False
            self.time_warning_shown = False
            print("Play session started with parental override")
            return True
        
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
        return (time.time() - self.session_start_time) / 60
    
    def get_remaining_time_minutes(self):
        """Get remaining playtime in minutes"""
        max_time = self.user_manager.get_max_playtime()
        total_used_time = self.user_manager.get_total_playtime_today() + self.get_session_time_minutes()
        return max(0, max_time - total_used_time)
    
    def check_time_limit(self):
        """Check if time limit has been reached"""
        if self.user_manager.get_parental_override_active():
            return False
        
        remaining_time = self.get_remaining_time_minutes()
        
        if remaining_time <= 5 and remaining_time > 0 and not self.time_warning_shown:
            self.time_warning_shown = True
            print(f"WARNING: Only {remaining_time:.1f} minutes of playtime remaining!")
            print("Ask a parent to activate parental override if more time is needed.")
            return False
        
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