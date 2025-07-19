"""
TDF System - Enhanced Version with Modern UI, Fixed Filters and Role-Based Access Control
Features: Modern Professional UI, Fixed filter functionality, Role-based permissions, Database path configuration, User account management
Updated: Modern typography, enhanced visual design, improved accessibility and readability
"""
#coderabbitai Request Full UI Redesign – Login & Add Ban Pages + Overall Stylin
#coderabbitai please prioritize visual clarity, modern layout, and practical usability improvements.


import sys
import os
import time
import logging
import threading
import sqlite3
import json
import hashlib
from PyQt5.QtCore import QThread, pyqtSignal
import resources
import queue
import weakref
from PyQt5.QtCore import QMutex, QWaitCondition, QPropertyAnimation, QEasingCurve
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional, Dict, Any
from dataclasses import dataclass
import queue

# PyQt5 imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Database imports
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

# Audio imports
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
# Test pull request to trigger CodeRabbit review
# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tdf_integrated.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Default Configuration
DEFAULT_CONFIG = {
    "server_db_path": r"C:\Users\abdalbagi.a\Desktop\APPPPNEW\TruckAutoId_copy.accdb",
    "local_sqlite_path": r"C:\Users\abdalbagi.a\Desktop\APP\tdf_app.db",
    "connection_timeout": 3,
    "cache_timeout": 5,
    "monitor_interval": 3,
    "auto_refresh_interval": 60,
    "default_user": "admin",
    "sound_enabled": True,
    "warning_sound_duration": 3.0
}

CONFIG_FILE = "config.json"

@dataclass
class TankerData:
    """Data structure for tanker information"""
    tanker_number: str
    company_name: str = ""
    balance: float = 0.0
    salik_tag: str = ""
    entry_date: str = ""
    entry_time: str = ""
    status: str = "UNKNOWN"
    reason: str = ""
    ban_info: dict = None
    voice_data: bytes = None
    voice_filename: str = None

class ModernUITheme:
    """Modern UI theme constants for consistent styling"""

    # Color Palette - Professional and modern
    PRIMARY = "#2563EB"           # Blue-600
    PRIMARY_DARK = "#1D4ED8"      # Blue-700
    PRIMARY_LIGHT = "#3B82F6"     # Blue-500
    DANGER = "#DC2626"  # Red for errors
    SURFACE = "#F3F4F6"  # Light gray for surfaces
    BORDER_LIGHT = "#E5E7EB"  # Light border color
    WARNING = "#F59E0B"  # Amber for warnings
    SUCCESS = "#10B981"  # Green for success

    SECONDARY = "#7C3AED"         # Violet-600
    ACCENT = "#059669"            # Emerald-600

    SUCCESS = "#059669"           # Emerald-600
    WARNING = "#D97706"           # Amber-600
    ERROR = "#DC2626"             # Red-600
    INFO = "#0284C7"              # Sky-600

    # Neutral colors
    BACKGROUND = "#FFFFFF"        # White
    SURFACE = "#F8FAFC"           # Slate-50
    CARD = "#FFFFFF"              # White

    BORDER = "#E2E8F0"            # Slate-200
    BORDER_LIGHT = "#F1F5F9"      # Slate-100

    TEXT_PRIMARY = "#0F172A"      # Slate-900
    TEXT_SECONDARY = "#475569"    # Slate-600
    TEXT_MUTED = "#94A3B8"        # Slate-400
    TEXT_DISABLED = "#CBD5E1"     # Slate-300

    # Dark theme variants for sidebar
    DARK_PRIMARY = "#1E293B"      # Slate-800
    DARK_SECONDARY = "#334155"    # Slate-700
    DARK_ACCENT = "#475569"       # Slate-600

    # Font configurations
    FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    FONT_FAMILY_MONO = "'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace"

    # Font sizes
    FONT_SIZE_XS = "11px"
    FONT_SIZE_SM = "13px"
    FONT_SIZE_BASE = "14px"
    FONT_SIZE_LG = "16px"
    FONT_SIZE_XL = "18px"
    FONT_SIZE_2XL = "20px"
    FONT_SIZE_3XL = "24px"
    FONT_SIZE_4XL = "30px"
    FONT_SIZE_5XL = "36px"

    # Spacing
    SPACE_XS = "4px"
    SPACE_SM = "8px"
    SPACE_MD = "12px"
    SPACE_LG = "16px"
    SPACE_XL = "20px"
    SPACE_2XL = "24px"
    SPACE_3XL = "32px"
    SPACE_4XL = "40px"

    # Border radius
    RADIUS_SM = "6px"
    RADIUS_MD = "8px"
    RADIUS_LG = "12px"
    RADIUS_XL = "16px"

    # Shadows
    SHADOW_SM = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    SHADOW_MD = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    SHADOW_LG = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
    SHADOW_XL = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"

class ConfigManager:
    """Configuration management class"""

    def __init__(self):
        self.config_file = CONFIG_FILE
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged_config = DEFAULT_CONFIG.copy()
                merged_config.update(config)
                return merged_config
            else:
                # Create default config file
                self.save_config(DEFAULT_CONFIG)
                return DEFAULT_CONFIG.copy()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return DEFAULT_CONFIG.copy()

    def save_config(self, config=None):
        """Save configuration to file"""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4, ensure_ascii=False)
            if config:
                self.config = config
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value

    def update_paths(self, server_path, local_path):
        """Update database paths"""
        self.config['server_db_path'] = server_path
        self.config['local_sqlite_path'] = local_path
        return self.save_config()

class UserManager:
    """User account management class"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.init_user_tables()

    def init_user_tables(self):
        """Initialize user management tables with schema migration support"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                # Check if users table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
                table_exists = cursor.fetchone() is not None

                if table_exists:
                    # Table exists, check and add missing columns
                    logger.info("Users table exists, checking schema...")

                    # Get existing columns
                    cursor.execute("PRAGMA table_info(users)")
                    existing_columns = [column[1] for column in cursor.fetchall()]
                    logger.info(f"Existing columns: {existing_columns}")

                    # Check if we have all required columns
                    required_columns = ['id', 'username', 'password_hash', 'full_name', 'role', 'is_active', 'created_at', 'last_login', 'created_by']
                    missing_columns = [col for col in required_columns if col not in existing_columns]

                    if missing_columns:
                        logger.info(f"Missing columns: {missing_columns}")

                        # If too many columns are missing or we have critical missing columns, recreate table
                        critical_missing = ['password_hash', 'role', 'is_active']
                        has_critical_missing = any(col in missing_columns for col in critical_missing)

                        if has_critical_missing or len(missing_columns) > 4:
                            logger.info("Too many critical columns missing, recreating table...")

                            # Backup existing data
                            cursor.execute("SELECT * FROM users")
                            existing_data = cursor.fetchall()
                            cursor.execute("PRAGMA table_info(users)")
                            old_columns = [column[1] for column in cursor.fetchall()]
                            logger.info(f"Backing up {len(existing_data)} users with columns: {old_columns}")

                            # Drop and recreate table
                            cursor.execute("DROP TABLE users")
                            cursor.execute('''
                                CREATE TABLE users (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    username TEXT UNIQUE NOT NULL,
                                    password_hash TEXT NOT NULL,
                                    full_name TEXT,
                                    role TEXT DEFAULT 'operator',
                                    is_active INTEGER DEFAULT 1,
                                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                    last_login DATETIME,
                                    created_by TEXT
                                )
                            ''')

                            # Restore data with proper mapping
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            for row in existing_data:
                                old_row_dict = dict(zip(old_columns, row))

                                # Map old data to new schema
                                username = old_row_dict.get('username', '')
                                password_hash = old_row_dict.get('password_hash') or old_row_dict.get('password', '')
                                full_name = old_row_dict.get('full_name', username.title())
                                role = old_row_dict.get('role', 'admin' if username == 'admin' else 'operator')
                                is_active = old_row_dict.get('is_active', 1)
                                created_at = old_row_dict.get('created_at', current_time)
                                created_by = old_row_dict.get('created_by', 'migration')

                                # If password is not hashed, hash it
                                if len(password_hash) < 50:  # Assume unhashed if too short
                                    if username == 'admin' and password_hash in ['admin', 'admin123']:
                                        password_hash = self.hash_password('admin123')
                                    elif username == 'operator' and password_hash in ['operator', 'operator123']:
                                        password_hash = self.hash_password('operator123')
                                    elif username == 'supervisor' and password_hash in ['supervisor', 'supervisor123']:
                                        password_hash = self.hash_password('supervisor123')
                                    else:
                                        password_hash = self.hash_password(password_hash)

                                cursor.execute("""
                                    INSERT INTO users (username, password_hash, full_name, role, is_active, created_at, created_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (username, password_hash, full_name, role, is_active, created_at, created_by))

                            logger.info("Table recreated and data migrated successfully")

                        else:
                            # Add missing columns one by one (for non-critical missing columns)
                            for column in missing_columns:
                                logger.info(f"Adding missing column: {column}")
                                if column == 'password_hash':
                                    cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
                                elif column == 'full_name':
                                    cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
                                elif column == 'role':
                                    cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'operator'")
                                elif column == 'is_active':
                                    cursor.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
                                elif column == 'created_at':
                                    # Use a constant default instead of CURRENT_TIMESTAMP
                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    cursor.execute(f"ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT '{current_time}'")
                                elif column == 'last_login':
                                    cursor.execute("ALTER TABLE users ADD COLUMN last_login DATETIME")
                                elif column == 'created_by':
                                    cursor.execute("ALTER TABLE users ADD COLUMN created_by TEXT DEFAULT 'system'")

                else:
                    # Create new table
                    logger.info("Creating new users table...")
                    cursor.execute('''
                        CREATE TABLE users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            full_name TEXT,
                            role TEXT DEFAULT 'operator',
                            is_active INTEGER DEFAULT 1,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            last_login DATETIME,
                            created_by TEXT
                        )
                    ''')

                # Check if users exist
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                logger.info(f"Found {user_count} existing users in database")

                # Create default users if none exist
                if user_count == 0:
                    logger.info("Creating default users...")

                    # Create default admin user
                    admin_password = self.hash_password("admin123")
                    cursor.execute("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("admin", admin_password, "System Administrator", "admin", "system"))

                    # Add sample operator user
                    operator_password = self.hash_password("operator123")
                    cursor.execute("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("operator", operator_password, "Default Operator", "operator", "system"))

                    # Add sample supervisor user
                    supervisor_password = self.hash_password("supervisor123")
                    cursor.execute("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("supervisor", supervisor_password, "Default Supervisor", "supervisor", "system"))

                    logger.info("Default users created: admin/admin123, operator/operator123, supervisor/supervisor123")

                else:
                    # Ensure default users exist and have correct data
                    logger.info("Checking and updating existing users...")

                    # Check admin user
                    cursor.execute("SELECT username, password_hash, role FROM users WHERE username = 'admin'")
                    admin_user = cursor.fetchone()

                    if not admin_user:
                        # Create admin user
                        admin_password = self.hash_password("admin123")
                        cursor.execute("""
                            INSERT INTO users (username, password_hash, full_name, role, created_by)
                            VALUES (?, ?, ?, ?, ?)
                        """, ("admin", admin_password, "System Administrator", "admin", "system"))
                        logger.info("Created missing admin user")
                    else:
                        # Update admin user if needed
                        if admin_user[2] != 'admin':
                            cursor.execute("UPDATE users SET role = 'admin' WHERE username = 'admin'")
                            logger.info("Updated admin user role")

                        # Check if password is properly hashed
                        if len(admin_user[1]) < 50:
                            new_hash = self.hash_password("admin123")
                            cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'admin'", (new_hash,))
                            logger.info("Updated admin password hash")

                    # Check operator user
                    cursor.execute("SELECT username, password_hash, role FROM users WHERE username = 'operator'")
                    operator_user = cursor.fetchone()

                    if not operator_user:
                        # Create operator user
                        operator_password = self.hash_password("operator123")
                        cursor.execute("""
                            INSERT INTO users (username, password_hash, full_name, role, created_by)
                            VALUES (?, ?, ?, ?, ?)
                        """, ("operator", operator_password, "Default Operator", "operator", "system"))
                        logger.info("Created missing operator user")
                    else:
                        # Update operator user if needed
                        if operator_user[2] != 'operator':
                            cursor.execute("UPDATE users SET role = 'operator' WHERE username = 'operator'")
                            logger.info("Updated operator user role")

                        # Check if password is properly hashed
                        if len(operator_user[1]) < 50:
                            new_hash = self.hash_password("operator123")
                            cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'operator'", (new_hash,))
                            logger.info("Updated operator password hash")

                    # Check supervisor user
                    cursor.execute("SELECT username, password_hash, role FROM users WHERE username = 'supervisor'")
                    supervisor_user = cursor.fetchone()

                    if not supervisor_user:
                        # Create supervisor user
                        supervisor_password = self.hash_password("supervisor123")
                        cursor.execute("""
                            INSERT INTO users (username, password_hash, full_name, role, created_by)
                            VALUES (?, ?, ?, ?, ?)
                        """, ("supervisor", supervisor_password, "Default Supervisor", "supervisor", "system"))
                        logger.info("Created missing supervisor user")
                    else:
                        # Update supervisor user if needed
                        if supervisor_user[2] != 'supervisor':
                            cursor.execute("UPDATE users SET role = 'supervisor' WHERE username = 'supervisor'")
                            logger.info("Updated supervisor user role")

                        # Check if password is properly hashed
                        if len(supervisor_user[1]) < 50:
                            new_hash = self.hash_password("supervisor123")
                            cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'supervisor'", (new_hash,))
                            logger.info("Updated supervisor password hash")

                # Final verification - list all users with their complete info
                cursor.execute("SELECT username, role, is_active FROM users")
                all_users = cursor.fetchall()
                logger.info(f"Final user list: {all_users}")

                conn.commit()
                logger.info("User tables initialized successfully with schema migration")

        except Exception as e:
            logger.error(f"Error initializing user tables: {e}")
            raise

    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def authenticate_user(self, username, password):
        """Authenticate user with improved debugging"""
        try:
            logger.info(f"Attempting authentication for user: {username}")

            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                # First check if user exists
                cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
                user_exists = cursor.fetchone()

                if not user_exists:
                    logger.warning(f"User '{username}' not found in database")
                    return None

                # Check authentication
                password_hash = self.hash_password(password)
                logger.debug(f"Generated password hash for {username}: {password_hash[:10]}...")

                cursor.execute("""
                    SELECT id, username, full_name, role, is_active, password_hash
                    FROM users 
                    WHERE username = ?
                """, (username,))

                result = cursor.fetchone()
                if result:
                    stored_hash = result[5]
                    is_active = result[4]

                    logger.debug(f"Stored password hash: {stored_hash[:10]}...")
                    logger.debug(f"User active status: {is_active}")

                    if not is_active:
                        logger.warning(f"User '{username}' account is inactive")
                        return None

                    if password_hash == stored_hash:
                        # Update last login
                        cursor.execute("""
                            UPDATE users SET last_login = CURRENT_TIMESTAMP 
                            WHERE username = ?
                        """, (username,))
                        conn.commit()

                        logger.info(f"Authentication successful for user: {username}")
                        return {
                            'id': result[0],
                            'username': result[1],
                            'full_name': result[2],
                            'role': result[3],
                            'is_active': result[4]
                        }
                    else:
                        logger.warning(f"Password mismatch for user: {username}")
                        return None
                else:
                    logger.warning(f"User query returned no results for: {username}")
                    return None

        except Exception as e:
            logger.error(f"Authentication error for {username}: {e}")
            return None

    def add_user(self, username, password, full_name, role, created_by):
        """Add new user"""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                password_hash = self.hash_password(password)
                cursor.execute("""
                    INSERT INTO users (username, password_hash, full_name, role, created_by)
                    VALUES (?, ?, ?, ?, ?)
                """, (username, password_hash, full_name, role, created_by))

                conn.commit()
                return True

        except sqlite3.IntegrityError:
            return False  # Username already exists
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False

    def update_user(self, user_id, full_name=None, role=None, is_active=None):
        """Update user information"""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                updates = []
                params = []

                if full_name is not None:
                    updates.append("full_name = ?")
                    params.append(full_name)

                if role is not None:
                    updates.append("role = ?")
                    params.append(role)

                if is_active is not None:
                    updates.append("is_active = ?")
                    params.append(is_active)

                if updates:
                    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
                    params.append(user_id)
                    cursor.execute(query, params)
                    conn.commit()
                    return True

                return False

        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False

    def change_password(self, user_id, new_password):
        """Change user password"""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                password_hash = self.hash_password(new_password)
                cursor.execute("""
                    UPDATE users SET password_hash = ? WHERE id = ?
                """, (password_hash, user_id))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error changing password: {e}")
            return False

    def get_all_users(self):
        """Get all users"""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, username, full_name, role, is_active, created_at, last_login, created_by
                    FROM users 
                    ORDER BY username
                """)

                return cursor.fetchall()

        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []

    def delete_user(self, user_id):
        """Delete user (soft delete by setting is_active to 0)"""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                cursor.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False

class WarningSound:
    """Warning sound generator and player"""

    def __init__(self, duration=3.0, frequency=800):
        self.duration = duration
        self.frequency = frequency
        self.sample_rate = 44100
        self._generate_warning_sound()

    def _generate_warning_sound(self):
        """Generate a warning sound pattern"""
        try:
            if not AUDIO_AVAILABLE:
                self.warning_data = None
                return

            # Generate a pulsating warning tone
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)

            # Create a pulsating effect (3 pulses per second)
            pulse_freq = 3
            pulse_envelope = 0.5 * (1 + np.sin(2 * np.pi * pulse_freq * t))

            # Generate the base tone
            tone = np.sin(2 * np.pi * self.frequency * t)

            # Apply pulsating envelope and fade in/out
            fade_samples = int(0.1 * self.sample_rate)  # 0.1 second fade
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)

            # Apply fades
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

            # Apply pulse envelope
            warning_tone = tone * pulse_envelope * 0.3  # Reduce volume

            self.warning_data = warning_tone.astype(np.float32).tobytes()

        except Exception as e:
            logger.error(f"Error generating warning sound: {e}")
            self.warning_data = None

    def play(self, callback=None):
        """Play warning sound"""
        if not AUDIO_AVAILABLE or not self.warning_data:
            if callback:
                callback(False, "Audio not available")
            return

        try:
            audio_array = np.frombuffer(self.warning_data, dtype=np.float32)
            sd.play(audio_array, self.sample_rate)
            if callback:
                callback(True, "Warning sound played")
        except Exception as e:
            logger.error(f"Error playing warning sound: {e}")
            if callback:
                callback(False, str(e))

    def stop(self):
        """Stop any playing sound"""
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"Error stopping sound: {e}")

class AudioWorker(QThread):
    """Separate thread for audio operations to prevent UI hanging"""
    finished = pyqtSignal(bool, str)

    def __init__(self, operation, audio_data=None):
        super().__init__()
        self.operation = operation  # 'play' or 'record'
        self.audio_data = audio_data
        self.sample_rate = 44100
        self.recording_data = []
        self.is_recording = False

    def run(self):
        try:
            if self.operation == 'play' and self.audio_data:
                self.play_audio()
            elif self.operation == 'record':
                self.record_audio()
        except Exception as e:
            self.finished.emit(False, str(e))

    def play_audio(self):
        try:
            sd.stop()
            audio_array = np.frombuffer(self.audio_data, dtype=np.float32)
            if len(audio_array) > 0:
                sd.play(audio_array, self.sample_rate)
                sd.wait()
                self.finished.emit(True, "Playback completed")
            else:
                self.finished.emit(False, "Empty audio data")
        except Exception as e:
            self.finished.emit(False, f"Playback error: {e}")

    def record_audio(self):
        try:
            self.is_recording = True
            self.recording_data = []

            def audio_callback(indata, frames, time, status):
                if self.is_recording and status is None:
                    self.recording_data.append(indata.copy())

            with sd.InputStream(callback=audio_callback, samplerate=self.sample_rate,
                               channels=1, dtype=np.float32):
                while self.is_recording:
                    self.msleep(100)

            if self.recording_data:
                audio_data = np.concatenate(self.recording_data, axis=0)
                self.finished.emit(True, audio_data.tobytes())
            else:
                self.finished.emit(False, "No audio recorded")
        except Exception as e:
            self.finished.emit(False, f"Recording error: {e}")

    def stop_recording(self):
        self.is_recording = False

class AudioRecorder:
    """Improved Audio recording and playback with threading"""

    def __init__(self):
        self.current_worker = None

    def start_recording(self, callback):
        """Start audio recording in separate thread"""
        if not AUDIO_AVAILABLE:
            callback(False, "Audio libraries not available")
            return

        self.stop_any_operation()
        self.current_worker = AudioWorker('record')
        self.current_worker.finished.connect(callback)
        self.current_worker.start()

    def stop_recording(self):
        """Stop current recording"""
        if self.current_worker and self.current_worker.operation == 'record':
            self.current_worker.stop_recording()

    def play_audio(self, audio_data, callback):
        """Play audio data in separate thread"""
        if not AUDIO_AVAILABLE or not audio_data:
            callback(False, "Audio not available or no data")
            return

        self.stop_any_operation()
        self.current_worker = AudioWorker('play', audio_data)
        self.current_worker.finished.connect(callback)
        self.current_worker.start()

    def stop_any_operation(self):
        """Stop any current audio operation"""
        try:
            sd.stop()
            if self.current_worker and self.current_worker.isRunning():
                if self.current_worker.operation == 'record':
                    self.current_worker.stop_recording()
                self.current_worker.terminate()
                self.current_worker.wait(1000)
        except Exception as e:
            logger.debug(f"Error stopping audio operation: {e}")

class DatabaseManager:
    """Enhanced database manager with configurable paths"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.server_db = self.config.get('server_db_path')
        self.sqlite_db = self.config.get('local_sqlite_path')
        self.cache = {}
        self.cache_timeout = self.config.get('cache_timeout', 5)
        self.server_available = False
        self.connection_strings = [
            f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={self.server_db};",
            f"Driver={{Microsoft Access Driver (*.mdb)}};DBQ={self.server_db};",
        ]
        self._connection_lock = threading.Lock()

        self.init_sqlite()
        self.test_server_connection()

    def update_paths(self, server_path, local_path):
        """Update database paths"""
        self.server_db = server_path
        self.sqlite_db = local_path
        self.connection_strings = [
            f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={server_path};",
            f"Driver={{Microsoft Access Driver (*.mdb)}};DBQ={server_path};",
        ]
        self.init_sqlite()
        self.test_server_connection()

    def test_server_connection(self):
        """Test server connection - ENSURE THIS METHOD EXISTS"""
        try:
            if not PYODBC_AVAILABLE:
                self.server_available = False
                logger.warning("pyodbc not available for server connection")
                return

            with self._connection_lock:
                with self.get_server_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM VehicleMaster")
                    count = cursor.fetchone()[0]
                    self.server_available = True
                    logger.info(f"Server connected - {count} vehicles in database")
        except Exception as e:
            self.server_available = False
            logger.warning(f"Server connection failed: {e}")
    def init_sqlite(self):
        """Initialize SQLite database with schema migration support"""
        try:
            os.makedirs(os.path.dirname(self.sqlite_db), exist_ok=True)

            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                # Create ban_records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ban_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tanker_number TEXT NOT NULL,
                        ban_reason TEXT NOT NULL,
                        ban_type TEXT CHECK (ban_type IN ('temporary', 'permanent', 'permission', 'reminder')) DEFAULT 'temporary',
                        start_date DATE,
                        end_date DATE,
                        created_by TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        voice_recording BLOB,
                        voice_filename TEXT
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tanker_number TEXT,
                        status TEXT,
                        reason TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        operator TEXT
                    )
                ''')

                # Check if is_active column exists, if not add it
                cursor.execute("PRAGMA table_info(ban_records)")
                columns = [column[1] for column in cursor.fetchall()]

                if 'is_active' not in columns:
                    logger.info("Adding is_active column to ban_records table")
                    cursor.execute('ALTER TABLE ban_records ADD COLUMN is_active INTEGER DEFAULT 1')

                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tanker_number ON ban_records(tanker_number)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_log_timestamp ON logs(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ban_active ON ban_records(is_active)')

                # Add sample data if empty
                cursor.execute("SELECT COUNT(*) FROM ban_records")
                if cursor.fetchone()[0] == 0:
                    sample_bans = [
                        ("TEST001", "Line Cross", "permanent", "2024-01-01", None, "System", 1),
                        ("40247", "Repair Not Done", "permission", "2024-01-01", "2025-12-31", "Admin", 1),
                        ("5001", "Schedule maintenance reminder", "reminder", "2024-01-01", "2025-12-31", "Admin", 1),
                    ]
                    cursor.executemany("""
                        INSERT INTO ban_records (tanker_number, ban_reason, ban_type, start_date, end_date, created_by, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, sample_bans)

                conn.commit()
                logger.info("SQLite database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise

    @contextmanager
    def get_server_connection(self):
        """Get server connection with improved error handling"""
        connection = None
        try:
            if not PYODBC_AVAILABLE:
                raise Exception("pyodbc not available")

            for conn_str in self.connection_strings:
                try:
                    connection = pyodbc.connect(conn_str, timeout=self.config.get('connection_timeout', 3))
                    self.server_available = True
                    yield connection
                    return
                except Exception:
                    continue

            self.server_available = False
            raise Exception("All connection attempts failed")

        except Exception as e:
            self.server_available = False
            raise
        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass

    def get_latest_tanker_from_server(self):
        """Get latest tanker from server with improved error handling"""
        try:
            with self.get_server_connection() as conn:
                cursor = conn.cursor()
                query = """
                   SELECT TOP 1 VT_RegNo, VT_InDate, VT_InTime, VT_ID 
                   FROM VehicleTransactions 
                   ORDER BY VT_ID DESC
                   """

                cursor.execute(query)
                result = cursor.fetchone()

                if result:
                    tanker_data = TankerData(
                        tanker_number=str(result[0]).strip(),
                        entry_date=str(result[1]),
                        entry_time=str(result[2])
                    )
                    return tanker_data
                return None

        except Exception as e:
            logger.error(f"Error getting latest tanker: {e}")
            return None

    def get_tanker_from_server(self, tanker_number):
        """Get specific tanker from server by number"""
        try:
            with self.get_server_connection() as conn:
                cursor = conn.cursor()
                query = """
                   SELECT TOP 1 VT_RegNo, VT_InDate, VT_InTime, VT_ID 
                   FROM VehicleTransactions 
                   WHERE VT_RegNo = ?
                   ORDER BY VT_ID DESC
                   """

                cursor.execute(query, (tanker_number,))
                result = cursor.fetchone()

                if result:
                    tanker_data = TankerData(
                        tanker_number=str(result[0]).strip(),
                        entry_date=str(result[1]),
                        entry_time=str(result[2])
                    )
                    return tanker_data
                return None

        except Exception as e:
            logger.error(f"Error getting tanker {tanker_number}: {e}")
            return None

    def get_complete_ban_record(self, tanker_number):
        """Get complete ban record with improved query"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT id, tanker_number, ban_reason, ban_type, start_date, end_date, 
                           created_by, created_at, voice_recording, voice_filename
                    FROM ban_records 
                    WHERE tanker_number = ? 
                    AND is_active = 1
                    AND (end_date IS NULL OR end_date >= date('now'))
                    ORDER BY created_at DESC
                    LIMIT 1
                """

                cursor.execute(query, (tanker_number,))
                result = cursor.fetchone()

                if result:
                    ban_record = {
                        'id': result[0],
                        'tanker_number': result[1],
                        'ban_reason': result[2],
                        'ban_type': result[3],
                        'start_date': result[4],
                        'end_date': result[5],
                        'created_by': result[6],
                        'created_at': result[7],
                        'voice_recording': result[8],
                        'voice_filename': result[9]
                    }
                    return ban_record

                return None

        except Exception as e:
            logger.error(f"Error getting ban record: {e}")
            return None

    def verify_specific_tanker(self, tanker_number, operator="System"):
        """Verify a specific tanker by number with improved logic"""
        try:
            tanker_data = None
            if self.server_available:
                tanker_data = self.get_tanker_from_server(tanker_number)

                if not tanker_data:
                    self.log_verification(tanker_number, "NOT_FOUND", f"Tanker {tanker_number} not found in database", operator)
                    return "NOT_FOUND", f"Tanker {tanker_number} not found in database", {
                        "tanker_number": tanker_number,
                        "ban_record": None,
                        "decision_color": "red",
                        "error": True
                    }

            ban_record = self.get_complete_ban_record(tanker_number)

            if ban_record:
                ban_reason = ban_record['ban_reason']
                ban_type = ban_record['ban_type']

                if ban_type == "permanent":
                    status = "DENIED"
                    reason = f"PERMANENT BAN: {ban_reason}"
                    decision_color = "red"
                    play_sound = True
                elif ban_type == "temporary":
                    status = "DENIED"
                    reason = f"TEMPORARY BAN: {ban_reason}"
                    decision_color = "red"
                    play_sound = True
                elif ban_type == "permission":
                    status = "ALLOWED_WITH_PERMISSION"
                    reason = f"PERMISSION REQUIRED: {ban_reason}"
                    decision_color = "orange"
                    play_sound = True
                elif ban_type == "reminder":
                    status = "ALLOWED_WITH_WARNING"
                    reason = f"REMINDER: {ban_reason}"
                    decision_color = "orange"
                    play_sound = True
                else:
                    status = "DENIED"
                    reason = f"BAN ACTIVE: {ban_reason}"
                    decision_color = "red"
                    play_sound = True

                self.log_verification(tanker_number, status, reason, operator)

                return status, reason, {
                    "tanker_number": tanker_number,
                    "ban_record": ban_record,
                    "decision_color": decision_color,
                    "tanker_data": tanker_data,
                    "play_sound": play_sound
                }
            else:
                status = "ALLOWED"
                reason = f"Vehicle {tanker_number} is not in ban records - Access granted"

                self.log_verification(tanker_number, status, reason, operator)

                return status, reason, {
                    "tanker_number": tanker_number,
                    "ban_record": None,
                    "decision_color": "green",
                    "tanker_data": tanker_data,
                    "play_sound": False
                }

        except Exception as e:
            logger.error(f"Verification error for {tanker_number}: {e}")
            error_msg = f"Verification failed: {str(e)}"
            self.log_verification(tanker_number, "ERROR", error_msg, operator)
            return "ERROR", error_msg, {"error": True, "tanker_number": tanker_number}

    def simple_tanker_verification(self, operator="System"):
        """Simplified tanker verification for latest tanker"""
        try:
            latest_tanker = self.get_latest_tanker_from_server()

            if not latest_tanker:
                error_msg = "No tanker entries found" if self.server_available else "Server unavailable"
                self.log_verification("UNKNOWN", "ERROR", error_msg, operator)
                return "ERROR", error_msg, {"error": True}

            tanker_number = latest_tanker.tanker_number
            return self.verify_specific_tanker(tanker_number, operator)

        except Exception as e:
            logger.error(f"Verification error: {e}")
            error_msg = f"Verification failed: {str(e)}"
            self.log_verification("UNKNOWN", "ERROR", error_msg, operator)
            return "ERROR", error_msg, {"error": True}

    def add_ban_record(self, tanker_number, ban_reason, ban_type, start_date, end_date, created_by, voice_data=None, voice_filename=None):
        """Add ban record with improved error handling"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO ban_records 
                    (tanker_number, ban_reason, ban_type, start_date, end_date, created_by, voice_recording, voice_filename, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (tanker_number, ban_reason, ban_type, start_date, end_date, created_by, voice_data, voice_filename))

                conn.commit()
                logger.info(f"Ban record added for {tanker_number}")
                return True

        except Exception as e:
            logger.error(f"Error adding ban record: {e}")
            return False

    def get_all_bans(self, filters=None):
        """Get all bans with improved filtering - Fixed filter logic"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT id, tanker_number, ban_reason, ban_type, start_date, end_date, 
                           created_by, voice_recording
                    FROM ban_records 
                    WHERE is_active = 1
                """
                params = []

                if filters:
                    # Fixed: Date filter now works correctly with created_at field
                    if filters.get('start_date') and filters.get('end_date'):
                        query += " AND DATE(created_at) BETWEEN ? AND ?"
                        params.extend([filters['start_date'], filters['end_date']])

                    # Fixed: Text filters now work with partial matches
                    if filters.get('reason'):
                        query += " AND LOWER(ban_reason) LIKE LOWER(?)"
                        params.append(f"%{filters['reason']}%")

                    if filters.get('ban_type'):
                        query += " AND ban_type = ?"
                        params.append(filters['ban_type'])

                    if filters.get('tanker_number'):
                        query += " AND LOWER(tanker_number) LIKE LOWER(?)"
                        params.append(f"%{filters['tanker_number']}%")

                query += " ORDER BY created_at DESC"

                cursor.execute(query, params)
                results = cursor.fetchall()
                logger.info(f"Ban query executed with {len(params)} parameters, returned {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"Error getting bans: {e}")
            return []

    def log_verification(self, tanker_number, status, reason, operator):
        """Log verification with improved error handling"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO logs (tanker_number, status, reason, operator)
                    VALUES (?, ?, ?, ?)
                """, (tanker_number, status, reason, operator))

                conn.commit()

        except Exception as e:
            logger.error(f"Error logging verification: {e}")

    def get_recent_logs(self, limit=50, filters=None):
        """Get recent logs with improved filtering - Fixed filter logic"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT id, tanker_number, status, reason, timestamp, operator
                    FROM logs 
                """
                params = []
                conditions = []

                if filters:
                    # Fixed: Date filters now work correctly
                    if filters.get('start_date') and filters.get('end_date'):
                        conditions.append("DATE(timestamp) BETWEEN ? AND ?")
                        params.extend([filters['start_date'], filters['end_date']])

                    # Fixed: Text filters with case-insensitive partial matching
                    if filters.get('reason'):
                        conditions.append("LOWER(reason) LIKE LOWER(?)")
                        params.append(f"%{filters['reason']}%")

                    if filters.get('status'):
                        conditions.append("LOWER(status) LIKE LOWER(?)")
                        params.append(f"%{filters['status']}%")

                    if filters.get('tanker_number'):
                        conditions.append("LOWER(tanker_number) LIKE LOWER(?)")
                        params.append(f"%{filters['tanker_number']}%")

                    if filters.get('operator'):
                        conditions.append("LOWER(operator) LIKE LOWER(?)")
                        params.append(f"%{filters['operator']}%")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                results = cursor.fetchall()
                logger.info(f"Log query executed with {len(params)} parameters, returned {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return []

    def get_ban_statistics(self, filters=None):
        """Get ban statistics from ban_records table - Fixed filter logic"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                query = "SELECT ban_type, created_at FROM ban_records WHERE is_active = 1"
                params = []

                if filters and filters.get('start_date') and filters.get('end_date'):
                    query += " AND DATE(created_at) BETWEEN ? AND ?"
                    params.extend([filters['start_date'], filters['end_date']])

                cursor.execute(query, params)
                results = cursor.fetchall()

                if not results:
                    return {'total_bans': 0, 'permanent': 0, 'temporary': 0, 'permission': 0, 'reminder': 0, 'active_bans': 0}

                total_bans = len(results)
                permanent_count = sum(1 for ban_type, _ in results if str(ban_type).lower() == "permanent")
                temporary_count = sum(1 for ban_type, _ in results if str(ban_type).lower() == "temporary")
                permission_count = sum(1 for ban_type, _ in results if str(ban_type).lower() == "permission")
                reminder_count = sum(1 for ban_type, _ in results if str(ban_type).lower() == "reminder")

                # Count currently active bans
                active_query = """
                    SELECT COUNT(*) FROM ban_records 
                    WHERE is_active = 1 
                    AND (end_date IS NULL OR end_date >= date('now'))
                """

                if filters and filters.get('start_date') and filters.get('end_date'):
                    active_query += " AND DATE(created_at) BETWEEN ? AND ?"
                    cursor.execute(active_query, params)
                else:
                    cursor.execute(active_query)

                active_bans = cursor.fetchone()[0]

                return {
                    'total_bans': total_bans,
                    'permanent': permanent_count,
                    'temporary': temporary_count,
                    'permission': permission_count,
                    'reminder': reminder_count,
                    'active_bans': active_bans
                }

        except Exception as e:
            logger.error(f"Error getting ban statistics: {e}")
            return {'total_bans': 0, 'permanent': 0, 'temporary': 0, 'permission': 0, 'reminder': 0, 'active_bans': 0}

    def get_verification_statistics(self, filters=None):
        """Get verification statistics from logs table - Fixed filter logic"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                query = "SELECT status FROM logs WHERE status IS NOT NULL"
                params = []

                if filters and filters.get('start_date') and filters.get('end_date'):
                    query += " AND DATE(timestamp) BETWEEN ? AND ?"
                    params.extend([filters['start_date'], filters['end_date']])

                cursor.execute(query, params)
                results = cursor.fetchall()

                if not results:
                    return {'total': 0, 'allowed': 0, 'rejected': 0, 'conditional': 0, 'errors': 0, 'success_rate': 0}

                total_count = len(results)
                allowed_count = 0
                rejected_count = 0
                conditional_count = 0
                error_count = 0

                for (status,) in results:
                    status_str = str(status).upper()

                    if "ALLOWED" in status_str:
                        allowed_count += 1
                    elif any(word in status_str for word in ["REJECTED", "DENIED"]):
                        rejected_count += 1
                    elif any(word in status_str for word in ["CONDITIONAL", "PERMISSION"]):
                        conditional_count += 1
                    elif "ERROR" in status_str:
                        error_count += 1

                success_rate = (allowed_count / total_count * 100) if total_count > 0 else 0

                return {
                    'total': total_count,
                    'allowed': allowed_count,
                    'rejected': rejected_count,
                    'conditional': conditional_count,
                    'errors': error_count,
                    'success_rate': success_rate
                }

        except Exception as e:
            logger.error(f"Error getting verification statistics: {e}")
            return {'total': 0, 'allowed': 0, 'rejected': 0, 'conditional': 0, 'errors': 0, 'success_rate': 0}


class ModernLoginDialog(QDialog):
    """Enhanced login dialog with modern UI design"""

    def __init__(self, user_manager):
        super().__init__()
        self.user_manager = user_manager
        self.user_info = None
        self.is_password_visible = False
        self.init_ui()
        self.apply_modern_styles()

    def init_ui(self):
        self.setWindowTitle("TDF System - Login")
        self.setFixedSize(1000, 650)
        self.setWindowFlags(Qt.FramelessWindowHint)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel with modern design
        left_panel = QFrame()
        left_panel.setFixedWidth(450)
        left_panel.setObjectName("leftPanel")

        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Modern logo design
        logo_container = QFrame()
        logo_container.setFixedSize(120, 120)
        logo_container.setObjectName("logoContainer")
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel("🚛")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet(f"font-size: {ModernUITheme.FONT_SIZE_5XL}; color: white; background: transparent;")
        logo_layout.addWidget(logo_label)

        welcome_label = QLabel("TDF System")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_4XL}; 
            font-weight: 700; 
            color: white; 
            margin-bottom: {ModernUITheme.SPACE_MD};
            letter-spacing: -0.5px;
        """)

        subtitle_label = QLabel("Professional Vehicle Management\nModern UI • Enhanced Security • Real-time Monitoring")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_LG}; 
            color: rgba(255, 255, 255, 0.9); 
            line-height: 1.6;
            font-weight: 400;
        """)

        feature_label = QLabel(
            "✓ Fixed Filters & Enhanced Search\n✓ Role-Based Access Control\n✓ Database Configuration\n✓ Modern Professional Interface")
        feature_label.setAlignment(Qt.AlignCenter)
        feature_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_SM}; 
            color: rgba(255, 255, 255, 0.8); 
            line-height: 1.8;
            margin-top: {ModernUITheme.SPACE_2XL};
            font-weight: 400;
        """)

        left_layout.addWidget(logo_container)
        left_layout.addWidget(welcome_label)
        left_layout.addWidget(subtitle_label)
        left_layout.addWidget(feature_label)
        left_panel.setLayout(left_layout)

        # Right panel with modern form design
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignCenter)
        right_layout.setContentsMargins(60, 40, 60, 40)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(40, 40)
        close_btn.setObjectName("closeButton")
        close_btn.clicked.connect(self.reject)

        header_layout = QHBoxLayout()
        header_layout.addStretch()
        header_layout.addWidget(close_btn)

        login_title = QLabel("Welcome Back")
        login_title.setAlignment(Qt.AlignCenter)
        login_title.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_3XL}; 
            color: {ModernUITheme.TEXT_PRIMARY}; 
            margin-bottom: {ModernUITheme.SPACE_4XL};
            font-weight: 600;
            letter-spacing: -0.5px;
        """)

        # Modern form inputs
        form_container = QFrame()
        form_container.setObjectName("formContainer")
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        form_layout.setContentsMargins(0, 0, 0, 0)

        # Username field
        username_label = QLabel("Username")
        username_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_SM}; 
            font-weight: 600; 
            color: {ModernUITheme.TEXT_SECONDARY};
            margin-bottom: {ModernUITheme.SPACE_SM};
        """)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setObjectName("modernInput")
        self.username_input.setText("admin")  # Default for testing

        # Add username icon
        username_container = QFrame()
        username_container.setObjectName("inputContainer")
        username_layout = QHBoxLayout(username_container)
        username_layout.setContentsMargins(15, 0, 15, 0)
        username_icon = QLabel("👤")
        username_icon.setStyleSheet("font-size: 16px;")
        username_layout.addWidget(username_icon)
        username_layout.addWidget(self.username_input)
        username_layout.setSpacing(10)

        password_label = QLabel("Password")
        password_label.setStyleSheet(f"""
                    font-size: {ModernUITheme.FONT_SIZE_SM}; 
                    font-weight: 600; 
                    color: {ModernUITheme.TEXT_SECONDARY};
                    margin-bottom: {ModernUITheme.SPACE_SM};
                """)

        # Create password container
        password_container = QFrame()
        password_container.setObjectName("inputContainer")
        password_layout = QHBoxLayout(password_container)
        password_layout.setContentsMargins(15, 0, 15, 0)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setObjectName("modernInput")
        self.password_input.setText("admin123")

        # Password toggle button
        self.toggle_password_btn = QToolButton()
        self.toggle_password_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_password_btn.setFixedSize(30, 30)
        self.toggle_password_btn.setObjectName("togglePasswordButton")
        self.toggle_password_btn.clicked.connect(self.toggle_password_visibility)

        password_layout.addWidget(self.password_input)
        password_layout.addWidget(self.toggle_password_btn)
        # Remember me checkbox
        remember_me = QCheckBox("Remember me")
        remember_me.setObjectName("rememberCheckbox")
        remember_me.setCursor(Qt.PointingHandCursor)

        # Forgot password link
        forgot_password = QLabel(
            "<a href='#' style='text-decoration:none; color: " + ModernUITheme.PRIMARY + ";'>Forgot password?</a>")
        forgot_password.setCursor(Qt.PointingHandCursor)
        forgot_password.setOpenExternalLinks(False)
        forgot_password.linkActivated.connect(self.show_password_reset)

        # Remember me + Forgot password layout
        remember_layout = QHBoxLayout()
        remember_layout.addWidget(remember_me)
        remember_layout.addStretch()
        remember_layout.addWidget(forgot_password)

        # Login button with loading animation
        self.login_btn = QPushButton("Sign In")
        self.login_btn.setObjectName("primaryButton")
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setDefault(True)

        # Create loading spinner (hidden by default)
        self.loading_spinner = QLabel()
        self.loading_spinner.setFixedSize(20, 20)
        self.loading_spinner.setObjectName("loadingSpinner")
        self.loading_spinner.hide()

        # Button layout to contain both button and spinner
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.loading_spinner)
        button_layout.addWidget(self.login_btn)
        button_layout.addStretch()
        button_layout.setSpacing(15)

        # Error message label
        self.error_label = QLabel()
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.hide()

        form_layout.addWidget(username_label)
        form_layout.addWidget(username_container)
        form_layout.addWidget(password_label)
        form_layout.addWidget(password_container)
        form_layout.addLayout(remember_layout)
        form_layout.addLayout(button_layout)
        form_layout.addWidget(self.error_label)

        # Info section with modern styling
        info_container = QFrame()
        info_container.setObjectName("infoContainer")
        info_layout = QVBoxLayout(info_container)
        info_layout.setSpacing(10)

        info_label = QLabel("Default Credentials")
        info_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_SM}; 
            font-weight: 600; 
            color: {ModernUITheme.TEXT_SECONDARY};
            margin-bottom: {ModernUITheme.SPACE_SM};
        """)

        credentials_label = QLabel("admin / admin123 • supervisor / supervisor123 • operator / operator123")
        credentials_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_XS}; 
            color: {ModernUITheme.TEXT_MUTED}; 
            font-family: {ModernUITheme.FONT_FAMILY_MONO};
            background-color: {ModernUITheme.SURFACE};
            padding: {ModernUITheme.SPACE_SM};
            border-radius: {ModernUITheme.RADIUS_SM};
            border: 1px solid {ModernUITheme.BORDER_LIGHT};
        """)

        role_info_label = QLabel("Operator: Basic access • Supervisor: Settings access • Admin: Full access")
        role_info_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_XS}; 
            color: {ModernUITheme.TEXT_MUTED}; 
            margin-top: {ModernUITheme.SPACE_SM};
        """)

        # Reset database button with modern styling
        reset_btn = QPushButton("🔄 Reset Database")
        reset_btn.setObjectName("secondaryButton")
        reset_btn.setToolTip("Reset user database and recreate default users")
        reset_btn.clicked.connect(self.reset_user_database)

        info_layout.addWidget(info_label)
        info_layout.addWidget(credentials_label)
        info_layout.addWidget(role_info_label)
        info_layout.addWidget(reset_btn)

        right_layout.addLayout(header_layout)
        right_layout.addWidget(login_title)
        right_layout.addWidget(form_container)
        right_layout.addStretch()
        right_layout.addWidget(info_container)
        right_panel.setLayout(right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        self.setLayout(main_layout)

        self.password_input.returnPressed.connect(self.login)

    def apply_modern_styles(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {ModernUITheme.BACKGROUND};
                font-family: {ModernUITheme.FONT_FAMILY};
            }}

            #leftPanel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 {ModernUITheme.PRIMARY}, stop:1 {ModernUITheme.PRIMARY_LIGHT});
                border-top-left-radius: {ModernUITheme.RADIUS_XL};
                border-bottom-left-radius: {ModernUITheme.RADIUS_XL};
            }}

            #logoContainer {{
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: {ModernUITheme.RADIUS_XL};
                border: 2px solid rgba(255, 255, 255, 0.2);
            }}

            #rightPanel {{
                background-color: {ModernUITheme.BACKGROUND};
                border-top-right-radius: {ModernUITheme.RADIUS_XL};
                border-bottom-right-radius: {ModernUITheme.RADIUS_XL};
            }}

            #closeButton {{
                background-color: transparent;
                border: none;
                color: {ModernUITheme.TEXT_MUTED};
                font-size: {ModernUITheme.FONT_SIZE_LG};
                font-weight: bold;
                border-radius: {ModernUITheme.RADIUS_MD};
            }}

            #closeButton:hover {{
                background-color: {ModernUITheme.SURFACE};
                color: {ModernUITheme.TEXT_SECONDARY};
            }}

            #formContainer {{
                background-color: transparent;
            }}

            #inputContainer {{
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                background-color: {ModernUITheme.BACKGROUND};
            }}

            #modernInput {{
                border: none;
                padding: {ModernUITheme.SPACE_LG} 0;
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                background-color: transparent;
                color: {ModernUITheme.TEXT_PRIMARY};
                min-height: 20px;
                font-weight: 500;
            }}

            #modernInput:focus {{
                outline: none;
            }}

            #inputContainer:focus-within {{
                border-color: {ModernUITheme.PRIMARY};
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }}

            #modernInput::placeholder {{
                color: {ModernUITheme.TEXT_MUTED};
            }}

            #togglePasswordButton {{
                background-color: transparent;
                border: none;
                padding: 0;
                margin: 0;
                color: {ModernUITheme.TEXT_MUTED};
                qproperty-icon: url(:/icons/eye_closed.svg);
                qproperty-iconSize: 20px;
            }}

            #togglePasswordButton:hover {{
                color: {ModernUITheme.PRIMARY};
            }}

            #primaryButton {{
                background-color: {ModernUITheme.PRIMARY};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-weight: 600;
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                padding: {ModernUITheme.SPACE_LG} {ModernUITheme.SPACE_2XL};
                min-height: 48px;
                min-width: 120px;
            }}

            #primaryButton:hover {{
                background-color: {ModernUITheme.PRIMARY_DARK};
            }}

            #primaryButton:pressed {{
                transform: translateY(0px);
            }}

            #secondaryButton {{
                background-color: {ModernUITheme.WARNING};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_SM};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_MD};
                margin-top: {ModernUITheme.SPACE_MD};
            }}

            #secondaryButton:hover {{
                background-color: #B45309;
            }}

            #infoContainer {{
                background-color: {ModernUITheme.SURFACE};
                border-radius: {ModernUITheme.RADIUS_LG};
                padding: {ModernUITheme.SPACE_2XL};
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
            }}

            #errorLabel {{
                color: {ModernUITheme.DANGER};
                background-color: rgba(220, 38, 38, 0.1);
                font-size: {ModernUITheme.FONT_SIZE_SM};
                padding: {ModernUITheme.SPACE_MD};
                border-radius: {ModernUITheme.RADIUS_MD};
                border: 1px solid rgba(220, 38, 38, 0.2);
            }}

            #loadingSpinner {{
                border: 2px solid rgba(59, 130, 246, 0.3);
                border-radius: 50%;
                border-top: 2px solid {ModernUITheme.PRIMARY};
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
            }}

            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        """)

    def toggle_password_visibility(self):
        """Toggle password visibility with icon change"""
        self.is_password_visible = not self.is_password_visible
        if self.is_password_visible:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_btn.setIcon(QIcon(":/icons/eye_open.svg"))
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_btn.setIcon(QIcon(":/icons/eye_closed.svg"))
    def show_password_reset(self):
        """Show password reset dialog (stub)"""
        QMessageBox.information(self, "Password Reset",
                                "Password reset functionality would be implemented here.\n\n" +
                                "For now, please contact your system administrator.")

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username:
            self.show_error("Please enter your username")
            return
        if not password:
            self.show_error("Please enter your password")
            return

        # Clear previous errors
        self.error_label.hide()

        # Show loading state
        self.set_loading(True)

        try:
            # Authenticate user
            self.user_info = self.user_manager.authenticate_user(username, password)

            if self.user_info:
                logger.info(f"Login successful for: {username}")
                self.accept()
            else:
                # Check if user exists for better error message
                try:
                    with sqlite3.connect(self.user_manager.db_path, timeout=10) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT username, is_active FROM users WHERE username = ?", (username,))
                        user_check = cursor.fetchone()

                        if user_check:
                            if user_check[1] == 0:
                                error_msg = f"Account '{username}' is inactive. Please contact administrator."
                            else:
                                error_msg = f"Invalid password for user '{username}'"
                        else:
                            error_msg = f"User '{username}' not found"
                except Exception as e:
                    error_msg = f"Authentication failed for '{username}'"
                    logger.error(f"Error checking user existence: {e}")

                self.show_error(error_msg)
                logger.warning(f"Login failed for user: {username}")

        except Exception as e:
            error_msg = f"Login error: {str(e)}"
            logger.error(f"Login exception for {username}: {e}")
            self.show_error(error_msg)

        finally:
            # Reset loading state
            self.set_loading(False)

    def show_error(self, message):
        """Show error message in the UI"""
        self.error_label.setText(message)
        self.error_label.show()

    def set_loading(self, loading):
        """Set loading state for login button"""
        if loading:
            self.login_btn.setText("Authenticating...")
            self.login_btn.setEnabled(False)
            self.loading_spinner.show()
        else:
            self.login_btn.setText("Sign In")
            self.login_btn.setEnabled(True)
            self.loading_spinner.hide()

        QApplication.processEvents()

    def reset_user_database(self):
        """Reset user database and recreate default users"""
        try:
            reply = QMessageBox.question(self, "Reset Database",
                                         "This will delete all users and recreate default accounts.\n\n" +
                                         "Default users:\n• admin / admin123\n• supervisor / supervisor123\n• operator / operator123\n\n" +
                                         "Continue?",
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                logger.info("Resetting user database...")

                # Drop and recreate users table
                with sqlite3.connect(self.user_manager.db_path, timeout=10) as conn:
                    cursor = conn.cursor()

                    # Drop existing table
                    cursor.execute("DROP TABLE IF EXISTS users")

                    # Recreate table and default users
                    self.user_manager.init_user_tables()

                    logger.info("User database reset completed")

                QMessageBox.information(self, "Reset Complete",
                                        "User database has been reset.\n\n" +
                                        "You can now login with:\n• admin / admin123\n• supervisor / supervisor123\n• operator / operator123")

        except Exception as e:
            error_msg = f"Error resetting database: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Reset Error", error_msg)
class AudioRecordDialog(QDialog):
    """Modern audio recording dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice Recording")
        self.setFixedSize(450, 350)
        self.recorded_data = None
        self.recorder = AudioRecorder() if AUDIO_AVAILABLE else None
        self.is_recording = False
        self.init_ui()
        self.apply_modern_styles()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel("Voice Recording")
        header_label.setObjectName("dialogHeader")
        layout.addWidget(header_label)

        # Status section
        status_container = QFrame()
        status_container.setObjectName("statusContainer")
        status_layout = QVBoxLayout(status_container)

        self.status_label = QLabel("Ready to record voice note")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)

        layout.addWidget(status_container)

        # Control buttons
        self.record_btn = QPushButton("🎙️ Start Recording")
        self.record_btn.setObjectName("recordButton")
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)

        self.play_btn = QPushButton("▶️ Play Recording")
        self.play_btn.setEnabled(False)
        self.play_btn.setObjectName("playButton")
        self.play_btn.clicked.connect(self.play_recording)
        layout.addWidget(self.play_btn)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        save_btn = QPushButton("Save Recording")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self.accept)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

        if not AUDIO_AVAILABLE:
            self.status_label.setText("Audio recording not available\n(Install sounddevice and soundfile)")
            self.record_btn.setEnabled(False)

        self.setLayout(layout)

    def apply_modern_styles(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {ModernUITheme.BACKGROUND};
                font-family: {ModernUITheme.FONT_FAMILY};
            }}
            
            #dialogHeader {{
                font-size: {ModernUITheme.FONT_SIZE_2XL};
                font-weight: 600;
                color: {ModernUITheme.TEXT_PRIMARY};
                text-align: center;
                margin-bottom: {ModernUITheme.SPACE_XL};
            }}
            
            #statusContainer {{
                background-color: {ModernUITheme.SURFACE};
                border-radius: {ModernUITheme.RADIUS_LG};
                padding: {ModernUITheme.SPACE_2XL};
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
            }}
            
            #statusLabel {{
                font-size: {ModernUITheme.FONT_SIZE_LG};
                color: {ModernUITheme.TEXT_SECONDARY};
                text-align: center;
                font-weight: 500;
            }}
            
            #recordButton {{
                background-color: {ModernUITheme.ERROR};
                color: white;
                font-size: {ModernUITheme.FONT_SIZE_LG};
                font-weight: 600;
                padding: {ModernUITheme.SPACE_LG} {ModernUITheme.SPACE_2XL};
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                min-height: 56px;
            }}
            
            #recordButton:hover {{
                background-color: #B91C1C;
            }}
            
            #recordButton:disabled {{
                background-color: {ModernUITheme.TEXT_DISABLED};
                color: {ModernUITheme.TEXT_MUTED};
            }}
            
            #playButton {{
                background-color: {ModernUITheme.SUCCESS};
                color: white;
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_XL};
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                min-height: 44px;
            }}
            
            #playButton:hover {{
                background-color: #047857;
            }}
            
            #playButton:disabled {{
                background-color: {ModernUITheme.TEXT_DISABLED};
                color: {ModernUITheme.TEXT_MUTED};
            }}
            
            #primaryButton {{
                background-color: {ModernUITheme.PRIMARY};
                color: white;
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 600;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                min-height: 44px;
            }}
            
            #primaryButton:hover {{
                background-color: {ModernUITheme.PRIMARY_DARK};
            }}
            
            #secondaryButton {{
                background-color: transparent;
                color: {ModernUITheme.TEXT_SECONDARY};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                min-height: 44px;
            }}
            
            #secondaryButton:hover {{
                background-color: {ModernUITheme.SURFACE};
                border-color: {ModernUITheme.TEXT_MUTED};
            }}
        """)

    def toggle_recording(self):
        if not self.recorder:
            return

        if not self.is_recording:
            try:
                self.is_recording = True
                self.record_btn.setText("⏹️ Stop Recording")
                self.record_btn.setStyleSheet(f"""
                    background-color: {ModernUITheme.WARNING};
                    color: white;
                    font-size: {ModernUITheme.FONT_SIZE_LG};
                    font-weight: 600;
                    padding: {ModernUITheme.SPACE_LG} {ModernUITheme.SPACE_2XL};
                    border: none;
                    border-radius: {ModernUITheme.RADIUS_MD};
                    min-height: 56px;
                """)
                self.status_label.setText("Recording... Click stop when finished")
                self.play_btn.setEnabled(False)

                self.recorder.start_recording(self.on_recording_finished)

            except Exception as e:
                QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {e}")
                self.is_recording = False
        else:
            self.recorder.stop_recording()

    def on_recording_finished(self, success, data):
        """Handle recording completion"""
        self.is_recording = False
        self.record_btn.setText("🎙️ Start Recording")
        self.record_btn.setStyleSheet(f"""
            background-color: {ModernUITheme.ERROR};
            color: white;
            font-size: {ModernUITheme.FONT_SIZE_LG};
            font-weight: 600;
            padding: {ModernUITheme.SPACE_LG} {ModernUITheme.SPACE_2XL};
            border: none;
            border-radius: {ModernUITheme.RADIUS_MD};
            min-height: 56px;
        """)

        if success and isinstance(data, bytes):
            self.recorded_data = data
            self.status_label.setText("Recording saved! You can play it back or save.")
            self.play_btn.setEnabled(True)
        else:
            self.status_label.setText(f"Recording failed: {data}")

    def play_recording(self):
        if self.recorded_data and self.recorder:
            self.status_label.setText("Playing recording...")
            self.recorder.play_audio(self.recorded_data, self.on_playback_finished)

    def on_playback_finished(self, success, message):
        """Handle playback completion"""
        if success:
            self.status_label.setText("Recording saved! You can play it back or save.")
        else:
            self.status_label.setText(f"Playback failed: {message}")

class UserManagementDialog(QDialog):
    """Modern user management dialog"""

    def __init__(self, user_manager, current_user, user_data=None, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.current_user = current_user
        self.user_data = user_data
        self.is_edit_mode = user_data is not None

        title = "Edit User" if self.is_edit_mode else "Add New User"
        self.setWindowTitle(title)
        self.setFixedSize(550, 450)

        self.init_ui()
        self.apply_modern_styles()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel("Edit User" if self.is_edit_mode else "Add New User")
        header_label.setObjectName("dialogHeader")
        layout.addWidget(header_label)

        # Form container
        form_container = QFrame()
        form_container.setObjectName("formContainer")
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(int(ModernUITheme.SPACE_XL.replace('px', '')))

        # Username field
        username_group = QVBoxLayout()
        username_label = QLabel("Username")
        username_label.setObjectName("fieldLabel")

        self.username_input = QLineEdit()
        self.username_input.setObjectName("modernInput")
        self.username_input.setPlaceholderText("Enter username")
        if self.is_edit_mode:
            self.username_input.setText(self.user_data[1])
            self.username_input.setEnabled(False)  # Don't allow username changes

        username_group.addWidget(username_label)
        username_group.addWidget(self.username_input)

        # Full name field
        fullname_group = QVBoxLayout()
        fullname_label = QLabel("Full Name")
        fullname_label.setObjectName("fieldLabel")

        self.full_name_input = QLineEdit()
        self.full_name_input.setObjectName("modernInput")
        self.full_name_input.setPlaceholderText("Enter full name")
        if self.is_edit_mode:
            self.full_name_input.setText(self.user_data[2] or "")

        fullname_group.addWidget(fullname_label)
        fullname_group.addWidget(self.full_name_input)

        # Role field
        role_group = QVBoxLayout()
        role_label = QLabel("Role")
        role_label.setObjectName("fieldLabel")

        self.role_combo = QComboBox()
        self.role_combo.setObjectName("modernCombo")
        self.role_combo.addItems(["operator", "supervisor", "admin"])
        if self.is_edit_mode:
            index = self.role_combo.findText(self.user_data[3])
            if index >= 0:
                self.role_combo.setCurrentIndex(index)

        role_group.addWidget(role_label)
        role_group.addWidget(self.role_combo)

        # Password fields
        password_group = QVBoxLayout()
        password_label = QLabel("Password")
        password_label.setObjectName("fieldLabel")

        self.password_input = QLineEdit()
        self.password_input.setObjectName("modernInput")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter password" if not self.is_edit_mode else "Leave empty to keep current password")

        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setObjectName("modernInput")
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setPlaceholderText("Confirm password")

        password_group.addWidget(password_label)
        password_group.addWidget(self.password_input)
        password_group.addWidget(self.confirm_password_input)

        # Status checkbox for edit mode
        if self.is_edit_mode:
            status_group = QVBoxLayout()
            status_label = QLabel("Account Status")
            status_label.setObjectName("fieldLabel")

            self.active_checkbox = QCheckBox("Account Active")
            self.active_checkbox.setObjectName("modernCheckbox")
            self.active_checkbox.setChecked(self.user_data[4] == 1)

            status_group.addWidget(status_label)
            status_group.addWidget(self.active_checkbox)
            form_layout.addLayout(status_group)

        form_layout.addLayout(username_group)
        form_layout.addLayout(fullname_group)
        form_layout.addLayout(role_group)
        form_layout.addLayout(password_group)

        layout.addWidget(form_container)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        save_btn = QPushButton("Save User")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self.save_user)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

    def apply_modern_styles(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {ModernUITheme.BACKGROUND};
                font-family: {ModernUITheme.FONT_FAMILY};
            }}
            
            #dialogHeader {{
                font-size: {ModernUITheme.FONT_SIZE_2XL};
                font-weight: 600;
                color: {ModernUITheme.TEXT_PRIMARY};
                margin-bottom: {ModernUITheme.SPACE_XL};
            }}
            
            #formContainer {{
                background-color: {ModernUITheme.SURFACE};
                border-radius: {ModernUITheme.RADIUS_LG};
                padding: {ModernUITheme.SPACE_2XL};
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
            }}
            
            #fieldLabel {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 600;
                color: {ModernUITheme.TEXT_SECONDARY};
                margin-bottom: {ModernUITheme.SPACE_SM};
            }}
            
            #modernInput {{
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                background-color: {ModernUITheme.BACKGROUND};
                color: {ModernUITheme.TEXT_PRIMARY};
                min-height: 20px;
                font-weight: 500;
            }}
            
            #modernInput:focus {{
                border-color: {ModernUITheme.PRIMARY};
                outline: none;
            }}
            
            #modernInput:disabled {{
                background-color: {ModernUITheme.SURFACE};
                color: {ModernUITheme.TEXT_MUTED};
            }}
            
            #modernCombo {{
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                background-color: {ModernUITheme.BACKGROUND};
                color: {ModernUITheme.TEXT_PRIMARY};
                min-height: 20px;
                font-weight: 500;
            }}
            
            #modernCombo:focus {{
                border-color: {ModernUITheme.PRIMARY};
            }}
            
            #modernCombo::drop-down {{
                border: none;
                width: 20px;
            }}
            
            #modernCombo::down-arrow {{
                image: none;
                border: none;
                width: 12px;
                height: 12px;
            }}
            
            #modernCheckbox {{
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                color: {ModernUITheme.TEXT_PRIMARY};
                font-weight: 500;
            }}
            
            #modernCheckbox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: 4px;
                background-color: {ModernUITheme.BACKGROUND};
            }}
            
            #modernCheckbox::indicator:checked {{
                background-color: {ModernUITheme.PRIMARY};
                border-color: {ModernUITheme.PRIMARY};
            }}
            
            #primaryButton {{
                background-color: {ModernUITheme.PRIMARY};
                color: white;
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 600;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                min-height: 44px;
            }}
            
            #primaryButton:hover {{
                background-color: {ModernUITheme.PRIMARY_DARK};
            }}
            
            #secondaryButton {{
                background-color: transparent;
                color: {ModernUITheme.TEXT_SECONDARY};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                min-height: 44px;
            }}
            
            #secondaryButton:hover {{
                background-color: {ModernUITheme.SURFACE};
                border-color: {ModernUITheme.TEXT_MUTED};
            }}
        """)

    def save_user(self):
        username = self.username_input.text().strip()
        full_name = self.full_name_input.text().strip()
        role = self.role_combo.currentText()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()

        # Validation
        if not username or not full_name:
            QMessageBox.warning(self, "Validation Error", "Username and full name are required")
            return

        if not self.is_edit_mode and not password:
            QMessageBox.warning(self, "Validation Error", "Password is required for new users")
            return

        if password and password != confirm_password:
            QMessageBox.warning(self, "Validation Error", "Passwords do not match")
            return

        if password and len(password) < 6:
            QMessageBox.warning(self, "Validation Error", "Password must be at least 6 characters long")
            return

        try:
            if self.is_edit_mode:
                # Update existing user
                user_id = self.user_data[0]
                is_active = self.active_checkbox.isChecked() if hasattr(self, 'active_checkbox') else True

                success = self.user_manager.update_user(user_id, full_name, role, is_active)

                if password:  # Update password if provided
                    success = success and self.user_manager.change_password(user_id, password)

                if success:
                    QMessageBox.information(self, "Success", "User updated successfully")
                    self.accept()
                else:
                    QMessageBox.critical(self, "Error", "Failed to update user")
            else:
                # Add new user
                success = self.user_manager.add_user(username, password, full_name, role, self.current_user['username'])

                if success:
                    QMessageBox.information(self, "Success", "User added successfully")
                    self.accept()
                else:
                    QMessageBox.critical(self, "Error", "Failed to add user. Username may already exist.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {str(e)}")
# TODO: This part is causing UI freezing. Please suggest a refactor using QThread or async technique to prevent Not Responding.
class DatabaseWorker(QThread):
    """CORRECTED Database Worker Thread for async database operations"""

    # Signal definitions
    operation_started = pyqtSignal(str, str)  # operation_id, description
    operation_progress = pyqtSignal(str, int, str)  # operation_id, percentage, message
    operation_completed = pyqtSignal(str, object)  # operation_id, result
    operation_error = pyqtSignal(str, str)  # operation_id, error_message

    # Specific data signals
    bans_loaded = pyqtSignal(list)
    logs_loaded = pyqtSignal(list)
    users_loaded = pyqtSignal(list)
    statistics_loaded = pyqtSignal(dict)
    verification_completed = pyqtSignal(str, str, dict)  # status, reason, details
    connection_tested = pyqtSignal(str, bool, str)  # connection_type, success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.operations_queue = queue.Queue()
        self.current_operation = None
        self.is_running = True
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()

    def add_operation(self, operation_type, operation_id, db_instance, *args, **kwargs):
        """Add operation to queue for async execution"""
        operation = {
            'type': operation_type,
            'id': operation_id,
            'db': db_instance,
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.time()
        }
        self.operations_queue.put(operation)

        # Wake up thread if sleeping
        self.mutex.lock()
        self.wait_condition.wakeOne()
        self.mutex.unlock()

        if not self.isRunning():
            self.start()

    def run(self):
        """Main thread execution loop"""
        logger.info("DatabaseWorker thread started")

        while self.is_running:
            try:
                # Get operation from queue with timeout
                try:
                    operation = self.operations_queue.get(timeout=1.0)
                except queue.Empty:
                    # No operations, wait for signal
                    self.mutex.lock()
                    self.wait_condition.wait(self.mutex, 5000)  # 5 second timeout
                    self.mutex.unlock()
                    continue

                self.current_operation = operation
                self._execute_operation(operation)
                self.current_operation = None

            except Exception as e:
                logger.error(f"DatabaseWorker thread error: {e}")
                if self.current_operation:
                    self.operation_error.emit(self.current_operation['id'], str(e))
                    self.current_operation = None

    def _execute_operation(self, operation):
        """Execute individual database operation"""
        try:
            op_type = operation['type']
            op_id = operation['id']
            db = operation['db']
            args = operation['args']
            kwargs = operation['kwargs']

            # Emit operation started signal
            description = self._get_operation_description(op_type)
            self.operation_started.emit(op_id, description)

            # Execute operation based on type
            result = None

            if op_type == 'load_bans':
                result = self._load_bans(db, *args, **kwargs)
                self.bans_loaded.emit(result)

            elif op_type == 'load_logs':
                result = self._load_logs(db, *args, **kwargs)
                self.logs_loaded.emit(result)

            elif op_type == 'load_users':
                result = self._load_users(db, *args, **kwargs)
                self.users_loaded.emit(result)

            elif op_type == 'load_statistics':
                result = self._load_statistics(db, *args, **kwargs)
                self.statistics_loaded.emit(result)

            elif op_type == 'verify_tanker':
                status, reason, details = self._verify_tanker(db, *args, **kwargs)
                result = (status, reason, details)
                self.verification_completed.emit(status, reason, details)

            elif op_type == 'test_connection':
                success, message = self._test_connection(db, *args, **kwargs)
                result = (success, message)
                connection_type = args[0] if args else 'unknown'
                self.connection_tested.emit(connection_type, success, message)

            else:
                raise ValueError(f"Unknown operation type: {op_type}")

            # Emit completion signal
            self.operation_completed.emit(op_id, result)

        except Exception as e:
            logger.error(f"Error executing operation {op_type}: {e}")
            self.operation_error.emit(op_id, str(e))

    def _get_operation_description(self, op_type):
        """Get human-readable description for operation type"""
        descriptions = {
            'load_bans': 'Loading ban records...',
            'load_logs': 'Loading activity logs...',
            'load_users': 'Loading user accounts...',
            'load_statistics': 'Calculating statistics...',
            'verify_tanker': 'Verifying tanker...',
            'test_connection': 'Testing database connection...'
        }
        return descriptions.get(op_type, f'Executing {op_type}...')

    # ===== CORRECTED OPERATION IMPLEMENTATIONS =====

    def _load_bans(self, db, filters=None):
        """Load ban records - CALLS EXISTING DatabaseManager methods"""
        self.operation_progress.emit(self.current_operation['id'], 10, "Loading ban records...")

        try:
            # Use existing DatabaseManager method
            result = db.get_all_bans(filters)

            self.operation_progress.emit(self.current_operation['id'], 100, "Ban records loaded")
            logger.info(f"Loaded {len(result)} ban records")
            return result

        except Exception as e:
            logger.error(f"Error loading bans: {e}")
            raise

    def _load_logs(self, db, limit=50, filters=None):
        """Load activity logs - CALLS EXISTING DatabaseManager methods"""
        self.operation_progress.emit(self.current_operation['id'], 10, "Loading activity logs...")

        try:
            # Use existing DatabaseManager method
            result = db.get_recent_logs(limit, filters)

            self.operation_progress.emit(self.current_operation['id'], 100, "Activity logs loaded")
            logger.info(f"Loaded {len(result)} log records")
            return result

        except Exception as e:
            logger.error(f"Error loading logs: {e}")
            raise

    def _load_users(self, user_manager):
        """Load user accounts - CALLS EXISTING UserManager methods"""
        self.operation_progress.emit(self.current_operation['id'], 20, "Loading user accounts...")

        try:
            # Use existing UserManager method
            result = user_manager.get_all_users()

            self.operation_progress.emit(self.current_operation['id'], 100, "User accounts loaded")
            logger.info(f"Loaded {len(result)} user accounts")
            return result

        except Exception as e:
            logger.error(f"Error loading users: {e}")
            raise

    def _load_statistics(self, db, filters=None):
        """Load dashboard statistics - CALLS EXISTING DatabaseManager methods"""
        self.operation_progress.emit(self.current_operation['id'], 10, "Calculating statistics...")

        try:
            self.operation_progress.emit(self.current_operation['id'], 30, "Loading ban statistics...")
            ban_stats = db.get_ban_statistics(filters)

            self.operation_progress.emit(self.current_operation['id'], 60, "Loading verification statistics...")
            verify_stats = db.get_verification_statistics(filters)

            self.operation_progress.emit(self.current_operation['id'], 80, "Loading recent data...")
            recent_bans = db.get_all_bans(filters)[:10] if hasattr(db, 'get_all_bans') else []
            recent_logs = db.get_recent_logs(15, filters) if hasattr(db, 'get_recent_logs') else []

            self.operation_progress.emit(self.current_operation['id'], 100, "Statistics complete")

            result = {
                'ban_stats': ban_stats,
                'verify_stats': verify_stats,
                'recent_bans': recent_bans,
                'recent_logs': recent_logs
            }

            logger.info("Dashboard statistics calculated successfully")
            return result

        except Exception as e:
            logger.error(f"Error loading statistics: {e}")
            raise

    def _verify_tanker(self, db, tanker_number, operator):
        """Verify tanker - CALLS EXISTING DatabaseManager methods"""
        tanker_display = tanker_number or 'latest tanker'
        self.operation_progress.emit(self.current_operation['id'], 20, f"Verifying {tanker_display}...")

        try:
            if tanker_number:
                # Use existing DatabaseManager method
                status, reason, details = db.verify_specific_tanker(tanker_number, operator)
            else:
                # Use existing DatabaseManager method
                status, reason, details = db.simple_tanker_verification(operator)

            self.operation_progress.emit(self.current_operation['id'], 100, "Verification complete")

            logger.info(f"Tanker verification completed: {status}")
            return status, reason, details

        except Exception as e:
            logger.error(f"Error verifying tanker: {e}")
            raise

    def _test_connection(self, db, connection_type):
        """Test database connection - CALLS EXISTING DatabaseManager methods"""
        self.operation_progress.emit(self.current_operation['id'], 30, f"Testing {connection_type} connection...")

        try:
            if connection_type == 'server':
                # Use existing DatabaseManager method
                db.test_server_connection()  # This is the CORRECT method name
                success = db.server_available
                message = "Server connection successful" if success else "Server connection failed"

            elif connection_type == 'local':
                # Test local SQLite connection directly
                try:
                    with sqlite3.connect(db.sqlite_db, timeout=10) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                    success = True
                    message = f"Local connection successful ({len(tables)} tables)"
                except Exception as e:
                    success = False
                    message = f"Local connection failed: {str(e)}"

            else:
                raise ValueError(f"Unknown connection type: {connection_type}")

            self.operation_progress.emit(self.current_operation['id'], 100, "Connection test complete")

            logger.info(f"Connection test {connection_type}: {'Success' if success else 'Failed'}")
            return success, message

        except Exception as e:
            logger.error(f"Error testing {connection_type} connection: {e}")
            return False, str(e)

    def stop_thread(self):
        """Stop the thread gracefully"""
        logger.info("Stopping DatabaseWorker thread...")
        self.is_running = False
        self.mutex.lock()
        self.wait_condition.wakeAll()
        self.mutex.unlock()

        if self.isRunning():
            self.wait(5000)  # Wait up to 5 seconds for thread to finish

        logger.info("DatabaseWorker thread stopped")

class LoadingOverlay(QWidget):
    """Loading overlay that covers the entire widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            LoadingOverlay {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Spinner
        self.spinner_label = QLabel("⏳")
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_label.setStyleSheet("""
            QLabel {
                color: #3B82F6;
                font-size: 32px;
                margin-bottom: 16px;
            }
        """)
        layout.addWidget(self.spinner_label)

        # Message
        self.message_label = QLabel("Loading...")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("""
            QLabel {
                color: #6B7280;
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 16px;
            }
        """)
        layout.addWidget(self.message_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #E5E7EB;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Animation timer for spinner
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spinner)
        self.spinner_chars = ["⏳", "⌛", "⏳", "⌛"]
        self.spinner_index = 0

        self.hide()

    def show_loading(self, message="Loading..."):
        """Show loading overlay"""
        self.message_label.setText(message)
        self.progress_bar.setValue(0)
        self.timer.start(500)  # Update every 500ms
        self.show()
        self.raise_()

    def hide_loading(self):
        """Hide loading overlay"""
        self.timer.stop()
        self.hide()

    def update_progress(self, percentage, message=None):
        """Update loading progress"""
        self.progress_bar.setValue(percentage)
        if message:
            self.message_label.setText(message)

    def update_spinner(self):
        """Update spinner animation"""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
        self.spinner_label.setText(self.spinner_chars[self.spinner_index])
class MainWindow(QMainWindow):
    """Enhanced main window with Modern UI and Fixed Filters"""

    def __init__(self, user_info, config_manager):
        super().__init__()
        self.user_info = user_info
        self.config = config_manager
        self.db_worker = DatabaseWorker()
        self.setup_database_signals()
        self.loading_overlays = {}
        self.active_operations = {}
        self.operation_counter = 0
        self.last_tanker = None
        self.auto_switch_enabled = False
        self.notification_enabled = True
        self.audio_recorder = AudioRecorder() if AUDIO_AVAILABLE else None

        # Initialize warning sound system
        self.warning_sound = WarningSound(
            duration=self.config.get('warning_sound_duration', 3.0)
        ) if AUDIO_AVAILABLE else None
        self.sound_enabled = self.config.get('sound_enabled', True)
        self.current_sound_playing = False

        # Initialize database with config
        self.db = DatabaseManager(self.config)

        # Initialize user manager
        self.user_manager = UserManager(self.config.get('local_sqlite_path'))

        self.setWindowTitle(f"TDF System - Modern UI - User: {user_info['full_name']} ({user_info['role']})")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize filter states
        self.dashboard_filters_applied = False
        self.ban_filters_applied = False
        self.current_dashboard_filters = None
        self.current_ban_filters = None

        self.init_ui()
        self.apply_modern_styles()

        # Use single shot timer to avoid blocking
        QTimer.singleShot(500, self.initial_dashboard_load_async)
        QTimer.singleShot(1000, self.start_monitoring)

        logger.info(f"Modern UI main window initialized for user: {user_info['username']} with role: {user_info['role']}")
        self.setup_loading_overlays()
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Modern status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready...")
        self.status_bar.setObjectName("modernStatusBar")

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.create_modern_sidebar()
        main_layout.addWidget(self.sidebar)

        self.create_modern_content_area()
        main_layout.addWidget(self.content_area, 1)

    def create_modern_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar.setObjectName("modernSidebar")
        self.sidebar.setFixedWidth(280)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))
        sidebar_layout.setContentsMargins(20, 30, 20, 30)

        # Modern title section
        title_container = QFrame()
        title_container.setObjectName("titleContainer")
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(int(ModernUITheme.SPACE_SM.replace('px', '')))

        title = QLabel("TDF System")
        title.setObjectName("sidebarTitle")
        title_layout.addWidget(title)

        subtitle = QLabel("Modern Interface")
        subtitle.setObjectName("sidebarSubtitle")
        title_layout.addWidget(subtitle)

        sidebar_layout.addWidget(title_container)

        # Modern navigation buttons
        nav_container = QFrame()
        nav_container.setObjectName("navContainer")
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setSpacing(int(ModernUITheme.SPACE_SM.replace('px', '')))

        nav_buttons = [
            ("📊", "Dashboard", self.show_dashboard, True),
            ("✅", "Auto Verification", self.show_verification, True),
            ("🔍", "Manual Verify", self.show_manual_verify, True),
            ("⛔", "Ban Management", self.show_bans, True),
            ("📝", "Activity Logs", self.show_logs, True),
        ]

        # Add Settings button with role-based access
        if self.user_info['role'] in ['admin', 'supervisor']:
            nav_buttons.append(("⚙️", "Settings", self.show_settings, True))

        for icon, text, callback, enabled in nav_buttons:
            btn_container = QFrame()
            btn_container.setObjectName("navButtonContainer")
            btn_layout = QHBoxLayout(btn_container)
            btn_layout.setContentsMargins(16, 12, 16, 12)
            btn_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

            icon_label = QLabel(icon)
            icon_label.setObjectName("navIcon")
            btn_layout.addWidget(icon_label)

            text_label = QLabel(text)
            text_label.setObjectName("navText")
            btn_layout.addWidget(text_label)

            btn_layout.addStretch()

            # Make the container clickable
            btn_container.mousePressEvent = lambda event, cb=callback: cb()
            btn_container.setEnabled(enabled)

            if not enabled:
                btn_container.setToolTip("Access restricted for your role")
                btn_container.setObjectName("navButtonContainerDisabled")

            nav_layout.addWidget(btn_container)

        sidebar_layout.addWidget(nav_container)

        # Add role restriction notice for operators
        if self.user_info['role'] == 'operator':
            restriction_container = QFrame()
            restriction_container.setObjectName("restrictionNotice")
            restriction_layout = QVBoxLayout(restriction_container)
            restriction_layout.setContentsMargins(16, 12, 16, 12)

            restriction_label = QLabel("⚠️ Operator Access")
            restriction_label.setObjectName("restrictionTitle")
            restriction_layout.addWidget(restriction_label)

            restriction_desc = QLabel("Settings restricted")
            restriction_desc.setObjectName("restrictionDesc")
            restriction_layout.addWidget(restriction_desc)

            sidebar_layout.addWidget(restriction_container)

        sidebar_layout.addStretch()

        # Modern sound control section
        sound_container = QFrame()
        sound_container.setObjectName("soundContainer")
        sound_layout = QVBoxLayout(sound_container)
        sound_layout.setContentsMargins(16, 16, 16, 16)
        sound_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        sound_header = QLabel("🔊 Sound Alerts")
        sound_header.setObjectName("soundHeader")
        sound_layout.addWidget(sound_header)

        sound_controls = QHBoxLayout()
        sound_controls.setSpacing(int(ModernUITheme.SPACE_SM.replace('px', '')))

        self.sound_toggle_btn = QPushButton("ON" if self.sound_enabled else "OFF")
        self.sound_toggle_btn.setObjectName("soundToggle")
        self.sound_toggle_btn.setCheckable(True)
        self.sound_toggle_btn.setChecked(self.sound_enabled)
        self.sound_toggle_btn.clicked.connect(self.toggle_sound)
        sound_controls.addWidget(self.sound_toggle_btn)

        self.stop_sound_btn = QPushButton("Stop")
        self.stop_sound_btn.setObjectName("soundStop")
        self.stop_sound_btn.clicked.connect(self.stop_warning_sound)
        sound_controls.addWidget(self.stop_sound_btn)

        sound_layout.addLayout(sound_controls)
        sidebar_layout.addWidget(sound_container)

        # Modern user info section
        user_container = QFrame()
        user_container.setObjectName("userContainer")
        user_layout = QVBoxLayout(user_container)
        user_layout.setContentsMargins(16, 16, 16, 16)
        user_layout.setSpacing(int(ModernUITheme.SPACE_SM.replace('px', '')))

        user_name = QLabel(f"👤 {self.user_info['full_name']}")
        user_name.setObjectName("userName")
        user_layout.addWidget(user_name)

        user_role = QLabel(f"🔧 {self.user_info['role'].title()}")
        user_role.setObjectName(f"userRole{self.user_info['role'].title()}")
        user_layout.addWidget(user_role)

        sidebar_layout.addWidget(user_container)

    def setup_database_signals(self):
        """Setup database worker signals"""
        self.db_worker.operation_started.connect(self.on_operation_started)
        self.db_worker.operation_progress.connect(self.on_operation_progress)
        self.db_worker.operation_completed.connect(self.on_operation_completed)
        self.db_worker.operation_error.connect(self.on_operation_error)

        # Specific data signals
        self.db_worker.bans_loaded.connect(self.on_bans_loaded)
        self.db_worker.logs_loaded.connect(self.on_logs_loaded)
        self.db_worker.users_loaded.connect(self.on_users_loaded)
        self.db_worker.statistics_loaded.connect(self.on_statistics_loaded)
        self.db_worker.verification_completed.connect(self.on_verification_completed)
        self.db_worker.connection_tested.connect(self.on_connection_tested)

    def setup_loading_overlays(self):
        """Setup loading overlays for each page"""
        if hasattr(self, 'dashboard_page'):
            self.loading_overlays['dashboard'] = LoadingOverlay(self.dashboard_page)

        if hasattr(self, 'bans_page'):
            self.loading_overlays['bans'] = LoadingOverlay(self.bans_page)

        if hasattr(self, 'logs_page'):
            self.loading_overlays['logs'] = LoadingOverlay(self.logs_page)

        # Position overlays
        for page_name, overlay in self.loading_overlays.items():
            if page_name == 'dashboard' and hasattr(self, 'dashboard_page'):
                self._setup_overlay_resize(self.dashboard_page, overlay)
            elif page_name == 'bans' and hasattr(self, 'bans_page'):
                self._setup_overlay_resize(self.bans_page, overlay)
            elif page_name == 'logs' and hasattr(self, 'logs_page'):
                self._setup_overlay_resize(self.logs_page, overlay)

    def _setup_overlay_resize(self, page_widget, overlay):
        """Setup overlay resizing for a page"""

        def resize_overlay():
            overlay.setGeometry(page_widget.rect())

        original_resize = page_widget.resizeEvent

        def new_resize_event(event):
            if original_resize:
                original_resize(event)
            else:
                QWidget.resizeEvent(page_widget, event)
            resize_overlay()

        page_widget.resizeEvent = new_resize_event
        resize_overlay()  # Initial positioning

    def generate_operation_id(self):
        """Generate unique operation ID"""
        self.operation_counter += 1
        return f"op_{self.operation_counter}_{int(time.time())}"

    def initial_dashboard_load_async(self):
        """Load initial dashboard data asynchronously"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'initial_dashboard_load'

            if 'dashboard' in self.loading_overlays:
                self.loading_overlays['dashboard'].show_loading("Loading initial dashboard data...")

            self.db_worker.add_operation('load_statistics', operation_id, self.db, self.current_dashboard_filters)

            logger.info("Initial dashboard load started asynchronously")
        except Exception as e:
            logger.error(f"Error starting initial dashboard load: {e}")
            self.status_bar.showMessage(f"Dashboard load error: {e}")

    def refresh_dashboard_async(self):
        """Refresh dashboard asynchronously"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'dashboard_refresh'

            if 'dashboard' in self.loading_overlays:
                self.loading_overlays['dashboard'].show_loading("Refreshing dashboard...")

            filters = self.current_dashboard_filters if self.dashboard_filters_applied else None
            self.db_worker.add_operation('load_statistics', operation_id, self.db, filters)

            logger.info("Dashboard refresh started asynchronously")
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")
            self.status_bar.showMessage(f"Dashboard refresh error: {e}")

    def load_bans_table_async(self):
        """Load bans table asynchronously"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'load_bans'

            if 'bans' in self.loading_overlays:
                self.loading_overlays['bans'].show_loading("Loading ban records...")

            if hasattr(self, 'bans_table'):
                self.bans_table.setEnabled(False)

            filters = self.current_ban_filters if self.ban_filters_applied else None
            self.db_worker.add_operation('load_bans', operation_id, self.db, filters)

            logger.info("Ban table load started asynchronously")
        except Exception as e:
            logger.error(f"Error loading bans table: {e}")
            self.status_bar.showMessage(f"Bans table load error: {e}")

    def load_logs_table_async(self, filters=None):
        """Load logs table asynchronously"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'load_logs'

            if 'logs' in self.loading_overlays:
                self.loading_overlays['logs'].show_loading("Loading activity logs...")

            if hasattr(self, 'logs_table'):
                self.logs_table.setEnabled(False)

            if filters is None:
                filters = self.get_current_log_filters()

            self.db_worker.add_operation('load_logs', operation_id, self.db, 100, filters)

            logger.info("Logs table load started asynchronously")
        except Exception as e:
            logger.error(f"Error loading logs table: {e}")
            self.status_bar.showMessage(f"Logs table load error: {e}")

    def verify_latest_tanker_async(self):
        """Verify latest tanker asynchronously"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'verify_latest'

            self.status_bar.showMessage("Verifying latest tanker...")

            self.db_worker.add_operation('verify_tanker', operation_id, self.db, None, self.user_info['username'])

            logger.info("Latest tanker verification started asynchronously")
        except Exception as e:
            logger.error(f"Error verifying latest tanker: {e}")
            self.status_bar.showMessage(f"Verification error: {e}")

    def verify_manual_tanker_async(self):
        """Verify manually entered tanker asynchronously"""
        try:
            if not hasattr(self, 'manual_tanker_input'):
                return

            tanker_number = self.manual_tanker_input.text().strip().upper()
            if not tanker_number:
                QMessageBox.warning(self, "Input Required", "Please enter a tanker number")
                return

            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'verify_manual'

            self.status_bar.showMessage(f"Verifying tanker: {tanker_number}")

            self.db_worker.add_operation('verify_tanker', operation_id, self.db, tanker_number,
                                         self.user_info['username'])

            logger.info(f"Manual tanker verification started asynchronously: {tanker_number}")
        except Exception as e:
            logger.error(f"Error verifying manual tanker: {e}")
            self.status_bar.showMessage(f"Manual verification error: {e}")

    def test_server_connection_async(self):
        """Test server connection asynchronously - CORRECTED"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'test_server'

            if hasattr(self, 'server_status_label'):
                self.server_status_label.setText("Server Status: Testing...")
                self.server_status_label.setStyleSheet("color: #6B7280;")

            self.status_bar.showMessage("Testing server connection...")

            # CORRECTED: Pass 'server' as argument, not call async method on db
            self.db_worker.add_operation('test_connection', operation_id, self.db, 'server')

            logger.info("Server connection test started asynchronously")
        except Exception as e:
            logger.error(f"Error testing server connection: {e}")
            if hasattr(self, 'server_status_label'):
                self.server_status_label.setText(f"Server Status: ❌ Error: {str(e)[:30]}")

    def test_local_connection_async(self):
        """Test local connection asynchronously - CORRECTED"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'test_local'

            if hasattr(self, 'local_status_label'):
                self.local_status_label.setText("Local Status: Testing...")
                self.local_status_label.setStyleSheet("color: #6B7280;")

            self.status_bar.showMessage("Testing local connection...")

            # CORRECTED: Pass 'local' as argument, not call async method on db
            self.db_worker.add_operation('test_connection', operation_id, self.db, 'local')

            logger.info("Local connection test started asynchronously")
        except Exception as e:
            logger.error(f"Error testing local connection: {e}")
            if hasattr(self, 'local_status_label'):
                self.local_status_label.setText(f"Local Status: ❌ Error: {str(e)[:30]}")
    def on_operation_started(self, operation_id, description):
        """Handle operation started signal"""
        logger.info(f"Operation started: {operation_id} - {description}")
        self.status_bar.showMessage(description)

    def on_operation_progress(self, operation_id, percentage, message):
        """Handle operation progress signal"""
        operation_type = self.active_operations.get(operation_id)
        if operation_type:
            if operation_type in ['dashboard_refresh', 'initial_dashboard_load']:
                if 'dashboard' in self.loading_overlays:
                    self.loading_overlays['dashboard'].update_progress(percentage, message)
            elif operation_type == 'load_bans':
                if 'bans' in self.loading_overlays:
                    self.loading_overlays['bans'].update_progress(percentage, message)
            elif operation_type == 'load_logs':
                if 'logs' in self.loading_overlays:
                    self.loading_overlays['logs'].update_progress(percentage, message)

        if percentage < 100:
            self.status_bar.showMessage(f"{message} ({percentage}%)")

    def on_operation_completed(self, operation_id, result):
        """Handle operation completed signal"""
        operation_type = self.active_operations.pop(operation_id, None)
        if operation_type:
            logger.info(f"Operation completed: {operation_type}")
            self.status_bar.showMessage(f"{operation_type.replace('_', ' ').title()} completed successfully")

    def on_operation_error(self, operation_id, error_message):
        """Handle operation error signal"""
        operation_type = self.active_operations.pop(operation_id, None)

        logger.error(f"Operation error: {operation_type} - {error_message}")

        # Hide all loading overlays
        for overlay in self.loading_overlays.values():
            overlay.hide_loading()

        # Re-enable tables
        if hasattr(self, 'bans_table'):
            self.bans_table.setEnabled(True)
        if hasattr(self, 'logs_table'):
            self.logs_table.setEnabled(True)

        # Show error message
        self.status_bar.showMessage(f"Error: {error_message}")

        error_title = f"Error in {operation_type.replace('_', ' ').title()}" if operation_type else "Operation Error"
        QMessageBox.critical(self, error_title, f"Operation failed:\n\n{error_message}")

    def on_bans_loaded(self, bans_data):
        """Handle bans loaded signal"""
        try:
            if 'bans' in self.loading_overlays:
                self.loading_overlays['bans'].hide_loading()

            if hasattr(self, 'bans_table'):
                self.bans_table.setEnabled(True)

            # Use your existing populate method or add this:
            self.populate_bans_table_from_data(bans_data)

            logger.info(f"Bans table updated with {len(bans_data)} records")

        except Exception as e:
            logger.error(f"Error handling bans loaded: {e}")

    def on_logs_loaded(self, logs_data):
        """Handle logs loaded signal"""
        try:
            if 'logs' in self.loading_overlays:
                self.loading_overlays['logs'].hide_loading()

            if hasattr(self, 'logs_table'):
                self.logs_table.setEnabled(True)

            # Use your existing populate method
            self.populate_logs_table_from_data(logs_data)

            logger.info(f"Logs table updated with {len(logs_data)} records")

        except Exception as e:
            logger.error(f"Error handling logs loaded: {e}")

    def on_users_loaded(self, users_data):
        """Handle users loaded signal"""
        try:
            if hasattr(self, 'users_table'):
                self.users_table.setEnabled(True)

            # Use your existing load_users_table logic here
            if hasattr(self, 'load_users_table'):
                # Call your existing method with the data
                pass

            logger.info(f"Users table updated with {len(users_data)} records")

        except Exception as e:
            logger.error(f"Error handling users loaded: {e}")

    def on_statistics_loaded(self, stats_data):
        """Handle statistics loaded signal"""
        try:
            if 'dashboard' in self.loading_overlays:
                self.loading_overlays['dashboard'].hide_loading()

            # Use your existing dashboard update methods
            self.update_dashboard_with_statistics(stats_data)

            logger.info("Dashboard statistics updated successfully")

        except Exception as e:
            logger.error(f"Error handling statistics loaded: {e}")

    def on_verification_completed(self, status, reason, details):
        """Handle verification completed signal"""
        try:
            operation_type = None
            for op_id, op_type in self.active_operations.items():
                if op_type in ['verify_latest', 'verify_manual']:
                    operation_type = op_type
                    break

            if operation_type == 'verify_latest':
                tanker_number = details.get("tanker_number", "UNKNOWN")
                # Use your existing update method
                if hasattr(self, 'update_auto_verification_display'):
                    self.update_auto_verification_display(tanker_number, status, reason, details)
                self.status_bar.showMessage(f"Latest tanker verified: {tanker_number}")
            elif operation_type == 'verify_manual':
                tanker_number = details.get("tanker_number", "UNKNOWN")
                # Use your existing update method
                if hasattr(self, 'update_manual_verification_display'):
                    self.update_manual_verification_display(tanker_number, status, reason, details)
                self.status_bar.showMessage(f"Manual verification completed: {tanker_number}")

            # Play warning sound if needed
            if details.get("play_sound", False) and hasattr(self, 'play_warning_sound_for_status'):
                self.play_warning_sound_for_status(status)

            logger.info(f"Verification completed: {status} - {reason}")

        except Exception as e:
            logger.error(f"Error handling verification completed: {e}")

    def on_connection_tested(self, connection_type, success, message):
        """Handle connection tested signal"""
        try:
            if connection_type == 'server':
                if hasattr(self, 'server_status_label'):
                    if success:
                        self.server_status_label.setText(f"Server Status: ✅ {message}")
                        self.server_status_label.setStyleSheet("color: #059669; font-weight: 600;")
                    else:
                        self.server_status_label.setText(f"Server Status: ❌ {message}")
                        self.server_status_label.setStyleSheet("color: #DC2626; font-weight: 600;")

            elif connection_type == 'local':
                if hasattr(self, 'local_status_label'):
                    if success:
                        self.local_status_label.setText(f"Local Status: ✅ {message}")
                        self.local_status_label.setStyleSheet("color: #059669; font-weight: 600;")
                    else:
                        self.local_status_label.setText(f"Local Status: ❌ {message}")
                        self.local_status_label.setStyleSheet("color: #DC2626; font-weight: 600;")

            status_msg = f"{connection_type.title()} connection: {'Success' if success else 'Failed'}"
            self.status_bar.showMessage(status_msg)

            logger.info(f"Connection test {connection_type}: {'Success' if success else 'Failed'} - {message}")

        except Exception as e:
            logger.error(f"Error handling connection tested: {e}")

    def get_current_log_filters(self):
        """Get current log filters"""
        filters = {}
        if hasattr(self, 'log_start_date') and hasattr(self, 'log_end_date'):
            filters['start_date'] = self.log_start_date.date().toString("yyyy-MM-dd")
            filters['end_date'] = self.log_end_date.date().toString("yyyy-MM-dd")
        return filters

    def populate_bans_table_from_data(self, bans_data):
        """Populate bans table from loaded data - customize this for your table structure"""
        if hasattr(self, 'bans_table'):
            # Clear existing data
            self.bans_table.setRowCount(0)

            # Populate with new data (adapt this to your existing populate logic)
            for row, ban in enumerate(bans_data):
                self.bans_table.insertRow(row)

                for col in range(min(8, len(ban))):
                    value = ban[col] if col < len(ban) else ""
                    item = QTableWidgetItem(str(value) if value else "")
                    self.bans_table.setItem(row, col, item)

    def populate_logs_table_from_data(self, logs_data):
        """Populate logs table from loaded data - customize this for your table structure"""
        if hasattr(self, 'logs_table'):
            self.logs_table.setRowCount(0)

            for row, log in enumerate(logs_data):
                self.logs_table.insertRow(row)

                for col, value in enumerate(log):
                    item = QTableWidgetItem(str(value) if value else "")
                    self.logs_table.setItem(row, col, item)

    def update_dashboard_with_statistics(self, stats_data):
        """Update dashboard with statistics data - use your existing logic"""
        # Use your existing dashboard update methods here
        # This is where you'd call your existing refresh_dashboard logic
        # but with the pre-loaded stats_data instead of loading it again
        pass

    def create_modern_content_area(self):
        self.content_area = QFrame()
        self.content_area.setObjectName("modernContentArea")

        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.content_layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("modernStackedWidget")

        self.dashboard_page = self.create_modern_dashboard_page()
        self.verification_page = self.create_modern_verification_page()
        self.manual_verify_page = self.create_modern_manual_verify_page()
        self.bans_page = self.create_modern_bans_page()
        self.logs_page = self.create_modern_logs_page()

        # Only create settings page if user has access
        if self.user_info['role'] in ['admin', 'supervisor']:
            self.settings_page = self.create_modern_settings_page()
            self.stacked_widget.addWidget(self.settings_page)

        self.stacked_widget.addWidget(self.dashboard_page)
        self.stacked_widget.addWidget(self.verification_page)
        self.stacked_widget.addWidget(self.manual_verify_page)
        self.stacked_widget.addWidget(self.bans_page)
        self.stacked_widget.addWidget(self.logs_page)

        self.content_layout.addWidget(self.stacked_widget)

    def create_modern_dashboard_page(self):
        """Create modern dashboard page with enhanced styling"""
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Modern header section
        header_container = QFrame()
        header_container.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)

        header_info = QVBoxLayout()
        header_title = QLabel("📊 Dashboard")
        header_title.setObjectName("pageTitle")
        header_info.addWidget(header_title)

        header_subtitle = QLabel("Ban Records & Verification Statistics")
        header_subtitle.setObjectName("pageSubtitle")
        header_info.addWidget(header_subtitle)

        header_layout.addLayout(header_info)
        header_layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setObjectName("refreshButton")
        refresh_btn.clicked.connect(self.refresh_dashboard)
        header_layout.addWidget(refresh_btn)

        layout.addWidget(header_container)

        # Modern filter section
        filter_container = QFrame()
        filter_container.setObjectName("filterContainer")
        filter_layout = QVBoxLayout(filter_container)
        filter_layout.setContentsMargins(24, 20, 24, 20)
        filter_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        filter_header = QLabel("📊 Dashboard Filters")
        filter_header.setObjectName("filterHeader")
        filter_layout.addWidget(filter_header)

        filter_controls = QHBoxLayout()
        filter_controls.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Date range controls
        date_group = QVBoxLayout()
        date_label = QLabel("Date Range")
        date_label.setObjectName("filterLabel")
        date_group.addWidget(date_label)

        date_controls = QHBoxLayout()
        date_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.dashboard_start_date = QDateEdit()
        self.dashboard_start_date.setObjectName("modernDateEdit")
        self.dashboard_start_date.setDate(QDate.currentDate().addDays(-30))
        self.dashboard_start_date.setCalendarPopup(True)
        date_controls.addWidget(self.dashboard_start_date)

        date_to_label = QLabel("to")
        date_to_label.setObjectName("dateToLabel")
        date_controls.addWidget(date_to_label)

        self.dashboard_end_date = QDateEdit()
        self.dashboard_end_date.setObjectName("modernDateEdit")
        self.dashboard_end_date.setDate(QDate.currentDate())
        self.dashboard_end_date.setCalendarPopup(True)
        date_controls.addWidget(self.dashboard_end_date)

        date_group.addLayout(date_controls)
        filter_controls.addLayout(date_group)

        filter_controls.addStretch()

        # Filter action buttons
        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        apply_filter_btn = QPushButton("🔍 Apply Filter")
        apply_filter_btn.setObjectName("applyFilterButton")
        apply_filter_btn.clicked.connect(self.apply_dashboard_filter)
        filter_actions.addWidget(apply_filter_btn)

        clear_filter_btn = QPushButton("🗑️ Clear")
        clear_filter_btn.setObjectName("clearFilterButton")
        clear_filter_btn.clicked.connect(self.clear_dashboard_filter)
        filter_actions.addWidget(clear_filter_btn)

        filter_controls.addLayout(filter_actions)
        filter_layout.addLayout(filter_controls)

        # Filter status indicator
        self.dashboard_filter_status = QLabel("📄 Showing all data")
        self.dashboard_filter_status.setObjectName("filterStatus")
        filter_layout.addWidget(self.dashboard_filter_status)

        layout.addWidget(filter_container)

        # Statistics sections with modern cards
        stats_section = QVBoxLayout()
        stats_section.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Ban statistics
        ban_stats_header = QLabel("⛔ Ban Records Statistics")
        ban_stats_header.setObjectName("statsHeader")
        stats_section.addWidget(ban_stats_header)

        ban_stats_container = QFrame()
        ban_stats_container.setObjectName("statsContainer")
        self.ban_stats_container = QHBoxLayout(ban_stats_container)
        self.ban_stats_container.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))
        stats_section.addWidget(ban_stats_container)

        # Verification statistics
        verify_stats_header = QLabel("✅ Verification Statistics")
        verify_stats_header.setObjectName("statsHeader")
        stats_section.addWidget(verify_stats_header)

        verify_stats_container = QFrame()
        verify_stats_container.setObjectName("statsContainer")
        self.verify_stats_container = QHBoxLayout(verify_stats_container)
        self.verify_stats_container.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))
        stats_section.addWidget(verify_stats_container)

        layout.addLayout(stats_section)

        # Recent activity tables with modern styling
        tables_section = QVBoxLayout()
        tables_section.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Recent ban records table
        recent_bans_header = QLabel("📋 Recent Ban Records")
        recent_bans_header.setObjectName("tableHeader")
        tables_section.addWidget(recent_bans_header)

        self.recent_bans_table = QTableWidget()
        self.recent_bans_table.setObjectName("modernTable")
        self.recent_bans_table.setColumnCount(6)
        self.recent_bans_table.setHorizontalHeaderLabels(["Tanker", "Reason", "Type", "Start Date", "End Date", "Created By"])
        self.recent_bans_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_bans_table.setMaximumHeight(250)
        tables_section.addWidget(self.recent_bans_table)

        # Recent verification activity table
        recent_logs_header = QLabel("📋 Recent Verification Activity")
        recent_logs_header.setObjectName("tableHeader")
        tables_section.addWidget(recent_logs_header)

        self.recent_table = QTableWidget()
        self.recent_table.setObjectName("modernTable")
        self.recent_table.setColumnCount(5)
        self.recent_table.setHorizontalHeaderLabels(["⏰ Time", "🚛 Tanker", "📊 Status", "📝 Reason", "👤 Operator"])
        self.recent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_table.setMaximumHeight(250)
        tables_section.addWidget(self.recent_table)

        layout.addLayout(tables_section)
        layout.addStretch()

        return page

    def create_modern_verification_page(self):
        """Create modern auto verification page"""
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Header
        header_container = QFrame()
        header_container.setObjectName("pageHeader")
        header_layout = QVBoxLayout(header_container)

        header_title = QLabel("✅ Auto Verification")
        header_title.setObjectName("pageTitle")
        header_layout.addWidget(header_title)

        header_subtitle = QLabel("Latest Entry Verification")
        header_subtitle.setObjectName("pageSubtitle")
        header_layout.addWidget(header_subtitle)

        layout.addWidget(header_container)

        # Control section
        control_container = QFrame()
        control_container.setObjectName("controlContainer")
        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(24, 20, 24, 20)

        verify_btn = QPushButton("🔍 Verify Latest Entry")
        verify_btn.setObjectName("verifyButton")
        verify_btn.clicked.connect(self.verify_latest_tanker)
        control_layout.addWidget(verify_btn)

        control_layout.addStretch()
        layout.addWidget(control_container)

        # Result display with modern styling
        self.auto_result_frame = QFrame()
        self.auto_result_frame.setObjectName("resultContainer")
        result_layout = QVBoxLayout(self.auto_result_frame)
        result_layout.setContentsMargins(32, 32, 32, 32)
        result_layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Tanker info
        self.auto_tanker_info_label = QLabel("🚛 Ready for verification...")
        self.auto_tanker_info_label.setObjectName("tankerInfoLabel")
        result_layout.addWidget(self.auto_tanker_info_label)

        # Status display
        self.auto_status_label = QLabel("System Ready")
        self.auto_status_label.setObjectName("statusDisplayLabel")
        result_layout.addWidget(self.auto_status_label)

        # Reason display
        self.auto_reason_label = QLabel("Click 'Verify Latest Entry' to check the most recent tanker")
        self.auto_reason_label.setObjectName("reasonDisplayLabel")
        result_layout.addWidget(self.auto_reason_label)

        # Voice playback section
        self.auto_voice_frame = QFrame()
        self.auto_voice_frame.setObjectName("voiceContainer")
        voice_layout = QHBoxLayout(self.auto_voice_frame)
        voice_layout.setContentsMargins(20, 16, 20, 16)

        self.auto_voice_info_label = QLabel("🎵 No voice note available")
        self.auto_voice_info_label.setObjectName("voiceInfoLabel")
        voice_layout.addWidget(self.auto_voice_info_label)

        voice_layout.addStretch()

        self.auto_play_voice_btn = QPushButton("🎵 Play Voice Note")
        self.auto_play_voice_btn.setObjectName("voicePlayButton")
        self.auto_play_voice_btn.clicked.connect(self.play_auto_verification_voice)
        self.auto_play_voice_btn.setVisible(False)
        voice_layout.addWidget(self.auto_play_voice_btn)

        self.auto_voice_frame.setVisible(False)
        result_layout.addWidget(self.auto_voice_frame)

        layout.addWidget(self.auto_result_frame)
        layout.addStretch()

        return page

    def create_modern_manual_verify_page(self):
        """Create modern manual verification page"""
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Header
        header_container = QFrame()
        header_container.setObjectName("pageHeader")
        header_layout = QVBoxLayout(header_container)

        header_title = QLabel("🔍 Manual Verification")
        header_title.setObjectName("pageTitle")
        header_layout.addWidget(header_title)

        header_subtitle = QLabel("Enter Tanker Number for Verification")
        header_subtitle.setObjectName("pageSubtitle")
        header_layout.addWidget(header_subtitle)

        layout.addWidget(header_container)

        # Input section with modern styling
        input_container = QFrame()
        input_container.setObjectName("inputContainer")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(24, 24, 24, 24)
        input_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Tanker input field
        input_group = QVBoxLayout()
        input_label = QLabel("Tanker Number")
        input_label.setObjectName("inputLabel")
        input_group.addWidget(input_label)

        input_controls = QHBoxLayout()
        input_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.manual_tanker_input = QLineEdit()
        self.manual_tanker_input.setObjectName("tankerInput")
        self.manual_tanker_input.setPlaceholderText("Enter tanker number (e.g., 40247, TEST001)")
        self.manual_tanker_input.returnPressed.connect(self.verify_manual_tanker)
        input_controls.addWidget(self.manual_tanker_input)

        verify_btn = QPushButton("🔍 Verify")
        verify_btn.setObjectName("verifyButton")
        verify_btn.clicked.connect(self.verify_manual_tanker)
        input_controls.addWidget(verify_btn)

        input_group.addLayout(input_controls)
        input_layout.addLayout(input_group)

        # Quick access buttons with modern styling
        quick_group = QVBoxLayout()
        quick_label = QLabel("Quick Access")
        quick_label.setObjectName("quickLabel")
        quick_group.addWidget(quick_label)

        quick_buttons_layout = QHBoxLayout()
        quick_buttons_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        quick_buttons = [
            ("TEST001", "TEST001"),
            ("40247", "40247"),
            ("5001", "5001"),
            ("Clear", "")
        ]

        for text, value in quick_buttons:
            btn = QPushButton(text)
            btn.setObjectName("quickButton")
            if value:
                btn.clicked.connect(lambda checked, v=value: self.manual_tanker_input.setText(v))
            else:
                btn.clicked.connect(lambda: self.manual_tanker_input.clear())
            quick_buttons_layout.addWidget(btn)

        quick_buttons_layout.addStretch()
        quick_group.addLayout(quick_buttons_layout)
        input_layout.addLayout(quick_group)

        layout.addWidget(input_container)

        # Result display (same structure as auto verification)
        self.manual_result_frame = QFrame()
        self.manual_result_frame.setObjectName("resultContainer")
        result_layout = QVBoxLayout(self.manual_result_frame)
        result_layout.setContentsMargins(32, 32, 32, 32)
        result_layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        self.manual_tanker_info_label = QLabel("🚛 Enter tanker number above")
        self.manual_tanker_info_label.setObjectName("tankerInfoLabel")
        result_layout.addWidget(self.manual_tanker_info_label)

        self.manual_status_label = QLabel("Ready for Manual Verification")
        self.manual_status_label.setObjectName("statusDisplayLabel")
        result_layout.addWidget(self.manual_status_label)

        self.manual_reason_label = QLabel("Enter a tanker number in the field above and click 'Verify' or press Enter")
        self.manual_reason_label.setObjectName("reasonDisplayLabel")
        result_layout.addWidget(self.manual_reason_label)

        # Voice playback section
        self.manual_voice_frame = QFrame()
        self.manual_voice_frame.setObjectName("voiceContainer")
        voice_layout = QHBoxLayout(self.manual_voice_frame)
        voice_layout.setContentsMargins(20, 16, 20, 16)

        self.manual_voice_info_label = QLabel("🎵 No voice note available")
        self.manual_voice_info_label.setObjectName("voiceInfoLabel")
        voice_layout.addWidget(self.manual_voice_info_label)

        voice_layout.addStretch()

        self.manual_play_voice_btn = QPushButton("🎵 Play Voice Note")
        self.manual_play_voice_btn.setObjectName("voicePlayButton")
        self.manual_play_voice_btn.clicked.connect(self.play_manual_verification_voice)
        self.manual_play_voice_btn.setVisible(False)
        voice_layout.addWidget(self.manual_play_voice_btn)

        self.manual_voice_frame.setVisible(False)
        result_layout.addWidget(self.manual_voice_frame)

        layout.addWidget(self.manual_result_frame)
        layout.addStretch()

        return page

    def create_modern_bans_page(self):
        """Create modern ban management page"""
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Header
        header_container = QFrame()
        header_container.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header_container)

        header_info = QVBoxLayout()
        header_title = QLabel("⛔ Ban Management")
        header_title.setObjectName("pageTitle")
        header_info.addWidget(header_title)

        header_subtitle = QLabel("Local Database (ban_records)")
        header_subtitle.setObjectName("pageSubtitle")
        header_info.addWidget(header_subtitle)

        header_layout.addLayout(header_info)
        header_layout.addStretch()

        add_btn = QPushButton("➕ Add Ban")
        add_btn.setObjectName("addBanButton")
        add_btn.clicked.connect(self.show_add_ban_dialog)
        header_layout.addWidget(add_btn)

        layout.addWidget(header_container)

        # Enhanced filter section for bans
        filter_container = QFrame()
        filter_container.setObjectName("filterContainer")
        filter_layout = QVBoxLayout(filter_container)
        filter_layout.setContentsMargins(24, 20, 24, 20)
        filter_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        filter_header = QLabel("🔍 Ban Record Filters")
        filter_header.setObjectName("filterHeader")
        filter_layout.addWidget(filter_header)

        # Filter controls in rows
        filter_row1 = QHBoxLayout()
        filter_row1.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Date range
        date_group = QVBoxLayout()
        date_label = QLabel("Date Range")
        date_label.setObjectName("filterLabel")
        date_group.addWidget(date_label)

        date_controls = QHBoxLayout()
        date_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.ban_start_date = QDateEdit()
        self.ban_start_date.setObjectName("modernDateEdit")
        self.ban_start_date.setDate(QDate.currentDate().addYears(-1))
        self.ban_start_date.setCalendarPopup(True)
        date_controls.addWidget(self.ban_start_date)

        date_to_label = QLabel("to")
        date_to_label.setObjectName("dateToLabel")
        date_controls.addWidget(date_to_label)

        self.ban_end_date = QDateEdit()
        self.ban_end_date.setObjectName("modernDateEdit")
        self.ban_end_date.setDate(QDate.currentDate())
        self.ban_end_date.setCalendarPopup(True)
        date_controls.addWidget(self.ban_end_date)

        date_group.addLayout(date_controls)
        filter_row1.addLayout(date_group)

        # Ban type filter
        type_group = QVBoxLayout()
        type_label = QLabel("Ban Type")
        type_label.setObjectName("filterLabel")
        type_group.addWidget(type_label)

        self.ban_type_filter = QComboBox()
        self.ban_type_filter.setObjectName("modernCombo")
        self.ban_type_filter.addItems(["All", "temporary", "permanent", "permission", "reminder"])
        type_group.addWidget(self.ban_type_filter)

        filter_row1.addLayout(type_group)
        filter_layout.addLayout(filter_row1)

        # Second row for text filters
        filter_row2 = QHBoxLayout()
        filter_row2.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Tanker number filter
        tanker_group = QVBoxLayout()
        tanker_label = QLabel("Tanker Number")
        tanker_label.setObjectName("filterLabel")
        tanker_group.addWidget(tanker_label)

        self.ban_tanker_filter = QLineEdit()
        self.ban_tanker_filter.setObjectName("filterInput")
        self.ban_tanker_filter.setPlaceholderText("Filter by tanker number...")
        tanker_group.addWidget(self.ban_tanker_filter)

        filter_row2.addLayout(tanker_group)

        # Reason filter
        reason_group = QVBoxLayout()
        reason_label = QLabel("Ban Reason")
        reason_label.setObjectName("filterLabel")
        reason_group.addWidget(reason_label)

        self.ban_reason_filter = QLineEdit()
        self.ban_reason_filter.setObjectName("filterInput")
        self.ban_reason_filter.setPlaceholderText("Filter by reason...")
        reason_group.addWidget(self.ban_reason_filter)

        filter_row2.addLayout(reason_group)

        filter_layout.addLayout(filter_row2)

        # Filter actions
        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        apply_ban_filter_btn = QPushButton("🔍 Apply Filters")
        apply_ban_filter_btn.setObjectName("applyFilterButton")
        apply_ban_filter_btn.clicked.connect(self.apply_ban_filters)
        filter_actions.addWidget(apply_ban_filter_btn)

        clear_ban_filter_btn = QPushButton("🗑️ Clear Filters")
        clear_ban_filter_btn.setObjectName("clearFilterButton")
        clear_ban_filter_btn.clicked.connect(self.clear_ban_filters)
        filter_actions.addWidget(clear_ban_filter_btn)

        filter_actions.addStretch()
        filter_layout.addLayout(filter_actions)

        # Filter status
        self.ban_filter_status = QLabel("📄 Showing all records")
        self.ban_filter_status.setObjectName("filterStatus")
        filter_layout.addWidget(self.ban_filter_status)

        layout.addWidget(filter_container)

        # Ban count display
        self.ban_count_label = QLabel("Loading ban records...")
        self.ban_count_label.setObjectName("countLabel")
        layout.addWidget(self.ban_count_label)

        # Bans table with modern styling
        self.bans_table = QTableWidget()
        self.bans_table.setObjectName("modernTable")
        self.bans_table.setColumnCount(8)
        self.bans_table.setHorizontalHeaderLabels([
            "ID", "Tanker", "Reason", "Type", "Start Date", "End Date", "Created By", "Voice"
        ])

        # Set column widths
        header = self.bans_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Tanker
        header.setSectionResizeMode(2, QHeaderView.Stretch)           # Reason
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Start Date
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # End Date
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Created By
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Voice

        layout.addWidget(self.bans_table)

        return page

    def create_modern_logs_page(self):
        """Create modern logs page"""
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Header
        header_container = QFrame()
        header_container.setObjectName("pageHeader")
        header_layout = QVBoxLayout(header_container)

        header_title = QLabel("📝 Activity Logs")
        header_title.setObjectName("pageTitle")
        header_layout.addWidget(header_title)

        header_subtitle = QLabel("System Verification Activity")
        header_subtitle.setObjectName("pageSubtitle")
        header_layout.addWidget(header_subtitle)

        layout.addWidget(header_container)

        # Enhanced filter section for logs
        filter_container = QFrame()
        filter_container.setObjectName("filterContainer")
        filter_layout = QVBoxLayout(filter_container)
        filter_layout.setContentsMargins(24, 20, 24, 20)
        filter_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        filter_header = QLabel("🔍 Activity Log Filters")
        filter_header.setObjectName("filterHeader")
        filter_layout.addWidget(filter_header)

        # Filter controls
        filter_row1 = QHBoxLayout()
        filter_row1.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Date range
        date_group = QVBoxLayout()
        date_label = QLabel("Date Range")
        date_label.setObjectName("filterLabel")
        date_group.addWidget(date_label)

        date_controls = QHBoxLayout()
        date_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.log_start_date = QDateEdit()
        self.log_start_date.setObjectName("modernDateEdit")
        self.log_start_date.setDate(QDate.currentDate().addDays(-7))
        self.log_start_date.setCalendarPopup(True)
        date_controls.addWidget(self.log_start_date)

        date_to_label = QLabel("to")
        date_to_label.setObjectName("dateToLabel")
        date_controls.addWidget(date_to_label)

        self.log_end_date = QDateEdit()
        self.log_end_date.setObjectName("modernDateEdit")
        self.log_end_date.setDate(QDate.currentDate())
        self.log_end_date.setCalendarPopup(True)
        date_controls.addWidget(self.log_end_date)

        date_group.addLayout(date_controls)
        filter_row1.addLayout(date_group)

        # Status filter
        status_group = QVBoxLayout()
        status_label = QLabel("Status")
        status_label.setObjectName("filterLabel")
        status_group.addWidget(status_label)

        self.log_status_filter = QComboBox()
        self.log_status_filter.setObjectName("modernCombo")
        self.log_status_filter.addItems(["All", "ALLOWED", "DENIED", "REJECTED", "ERROR", "CONDITIONAL"])
        status_group.addWidget(self.log_status_filter)

        filter_row1.addLayout(status_group)
        filter_layout.addLayout(filter_row1)

        # Text filters
        filter_row2 = QHBoxLayout()
        filter_row2.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Tanker filter
        tanker_group = QVBoxLayout()
        tanker_label = QLabel("Tanker Number")
        tanker_label.setObjectName("filterLabel")
        tanker_group.addWidget(tanker_label)

        self.log_tanker_filter = QLineEdit()
        self.log_tanker_filter.setObjectName("filterInput")
        self.log_tanker_filter.setPlaceholderText("Filter by tanker...")
        tanker_group.addWidget(self.log_tanker_filter)

        filter_row2.addLayout(tanker_group)

        # Operator filter
        operator_group = QVBoxLayout()
        operator_label = QLabel("Operator")
        operator_label.setObjectName("filterLabel")
        operator_group.addWidget(operator_label)

        self.log_operator_filter = QLineEdit()
        self.log_operator_filter.setObjectName("filterInput")
        self.log_operator_filter.setPlaceholderText("Filter by operator...")
        operator_group.addWidget(self.log_operator_filter)

        filter_row2.addLayout(operator_group)

        filter_layout.addLayout(filter_row2)

        # Filter actions
        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        apply_log_filter_btn = QPushButton("🔍 Apply Filters")
        apply_log_filter_btn.setObjectName("applyFilterButton")
        apply_log_filter_btn.clicked.connect(self.apply_log_filters)
        filter_actions.addWidget(apply_log_filter_btn)

        clear_log_filter_btn = QPushButton("🗑️ Clear")
        clear_log_filter_btn.setObjectName("clearFilterButton")
        clear_log_filter_btn.clicked.connect(self.clear_log_filters)
        filter_actions.addWidget(clear_log_filter_btn)

        filter_actions.addStretch()
        filter_layout.addLayout(filter_actions)

        # Filter status
        self.log_filter_status = QLabel("📄 Showing last 7 days")
        self.log_filter_status.setObjectName("filterStatus")
        filter_layout.addWidget(self.log_filter_status)

        layout.addWidget(filter_container)

        # Logs table
        self.logs_table = QTableWidget()
        self.logs_table.setObjectName("modernTable")
        self.logs_table.setColumnCount(6)
        self.logs_table.setHorizontalHeaderLabels([
            "ID", "Tanker", "Status", "Reason", "Timestamp", "Operator"
        ])
        self.logs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.logs_table)

        return page

    def create_modern_settings_page(self):
        """Create modern settings page"""
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Header
        header_container = QFrame()
        header_container.setObjectName("pageHeader")
        header_layout = QVBoxLayout(header_container)

        header_title = QLabel("⚙️ Settings")
        header_title.setObjectName("pageTitle")
        header_layout.addWidget(header_title)

        header_subtitle = QLabel("Database Configuration & User Management")
        header_subtitle.setObjectName("pageSubtitle")
        header_layout.addWidget(header_subtitle)

        layout.addWidget(header_container)

        # Create modern tab widget
        tab_widget = QTabWidget()
        tab_widget.setObjectName("modernTabWidget")

        # Database Settings Tab
        db_tab = self.create_modern_database_tab()
        tab_widget.addTab(db_tab, "🗄️ Database")

        # User Management Tab (admin only)
        if self.user_info['role'] == 'admin':
            user_tab = self.create_modern_user_tab()
            tab_widget.addTab(user_tab, "👥 Users")

        # System Settings Tab
        system_tab = self.create_modern_system_tab()
        tab_widget.addTab(system_tab, "🔧 System")

        # Role restriction notice for supervisors
        if self.user_info['role'] == 'supervisor':
            notice_container = QFrame()
            notice_container.setObjectName("noticeContainer")
            notice_layout = QVBoxLayout(notice_container)
            notice_layout.setContentsMargins(20, 16, 20, 16)

            notice_label = QLabel("ℹ️ Supervisor Access: You can modify database and system settings, but user management is restricted to administrators.")
            notice_label.setObjectName("noticeLabel")
            notice_layout.addWidget(notice_label)

            layout.addWidget(notice_container)

        layout.addWidget(tab_widget)
        return page

    def create_modern_database_tab(self):
        """Create modern database settings tab"""
        tab = QWidget()
        tab.setObjectName("modernTabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        layout.setContentsMargins(24, 24, 24, 24)

        # Database Configuration Section
        db_container = QFrame()
        db_container.setObjectName("settingsContainer")
        db_layout = QVBoxLayout(db_container)
        db_layout.setContentsMargins(24, 20, 24, 24)
        db_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        db_header = QLabel("🗄️ Database Configuration")
        db_header.setObjectName("settingsHeader")
        db_layout.addWidget(db_header)

        # Server database path
        server_group = QVBoxLayout()
        server_label = QLabel("Server Database (.mdb)")
        server_label.setObjectName("settingsLabel")
        server_group.addWidget(server_label)

        server_controls = QHBoxLayout()
        server_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.server_path_input = QLineEdit()
        self.server_path_input.setObjectName("pathInput")
        self.server_path_input.setText(self.config.get('server_db_path', ''))
        self.server_path_input.setPlaceholderText("Path to server database (.mdb file)")
        server_controls.addWidget(self.server_path_input)

        server_browse_btn = QPushButton("📁 Browse")
        server_browse_btn.setObjectName("browseButton")
        server_browse_btn.clicked.connect(self.browse_server_db)
        server_controls.addWidget(server_browse_btn)

        server_group.addLayout(server_controls)
        db_layout.addLayout(server_group)

        # Local database path
        local_group = QVBoxLayout()
        local_label = QLabel("Local Database (.db)")
        local_label.setObjectName("settingsLabel")
        local_group.addWidget(local_label)

        local_controls = QHBoxLayout()
        local_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.local_path_input = QLineEdit()
        self.local_path_input.setObjectName("pathInput")
        self.local_path_input.setText(self.config.get('local_sqlite_path', ''))
        self.local_path_input.setPlaceholderText("Path to local SQLite database")
        local_controls.addWidget(self.local_path_input)

        local_browse_btn = QPushButton("📁 Browse")
        local_browse_btn.setObjectName("browseButton")
        local_browse_btn.clicked.connect(self.browse_local_db)
        local_controls.addWidget(local_browse_btn)

        local_group.addLayout(local_controls)
        db_layout.addLayout(local_group)

        layout.addWidget(db_container)

        # Connection Test Section
        test_container = QFrame()
        test_container.setObjectName("settingsContainer")
        test_layout = QVBoxLayout(test_container)
        test_layout.setContentsMargins(24, 20, 24, 24)
        test_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        test_header = QLabel("🔍 Connection Testing")
        test_header.setObjectName("settingsHeader")
        test_layout.addWidget(test_header)

        # Status displays
        self.server_status_label = QLabel("Server Status: Testing...")
        self.server_status_label.setObjectName("statusLabel")
        test_layout.addWidget(self.server_status_label)

        self.local_status_label = QLabel("Local Status: Testing...")
        self.local_status_label.setObjectName("statusLabel")
        test_layout.addWidget(self.local_status_label)

        # Test buttons
        test_actions = QHBoxLayout()
        test_actions.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        test_server_btn = QPushButton("🔍 Test Server")
        test_server_btn.setObjectName("testButton")
        test_server_btn.clicked.connect(self.test_server_connection)
        test_actions.addWidget(test_server_btn)

        test_local_btn = QPushButton("🔍 Test Local")
        test_local_btn.setObjectName("testButton")
        test_local_btn.clicked.connect(self.test_local_connection)
        test_actions.addWidget(test_local_btn)

        test_all_btn = QPushButton("🔍 Test All")
        test_all_btn.setObjectName("testButton")
        test_all_btn.clicked.connect(self.test_all_connections)
        test_actions.addWidget(test_all_btn)

        test_actions.addStretch()
        test_layout.addLayout(test_actions)

        layout.addWidget(test_container)

        # Save actions
        save_actions = QHBoxLayout()
        save_actions.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))
        save_actions.addStretch()

        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setObjectName("resetButton")
        reset_btn.clicked.connect(self.reset_database_settings)
        save_actions.addWidget(reset_btn)

        save_btn = QPushButton("💾 Save Database Settings")
        save_btn.setObjectName("saveButton")
        save_btn.clicked.connect(self.save_database_settings)
        save_actions.addWidget(save_btn)

        layout.addLayout(save_actions)
        layout.addStretch()

        # Initial connection test
        QTimer.singleShot(500, self.test_all_connections)

        return tab

    def create_modern_user_tab(self):
        """Create modern user management tab"""
        tab = QWidget()
        tab.setObjectName("modernTabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header_container = QFrame()
        header_container.setObjectName("settingsContainer")
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(24, 20, 24, 20)

        header_info = QVBoxLayout()
        users_title = QLabel("👥 User Account Management")
        users_title.setObjectName("settingsHeader")
        header_info.addWidget(users_title)

        users_subtitle = QLabel("Manage user accounts and permissions")
        users_subtitle.setObjectName("settingsSubtext")
        header_info.addWidget(users_subtitle)

        header_layout.addLayout(header_info)
        header_layout.addStretch()

        add_user_btn = QPushButton("➕ Add New User")
        add_user_btn.setObjectName("addUserButton")
        add_user_btn.clicked.connect(self.add_new_user)
        header_layout.addWidget(add_user_btn)

        layout.addWidget(header_container)

        # Users table with modern styling
        self.users_table = QTableWidget()
        self.users_table.setObjectName("modernTable")
        self.users_table.setColumnCount(8)
        self.users_table.setHorizontalHeaderLabels([
            "ID", "Username", "Full Name", "Role", "Status", "Created", "Last Login", "Actions"
        ])

        # Set column widths
        header = self.users_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Username
        header.setSectionResizeMode(2, QHeaderView.Stretch)           # Full Name
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Role
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Created
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Last Login
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Actions

        layout.addWidget(self.users_table)

        # Load users
        self.load_users_table()

        return tab

    def create_modern_system_tab(self):
        """Create modern system settings tab"""
        tab = QWidget()
        tab.setObjectName("modernTabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        layout.setContentsMargins(24, 24, 24, 24)

        # System Configuration
        system_container = QFrame()
        system_container.setObjectName("settingsContainer")
        system_layout = QVBoxLayout(system_container)
        system_layout.setContentsMargins(24, 20, 24, 24)
        system_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        system_header = QLabel("🔧 System Configuration")
        system_header.setObjectName("settingsHeader")
        system_layout.addWidget(system_header)

        # Settings grid
        settings_grid = QGridLayout()
        settings_grid.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Connection timeout
        timeout_label = QLabel("Connection Timeout")
        timeout_label.setObjectName("settingsLabel")
        self.connection_timeout_spin = QSpinBox()
        self.connection_timeout_spin.setObjectName("modernSpinBox")
        self.connection_timeout_spin.setRange(1, 30)
        self.connection_timeout_spin.setValue(self.config.get('connection_timeout', 3))
        self.connection_timeout_spin.setSuffix(" seconds")
        settings_grid.addWidget(timeout_label, 0, 0)
        settings_grid.addWidget(self.connection_timeout_spin, 0, 1)

        # Cache timeout
        cache_label = QLabel("Cache Timeout")
        cache_label.setObjectName("settingsLabel")
        self.cache_timeout_spin = QSpinBox()
        self.cache_timeout_spin.setObjectName("modernSpinBox")
        self.cache_timeout_spin.setRange(1, 60)
        self.cache_timeout_spin.setValue(self.config.get('cache_timeout', 5))
        self.cache_timeout_spin.setSuffix(" seconds")
        settings_grid.addWidget(cache_label, 1, 0)
        settings_grid.addWidget(self.cache_timeout_spin, 1, 1)

        # Monitor interval
        monitor_label = QLabel("Monitor Interval")
        monitor_label.setObjectName("settingsLabel")
        self.monitor_interval_spin = QSpinBox()
        self.monitor_interval_spin.setObjectName("modernSpinBox")
        self.monitor_interval_spin.setRange(1, 30)
        self.monitor_interval_spin.setValue(self.config.get('monitor_interval', 3))
        self.monitor_interval_spin.setSuffix(" seconds")
        settings_grid.addWidget(monitor_label, 2, 0)
        settings_grid.addWidget(self.monitor_interval_spin, 2, 1)

        # Auto refresh interval
        refresh_label = QLabel("Auto Refresh Interval")
        refresh_label.setObjectName("settingsLabel")
        self.auto_refresh_spin = QSpinBox()
        self.auto_refresh_spin.setObjectName("modernSpinBox")
        self.auto_refresh_spin.setRange(10, 300)
        self.auto_refresh_spin.setValue(self.config.get('auto_refresh_interval', 60))
        self.auto_refresh_spin.setSuffix(" seconds")
        settings_grid.addWidget(refresh_label, 3, 0)
        settings_grid.addWidget(self.auto_refresh_spin, 3, 1)

        # Sound duration
        sound_label = QLabel("Warning Sound Duration")
        sound_label.setObjectName("settingsLabel")
        self.warning_duration_spin = QDoubleSpinBox()
        self.warning_duration_spin.setObjectName("modernDoubleSpinBox")
        self.warning_duration_spin.setRange(1.0, 10.0)
        self.warning_duration_spin.setValue(self.config.get('warning_sound_duration', 3.0))
        self.warning_duration_spin.setSuffix(" seconds")
        self.warning_duration_spin.setSingleStep(0.5)
        settings_grid.addWidget(sound_label, 4, 0)
        settings_grid.addWidget(self.warning_duration_spin, 4, 1)

        system_layout.addLayout(settings_grid)
        layout.addWidget(system_container)

        # System Information
        info_container = QFrame()
        info_container.setObjectName("settingsContainer")
        info_layout = QVBoxLayout(info_container)
        info_layout.setContentsMargins(24, 20, 24, 24)
        info_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        info_header = QLabel("ℹ️ System Information")
        info_header.setObjectName("settingsHeader")
        info_layout.addWidget(info_header)

        # Info items
        info_items = [
            (f"📄 Configuration File: {os.path.abspath(CONFIG_FILE)}"),
            (f"🗄️ Server DB: {self.config.get('server_db_path', 'Not set')}"),
            (f"💾 Local DB: {self.config.get('local_sqlite_path', 'Not set')}"),
            (f"🎵 Audio Support: {'✅ Available' if AUDIO_AVAILABLE else '❌ Not Available'}"),
            (f"🔊 Sound Alerts: {'✅ Enabled' if self.sound_enabled else '❌ Disabled'}")
        ]

        for info_text in info_items:
            info_label = QLabel(info_text)
            info_label.setObjectName("infoLabel")
            info_layout.addWidget(info_label)

        layout.addWidget(info_container)

        # Save button
        save_actions = QHBoxLayout()
        save_actions.addStretch()

        save_system_btn = QPushButton("💾 Save System Settings")
        save_system_btn.setObjectName("saveButton")
        save_system_btn.clicked.connect(self.save_system_settings)
        save_actions.addWidget(save_system_btn)

        layout.addLayout(save_actions)
        layout.addStretch()

        return tab

    def apply_modern_styles(self):
        """Apply comprehensive modern styling"""
        self.setStyleSheet(f"""
            /* Main Window Styling */
            QMainWindow {{
                background-color: {ModernUITheme.BACKGROUND};
                font-family: {ModernUITheme.FONT_FAMILY};
                color: {ModernUITheme.TEXT_PRIMARY};
            }}
            
            /* Status Bar */
            #modernStatusBar {{
                background-color: {ModernUITheme.SURFACE};
                border-top: 1px solid {ModernUITheme.BORDER_LIGHT};
                color: {ModernUITheme.TEXT_SECONDARY};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                padding: {ModernUITheme.SPACE_SM};
            }}
            
            /* Sidebar Styling */
            #modernSidebar {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {ModernUITheme.DARK_PRIMARY}, stop:1 {ModernUITheme.DARK_SECONDARY});
                border-right: 1px solid {ModernUITheme.BORDER};
            }}
            
            #titleContainer {{
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: {ModernUITheme.RADIUS_LG};
                padding: {ModernUITheme.SPACE_LG};
                margin-bottom: {ModernUITheme.SPACE_XL};
            }}
            
            #sidebarTitle {{
                font-size: {ModernUITheme.FONT_SIZE_2XL};
                font-weight: 700;
                color: white;
                margin: 0;
            }}
            
            #sidebarSubtitle {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: rgba(255, 255, 255, 0.8);
                font-weight: 500;
                margin: 0;
            }}
            
            /* Navigation Styling */
            #navContainer {{
                background-color: transparent;
            }}
            
            #navButtonContainer {{
                background-color: transparent;
                border-radius: {ModernUITheme.RADIUS_MD};
                border: none;
                margin: {ModernUITheme.SPACE_XS} 0;
                transition: all 0.2s ease;
            }}
            
            #navButtonContainer:hover {{
                background-color: rgba(255, 255, 255, 0.1);
                transform: translateX(4px);
            }}
            
            #navButtonContainerDisabled {{
                background-color: transparent;
                border-radius: {ModernUITheme.RADIUS_MD};
                opacity: 0.5;
            }}
            
            #navIcon {{
                font-size: {ModernUITheme.FONT_SIZE_LG};
                color: white;
                min-width: 24px;
            }}
            
            #navText {{
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 500;
                color: white;
            }}
            
            /* Sidebar Components */
            #restrictionNotice {{
                background-color: rgba(255, 183, 77, 0.1);
                border: 1px solid rgba(255, 183, 77, 0.3);
                border-radius: {ModernUITheme.RADIUS_MD};
                margin: {ModernUITheme.SPACE_MD} 0;
            }}
            
            #restrictionTitle {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 600;
                color: #FFB74D;
            }}
            
            #restrictionDesc {{
                font-size: {ModernUITheme.FONT_SIZE_XS};
                color: rgba(255, 183, 77, 0.8);
            }}
            
            #soundContainer {{
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: {ModernUITheme.RADIUS_MD};
                margin: {ModernUITheme.SPACE_MD} 0;
            }}
            
            #soundHeader {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 600;
                color: white;
                text-align: center;
            }}
            
            #soundToggle {{
                background-color: {ModernUITheme.SUCCESS};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_SM};
                font-size: {ModernUITheme.FONT_SIZE_XS};
                font-weight: 600;
                padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_MD};
                min-width: 50px;
            }}
            
            #soundToggle:!checked {{
                background-color: {ModernUITheme.ERROR};
            }}
            
            #soundStop {{
                background-color: {ModernUITheme.WARNING};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_SM};
                font-size: {ModernUITheme.FONT_SIZE_XS};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_MD};
            }}
            
            #userContainer {{
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: {ModernUITheme.RADIUS_MD};
            }}
            
            #userName {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 600;
                color: white;
            }}
            
            #userRoleAdmin {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: #FFB74D;
                font-weight: 500;
            }}
            
            #userRoleSupervisor {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: #81C784;
                font-weight: 500;
            }}
            
            #userRoleOperator {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: #64B5F6;
                font-weight: 500;
            }}
            
            /* Content Area */
            #modernContentArea {{
                background-color: {ModernUITheme.BACKGROUND};
            }}
            
            #modernStackedWidget {{
                background-color: transparent;
            }}
            
            #modernPage {{
                background-color: transparent;
            }}
            
            /* Page Headers */
            #pageHeader {{
                background-color: transparent;
                margin-bottom: {ModernUITheme.SPACE_XL};
            }}
            
            #pageTitle {{
                font-size: {ModernUITheme.FONT_SIZE_3XL};
                font-weight: 700;
                color: {ModernUITheme.TEXT_PRIMARY};
                margin: 0;
                letter-spacing: -0.5px;
            }}
            
            #pageSubtitle {{
                font-size: {ModernUITheme.FONT_SIZE_LG};
                color: {ModernUITheme.TEXT_SECONDARY};
                font-weight: 400;
                margin: {ModernUITheme.SPACE_SM} 0 0 0;
            }}
            
            /* Buttons */
            #refreshButton, #verifyButton, #addBanButton, #saveButton {{
                background-color: {ModernUITheme.PRIMARY};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 600;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                min-height: 44px;
            }}
            
            #refreshButton:hover, #verifyButton:hover, #addBanButton:hover, #saveButton:hover {{
                background-color: {ModernUITheme.PRIMARY_DARK};
                transform: translateY(-1px);
            }}
            
            #addBanButton {{
                background-color: {ModernUITheme.ERROR};
            }}
            
            #addBanButton:hover {{
                background-color: #B91C1C;
            }}
            
            #addUserButton {{
                background-color: {ModernUITheme.SUCCESS};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 600;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                min-height: 44px;
            }}
            
            #addUserButton:hover {{
                background-color: #047857;
            }}
            
            #resetButton {{
                background-color: {ModernUITheme.WARNING};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                min-height: 44px;
            }}
            
            #resetButton:hover {{
                background-color: #B45309;
            }}
            
            #browseButton, #testButton {{
                background-color: {ModernUITheme.TEXT_SECONDARY};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                min-height: 40px;
            }}
            
            #browseButton:hover, #testButton:hover {{
                background-color: #374151;
            }}
            
            #quickButton {{
                background-color: {ModernUITheme.WARNING};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_SM};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_MD};
                min-height: 36px;
            }}
            
            #quickButton:hover {{
                background-color: #B45309;
            }}
            
            #applyFilterButton {{
                background-color: {ModernUITheme.PRIMARY};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 600;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                min-height: 40px;
            }}
            
            #applyFilterButton:hover {{
                background-color: {ModernUITheme.PRIMARY_DARK};
            }}
            
            #clearFilterButton {{
                background-color: {ModernUITheme.TEXT_SECONDARY};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                min-height: 40px;
            }}
            
            #clearFilterButton:hover {{
                background-color: #374151;
            }}
            
            #voicePlayButton {{
                background-color: {ModernUITheme.SUCCESS};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_LG};
                min-height: 36px;
            }}
            
            #voicePlayButton:hover {{
                background-color: #047857;
            }}
            
            /* Containers and Cards */
            #filterContainer, #controlContainer, #inputContainer, #settingsContainer {{
                background-color: {ModernUITheme.SURFACE};
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
                border-radius: {ModernUITheme.RADIUS_LG};
                box-shadow: {ModernUITheme.SHADOW_SM};
            }}
            
            #resultContainer {{
                background-color: {ModernUITheme.SURFACE};
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
                border-radius: {ModernUITheme.RADIUS_XL};
                box-shadow: {ModernUITheme.SHADOW_MD};
            }}
            
            #voiceContainer {{
                background-color: rgba(5, 150, 105, 0.1);
                border: 1px solid rgba(5, 150, 105, 0.2);
                border-radius: {ModernUITheme.RADIUS_MD};
            }}
            
            #statsContainer {{
                background-color: transparent;
            }}
            
            #noticeContainer {{
                background-color: rgba(251, 191, 36, 0.1);
                border: 1px solid rgba(251, 191, 36, 0.2);
                border-radius: {ModernUITheme.RADIUS_MD};
            }}
            
            #noticeLabel {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: {ModernUITheme.WARNING};
                font-weight: 500;
                line-height: 1.5;
            }}
            
            /* Labels and Text */
            #filterHeader, #settingsHeader, #tableHeader, #statsHeader {{
                font-size: {ModernUITheme.FONT_SIZE_XL};
                font-weight: 600;
                color: {ModernUITheme.TEXT_PRIMARY};
                margin: 0;
            }}
            
            #filterLabel, #settingsLabel, #inputLabel, #quickLabel {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 600;
                color: {ModernUITheme.TEXT_SECONDARY};
                margin: 0;
            }}
            
            #filterStatus, #countLabel {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: {ModernUITheme.TEXT_MUTED};
                font-style: italic;
                margin: 0;
            }}
            
            #tankerInfoLabel {{
                font-size: {ModernUITheme.FONT_SIZE_XL};
                font-weight: 600;
                color: {ModernUITheme.TEXT_PRIMARY};
                text-align: center;
                margin: 0;
            }}
            
            #statusDisplayLabel {{
                font-size: {ModernUITheme.FONT_SIZE_3XL};
                font-weight: 700;
                text-align: center;
                margin: {ModernUITheme.SPACE_LG} 0;
                letter-spacing: -0.5px;
            }}
            
            #reasonDisplayLabel {{
                font-size: {ModernUITheme.FONT_SIZE_LG};
                color: {ModernUITheme.TEXT_SECONDARY};
                text-align: center;
                line-height: 1.5;
                margin: 0;
            }}
            
            #voiceInfoLabel {{
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                color: {ModernUITheme.TEXT_SECONDARY};
                font-weight: 500;
            }}
            
            #dateToLabel {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: {ModernUITheme.TEXT_MUTED};
                font-weight: 500;
                margin: 0 {ModernUITheme.SPACE_SM};
            }}
            
            #statusLabel {{
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                color: {ModernUITheme.TEXT_SECONDARY};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_SM} 0;
            }}
            
            #infoLabel {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: {ModernUITheme.TEXT_MUTED};
                font-family: {ModernUITheme.FONT_FAMILY_MONO};
                line-height: 1.4;
            }}
            
            #settingsSubtext {{
                font-size: {ModernUITheme.FONT_SIZE_SM};
                color: {ModernUITheme.TEXT_MUTED};
                margin: 0;
            }}
            
            /* Form Inputs */
            #tankerInput, #filterInput, #pathInput {{
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                background-color: {ModernUITheme.BACKGROUND};
                color: {ModernUITheme.TEXT_PRIMARY};
                font-weight: 500;
                min-height: 20px;
            }}
            
            #tankerInput:focus, #filterInput:focus, #pathInput:focus {{
                border-color: {ModernUITheme.PRIMARY};
                outline: none;
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }}
            
            #modernDateEdit, #modernCombo {{
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                background-color: {ModernUITheme.BACKGROUND};
                color: {ModernUITheme.TEXT_PRIMARY};
                font-weight: 500;
                min-height: 20px;
            }}
            
            #modernDateEdit:focus, #modernCombo:focus {{
                border-color: {ModernUITheme.PRIMARY};
                outline: none;
            }}
            
            #modernSpinBox, #modernDoubleSpinBox {{
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                background-color: {ModernUITheme.BACKGROUND};
                color: {ModernUITheme.TEXT_PRIMARY};
                font-weight: 500;
                min-height: 20px;
            }}
            
            #modernSpinBox:focus, #modernDoubleSpinBox:focus {{
                border-color: {ModernUITheme.PRIMARY};
                outline: none;
            }}
            
            /* Tables */
            #modernTable {{
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
                border-radius: {ModernUITheme.RADIUS_LG};
                background-color: {ModernUITheme.BACKGROUND};
                gridline-color: {ModernUITheme.BORDER_LIGHT};
                selection-background-color: rgba(37, 99, 235, 0.1);
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 500;
            }}
            
            #modernTable::item {{
                padding: {ModernUITheme.SPACE_MD};
                border-bottom: 1px solid {ModernUITheme.BORDER_LIGHT};
            }}
            
            #modernTable::item:selected {{
                background-color: rgba(37, 99, 235, 0.1);
                color: {ModernUITheme.TEXT_PRIMARY};
            }}
            
            #modernTable QHeaderView::section {{
                background-color: {ModernUITheme.SURFACE};
                color: {ModernUITheme.TEXT_SECONDARY};
                font-weight: 600;
                font-size: {ModernUITheme.FONT_SIZE_SM};
                padding: {ModernUITheme.SPACE_MD};
                border: none;
                border-bottom: 2px solid {ModernUITheme.BORDER};
                border-right: 1px solid {ModernUITheme.BORDER_LIGHT};
            }}
            
            #modernTable QHeaderView::section:first {{
                border-top-left-radius: {ModernUITheme.RADIUS_LG};
            }}
            
            #modernTable QHeaderView::section:last {{
                border-top-right-radius: {ModernUITheme.RADIUS_LG};
                border-right: none;
            }}
            
            /* Tab Widget */
            #modernTabWidget {{
                background-color: transparent;
            }}
            
            #modernTabWidget::pane {{
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
                border-radius: {ModernUITheme.RADIUS_LG};
                background-color: {ModernUITheme.BACKGROUND};
                margin-top: 8px;
            }}
            
            #modernTabWidget QTabBar::tab {{
                background-color: {ModernUITheme.SURFACE};
                color: {ModernUITheme.TEXT_SECONDARY};
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL};
                margin-right: {ModernUITheme.SPACE_XS};
                border-radius: {ModernUITheme.RADIUS_MD} {ModernUITheme.RADIUS_MD} 0 0;
                font-weight: 500;
                font-size: {ModernUITheme.FONT_SIZE_BASE};
            }}
            
            #modernTabWidget QTabBar::tab:selected {{
                background-color: {ModernUITheme.PRIMARY};
                color: white;
                border-color: {ModernUITheme.PRIMARY};
                font-weight: 600;
            }}
            
            #modernTabWidget QTabBar::tab:hover:!selected {{
                background-color: rgba(37, 99, 235, 0.1);
                color: {ModernUITheme.PRIMARY};
            }}
            
            #modernTabPage {{
                background-color: transparent;
            }}
            
            /* Scrollbars */
            QScrollBar:vertical {{
                background-color: {ModernUITheme.SURFACE};
                width: 12px;
                border-radius: 6px;
                margin: 0;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {ModernUITheme.TEXT_MUTED};
                border-radius: 6px;
                min-height: 20px;
                margin: 2px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {ModernUITheme.TEXT_SECONDARY};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            
            QScrollBar:horizontal {{
                background-color: {ModernUITheme.SURFACE};
                height: 12px;
                border-radius: 6px;
                margin: 0;
            }}
            
            QScrollBar::handle:horizontal {{
                background-color: {ModernUITheme.TEXT_MUTED};
                border-radius: 6px;
                min-width: 20px;
                margin: 2px;
            }}
            
            QScrollBar::handle:horizontal:hover {{
                background-color: {ModernUITheme.TEXT_SECONDARY};
            }}
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
        """)

    # Include all the existing methods from the original class
    # (toggle_sound, stop_warning_sound, play_warning_sound_for_status, etc.)
    # Continue with rest of methods...

    def toggle_sound(self):
        """Toggle sound enabled/disabled"""
        self.sound_enabled = not self.sound_enabled
        self.config.set('sound_enabled', self.sound_enabled)
        self.config.save_config()

        self.sound_toggle_btn.setText("ON" if self.sound_enabled else "OFF")
        self.sound_toggle_btn.setChecked(self.sound_enabled)

        status = "enabled" if self.sound_enabled else "disabled"
        self.status_bar.showMessage(f"Sound alerts {status}")

    def stop_warning_sound(self):
        """Stop any playing warning sound"""
        if self.warning_sound:
            self.warning_sound.stop()
            self.current_sound_playing = False
            self.status_bar.showMessage("Warning sound stopped")

    def play_warning_sound_for_status(self, status):
        """Play warning sound based on verification status"""
        if not self.sound_enabled or not self.warning_sound:
            return

        should_play = False

        # Check if sound should be played based on status
        if "DENIED" in status or "REJECTED" in status:
            should_play = True
        elif "PERMISSION" in status or "ALLOWED_WITH_PERMISSION" in status:
            should_play = True
        elif "WARNING" in status or "CONDITIONAL" in status:
            should_play = True

        if should_play and not self.current_sound_playing:
            self.current_sound_playing = True
            self.warning_sound.play(self.on_warning_sound_finished)
            logger.info(f"Playing warning sound for status: {status}")

    def on_warning_sound_finished(self, success, message):
        """Handle warning sound completion"""
        self.current_sound_playing = False
        if not success:
            logger.debug(f"Warning sound playback issue: {message}")

    def create_stats_card(self, title, value, color):
        """Create a modern statistics card widget"""
        card = QFrame()
        card.setFixedSize(200, 140)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {ModernUITheme.BACKGROUND}; 
                border-radius: {ModernUITheme.RADIUS_LG}; 
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
                margin: {ModernUITheme.SPACE_SM};
                box-shadow: {ModernUITheme.SHADOW_SM};
            }}
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 20, 16)
        card_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_SM}; 
            color: {ModernUITheme.TEXT_SECONDARY}; 
            font-weight: 600;
            margin: 0;
        """)
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setWordWrap(True)
        card_layout.addWidget(title_label)

        value_label = QLabel(str(value))
        value_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_4XL}; 
            font-weight: 700; 
            color: {color};
            margin: {ModernUITheme.SPACE_SM} 0;
        """)
        value_label.setAlignment(Qt.AlignLeft)
        card_layout.addWidget(value_label)

        card_layout.addStretch()

        indicator = QFrame()
        indicator.setFixedHeight(4)
        indicator.setStyleSheet(f"""
            background-color: {color}; 
            border-radius: 2px;
            margin: 0;
        """)
        card_layout.addWidget(indicator)

        return card

    def initial_dashboard_load(self):
        """Load initial dashboard data"""
        try:
            self.refresh_dashboard()
            self.status_bar.showMessage("Dashboard loaded successfully")
            logger.info("Initial dashboard data loaded")
        except Exception as e:
            logger.error(f"Error loading initial dashboard data: {e}")
            self.status_bar.showMessage(f"Dashboard load error: {e}")

    def apply_dashboard_filter(self):
        """Apply dashboard filter with proper filter handling"""
        try:
            self.dashboard_filters_applied = True
            self.current_dashboard_filters = {
                'start_date': self.dashboard_start_date.date().toString("yyyy-MM-dd"),
                'end_date': self.dashboard_end_date.date().toString("yyyy-MM-dd")
            }

            start_date_str = self.dashboard_start_date.date().toString("dd/MM/yyyy")
            end_date_str = self.dashboard_end_date.date().toString("dd/MM/yyyy")
            self.dashboard_filter_status.setText(f"🔍 Filtered: {start_date_str} to {end_date_str}")
            self.dashboard_filter_status.setStyleSheet(f"color: {ModernUITheme.WARNING}; font-weight: 600; font-style: italic;")

            self.refresh_dashboard()
            self.status_bar.showMessage("Dashboard filter applied")
            logger.info(f"Dashboard filter applied: {self.current_dashboard_filters}")

        except Exception as e:
            logger.error(f"Error applying dashboard filter: {e}")
            self.status_bar.showMessage(f"Dashboard filter error: {e}")

    def clear_dashboard_filter(self):
        """Clear dashboard filter with proper state management"""
        try:
            self.dashboard_filters_applied = False
            self.current_dashboard_filters = None

            self.dashboard_start_date.setDate(QDate.currentDate().addDays(-30))
            self.dashboard_end_date.setDate(QDate.currentDate())

            self.dashboard_filter_status.setText("📄 Showing all data")
            self.dashboard_filter_status.setStyleSheet(f"color: {ModernUITheme.TEXT_MUTED}; font-style: italic;")

            self.refresh_dashboard()
            self.status_bar.showMessage("Dashboard filter cleared")
            logger.info("Dashboard filter cleared")

        except Exception as e:
            logger.error(f"Error clearing dashboard filter: {e}")

    def refresh_dashboard(self):
        """Legacy method redirects to async version"""
        self.refresh_dashboard_async()

    def load_recent_ban_records(self, filters=None):
        """Load recent ban records from ban_records table"""
        try:
            recent_bans = self.db.get_all_bans(filters)[:10]
            self.recent_bans_table.setRowCount(0)

            for row, ban in enumerate(recent_bans):
                self.recent_bans_table.insertRow(row)

                data = [
                    str(ban[1]) if len(ban) > 1 else "",
                    str(ban[2])[:50] + "..." if len(ban) > 2 and len(str(ban[2])) > 50 else str(ban[2]) if len(ban) > 2 else "",
                    str(ban[3]) if len(ban) > 3 else "",
                    self.format_date(ban[4]) if len(ban) > 4 else "",
                    self.format_date(ban[5]) if len(ban) > 5 else "No End Date",
                    str(ban[6]) if len(ban) > 6 else ""
                ]

                for col, value in enumerate(data):
                    item = QTableWidgetItem(str(value))

                    # Color code ban types with modern colors
                    if col == 2:  # Ban type column
                        ban_type = str(value).lower()
                        if ban_type == "permanent":
                            item.setForeground(QColor(ModernUITheme.ERROR))
                        elif ban_type == "temporary":
                            item.setForeground(QColor(ModernUITheme.WARNING))
                        elif ban_type == "permission":
                            item.setForeground(QColor(ModernUITheme.PRIMARY))
                        elif ban_type == "reminder":
                            item.setForeground(QColor(ModernUITheme.SUCCESS))

                    self.recent_bans_table.setItem(row, col, item)

        except Exception as e:
            logger.error(f"Error loading recent ban records: {e}")

    def format_date(self, date_str):
        """Format date string for display"""
        if not date_str:
            return ""
        try:
            date_obj = datetime.strptime(str(date_str), "%Y-%m-%d").date()
            return date_obj.strftime("%d/%m/%Y")
        except:
            return str(date_str)

    def load_recent_activity(self):
        """Load recent activity with improved error handling"""
        try:
            logs = self.db.get_recent_logs(15)
            self.recent_table.setRowCount(0)

            for row, log in enumerate(logs):
                self.recent_table.insertRow(row)

                # Format timestamp
                timestamp = log[4] if len(log) > 4 else None
                if timestamp:
                    try:
                        dt = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
                        if dt.date() == datetime.now().date():
                            formatted_time = dt.strftime("%H:%M:%S")
                        else:
                            formatted_time = dt.strftime("%m/%d %H:%M")
                    except:
                        formatted_time = str(timestamp)[:16]
                else:
                    formatted_time = "Unknown"

                data = [
                    formatted_time,
                    str(log[1]) if len(log) > 1 else "N/A",
                    str(log[2]) if len(log) > 2 else "Unknown",
                    str(log[3])[:50] + "..." if len(log) > 3 and len(str(log[3])) > 50 else str(log[3]) if len(log) > 3 else "No reason",
                    str(log[5]) if len(log) > 5 else "System"
                ]

                for col, value in enumerate(data):
                    item = QTableWidgetItem(str(value))

                    # Color coding with modern colors
                    if col == 2:  # Status column
                        status_upper = str(value).upper()
                        if "ALLOWED" in status_upper:
                            item.setForeground(QColor(ModernUITheme.SUCCESS))
                        elif any(word in status_upper for word in ["REJECTED", "DENIED"]):
                            item.setForeground(QColor(ModernUITheme.ERROR))
                        elif "CONDITIONAL" in status_upper:
                            item.setForeground(QColor(ModernUITheme.WARNING))
                        elif "ERROR" in status_upper:
                            item.setForeground(QColor(ModernUITheme.TEXT_MUTED))

                    self.recent_table.setItem(row, col, item)

        except Exception as e:
            logger.error(f"Error loading recent activity: {e}")

    def verify_latest_tanker(self):
        """Legacy method redirects to async version"""
        self.verify_latest_tanker_async()

    def verify_manual_tanker(self):
        """Legacy method redirects to async version"""
        self.verify_manual_tanker_async()

    def update_auto_verification_display(self, tanker_number, status, reason, details):
        """Update auto verification display with modern styling"""
        try:
            self.auto_tanker_info_label.setText(f"🚛 Vehicle: {tanker_number}")
            self.auto_status_label.setText(status)
            self.auto_reason_label.setText(reason)

            # Set colors based on status with modern theme colors
            color = self.get_modern_status_color(status)
            self.auto_status_label.setStyleSheet(f"""
                font-size: {ModernUITheme.FONT_SIZE_3XL}; 
                font-weight: 700; 
                color: {color}; 
                text-align: center;
                margin: {ModernUITheme.SPACE_LG} 0;
                letter-spacing: -0.5px;
            """)

            # Handle voice playback if available
            ban_record = details.get("ban_record")
            if ban_record and ban_record.get('voice_recording'):
                self.auto_voice_frame.setVisible(True)
                self.auto_play_voice_btn.setVisible(True)
                self.auto_voice_info_label.setText(f"🎵 Voice note: {ban_record.get('ban_reason', 'Available')}")
                self.auto_current_voice_data = ban_record.get('voice_recording')
            else:
                self.auto_voice_frame.setVisible(False)
                self.auto_current_voice_data = None

        except Exception as e:
            logger.error(f"Error updating auto verification display: {e}")

    def update_manual_verification_display(self, tanker_number, status, reason, details):
        """Update manual verification display with modern styling"""
        try:
            self.manual_tanker_info_label.setText(f"🚛 Vehicle: {tanker_number}")
            self.manual_status_label.setText(status)
            self.manual_reason_label.setText(reason)

            # Set colors based on status with modern theme colors
            color = self.get_modern_status_color(status)
            self.manual_status_label.setStyleSheet(f"""
                font-size: {ModernUITheme.FONT_SIZE_3XL}; 
                font-weight: 700; 
                color: {color}; 
                text-align: center;
                margin: {ModernUITheme.SPACE_LG} 0;
                letter-spacing: -0.5px;
            """)

            # Handle voice playback if available
            ban_record = details.get("ban_record")
            if ban_record and ban_record.get('voice_recording'):
                self.manual_voice_frame.setVisible(True)
                self.manual_play_voice_btn.setVisible(True)
                self.manual_voice_info_label.setText(f"🎵 Voice note: {ban_record.get('ban_reason', 'Available')}")
                self.manual_current_voice_data = ban_record.get('voice_recording')
            else:
                self.manual_voice_frame.setVisible(False)
                self.manual_current_voice_data = None

        except Exception as e:
            logger.error(f"Error updating manual verification display: {e}")

    def get_modern_status_color(self, status):
        """Get modern theme color based on status"""
        if "ALLOWED" in status and "PERMISSION" not in status and "WARNING" not in status:
            return ModernUITheme.SUCCESS
        elif any(word in status for word in ["DENIED", "REJECTED"]):
            return ModernUITheme.ERROR
        elif "PERMISSION" in status or "CONDITIONAL" in status or "WARNING" in status:
            return ModernUITheme.WARNING
        else:
            return ModernUITheme.PRIMARY

    def play_auto_verification_voice(self):
        """Play voice note from auto verification page"""
        if hasattr(self, 'auto_current_voice_data') and self.auto_current_voice_data and self.audio_recorder:
            try:
                self.auto_voice_info_label.setText("🔊 Playing voice note...")
                self.audio_recorder.play_audio(self.auto_current_voice_data, self.on_auto_voice_finished)
            except Exception as e:
                logger.error(f"Error playing auto verification voice: {e}")
                QMessageBox.warning(self, "Error", f"Audio playback failed: {e}")

    def play_manual_verification_voice(self):
        """Play voice note from manual verification page"""
        if hasattr(self, 'manual_current_voice_data') and self.manual_current_voice_data and self.audio_recorder:
            try:
                self.manual_voice_info_label.setText("🔊 Playing voice note...")
                self.audio_recorder.play_audio(self.manual_current_voice_data, self.on_manual_voice_finished)
            except Exception as e:
                logger.error(f"Error playing manual verification voice: {e}")
                QMessageBox.warning(self, "Error", f"Audio playback failed: {e}")

    def on_auto_voice_finished(self, success, message):
        """Handle auto voice playback completion"""
        if success:
            self.auto_voice_info_label.setText("🎵 Voice note available - click to play")
        else:
            self.auto_voice_info_label.setText(f"❌ Playback failed: {message}")

    def on_manual_voice_finished(self, success, message):
        """Handle manual voice playback completion"""
        if success:
            self.manual_voice_info_label.setText("🎵 Voice note available - click to play")
        else:
            self.manual_voice_info_label.setText(f"❌ Playback failed: {message}")

    def show_add_ban_dialog(self):
        """Show modern add ban dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Ban Record")
        dialog.setFixedSize(650, 550)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {ModernUITheme.BACKGROUND};
                font-family: {ModernUITheme.FONT_FAMILY};
            }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel("Add New Ban Record")
        header_label.setStyleSheet(f"""
            font-size: {ModernUITheme.FONT_SIZE_2XL};
            font-weight: 600;
            color: {ModernUITheme.TEXT_PRIMARY};
            margin-bottom: {ModernUITheme.SPACE_XL};
        """)
        layout.addWidget(header_label)

        # Form container
        form_container = QFrame()
        form_container.setStyleSheet(f"""
            QFrame {{
                background-color: {ModernUITheme.SURFACE};
                border-radius: {ModernUITheme.RADIUS_LG};
                padding: {ModernUITheme.SPACE_2XL};
                border: 1px solid {ModernUITheme.BORDER_LIGHT};
            }}
        """)
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        # Form fields
        tanker_input = QLineEdit()
        tanker_input.setObjectName("modernInput")
        tanker_input.setPlaceholderText("Enter tanker number")

        reason_input = QTextEdit()
        reason_input.setObjectName("modernTextEdit")
        reason_input.setPlaceholderText("Enter ban reason")
        reason_input.setMaximumHeight(80)

        type_combo = QComboBox()
        type_combo.setObjectName("modernCombo")
        type_combo.addItems(["temporary", "permanent", "permission", "reminder"])

        start_date = QDateEdit()
        start_date.setObjectName("modernDateEdit")
        start_date.setDate(QDate.currentDate())
        start_date.setCalendarPopup(True)

        end_date = QDateEdit()
        end_date.setObjectName("modernDateEdit")
        end_date.setDate(QDate.currentDate().addDays(30))
        end_date.setCalendarPopup(True)

        # Add form fields with labels
        fields = [
            ("Tanker Number:", tanker_input),
            ("Ban Reason:", reason_input),
            ("Ban Type:", type_combo),
            ("Start Date:", start_date),
            ("End Date:", end_date)
        ]

        for label_text, widget in fields:
            field_layout = QVBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet(f"""
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 600;
                color: {ModernUITheme.TEXT_SECONDARY};
                margin-bottom: {ModernUITheme.SPACE_SM};
            """)
            field_layout.addWidget(label)
            field_layout.addWidget(widget)
            form_layout.addLayout(field_layout)

        layout.addWidget(form_container)

        # Audio recording section
        audio_container = QFrame()
        audio_container.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(5, 150, 105, 0.1);
                border: 1px solid rgba(5, 150, 105, 0.2);
                border-radius: {ModernUITheme.RADIUS_MD};
                padding: {ModernUITheme.SPACE_LG};
            }}
        """)
        audio_layout = QHBoxLayout(audio_container)

        audio_label = QLabel("Voice Note:")
        audio_label.setStyleSheet(f"""
            font-weight: 600;
            color: {ModernUITheme.TEXT_SECONDARY};
        """)
        audio_layout.addWidget(audio_label)

        voice_status_label = QLabel("No recording")
        voice_status_label.setStyleSheet(f"color: {ModernUITheme.TEXT_MUTED}; font-style: italic;")
        audio_layout.addWidget(voice_status_label)

        audio_layout.addStretch()

        record_voice_btn = QPushButton("🎙️ Record Voice")
        record_voice_btn.setObjectName("voiceRecordButton")
        if not AUDIO_AVAILABLE:
            record_voice_btn.setEnabled(False)
            record_voice_btn.setText("🎙️ Audio Not Available")

        recorded_data = None

        def record_voice():
            nonlocal recorded_data
            audio_dialog = AudioRecordDialog(dialog)
            if audio_dialog.exec_() == QDialog.Accepted:
                recorded_data = audio_dialog.recorded_data
                if recorded_data:
                    voice_status_label.setText("✅ Voice recorded")
                    voice_status_label.setStyleSheet(f"color: {ModernUITheme.SUCCESS}; font-weight: 600;")

        record_voice_btn.clicked.connect(record_voice)
        audio_layout.addWidget(record_voice_btn)

        layout.addWidget(audio_container)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(dialog.reject)

        save_btn = QPushButton("Save Ban")
        save_btn.setObjectName("primaryButton")

        def save_ban():
            tanker = tanker_input.text().strip().upper()
            reason = reason_input.toPlainText().strip()
            ban_type = type_combo.currentText()
            start_str = start_date.date().toString("yyyy-MM-dd")
            end_str = end_date.date().toString("yyyy-MM-dd") if ban_type != "permanent" else None

            if not tanker or not reason:
                QMessageBox.warning(dialog, "Input Error", "Please fill all required fields")
                return

            voice_filename = f"voice_{tanker}_{int(time.time())}.wav" if recorded_data else None

            success = self.db.add_ban_record(
                tanker, reason, ban_type, start_str, end_str, self.user_info['username'],
                recorded_data, voice_filename
            )

            if success:
                QMessageBox.information(dialog, "Success", f"Ban record created for {tanker}")
                dialog.accept()
                self.load_bans_table()
                self.refresh_dashboard()
                self.status_bar.showMessage(f"Ban record added for {tanker}")
            else:
                QMessageBox.critical(dialog, "Error", "Failed to create ban record")

        save_btn.clicked.connect(save_ban)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        # Apply modern input styles to the dialog
        dialog.setStyleSheet(dialog.styleSheet() + f"""
            #modernInput, #modernTextEdit {{
                border: 2px solid {ModernUITheme.BORDER};
                border-radius: {ModernUITheme.RADIUS_MD};
                padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG};
                font-size: {ModernUITheme.FONT_SIZE_BASE};
                background-color: {ModernUITheme.BACKGROUND};
                color: {ModernUITheme.TEXT_PRIMARY};
                font-weight: 500;
            }}
            
            #modernInput:focus, #modernTextEdit:focus {{
                border-color: {ModernUITheme.PRIMARY};
                outline: none;
            }}
            
            #voiceRecordButton {{
                background-color: {ModernUITheme.SUCCESS};
                color: white;
                border: none;
                border-radius: {ModernUITheme.RADIUS_MD};
                font-size: {ModernUITheme.FONT_SIZE_SM};
                font-weight: 500;
                padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_LG};
            }}
            
            #voiceRecordButton:hover {{
                background-color: #047857;
            }}
            
            #voiceRecordButton:disabled {{
                background-color: {ModernUITheme.TEXT_DISABLED};
                color: {ModernUITheme.TEXT_MUTED};
            }}
        """)

        dialog.exec_()

    def apply_ban_filters(self):
        """Apply ban filters with proper filter handling"""
        try:
            self.ban_filters_applied = True
            self.current_ban_filters = {}

            # Date filter
            start_date = self.ban_start_date.date().toString("yyyy-MM-dd")
            end_date = self.ban_end_date.date().toString("yyyy-MM-dd")
            self.current_ban_filters['start_date'] = start_date
            self.current_ban_filters['end_date'] = end_date

            # Text filters
            tanker_text = self.ban_tanker_filter.text().strip()
            if tanker_text:
                self.current_ban_filters['tanker_number'] = tanker_text

            reason_text = self.ban_reason_filter.text().strip()
            if reason_text:
                self.current_ban_filters['reason'] = reason_text

            ban_type = self.ban_type_filter.currentText()
            if ban_type != "All":
                self.current_ban_filters['ban_type'] = ban_type

            # Update filter status display
            filter_parts = []
            if tanker_text:
                filter_parts.append(f"Tanker: {tanker_text}")
            if reason_text:
                filter_parts.append(f"Reason: {reason_text}")
            if ban_type != "All":
                filter_parts.append(f"Type: {ban_type}")

            start_date_str = self.ban_start_date.date().toString("dd/MM/yyyy")
            end_date_str = self.ban_end_date.date().toString("dd/MM/yyyy")
            filter_parts.append(f"Date: {start_date_str} to {end_date_str}")

            filter_text = ", ".join(filter_parts)
            self.ban_filter_status.setText(f"🔍 Filtered: {filter_text}")
            self.ban_filter_status.setStyleSheet(f"color: {ModernUITheme.WARNING}; font-weight: 600; font-style: italic;")

            self.load_bans_table()
            self.status_bar.showMessage("Ban filters applied")
            logger.info(f"Ban filters applied: {self.current_ban_filters}")

        except Exception as e:
            logger.error(f"Error applying ban filters: {e}")
            self.status_bar.showMessage(f"Ban filter error: {e}")

    def clear_ban_filters(self):
        """Clear ban filters with proper state management"""
        try:
            self.ban_filters_applied = False
            self.current_ban_filters = None

            self.ban_start_date.setDate(QDate.currentDate().addYears(-1))
            self.ban_end_date.setDate(QDate.currentDate())
            self.ban_tanker_filter.clear()
            self.ban_reason_filter.clear()
            self.ban_type_filter.setCurrentText("All")

            self.ban_filter_status.setText("📄 Showing all records")
            self.ban_filter_status.setStyleSheet(f"color: {ModernUITheme.TEXT_MUTED}; font-style: italic;")

            self.load_bans_table()
            self.status_bar.showMessage("Ban filters cleared")
            logger.info("Ban filters cleared")

        except Exception as e:
            logger.error(f"Error clearing ban filters: {e}")

    def apply_log_filters(self):
        """Apply log filters with proper filter handling"""
        try:
            filters = {}

            start_date = self.log_start_date.date().toString("yyyy-MM-dd")
            end_date = self.log_end_date.date().toString("yyyy-MM-dd")
            filters['start_date'] = start_date
            filters['end_date'] = end_date

            if self.log_tanker_filter.text().strip():
                filters['tanker_number'] = self.log_tanker_filter.text().strip()

            if self.log_operator_filter.text().strip():
                filters['operator'] = self.log_operator_filter.text().strip()

            status = self.log_status_filter.currentText()
            if status != "All":
                filters['status'] = status

            # Update filter status display
            filter_parts = []
            if filters.get('tanker_number'):
                filter_parts.append(f"Tanker: {filters['tanker_number']}")
            if filters.get('operator'):
                filter_parts.append(f"Operator: {filters['operator']}")
            if filters.get('status'):
                filter_parts.append(f"Status: {filters['status']}")

            start_date_str = self.log_start_date.date().toString("dd/MM/yyyy")
            end_date_str = self.log_end_date.date().toString("dd/MM/yyyy")
            filter_parts.append(f"Date: {start_date_str} to {end_date_str}")

            filter_text = ", ".join(filter_parts)
            self.log_filter_status.setText(f"🔍 Filtered: {filter_text}")
            self.log_filter_status.setStyleSheet(f"color: {ModernUITheme.WARNING}; font-weight: 600; font-style: italic;")

            self.load_logs_table(filters)
            self.status_bar.showMessage("Log filters applied")
            logger.info(f"Log filters applied: {filters}")

        except Exception as e:
            logger.error(f"Error applying log filters: {e}")
            self.status_bar.showMessage(f"Log filter error: {e}")

    def clear_log_filters(self):
        """Clear log filters with proper state management"""
        try:
            self.log_start_date.setDate(QDate.currentDate().addDays(-7))
            self.log_end_date.setDate(QDate.currentDate())
            self.log_tanker_filter.clear()
            self.log_operator_filter.clear()
            self.log_status_filter.setCurrentText("All")

            self.log_filter_status.setText("📄 Showing last 7 days")
            self.log_filter_status.setStyleSheet(f"color: {ModernUITheme.TEXT_MUTED}; font-style: italic;")

            self.load_logs_table()
            self.status_bar.showMessage("Log filters cleared")
            logger.info("Log filters cleared")

        except Exception as e:
            logger.error(f"Error clearing log filters: {e}")

    def load_bans_table(self):
        """Legacy method redirects to async version"""
        self.load_bans_table_async()
    def play_ban_voice(self, voice_data, tanker_number):
        """Play voice note from ban management table"""
        if voice_data and self.audio_recorder:
            try:
                self.status_bar.showMessage(f"Playing voice note for {tanker_number}...")
                self.audio_recorder.play_audio(voice_data,
                    lambda success, msg: self.status_bar.showMessage(
                        f"Voice playback {'completed' if success else 'failed'} for {tanker_number}"
                    )
                )
            except Exception as e:
                logger.error(f"Error playing ban voice: {e}")
                QMessageBox.warning(self, "Error", f"Audio playback failed: {e}")

    def load_logs_table(self, filters=None):
        """Load logs table with proper filter handling"""
        try:
            logs = self.db.get_recent_logs(100, filters)
            self.logs_table.setRowCount(0)

            for row, log in enumerate(logs):
                self.logs_table.insertRow(row)

                for col, value in enumerate(log):
                    if col == 4:  # Timestamp column
                        if value:
                            try:
                                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                                formatted_time = dt.strftime("%d/%m/%Y %H:%M")
                            except:
                                formatted_time = value
                        else:
                            formatted_time = ""
                        item = QTableWidgetItem(formatted_time)
                    else:
                        item = QTableWidgetItem(str(value) if value else "")

                    # Color code status with modern colors
                    if col == 2:  # Status column
                        status_upper = str(value).upper()
                        if "ALLOWED" in status_upper:
                            item.setForeground(QColor(ModernUITheme.SUCCESS))
                        elif any(word in status_upper for word in ["REJECTED", "DENIED"]):
                            item.setForeground(QColor(ModernUITheme.ERROR))
                        elif "CONDITIONAL" in status_upper:
                            item.setForeground(QColor(ModernUITheme.WARNING))
                        elif "ERROR" in status_upper:
                            item.setForeground(QColor(ModernUITheme.TEXT_MUTED))

                    self.logs_table.setItem(row, col, item)

            self.status_bar.showMessage(f"Loaded {len(logs)} log entries")
            logger.info(f"Log table loaded with {len(logs)} records, filters: {filters is not None}")

        except Exception as e:
            logger.error(f"Error loading logs table: {e}")
            self.status_bar.showMessage(f"Error loading logs: {e}")

    # Navigation methods
    def show_dashboard(self):
        """Show dashboard page with async loading"""
        try:
            if hasattr(self, 'stacked_widget') and hasattr(self, 'dashboard_page'):
                self.stacked_widget.setCurrentWidget(self.dashboard_page)
                self.refresh_dashboard_async()  # Changed to async
                self.status_bar.showMessage("Dashboard page loaded")
        except Exception as e:
            logger.error(f"Error showing dashboard: {e}")


    def show_verification(self):
        """Show auto verification page"""
        try:
            self.stacked_widget.setCurrentWidget(self.verification_page)
            self.status_bar.showMessage("Auto verification page loaded")
        except Exception as e:
            logger.error(f"Error showing verification page: {e}")

    def show_manual_verify(self):
        """Show manual verification page"""
        try:
            self.stacked_widget.setCurrentWidget(self.manual_verify_page)
            self.status_bar.showMessage("Manual verification page loaded")
        except Exception as e:
            logger.error(f"Error showing manual verify page: {e}")

    def show_bans(self):
        """Show ban management page with async loading"""
        try:
            if hasattr(self, 'stacked_widget') and hasattr(self, 'bans_page'):
                self.stacked_widget.setCurrentWidget(self.bans_page)
                self.ban_filters_applied = False
                self.current_ban_filters = None
                self.load_bans_table_async()  # Changed to async
                self.status_bar.showMessage("Ban management page loaded")
        except Exception as e:
            logger.error(f"Error showing bans page: {e}")

    def show_logs(self):
        """Show activity logs page with async loading"""
        try:
            if hasattr(self, 'stacked_widget') and hasattr(self, 'logs_page'):
                self.stacked_widget.setCurrentWidget(self.logs_page)
                self.load_logs_table_async()  # Changed to async
                self.status_bar.showMessage("Activity logs page loaded")
        except Exception as e:
            logger.error(f"Error showing logs page: {e}")

    def show_settings(self):
        """Show settings page with role-based access control"""
        try:
            if self.user_info['role'] not in ['admin', 'supervisor']:
                QMessageBox.warning(self, "Access Denied",
                                  f"Settings access is restricted.\n\nYour role: {self.user_info['role']}\nRequired: Admin or Supervisor")
                logger.warning(f"Settings access denied for role: {self.user_info['role']}")
                return

            if hasattr(self, 'settings_page'):
                self.stacked_widget.setCurrentWidget(self.settings_page)

                if self.user_info['role'] == 'admin':
                    self.load_users_table()

                self.status_bar.showMessage("Settings page loaded")
            else:
                QMessageBox.warning(self, "Access Error", "Settings page not available for your role")

        except Exception as e:
            logger.error(f"Error showing settings page: {e}")

    # Settings methods
    def browse_server_db(self):
        """Browse for server database file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Server Database",
            self.config.get('server_db_path', ''),
            "Access Database (*.mdb *.accdb);;All Files (*)"
        )
        if file_path:
            self.server_path_input.setText(file_path)

    def browse_local_db(self):
        """Browse for local database file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Local Database",
            self.config.get('local_sqlite_path', ''),
            "SQLite Database (*.db);;All Files (*)"
        )
        if file_path:
            self.local_path_input.setText(file_path)

    def test_server_connection(self):
        """Legacy method redirects to async version"""
        self.test_server_connection_async()
    def test_local_connection(self):
        """Legacy method redirects to async version"""
        self.test_local_connection_async()

    def test_all_connections(self):
        """Test all database connections"""
        self.test_server_connection()
        self.test_local_connection()

    def save_database_settings(self):
        """Save database settings"""
        try:
            server_path = self.server_path_input.text().strip()
            local_path = self.local_path_input.text().strip()

            if not server_path or not local_path:
                QMessageBox.warning(self, "Validation Error", "Both database paths are required")
                return

            success = self.config.update_paths(server_path, local_path)

            if success:
                self.db.update_paths(server_path, local_path)
                self.user_manager = UserManager(local_path)

                QMessageBox.information(self, "Success", "Database settings saved successfully!\n\nConnections will be updated immediately.")
                self.status_bar.showMessage("Database settings updated")
                QTimer.singleShot(500, self.test_all_connections)
            else:
                QMessageBox.critical(self, "Error", "Failed to save database settings")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    def reset_database_settings(self):
        """Reset database settings to defaults"""
        reply = QMessageBox.question(self, "Reset Settings",
                                   "Are you sure you want to reset database settings to defaults?",
                                   QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.server_path_input.setText(DEFAULT_CONFIG['server_db_path'])
            self.local_path_input.setText(DEFAULT_CONFIG['local_sqlite_path'])
            QTimer.singleShot(100, self.test_all_connections)

    def save_system_settings(self):
        """Save system settings"""
        try:
            self.config.set('connection_timeout', self.connection_timeout_spin.value())
            self.config.set('cache_timeout', self.cache_timeout_spin.value())
            self.config.set('monitor_interval', self.monitor_interval_spin.value())
            self.config.set('auto_refresh_interval', self.auto_refresh_spin.value())
            self.config.set('warning_sound_duration', self.warning_duration_spin.value())

            if self.config.save_config():
                if self.warning_sound:
                    self.warning_sound = WarningSound(
                        duration=self.warning_duration_spin.value()
                    )

                QMessageBox.information(self, "Success", "System settings saved successfully!")
                self.status_bar.showMessage("System settings updated")
                self.restart_monitoring()
            else:
                QMessageBox.critical(self, "Error", "Failed to save system settings")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save system settings: {str(e)}")

    def load_users_table(self):
        """Load users into the table"""
        try:
            users = self.user_manager.get_all_users()
            self.users_table.setRowCount(0)

            for row, user in enumerate(users):
                self.users_table.insertRow(row)

                data = [
                    str(user[0]),  # ID
                    str(user[1]),  # Username
                    str(user[2]) if user[2] else "",  # Full Name
                    str(user[3]),  # Role
                    "Active" if user[4] == 1 else "Inactive",  # Status
                    self.format_datetime(user[5]) if user[5] else "",  # Created
                    self.format_datetime(user[6]) if user[6] else "Never",  # Last Login
                ]

                for col, value in enumerate(data):
                    item = QTableWidgetItem(str(value))

                    # Color code by role with modern colors
                    if col == 3:  # Role column
                        if value == "admin":
                            item.setForeground(QColor(ModernUITheme.ERROR))
                        elif value == "supervisor":
                            item.setForeground(QColor(ModernUITheme.WARNING))
                        else:
                            item.setForeground(QColor(ModernUITheme.SUCCESS))

                    # Color code by status
                    elif col == 4:  # Status column
                        if value == "Inactive":
                            item.setForeground(QColor(ModernUITheme.TEXT_MUTED))

                    self.users_table.setItem(row, col, item)

                # Add action buttons
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(4, 4, 4, 4)
                actions_layout.setSpacing(4)

                edit_btn = QPushButton("✏️")
                edit_btn.setToolTip("Edit User")
                edit_btn.setFixedSize(30, 25)
                edit_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ModernUITheme.PRIMARY};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {ModernUITheme.PRIMARY_DARK};
                    }}
                """)
                edit_btn.clicked.connect(lambda checked, u=user: self.edit_user(u))

                delete_btn = QPushButton("🗑️")
                delete_btn.setToolTip("Delete User")
                delete_btn.setFixedSize(30, 25)
                delete_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ModernUITheme.ERROR};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #B91C1C;
                    }}
                """)
                delete_btn.clicked.connect(lambda checked, u=user: self.delete_user(u))

                if user[1] == self.user_info['username']:
                    delete_btn.setEnabled(False)
                    delete_btn.setToolTip("Cannot delete current user")

                actions_layout.addWidget(edit_btn)
                actions_layout.addWidget(delete_btn)
                actions_layout.addStretch()

                self.users_table.setCellWidget(row, 7, actions_widget)

        except Exception as e:
            logger.error(f"Error loading users table: {e}")

    def format_datetime(self, datetime_str):
        """Format datetime string for display"""
        if not datetime_str:
            return ""
        try:
            dt = datetime.strptime(str(datetime_str), "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%d/%m/%Y %H:%M")
        except:
            return str(datetime_str)

    def add_new_user(self):
        """Add new user"""
        dialog = UserManagementDialog(self.user_manager, self.user_info, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.load_users_table()
            self.status_bar.showMessage("User added successfully")

    def edit_user(self, user_data):
        """Edit existing user"""
        dialog = UserManagementDialog(self.user_manager, self.user_info, user_data, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.load_users_table()
            self.status_bar.showMessage("User updated successfully")

    def delete_user(self, user_data):
        """Delete user"""
        username = user_data[1]

        if username == self.user_info['username']:
            QMessageBox.warning(self, "Cannot Delete", "You cannot delete your own account")
            return

        reply = QMessageBox.question(self, "Delete User",
                                   f"Are you sure you want to delete user '{username}'?\n\nThis action cannot be undone.",
                                   QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                success = self.user_manager.delete_user(user_data[0])
                if success:
                    QMessageBox.information(self, "Success", f"User '{username}' has been deleted")
                    self.load_users_table()
                    self.status_bar.showMessage(f"User {username} deleted")
                else:
                    QMessageBox.critical(self, "Error", "Failed to delete user")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete user: {str(e)}")

    def restart_monitoring(self):
        """Restart monitoring with new settings"""
        try:
            if hasattr(self, 'monitor_timer') and self.monitor_timer:
                self.monitor_timer.stop()
            if hasattr(self, 'dashboard_timer') and self.dashboard_timer:
                self.dashboard_timer.stop()

            QTimer.singleShot(1000, self.start_monitoring)

        except Exception as e:
            logger.error(f"Error restarting monitoring: {e}")

    def start_monitoring(self):
        """Start monitoring with improved error handling"""
        try:
            current_latest = self.db.get_latest_tanker_from_server()
            if current_latest:
                self.last_tanker = current_latest.tanker_number
            else:
                self.last_tanker = None

            self.monitor_timer = QTimer(self)
            self.monitor_timer.timeout.connect(self.check_new_tanker)
            monitor_interval = self.config.get('monitor_interval', 3) * 1000
            self.monitor_timer.start(monitor_interval)

            self.dashboard_timer = QTimer(self)
            self.dashboard_timer.timeout.connect(self.auto_refresh_dashboard)
            refresh_interval = self.config.get('auto_refresh_interval', 60) * 1000
            self.dashboard_timer.start(refresh_interval)

            logger.info("Monitoring started successfully")
            self.status_bar.showMessage("Monitoring system started")

        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            self.status_bar.showMessage(f"Monitoring error: {e}")

    def auto_refresh_dashboard(self):
        """Auto refresh dashboard if currently displayed"""
        try:
            if (hasattr(self, 'stacked_widget') and hasattr(self, 'dashboard_page') and
                self.stacked_widget.currentWidget() == self.dashboard_page):
                self.refresh_dashboard()
                logger.debug("Dashboard auto-refreshed")
        except Exception as e:
            logger.debug(f"Dashboard auto-refresh error: {e}")

    def check_new_tanker(self):
        """Check for new tanker and auto-verify"""
        try:
            latest_tanker = self.db.get_latest_tanker_from_server()

            if latest_tanker and latest_tanker.tanker_number != self.last_tanker:
                self.last_tanker = latest_tanker.tanker_number
                logger.info(f"New tanker detected: {latest_tanker.tanker_number}")

                status, reason, details = self.db.simple_tanker_verification("AUTO_MONITOR")

                if details.get("play_sound", False):
                    self.play_warning_sound_for_status(status)

                if (hasattr(self, 'stacked_widget') and hasattr(self, 'verification_page') and
                    self.stacked_widget.currentWidget() == self.verification_page):
                    self.update_auto_verification_display(latest_tanker.tanker_number, status, reason, details)

                if (hasattr(self, 'stacked_widget') and hasattr(self, 'dashboard_page') and
                    self.stacked_widget.currentWidget() == self.dashboard_page):
                    self.refresh_dashboard()

                sound_status = " (Sound played)" if details.get("play_sound", False) else ""
                self.status_bar.showMessage(f"New tanker auto-verified: {latest_tanker.tanker_number} - {status}{sound_status}")

        except Exception as e:
            logger.debug(f"Auto-verification error: {e}")

    def closeEvent(self, event):
        """Clean shutdown with improved cleanup"""
        try:
            logger.info(f"TDF System shutdown initiated - User: {self.user_info['username']}")
            if hasattr(self, 'db_worker') and self.db_worker:
                logger.info("Stopping database worker thread...")
                self.db_worker.stop_thread()
            for overlay in self.loading_overlays.values():
                overlay.hide_loading()
            if hasattr(self, 'audio_recorder') and self.audio_recorder:
                self.audio_recorder.stop_any_operation()

            if hasattr(self, 'warning_sound') and self.warning_sound:
                self.warning_sound.stop()

            if hasattr(self, 'monitor_timer') and self.monitor_timer:
                self.monitor_timer.stop()
            if hasattr(self, 'dashboard_timer') and self.dashboard_timer:
                self.dashboard_timer.stop()

            QApplication.processEvents()

            logger.info(f"TDF System shutdown completed - User: {self.user_info['username']}")
            event.accept()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            event.accept()
def main():
    """Main entry point with modern UI and enhanced functionality"""
    try:
        logger.info("TDF SYSTEM - MODERN UI VERSION WITH ENHANCED FUNCTIONALITY STARTING")
        logger.info("Features: Modern Professional Interface, Fixed Filters, Role-based Access Control, Enhanced Accessibility")

        if not PYODBC_AVAILABLE:
            error_msg = "pyodbc not available. Install with: pip install pyodbc"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            QMessageBox.critical(None, "Missing Dependency",
                               f"Required library missing:\n\n{error_msg}\n\nPlease install pyodbc and restart the application.")
            return 1

        # Create application with modern settings
        app = QApplication(sys.argv)
        app.setApplicationName("TDF System - Modern Professional Interface")
        app.setStyle("Fusion")

        # Set modern application properties
        app.setAttribute(Qt.AA_DisableWindowContextHelpButton, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app.setQuitOnLastWindowClosed(True)

        # Set application font
        font = QFont(ModernUITheme.FONT_FAMILY.split(',')[0])
        font.setPixelSize(14)
        app.setFont(font)

        # Initialize configuration
        try:
            logger.info("Initializing configuration...")
            config_manager = ConfigManager()
            logger.info("Configuration initialized successfully")
        except Exception as e:
            error_msg = f"Configuration initialization failed: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(None, "Configuration Error",
                               f"Failed to initialize configuration:\n\n{error_msg}\n\nCheck logs for details.")
            return 1

        # Test database initialization
        try:
            logger.info("Testing database initialization...")
            test_db = DatabaseManager(config_manager)
            logger.info("Database initialization successful")
        except Exception as e:
            error_msg = f"Database initialization failed: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(None, "Database Error",
                               f"Failed to initialize database:\n\n{error_msg}\n\nCheck logs for details.")
            return 1

        # Initialize user manager
        try:
            logger.info("Initializing user manager...")
            local_db_path = config_manager.get('local_sqlite_path')
            logger.info(f"Using local database path: {local_db_path}")

            user_manager = UserManager(local_db_path)
            logger.info("User manager initialized successfully")

            # Verify default users exist
            with sqlite3.connect(local_db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users WHERE username IN ('admin', 'operator', 'supervisor')")
                default_user_count = cursor.fetchone()[0]
                logger.info(f"Found {default_user_count} default users in database")

                if default_user_count < 3:
                    logger.warning("Missing default users, recreating...")
                    cursor.execute("DELETE FROM users WHERE username IN ('admin', 'operator', 'supervisor')")

                    # Recreate default users
                    admin_password = user_manager.hash_password("admin123")
                    operator_password = user_manager.hash_password("operator123")
                    supervisor_password = user_manager.hash_password("supervisor123")

                    cursor.execute("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("admin", admin_password, "System Administrator", "admin", "system"))

                    cursor.execute("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("operator", operator_password, "Default Operator", "operator", "system"))

                    cursor.execute("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, ("supervisor", supervisor_password, "Default Supervisor", "supervisor", "system"))

                    conn.commit()
                    logger.info("Default users recreated successfully")

        except Exception as e:
            error_msg = f"User manager initialization failed: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(None, "User Management Error",
                               f"Failed to initialize user management:\n\n{error_msg}\n\nTry using the 'Reset Database' button on the login screen.")
            return 1

        # Show modern login dialog
        login = ModernLoginDialog(user_manager)

        if login.exec_() != QDialog.Accepted:
            logger.info("Login cancelled by user")
            return 0

        user_info = login.user_info
        if not user_info:
            QMessageBox.critical(None, "Login Error", "Authentication failed!")
            logger.error("Authentication failed")
            return 1

        logger.info(f"Login successful for user: {user_info['username']} (Role: {user_info['role']})")

        # Create modern main window
        try:
            window = MainWindow(user_info, config_manager)
            window.show()
            logger.info("Modern main window created and displayed successfully")
        except Exception as e:
            error_msg = f"Error creating main window: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(None, "Initialization Error",
                               f"Failed to initialize main window:\n\n{error_msg}\n\nCheck logs for details.")
            return 1

        # Log system status
        logger.info("TDF System started successfully with Modern Professional Interface")
        logger.info("Enhanced features:")
        logger.info("- MODERN UI: Professional typography, colors, and visual design")
        logger.info("- ACCESSIBILITY: High contrast, readable fonts, clear spacing")
        logger.info("- FIXED FILTERS: Properly working filter system with clear indicators")
        logger.info("- ROLE-BASED ACCESS: Secure user permission system")
        logger.info("- ENHANCED UX: Intuitive navigation and responsive controls")
        logger.info("- PROFESSIONAL STYLING: Modern cards, buttons, and layouts")
        logger.info(f"Current user role: {user_info['role']} - Modern interface ready")

        return app.exec_()

    except Exception as e:
        error_msg = f"Fatal error in main: {str(e)}"
        logger.error(error_msg)
        try:
            QMessageBox.critical(None, "Fatal Error",
                               f"Application failed to start:\n\n{error_msg}\n\nCheck logs for details.")
        except:
            print(f"FATAL ERROR: {e}")
        return 1
    window = MainWindow(user_info, config_manager)
    if hasattr(window, 'db_worker'):
        window.db_worker.start()
        logger.info("Database worker thread started")

    window.show()
if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
# Triggering CodeRabbit for unused code detection
