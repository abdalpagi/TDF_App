import os
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_SCALE_FACTOR'] = '1'

"""
TDF System - Enhanced Version with Modern UI, Fixed Filters and Role-Based Access Control
Refactored for cleaner code while preserving all functionality and design
CLEANED VERSION - Duplicates Removed
"""

import json
import shutil
import zipfile
from datetime import datetime, timedelta
import sys
import os
import time
import logging
import threading
import sqlite3
import json
import hashlib
import tempfile
from enhanced_db_manager import EnhancedDatabaseManager
import threading
import queue
import time
import tempfile
import os
import wave
import logging
from typing import Optional, Callable

import sys

logger = logging.getLogger(__name__)
import wave
import queue
import weakref
from typing import Optional, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


# Add timeout wrapper class
class TimeoutWrapper:
    """Quick fix to prevent hanging operations"""

    @staticmethod
    def execute_with_ui_timeout(operation, timeout=15, progress_callback=None):
        """Execute operation with timeout and UI updates"""
        import threading
        import time
        import queue

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def operation_thread():
            try:
                result = operation()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        # Start operation in background
        thread = threading.Thread(target=operation_thread, daemon=True)
        thread.start()

        # Wait with UI updates
        start_time = time.time()
        while thread.is_alive() and (time.time() - start_time) < timeout:
            # Keep UI responsive
            QApplication.processEvents()

            # Call progress callback if provided
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(elapsed, timeout)

            time.sleep(0.1)  # Small delay to prevent CPU spinning

        # Check if operation completed
        if thread.is_alive():
            # Operation timed out
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        # Get result or exception
        if not exception_queue.empty():
            raise exception_queue.get()
        elif not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Operation completed but no result available")
# Configure logging (single configuration)
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

# Audio libraries (consolidated audio availability check)
AUDIO_AVAILABLE = False
try:
    import pyaudio
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
    logger.info("Audio libraries available (PyAudio, sounddevice, soundfile, numpy)")
except ImportError as e:
    logger.warning(f"Audio not available: {e} - audio features will be disabled")
    pyaudio = None
    sd = None
    sf = None
    np = None

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16 if AUDIO_AVAILABLE else None
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 30

# Keep pyodbc for server Access database
try:
    import pyodbc

    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

# Update DEFAULT_CONFIG
DEFAULT_CONFIG = {
    "server_db_path": "",  # Microsoft Access .mdb/.accdb file path
    "local_sqlite_path": "data/local_database.db",  # SQLite .db file for ban records
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
def validate_audio_data(audio_data):
    """Enhanced audio data validation"""
    if not audio_data:
        return False, "No audio data provided"

    if len(audio_data) < 1000:  # Increased minimum size
        return False, f"Audio data too small: {len(audio_data)} bytes (minimum 1000 bytes required)"

    if not audio_data.startswith(b'RIFF'):
        return False, "Missing WAV RIFF header"

    if b'WAVE' not in audio_data[:20]:
        return False, "Missing WAVE format identifier"

    # Additional validation: check for data chunk
    if b'data' not in audio_data[:100]:
        return False, "Missing WAV data chunk"

    return True, "Audio data is valid"

class ModernUITheme:
    """Modern UI theme constants for consistent styling"""
    PRIMARY = "#2563EB"
    PRIMARY_DARK = "#1D4ED8"
    PRIMARY_LIGHT = "#3B82F6"
    SECONDARY = "#7C3AED"
    ACCENT = "#059669"
    SUCCESS = "#059669"
    WARNING = "#D97706"
    ERROR = "#DC2626"
    INFO = "#0284C7"
    DANGER = "#DC2626"
    SURFACE = "#F3F4F6"
    BORDER_LIGHT = "#E5E7EB"
    BACKGROUND = "#FFFFFF"
    CARD = "#FFFFFF"
    BORDER = "#E2E8F0"
    TEXT_PRIMARY = "#0F172A"
    TEXT_SECONDARY = "#475569"
    TEXT_MUTED = "#94A3B8"
    TEXT_DISABLED = "#CBD5E1"
    DARK_PRIMARY = "#1E293B"
    DARK_SECONDARY = "#334155"
    DARK_ACCENT = "#475569"

    FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    FONT_FAMILY_MONO = "'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace"

    FONT_SIZE_XS = "11px"
    FONT_SIZE_SM = "13px"
    FONT_SIZE_BASE = "14px"
    FONT_SIZE_LG = "16px"
    FONT_SIZE_XL = "18px"
    FONT_SIZE_2XL = "20px"
    FONT_SIZE_3XL = "24px"
    FONT_SIZE_4XL = "30px"
    FONT_SIZE_5XL = "36px"

    SPACE_XS = "4px"
    SPACE_SM = "8px"
    SPACE_MD = "12px"
    SPACE_LG = "16px"
    SPACE_XL = "20px"
    SPACE_2XL = "24px"
    SPACE_3XL = "32px"
    SPACE_4XL = "40px"

    RADIUS_SM = "6px"
    RADIUS_MD = "8px"
    RADIUS_LG = "12px"
    RADIUS_XL = "16px"

    SHADOW_SM = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    SHADOW_MD = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    SHADOW_LG = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
    SHADOW_XL = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"


class ConfigManager:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.config = self.load_config()

    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                merged_config = DEFAULT_CONFIG.copy()
                merged_config.update(config)
                return merged_config
            else:
                self.save_config(DEFAULT_CONFIG)
                return DEFAULT_CONFIG.copy()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return DEFAULT_CONFIG.copy()

    def save_config(self, config=None):
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
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def update_paths(self, server_path, local_path):
        self.config['server_db_path'] = server_path
        self.config['local_sqlite_path'] = local_path
        return self.save_config()


class UserManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_user_tables()

    def init_user_tables(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
                table_exists = cursor.fetchone() is not None

                if table_exists:
                    cursor.execute("PRAGMA table_info(users)")
                    existing_columns = [column[1] for column in cursor.fetchall()]

                    required_columns = ['id', 'username', 'password_hash', 'full_name', 'role', 'is_active',
                                        'created_at', 'last_login', 'created_by']
                    missing_columns = [col for col in required_columns if col not in existing_columns]

                    if missing_columns:
                        cursor.execute("SELECT * FROM users")
                        existing_data = cursor.fetchall()
                        cursor.execute("PRAGMA table_info(users)")
                        old_columns = [column[1] for column in cursor.fetchall()]

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

                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for row in existing_data:
                            old_row_dict = dict(zip(old_columns, row))
                            username = old_row_dict.get('username', '')
                            password_hash = old_row_dict.get('password_hash') or old_row_dict.get('password', '')

                            if len(password_hash) < 50:
                                if username in ['admin', 'operator', 'supervisor']:
                                    password_hash = self.hash_password(f'{username}123')
                                else:
                                    password_hash = self.hash_password(password_hash)

                            cursor.execute("""
                                INSERT INTO users (username, password_hash, full_name, role, is_active, created_at, created_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (username, password_hash, old_row_dict.get('full_name', username.title()),
                                  old_row_dict.get('role', 'admin' if username == 'admin' else 'operator'),
                                  old_row_dict.get('is_active', 1), current_time, 'migration'))
                else:
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

                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]

                if user_count == 0:
                    default_users = [
                        ("admin", self.hash_password("admin123"), "System Administrator", "admin", "system"),
                        ("operator", self.hash_password("operator123"), "Default Operator", "operator", "system"),
                        ("supervisor", self.hash_password("supervisor123"), "Default Supervisor", "supervisor", "system")
                    ]
                    cursor.executemany("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, default_users)

                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing user tables: {e}")
            raise

    def hash_password(self, password):
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def authenticate_user(self, username, password):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                password_hash = self.hash_password(password)

                cursor.execute("""
                    SELECT id, username, full_name, role, is_active, password_hash
                    FROM users WHERE username = ?
                """, (username,))

                result = cursor.fetchone()
                if result and result[4] and password_hash == result[5]:
                    cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?", (username,))
                    conn.commit()
                    return {'id': result[0], 'username': result[1], 'full_name': result[2], 'role': result[3],
                            'is_active': result[4]}
                return None
        except Exception as e:
            logger.error(f"Authentication error for {username}: {e}")
            return None

    def add_user(self, username, password, full_name, role, created_by):
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
            return False
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False

    def get_all_users(self):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, full_name, role, is_active, created_at, last_login, created_by
                    FROM users ORDER BY username
                """)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []


class WarningSound:
    def __init__(self, duration=3.0, frequency=800):
        self.duration = duration
        self.frequency = frequency
        self.sample_rate = 44100
        self._generate_warning_sound()

    def _generate_warning_sound(self):
        try:
            if not AUDIO_AVAILABLE:
                self.warning_data = None
                return

            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
            pulse_freq = 3
            pulse_envelope = 0.5 * (1 + np.sin(2 * np.pi * pulse_freq * t))
            tone = np.sin(2 * np.pi * self.frequency * t)

            fade_samples = int(0.1 * self.sample_rate)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)

            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
            warning_tone = tone * pulse_envelope * 0.3
            self.warning_data = warning_tone.astype(np.float32).tobytes()
        except Exception as e:
            logger.error(f"Error generating warning sound: {e}")
            self.warning_data = None

    def play(self, callback=None):
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
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"Error stopping sound: {e}")


class ThreadSafeAudioRecorder:
    """Completely rewritten thread-safe audio recorder to prevent heap corruption"""

    def __init__(self):
        self.audio = None
        self.recording = False
        self.frames = []
        self.recording_thread = None
        self.recording_stream = None

        # Thread-safe playback management
        self._playback_lock = threading.Lock()
        self._current_playback_thread = None
        self._stop_playback_event = threading.Event()

        if AUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                logger.info("ThreadSafeAudioRecorder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PyAudio: {e}")
                self.audio = None

    def start_recording(self):
        """Thread-safe recording start"""
        try:
            if not AUDIO_AVAILABLE or not self.audio:
                logger.error("Audio not available for recording")
                return False

            if self.recording:
                logger.warning("Recording already in progress")
                return False

            # Stop any existing playback first
            self.stop_playback()

            self.frames = []
            self.recording = True

            def recording_worker():
                """Isolated recording worker thread"""
                stream = None
                try:
                    # Create stream in the worker thread
                    stream = self.audio.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK
                    )

                    logger.info("Recording started in worker thread")
                    start_time = time.time()

                    while self.recording and (time.time() - start_time) < RECORD_SECONDS:
                        try:
                            if stream and stream.is_active():
                                data = stream.read(CHUNK, exception_on_overflow=False)
                                if self.recording and len(data) > 0:
                                    self.frames.append(data)
                            else:
                                break
                        except Exception as e:
                            logger.error(f"Error reading audio data: {e}")
                            break

                    logger.info(f"Recording finished. Captured {len(self.frames)} frames")

                except Exception as e:
                    logger.error(f"Recording worker error: {e}")
                    self.recording = False
                finally:
                    # Always clean up stream in the same thread
                    if stream:
                        try:
                            if stream.is_active():
                                stream.stop_stream()
                            stream.close()
                        except Exception as e:
                            logger.debug(f"Error closing recording stream: {e}")

            # Start recording thread
            self.recording_thread = threading.Thread(target=recording_worker, daemon=True)
            self.recording_thread.start()

            return True

        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.recording = False
            return False

    def stop_recording(self):
        """Thread-safe recording stop"""
        try:
            if not self.recording:
                logger.debug("No recording in progress")
                return None

            logger.info("Stopping recording...")
            self.recording = False

            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=3.0)

            if not self.frames:
                logger.warning("No audio frames recorded")
                return None

            logger.info(f"Processing {len(self.frames)} audio frames")

            # Create WAV file safely
            return self._create_wav_file_safe()

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None
        finally:
            self.recording = False
            self.recording_thread = None

    def _create_wav_file_safe(self):
        """Safely create WAV file with proper error handling"""
        temp_file = None
        try:
            # Calculate total size
            total_size = sum(len(frame) for frame in self.frames)
            if total_size < 1000:
                logger.error(f"Audio data too small: {total_size} bytes")
                return None

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_filename = temp_file.name
            temp_file.close()

            # Write WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)

                # Write frames safely
                for frame in self.frames:
                    wf.writeframes(frame)

            # Read complete file
            with open(temp_filename, 'rb') as f:
                wav_data = f.read()

            # Cleanup
            os.unlink(temp_filename)

            # Validate
            if len(wav_data) < 1000 or not wav_data.startswith(b'RIFF'):
                logger.error(f"Invalid WAV file created: {len(wav_data)} bytes")
                return None

            logger.info(f"WAV file created successfully: {len(wav_data)} bytes")
            return wav_data

        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            return None

    def play_audio(self, audio_data_or_file, callback: Optional[Callable] = None):
        """COMPLETELY REWRITTEN thread-safe audio playback to prevent heap corruption"""
        if not AUDIO_AVAILABLE or not self.audio:
            if callback:
                callback(False, "Audio playback not available")
            return

        # Use lock to prevent multiple playback threads
        if not self._playback_lock.acquire(blocking=False):
            logger.warning("Another playback is already in progress")
            if callback:
                callback(False, "Another playback is in progress")
            return

        try:
            # Stop any existing playback
            self._stop_playback_event.set()
            if self._current_playback_thread and self._current_playback_thread.is_alive():
                self._current_playback_thread.join(timeout=1.0)

            # Reset stop event
            self._stop_playback_event.clear()

            def playback_worker():
                """Isolated playback worker to prevent memory corruption"""
                temp_file_path = None
                stream = None

                try:
                    # Prepare audio file
                    if isinstance(audio_data_or_file, str):
                        audio_file = audio_data_or_file
                        if not os.path.exists(audio_file):
                            raise FileNotFoundError(f"Audio file not found: {audio_file}")

                    elif isinstance(audio_data_or_file, bytes):
                        # Validate audio data
                        if len(audio_data_or_file) < 1000:
                            raise ValueError(f"Audio data too small: {len(audio_data_or_file)} bytes")

                        if not audio_data_or_file.startswith(b'RIFF'):
                            raise ValueError("Invalid audio data - not a WAV file")

                        # Create temporary file
                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_file_path = temp_file.name
                        temp_file.write(audio_data_or_file)
                        temp_file.close()
                        audio_file = temp_file_path

                    else:
                        raise ValueError("Invalid audio input type")

                    # Open and validate WAV file
                    with wave.open(audio_file, 'rb') as wf:
                        frames = wf.getnframes()
                        if frames == 0:
                            raise ValueError("WAV file contains no audio frames")

                        channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                        framerate = wf.getframerate()

                        logger.info(f"Playing audio: {channels}ch, {sample_width}bytes, {framerate}Hz, {frames} frames")

                        # Create playback stream IN THE WORKER THREAD
                        try:
                            stream = self.audio.open(
                                format=self.audio.get_format_from_width(sample_width),
                                channels=channels,
                                rate=framerate,
                                output=True,
                                frames_per_buffer=CHUNK
                            )
                        except Exception as e:
                            raise Exception(f"Failed to create audio stream: {e}")

                        # Play audio in chunks with stop check
                        wf.rewind()
                        data = wf.readframes(CHUNK)

                        while data and not self._stop_playback_event.is_set():
                            try:
                                if stream and stream.is_active():
                                    stream.write(data)
                                    data = wf.readframes(CHUNK)
                                else:
                                    break
                            except Exception as e:
                                logger.error(f"Error during playback: {e}")
                                break

                    # Successful completion
                    if not self._stop_playback_event.is_set() and callback:
                        callback(True, "Playback completed")

                except Exception as e:
                    logger.error(f"Playback worker error: {e}")
                    if callback:
                        callback(False, str(e))

                finally:
                    # CRITICAL: Always cleanup in the same thread
                    if stream:
                        try:
                            if stream.is_active():
                                stream.stop_stream()
                            stream.close()
                        except Exception as e:
                            logger.debug(f"Error closing playback stream: {e}")

                    # Cleanup temp file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            logger.debug(f"Error removing temp file: {e}")

            # Start playback thread
            self._current_playback_thread = threading.Thread(target=playback_worker, daemon=True)
            self._current_playback_thread.start()

        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            if callback:
                callback(False, str(e))
        finally:
            # Always release the lock
            try:
                self._playback_lock.release()
            except:
                pass

    def stop_playback(self):
        """Thread-safe playback stop"""
        try:
            logger.debug("Stopping audio playback...")

            # Signal stop to playback thread
            self._stop_playback_event.set()

            # Wait for playback thread to finish
            if self._current_playback_thread and self._current_playback_thread.is_alive():
                self._current_playback_thread.join(timeout=2.0)

            self._current_playback_thread = None
            logger.debug("Audio playback stopped")

        except Exception as e:
            logger.error(f"Error stopping playback: {e}")

    def cleanup(self):
        """Cleanup all resources"""
        try:
            self.recording = False
            self.stop_playback()

            if self.audio:
                try:
                    self.audio.terminate()
                except:
                    pass
                self.audio = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class DatabaseWorker(QThread):
    operation_started = pyqtSignal(str, str)
    operation_progress = pyqtSignal(str, int, str)
    operation_completed = pyqtSignal(str, object)
    operation_error = pyqtSignal(str, str)
    bans_loaded = pyqtSignal(list)
    logs_loaded = pyqtSignal(list)
    statistics_loaded = pyqtSignal(dict)
    verification_completed = pyqtSignal(str, str, dict)
    connection_tested = pyqtSignal(str, bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.operations_queue = queue.Queue()
        self.current_operation = None
        self.is_running = True
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()

    def add_operation(self, operation_type, operation_id, db_instance, *args, **kwargs):
        operation = {
            'type': operation_type, 'id': operation_id, 'db': db_instance,
            'args': args, 'kwargs': kwargs, 'timestamp': time.time()
        }
        self.operations_queue.put(operation)
        self.mutex.lock()
        self.wait_condition.wakeOne()
        self.mutex.unlock()
        if not self.isRunning():
            self.start()

    def _execute_with_timeout(self, operation, timeout=15):
        """Execute operation with timeout protection - MISSING METHOD"""
        import threading
        import queue

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def operation_thread():
            try:
                result = operation()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=operation_thread, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            logger.warning(f"DatabaseWorker operation timed out after {timeout} seconds")
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        if not exception_queue.empty():
            raise exception_queue.get()

        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Operation completed but no result available")
    def run(self):
        while self.is_running:
            try:
                try:
                    operation = self.operations_queue.get(timeout=1.0)
                except queue.Empty:
                    self.mutex.lock()
                    self.wait_condition.wait(self.mutex, 5000)
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

    # Improved DatabaseWorker _execute_operation method
    def _execute_operation(self, operation):
        """FIXED: Enhanced operation execution with better error handling and verification logic"""
        try:
            op_type, op_id, db = operation['type'], operation['id'], operation['db']
            args, kwargs = operation['args'], operation['kwargs']

            description = {
                'load_bans': 'Loading ban records...',
                'load_logs': 'Loading activity logs...',
                'load_statistics': 'Calculating statistics...',
                'verify_tanker': 'Verifying tanker with Red Entry check...',
                'test_connection': 'Testing database connection...'
            }.get(op_type, f'Executing {op_type}...')

            self.operation_started.emit(op_id, description)

            if op_type == 'verify_tanker':
                # FIXED: Enhanced verification parameter handling
                logger.info(f"DatabaseWorker: Starting verification with args: {args}")

                try:
                    # Validate database connection first
                    if not hasattr(db, 'verify_specific_tanker') or not hasattr(db, 'simple_tanker_verification'):
                        raise Exception("Database verification methods not available")

                    # Enhanced parameter validation and handling
                    if len(args) == 1:
                        # Auto verification: only operator provided
                        operator = str(args[0]).strip() if args[0] else "System"
                        logger.info(f"DatabaseWorker: Auto verification for operator: {operator}")

                        self.operation_progress.emit(op_id, 25, "Getting latest tanker...")

                        # FIXED: Add timeout protection for database operations
                        try:
                            status, reason, details = self._execute_with_timeout(
                                lambda: db.simple_tanker_verification(operator),
                                timeout=15  # 15 second timeout
                            )
                        except Exception as timeout_error:
                            logger.error(f"Auto verification timeout: {timeout_error}")
                            status = "ERROR"
                            reason = "Verification timed out - please try again"
                            details = {"error": True, "tanker_number": "TIMEOUT", "duplicate_detected": False}

                    elif len(args) == 2:
                        # Manual verification: tanker_number and operator
                        tanker_number = str(args[0]).strip().upper() if args[0] else ""
                        operator = str(args[1]).strip() if args[1] else "System"

                        # CRITICAL FIX: Enhanced tanker number validation
                        if not tanker_number or len(tanker_number) < 1:
                            logger.error(f"DatabaseWorker: Invalid tanker number: '{tanker_number}'")
                            status = "ERROR"
                            reason = "Invalid tanker number - please enter a valid tanker number"
                            details = {"error": True, "tanker_number": "INVALID", "duplicate_detected": False}
                        else:
                            logger.info(
                                f"DatabaseWorker: Manual verification for tanker: {tanker_number}, operator: {operator}")

                            self.operation_progress.emit(op_id, 25, f"Verifying {tanker_number}...")
                            self.operation_progress.emit(op_id, 50, "Checking Red Entry duplicates...")
                            self.operation_progress.emit(op_id, 75, "Checking ban records...")

                            # FIXED: Add timeout protection for manual verification
                            try:
                                status, reason, details = self._execute_with_timeout(
                                    lambda: db.verify_specific_tanker(tanker_number, operator),
                                    timeout=20  # 20 second timeout for manual verification
                                )
                            except Exception as timeout_error:
                                logger.error(f"Manual verification timeout: {timeout_error}")
                                status = "ERROR"
                                reason = f"Verification timed out for {tanker_number} - please try again"
                                details = {"error": True, "tanker_number": tanker_number, "duplicate_detected": False}
                    else:
                        # Fallback for unexpected arguments
                        logger.warning(f"DatabaseWorker: Unexpected args count: {len(args)}, using error response")
                        status = "ERROR"
                        reason = "Invalid verification parameters"
                        details = {"error": True, "tanker_number": "INVALID", "duplicate_detected": False}

                    # FIXED: Ensure details has all required keys
                    if not isinstance(details, dict):
                        details = {"error": True, "duplicate_detected": False, "tanker_number": "UNKNOWN"}

                    # Ensure required keys exist
                    required_keys = ["duplicate_detected", "tanker_number", "error"]
                    for key in required_keys:
                        if key not in details:
                            if key == "duplicate_detected":
                                details[key] = False
                            elif key == "error":
                                details[key] = status == "ERROR"
                            elif key == "tanker_number":
                                details[key] = "UNKNOWN"

                    # Enhanced progress feedback
                    self.operation_progress.emit(op_id, 100, "Verification complete")

                    # Emit completion signal
                    logger.info(f"DatabaseWorker: Verification completed with status: {status}")
                    self.verification_completed.emit(status, reason, details)

                except Exception as verify_error:
                    logger.error(f"DatabaseWorker: Verification failed: {verify_error}")
                    error_status = "ERROR"
                    error_reason = f"Verification failed: {str(verify_error)}"
                    error_details = {
                        "error": True,
                        "exception": str(verify_error),
                        "duplicate_detected": False,
                        "tanker_number": "ERROR"
                    }
                    self.verification_completed.emit(error_status, error_reason, error_details)

            else:
                # Handle other operations...
                if op_type == 'load_bans':
                    filters = args[0] if len(args) > 0 else None
                    exclude_blob = kwargs.get('exclude_blob', True)
                    include_inactive = kwargs.get('include_inactive', False)
                    result = db.get_all_bans(filters, exclude_blob, include_inactive)
                    self.bans_loaded.emit(result)

                elif op_type == 'load_logs':
                    result = db.get_recent_logs(*args, **kwargs)
                    self.logs_loaded.emit(result)

                elif op_type == 'load_statistics':
                    self.operation_progress.emit(op_id, 30, "Loading ban statistics...")
                    ban_stats = db.get_ban_statistics(*args, **kwargs)
                    self.operation_progress.emit(op_id, 60, "Loading verification statistics...")
                    verify_stats = db.get_verification_statistics(*args, **kwargs)
                    self.operation_progress.emit(op_id, 80, "Loading recent data...")
                    recent_bans = db.get_all_bans(*args, exclude_blob=True, **kwargs) if hasattr(db,
                                                                                                 'get_all_bans') else []
                    recent_logs = db.get_recent_logs(15, *args, **kwargs) if hasattr(db, 'get_recent_logs') else []
                    result = {'ban_stats': ban_stats, 'verify_stats': verify_stats, 'recent_bans': recent_bans,
                              'recent_logs': recent_logs}
                    self.statistics_loaded.emit(result)

                # Emit completion for non-verification operations
                if op_type != 'verify_tanker':
                    self.operation_completed.emit(op_id, result if 'result' in locals() else None)

        except Exception as e:
            logger.error(f"DatabaseWorker: Error executing operation {op_type}: {e}")
            if op_type == 'verify_tanker':
                error_details = {
                    "error": True,
                    "exception": str(e),
                    "duplicate_detected": False,
                    "tanker_number": "ERROR"
                }
                self.verification_completed.emit("ERROR", f"Operation failed: {str(e)}", error_details)
            else:
                self.operation_error.emit(op_id, str(e))

    def _execute_with_timeout(self, operation, timeout=15):
        """Execute operation with timeout protection"""
        import threading
        import queue

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def operation_thread():
            try:
                result = operation()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        # Start operation in background
        thread = threading.Thread(target=operation_thread, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Thread is still running - timeout
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()

        # Get result
        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Operation completed but no result available")
    def add_verification_timeout_protection(self):
        """Add timeout protection for verification operations"""
        self.verification_timeout_timer = QTimer()
        self.verification_timeout_timer.setSingleShot(True)
        self.verification_timeout_timer.timeout.connect(self.handle_verification_timeout)

    def handle_verification_timeout(self):
        """Handle verification timeout"""
        logger.warning("Verification operation timed out")

        # Reset UI states
        self.auto_status_label.setText("⏰ TIMEOUT")
        self.auto_reason_label.setText("Verification timed out - the operation took too long")

        self.manual_status_label.setText("⏰ TIMEOUT")
        self.manual_reason_label.setText("Verification timed out - the operation took too long")

        # Clear active operations
        self.active_operations.clear()

        self.statusBar().showMessage("Verification timed out")

    def verify_latest_tanker(self):
        """Auto verification with timeout protection"""
        try:
            # Start timeout timer (30 seconds)
            if hasattr(self, 'verification_timeout_timer'):
                self.verification_timeout_timer.start(30000)

            # Immediate UI feedback
            self.auto_status_label.setText("⏳ Verifying...")
            self.auto_reason_label.setText("Checking latest entry for ban records and Red Entry duplicates...")
            self.auto_tanker_info_label.setText("🚛 Retrieving latest entry...")

            # Hide voice section initially
            if hasattr(self, 'auto_voice_frame'):
                self.auto_voice_frame.setVisible(False)

            QApplication.processEvents()

            # Generate operation ID and start verification
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'verify_latest'

            logger.info("Starting auto verification for latest tanker")

            # Add to database worker queue with timeout protection
            self.db_worker.add_operation('verify_tanker', operation_id, self.db, self.user_info['username'])

        except Exception as e:
            logger.error(f"Auto verification error: {e}")
            self.auto_status_label.setText("❌ ERROR")
            self.auto_reason_label.setText(f"Auto verification failed: {str(e)}")


    def on_verification_completed(self, status, reason, details):
        """Enhanced verification completion handler with timeout cleanup"""
        try:
            # Stop timeout timer
            if hasattr(self, 'verification_timeout_timer'):
                self.verification_timeout_timer.stop()

            operation_type = None
            operation_id_to_remove = None

            # Find the operation type
            for op_id, op_type in self.active_operations.items():
                if op_type in ['verify_latest', 'verify_manual']:
                    operation_type = op_type
                    operation_id_to_remove = op_id
                    break

            # Clean up operation
            if operation_id_to_remove:
                self.active_operations.pop(operation_id_to_remove, None)

            tanker_number = details.get("tanker_number", "UNKNOWN")

            # Update appropriate display
            if operation_type == 'verify_latest':
                self.update_auto_verification_display(tanker_number, status, reason, details)
                self.statusBar().showMessage(f"Auto verification: {tanker_number} - {status}")

            elif operation_type == 'verify_manual':
                self.update_manual_verification_display(tanker_number, status, reason, details)
                self.statusBar().showMessage(f"Manual verification: {tanker_number} - {status}")

            # Play sound for important statuses (including duplicates)
            if details.get("play_sound", False) or details.get("duplicate_detected", False):
                self.play_warning_sound_for_status(status)

            # Log the result
            if details.get("duplicate_detected"):
                logger.warning(f"RED ENTRY DUPLICATE DETECTED: {tanker_number} - {reason}")
            else:
                logger.info(f"Verification completed: {tanker_number} - {status}")

        except Exception as e:
            logger.error(f"Error handling verification completed: {e}")
            # Fallback status update
            self.statusBar().showMessage(f"Verification completed with errors: {str(e)}")

    def stop_thread(self):
        self.is_running = False
        self.mutex.lock()
        self.wait_condition.wakeAll()
        self.mutex.unlock()
        if self.isRunning():
            self.wait(5000)


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("LoadingOverlay { background-color: rgba(255, 255, 255, 0.9); border-radius: 8px; }")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.spinner_label = QLabel("⏳")
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_label.setStyleSheet("QLabel { color: #3B82F6; font-size: 32px; margin-bottom: 16px; }")
        layout.addWidget(self.spinner_label)

        self.message_label = QLabel("Loading...")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet(
            "QLabel { color: #6B7280; font-size: 16px; font-weight: 500; margin-bottom: 16px; }")
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; border-radius: 3px; background-color: #E5E7EB; }
            QProgressBar::chunk { background-color: #3B82F6; border-radius: 3px; }
        """)
        layout.addWidget(self.progress_bar)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spinner)
        self.spinner_chars = ["⏳", "⌛", "⏳", "⌛"]
        self.spinner_index = 0
        self.hide()

    def show_loading(self, message="Loading..."):
        self.message_label.setText(message)
        self.progress_bar.setValue(0)
        self.timer.start(500)
        self.show()
        self.raise_()

    def hide_loading(self):
        self.timer.stop()
        self.hide()

    def update_progress(self, percentage, message=None):
        self.progress_bar.setValue(percentage)
        if message:
            self.message_label.setText(message)

    def update_spinner(self):
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
        self.spinner_label.setText(self.spinner_chars[self.spinner_index])


class ModernLoginDialog(QDialog):
    def __init__(self, user_manager):
        super().__init__()
        self.user_manager = user_manager
        self.user_info = None
        self.is_password_visible = False
        self.init_ui()
        self.apply_modern_styles()

    def init_ui(self):
        self.setWindowTitle("TDF System - Login")
        # Calculate responsive size
        screen = QApplication.primaryScreen().geometry()
        dialog_width = min(900, int(screen.width() * 0.7))
        dialog_height = min(600, int(screen.height() * 0.7))
        self.setFixedSize(dialog_width, dialog_height)
        self.setWindowFlags(Qt.FramelessWindowHint)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_panel = self.create_branding_panel()
        right_panel = self.create_form_panel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        self.setLayout(main_layout)
        self.setup_keyboard_shortcuts()

    def create_branding_panel(self):
        left_panel = QFrame()
        left_panel.setFixedWidth(450)
        left_panel.setObjectName("leftPanel")

        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setSpacing(int(ModernUITheme.SPACE_4XL.replace('px', '')))
        left_layout.setContentsMargins(40, 60, 40, 60)

        logo_container = QFrame()
        logo_container.setFixedSize(140, 140)
        logo_container.setObjectName("logoContainer")

        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setAlignment(Qt.AlignCenter)

        logo_label = QLabel("🚛")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet(f"font-size: {ModernUITheme.FONT_SIZE_5XL}; color: white; background: transparent;")
        logo_layout.addWidget(logo_label)

        brand_title = QLabel("TDF System")
        brand_title.setAlignment(Qt.AlignCenter)
        brand_title.setStyleSheet(
            f"font-size: {ModernUITheme.FONT_SIZE_4XL}; font-weight: 700; color: white; letter-spacing: -0.8px; margin-bottom: {ModernUITheme.SPACE_LG};")

        subtitle = QLabel("Professional Vehicle Management")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: {ModernUITheme.FONT_SIZE_XL}; color: rgba(255, 255, 255, 0.9); font-weight: 500; margin-bottom: {ModernUITheme.SPACE_2XL};")

        features = QLabel(
            "✨ Modern Professional Interface\n🔒 Enhanced Security\n📊 Real-time Monitoring\n🔍 Advanced Search & Filters")
        features.setAlignment(Qt.AlignCenter)
        features.setStyleSheet(
            f"font-size: {ModernUITheme.FONT_SIZE_BASE}; color: rgba(255, 255, 255, 0.85); line-height: 2.0; font-weight: 400;")

        left_layout.addWidget(logo_container)
        left_layout.addWidget(brand_title)
        left_layout.addWidget(subtitle)
        left_layout.addWidget(features)
        left_layout.addStretch()

        left_panel.setLayout(left_layout)
        return left_panel

    def create_form_panel(self):
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")

        right_layout = QVBoxLayout()
        right_layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        right_layout.setContentsMargins(60, 40, 60, 40)

        header = self.create_header()
        form_content = self.create_form_content()

        right_layout.addWidget(header)
        right_layout.addStretch(1)
        right_layout.addWidget(form_content)
        right_layout.addStretch(2)

        right_panel.setLayout(right_layout)
        return right_panel

    def create_header(self):
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(44, 44)
        close_btn.setObjectName("closeButton")
        close_btn.clicked.connect(self.reject)
        close_btn.setToolTip("Close")

        header_layout.addWidget(close_btn)
        return header_widget

    def create_form_content(self):
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(int(ModernUITheme.SPACE_3XL.replace('px', '')))
        content_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Welcome Back")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("welcomeTitle")

        form_container = self.create_form_inputs()
        actions_container = self.create_action_buttons()

        self.error_label = QLabel()
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.hide()

        content_layout.addWidget(title)
        content_layout.addWidget(form_container)
        content_layout.addWidget(actions_container)
        content_layout.addWidget(self.error_label)

        return content_widget

    def create_form_inputs(self):
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(int(ModernUITheme.SPACE_3XL.replace('px', '')))
        form_layout.setContentsMargins(0, 0, 0, 0)

        username_group = self.create_input_group("Username", "Enter your username", "👤", "admin")
        self.username_input = username_group['input']

        password_group = self.create_password_input_group()
        self.password_input = password_group['input']
        self.toggle_password_btn = password_group['toggle_btn']

        remember_section = self.create_remember_section()

        form_layout.addWidget(username_group['widget'])
        form_layout.addWidget(password_group['widget'])
        form_layout.addWidget(remember_section)

        return form_widget

    def create_input_group(self, label_text, placeholder, icon, default_value=""):
        group_widget = QWidget()
        group_layout = QVBoxLayout(group_widget)
        group_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))
        group_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        label.setObjectName("inputLabel")

        container = QFrame()
        container.setObjectName("inputContainer")
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(20, 0, 20, 0)
        container_layout.setSpacing(12)

        icon_label = QLabel(icon)
        icon_label.setObjectName("inputIcon")

        input_field = QLineEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setObjectName("modernInput")
        if default_value:
            input_field.setText(default_value)

        container_layout.addWidget(icon_label)
        container_layout.addWidget(input_field)

        group_layout.addWidget(label)
        group_layout.addWidget(container)

        return {'widget': group_widget, 'input': input_field, 'container': container}

    def create_password_input_group(self):
        group_widget = QWidget()
        group_layout = QVBoxLayout(group_widget)
        group_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))
        group_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Password")
        label.setObjectName("inputLabel")

        container = QFrame()
        container.setObjectName("inputContainer")
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(20, 0, 20, 0)
        container_layout.setSpacing(12)

        icon_label = QLabel("🔒")
        icon_label.setObjectName("inputIcon")

        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        password_input.setPlaceholderText("Enter your password")
        password_input.setObjectName("modernInput")

        toggle_btn = QToolButton()
        toggle_btn.setText("👁")
        toggle_btn.setObjectName("togglePasswordButton")
        toggle_btn.setFixedSize(32, 32)
        toggle_btn.setCursor(Qt.PointingHandCursor)
        toggle_btn.clicked.connect(self.toggle_password_visibility)
        toggle_btn.setToolTip("Show/Hide password")

        container_layout.addWidget(icon_label)
        container_layout.addWidget(password_input)
        container_layout.addWidget(toggle_btn)

        group_layout.addWidget(label)
        group_layout.addWidget(container)

        return {'widget': group_widget, 'input': password_input, 'toggle_btn': toggle_btn}

    def create_remember_section(self):
        section_widget = QWidget()
        section_layout = QHBoxLayout(section_widget)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        remember_me = QCheckBox("Remember me")
        remember_me.setObjectName("rememberCheckbox")
        remember_me.setCursor(Qt.PointingHandCursor)

        forgot_password = QLabel(
            f"<a href='#' style='text-decoration:none; color: {ModernUITheme.PRIMARY};'>Forgot password?</a>")
        forgot_password.setObjectName("forgotPasswordLink")
        forgot_password.setCursor(Qt.PointingHandCursor)
        forgot_password.setOpenExternalLinks(False)
        forgot_password.linkActivated.connect(self.show_password_reset)

        section_layout.addWidget(remember_me)
        section_layout.addStretch()
        section_layout.addWidget(forgot_password)

        return section_widget

    def create_action_buttons(self):
        actions_widget = QWidget()
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(15)

        self.loading_spinner = QLabel()
        self.loading_spinner.setFixedSize(24, 24)
        self.loading_spinner.setObjectName("loadingSpinner")
        self.loading_spinner.hide()

        self.login_btn = QPushButton("Sign In")
        self.login_btn.setObjectName("primaryButton")
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setDefault(True)

        button_layout.addStretch()
        button_layout.addWidget(self.loading_spinner)
        button_layout.addWidget(self.login_btn)
        button_layout.addStretch()

        actions_layout.addWidget(button_container)
        return actions_widget

    def setup_keyboard_shortcuts(self):
        self.password_input.returnPressed.connect(self.login)
        self.username_input.returnPressed.connect(self.password_input.setFocus)

        escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        escape_shortcut.activated.connect(self.reject)

    def toggle_password_visibility(self):
        self.is_password_visible = not self.is_password_visible
        if self.is_password_visible:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_btn.setText("🙈")
            self.toggle_password_btn.setToolTip("Hide password")
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_btn.setText("👁")
            self.toggle_password_btn.setToolTip("Show password")

    def show_password_reset(self):
        QMessageBox.information(self, "Password Reset",
                                "Password reset functionality would be implemented here.\n\nFor now, please contact your system administrator.",
                                QMessageBox.Ok)

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username:
            self.show_error("Please enter your username")
            self.username_input.setFocus()
            return
        if not password:
            self.show_error("Please enter your password")
            self.password_input.setFocus()
            return

        self.error_label.hide()
        self.set_loading(True)

        try:
            self.user_info = self.user_manager.authenticate_user(username, password)
            if self.user_info:
                logger.info(f"Login successful for: {username}")
                self.accept()
            else:
                error_msg = self.get_detailed_error_message(username)
                self.show_error(error_msg)
                logger.warning(f"Login failed for user: {username}")
        except Exception as e:
            error_msg = f"Login error: {str(e)}"
            logger.error(f"Login exception for {username}: {e}")
            self.show_error(error_msg)
        finally:
            self.set_loading(False)

    def get_detailed_error_message(self, username):
        try:
            with sqlite3.connect(self.user_manager.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT username, is_active FROM users WHERE username = ?", (username,))
                user_check = cursor.fetchone()

                if user_check:
                    if user_check[1] == 0:
                        return f"Account '{username}' is inactive. Please contact administrator."
                    else:
                        return f"Invalid password for user '{username}'"
                else:
                    return f"User '{username}' not found"
        except Exception as e:
            logger.error(f"Error checking user existence: {e}")
            return f"Authentication failed for '{username}'"

    def show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()

    def set_loading(self, loading):
        if loading:
            self.login_btn.setText("Authenticating...")
            self.login_btn.setEnabled(False)
            self.loading_spinner.show()
            self.username_input.setEnabled(False)
            self.password_input.setEnabled(False)
        else:
            self.login_btn.setText("Sign In")
            self.login_btn.setEnabled(True)
            self.loading_spinner.hide()
            self.username_input.setEnabled(True)
            self.password_input.setEnabled(True)
        QApplication.processEvents()

    def apply_modern_styles(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {ModernUITheme.BACKGROUND}; font-family: {ModernUITheme.FONT_FAMILY}; }}
            #leftPanel {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ModernUITheme.PRIMARY}, stop:1 {ModernUITheme.PRIMARY_LIGHT}); border-top-left-radius: {ModernUITheme.RADIUS_XL}; border-bottom-left-radius: {ModernUITheme.RADIUS_XL}; }}
            #logoContainer {{ background-color: rgba(255, 255, 255, 0.1); border-radius: {ModernUITheme.RADIUS_XL}; border: 2px solid rgba(255, 255, 255, 0.2); }}
            #rightPanel {{ background-color: {ModernUITheme.BACKGROUND}; border-top-right-radius: {ModernUITheme.RADIUS_XL}; border-bottom-right-radius: {ModernUITheme.RADIUS_XL}; }}
            #closeButton {{ background-color: transparent; border: none; color: {ModernUITheme.TEXT_MUTED}; font-size: {ModernUITheme.FONT_SIZE_LG}; font-weight: bold; border-radius: {ModernUITheme.RADIUS_MD}; }}
            #closeButton:hover {{ background-color: {ModernUITheme.SURFACE}; color: {ModernUITheme.TEXT_SECONDARY}; }}
            #inputContainer {{ border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; background-color: {ModernUITheme.BACKGROUND}; }}
            #modernInput {{ border: none; padding: {ModernUITheme.SPACE_LG} 0; font-size: {ModernUITheme.FONT_SIZE_BASE}; background-color: transparent; color: {ModernUITheme.TEXT_PRIMARY}; min-height: 20px; font-weight: 500; }}
            #modernInput:focus {{ outline: none; }}
            #inputContainer:focus-within {{ border-color: {ModernUITheme.PRIMARY}; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }}
            #modernInput::placeholder {{ color: {ModernUITheme.TEXT_MUTED}; }}
            #togglePasswordButton {{ background-color: transparent; border: none; color: {ModernUITheme.TEXT_MUTED}; }}
            #togglePasswordButton:hover {{ color: {ModernUITheme.PRIMARY}; }}
            #primaryButton {{ background-color: {ModernUITheme.PRIMARY}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-weight: 600; font-size: {ModernUITheme.FONT_SIZE_BASE}; padding: {ModernUITheme.SPACE_LG} {ModernUITheme.SPACE_2XL}; min-height: 48px; min-width: 120px; }}
            #primaryButton:hover {{ background-color: {ModernUITheme.PRIMARY_DARK}; }}
            #errorLabel {{ color: {ModernUITheme.DANGER}; background-color: rgba(220, 38, 38, 0.1); font-size: {ModernUITheme.FONT_SIZE_SM}; padding: {ModernUITheme.SPACE_MD}; border-radius: {ModernUITheme.RADIUS_MD}; border: 1px solid rgba(220, 38, 38, 0.2); }}
        """)


def create_audio_recorder():
    try:
        if AUDIO_AVAILABLE:
            return ThreadSafeAudioRecorder()
        else:
            return FallbackAudioRecorder()
    except Exception as e:
        logger.error(f"AudioRecorder failed: {e}")
        return FallbackAudioRecorder()


class FallbackAudioRecorder:
    def __init__(self):
        self.audio = None
        self.recording = False
        self.frames = []
        self.current_playback_stream = None

    def play_audio(self, audio_data_or_file, callback=None):
        QMessageBox.information(None, "Audio Playback", "Audio playback not available")
        if callback:
            callback(False, "Audio not available")

    def stop_playback(self):
        pass

    def start_recording(self):
        self.recording = True
        return True

    def stop_recording(self):
        self.recording = False
        return b"dummy_audio_data"

    def stop_any_operation(self):
        pass


class VoiceRecordingDialog(QDialog):
    """Complete voice recording dialog with all required methods"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_recorder = None
        self.recorded_data = None
        self.is_recording = False
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_timer)
        self.recording_time = 0

        # Use thread-safe audio recorder
        if AUDIO_AVAILABLE:
            try:
                self.audio_recorder = ThreadSafeAudioRecorder()
                logger.info("Voice dialog: ThreadSafeAudioRecorder initialized")
            except Exception as e:
                logger.error(f"Voice dialog: Failed to initialize audio: {e}")
                self.audio_recorder = None
        else:
            logger.warning("Voice dialog: Audio not available")

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        """Initialize the complete user interface"""
        self.setWindowTitle("Voice Recording")
        self.setFixedSize(500, 400)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Header section
        header_layout = QHBoxLayout()

        icon_label = QLabel("🎤")
        icon_label.setStyleSheet("font-size: 48px;")
        header_layout.addWidget(icon_label)

        title_container = QVBoxLayout()
        title_label = QLabel("Voice Recording")
        title_label.setObjectName("dialogTitle")

        subtitle_label = QLabel("Record a voice note for this ban record")
        subtitle_label.setObjectName("dialogSubtitle")

        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)

        header_layout.addLayout(title_container)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Status display section
        self.status_container = QFrame()
        self.status_container.setObjectName("statusContainer")
        status_layout = QVBoxLayout(self.status_container)
        status_layout.setContentsMargins(20, 20, 20, 20)

        self.status_label = QLabel("⚪ Ready to record")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)

        self.timer_label = QLabel("00:00")
        self.timer_label.setObjectName("timerLabel")
        self.timer_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.timer_label)

        main_layout.addWidget(self.status_container)

        # Control buttons section
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.record_btn = QPushButton("🎙️ Start Recording")
        self.record_btn.setObjectName("recordButton")
        self.record_btn.clicked.connect(self.toggle_recording)

        # Check if audio is available
        if not AUDIO_AVAILABLE or not self.audio_recorder:
            self.record_btn.setEnabled(False)
            self.record_btn.setText("🚫 Audio Not Available")
            self.status_label.setText("❌ Audio recording not available on this system")

        control_layout.addWidget(self.record_btn)

        self.play_btn = QPushButton("🎵 Play")
        self.play_btn.setObjectName("playButton")
        self.play_btn.clicked.connect(self.play_recording)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)

        main_layout.addLayout(control_layout)

        # Dialog buttons section
        button_layout = QHBoxLayout()

        info_label = QLabel("💡 Click 'Start Recording' to begin, click again to stop")
        info_label.setObjectName("infoLabel")
        button_layout.addWidget(info_label)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelButton")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.save_btn = QPushButton("Save Recording")
        self.save_btn.setObjectName("saveButton")
        self.save_btn.clicked.connect(self.accept)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        main_layout.addLayout(button_layout)

    def toggle_recording(self):
        """Toggle recording state with enhanced error handling"""
        if not self.audio_recorder:
            QMessageBox.warning(self, "Audio Error",
                                "Audio recording is not available.\n\n"
                                "Please ensure your system has a microphone and the required audio libraries are installed.")
            return

        try:
            if not self.is_recording:
                # Start recording
                logger.info("Starting voice recording...")
                self.recorded_data = None  # Clear any previous data

                success = self.audio_recorder.start_recording()
                if success:
                    self.is_recording = True
                    self.recording_time = 0
                    self.recording_timer.start(1000)  # Update every second

                    self.record_btn.setText("⏹️ Stop Recording")
                    self.status_label.setText("🔴 Recording in progress...")
                    self.play_btn.setEnabled(False)
                    self.save_btn.setEnabled(False)

                    logger.info("Voice recording started successfully")
                else:
                    QMessageBox.warning(self, "Recording Error", "Failed to start recording")

            else:
                # Stop recording
                logger.info("Stopping voice recording...")

                # Stop timer first
                self.recording_timer.stop()

                # Stop recording
                self.is_recording = False

                # Allow a brief moment for the recording to settle
                QApplication.processEvents()
                time.sleep(0.2)

                # Get recorded data
                self.recorded_data = self.audio_recorder.stop_recording()

                if self.recorded_data and len(self.recorded_data) > 1000:
                    # Validate WAV format
                    if self.recorded_data.startswith(b'RIFF') and b'WAVE' in self.recorded_data[:20]:
                        self.record_btn.setText("🎙️ Record Again")
                        self.status_label.setText("🟢 Recording completed successfully")
                        self.play_btn.setEnabled(True)
                        self.save_btn.setEnabled(True)
                        logger.info(f"Voice recording completed: {len(self.recorded_data)} bytes")
                    else:
                        # Try to fix the audio data
                        logger.warning("Audio data missing proper WAV header, attempting to fix...")
                        self.recorded_data = self._create_proper_wav_file(self.recorded_data)
                        if self.recorded_data and len(self.recorded_data) > 1000:
                            self.record_btn.setText("🎙️ Record Again")
                            self.status_label.setText("🟡 Recording completed (format corrected)")
                            self.play_btn.setEnabled(True)
                            self.save_btn.setEnabled(True)
                            logger.info(f"Voice recording completed with correction: {len(self.recorded_data)} bytes")
                        else:
                            self._handle_recording_failure()
                else:
                    self._handle_recording_failure()

        except Exception as e:
            logger.error(f"Error during recording toggle: {e}")
            self.is_recording = False
            self.recording_timer.stop()
            self.record_btn.setText("🎙️ Start Recording")
            self.status_label.setText("❌ Recording error occurred")
            QMessageBox.critical(self, "Recording Error", f"An error occurred during recording:\n\n{str(e)}")

    def play_recording(self):
        """Play recorded audio with enhanced error handling"""
        if not self.recorded_data or not self.audio_recorder:
            QMessageBox.warning(self, "Playback Error", "No recording available to play")
            return

        if len(self.recorded_data) < 1000:
            QMessageBox.warning(self, "Playback Error",
                                f"Recording data is too small to play ({len(self.recorded_data)} bytes)")
            return

        try:
            self.status_label.setText("🔊 Playing recording...")
            self.play_btn.setEnabled(False)

            def on_playback_finished(success, message):
                try:
                    if success:
                        self.status_label.setText("🟢 Recording completed successfully")
                        logger.info("Voice playback completed successfully")
                    else:
                        self.status_label.setText(f"❌ Playback failed: {message}")
                        logger.error(f"Voice playback failed: {message}")
                    self.play_btn.setEnabled(True)
                except Exception as e:
                    logger.error(f"Error in playback callback: {e}")
                    self.play_btn.setEnabled(True)

            self.audio_recorder.play_audio(self.recorded_data, callback=on_playback_finished)

        except Exception as e:
            logger.error(f"Error playing recording: {e}")
            self.status_label.setText("❌ Playback error occurred")
            self.play_btn.setEnabled(True)
            QMessageBox.critical(self, "Playback Error", f"Failed to play recording:\n\n{str(e)}")

    def update_recording_timer(self):
        """Update the recording timer display - REQUIRED METHOD"""
        self.recording_time += 1
        minutes = self.recording_time // 60
        seconds = self.recording_time % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

        # Stop recording after 30 seconds (configurable)
        if self.recording_time >= RECORD_SECONDS:
            self.toggle_recording()

    def _handle_recording_failure(self):
        """Handle recording failure consistently"""
        self.record_btn.setText("🎙️ Start Recording")
        self.status_label.setText("❌ Recording failed - please try again")
        self.timer_label.setText("00:00")
        self.recorded_data = None
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        logger.warning("Voice recording failed - insufficient data")

        # Show helpful message
        QMessageBox.information(self, "Recording Issue",
                                "Recording failed - the audio data is too small.\n\n"
                                "Tips for better recording:\n"
                                "• Speak closer to the microphone\n"
                                "• Record for at least 2-3 seconds\n"
                                "• Check microphone permissions\n"
                                "• Ensure microphone is not muted")

    def _create_proper_wav_file(self, raw_data):
        """Create a proper WAV file from raw audio data"""
        import struct

        if not raw_data or len(raw_data) < 100:
            return None

        try:
            # Standard WAV parameters
            sample_rate = 44100
            bits_per_sample = 16
            channels = 1
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            data_size = len(raw_data)

            # Create WAV header
            wav_header = struct.pack('<4sI4s4s4sIHHIIHH4sI',
                                     b'RIFF',  # Chunk ID
                                     36 + data_size,  # Chunk size
                                     b'WAVE',  # Format
                                     b'fmt ',  # Subchunk1 ID
                                     16,  # Subchunk1 size (PCM)
                                     1,  # Audio format (PCM)
                                     channels,  # Number of channels
                                     sample_rate,  # Sample rate
                                     byte_rate,  # Byte rate
                                     block_align,  # Block align
                                     bits_per_sample,  # Bits per sample
                                     b'data',  # Subchunk2 ID
                                     data_size  # Subchunk2 size
                                     )

            complete_wav = wav_header + raw_data
            logger.info(f"Created proper WAV file: {len(complete_wav)} bytes")
            return complete_wav

        except Exception as e:
            logger.error(f"Error creating proper WAV file: {e}")
            return None

    def get_recorded_data(self):
        """Get the recorded audio data"""
        return self.recorded_data

    def closeEvent(self, event):
        """Handle dialog close event"""
        try:
            if self.is_recording:
                reply = QMessageBox.question(self, "Recording in Progress",
                                             "Recording is still in progress. Stop recording and close?",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.toggle_recording()
                    event.accept()
                else:
                    event.ignore()
                    return

            # Clean up audio recorder
            if self.audio_recorder:
                try:
                    self.audio_recorder.cleanup()
                except Exception as e:
                    logger.debug(f"Error cleaning up audio recorder: {e}")

            event.accept()

        except Exception as e:
            logger.error(f"Error closing voice dialog: {e}")
            event.accept()

    def apply_styles(self):
        """Apply modern styling to the dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #0F172A;
            }
            #dialogTitle {
                font-size: 20px;
                font-weight: 700;
                color: #0F172A;
                margin: 0;
            }
            #dialogSubtitle {
                font-size: 13px;
                color: #94A3B8;
                margin: 0;
            }
            #statusContainer {
                background-color: #F3F4F6;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                margin: 20px 0;
            }
            #statusLabel {
                font-size: 16px;
                font-weight: 600;
                color: #0F172A;
                margin-bottom: 10px;
            }
            #timerLabel {
                font-size: 24px;
                font-weight: 700;
                color: #2563EB;
                font-family: 'Consolas', monospace;
            }
            #recordButton {
                background-color: #DC2626;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                min-height: 44px;
            }
            #recordButton:hover {
                background-color: #B91C1C;
            }
            #recordButton:disabled {
                background-color: #94A3B8;
            }
            #playButton {
                background-color: #059669;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                min-height: 44px;
            }
            #playButton:hover {
                background-color: #047857;
            }
            #playButton:disabled {
                background-color: #94A3B8;
            }
            #saveButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                min-height: 44px;
            }
            #saveButton:hover {
                background-color: #1D4ED8;
            }
            #saveButton:disabled {
                background-color: #94A3B8;
            }
            #cancelButton {
                background-color: transparent;
                color: #475569;
                border: 2px solid #E2E8F0;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                min-height: 44px;
            }
            #cancelButton:hover {
                background-color: #F3F4F6;
                border-color: #94A3B8;
            }
            #infoLabel {
                font-size: 13px;
                color: #94A3B8;
                font-style: italic;
            }
        """)


# ALSO NEED THE COMPLETE THREAD-SAFE AUDIO RECORDER

class ThreadSafeAudioRecorder:
    """Complete thread-safe audio recorder to prevent heap corruption"""

    def __init__(self):
        self.audio = None
        self.recording = False
        self.frames = []
        self.recording_thread = None
        self.recording_stream = None

        # Thread-safe playback management
        self._playback_lock = threading.Lock()
        self._current_playback_thread = None
        self._stop_playback_event = threading.Event()

        if AUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                logger.info("ThreadSafeAudioRecorder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PyAudio: {e}")
                self.audio = None

    def start_recording(self):
        """Thread-safe recording start"""
        try:
            if not AUDIO_AVAILABLE or not self.audio:
                logger.error("Audio not available for recording")
                return False

            if self.recording:
                logger.warning("Recording already in progress")
                return False

            # Stop any existing playback first
            self.stop_playback()

            self.frames = []
            self.recording = True

            def recording_worker():
                """Isolated recording worker thread"""
                stream = None
                try:
                    # Create stream in the worker thread
                    stream = self.audio.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK
                    )

                    logger.info("Recording started in worker thread")
                    start_time = time.time()

                    while self.recording and (time.time() - start_time) < RECORD_SECONDS:
                        try:
                            if stream and stream.is_active():
                                data = stream.read(CHUNK, exception_on_overflow=False)
                                if self.recording and len(data) > 0:
                                    self.frames.append(data)
                            else:
                                break
                        except Exception as e:
                            logger.error(f"Error reading audio data: {e}")
                            break

                    logger.info(f"Recording finished. Captured {len(self.frames)} frames")

                except Exception as e:
                    logger.error(f"Recording worker error: {e}")
                    self.recording = False
                finally:
                    # Always clean up stream in the same thread
                    if stream:
                        try:
                            if stream.is_active():
                                stream.stop_stream()
                            stream.close()
                        except Exception as e:
                            logger.debug(f"Error closing recording stream: {e}")

            # Start recording thread
            self.recording_thread = threading.Thread(target=recording_worker, daemon=True)
            self.recording_thread.start()

            return True

        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.recording = False
            return False

    def stop_recording(self):
        """Thread-safe recording stop"""
        try:
            if not self.recording:
                logger.debug("No recording in progress")
                return None

            logger.info("Stopping recording...")
            self.recording = False

            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=3.0)

            if not self.frames:
                logger.warning("No audio frames recorded")
                return None

            logger.info(f"Processing {len(self.frames)} audio frames")

            # Create WAV file safely
            return self._create_wav_file_safe()

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None
        finally:
            self.recording = False
            self.recording_thread = None

    def _create_wav_file_safe(self):
        """Safely create WAV file with proper error handling"""
        temp_file = None
        try:
            # Calculate total size
            total_size = sum(len(frame) for frame in self.frames)
            if total_size < 1000:
                logger.error(f"Audio data too small: {total_size} bytes")
                return None

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_filename = temp_file.name
            temp_file.close()

            # Write WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)

                # Write frames safely
                for frame in self.frames:
                    wf.writeframes(frame)

            # Read complete file
            with open(temp_filename, 'rb') as f:
                wav_data = f.read()

            # Cleanup
            os.unlink(temp_filename)

            # Validate
            if len(wav_data) < 1000 or not wav_data.startswith(b'RIFF'):
                logger.error(f"Invalid WAV file created: {len(wav_data)} bytes")
                return None

            logger.info(f"WAV file created successfully: {len(wav_data)} bytes")
            return wav_data

        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            return None

    def play_audio(self, audio_data_or_file, callback: Optional[Callable] = None):
        """Thread-safe audio playback to prevent heap corruption"""
        if not AUDIO_AVAILABLE or not self.audio:
            if callback:
                callback(False, "Audio playback not available")
            return

        # Use lock to prevent multiple playback threads
        if not self._playback_lock.acquire(blocking=False):
            logger.warning("Another playback is already in progress")
            if callback:
                callback(False, "Another playback is in progress")
            return

        try:
            # Stop any existing playback
            self._stop_playback_event.set()
            if self._current_playback_thread and self._current_playback_thread.is_alive():
                self._current_playback_thread.join(timeout=1.0)

            # Reset stop event
            self._stop_playback_event.clear()

            def playback_worker():
                """Isolated playback worker to prevent memory corruption"""
                temp_file_path = None
                stream = None

                try:
                    # Prepare audio file
                    if isinstance(audio_data_or_file, str):
                        audio_file = audio_data_or_file
                        if not os.path.exists(audio_file):
                            raise FileNotFoundError(f"Audio file not found: {audio_file}")

                    elif isinstance(audio_data_or_file, bytes):
                        # Validate audio data
                        if len(audio_data_or_file) < 1000:
                            raise ValueError(f"Audio data too small: {len(audio_data_or_file)} bytes")

                        if not audio_data_or_file.startswith(b'RIFF'):
                            raise ValueError("Invalid audio data - not a WAV file")

                        # Create temporary file
                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_file_path = temp_file.name
                        temp_file.write(audio_data_or_file)
                        temp_file.close()
                        audio_file = temp_file_path

                    else:
                        raise ValueError("Invalid audio input type")

                    # Open and validate WAV file
                    with wave.open(audio_file, 'rb') as wf:
                        frames = wf.getnframes()
                        if frames == 0:
                            raise ValueError("WAV file contains no audio frames")

                        channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                        framerate = wf.getframerate()

                        logger.info(f"Playing audio: {channels}ch, {sample_width}bytes, {framerate}Hz, {frames} frames")

                        # Create playback stream IN THE WORKER THREAD
                        try:
                            stream = self.audio.open(
                                format=self.audio.get_format_from_width(sample_width),
                                channels=channels,
                                rate=framerate,
                                output=True,
                                frames_per_buffer=CHUNK
                            )
                        except Exception as e:
                            raise Exception(f"Failed to create audio stream: {e}")

                        # Play audio in chunks with stop check
                        wf.rewind()
                        data = wf.readframes(CHUNK)

                        while data and not self._stop_playback_event.is_set():
                            try:
                                if stream and stream.is_active():
                                    stream.write(data)
                                    data = wf.readframes(CHUNK)
                                else:
                                    break
                            except Exception as e:
                                logger.error(f"Error during playback: {e}")
                                break

                    # Successful completion
                    if not self._stop_playback_event.is_set() and callback:
                        callback(True, "Playback completed")

                except Exception as e:
                    logger.error(f"Playback worker error: {e}")
                    if callback:
                        callback(False, str(e))

                finally:
                    # CRITICAL: Always cleanup in the same thread
                    if stream:
                        try:
                            if stream.is_active():
                                stream.stop_stream()
                            stream.close()
                        except Exception as e:
                            logger.debug(f"Error closing playback stream: {e}")

                    # Cleanup temp file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            logger.debug(f"Error removing temp file: {e}")

            # Start playback thread
            self._current_playback_thread = threading.Thread(target=playback_worker, daemon=True)
            self._current_playback_thread.start()

        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            if callback:
                callback(False, str(e))
        finally:
            # Always release the lock
            try:
                self._playback_lock.release()
            except:
                pass

    def stop_playback(self):
        """Thread-safe playback stop"""
        try:
            logger.debug("Stopping audio playback...")

            # Signal stop to playback thread
            self._stop_playback_event.set()

            # Wait for playback thread to finish
            if self._current_playback_thread and self._current_playback_thread.is_alive():
                self._current_playback_thread.join(timeout=2.0)

            self._current_playback_thread = None
            logger.debug("Audio playback stopped")

        except Exception as e:
            logger.error(f"Error stopping playback: {e}")

    def cleanup(self):
        """Cleanup all resources"""
        try:
            self.recording = False
            self.stop_playback()

            if self.audio:
                try:
                    self.audio.terminate()
                except:
                    pass
                self.audio = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class MainWindow(QMainWindow):
    def __init__(self, user_info, config_manager):
        super().__init__()
        self.data_cache = {
            'ban_records': None,
            'logs': None,
            'statistics': None,
            'cache_time': {}
        }
        self._max_cache_age = 30  # seconds
        self._operation_cache = {}
        self._last_operation_time = {}

        # Initialize performance optimizations safely
        try:
            self.init_performance_optimizations()
        except Exception as e:
            logger.error(f"Performance optimization error (non-critical): {e}")
            # Create fallback attributes
            self.operation_debounce_timer = None

        # Initialize basic attributes first
        self.user_info = user_info
        self.config = config_manager
        self.currently_playing_row = None
        self.last_tanker = None
        self.operation_counter = 0
        self.active_operations = {}
        self.loading_overlays = {}
        self._button_handlers = {}


        # Initialize performance optimizations
        self.init_performance_optimizations()
        # Use thread-safe audio recorder
        try:
            if AUDIO_AVAILABLE:
                self.audio_recorder = ThreadSafeAudioRecorder()
                logger.info("MainWindow: ThreadSafeAudioRecorder initialized")
            else:
                self.audio_recorder = None
                logger.warning("MainWindow: Audio not available")
        except Exception as e:
            logger.error(f"MainWindow: Failed to initialize audio: {e}")
            self.audio_recorder = None

        # Initialize warning sound safely
        try:
            if AUDIO_AVAILABLE and self.audio_recorder:
                self.warning_sound = WarningSound(duration=self.config.get('warning_sound_duration', 3.0))
            else:
                self.warning_sound = None
                logger.warning("WarningSound not available")
        except Exception as e:
            logger.error(f"Failed to initialize WarningSound: {e}")
            self.warning_sound = None

        # Initialize sound settings
        self.sound_enabled = self.config.get('sound_enabled', True)
        self.current_sound_playing = False

        # Initialize database worker
        self.db_worker = DatabaseWorker()
        self.setup_database_signals()

        # Initialize database manager
        try:
            logger.info("Initializing enhanced database manager for network shares...")
            self.db = EnhancedDatabaseManager(self.config)
            logger.info("Enhanced DatabaseManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced DatabaseManager: {e}")
            logger.info("Falling back to original DatabaseManager...")
            try:
                self.db = EnhancedDatabaseManager(self.config)
                logger.info("Fallback DatabaseManager initialized")
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise

        # Initialize user manager
        try:
            self.user_manager = UserManager(self.config.get('local_sqlite_path'))
            logger.info("UserManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize UserManager: {e}")
            raise

        # Set window properties
        self.setWindowTitle(f"TDF System - Modern UI - User: {user_info['full_name']} ({user_info['role']})")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize filter states
        self.dashboard_filters_applied = False
        self.ban_filters_applied = False
        self.current_dashboard_filters = None
        self.current_ban_filters = None
        self.setup_cross_platform_window()
        # Initialize UI
        try:
            self.init_ui()
            self.apply_modern_styles()
            logger.info("UI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize UI: {e}")
            raise

        # Setup loading overlays
        self.setup_loading_overlays()
        self.add_verification_timeout_protection()
        # Start delayed initialization
        QTimer.singleShot(500, self.initial_dashboard_load_async)
        QTimer.singleShot(1000, self.start_monitoring)

        logger.info(
            f"Modern UI main window initialized for user: {user_info['username']} with role: {user_info['role']}")

    def setup_cross_platform_window(self):
        """Setup window for better cross-platform compatibility"""
        # Get screen size
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Set responsive window size (never bigger than screen)
        window_width = min(1600, int(screen_width * 0.9))
        window_height = min(1000, int(screen_height * 0.9))

        # Set minimum size to prevent layout breaking
        self.setMinimumSize(1200, 800)
        self.resize(window_width, window_height)

        # Center the window
        self.center_window()

    def center_window(self):
        """Center window on screen"""
        screen = QApplication.primaryScreen().geometry()
        window = self.frameGeometry()
        center_point = screen.center()
        window.moveCenter(center_point)
        self.move(window.topLeft())
    def setup_database_signals(self):
        self.db_worker.operation_started.connect(self.on_operation_started)
        self.db_worker.operation_progress.connect(self.on_operation_progress)
        self.db_worker.operation_completed.connect(self.on_operation_completed)
        self.db_worker.operation_error.connect(self.on_operation_error)
        self.db_worker.bans_loaded.connect(self.on_bans_loaded)
        self.db_worker.logs_loaded.connect(self.on_logs_loaded)
        self.db_worker.statistics_loaded.connect(self.on_statistics_loaded)
        self.db_worker.verification_completed.connect(self.on_verification_completed)
        self.db_worker.connection_tested.connect(self.on_connection_tested)

    def initialize_local_database(self):
        """Initialize SQLite local database with required tables"""
        try:
            if not hasattr(self, 'db') or not self.db:
                QMessageBox.warning(self, "Database Error", "Database manager not available")
                return

            local_path = self.local_path_input.text().strip()
            if not local_path:
                QMessageBox.warning(self, "Path Required", "Please enter a local database path first")
                return

            # Update the database path
            self.db.sqlite_db = local_path

            # Initialize the database
            self.db.init_ban_records_table()

            QMessageBox.information(self, "Success",
                                    f"SQLite local database initialized successfully!\n\nPath: {local_path}\n\n"
                                    "Tables created:\n• ban_records\n• logs")

            # Test the connection
            self.test_local_connection()

        except Exception as e:
            logger.error(f"Error initializing local database: {e}")
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize local database:\n\n{str(e)}")
    def toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        self.sound_toggle_btn.setText("ON" if self.sound_enabled else "OFF")
        logger.info(f"Sound toggled to {'enabled' if self.sound_enabled else 'disabled'}")

    def setup_loading_overlays(self):
        for page_name in ['dashboard', 'bans', 'logs']:
            page_widget = getattr(self, f'{page_name}_page', None)
            if page_widget:
                overlay = LoadingOverlay(page_widget)
                self.loading_overlays[page_name] = overlay
                self._setup_overlay_resize(page_widget, overlay)

    def _setup_overlay_resize(self, page_widget, overlay):
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
        resize_overlay()

    def debug_ban_records_table(self):
        """Debug function to validate ban_records table structure and data"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                print("=== BAN RECORDS TABLE DEBUG ===")

                # Check if table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ban_records'")
                table_exists = cursor.fetchone() is not None
                print(f"Table exists: {table_exists}")

                if not table_exists:
                    print("❌ ban_records table does not exist!")
                    return False

                # Check table structure
                cursor.execute("PRAGMA table_info(ban_records)")
                columns = cursor.fetchall()
                print(f"Table columns ({len(columns)}):")
                for col in columns:
                    print(
                        f"  - {col[1]} ({col[2]}) {'NOT NULL' if col[3] else 'NULL'} {'DEFAULT: ' + str(col[4]) if col[4] else ''}")

                # Check total records
                cursor.execute("SELECT COUNT(*) FROM ban_records")
                total_count = cursor.fetchone()[0]
                print(f"Total records: {total_count}")

                # Check active/inactive counts
                cursor.execute("SELECT COUNT(*) FROM ban_records WHERE is_active = 1")
                active_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM ban_records WHERE is_active = 0")
                inactive_count = cursor.fetchone()[0]
                print(f"Active records: {active_count}")
                print(f"Inactive records: {inactive_count}")

                # Show sample records
                cursor.execute("SELECT id, tanker_number, ban_type, is_active FROM ban_records LIMIT 5")
                sample_records = cursor.fetchall()
                print(f"Sample records:")
                for record in sample_records:
                    print(f"  ID: {record[0]}, Tanker: {record[1]}, Type: {record[2]}, Active: {record[3]}")

                print("=== DEBUG COMPLETE ===")
                return True

        except Exception as e:
            print(f"❌ Debug failed: {e}")
            return False

    def validate_system_setup(self):
        """Validate that all system components are properly set up"""
        try:
            print("=== SYSTEM VALIDATION ===")

            # Check database manager
            if not hasattr(self, 'db') or not self.db:
                print("❌ Database manager not initialized")
                return False
            print("✅ Database manager OK")

            # Check ban records table
            if not self.db.debug_ban_records_table():
                print("❌ Ban records table validation failed")
                return False
            print("✅ Ban records table OK")

            # Check manual verification
            if not self.db.test_manual_verification():
                print("❌ Manual verification test failed")
                return False
            print("✅ Manual verification OK")

            # Check UI components
            required_components = [
                'show_inactive_checkbox', 'bans_table', 'manual_tanker_input',
                'manual_status_label', 'manual_reason_label'
            ]

            for component in required_components:
                if not hasattr(self, component):
                    print(f"❌ Missing UI component: {component}")
                    return False
            print("✅ UI components OK")

            print("🎉 SYSTEM VALIDATION PASSED!")
            return True

        except Exception as e:
            print(f"❌ System validation failed: {e}")
            return False

    def test_ban_management_features(self):
        """Test all ban management features"""
        try:
            reply = QMessageBox.question(self, "Test Ban Management",
                                         "This will create, edit, and delete test records.\n\n"
                                         "Do you want to proceed with the test?",
                                         QMessageBox.Yes | QMessageBox.No)

            if reply != QMessageBox.Yes:
                return

            test_tanker = f"TEST_{int(time.time())}"

            # Test 1: Create ban record
            success = self.db.add_ban_record(
                test_tanker,
                "Test ban for feature testing",
                "temporary",
                datetime.now().strftime("%Y-%m-%d"),
                (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                self.user_info['username']
            )

            if not success:
                QMessageBox.critical(self, "Test Failed", "Failed to create test ban record")
                return

            # Get the created record ID
            with sqlite3.connect(self.db.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM ban_records WHERE tanker_number = ? ORDER BY created_at DESC LIMIT 1",
                               (test_tanker,))
                result = cursor.fetchone()
                if not result:
                    QMessageBox.critical(self, "Test Failed", "Test record not found after creation")
                    return
                test_id = result[0]

            # Test 2: Update ban record
            success = self.db.update_ban_record(
                test_id,
                test_tanker,
                "Updated test ban reason",
                "permission",
                datetime.now().strftime("%Y-%m-%d"),
                (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                self.user_info['username']
            )

            if not success:
                QMessageBox.critical(self, "Test Failed", "Failed to update test ban record")
                return

            # Test 3: Deactivate ban record
            success = self.db.deactivate_ban_record(test_id, self.user_info['username'])
            if not success:
                QMessageBox.critical(self, "Test Failed", "Failed to deactivate test ban record")
                return

            # Test 4: Reactivate ban record
            success = self.db.reactivate_ban_record(test_id, self.user_info['username'])
            if not success:
                QMessageBox.critical(self, "Test Failed", "Failed to reactivate test ban record")
                return

            # Test 5: Manual verification
            status, reason, details = self.db.verify_specific_tanker(test_tanker, self.user_info['username'])
            if status != "ALLOWED_WITH_PERMISSION":
                QMessageBox.warning(self, "Test Warning",
                                    f"Expected 'ALLOWED_WITH_PERMISSION' but got '{status}'")

            # Cleanup: Delete test record
            success = self.db.delete_ban_record(test_id, self.user_info['username'])
            if not success:
                QMessageBox.warning(self, "Cleanup Warning", "Failed to delete test record")

            # Refresh the table
            self.load_bans_table()

            QMessageBox.information(self, "Test Complete",
                                    "✅ All ban management features tested successfully!\n\n"
                                    "Features tested:\n"
                                    "• Create ban record\n"
                                    "• Update ban record\n"
                                    "• Deactivate ban record\n"
                                    "• Reactivate ban record\n"
                                    "• Manual verification\n"
                                    "• Delete ban record")

        except Exception as e:
            logger.error(f"Ban management test failed: {e}")
            QMessageBox.critical(self, "Test Error", f"Test failed with error: {e}")
    def generate_operation_id(self):
        self.operation_counter += 1
        return f"op_{self.operation_counter}_{int(time.time())}"

    def init_performance_optimizations(self):
        """Initialize performance optimizations - FIXED VERSION"""
        try:
            # Create debounce timer
            self.operation_debounce_timer = QTimer()
            self.operation_debounce_timer.setSingleShot(True)
            self.operation_debounce_timer.timeout.connect(self.execute_debounced_operation)

            # Pending operation storage
            self.pending_operation = None

            logger.info("Performance optimizations initialized")

        except Exception as e:
            logger.error(f"Error initializing performance optimizations: {e}")
            # Create fallback
            self.operation_debounce_timer = None

    def execute_debounced_operation(self):
        """Execute the debounced operation - MISSING METHOD"""
        try:
            if hasattr(self, 'pending_operation') and self.pending_operation:
                if self.pending_operation == 'load_bans':
                    self.load_bans_table_safe()
                # Add other operations as needed

        except Exception as e:
            logger.error(f"Error executing debounced operation: {e}")

    def load_bans_table_safe(self):
        """SAFE version of load_bans_table that won't crash"""
        try:
            # Check cache first
            cache_key = 'ban_records'
            cache_time_key = f'{cache_key}_time'
            current_time = time.time()

            # Use cache if available and fresh
            if (hasattr(self, 'data_cache') and
                    self.data_cache and
                    cache_key in self.data_cache and
                    self.data_cache[cache_key] is not None and
                    cache_time_key in self.data_cache.get('cache_time', {}) and
                    current_time - self.data_cache['cache_time'][cache_time_key] < self._max_cache_age):
                logger.info("Using cached ban records data")
                self.populate_bans_table_from_data(self.data_cache[cache_key])
                return

            # Load from database with timeout protection
            if hasattr(self, 'bans_table'):
                self.bans_table.setEnabled(False)

            self.statusBar().showMessage("Loading ban records...")

            try:
                # Quick timeout for database operations
                def load_operation():
                    filters = getattr(self, 'current_ban_filters', None) if getattr(self, 'ban_filters_applied',
                                                                                    False) else None
                    include_inactive = (hasattr(self, 'show_inactive_checkbox') and
                                        self.show_inactive_checkbox.isChecked())
                    return self.db.get_all_bans(filters, True, include_inactive)

                # Execute with shorter timeout to prevent hanging
                bans_data = self._execute_with_timeout_safe(load_operation, timeout=10)

                # Cache the results
                if hasattr(self, 'data_cache') and self.data_cache:
                    self.data_cache[cache_key] = bans_data
                    if 'cache_time' not in self.data_cache:
                        self.data_cache['cache_time'] = {}
                    self.data_cache['cache_time'][cache_time_key] = current_time

                # Populate table
                self.populate_bans_table_from_data(bans_data)
                self.statusBar().showMessage(f"Loaded {len(bans_data)} ban records")

            except Exception as load_error:
                logger.error(f"Ban loading error: {load_error}")
                self.statusBar().showMessage("Failed to load ban records")
                if "timeout" in str(load_error).lower():
                    QMessageBox.warning(self, "Timeout", "Loading timed out. Database may be slow.")

        except Exception as e:
            logger.error(f"Error in load_bans_table_safe: {e}")
        finally:
            if hasattr(self, 'bans_table'):
                self.bans_table.setEnabled(True)

    def _execute_with_timeout_safe(self, operation, timeout=10):
        """Safe timeout execution that won't hang the UI"""
        import threading
        import queue

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def operation_thread():
            try:
                result = operation()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        # Start operation
        thread = threading.Thread(target=operation_thread, daemon=True)
        thread.start()

        # Wait with UI updates
        start_time = time.time()
        while thread.is_alive() and (time.time() - start_time) < timeout:
            QApplication.processEvents()
            time.sleep(0.05)  # Shorter sleep for more responsiveness

        # Check if timed out
        if thread.is_alive():
            logger.warning(f"Operation timed out after {timeout} seconds")
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        # Get results
        if not exception_queue.empty():
            raise exception_queue.get()
        elif not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Operation completed but no result available")

    def clear_cache(self):
        """Clear cached data - MISSING METHOD"""
        try:
            if hasattr(self, 'data_cache'):
                self.data_cache = {
                    'ban_records': None,
                    'logs': None,
                    'statistics': None,
                    'cache_time': {}
                }
            logger.info("Data cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def cleanup_resources(self):
        """Enhanced resource cleanup - FIXED VERSION"""
        try:
            # Clear caches
            self.clear_cache()

            # Stop timers safely
            timers = ['operation_debounce_timer', 'monitor_timer', 'dashboard_timer', 'verification_timeout_timer']
            for timer_name in timers:
                if hasattr(self, timer_name):
                    timer = getattr(self, timer_name)
                    if timer and hasattr(timer, 'isActive') and timer.isActive():
                        timer.stop()

            # Clear button handlers
            if hasattr(self, '_button_handlers'):
                self._button_handlers.clear()

            # Stop audio
            if hasattr(self, 'audio_recorder') and self.audio_recorder:
                try:
                    self.audio_recorder.stop_playback()
                except:
                    pass

            logger.info("Resources cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    def initial_dashboard_load_async(self):
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'initial_dashboard_load'
            if 'dashboard' in self.loading_overlays:
                self.loading_overlays['dashboard'].show_loading("Loading initial dashboard data...")
            self.db_worker.add_operation('load_statistics', operation_id, self.db, self.current_dashboard_filters)
        except Exception as e:
            logger.error(f"Error starting initial dashboard load: {e}")

    def on_operation_started(self, operation_id, description):
        self.statusBar().showMessage(description)

    def cleanup_resources(self):
        """Enhanced resource cleanup"""
        try:
            # Clear caches
            if hasattr(self, 'clear_cache'):
                self.clear_cache()

            # Stop timers
            timers = ['operation_debounce_timer', 'monitor_timer', 'dashboard_timer', 'verification_timeout_timer']
            for timer_name in timers:
                if hasattr(self, timer_name):
                    timer = getattr(self, timer_name)
                    if timer and timer.isActive():
                        timer.stop()

            # Clear button handlers
            if hasattr(self, '_button_handlers'):
                self._button_handlers.clear()

            # Stop audio
            if hasattr(self, 'audio_recorder') and self.audio_recorder:
                self.audio_recorder.stop_playback()

            logger.info("Resources cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def on_operation_progress(self, operation_id, percentage, message):
        operation_type = self.active_operations.get(operation_id)
        if operation_type and operation_type in self.loading_overlays:
            self.loading_overlays[operation_type.split('_')[0]].update_progress(percentage, message)

    def on_operation_completed(self, operation_id, result):
        operation_type = self.active_operations.pop(operation_id, None)
        if operation_type:
            self.statusBar().showMessage(f"{operation_type.replace('_', ' ').title()} completed successfully")

    def on_operation_error(self, operation_id, error_message):
        operation_type = self.active_operations.pop(operation_id, None)
        for overlay in self.loading_overlays.values():
            overlay.hide_loading()
        self.statusBar().showMessage(f"Error: {error_message}")
        error_title = f"Error in {operation_type.replace('_', ' ').title()}" if operation_type else "Operation Error"
        QMessageBox.critical(self, error_title, f"Operation failed:\n\n{error_message}")

    def on_statistics_loaded(self, stats):
        try:
            if 'dashboard' in self.loading_overlays:
                self.loading_overlays['dashboard'].hide_loading()
            self.update_dashboard_with_statistics(stats)
        except Exception as e:
            logger.error(f"Error handling statistics loaded: {e}")

    def on_bans_loaded(self, bans_data):
        try:
            if 'bans' in self.loading_overlays:
                self.loading_overlays['bans'].hide_loading()
            if hasattr(self, 'bans_table'):
                self.bans_table.setEnabled(True)
            self.populate_bans_table_from_data(bans_data)
        except Exception as e:
            logger.error(f"Error handling bans loaded: {e}")

    def on_logs_loaded(self, logs_data):
        try:
            if 'logs' in self.loading_overlays:
                self.loading_overlays['logs'].hide_loading()
            if hasattr(self, 'logs_table'):
                self.logs_table.setEnabled(True)
            self.populate_logs_table_from_data(logs_data)
        except Exception as e:
            logger.error(f"Error handling logs loaded: {e}")


    def test_red_entry_config(self):
        """Test Red Entry configuration from settings page"""
        try:
            if not hasattr(self, 'db') or not self.db:
                QMessageBox.warning(self, "Error", "Database not available")
                return

            # Show loading
            self.statusBar().showMessage("Testing Red Entry configuration...")
            QApplication.processEvents()

            # Test configuration
            success, message = self.db.test_red_entry_configuration()

            if success:
                # Update status label
                if hasattr(self, 'red_entry_status_label'):
                    self.red_entry_status_label.setText(f"✅ {message}")
                    self.red_entry_status_label.setStyleSheet("color: #059669; font-weight: 600;")

                # Show detailed information
                try:
                    # Get additional statistics if available
                    stats_query = f"""
                        SELECT COUNT(*) as total,
                               COUNT(CASE WHEN {self.db.red_entry_date_column} >= DateAdd('d', -1, Now()) THEN 1 END) as recent
                        FROM {self.db.red_entry_table}
                    """

                    with self.db.get_server_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(stats_query)
                        result = cursor.fetchone()
                        total_records = result[0] if result else 0
                        recent_records = result[1] if result else 0

                    details = (f"Red Entry Configuration Test Successful!\n\n"
                               f"✅ Table: {self.db.red_entry_table}\n"
                               f"✅ Registration Column: {self.db.red_entry_reg_column}\n"
                               f"✅ Date Column: {self.db.red_entry_date_column}\n"
                               f"✅ Time Column: {self.db.red_entry_time_column}\n"
                               f"✅ Time Margin: {self.db.time_match_margin} minutes\n\n"
                               f"📊 Total Records: {total_records}\n"
                               f"📊 Recent Records (24h): {recent_records}\n\n"
                               f"Red Entry duplicate detection is working correctly!")

                    QMessageBox.information(self, "Red Entry Test Success", details)

                except Exception as stats_error:
                    logger.warning(f"Could not get detailed statistics: {stats_error}")
                    QMessageBox.information(self, "Red Entry Test Success",
                                            f"{message}\n\nRed Entry duplicate detection is configured correctly!")

                self.statusBar().showMessage("Red Entry test completed successfully")

            else:
                # Update status label for failure
                if hasattr(self, 'red_entry_status_label'):
                    self.red_entry_status_label.setText(f"❌ {message}")
                    self.red_entry_status_label.setStyleSheet("color: #DC2626; font-weight: 600;")

                QMessageBox.warning(self, "Red Entry Test Failed",
                                    f"Red Entry configuration test failed:\n\n{message}\n\n"
                                    f"Please check:\n"
                                    f"• Database connection\n"
                                    f"• Table name: {self.db.red_entry_table}\n"
                                    f"• Column names in config.json\n"
                                    f"• Table permissions")

                self.statusBar().showMessage("Red Entry test failed")

        except Exception as e:
            error_msg = f"Red Entry test error: {str(e)}"
            logger.error(error_msg)

            if hasattr(self, 'red_entry_status_label'):
                self.red_entry_status_label.setText(f"❌ {error_msg}")
                self.red_entry_status_label.setStyleSheet("color: #DC2626; font-weight: 600;")

            QMessageBox.critical(self, "Red Entry Test Error",
                                 f"An error occurred during Red Entry testing:\n\n{error_msg}")

            self.statusBar().showMessage("Red Entry test error")
    def on_connection_tested(self, connection_type, success, message):
        try:
            if connection_type == 'server' and hasattr(self, 'server_status_label'):
                color = ModernUITheme.SUCCESS if success else ModernUITheme.ERROR
                self.server_status_label.setText(f"Server Status: {'✅' if success else '❌'} {message}")
                self.server_status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
            elif connection_type == 'local' and hasattr(self, 'local_status_label'):
                color = ModernUITheme.SUCCESS if success else ModernUITheme.ERROR
                self.local_status_label.setText(f"Local Status: {'✅' if success else '❌'} {message}")
                self.local_status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

            self.statusBar().showMessage(f"{connection_type.title()} connection: {'Success' if success else 'Failed'}")
        except Exception as e:
            logger.error(f"Error handling connection tested: {e}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

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
        self.sidebar.setFixedWidth(260)
        self.sidebar.setMinimumWidth(260)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))
        sidebar_layout.setContentsMargins(20, 30, 20, 30)

        # Title section
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

        # Navigation buttons
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

            btn_container.mousePressEvent = lambda event, cb=callback: cb()
            btn_container.setEnabled(enabled)

            nav_layout.addWidget(btn_container)

        sidebar_layout.addWidget(nav_container)
        sidebar_layout.addStretch()

        # Sound control section
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

        # User info section
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
        # Logout button section
        logout_container = QFrame()
        logout_container.setObjectName("logoutContainer")
        logout_layout = QVBoxLayout(logout_container)
        logout_layout.setContentsMargins(16, 16, 16, 16)
        logout_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        logout_btn = QPushButton("🚪 Logout")
        logout_btn.setObjectName("logoutButton")
        logout_btn.clicked.connect(self.logout_user)
        logout_layout.addWidget(logout_btn)
        logout_btn.setStyleSheet("""
                QPushButton {
                    background-color: #DC2626;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    padding: 12px 16px;
                    min-height: 40px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #B91C1C;
                }
            """)
        logout_layout.addWidget(logout_btn)

        # Add to sidebar (adjust this line based on your sidebar layout structure)
        sidebar_layout.addWidget(logout_container)

    def logout_user(self):
        """Simple logout method that works"""
        try:
            reply = QMessageBox.question(
                self,
                "Confirm Logout",
                f"Logout user '{self.user_info['full_name']}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            logger.info(f"User logout initiated: {self.user_info['username']}")

            # Clean up resources
            self.cleanup_resources()

            # Hide current window
            self.hide()

            # Create new login dialog
            login_dialog = ModernLoginDialog(self.user_manager)

            if login_dialog.exec_() == QDialog.Accepted:
                # User logged in successfully
                new_user_info = login_dialog.user_info
                self.user_info = new_user_info

                # Update window title
                self.setWindowTitle(
                    f"TDF System - Modern UI - User: {new_user_info['full_name']} ({new_user_info['role']})"
                )

                # Clear cache and refresh
                self.clear_cache()

                # Show window again
                self.show()

                logger.info(f"User switched to: {new_user_info['username']}")
                self.statusBar().showMessage(f"Welcome back, {new_user_info['full_name']}")

            else:
                # User cancelled - close application
                logger.info("Logout completed - application closing")
                QApplication.quit()

        except Exception as e:
            logger.error(f"Error during logout: {e}")
            QMessageBox.critical(self, "Logout Error", f"Error during logout: {e}")
            # Show window again in case of error
            self.show()
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

        if self.user_info['role'] in ['admin', 'supervisor']:
            self.settings_page = self.create_modern_settings_page()
            self.stacked_widget.addWidget(self.settings_page)

        self.stacked_widget.addWidget(self.dashboard_page)
        self.stacked_widget.addWidget(self.verification_page)
        self.stacked_widget.addWidget(self.manual_verify_page)
        self.stacked_widget.addWidget(self.bans_page)
        self.stacked_widget.addWidget(self.logs_page)

        self.content_layout.addWidget(self.stacked_widget)

    def _create_stat_card(self, value: str, label: str) -> QWidget:
        card = QFrame()
        card.setObjectName("statCard")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card.setMinimumHeight(120)
        card.setMaximumHeight(160)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignCenter)

        value_label = QLabel(value)
        value_label.setObjectName("valueLabel")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        value_label.setWordWrap(True)
        layout.addWidget(value_label)

        desc_label = QLabel(label)
        desc_label.setObjectName("descLabel")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        return card

    def create_modern_dashboard_page(self):
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Header
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

        # Filters
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

        # Filter buttons
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

        self.dashboard_filter_status = QLabel("📄 Showing all data")
        self.dashboard_filter_status.setObjectName("filterStatus")
        filter_layout.addWidget(self.dashboard_filter_status)

        layout.addWidget(filter_container)

        # Stats section
        stats_section = QVBoxLayout()
        stats_section.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Ban stats
        ban_stats_header = QLabel("⛔ Ban Records Statistics")
        ban_stats_header.setObjectName("statsHeader")
        stats_section.addWidget(ban_stats_header)

        ban_stats_container = QFrame()
        ban_stats_container.setObjectName("statsContainer")
        self.ban_stats_container = QHBoxLayout(ban_stats_container)
        self.ban_stats_container.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        self.label_total_bans = self._create_stat_card("0", "Total Bans")
        self.ban_stats_container.addWidget(self.label_total_bans)

        self.label_active_bans = self._create_stat_card("0", "Active Bans")
        self.ban_stats_container.addWidget(self.label_active_bans)

        stats_section.addWidget(ban_stats_container)

        # Verification stats
        verify_stats_header = QLabel("✅ Verification Statistics")
        verify_stats_header.setObjectName("statsHeader")
        stats_section.addWidget(verify_stats_header)

        verify_stats_container = QFrame()
        verify_stats_container.setObjectName("statsContainer")
        self.verify_stats_container = QHBoxLayout(verify_stats_container)
        self.verify_stats_container.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        self.label_total_verifications = self._create_stat_card("0", "Total Verifications")
        self.verify_stats_container.addWidget(self.label_total_verifications)

        self.label_success_rate = self._create_stat_card("0%", "Success Rate")
        self.verify_stats_container.addWidget(self.label_success_rate)

        stats_section.addWidget(verify_stats_container)
        layout.addLayout(stats_section)

        # Tables section
        tables_section = QVBoxLayout()
        tables_section.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        recent_bans_header = QLabel("📋 Recent Ban Records")
        recent_bans_header.setObjectName("tableHeader")
        tables_section.addWidget(recent_bans_header)

        self.recent_bans_table = QTableWidget()
        self.recent_bans_table.setObjectName("modernTable")
        self.recent_bans_table.setColumnCount(6)
        self.recent_bans_table.setHorizontalHeaderLabels(
            ["Tanker", "Reason", "Type", "Start Date", "End Date", "Created By"])
        self.recent_bans_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_bans_table.setMaximumHeight(250)
        tables_section.addWidget(self.recent_bans_table)

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

        # Result display
        self.auto_result_frame = QFrame()
        self.auto_result_frame.setObjectName("resultContainer")
        result_layout = QVBoxLayout(self.auto_result_frame)
        result_layout.setContentsMargins(32, 32, 32, 32)
        result_layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        self.auto_tanker_info_label = QLabel("🚛 Ready for verification...")
        self.auto_tanker_info_label.setObjectName("tankerInfoLabel")
        result_layout.addWidget(self.auto_tanker_info_label)

        self.auto_status_label = QLabel("System Ready")
        self.auto_status_label.setObjectName("statusDisplayLabel")
        result_layout.addWidget(self.auto_status_label)

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

        # Input section
        input_container = QFrame()
        input_container.setObjectName("inputContainer")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(24, 24, 24, 24)
        input_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

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

        # Quick access buttons
        quick_group = QVBoxLayout()
        quick_label = QLabel("Quick Access")
        quick_label.setObjectName("quickLabel")
        quick_group.addWidget(quick_label)

        quick_buttons_layout = QHBoxLayout()
        quick_buttons_layout.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        quick_buttons = [("Clear", "")]

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

        # Result display
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
        """ENHANCED: Complete ban management with CRUD operations and improved UI"""
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))

        # Enhanced Header with new controls
        header_container = QFrame()
        header_container.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header_container)

        header_info = QVBoxLayout()
        header_title = QLabel("⛔ Ban Management")
        header_title.setObjectName("pageTitle")
        header_info.addWidget(header_title)

        header_subtitle = QLabel("Comprehensive Ban Record Management - Create, Edit, Delete & Control")
        header_subtitle.setObjectName("pageSubtitle")
        header_info.addWidget(header_subtitle)

        header_layout.addLayout(header_info)
        header_layout.addStretch()

        # Enhanced header controls
        header_controls = QHBoxLayout()
        header_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        # Show inactive toggle
        self.show_inactive_checkbox = QCheckBox("Show Inactive Records")
        self.show_inactive_checkbox.setObjectName("showInactiveCheckbox")
        self.show_inactive_checkbox.stateChanged.connect(self.on_show_inactive_changed)
        header_controls.addWidget(self.show_inactive_checkbox)

        add_btn = QPushButton("➕ Add New Ban")
        add_btn.setObjectName("addBanButton")
        add_btn.clicked.connect(self.show_add_ban_dialog)
        header_controls.addWidget(add_btn)

        header_layout.addLayout(header_controls)
        layout.addWidget(header_container)

        # Filter section
        filter_container = QFrame()
        filter_container.setObjectName("filterContainer")
        filter_layout = QVBoxLayout(filter_container)
        filter_layout.setContentsMargins(24, 20, 24, 20)
        filter_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        filter_header = QLabel("🔍 Ban Record Filters")
        filter_header.setObjectName("filterHeader")
        filter_layout.addWidget(filter_header)

        # Filter controls - Date range
        filter_row1 = QHBoxLayout()
        filter_row1.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

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

        # Text filters
        filter_row2 = QHBoxLayout()
        filter_row2.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        tanker_group = QVBoxLayout()
        tanker_label = QLabel("Tanker Number")
        tanker_label.setObjectName("filterLabel")
        tanker_group.addWidget(tanker_label)

        self.ban_tanker_filter = QLineEdit()
        self.ban_tanker_filter.setObjectName("filterInput")
        self.ban_tanker_filter.setPlaceholderText("Filter by tanker number...")
        tanker_group.addWidget(self.ban_tanker_filter)

        filter_row2.addLayout(tanker_group)

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
        self.ban_filter_status = QLabel("📄 Showing active records")
        self.ban_filter_status.setObjectName("filterStatus")
        filter_layout.addWidget(self.ban_filter_status)

        layout.addWidget(filter_container)

        # Ban count display
        self.ban_count_label = QLabel("Loading ban records...")
        self.ban_count_label.setObjectName("countLabel")
        layout.addWidget(self.ban_count_label)

        # ENHANCED: Bans table with new action columns
        self.bans_table = QTableWidget()
        self.bans_table.setObjectName("modernTable")
        self.bans_table.setColumnCount(12)  # Increased for new action columns
        self.bans_table.setHorizontalHeaderLabels([
            "ID", "Tanker", "Reason", "Type", "Start Date", "End Date",
            "Created By", "Status", "Created", "🎧 Audio", "📝 Edit", "🗑️ Actions"
        ])

        # ENHANCED: Header configuration for better layout
        header = self.bans_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Tanker
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Reason
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Start Date
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # End Date
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Created By
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # Created
        header.setSectionResizeMode(9, QHeaderView.Fixed)  # Audio - Fixed width
        header.setSectionResizeMode(10, QHeaderView.Fixed)  # Edit - Fixed width
        header.setSectionResizeMode(11, QHeaderView.Fixed)  # Actions - Fixed width

        # Set fixed widths for action columns for better layout
        header.resizeSection(9, 100)  # Audio column
        header.resizeSection(10, 80)  # Edit column
        header.resizeSection(11, 120)  # Actions column

        # Set minimum row height for better button layout
        self.bans_table.setMinimumHeight(400)
        self.bans_table.verticalHeader().setDefaultSectionSize(45)

        layout.addWidget(self.bans_table)
        return page

    def init_performance_optimizations(self):
        """Initialize performance optimizations"""
        try:
            # Debounce timer for operations
            self.operation_debounce_timer = QTimer()
            self.operation_debounce_timer.setSingleShot(True)
            self.operation_debounce_timer.timeout.connect(self.execute_debounced_operation)

            # Data cache
            self.data_cache = {
                'ban_records': None,
                'logs': None,
                'statistics': None,
                'cache_time': {}
            }

            logger.info("Performance optimizations initialized")

        except Exception as e:
            logger.error(f"Error initializing performance optimizations: {e}")

    def _execute_with_timeout_ui(self, operation, timeout=15):
        """Execute operation with timeout and UI responsiveness"""
        import threading
        import queue

        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def operation_thread():
            try:
                result = operation()
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=operation_thread, daemon=True)
        thread.start()

        start_time = time.time()
        while thread.is_alive() and (time.time() - start_time) < timeout:
            QApplication.processEvents()
            time.sleep(0.1)

        if thread.is_alive():
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        if not exception_queue.empty():
            raise exception_queue.get()
        elif not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Operation completed but no result available")

    def clear_cache(self):
        """Clear cached data"""
        try:
            self.data_cache = {
                'ban_records': None,
                'logs': None,
                'statistics': None,
                'cache_time': {}
            }
            logger.info("Data cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    def create_modern_logs_page(self):
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

        # Filter section
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
        self.logs_table.setHorizontalHeaderLabels(["ID", "Tanker", "Status", "Reason", "Timestamp", "Operator"])
        self.logs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.logs_table)

        return page

    def create_modern_settings_page(self):
        page = QWidget()
        page.setObjectName("modernPage")
        layout = QVBoxLayout(page)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        # Create tab widget
        tab_widget = QTabWidget()
        tab_widget.setObjectName("modernTabWidget")
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

        # Create tab widget
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
        # Maintenance Tab
        maintenance_tab = self.create_maintenance_tab()
        tab_widget.addTab(maintenance_tab, "🔧 Maintenance")
        layout.addWidget(tab_widget)
        return page

    def create_maintenance_tab(self):
        """Create maintenance tab with backup and cleanup features"""
        tab = QWidget()
        tab.setObjectName("modernTabPage")
        layout = QVBoxLayout(tab)
        layout.setSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
        layout.setContentsMargins(24, 24, 24, 24)

        # Backup Section
        backup_container = QFrame()
        backup_container.setObjectName("settingsContainer")
        backup_layout = QVBoxLayout(backup_container)
        backup_layout.setContentsMargins(24, 20, 24, 24)
        backup_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        backup_header = QLabel("💾 Database Backup")
        backup_header.setObjectName("settingsHeader")
        backup_layout.addWidget(backup_header)

        backup_desc = QLabel("Create a complete backup of all ban records, logs, and configuration data")
        backup_desc.setObjectName("settingsSubtext")
        backup_layout.addWidget(backup_desc)

        # Backup path selection
        backup_path_layout = QHBoxLayout()
        self.backup_path_input = QLineEdit()
        self.backup_path_input.setObjectName("pathInput")
        self.backup_path_input.setText(os.path.join(os.path.expanduser("~"), "Desktop", "TDF_Backup"))

        browse_backup_btn = QPushButton("📁 Browse")
        browse_backup_btn.setObjectName("browseButton")
        browse_backup_btn.clicked.connect(self.browse_backup_location)

        backup_path_layout.addWidget(self.backup_path_input)
        backup_path_layout.addWidget(browse_backup_btn)
        backup_layout.addLayout(backup_path_layout)

        # Backup options
        self.include_voice_checkbox = QCheckBox("Include voice recordings")
        self.include_voice_checkbox.setChecked(True)
        self.compress_backup_checkbox = QCheckBox("Compress backup (ZIP)")
        self.compress_backup_checkbox.setChecked(True)

        backup_layout.addWidget(self.include_voice_checkbox)
        backup_layout.addWidget(self.compress_backup_checkbox)

        # Backup button
        backup_btn = QPushButton("💾 Create Backup")
        backup_btn.setObjectName("saveButton")
        backup_btn.clicked.connect(self.create_backup)
        backup_layout.addWidget(backup_btn)

        layout.addWidget(backup_container)

        # Cleanup Section
        cleanup_container = QFrame()
        cleanup_container.setObjectName("settingsContainer")
        cleanup_layout = QVBoxLayout(cleanup_container)
        cleanup_layout.setContentsMargins(24, 20, 24, 24)
        cleanup_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

        cleanup_header = QLabel("🗑️ Database Cleanup")
        cleanup_header.setObjectName("settingsHeader")
        cleanup_layout.addWidget(cleanup_header)

        # Cleanup options
        cleanup_options_layout = QFormLayout()

        self.cleanup_logs_days = QSpinBox()
        self.cleanup_logs_days.setObjectName("modernSpinBox")
        self.cleanup_logs_days.setRange(1, 365)
        self.cleanup_logs_days.setValue(90)
        self.cleanup_logs_days.setSuffix(" days")

        self.cleanup_inactive_bans = QCheckBox("Delete inactive ban records older than:")
        self.cleanup_inactive_days = QSpinBox()
        self.cleanup_inactive_days.setObjectName("modernSpinBox")
        self.cleanup_inactive_days.setRange(1, 365)
        self.cleanup_inactive_days.setValue(180)
        self.cleanup_inactive_days.setSuffix(" days")

        cleanup_options_layout.addRow("Delete logs older than:", self.cleanup_logs_days)
        cleanup_options_layout.addRow(self.cleanup_inactive_bans, self.cleanup_inactive_days)
        cleanup_layout.addLayout(cleanup_options_layout)

        # Cleanup buttons
        cleanup_actions = QHBoxLayout()

        preview_cleanup_btn = QPushButton("👁️ Preview Cleanup")
        preview_cleanup_btn.setObjectName("testButton")
        preview_cleanup_btn.clicked.connect(self.preview_cleanup)

        execute_cleanup_btn = QPushButton("🗑️ Execute Cleanup")
        execute_cleanup_btn.setObjectName("clearFilterButton")
        execute_cleanup_btn.clicked.connect(self.execute_cleanup)

        cleanup_actions.addWidget(preview_cleanup_btn)
        cleanup_actions.addWidget(execute_cleanup_btn)
        cleanup_layout.addLayout(cleanup_actions)

        layout.addWidget(cleanup_container)
        layout.addStretch()
        return tab

    def browse_backup_location(self):
        """Browse for backup location"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Backup Location", os.path.dirname(self.backup_path_input.text())
        )
        if folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(folder, f"TDF_Backup_{timestamp}")
            self.backup_path_input.setText(backup_path)

    def create_backup(self):
        """Create complete database backup"""
        try:
            backup_path = self.backup_path_input.text().strip()
            if not backup_path:
                QMessageBox.warning(self, "Invalid Path", "Please specify a backup location")
                return

            # Show progress
            progress = QProgressDialog("Creating backup...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            progress.setValue(20)

            # Backup SQLite database
            sqlite_backup = os.path.join(backup_path, "database_backup.db")
            shutil.copy2(self.db.sqlite_db, sqlite_backup)
            progress.setValue(60)

            # Export ban records
            ban_records = self.db.get_all_bans(None, exclude_blob=not self.include_voice_checkbox.isChecked(),
                                               include_inactive=True)

            ban_data = []
            for record in ban_records:
                record_dict = {}
                for i, value in enumerate(record):
                    if i == 8 and isinstance(value, bytes):  # Voice data
                        record_dict[f'field_{i}'] = "BINARY_DATA" if value else None
                    else:
                        record_dict[f'field_{i}'] = value
                ban_data.append(record_dict)

            ban_backup = os.path.join(backup_path, "ban_records.json")
            with open(ban_backup, 'w', encoding='utf-8') as f:
                json.dump(ban_data, f, indent=2, ensure_ascii=False, default=str)
            progress.setValue(80)

            # Compress if requested
            final_path = backup_path
            if self.compress_backup_checkbox.isChecked():
                import zipfile
                zip_path = f"{backup_path}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(backup_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, backup_path)
                            zipf.write(file_path, arcname)

                shutil.rmtree(backup_path)
                final_path = zip_path

            progress.setValue(100)
            progress.close()

            QMessageBox.information(self, "Backup Complete", f"Backup created: {final_path}")
            logger.info(f"Backup created: {final_path}")

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            QMessageBox.critical(self, "Backup Failed", f"Backup creation failed: {e}")

    def preview_cleanup(self):
        """Preview what will be cleaned up"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.cleanup_logs_days.value())

            with sqlite3.connect(self.db.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT COUNT(*) FROM logs WHERE timestamp < ?",
                    (cutoff_date.strftime("%Y-%m-%d %H:%M:%S"),)
                )
                logs_count = cursor.fetchone()[0]

                inactive_count = 0
                if self.cleanup_inactive_bans.isChecked():
                    cutoff_date_bans = datetime.now() - timedelta(days=self.cleanup_inactive_days.value())
                    cursor.execute(
                        "SELECT COUNT(*) FROM ban_records WHERE is_active = 0 AND created_at < ?",
                        (cutoff_date_bans.strftime("%Y-%m-%d %H:%M:%S"),)
                    )
                    inactive_count = cursor.fetchone()[0]

            message = f"Cleanup Preview:\n\n"
            message += f"• {logs_count} log entries\n"
            if self.cleanup_inactive_bans.isChecked():
                message += f"• {inactive_count} inactive ban records\n"
            message += f"\nTotal: {logs_count + inactive_count} records"

            QMessageBox.information(self, "Cleanup Preview", message)

        except Exception as e:
            QMessageBox.critical(self, "Preview Failed", f"Failed to preview: {e}")

    def execute_cleanup(self):
        """Execute database cleanup"""
        try:
            reply = QMessageBox.question(
                self, "Confirm Cleanup",
                "This will permanently delete old records!\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            cutoff_date = datetime.now() - timedelta(days=self.cleanup_logs_days.value())

            with sqlite3.connect(self.db.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()

                # Delete old logs
                cursor.execute(
                    "DELETE FROM logs WHERE timestamp < ?",
                    (cutoff_date.strftime("%Y-%m-%d %H:%M:%S"),)
                )
                logs_deleted = cursor.rowcount

                # Delete old inactive bans
                bans_deleted = 0
                if self.cleanup_inactive_bans.isChecked():
                    cutoff_date_bans = datetime.now() - timedelta(days=self.cleanup_inactive_days.value())
                    cursor.execute(
                        "DELETE FROM ban_records WHERE is_active = 0 AND created_at < ?",
                        (cutoff_date_bans.strftime("%Y-%m-%d %H:%M:%S"),)
                    )
                    bans_deleted = cursor.rowcount

                conn.commit()

            self.clear_cache()
            self.refresh_dashboard()

            QMessageBox.information(
                self, "Cleanup Complete",
                f"Deleted:\n• {logs_deleted} logs\n• {bans_deleted} inactive bans"
            )

        except Exception as e:
            QMessageBox.critical(self, "Cleanup Failed", f"Cleanup failed: {e}")
    def create_modern_database_tab(self):
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

        # Server database path - Access format
        server_group = QVBoxLayout()
        server_label = QLabel("Server Database (.mdb/.accdb)")  # Access format
        server_label.setObjectName("settingsLabel")
        server_group.addWidget(server_label)

        server_description = QLabel("Microsoft Access database containing vehicle master data and transactions")
        server_description.setObjectName("settingsSubtext")
        server_group.addWidget(server_description)

        server_controls = QHBoxLayout()
        server_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.server_path_input = QLineEdit()
        self.server_path_input.setObjectName("pathInput")
        self.server_path_input.setText(self.config.get('server_db_path', ''))
        self.server_path_input.setPlaceholderText("Path to Access database (.mdb/.accdb file)")
        server_controls.addWidget(self.server_path_input)

        server_browse_btn = QPushButton("📁 Browse")
        server_browse_btn.setObjectName("browseButton")
        server_browse_btn.clicked.connect(self.browse_server_db)
        server_controls.addWidget(server_browse_btn)

        server_group.addLayout(server_controls)
        db_layout.addLayout(server_group)

        # Local database path - SQLite format
        local_group = QVBoxLayout()
        local_label = QLabel("Local Database (.db)")  # SQLite format
        local_label.setObjectName("settingsLabel")
        local_group.addWidget(local_label)

        local_description = QLabel("SQLite database for ban records, logs, and local data storage")
        local_description.setObjectName("settingsSubtext")
        local_group.addWidget(local_description)

        local_controls = QHBoxLayout()
        local_controls.setSpacing(int(ModernUITheme.SPACE_MD.replace('px', '')))

        self.local_path_input = QLineEdit()
        self.local_path_input.setObjectName("pathInput")
        self.local_path_input.setText(self.config.get('local_sqlite_path', ''))
        self.local_path_input.setPlaceholderText("Path to SQLite database (.db file)")
        local_controls.addWidget(self.local_path_input)

        local_browse_btn = QPushButton("📁 Browse")
        local_browse_btn.setObjectName("browseButton")
        local_browse_btn.clicked.connect(self.browse_local_db)
        local_controls.addWidget(local_browse_btn)

        # Add Initialize Local DB button
        init_local_btn = QPushButton("🔧 Initialize Local DB")
        init_local_btn.setObjectName("testButton")
        init_local_btn.clicked.connect(self.initialize_local_database)
        local_controls.addWidget(init_local_btn)

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

        test_server_btn = QPushButton("🔍 Test Access Server")
        test_server_btn.setObjectName("testButton")
        test_server_btn.clicked.connect(self.test_server_connection)
        test_actions.addWidget(test_server_btn)

        test_local_btn = QPushButton("🔍 Test SQLite Local")
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

        save_btn = QPushButton("💾 Save Database Settings")
        save_btn.setObjectName("saveButton")
        save_btn.clicked.connect(self.save_database_settings)
        save_actions.addWidget(save_btn)

        layout.addLayout(save_actions)
        layout.addStretch()

        QTimer.singleShot(500, self.test_all_connections)
        return tab

    def create_modern_user_tab(self):
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

        # Users table
        self.users_table = QTableWidget()
        self.users_table.setObjectName("modernTable")
        self.users_table.setColumnCount(8)
        self.users_table.setHorizontalHeaderLabels(
            ["ID", "Username", "Full Name", "Role", "Status", "Created", "Last Login", "Actions"])

        header = self.users_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        layout.addWidget(self.users_table)
        self.load_users_table()
        return tab

    def create_modern_system_tab(self):
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

        # Auto refresh interval
        refresh_label = QLabel("Auto Refresh Interval")
        refresh_label.setObjectName("settingsLabel")
        self.auto_refresh_spin = QSpinBox()
        self.auto_refresh_spin.setObjectName("modernSpinBox")
        self.auto_refresh_spin.setRange(10, 300)
        self.auto_refresh_spin.setValue(self.config.get('auto_refresh_interval', 60))
        self.auto_refresh_spin.setSuffix(" seconds")
        settings_grid.addWidget(refresh_label, 1, 0)
        settings_grid.addWidget(self.auto_refresh_spin, 1, 1)

        system_layout.addLayout(settings_grid)
        layout.addWidget(system_container)

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


    def verify_manual_tanker(self):
        """FIXED: Manual verification without hanging - Red Entry enabled"""
        try:
            tanker_number = self.manual_tanker_input.text().strip().upper()

            if not tanker_number:
                QMessageBox.warning(self, "Input Required", "Please enter a tanker number")
                self.manual_tanker_input.setFocus()
                return

            # Immediate UI feedback
            self.manual_status_label.setText("⏳ Verifying...")
            self.manual_reason_label.setText("Checking ban records and Red Entry duplicates...")
            self.manual_tanker_info_label.setText(f"🚛 Vehicle: {tanker_number}")
            QApplication.processEvents()

            # Generate operation ID and start verification
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'verify_manual'

            logger.info(f"Starting manual verification for: {tanker_number}")

            # Add to database worker queue - NO HANGING
            self.db_worker.add_operation('verify_tanker', operation_id, self.db, tanker_number,
                                         self.user_info['username'])

        except Exception as e:
            logger.error(f"Manual verification error: {e}")
            self.manual_status_label.setText("❌ ERROR")
            self.manual_reason_label.setText(f"Verification failed: {str(e)}")

    def add_debug_menu(self):
        """Add debug menu for testing (call this in __init__)"""
        try:
            # Create debug menu
            menubar = self.menuBar()
            debug_menu = menubar.addMenu('🔧 Debug')

            # Validate system action
            validate_action = QAction('✅ Validate System', self)
            validate_action.triggered.connect(self.validate_system_setup)
            debug_menu.addAction(validate_action)

            # Test ban management action
            test_ban_action = QAction('🧪 Test Ban Management', self)
            test_ban_action.triggered.connect(self.test_ban_management_features)
            debug_menu.addAction(test_ban_action)

            # Debug database action
            debug_db_action = QAction('🗄️ Debug Database', self)
            debug_db_action.triggered.connect(lambda: self.db.debug_ban_records_table())
            debug_menu.addAction(debug_db_action)

            # Manual verification test
            test_verify_action = QAction('🔍 Test Manual Verification', self)
            test_verify_action.triggered.connect(lambda: self.db.test_manual_verification())
            debug_menu.addAction(test_verify_action)

            debug_menu.addSeparator()

            # Quick add test data
            add_test_data_action = QAction('📊 Add Test Data', self)
            add_test_data_action.triggered.connect(self.add_test_ban_data)
            debug_menu.addAction(add_test_data_action)

            logger.info("Debug menu added successfully")

        except Exception as e:
            logger.error(f"Error adding debug menu: {e}")

    # Add these methods to your MainWindow class in main.py




    def add_test_ban_data(self):
        """Add test ban data for testing purposes"""
        try:
            reply = QMessageBox.question(self, "Add Test Data",
                                         "This will add several test ban records.\n\n"
                                         "Do you want to proceed?",
                                         QMessageBox.Yes | QMessageBox.No)

            if reply != QMessageBox.Yes:
                return

            test_data = [
                ("TEST001", "Test permanent ban", "permanent", None),
                ("TEST002", "Test temporary ban", "temporary", 30),
                ("TEST003", "Test permission required", "permission", 60),
                ("TEST004", "Test reminder notice", "reminder", 7),
                ("TRUCK123", "Safety violation - expired license", "temporary", 14),
                ("VEH456", "Maintenance overdue", "permission", None),
            ]

            success_count = 0
            for tanker, reason, ban_type, days_duration in test_data:
                start_date = datetime.now().strftime("%Y-%m-%d")
                end_date = None
                if days_duration:
                    end_date = (datetime.now() + timedelta(days=days_duration)).strftime("%Y-%m-%d")

                success = self.db.add_ban_record(
                    tanker, reason, ban_type, start_date, end_date,
                    f"{self.user_info['username']}_TEST"
                )
                if success:
                    success_count += 1

            self.load_bans_table()
            QMessageBox.information(self, "Test Data Added",
                                    f"✅ Successfully added {success_count}/{len(test_data)} test records.\n\n"
                                    "You can now test:\n"
                                    "• Manual verification with TEST001, TEST002, etc.\n"
                                    "• Different ban types and their effects\n"
                                    "• Edit, deactivate, and delete functions")

        except Exception as e:
            logger.error(f"Error adding test data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add test data: {e}")




    def play_auto_verification_voice(self):
        if hasattr(self, 'auto_current_voice_data') and self.auto_current_voice_data and self.audio_recorder:
            try:
                self.auto_voice_info_label.setText("🔊 Playing voice note...")
                self.audio_recorder.play_audio(self.auto_current_voice_data, self.on_auto_voice_finished)
            except Exception as e:
                logger.error(f"Error playing auto verification voice: {e}")

    def play_manual_verification_voice(self):
        if hasattr(self, 'manual_current_voice_data') and self.manual_current_voice_data and self.audio_recorder:
            try:
                self.manual_voice_info_label.setText("🔊 Playing voice note...")
                self.audio_recorder.play_audio(self.manual_current_voice_data, self.on_manual_voice_finished)
            except Exception as e:
                logger.error(f"Error playing manual verification voice: {e}")

    def on_auto_voice_finished(self, success, message):
        if success:
            self.auto_voice_info_label.setText("🎵 Voice note available - click to play")
        else:
            self.auto_voice_info_label.setText(f"❌ Playback failed: {message}")

    def on_manual_voice_finished(self, success, message):
        if success:
            self.manual_voice_info_label.setText("🎵 Voice note available - click to play")
        else:
            self.manual_voice_info_label.setText(f"❌ Playback failed: {message}")

    def stop_warning_sound(self):
        if self.warning_sound:
            self.warning_sound.stop()
            self.current_sound_playing = False
            self.statusBar().showMessage("Warning sound stopped")

    def on_warning_sound_finished(self, success, message):
        self.current_sound_playing = False

    def show_add_ban_dialog(self):
        """Completely corrected add ban dialog with proper voice recording"""
        try:
            logger.info("Opening add ban dialog...")

            # Validate database worker
            if not hasattr(self, 'db_worker') or not self.db_worker or not self.db_worker.isRunning():
                QMessageBox.critical(self, "System Error", "Database worker not available")
                return

            dialog = QDialog(self)
            dialog.setWindowTitle("Add New Ban Record")
            dialog.setFixedSize(700, 800)

            layout = QVBoxLayout(dialog)
            layout.setSpacing(int(ModernUITheme.SPACE_3XL.replace('px', '')))
            layout.setContentsMargins(40, 30, 40, 30)

            # Header
            header_section = QWidget()
            header_layout = QHBoxLayout(header_section)
            header_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

            icon_label = QLabel("🚫")
            icon_label.setFixedSize(48, 48)
            icon_label.setAlignment(Qt.AlignCenter)

            title_container = QWidget()
            title_layout = QVBoxLayout(title_container)
            title_layout.setContentsMargins(0, 0, 0, 0)
            title_layout.setSpacing(4)

            title_label = QLabel("Add New Ban Record")
            title_label.setStyleSheet(
                f"font-size: {ModernUITheme.FONT_SIZE_2XL}; font-weight: 700; color: {ModernUITheme.TEXT_PRIMARY};")

            subtitle_label = QLabel("Fill in the details to create a new ban record")
            subtitle_label.setStyleSheet(f"font-size: {ModernUITheme.FONT_SIZE_SM}; color: {ModernUITheme.TEXT_MUTED};")

            title_layout.addWidget(title_label)
            title_layout.addWidget(subtitle_label)

            header_layout.addWidget(icon_label)
            header_layout.addWidget(title_container)
            header_layout.addStretch()

            layout.addWidget(header_section)

            # Form section
            form_container = QFrame()
            form_container.setStyleSheet(
                f"background-color: {ModernUITheme.SURFACE}; border-radius: {ModernUITheme.RADIUS_LG}; border: 1px solid {ModernUITheme.BORDER_LIGHT};")

            form_layout = QFormLayout(form_container)
            form_layout.setVerticalSpacing(int(ModernUITheme.SPACE_2XL.replace('px', '')))
            form_layout.setContentsMargins(32, 32, 32, 32)

            tanker_input = QLineEdit()
            tanker_input.setPlaceholderText("Enter tanker number (e.g., TR001)")
            tanker_input.setStyleSheet(
                f"border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; min-height: 20px;")

            reason_input = QTextEdit()
            reason_input.setPlaceholderText("Enter detailed ban reason...")
            reason_input.setMaximumHeight(120)
            reason_input.setStyleSheet(
                f"border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; min-height: 20px;")

            type_combo = QComboBox()
            type_combo.addItems(["temporary", "permanent", "permission", "reminder"])
            type_combo.setStyleSheet(
                f"border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; min-height: 20px;")

            start_date = QDateEdit()
            start_date.setDate(QDate.currentDate())
            start_date.setCalendarPopup(True)
            start_date.setStyleSheet(
                f"border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; min-height: 20px;")

            end_date = QDateEdit()
            end_date.setDate(QDate.currentDate().addDays(30))
            end_date.setCalendarPopup(True)
            end_date.setStyleSheet(
                f"border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; min-height: 20px;")

            form_layout.addRow("🚛 Tanker Number:", tanker_input)
            form_layout.addRow("📝 Ban Reason:", reason_input)
            form_layout.addRow("🏷️ Ban Type:", type_combo)
            form_layout.addRow("📅 Start Date:", start_date)
            form_layout.addRow("📅 End Date:", end_date)

            layout.addWidget(form_container)

            # Audio recording section
            audio_container = QFrame()
            audio_container.setStyleSheet(
                f"background-color: rgba(5, 150, 105, 0.05); border: 2px solid rgba(5, 150, 105, 0.15); border-radius: {ModernUITheme.RADIUS_LG};")

            audio_layout = QVBoxLayout(audio_container)
            audio_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))
            audio_layout.setContentsMargins(24, 20, 24, 20)

            audio_header = QLabel("🎙️ Voice Note (Optional)")
            audio_header.setStyleSheet(
                f"font-size: {ModernUITheme.FONT_SIZE_LG}; font-weight: 600; color: {ModernUITheme.TEXT_PRIMARY};")
            audio_layout.addWidget(audio_header)

            voice_controls = QHBoxLayout()
            voice_controls.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

            voice_status_label = QLabel("⚪ No recording")
            voice_status_label.setStyleSheet(
                f"font-size: {ModernUITheme.FONT_SIZE_SM}; color: {ModernUITheme.TEXT_MUTED}; font-weight: 500;")

            record_voice_btn = QPushButton("🎙️ Start Recording")
            record_voice_btn.setStyleSheet(
                f"background-color: {ModernUITheme.SUCCESS}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 600; padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_LG}; min-height: 40px;")

            # Check audio availability
            if not AUDIO_AVAILABLE or not hasattr(self, 'audio_recorder') or not self.audio_recorder:
                record_voice_btn.setEnabled(False)
                record_voice_btn.setText("🎙️ Audio Not Available")
                voice_status_label.setText("❌ Audio recording not available")

            voice_controls.addWidget(voice_status_label)
            voice_controls.addStretch()
            voice_controls.addWidget(record_voice_btn)

            audio_layout.addLayout(voice_controls)
            layout.addWidget(audio_container)

            # Buttons
            button_container = QWidget()
            button_layout = QHBoxLayout(button_container)
            button_layout.setSpacing(int(ModernUITheme.SPACE_LG.replace('px', '')))

            info_label = QLabel("* Required fields")
            info_label.setStyleSheet(
                f"font-size: {ModernUITheme.FONT_SIZE_XS}; color: {ModernUITheme.TEXT_MUTED}; font-style: italic;")

            button_layout.addWidget(info_label)
            button_layout.addStretch()

            cancel_btn = QPushButton("Cancel")
            cancel_btn.setStyleSheet(
                f"background-color: transparent; color: {ModernUITheme.TEXT_SECONDARY}; border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 600; padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_LG}; min-height: 44px;")
            cancel_btn.clicked.connect(dialog.reject)

            save_btn = QPushButton("💾 Save Ban Record")
            save_btn.setStyleSheet(
                f"background-color: {ModernUITheme.PRIMARY}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 600; padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_2XL}; min-height: 44px;")

            button_layout.addWidget(cancel_btn)
            button_layout.addWidget(save_btn)

            layout.addWidget(button_container)

            # Store recorded data
            recorded_data = None

            def record_voice():
                """FIXED voice recording function with proper error handling"""
                nonlocal recorded_data

                try:
                    if not AUDIO_AVAILABLE:
                        QMessageBox.warning(dialog, "Audio Error",
                                            "Audio recording is not available on this system.\n\n"
                                            "Please ensure your system has:\n"
                                            "• A working microphone\n"
                                            "• Required audio libraries (PyAudio, sounddevice)")
                        return

                    if not hasattr(self, 'audio_recorder') or not self.audio_recorder:
                        QMessageBox.warning(dialog, "Audio Error",
                                            "Audio recorder is not available.\n\n"
                                            "The audio system may not be properly initialized.")
                        return

                    # Create voice recording dialog with error handling
                    try:
                        voice_dialog = VoiceRecordingDialog(dialog)

                        # Verify dialog was created successfully
                        if not voice_dialog:
                            raise Exception("Dialog creation returned None")

                        if not hasattr(voice_dialog, 'audio_recorder'):
                            raise Exception("Dialog audio recorder not initialized")

                    except Exception as dialog_error:
                        logger.error(f"Failed to create voice recording dialog: {dialog_error}")
                        QMessageBox.critical(dialog, "Dialog Error",
                                             f"Cannot create voice recording dialog:\n\n{str(dialog_error)}\n\n"
                                             "Please check that your audio system is working properly.")
                        return

                    # Show dialog and handle result
                    dialog_result = voice_dialog.exec_()

                    if dialog_result == QDialog.Accepted:
                        recorded_data = voice_dialog.get_recorded_data()
                        if recorded_data and len(recorded_data) > 100:  # Enhanced validation
                            # Additional validation for WAV format
                            if recorded_data.startswith(b'RIFF') and b'WAVE' in recorded_data[:20]:
                                voice_status_label.setText("🟢 Voice recorded successfully")
                                record_voice_btn.setText("🎙️ Re-record Voice Note")
                                logger.info(
                                    f"Voice recorded successfully: {len(recorded_data)} bytes, valid WAV format")
                            else:
                                voice_status_label.setText("🟡 Voice recorded (format corrected)")
                                record_voice_btn.setText("🎙️ Re-record Voice Note")
                                logger.info(
                                    f"Voice recorded: {len(recorded_data)} bytes, format will be corrected during save")
                        else:
                            voice_status_label.setText("❌ Recording failed - data too small")
                            recorded_data = None
                            logger.warning("Voice recording failed - insufficient data")

                            # Show helpful message
                            QMessageBox.information(dialog, "Recording Issue",
                                                    "Recording failed - the audio data is too small.\n\n"
                                                    "Tips for better recording:\n"
                                                    "• Speak closer to the microphone\n"
                                                    "• Record for at least 2-3 seconds\n"
                                                    "• Check microphone permissions\n"
                                                    "• Ensure microphone is not muted")
                    else:
                        logger.info("Voice recording cancelled by user")

                except Exception as e:
                    logger.error(f"Error during voice recording: {e}")
                    voice_status_label.setText("❌ Recording error occurred")
                    recorded_data = None
                    QMessageBox.critical(dialog, "Recording Error",
                                         f"An error occurred during recording:\n\n{str(e)}")

            def save_ban():
                """Enhanced save ban function with better validation"""
                try:
                    # Validate inputs
                    tanker = tanker_input.text().strip().upper()
                    reason = reason_input.toPlainText().strip()

                    if not tanker or not reason:
                        QMessageBox.warning(dialog, "Input Error",
                                            "Please fill all required fields:\n\n"
                                            "• Tanker Number\n"
                                            "• Ban Reason")
                        return

                    # Validate tanker number format
                    if len(tanker) < 2:
                        QMessageBox.warning(dialog, "Validation Error",
                                            "Tanker number must be at least 2 characters long.")
                        tanker_input.setFocus()
                        return

                    # Prepare ban data
                    ban_type = type_combo.currentText()
                    start_str = start_date.date().toString("yyyy-MM-dd")
                    end_str = end_date.date().toString("yyyy-MM-dd") if ban_type != "permanent" else None
                    filename = f"voice_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav" if recorded_data else None

                    # Show progress
                    save_btn.setEnabled(False)
                    save_btn.setText("💾 Saving...")
                    QApplication.processEvents()

                    # Validate database connection
                    if not hasattr(self, 'db') or not self.db:
                        QMessageBox.critical(dialog, "Database Error", "Database connection not available")
                        return

                    # Save ban record
                    success = self.db.add_ban_record(
                        tanker, reason, ban_type, start_str, end_str,
                        self.user_info['username'],
                        voice_data=recorded_data,
                        voice_filename=filename
                    )

                    if success:
                        voice_info = "with voice note" if recorded_data else "without voice note"
                        QMessageBox.information(dialog, "Success",
                                                f"Ban record created successfully for {tanker} {voice_info}")
                        dialog.accept()

                        # Refresh related views
                        if hasattr(self, 'load_bans_table'):
                            QTimer.singleShot(100, self.load_bans_table)
                        if hasattr(self, 'refresh_dashboard'):
                            QTimer.singleShot(200, self.refresh_dashboard)

                        self.statusBar().showMessage(f"Ban record added for {tanker}")
                        logger.info(f"Ban record created for {tanker} with voice: {'Yes' if recorded_data else 'No'}")
                    else:
                        QMessageBox.critical(dialog, "Error",
                                             "Failed to create ban record. Please try again.")
                        logger.error(f"Failed to create ban record for {tanker}")

                except Exception as e:
                    logger.error(f"Error saving ban record: {e}")
                    QMessageBox.critical(dialog, "Error", f"Failed to save ban record:\n\n{str(e)}")
                finally:
                    # Always re-enable the save button
                    save_btn.setEnabled(True)
                    save_btn.setText("💾 Save Ban Record")

            # Connect button events
            record_voice_btn.clicked.connect(record_voice)
            save_btn.clicked.connect(save_ban)

            # Show the dialog
            dialog.exec_()

        except Exception as e:
            logger.error(f"Error creating add ban dialog: {e}")
            QMessageBox.critical(self, "Dialog Error", f"Failed to open add ban dialog:\n\n{str(e)}")

    # Additional helper method to add to your MainWindow class
    def create_voice_dialog_safely(self, parent=None):
        """Safely create a VoiceRecordingDialog with error handling"""
        try:
            # Check prerequisites
            if not AUDIO_AVAILABLE:
                logger.warning("Cannot create voice dialog - audio not available")
                return None

            if not hasattr(self, 'audio_recorder') or not self.audio_recorder:
                logger.warning("Cannot create voice dialog - audio recorder not available")
                return None

            # Create dialog
            dialog = VoiceRecordingDialog(parent)

            # Verify dialog was created properly
            if not hasattr(dialog, 'audio_recorder'):
                logger.error("Voice dialog created but audio_recorder not initialized")
                return None

            logger.info("Voice recording dialog created successfully")
            return dialog

        except Exception as e:
            logger.error(f"Failed to create voice recording dialog: {e}")
            return None
    def apply_dashboard_filter(self):
        try:
            self.dashboard_filters_applied = True
            self.current_dashboard_filters = {
                'start_date': self.dashboard_start_date.date().toString("yyyy-MM-dd"),
                'end_date': self.dashboard_end_date.date().toString("yyyy-MM-dd")
            }

            start_date_str = self.dashboard_start_date.date().toString("dd/MM/yyyy")
            end_date_str = self.dashboard_end_date.date().toString("dd/MM/yyyy")
            self.dashboard_filter_status.setText(f"🔍 Filtered: {start_date_str} to {end_date_str}")
            self.dashboard_filter_status.setStyleSheet(
                f"color: {ModernUITheme.WARNING}; font-weight: 600; font-style: italic;")

            self.refresh_dashboard()
            self.statusBar().showMessage("Dashboard filter applied")
        except Exception as e:
            logger.error(f"Error applying dashboard filter: {e}")

    def clear_dashboard_filter(self):
        try:
            self.dashboard_filters_applied = False
            self.current_dashboard_filters = None

            self.dashboard_start_date.setDate(QDate.currentDate().addDays(-30))
            self.dashboard_end_date.setDate(QDate.currentDate())

            self.dashboard_filter_status.setText("📄 Showing all data")
            self.dashboard_filter_status.setStyleSheet(f"color: {ModernUITheme.TEXT_MUTED}; font-style: italic;")

            self.refresh_dashboard()
            self.statusBar().showMessage("Dashboard filter cleared")
        except Exception as e:
            logger.error(f"Error clearing dashboard filter: {e}")

    def refresh_dashboard(self):
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'dashboard_refresh'
            if 'dashboard' in self.loading_overlays:
                self.loading_overlays['dashboard'].show_loading("Refreshing dashboard...")
            filters = self.current_dashboard_filters if self.dashboard_filters_applied else None
            self.db_worker.add_operation('load_statistics', operation_id, self.db, filters)
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")

    def apply_ban_filters(self):
        try:
            self.ban_filters_applied = True
            self.current_ban_filters = {
                'start_date': self.ban_start_date.date().toString("yyyy-MM-dd"),
                'end_date': self.ban_end_date.date().toString("yyyy-MM-dd")
            }

            if self.ban_tanker_filter.text().strip():
                self.current_ban_filters['tanker_number'] = self.ban_tanker_filter.text().strip()
            if self.ban_reason_filter.text().strip():
                self.current_ban_filters['reason'] = self.ban_reason_filter.text().strip()
            if self.ban_type_filter.currentText() != "All":
                self.current_ban_filters['ban_type'] = self.ban_type_filter.currentText()

            self.load_bans_table()
            self.statusBar().showMessage("Ban filters applied")
        except Exception as e:
            logger.error(f"Error applying ban filters: {e}")

    def clear_ban_filters(self):
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
            self.statusBar().showMessage("Ban filters cleared")
        except Exception as e:
            logger.error(f"Error clearing ban filters: {e}")

    def apply_log_filters(self):
        try:
            filters = {
                'start_date': self.log_start_date.date().toString("yyyy-MM-dd"),
                'end_date': self.log_end_date.date().toString("yyyy-MM-dd")
            }

            if self.log_tanker_filter.text().strip():
                filters['tanker_number'] = self.log_tanker_filter.text().strip()
            if self.log_operator_filter.text().strip():
                filters['operator'] = self.log_operator_filter.text().strip()
            if self.log_status_filter.currentText() != "All":
                filters['status'] = self.log_status_filter.currentText()

            self.load_logs_table(filters)
            self.statusBar().showMessage("Log filters applied")
        except Exception as e:
            logger.error(f"Error applying log filters: {e}")

    def clear_log_filters(self):
        try:
            self.log_start_date.setDate(QDate.currentDate().addDays(-7))
            self.log_end_date.setDate(QDate.currentDate())
            self.log_tanker_filter.clear()
            self.log_operator_filter.clear()
            self.log_status_filter.setCurrentText("All")

            self.log_filter_status.setText("📄 Showing last 7 days")
            self.log_filter_status.setStyleSheet(f"color: {ModernUITheme.TEXT_MUTED}; font-style: italic;")

            self.load_logs_table()
            self.statusBar().showMessage("Log filters cleared")
        except Exception as e:
            logger.error(f"Error clearing log filters: {e}")

    def load_bans_table(self):
        """Load bans table - FIXED to prevent crashes"""
        try:
            # Use the safe version
            self.load_bans_table_safe()
        except Exception as e:
            logger.error(f"Error in load_bans_table: {e}")
            # Fallback to basic loading
            try:
                if hasattr(self, 'bans_table'):
                    self.bans_table.setEnabled(False)

                # Quick load without cache
                filters = getattr(self, 'current_ban_filters', None) if getattr(self, 'ban_filters_applied',
                                                                                False) else None
                include_inactive = (hasattr(self, 'show_inactive_checkbox') and
                                    self.show_inactive_checkbox.isChecked())

                bans_data = self.db.get_all_bans(filters, True, include_inactive)
                self.populate_bans_table_from_data(bans_data)

            except Exception as fallback_error:
                logger.error(f"Fallback loading also failed: {fallback_error}")
                self.statusBar().showMessage("Failed to load ban records")
            finally:
                if hasattr(self, 'bans_table'):
                    self.bans_table.setEnabled(True)
    def on_show_inactive_changed(self):
        """Handle show inactive checkbox change"""
        self.load_bans_table()

    def safe_edit_ban_record(self, ban_id):
        """SAFE: Edit ban record"""
        try:
            logger.info(f"SAFE EDIT: Starting for ban_id={ban_id}")

            # Stop audio first
            try:
                if hasattr(self, 'audio_recorder') and self.audio_recorder:
                    self.audio_recorder.stop_playback()
            except:
                pass

            # Get ban record details safely
            with sqlite3.connect(self.db.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, tanker_number, ban_reason, ban_type, start_date, end_date
                    FROM ban_records WHERE id = ?
                """, (ban_id,))
                ban_data = cursor.fetchone()

            if not ban_data:
                QMessageBox.warning(self, "Not Found", "Ban record not found")
                return

            dialog = self.create_ban_edit_dialog(ban_data)
            if dialog.exec_() == QDialog.Accepted:
                QTimer.singleShot(200, self.safe_refresh_table)

        except Exception as e:
            logger.error(f"Error in safe_edit_ban_record: {e}")
            QMessageBox.critical(self, "Edit Error", f"Failed to edit ban record: {e}")

    def create_ban_edit_dialog(self, ban_data):
        """ENHANCED: Edit dialog with voice recording support"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Ban Record - ID: {ban_data[0]}")
        dialog.setFixedSize(700, 600)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel(f"📝 Editing Ban Record #{ban_data[0]}")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2563EB; margin-bottom: 20px;")
        layout.addWidget(header_label)

        # Form fields
        form_layout = QFormLayout()

        tanker_input = QLineEdit(ban_data[1])
        tanker_input.setObjectName("modernInput")

        reason_input = QTextEdit()
        reason_input.setPlainText(ban_data[2])
        reason_input.setMaximumHeight(120)
        reason_input.setObjectName("modernTextEdit")

        type_combo = QComboBox()
        type_combo.addItems(["temporary", "permanent", "permission", "reminder"])
        type_combo.setCurrentText(ban_data[3])
        type_combo.setObjectName("modernCombo")

        start_date = QDateEdit()
        start_date.setDate(QDate.fromString(ban_data[4], "yyyy-MM-dd") if ban_data[4] else QDate.currentDate())
        start_date.setCalendarPopup(True)
        start_date.setObjectName("modernDateEdit")

        end_date = QDateEdit()
        if ban_data[5]:
            end_date.setDate(QDate.fromString(ban_data[5], "yyyy-MM-dd"))
        else:
            end_date.setDate(QDate.currentDate().addDays(30))
        end_date.setCalendarPopup(True)
        end_date.setObjectName("modernDateEdit")

        form_layout.addRow("Tanker Number:", tanker_input)
        form_layout.addRow("Ban Reason:", reason_input)
        form_layout.addRow("Ban Type:", type_combo)
        form_layout.addRow("Start Date:", start_date)
        form_layout.addRow("End Date:", end_date)

        layout.addLayout(form_layout)

        # Voice recording section
        voice_container = QFrame()
        voice_container.setStyleSheet(
            "background-color: #F3F4F6; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px;")
        voice_layout = QVBoxLayout(voice_container)

        voice_header = QLabel("🎙️ Voice Note (Optional)")
        voice_header.setStyleSheet("font-weight: bold; color: #374151;")
        voice_layout.addWidget(voice_header)

        voice_controls = QHBoxLayout()
        voice_status_label = QLabel("⚪ Current voice note will be preserved")
        voice_status_label.setStyleSheet("color: #6B7280;")

        record_new_btn = QPushButton("🎙️ Record New Voice Note")
        record_new_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #047857; }
        """)

        # Store recorded data
        new_voice_data = None

        def record_new_voice():
            nonlocal new_voice_data
            try:
                if AUDIO_AVAILABLE and hasattr(self, 'audio_recorder') and self.audio_recorder:
                    voice_dialog = VoiceRecordingDialog(dialog)
                    if voice_dialog.exec_() == QDialog.Accepted:
                        new_voice_data = voice_dialog.get_recorded_data()
                        if new_voice_data:
                            voice_status_label.setText("🟢 New voice note recorded")
                            record_new_btn.setText("🎙️ Record Again")
                        else:
                            voice_status_label.setText("❌ Recording failed")
                else:
                    QMessageBox.warning(dialog, "Audio Error", "Voice recording not available")
            except Exception as e:
                logger.error(f"Error recording voice: {e}")
                QMessageBox.critical(dialog, "Recording Error", f"Failed to record voice: {e}")

        record_new_btn.clicked.connect(record_new_voice)

        voice_controls.addWidget(voice_status_label)
        voice_controls.addStretch()
        voice_controls.addWidget(record_new_btn)

        voice_layout.addLayout(voice_controls)
        layout.addWidget(voice_container)

        # Buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6B7280;
                border: 2px solid #E5E7EB;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #F3F4F6; }
        """)
        cancel_btn.clicked.connect(dialog.reject)

        save_btn = QPushButton("💾 Save Changes")
        save_btn.setObjectName("saveButton")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #1D4ED8; }
        """)

        def save_changes():
            try:
                # Ensure database method exists
                if not hasattr(self.db, 'update_ban_record'):
                    QMessageBox.critical(dialog, "System Error", "Update method not available")
                    return

                success = self.db.update_ban_record(
                    ban_data[0],
                    tanker_input.text().strip().upper(),
                    reason_input.toPlainText().strip(),
                    type_combo.currentText(),
                    start_date.date().toString("yyyy-MM-dd"),
                    end_date.date().toString("yyyy-MM-dd") if type_combo.currentText() != "permanent" else None,
                    self.user_info['username']
                )

                # TODO: Handle new voice data if needed
                # This would require extending update_ban_record to handle voice data

                if success:
                    QMessageBox.information(dialog, "Success", "Ban record updated successfully")
                    dialog.accept()
                else:
                    QMessageBox.critical(dialog, "Error", "Failed to update ban record")
            except Exception as e:
                logger.error(f"Error saving changes: {e}")
                QMessageBox.critical(dialog, "Error", f"Update failed: {e}")

        save_btn.clicked.connect(save_changes)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)
        return dialog

    def safe_deactivate_ban_record(self, ban_id):
        """CORRECTED: MainWindow method with proper QMessageBox calls"""
        try:
            logger.info(f"SAFE DEACTIVATE: Starting for ban_id={ban_id}")

            # Immediate validation
            if not ban_id:
                logger.error("Invalid ban_id: None or empty")
                return

            # Convert to int safely
            try:
                ban_id = int(ban_id)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid ban_id format: {ban_id}, error: {e}")
                QMessageBox.warning(self, "Invalid Request", "Invalid ban record ID")
                return

            # Stop all audio operations immediately
            try:
                if hasattr(self, 'audio_recorder') and self.audio_recorder:
                    self.audio_recorder.stop_playback()
                if hasattr(self, 'currently_playing_row'):
                    self.currently_playing_row = None
            except Exception as e:
                logger.debug(f"Audio cleanup error (non-critical): {e}")

            # Ensure database connection is available
            if not hasattr(self, 'db') or not self.db:
                logger.error("Database not available")
                QMessageBox.critical(self, "System Error", "Database connection not available")
                return

            # Check if method exists
            if not hasattr(self.db, 'deactivate_ban_record'):
                logger.error("Deactivate method not available")
                QMessageBox.critical(self, "System Error", "Deactivate method not available")
                return

            # Show confirmation dialog (THIS is where QMessageBox should be - in MainWindow)
            reply = QMessageBox.question(
                self,  # self is MainWindow (QWidget) - CORRECT
                "Confirm Deactivation",
                f"Deactivate ban record ID {ban_id}?\n\nThis will make the ban inactive.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                logger.info("Deactivation cancelled by user")
                return

            # Call database method (NO QMessageBox in database method)
            try:
                logger.info(f"Calling database deactivate for ban_id={ban_id}")
                success = self.db.deactivate_ban_record(ban_id, self.user_info['username'])

                if success:
                    logger.info(f"Deactivation successful for ban_id={ban_id}")
                    QMessageBox.information(self, "Success", f"Ban record {ban_id} deactivated")

                    # Refresh table with delay
                    QTimer.singleShot(500, self.safe_refresh_table)

                else:
                    logger.error(f"Database deactivation failed for ban_id={ban_id}")
                    QMessageBox.critical(self, "Error", "Failed to deactivate ban record")

            except Exception as db_error:
                logger.error(f"Database error during deactivation: {db_error}")
                QMessageBox.critical(self, "Database Error", f"Deactivation failed: {db_error}")

        except Exception as e:
            logger.error(f"Critical error in deactivate_ban_record: {e}")
            QMessageBox.critical(self, "Critical Error", f"Operation failed: {e}")

    def safe_delete_ban_record(self, ban_id):
        """CORRECTED: MainWindow method with proper QMessageBox calls"""
        try:
            logger.info(f"SAFE DELETE: Starting for ban_id={ban_id}")

            # Immediate validation
            if not ban_id:
                logger.error("Invalid ban_id: None or empty")
                return

            # Convert to int safely
            try:
                ban_id = int(ban_id)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid ban_id format: {ban_id}, error: {e}")
                QMessageBox.warning(self, "Invalid Request", "Invalid ban record ID")
                return

            # Stop all audio operations immediately
            try:
                if hasattr(self, 'audio_recorder') and self.audio_recorder:
                    self.audio_recorder.stop_playback()
                if hasattr(self, 'currently_playing_row'):
                    self.currently_playing_row = None
            except Exception as e:
                logger.debug(f"Audio cleanup error (non-critical): {e}")

            # Ensure database connection is available
            if not hasattr(self, 'db') or not self.db:
                logger.error("Database not available")
                QMessageBox.critical(self, "System Error", "Database connection not available")
                return

            # Check if method exists
            if not hasattr(self.db, 'delete_ban_record'):
                logger.error("Delete method not available")
                QMessageBox.critical(self, "System Error", "Delete method not available")
                return

            # Double confirmation for deletion (QMessageBox calls in MainWindow - CORRECT)
            reply1 = QMessageBox.question(
                self,  # self is MainWindow (QWidget) - CORRECT
                "⚠️ PERMANENT DELETION WARNING",
                f"PERMANENTLY DELETE ban record {ban_id}?\n\nThis cannot be undone!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply1 != QMessageBox.Yes:
                logger.info("Deletion cancelled by user (first confirmation)")
                return

            reply2 = QMessageBox.question(
                self,  # self is MainWindow (QWidget) - CORRECT
                "Final Confirmation",
                f"LAST CHANCE TO CANCEL!\n\nDelete ban record {ban_id}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply2 != QMessageBox.Yes:
                logger.info("Deletion cancelled by user (second confirmation)")
                return

            # Call database method (NO QMessageBox in database method)
            try:
                logger.info(f"Calling database delete for ban_id={ban_id}")
                success = self.db.delete_ban_record(ban_id, self.user_info['username'])

                if success:
                    logger.info(f"Deletion successful for ban_id={ban_id}")
                    QMessageBox.information(self, "Success", f"Ban record {ban_id} deleted permanently")

                    # Refresh table with delay
                    QTimer.singleShot(500, self.safe_refresh_table)

                else:
                    logger.error(f"Database deletion failed for ban_id={ban_id}")
                    QMessageBox.critical(self, "Error", "Failed to delete ban record")

            except Exception as db_error:
                logger.error(f"Database error during deletion: {db_error}")
                QMessageBox.critical(self, "Database Error", f"Deletion failed: {db_error}")

        except Exception as e:
            logger.error(f"Critical error in delete_ban_record: {e}")
            QMessageBox.critical(self, "Critical Error", f"Operation failed: {e}")

    def safe_reactivate_ban_record(self, ban_id):
        """CORRECTED: MainWindow method with proper QMessageBox calls"""
        try:
            logger.info(f"SAFE REACTIVATE: Starting for ban_id={ban_id}")

            # Similar validation as deactivate...
            if not ban_id:
                return

            try:
                ban_id = int(ban_id)
            except (ValueError, TypeError):
                QMessageBox.warning(self, "Invalid Request", "Invalid ban record ID")
                return

            # Stop audio
            try:
                if hasattr(self, 'audio_recorder') and self.audio_recorder:
                    self.audio_recorder.stop_playback()
            except:
                pass

            # Show confirmation (QMessageBox in MainWindow - CORRECT)
            reply = QMessageBox.question(
                self,  # self is MainWindow (QWidget) - CORRECT
                "Confirm Reactivation",
                f"Reactivate ban record ID {ban_id}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes and hasattr(self.db, 'reactivate_ban_record'):
                # Call database method (NO QMessageBox in database method)
                success = self.db.reactivate_ban_record(ban_id, self.user_info['username'])
                if success:
                    QMessageBox.information(self, "Success", f"Ban record {ban_id} reactivated")
                    QTimer.singleShot(500, self.safe_refresh_table)
                else:
                    QMessageBox.critical(self, "Error", "Failed to reactivate ban record")

        except Exception as e:
            logger.error(f"Error in reactivate_ban_record: {e}")
            QMessageBox.critical(self, "Critical Error", f"Operation failed: {e}")

    def update_ban_record(self, ban_id, tanker_number, ban_reason, ban_type, start_date, end_date, updated_by):
        """Update an existing ban record"""
        try:
            with sqlite3.connect(self.sqlite_db, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ban_records 
                    SET tanker_number = ?, ban_reason = ?, ban_type = ?, start_date = ?, end_date = ?
                    WHERE id = ?
                """, (tanker_number, ban_reason, ban_type, start_date, end_date, ban_id))

                # Log the update
                cursor.execute("""
                    INSERT INTO logs (tanker_number, status, reason, operator)
                    VALUES (?, ?, ?, ?)
                """, (tanker_number, "BAN_UPDATED", f"Ban record updated: {ban_reason}", updated_by))

                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating ban record {ban_id}: {e}")
            return False

    def safe_refresh_table(self):
        """SAFE: Refresh table without causing crashes"""
        try:
            logger.info("SAFE REFRESH: Starting table refresh")

            # Clear button handlers to prevent memory leaks
            if hasattr(self, '_button_handlers'):
                self._button_handlers.clear()

            # Call load with error handling
            self.load_bans_table()
            logger.info("SAFE REFRESH: Table refreshed successfully")

        except Exception as e:
            logger.error(f"Error in safe_refresh_table: {e}")

    def load_logs_table(self, filters=None):
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'load_logs'
            if 'logs' in self.loading_overlays:
                self.loading_overlays['logs'].show_loading("Loading activity logs...")
            if hasattr(self, 'logs_table'):
                self.logs_table.setEnabled(False)
            self.db_worker.add_operation('load_logs', operation_id, self.db, 100, filters)
        except Exception as e:
            logger.error(f"Error loading logs table: {e}")

    def populate_bans_table_from_data(self, bans_data):
        """CRASH-SAFE: Enhanced table population with safe button handling"""
        try:
            if not hasattr(self, 'bans_table'):
                return

            # Clear any existing button references to prevent memory issues
            if hasattr(self, '_button_handlers'):
                self._button_handlers.clear()
            else:
                self._button_handlers = {}

            self.bans_table.setRowCount(0)
            self.bans_table.setColumnCount(12)

            self.bans_table.setHorizontalHeaderLabels([
                "ID", "Tanker", "Reason", "Type", "Start", "End", "By", "Status", "Created", "🎧 Audio", "📝 Edit",
                "🗑️ Actions"
            ])

            self.currently_playing_row = None

            for row, ban in enumerate(bans_data):
                self.bans_table.insertRow(row)

                # Populate data columns (0-8)
                for col in range(min(9, len(ban))):
                    value = ban[col] if col < len(ban) else ""
                    item = QTableWidgetItem(str(value) if value is not None else "")

                    # Enhanced styling for different columns
                    if col == 3:  # Ban type column
                        ban_type = str(value).lower() if value else ""
                        if ban_type == "permanent":
                            item.setForeground(QColor(ModernUITheme.ERROR))
                            item.setFont(QFont("Arial", weight=QFont.Bold))
                        elif ban_type == "temporary":
                            item.setForeground(QColor(ModernUITheme.WARNING))
                        elif ban_type in ["permission", "reminder"]:
                            item.setForeground(QColor(ModernUITheme.INFO))

                    self.bans_table.setItem(row, col, item)

                # Add Status column (7)
                is_active = ban[9] if len(ban) > 9 else 1
                status_item = QTableWidgetItem("Active" if is_active else "Inactive")
                if is_active:
                    status_item.setForeground(QColor(ModernUITheme.SUCCESS))
                    status_item.setFont(QFont("Arial", weight=QFont.Bold))
                else:
                    status_item.setForeground(QColor(ModernUITheme.TEXT_MUTED))
                    status_item.setFont(QFont("Arial", style=QFont.StyleItalic))
                self.bans_table.setItem(row, 7, status_item)

                # SAFE: Audio Button (Column 9)
                self._create_safe_audio_button(row, ban)

                # SAFE: Edit Button (Column 10)
                self._create_safe_edit_button(row, ban[0])

                # SAFE: Actions Button Group (Column 11)
                self._create_safe_action_buttons(row, ban[0], is_active)

            # Update count label
            active_count = sum(1 for ban in bans_data if len(ban) > 9 and ban[9])
            inactive_count = len(bans_data) - active_count
            total_count = len(bans_data)

            if hasattr(self, 'show_inactive_checkbox') and self.show_inactive_checkbox.isChecked():
                self.ban_count_label.setText(
                    f"📊 Total: {total_count} records (Active: {active_count}, Inactive: {inactive_count})")
            else:
                self.ban_count_label.setText(f"📊 Showing {active_count} active ban records")

            logger.info(f"SAFE: Enhanced ban table populated with {len(bans_data)} records")

        except Exception as e:
            logger.error(f"Error populating enhanced bans table: {e}")
            QMessageBox.critical(self, "Table Error", f"Failed to populate ban records table: {e}")

    def _create_safe_audio_button(self, row, ban):
        """Create audio button safely without memory issues"""
        try:
            audio_container = QWidget()
            audio_layout = QHBoxLayout(audio_container)
            audio_layout.setContentsMargins(5, 5, 5, 5)
            audio_layout.setAlignment(Qt.AlignCenter)

            audio_button = QPushButton("🎧")
            audio_button.setObjectName("audioPlayButton")
            audio_button.setFixedSize(35, 35)
            audio_button.setToolTip("Play voice recording")

            # Store row info in button without using properties
            button_id = f"audio_{row}_{ban[0]}"
            self._button_handlers[button_id] = {
                'type': 'audio',
                'row': row,
                'record_id': ban[0]
            }

            # Check if audio data exists
            has_audio = len(ban) > 7 and ban[7] is not None and ban[7] != ""
            if has_audio:
                audio_button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ModernUITheme.SUCCESS};
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #047857;
                        transform: scale(1.05);
                    }}
                """)
                # SAFE: Use button_id instead of lambda to avoid circular references
                audio_button.clicked.connect(lambda checked, bid=button_id: self._handle_audio_click(bid))
            else:
                audio_button.setText("🚫")
                audio_button.setEnabled(False)
                audio_button.setToolTip("No voice recording available")
                audio_button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ModernUITheme.TEXT_MUTED};
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 16px;
                    }}
                """)

            audio_layout.addWidget(audio_button)
            self.bans_table.setCellWidget(row, 9, audio_container)

        except Exception as e:
            logger.error(f"Error creating audio button for row {row}: {e}")

    def _create_safe_edit_button(self, row, ban_id):
        """Create edit button safely"""
        try:
            edit_container = QWidget()
            edit_layout = QHBoxLayout(edit_container)
            edit_layout.setContentsMargins(5, 5, 5, 5)
            edit_layout.setAlignment(Qt.AlignCenter)

            edit_button = QPushButton("✏️")
            edit_button.setFixedSize(35, 35)
            edit_button.setToolTip("Edit ban record")
            edit_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ModernUITheme.PRIMARY};
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {ModernUITheme.PRIMARY_DARK};
                    transform: scale(1.05);
                }}
            """)

            # SAFE: Store handler info and use direct connection
            button_id = f"edit_{row}_{ban_id}"
            self._button_handlers[button_id] = {
                'type': 'edit',
                'row': row,
                'record_id': ban_id
            }

            edit_button.clicked.connect(lambda checked, bid=button_id: self._handle_edit_click(bid))

            edit_layout.addWidget(edit_button)
            self.bans_table.setCellWidget(row, 10, edit_container)

        except Exception as e:
            logger.error(f"Error creating edit button for row {row}: {e}")

    def _create_safe_action_buttons(self, row, ban_id, is_active):
        """Create action buttons safely to prevent crashes"""
        try:
            actions_container = QWidget()
            actions_layout = QHBoxLayout(actions_container)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            actions_layout.setAlignment(Qt.AlignCenter)

            if is_active:
                # Deactivate button
                deactivate_btn = QPushButton("⏸️")
                deactivate_btn.setFixedSize(25, 25)
                deactivate_btn.setToolTip("Deactivate ban")
                deactivate_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ModernUITheme.WARNING};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 12px;
                    }}
                    QPushButton:hover {{ 
                        background-color: #B45309; 
                    }}
                """)

                # SAFE: Store handler info
                button_id = f"deactivate_{row}_{ban_id}"
                self._button_handlers[button_id] = {
                    'type': 'deactivate',
                    'row': row,
                    'record_id': ban_id
                }

                deactivate_btn.clicked.connect(lambda checked, bid=button_id: self._handle_deactivate_click(bid))
                actions_layout.addWidget(deactivate_btn)
            else:
                # Reactivate button
                reactivate_btn = QPushButton("▶️")
                reactivate_btn.setFixedSize(25, 25)
                reactivate_btn.setToolTip("Reactivate ban")
                reactivate_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ModernUITheme.SUCCESS};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 12px;
                    }}
                    QPushButton:hover {{ 
                        background-color: #047857; 
                    }}
                """)

                # SAFE: Store handler info
                button_id = f"reactivate_{row}_{ban_id}"
                self._button_handlers[button_id] = {
                    'type': 'reactivate',
                    'row': row,
                    'record_id': ban_id
                }

                reactivate_btn.clicked.connect(lambda checked, bid=button_id: self._handle_reactivate_click(bid))
                actions_layout.addWidget(reactivate_btn)

            # Delete button
            delete_btn = QPushButton("🗑️")
            delete_btn.setFixedSize(25, 25)
            delete_btn.setToolTip("Permanently delete ban")
            delete_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ModernUITheme.ERROR};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 12px;
                }}
                QPushButton:hover {{ 
                    background-color: #B91C1C; 
                }}
            """)

            # SAFE: Store handler info
            button_id = f"delete_{row}_{ban_id}"
            self._button_handlers[button_id] = {
                'type': 'delete',
                'row': row,
                'record_id': ban_id
            }

            delete_btn.clicked.connect(lambda checked, bid=button_id: self._handle_delete_click(bid))
            actions_layout.addWidget(delete_btn)

            self.bans_table.setCellWidget(row, 11, actions_container)

        except Exception as e:
            logger.error(f"Error creating action buttons for row {row}: {e}")

    def _handle_audio_click(self, button_id):
        """SAFE: Handle audio button click"""
        try:
            if button_id not in self._button_handlers:
                logger.error(f"Unknown button ID: {button_id}")
                return

            handler_info = self._button_handlers[button_id]
            row = handler_info['row']
            record_id = handler_info['record_id']

            logger.info(f"Audio click: row={row}, record_id={record_id}")
            self.toggle_audio_playback(row)

        except Exception as e:
            logger.error(f"Error handling audio click {button_id}: {e}")

    def _handle_edit_click(self, button_id):
        """SAFE: Handle edit button click"""
        try:
            if button_id not in self._button_handlers:
                logger.error(f"Unknown button ID: {button_id}")
                return

            handler_info = self._button_handlers[button_id]
            record_id = handler_info['record_id']

            logger.info(f"Edit click: record_id={record_id}")
            self.safe_edit_ban_record(record_id)

        except Exception as e:
            logger.error(f"Error handling edit click {button_id}: {e}")

    def _handle_deactivate_click(self, button_id):
        """SAFE: Handle deactivate button click"""
        try:
            if button_id not in self._button_handlers:
                logger.error(f"Unknown button ID: {button_id}")
                return

            handler_info = self._button_handlers[button_id]
            record_id = handler_info['record_id']

            logger.info(f"Deactivate click: record_id={record_id}")
            self.safe_deactivate_ban_record(record_id)

        except Exception as e:
            logger.error(f"Error handling deactivate click {button_id}: {e}")

    def _handle_reactivate_click(self, button_id):
        """SAFE: Handle reactivate button click"""
        try:
            if button_id not in self._button_handlers:
                logger.error(f"Unknown button ID: {button_id}")
                return

            handler_info = self._button_handlers[button_id]
            record_id = handler_info['record_id']

            logger.info(f"Reactivate click: record_id={record_id}")
            self.safe_reactivate_ban_record(record_id)

        except Exception as e:
            logger.error(f"Error handling reactivate click {button_id}: {e}")

    def _handle_delete_click(self, button_id):
        """SAFE: Handle delete button click"""
        try:
            if button_id not in self._button_handlers:
                logger.error(f"Unknown button ID: {button_id}")
                return

            handler_info = self._button_handlers[button_id]
            record_id = handler_info['record_id']

            logger.info(f"Delete click: record_id={record_id}")
            self.safe_delete_ban_record(record_id)

        except Exception as e:
            logger.error(f"Error handling delete click {button_id}: {e}")

    def toggle_audio_playback(self, row_index):
        """FIXED: Audio playback with proper error handling and UI updates"""
        try:
            if row_index < 0 or row_index >= self.bans_table.rowCount():
                logger.error(f"Invalid row index: {row_index}")
                return

            # Get the button widget safely
            button_widget = self.bans_table.cellWidget(row_index, 9)  # Audio column
            if not button_widget:
                logger.error(f"No audio widget found at row {row_index}")
                QMessageBox.warning(self, "Error", "Audio button not found")
                return

            # Find the actual button inside the widget container
            audio_button = None
            for child in button_widget.findChildren(QPushButton):
                if child.objectName() == "audioPlayButton":
                    audio_button = child
                    break

            if not audio_button:
                logger.error(f"No audio button found at row {row_index}")
                QMessageBox.warning(self, "Error", "Audio button not found")
                return

            # If same row is playing → stop it
            if self.currently_playing_row == row_index:
                self.stop_current_audio_playback()
                return

            # Stop any previous playback
            if self.currently_playing_row is not None:
                self.stop_current_audio_playback()

            # Get the record ID safely
            record_id = audio_button.property("record_id")
            if not record_id:
                id_item = self.bans_table.item(row_index, 0)
                if not id_item:
                    QMessageBox.warning(self, "Error", "Could not find record ID")
                    return
                try:
                    record_id = int(id_item.text())
                except ValueError:
                    QMessageBox.warning(self, "Error", "Invalid record ID")
                    return

            # Retrieve audio data from database
            try:
                with sqlite3.connect(self.db.sqlite_db, timeout=10) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT voice_recording, voice_filename, ban_reason FROM ban_records WHERE id = ?",
                        (record_id,)
                    )
                    result = cursor.fetchone()

                    if not result:
                        QMessageBox.information(self, "Not Found", "Ban record not found in database")
                        return

                    voice_data, voice_filename, ban_reason = result

                    if not voice_data:
                        QMessageBox.information(self, "No Audio",
                                                f"No voice recording found for this ban record.\n\nReason: {ban_reason}")
                        return

                    # Update button state before playing
                    audio_button.setText("⏹️")
                    audio_button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {ModernUITheme.ERROR};
                            color: white;
                            border: none;
                            border-radius: 6px;
                            font-size: 16px;
                            font-weight: bold;
                        }}
                        QPushButton:hover {{
                            background-color: #B91C1C;
                        }}
                    """)

                    self.currently_playing_row = row_index
                    self.statusBar().showMessage(f"Playing audio for ban record {record_id}...")

                    # Play audio with proper callback
                    if self.audio_recorder and AUDIO_AVAILABLE:
                        def on_playback_finished(success, message):
                            try:
                                # Make sure we're still on the same row and button exists
                                if (self.currently_playing_row == row_index and
                                        row_index < self.bans_table.rowCount()):

                                    current_widget = self.bans_table.cellWidget(row_index, 9)
                                    if current_widget:
                                        # Find the button again
                                        for child in current_widget.findChildren(QPushButton):
                                            if child.objectName() == "audioPlayButton":
                                                child.setText("🎧")
                                                child.setStyleSheet(f"""
                                                    QPushButton {{
                                                        background-color: {ModernUITheme.SUCCESS};
                                                        color: white;
                                                        border: none;
                                                        border-radius: 6px;
                                                        font-size: 16px;
                                                        font-weight: bold;
                                                    }}
                                                    QPushButton:hover {{
                                                        background-color: #047857;
                                                    }}
                                                """)
                                                break

                                self.currently_playing_row = None

                                if success:
                                    self.statusBar().showMessage(f"Audio playback completed for record {record_id}")
                                else:
                                    self.statusBar().showMessage(f"Audio playback failed: {message}")
                                    logger.error(f"Audio playback failed: {message}")

                            except Exception as e:
                                logger.error(f"Error in playback callback: {e}")

                        self.audio_recorder.play_audio(voice_data, callback=on_playback_finished)

                    else:
                        QMessageBox.warning(self, "Audio Error",
                                            "Audio playback not available. Audio system may not be initialized.")
                        self.stop_current_audio_playback()

            except sqlite3.Error as e:
                logger.error(f"Database error during audio retrieval: {e}")
                QMessageBox.critical(self, "Database Error", f"Could not retrieve audio data: {e}")
                self.stop_current_audio_playback()

            except Exception as e:
                logger.error(f"Error retrieving audio data: {e}")
                QMessageBox.critical(self, "Error", f"Failed to retrieve audio: {e}")
                self.stop_current_audio_playback()

        except Exception as e:
            logger.error(f"Audio playback failed for row {row_index}: {e}")
            QMessageBox.critical(self, "Playback Error", f"Could not play audio: {e}")
            if hasattr(self, 'currently_playing_row'):
                self.currently_playing_row = None
    def stop_current_audio_playback(self):
        """Stop current audio and reset button state"""
        try:
            if self.currently_playing_row is not None:
                if self.currently_playing_row < self.bans_table.rowCount():
                    button = self.bans_table.cellWidget(self.currently_playing_row, 9)
                    if button and button.isEnabled():
                        button.setText("🎧 Play")
                        button.setStyleSheet(
                            "background-color: #059669; color: white; border: none; border-radius: 4px; padding: 4px 8px;")

                if self.audio_recorder and hasattr(self.audio_recorder, 'stop_playback'):
                    self.audio_recorder.stop_playback()

                self.currently_playing_row = None
                self.statusBar().showMessage("Audio stopped")

        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
            self.currently_playing_row = None

    def populate_logs_table_from_data(self, logs_data):
        if hasattr(self, 'logs_table'):
            self.logs_table.setRowCount(0)
            for row, log in enumerate(logs_data):
                self.logs_table.insertRow(row)
                for col, value in enumerate(log):
                    item = QTableWidgetItem(str(value) if value else "")
                    self.logs_table.setItem(row, col, item)

    def update_dashboard_with_statistics(self, stats):
        try:
            def update_stat_card(card_widget, new_value):
                if card_widget:
                    for child in card_widget.findChildren(QLabel):
                        if child.objectName() == "valueLabel":
                            child.setText(str(new_value))
                            break

            if hasattr(self, "label_total_bans"):
                update_stat_card(self.label_total_bans, stats.get("ban_stats", {}).get("total_bans", 0))
            if hasattr(self, "label_active_bans"):
                update_stat_card(self.label_active_bans, stats.get("ban_stats", {}).get("active_bans", 0))
            if hasattr(self, "label_total_verifications"):
                update_stat_card(self.label_total_verifications, stats.get("verify_stats", {}).get("total", 0))
            if hasattr(self, "label_success_rate"):
                success_rate = stats.get("verify_stats", {}).get("success_rate", 0.0)
                update_stat_card(self.label_success_rate, f"{round(success_rate, 1)}%")

            if hasattr(self, "recent_bans_table"):
                self.recent_bans_table.setRowCount(0)
                for row_data in stats.get("recent_bans", []):
                    row = self.recent_bans_table.rowCount()
                    self.recent_bans_table.insertRow(row)
                    for col, value in enumerate(
                            [row_data[1], row_data[2], row_data[3], row_data[4], row_data[5], row_data[6]]):
                        self.recent_bans_table.setItem(row, col, QTableWidgetItem(str(value)))

            if hasattr(self, "recent_table"):
                self.recent_table.setRowCount(0)
                for row_data in stats.get("recent_logs", []):
                    row = self.recent_table.rowCount()
                    self.recent_table.insertRow(row)
                    for col, value in enumerate([row_data[4], row_data[1], row_data[2], row_data[3], row_data[5]]):
                        self.recent_table.setItem(row, col, QTableWidgetItem(str(value)))
        except Exception as e:
            logger.error(f"Error updating dashboard statistics: {e}")

    def test_server_connection(self):
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'test_server'
            if hasattr(self, 'server_status_label'):
                self.server_status_label.setText("Server Status: Testing...")
            self.db_worker.add_operation('test_connection', operation_id, self.db, 'server')
        except Exception as e:
            logger.error(f"Error testing server connection: {e}")

    def test_local_connection(self):
        """Test SQLite local database connection"""
        try:
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'test_local'
            if hasattr(self, 'local_status_label'):
                self.local_status_label.setText("Local Status: Testing...")

            # Test SQLite connection
            try:
                with sqlite3.connect(self.db.sqlite_db, timeout=10) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                success = True
                message = f"SQLite connection successful ({len(tables)} tables)"
            except Exception as e:
                success = False
                message = f"SQLite connection failed: {str(e)}"

            self.on_connection_tested('local', success, message)

        except Exception as e:
            logger.error(f"Error testing local connection: {e}")
            self.on_connection_tested('local', False, f"Test failed: {str(e)}")
    def test_all_connections(self):
        self.test_server_connection()
        self.test_local_connection()

    def browse_server_db(self):
        """Browse for Access server database file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Server Database",
            self.config.get('server_db_path', ''),
            "Access Database (*.mdb *.accdb);;All Files (*)"  # Keep Access format
        )
        if file_path:
            self.server_path_input.setText(file_path)

    def browse_local_db(self):
        """Browse for SQLite local database file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Local Database",
            self.config.get('local_sqlite_path', ''),
            "SQLite Database (*.db);;All Files (*)"  # SQLite format for local
        )
        if file_path:
            self.local_path_input.setText(file_path)
    def save_database_settings(self):
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
                QMessageBox.information(self, "Success",
                                        "Database settings saved successfully!\n\nConnections will be updated immediately.")
                self.statusBar().showMessage("Database settings updated")
                QTimer.singleShot(500, self.test_all_connections)
            else:
                QMessageBox.critical(self, "Error", "Failed to save database settings")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

    def save_system_settings(self):
        try:
            self.config.set('connection_timeout', self.connection_timeout_spin.value())
            self.config.set('auto_refresh_interval', self.auto_refresh_spin.value())

            if self.config.save_config():
                QMessageBox.information(self, "Success", "System settings saved successfully!")
                self.statusBar().showMessage("System settings updated")
                self.restart_monitoring()
            else:
                QMessageBox.critical(self, "Error", "Failed to save system settings")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save system settings: {str(e)}")

    def load_users_table(self):
        try:
            users = self.user_manager.get_all_users()
            self.users_table.setRowCount(0)

            for row, user in enumerate(users):
                self.users_table.insertRow(row)
                data = [str(user[0]), str(user[1]), str(user[2]) if user[2] else "", str(user[3]),
                        "Active" if user[4] == 1 else "Inactive", self.format_datetime(user[5]) if user[5] else "",
                        self.format_datetime(user[6]) if user[6] else "Never"]

                for col, value in enumerate(data):
                    item = QTableWidgetItem(str(value))
                    if col == 3:  # Role column
                        if value == "admin":
                            item.setForeground(QColor(ModernUITheme.ERROR))
                        elif value == "supervisor":
                            item.setForeground(QColor(ModernUITheme.WARNING))
                        else:
                            item.setForeground(QColor(ModernUITheme.SUCCESS))
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
                edit_btn.setFixedSize(30, 25)
                edit_btn.setStyleSheet(
                    f"QPushButton {{ background-color: {ModernUITheme.PRIMARY}; color: white; border: none; border-radius: 4px; font-weight: bold; }} QPushButton:hover {{ background-color: {ModernUITheme.PRIMARY_DARK}; }}")
                edit_btn.clicked.connect(lambda checked, u=user: self.edit_user(u))

                actions_layout.addWidget(edit_btn)
                actions_layout.addStretch()

                self.users_table.setCellWidget(row, 7, actions_widget)
        except Exception as e:
            logger.error(f"Error loading users table: {e}")

    def format_datetime(self, datetime_str):
        if not datetime_str:
            return ""
        try:
            dt = datetime.strptime(str(datetime_str), "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%d/%m/%Y %H:%M")
        except:
            return str(datetime_str)

    def add_new_user(self):
        from user_management_dialog import UserManagementDialog
        dialog = UserManagementDialog(self.user_manager, self.user_info, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.load_users_table()
            self.statusBar().showMessage("User added successfully")

    def edit_user(self, user_data):
        from user_management_dialog import UserManagementDialog
        dialog = UserManagementDialog(self.user_manager, self.user_info, user_data, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.load_users_table()
            self.statusBar().showMessage("User updated successfully")

    def restart_monitoring(self):
        try:
            if hasattr(self, 'monitor_timer') and self.monitor_timer:
                self.monitor_timer.stop()
            if hasattr(self, 'dashboard_timer') and self.dashboard_timer:
                self.dashboard_timer.stop()
            QTimer.singleShot(1000, self.start_monitoring)
        except Exception as e:
            logger.error(f"Error restarting monitoring: {e}")

    def start_monitoring(self):
        try:
            try:
                current_latest = self.db.get_latest_tanker_from_server()
                if current_latest:
                    self.last_tanker = current_latest.tanker_number
                else:
                    self.last_tanker = None
            except Exception as e:
                logger.error(f"Error getting latest tanker during monitoring start: {e}")
                self.last_tanker = None
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

            self.statusBar().showMessage("Monitoring system started")
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")

    def auto_refresh_dashboard(self):
        try:
            if (hasattr(self, 'stacked_widget') and hasattr(self,
                                                            'dashboard_page') and self.stacked_widget.currentWidget() == self.dashboard_page):
                self.refresh_dashboard()
        except Exception as e:
            logger.debug(f"Dashboard auto-refresh error: {e}")

    def check_new_tanker(self):
        try:
            latest_tanker = self.db.get_latest_tanker_from_server()
            if latest_tanker and latest_tanker.tanker_number != self.last_tanker:
                self.last_tanker = latest_tanker.tanker_number
                status, reason, details = self.db.simple_tanker_verification("AUTO_MONITOR")

                if details.get("play_sound", False):
                    self.play_warning_sound_for_status(status)

                if (hasattr(self, 'stacked_widget') and hasattr(self,
                                                                'verification_page') and self.stacked_widget.currentWidget() == self.verification_page):
                    self.update_auto_verification_display(latest_tanker.tanker_number, status, reason, details)

                if (hasattr(self, 'stacked_widget') and hasattr(self,
                                                                'dashboard_page') and self.stacked_widget.currentWidget() == self.dashboard_page):
                    self.refresh_dashboard()

                sound_status = " (Sound played)" if details.get("play_sound", False) else ""
                self.statusBar().showMessage(
                    f"New tanker auto-verified: {latest_tanker.tanker_number} - {status}{sound_status}")
        except Exception as e:
            logger.debug(f"Auto-verification error: {e}")

    # Navigation methods
    def show_dashboard(self):
        try:
            if hasattr(self, 'stacked_widget') and hasattr(self, 'dashboard_page'):
                self.stacked_widget.setCurrentWidget(self.dashboard_page)
                self.refresh_dashboard()
                self.statusBar().showMessage("Dashboard page loaded")
        except Exception as e:
            logger.error(f"Error showing dashboard: {e}")

    def create_simple_backup(self):
        """Simple backup method for testing"""
        try:
            # Get backup location
            folder = QFileDialog.getExistingDirectory(self, "Select Backup Location")
            if not folder:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(folder, f"TDF_Backup_{timestamp}")
            os.makedirs(backup_path, exist_ok=True)

            # Copy database file
            if hasattr(self.db, 'sqlite_db') and os.path.exists(self.db.sqlite_db):
                backup_db = os.path.join(backup_path, "database_backup.db")
                shutil.copy2(self.db.sqlite_db, backup_db)

            # Copy config file
            if os.path.exists("config.json"):
                backup_config = os.path.join(backup_path, "config_backup.json")
                shutil.copy2("config.json", backup_config)

            QMessageBox.information(self, "Backup Complete", f"Backup created at:\n{backup_path}")
            logger.info(f"Simple backup created: {backup_path}")

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            QMessageBox.critical(self, "Backup Failed", f"Backup failed: {e}")
    def show_verification(self):
        try:
            self.stacked_widget.setCurrentWidget(self.verification_page)
            self.statusBar().showMessage("Auto verification page loaded")
        except Exception as e:
            logger.error(f"Error showing verification page: {e}")

    def show_manual_verify(self):
        try:
            self.stacked_widget.setCurrentWidget(self.manual_verify_page)
            self.statusBar().showMessage("Manual verification page loaded")
        except Exception as e:
            logger.error(f"Error showing manual verify page: {e}")

    def show_bans(self):
        try:
            if hasattr(self, 'stacked_widget') and hasattr(self, 'bans_page'):
                self.stacked_widget.setCurrentWidget(self.bans_page)
                self.ban_filters_applied = False
                self.current_ban_filters = None
                self.load_bans_table()
                self.statusBar().showMessage("Ban management page loaded")
        except Exception as e:
            logger.error(f"Error showing bans page: {e}")

    def show_logs(self):
        try:
            if hasattr(self, 'stacked_widget') and hasattr(self, 'logs_page'):
                self.stacked_widget.setCurrentWidget(self.logs_page)
                self.load_logs_table()
                self.statusBar().showMessage("Activity logs page loaded")
        except Exception as e:
            logger.error(f"Error showing logs page: {e}")

    def show_settings(self):
        try:
            if self.user_info['role'] not in ['admin', 'supervisor']:
                QMessageBox.warning(self, "Access Denied",
                                    f"Settings access is restricted.\n\nYour role: {self.user_info['role']}\nRequired: Admin or Supervisor")
                return

            if hasattr(self, 'settings_page'):
                self.stacked_widget.setCurrentWidget(self.settings_page)
                if self.user_info['role'] == 'admin':
                    self.load_users_table()
                self.statusBar().showMessage("Settings page loaded")
            else:
                QMessageBox.warning(self, "Access Error", "Settings page not available for your role")
        except Exception as e:
            logger.error(f"Error showing settings page: {e}")

    def initialize_server_database(self):
        """Initialize server database with required tables"""
        try:
            if not hasattr(self, 'db') or not self.db:
                QMessageBox.warning(self, "Database Error", "Database manager not available")
                return

            server_path = self.server_path_input.text().strip()
            if not server_path:
                QMessageBox.warning(self, "Path Required", "Please enter a server database path first")
                return

            # Update the database path
            self.db.server_db = server_path

            # Initialize the database
            self.db.init_server_database()

            QMessageBox.information(self, "Success",
                                    f"Server database initialized successfully!\n\nPath: {server_path}\n\n"
                                    "Tables created:\n• VehicleMaster\n• VehicleTransactions")

            # Test the connection
            self.test_server_connection()

        except Exception as e:
            logger.error(f"Error initializing server database: {e}")
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize server database:\n\n{str(e)}")

    def add_sample_data(self):
        """Add sample data to server database"""
        try:
            if not hasattr(self, 'db') or not self.db:
                QMessageBox.warning(self, "Database Error", "Database manager not available")
                return

            if not self.db.server_available:
                QMessageBox.warning(self, "Server Unavailable",
                                    "Server database is not available. Please check connection and try again.")
                return

            reply = QMessageBox.question(self, "Add Sample Data",
                                         "This will add sample vehicle and transaction data to the server database.\n\n"
                                         "Do you want to proceed?",
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.db.add_sample_server_data()
                QMessageBox.information(self, "Success",
                                        "Sample data added successfully!\n\n"
                                        "Added:\n• Sample vehicles (TR001, TR002, 40247, TEST001)\n"
                                        "• Sample transactions for each vehicle")

                # Refresh connection test to show updated counts
                self.test_server_connection()

        except Exception as e:
            logger.error(f"Error adding sample data: {e}")
            QMessageBox.critical(self, "Sample Data Error", f"Failed to add sample data:\n\n{str(e)}")

    # Add these methods to your MainWindow class in main.py

    def add_verification_timeout_protection(self):
        """Add timeout protection for verification operations"""
        try:
            self.verification_timeout_timer = QTimer()
            self.verification_timeout_timer.setSingleShot(True)
            self.verification_timeout_timer.timeout.connect(self.handle_verification_timeout)
            logger.info("Verification timeout protection initialized")
        except Exception as e:
            logger.error(f"Error initializing timeout protection: {e}")

    def handle_verification_timeout(self):
        """Handle verification timeout"""
        try:
            logger.warning("Verification operation timed out")

            # Reset UI states for auto verification
            if hasattr(self, 'auto_status_label'):
                self.auto_status_label.setText("⏰ TIMEOUT")
                self.auto_status_label.setStyleSheet(
                    f"font-size: 30px; font-weight: 700; color: #D97706; text-align: center; margin: 16px 0;"
                )

            if hasattr(self, 'auto_reason_label'):
                self.auto_reason_label.setText(
                    "Verification timed out - the operation took too long. Please try again.")

            if hasattr(self, 'auto_tanker_info_label'):
                self.auto_tanker_info_label.setText("🚛 Verification timeout")

            # Reset UI states for manual verification
            if hasattr(self, 'manual_status_label'):
                self.manual_status_label.setText("⏰ TIMEOUT")
                self.manual_status_label.setStyleSheet(
                    f"font-size: 30px; font-weight: 700; color: #D97706; text-align: center; margin: 16px 0;"
                )

            if hasattr(self, 'manual_reason_label'):
                self.manual_reason_label.setText(
                    "Verification timed out - the operation took too long. Please try again.")

            if hasattr(self, 'manual_tanker_info_label'):
                self.manual_tanker_info_label.setText("🚛 Verification timeout")

            # Clear active operations
            if hasattr(self, 'active_operations'):
                self.active_operations.clear()

            # Update status bar
            self.statusBar().showMessage("Verification timed out - please try again")

            # Hide loading overlays
            for overlay in getattr(self, 'loading_overlays', {}).values():
                try:
                    overlay.hide_loading()
                except:
                    pass

        except Exception as e:
            logger.error(f"Error handling verification timeout: {e}")

    def verify_latest_tanker(self):
        """FIXED: Auto verification with timeout protection and Red Entry check"""
        try:
            # Start timeout timer (30 seconds)
            if hasattr(self, 'verification_timeout_timer'):
                self.verification_timeout_timer.stop()  # Stop any existing timer
                self.verification_timeout_timer.start(30000)  # 30 seconds

            # Immediate UI feedback
            self.auto_status_label.setText("⏳ Verifying...")
            self.auto_status_label.setStyleSheet(
                f"font-size: 30px; font-weight: 700; color: #2563EB; text-align: center; margin: 16px 0;"
            )
            self.auto_reason_label.setText("Checking latest entry for ban records and Red Entry duplicates...")
            self.auto_tanker_info_label.setText("🚛 Retrieving latest entry...")

            # Hide voice section initially
            if hasattr(self, 'auto_voice_frame'):
                self.auto_voice_frame.setVisible(False)

            QApplication.processEvents()

            # Generate operation ID and start verification
            operation_id = self.generate_operation_id()
            self.active_operations[operation_id] = 'verify_latest'

            logger.info("Starting auto verification for latest tanker with Red Entry check")

            # Add to database worker queue with timeout protection
            self.db_worker.add_operation('verify_tanker', operation_id, self.db, self.user_info['username'])

        except Exception as e:
            logger.error(f"Auto verification error: {e}")
            if hasattr(self, 'verification_timeout_timer'):
                self.verification_timeout_timer.stop()
            self.auto_status_label.setText("❌ ERROR")
            self.auto_reason_label.setText(f"Auto verification failed: {str(e)}")


    def on_verification_completed(self, status, reason, details):
        """Enhanced verification completion handler with timeout cleanup and Red Entry support"""
        try:
            # Stop timeout timer
            if hasattr(self, 'verification_timeout_timer'):
                self.verification_timeout_timer.stop()

            operation_type = None
            operation_id_to_remove = None

            # Find the operation type
            for op_id, op_type in list(self.active_operations.items()):
                if op_type in ['verify_latest', 'verify_manual']:
                    operation_type = op_type
                    operation_id_to_remove = op_id
                    break

            # Clean up operation
            if operation_id_to_remove:
                self.active_operations.pop(operation_id_to_remove, None)

            tanker_number = details.get("tanker_number", "UNKNOWN")

            # Update appropriate display
            if operation_type == 'verify_latest':
                self.update_auto_verification_display(tanker_number, status, reason, details)
                self.statusBar().showMessage(f"Auto verification: {tanker_number} - {status}")

            elif operation_type == 'verify_manual':
                self.update_manual_verification_display(tanker_number, status, reason, details)
                self.statusBar().showMessage(f"Manual verification: {tanker_number} - {status}")

            # Play sound for important statuses (including duplicates)
            if details.get("play_sound", False) or details.get("duplicate_detected", False):
                self.play_warning_sound_for_status(status)

            # Log the result
            if details.get("duplicate_detected"):
                logger.warning(f"RED ENTRY DUPLICATE DETECTED: {tanker_number} - {reason}")
            else:
                logger.info(f"Verification completed: {tanker_number} - {status}")

        except Exception as e:
            logger.error(f"Error handling verification completed: {e}")
            # Fallback status update
            if hasattr(self, 'verification_timeout_timer'):
                self.verification_timeout_timer.stop()
            self.statusBar().showMessage(f"Verification completed with errors: {str(e)}")

    def update_auto_verification_display(self, tanker_number, status, reason, details):
        """Enhanced auto verification display with Red Entry detection - NO HANGING"""
        try:
            self.auto_tanker_info_label.setText(f"🚛 Vehicle: {tanker_number}")
            self.auto_status_label.setText(status)

            # Enhanced handling for Red Entry duplicates
            if details.get("duplicate_detected"):
                # Special styling for duplicates
                color = "#DC2626"  # Red for duplicates
                self.auto_status_label.setStyleSheet(
                    f"font-size: 30px; font-weight: 700; color: {color}; text-align: center; "
                    f"margin: 16px 0; letter-spacing: -0.5px; "
                    f"background-color: rgba(220, 38, 38, 0.1); padding: 16px; "
                    f"border-radius: 8px; border: 2px solid rgba(220, 38, 38, 0.3);"
                )

                # Show Red Entry information
                red_entry = details.get("red_entry_record")
                if red_entry:
                    time_diff = red_entry.get("time_diff_minutes", 0)
                    enhanced_reason = (f"🚨 DUPLICATE ENTRY DETECTED\n\n{reason}\n\n"
                                       f"⏰ Last processed: {time_diff:.1f} minutes ago\n"
                                       f"🆔 Red Entry ID: {red_entry.get('id', 'N/A')}")
                    self.auto_reason_label.setText(enhanced_reason)
                else:
                    self.auto_reason_label.setText(f"🚨 DUPLICATE ENTRY DETECTED\n\n{reason}")
            else:
                # Normal styling for non-duplicates
                color = self.get_modern_status_color(status)
                self.auto_status_label.setStyleSheet(
                    f"font-size: 30px; font-weight: 700; color: {color}; text-align: center; "
                    f"margin: 16px 0; letter-spacing: -0.5px;"
                )
                self.auto_reason_label.setText(reason)

            # Handle voice notes (existing functionality)
            ban_record = details.get("ban_record")
            if ban_record and ban_record.get('voice_recording'):
                self.auto_voice_frame.setVisible(True)
                self.auto_play_voice_btn.setVisible(True)
                self.auto_voice_info_label.setText(f"🎵 Voice note: {ban_record.get('ban_reason', 'Available')}")
                self.auto_current_voice_data = ban_record.get('voice_recording')
            else:
                self.auto_voice_frame.setVisible(False)
                self.auto_current_voice_data = None

            logger.info(f"Auto verification display updated: {status} for {tanker_number}")

        except Exception as e:
            logger.error(f"Error updating auto verification display: {e}")
            # Fallback display
            self.auto_tanker_info_label.setText(f"🚛 Vehicle: {tanker_number}")
            self.auto_status_label.setText("ERROR")
            self.auto_reason_label.setText(f"Display error: {str(e)}")

    def update_manual_verification_display(self, tanker_number, status, reason, details):
        """Enhanced manual verification display with Red Entry detection - NO HANGING"""
        try:
            self.manual_tanker_info_label.setText(f"🚛 Vehicle: {tanker_number}")
            self.manual_status_label.setText(status)

            # Enhanced handling for Red Entry duplicates
            if details.get("duplicate_detected"):
                # Special styling for duplicates
                color = "#DC2626"  # Red for duplicates
                self.manual_status_label.setStyleSheet(
                    f"font-size: 30px; font-weight: 700; color: {color}; text-align: center; "
                    f"margin: 16px 0; letter-spacing: -0.5px; "
                    f"background-color: rgba(220, 38, 38, 0.1); padding: 16px; "
                    f"border-radius: 8px; border: 2px solid rgba(220, 38, 38, 0.3);"
                )

                # Show Red Entry information
                red_entry = details.get("red_entry_record")
                if red_entry:
                    time_diff = red_entry.get("time_diff_minutes", 0)
                    enhanced_reason = (f"🚨 DUPLICATE ENTRY DETECTED\n\n{reason}\n\n"
                                       f"⏰ Last processed: {time_diff:.1f} minutes ago\n"
                                       f"🆔 Red Entry ID: {red_entry.get('id', 'N/A')}")
                    self.manual_reason_label.setText(enhanced_reason)
                else:
                    self.manual_reason_label.setText(f"🚨 DUPLICATE ENTRY DETECTED\n\n{reason}")
            else:
                # Normal styling for non-duplicates
                color = self.get_modern_status_color(status)
                self.manual_status_label.setStyleSheet(
                    f"font-size: 30px; font-weight: 700; color: {color}; text-align: center; "
                    f"margin: 16px 0; letter-spacing: -0.5px;"
                )
                self.manual_reason_label.setText(reason)

            # Handle voice notes (existing functionality)
            ban_record = details.get("ban_record")
            if ban_record and ban_record.get('voice_recording'):
                self.manual_voice_frame.setVisible(True)
                self.manual_play_voice_btn.setVisible(True)
                self.manual_voice_info_label.setText(f"🎵 Voice note: {ban_record.get('ban_reason', 'Available')}")
                self.manual_current_voice_data = ban_record.get('voice_recording')
            else:
                self.manual_voice_frame.setVisible(False)
                self.manual_current_voice_data = None

            logger.info(f"Manual verification display updated: {status} for {tanker_number}")

        except Exception as e:
            logger.error(f"Error updating manual verification display: {e}")
            # Fallback display
            self.manual_tanker_info_label.setText(f"🚛 Vehicle: {tanker_number}")
            self.manual_status_label.setText("ERROR")
            self.manual_reason_label.setText(f"Display error: {str(e)}")

    def get_modern_status_color(self, status):
        """Enhanced status color coding with Red Entry support"""
        try:
            status_upper = status.upper()

            if "ALLOWED" in status_upper and "PERMISSION" not in status_upper and "WARNING" not in status_upper:
                return "#059669"  # Green for allowed
            elif any(word in status_upper for word in ["DENIED", "REJECTED", "DUPLICATE"]):  # NEW: Red for duplicates
                return "#DC2626"  # Red for denied/duplicates
            elif any(word in status_upper for word in ["PERMISSION", "CONDITIONAL", "WARNING"]):
                return "#D97706"  # Orange for warnings
            else:
                return "#2563EB"  # Blue for others
        except Exception as e:
            logger.error(f"Error getting status color: {e}")
            return "#2563EB"  # Default blue

    def play_warning_sound_for_status(self, status):
        """Enhanced warning sound with Red Entry duplicate detection"""
        try:
            if not getattr(self, 'sound_enabled', True) or not getattr(self, 'warning_sound', None):
                return

            # Play sound for bans, duplicates, and warnings
            status_upper = status.upper()
            should_play = (
                    "DENIED" in status_upper or
                    "REJECTED" in status_upper or
                    "PERMISSION" in status_upper or
                    "WARNING" in status_upper or
                    "DUPLICATE" in status_upper  # NEW: Play sound for Red Entry duplicates
            )

            if should_play and not getattr(self, 'current_sound_playing', False):
                self.current_sound_playing = True
                self.warning_sound.play(self.on_warning_sound_finished)
                logger.info(f"Warning sound played for status: {status}")
        except Exception as e:
            logger.error(f"Error playing warning sound: {e}")

    def on_warning_sound_finished(self, success, message):
        """Handle warning sound completion"""
        try:
            self.current_sound_playing = False
            if success:
                logger.debug("Warning sound completed successfully")
            else:
                logger.debug(f"Warning sound failed: {message}")
        except Exception as e:
            logger.error(f"Error in warning sound callback: {e}")
            self.current_sound_playing = False





    def apply_modern_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {ModernUITheme.BACKGROUND}; font-family: {ModernUITheme.FONT_FAMILY}; color: {ModernUITheme.TEXT_PRIMARY}; }}
            #modernStatusBar {{ background-color: {ModernUITheme.SURFACE}; border-top: 1px solid {ModernUITheme.BORDER_LIGHT}; color: {ModernUITheme.TEXT_SECONDARY}; font-size: {ModernUITheme.FONT_SIZE_SM}; padding: {ModernUITheme.SPACE_SM}; }}
            #modernSidebar {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {ModernUITheme.DARK_PRIMARY}, stop:1 {ModernUITheme.DARK_SECONDARY}); border-right: 1px solid {ModernUITheme.BORDER}; }}
            #titleContainer {{ background-color: rgba(255, 255, 255, 0.05); border-radius: {ModernUITheme.RADIUS_LG}; padding: {ModernUITheme.SPACE_LG}; margin-bottom: {ModernUITheme.SPACE_XL}; }}
            #sidebarTitle {{ font-size: {ModernUITheme.FONT_SIZE_2XL}; font-weight: 700; color: white; margin: 0; }}
            #sidebarSubtitle {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: rgba(255, 255, 255, 0.8); font-weight: 500; margin: 0; }}
            #navContainer {{ background-color: transparent; }}
            #navButtonContainer {{ background-color: transparent; border-radius: {ModernUITheme.RADIUS_MD}; border: none; margin: {ModernUITheme.SPACE_XS} 0; }}
            #navButtonContainer:hover {{ background-color: rgba(255, 255, 255, 0.1); transform: translateX(4px); }}
            #navIcon {{ font-size: {ModernUITheme.FONT_SIZE_LG}; color: white; min-width: 24px; }}
            #navText {{ font-size: {ModernUITheme.FONT_SIZE_BASE}; font-weight: 500; color: white; }}
            #soundContainer {{ background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: {ModernUITheme.RADIUS_MD}; margin: {ModernUITheme.SPACE_MD} 0; }}
            #soundHeader {{ font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 600; color: white; text-align: center; }}
            #soundToggle {{ background-color: {ModernUITheme.SUCCESS}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_SM}; font-size: {ModernUITheme.FONT_SIZE_XS}; font-weight: 600; padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_MD}; min-width: 50px; }}
            #soundToggle:!checked {{ background-color: {ModernUITheme.ERROR}; }}
            #soundStop {{ background-color: {ModernUITheme.WARNING}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_SM}; font-size: {ModernUITheme.FONT_SIZE_XS}; font-weight: 500; padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_MD}; }}
            #userContainer {{ background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: {ModernUITheme.RADIUS_MD}; }}
            #userName {{ font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 600; color: white; }}
            #userRoleAdmin {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: #FFB74D; font-weight: 500; }}
            #userRoleSupervisor {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: #81C784; font-weight: 500; }}
            #userRoleOperator {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: #64B5F6; font-weight: 500; }}
            #modernContentArea {{ background-color: {ModernUITheme.BACKGROUND}; }}
            #modernPage {{ background-color: transparent; }}
            #pageHeader {{ background-color: transparent; margin-bottom: {ModernUITheme.SPACE_XL}; }}
            #pageTitle {{ font-size: {ModernUITheme.FONT_SIZE_3XL}; font-weight: 700; color: {ModernUITheme.TEXT_PRIMARY}; margin: 0; letter-spacing: -0.5px; }}
            #pageSubtitle {{ font-size: {ModernUITheme.FONT_SIZE_LG}; color: {ModernUITheme.TEXT_SECONDARY}; font-weight: 400; margin: {ModernUITheme.SPACE_SM} 0 0 0; }}
            #refreshButton, #verifyButton, #addBanButton, #saveButton {{ background-color: {ModernUITheme.PRIMARY}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_BASE}; font-weight: 600; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL}; min-height: 44px; }}
            #refreshButton:hover, #verifyButton:hover, #addBanButton:hover, #saveButton:hover {{ background-color: {ModernUITheme.PRIMARY_DARK}; transform: translateY(-1px); }}
            #addBanButton {{ background-color: {ModernUITheme.ERROR}; }}
            #addBanButton:hover {{ background-color: #B91C1C; }}
            #addUserButton {{ background-color: {ModernUITheme.SUCCESS}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_BASE}; font-weight: 600; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL}; min-height: 44px; }}
            #addUserButton:hover {{ background-color: #047857; }}
            #browseButton, #testButton {{ background-color: {ModernUITheme.TEXT_SECONDARY}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 500; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; min-height: 40px; }}
            #browseButton:hover, #testButton:hover {{ background-color: #374151; }}
            #quickButton {{ background-color: {ModernUITheme.WARNING}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_SM}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 500; padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_MD}; min-height: 36px; }}
            #quickButton:hover {{ background-color: #B45309; }}
            #applyFilterButton {{ background-color: {ModernUITheme.PRIMARY}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 600; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; min-height: 40px; }}
            #applyFilterButton:hover {{ background-color: {ModernUITheme.PRIMARY_DARK}; }}
            #clearFilterButton {{ background-color: {ModernUITheme.TEXT_SECONDARY}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 500; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; min-height: 40px; }}
            #clearFilterButton:hover {{ background-color: #374151; }}
            #voicePlayButton {{ background-color: {ModernUITheme.SUCCESS}; color: white; border: none; border-radius: {ModernUITheme.RADIUS_MD}; font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 500; padding: {ModernUITheme.SPACE_SM} {ModernUITheme.SPACE_LG}; min-height: 36px; }}
            #voicePlayButton:hover {{ background-color: #047857; }}
            #filterContainer, #controlContainer, #inputContainer, #settingsContainer {{ background-color: {ModernUITheme.SURFACE}; border: 1px solid {ModernUITheme.BORDER_LIGHT}; border-radius: {ModernUITheme.RADIUS_LG}; box-shadow: {ModernUITheme.SHADOW_SM}; }}
            #resultContainer {{ background-color: {ModernUITheme.SURFACE}; border: 1px solid {ModernUITheme.BORDER_LIGHT}; border-radius: {ModernUITheme.RADIUS_XL}; box-shadow: {ModernUITheme.SHADOW_MD}; }}
            #voiceContainer {{ background-color: rgba(5, 150, 105, 0.1); border: 1px solid rgba(5, 150, 105, 0.2); border-radius: {ModernUITheme.RADIUS_MD}; }}
            #statsContainer {{ background-color: transparent; }}
            #filterHeader, #settingsHeader, #tableHeader, #statsHeader {{ font-size: {ModernUITheme.FONT_SIZE_XL}; font-weight: 600; color: {ModernUITheme.TEXT_PRIMARY}; margin: 0; }}
            #filterLabel, #settingsLabel, #inputLabel, #quickLabel {{ font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 600; color: {ModernUITheme.TEXT_SECONDARY}; margin: 0; }}
            #filterStatus, #countLabel {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: {ModernUITheme.TEXT_MUTED}; font-style: italic; margin: 0; }}
            #tankerInfoLabel {{ font-size: {ModernUITheme.FONT_SIZE_XL}; font-weight: 600; color: {ModernUITheme.TEXT_PRIMARY}; text-align: center; margin: 0; }}
            #statusDisplayLabel {{ font-size: {ModernUITheme.FONT_SIZE_3XL}; font-weight: 700; text-align: center; margin: {ModernUITheme.SPACE_LG} 0; letter-spacing: -0.5px; }}
            #reasonDisplayLabel {{ font-size: {ModernUITheme.FONT_SIZE_LG}; color: {ModernUITheme.TEXT_SECONDARY}; text-align: center; line-height: 1.5; margin: 0; }}
            #voiceInfoLabel {{ font-size: {ModernUITheme.FONT_SIZE_BASE}; color: {ModernUITheme.TEXT_SECONDARY}; font-weight: 500; }}
            #dateToLabel {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: {ModernUITheme.TEXT_MUTED}; font-weight: 500; margin: 0 {ModernUITheme.SPACE_SM}; }}
            #statusLabel {{ font-size: {ModernUITheme.FONT_SIZE_BASE}; color: {ModernUITheme.TEXT_SECONDARY}; font-weight: 500; padding: {ModernUITheme.SPACE_SM} 0; }}
            #settingsSubtext {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: {ModernUITheme.TEXT_MUTED}; margin: 0; }}
            #tankerInput, #filterInput, #pathInput {{ border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; background-color: {ModernUITheme.BACKGROUND}; color: {ModernUITheme.TEXT_PRIMARY}; font-weight: 500; min-height: 20px; }}
            #tankerInput:focus, #filterInput:focus, #pathInput:focus {{ border-color: {ModernUITheme.PRIMARY}; outline: none; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }}
            #modernDateEdit, #modernCombo {{ border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; background-color: {ModernUITheme.BACKGROUND}; color: {ModernUITheme.TEXT_PRIMARY}; font-weight: 500; min-height: 20px; }}
            #modernDateEdit:focus, #modernCombo:focus {{ border-color: {ModernUITheme.PRIMARY}; outline: none; }}
            #modernSpinBox {{ border: 2px solid {ModernUITheme.BORDER}; border-radius: {ModernUITheme.RADIUS_MD}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; font-size: {ModernUITheme.FONT_SIZE_BASE}; background-color: {ModernUITheme.BACKGROUND}; color: {ModernUITheme.TEXT_PRIMARY}; font-weight: 500; min-height: 20px; }}
            #modernSpinBox:focus {{ border-color: {ModernUITheme.PRIMARY}; outline: none; }}
            #modernTable {{ border: 1px solid {ModernUITheme.BORDER_LIGHT}; border-radius: {ModernUITheme.RADIUS_LG}; background-color: {ModernUITheme.BACKGROUND}; gridline-color: {ModernUITheme.BORDER_LIGHT}; selection-background-color: rgba(37, 99, 235, 0.1); font-size: {ModernUITheme.FONT_SIZE_SM}; font-weight: 500; }}
            #modernTable::item {{ padding: {ModernUITheme.SPACE_MD}; border-bottom: 1px solid {ModernUITheme.BORDER_LIGHT}; }}
            #modernTable::item:selected {{ background-color: rgba(37, 99, 235, 0.1); color: {ModernUITheme.TEXT_PRIMARY}; }}
            #modernTable QHeaderView::section {{ background-color: {ModernUITheme.SURFACE}; color: {ModernUITheme.TEXT_SECONDARY}; font-weight: 600; font-size: {ModernUITheme.FONT_SIZE_SM}; padding: {ModernUITheme.SPACE_MD}; border: none; border-bottom: 2px solid {ModernUITheme.BORDER}; border-right: 1px solid {ModernUITheme.BORDER_LIGHT}; }}
            #modernTable QHeaderView::section:first {{ border-top-left-radius: {ModernUITheme.RADIUS_LG}; }}
            #modernTable QHeaderView::section:last {{ border-top-right-radius: {ModernUITheme.RADIUS_LG}; border-right: none; }}
            #modernTabWidget {{ background-color: transparent; }}
            #modernTabWidget::pane {{ border: 1px solid {ModernUITheme.BORDER_LIGHT}; border-radius: {ModernUITheme.RADIUS_LG}; background-color: {ModernUITheme.BACKGROUND}; margin-top: 8px; }}
            #modernTabWidget QTabBar::tab {{ background-color: {ModernUITheme.SURFACE}; color: {ModernUITheme.TEXT_SECONDARY}; border: 1px solid {ModernUITheme.BORDER_LIGHT}; padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_2XL}; margin-right: {ModernUITheme.SPACE_XS}; border-radius: {ModernUITheme.RADIUS_MD} {ModernUITheme.RADIUS_MD} 0 0; font-weight: 500; font-size: {ModernUITheme.FONT_SIZE_BASE}; }}
            #modernTabWidget QTabBar::tab:selected {{ background-color: {ModernUITheme.PRIMARY}; color: white; border-color: {ModernUITheme.PRIMARY}; font-weight: 600; }}
            #modernTabWidget QTabBar::tab:hover:!selected {{ background-color: rgba(37, 99, 235, 0.1); color: {ModernUITheme.PRIMARY}; }}
            #modernTabPage {{ background-color: transparent; }}
            #statCard {{ background-color: {ModernUITheme.SURFACE}; border: 1px solid {ModernUITheme.BORDER_LIGHT}; border-radius: {ModernUITheme.RADIUS_LG}; box-shadow: {ModernUITheme.SHADOW_SM}; }}
            #valueLabel {{ font-size: {ModernUITheme.FONT_SIZE_3XL}; font-weight: 700; color: {ModernUITheme.PRIMARY}; }}
            #descLabel {{ font-size: {ModernUITheme.FONT_SIZE_SM}; color: {ModernUITheme.TEXT_SECONDARY}; font-weight: 500; }}
            QScrollBar:vertical {{ background-color: {ModernUITheme.SURFACE}; width: 12px; border-radius: 6px; margin: 0; }}
            QScrollBar::handle:vertical {{ background-color: {ModernUITheme.TEXT_MUTED}; border-radius: 6px; min-height: 20px; margin: 2px; }}
            QScrollBar::handle:vertical:hover {{ background-color: {ModernUITheme.TEXT_SECONDARY}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{ background-color: {ModernUITheme.SURFACE}; height: 12px; border-radius: 6px; margin: 0; }}
            QScrollBar::handle:horizontal {{ background-color: {ModernUITheme.TEXT_MUTED}; border-radius: 6px; min-width: 20px; margin: 2px; }}
            QScrollBar::handle:horizontal:hover {{ background-color: {ModernUITheme.TEXT_SECONDARY}; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

        """)
        # ADD THESE NEW STYLES:
        additional_styles = f"""
                #logoutContainer {{ 
                    background-color: rgba(255, 255, 255, 0.05); 
                    border: 1px solid rgba(255, 255, 255, 0.1); 
                    border-radius: {ModernUITheme.RADIUS_MD}; 
                }}
                #logoutButton {{ 
                    background-color: {ModernUITheme.ERROR}; 
                    color: white; 
                    border: none; 
                    border-radius: {ModernUITheme.RADIUS_MD}; 
                    font-weight: 600; 
                    padding: {ModernUITheme.SPACE_MD} {ModernUITheme.SPACE_LG}; 
                    min-height: 40px; 
                }}
                #logoutButton:hover {{ 
                    background-color: #B91C1C; 
                }}
            """

        # Add to existing stylesheet
        current_style = self.styleSheet()
        self.setStyleSheet(current_style + additional_styles)

    def closeEvent(self, event):
        """Enhanced shutdown with better cleanup"""
        try:
            logger.info(f"TDF System shutdown initiated - User: {self.user_info['username']}")

            # Stop all audio operations first
            if hasattr(self, 'audio_recorder') and self.audio_recorder:
                try:
                    self.audio_recorder.stop_playback()
                    if hasattr(self.audio_recorder, 'recording') and self.audio_recorder.recording:
                        self.audio_recorder.recording = False
                except Exception as e:
                    logger.debug(f"Error stopping audio: {e}")

            # Stop database worker
            if hasattr(self, 'db_worker') and self.db_worker:
                logger.info("Stopping database worker thread...")
                self.db_worker.stop_thread()

            # Enhanced database cleanup - NEW SECTION
            if hasattr(self, 'db'):
                try:
                    # Check if using enhanced database manager
                    if hasattr(self.db, 'cleanup'):
                        logger.info("Cleaning up enhanced database connections...")
                        self.db.cleanup()
                    else:
                        logger.info("Using standard database cleanup...")
                        # For standard DatabaseManager, just log
                        pass
                except Exception as e:
                    logger.debug(f"Database cleanup error (non-critical): {e}")

            # Hide all overlays
            for overlay in self.loading_overlays.values():
                try:
                    overlay.hide_loading()
                except:
                    pass

            # Stop timers
            if hasattr(self, 'monitor_timer') and self.monitor_timer:
                self.monitor_timer.stop()
            if hasattr(self, 'dashboard_timer') and self.dashboard_timer:
                self.dashboard_timer.stop()

            # Audio recorder cleanup (consolidated)
            if hasattr(self, 'audio_recorder') and self.audio_recorder:
                try:
                    self.audio_recorder.cleanup()
                except Exception as e:
                    logger.debug(f"Audio recorder cleanup error: {e}")

            # Process any remaining events
            QApplication.processEvents()

            logger.info(f"Enhanced TDF System shutdown completed - User: {self.user_info['username']}")
            event.accept()

        except Exception as e:
            logger.error(f"Error during enhanced shutdown: {e}")
            event.accept()  # Accept anyway to avoid hanging
def main():
    """Main entry point with modern UI and enhanced functionality"""
    try:
        logger.info("TDF SYSTEM - MODERN UI VERSION WITH ENHANCED FUNCTIONALITY STARTING")
        logger.info(
            "Features: Modern Professional Interface, Fixed Filters, Role-based Access Control, Enhanced Accessibility")
        if not PYODBC_AVAILABLE:
            error_msg = "pyodbc not available. Install with: pip install pyodbc"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            QMessageBox.critical(None, "Missing Dependency",
                                 f"Required library missing:\n\n{error_msg}\n\nPlease install pyodbc and restart the application.")
            return 1


        # Create application with modern settings
        # ADD THESE LINES BEFORE QApplication
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

        app.setApplicationName("TDF System - Modern Professional Interface")
        app.setStyle("Fusion")
        # Set consistent font for all systems
        font = QFont("Arial", 9)
        app.setFont(font)
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
            test_db = EnhancedDatabaseManager(config_manager)
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

                    default_users = [
                        ("admin", admin_password, "System Administrator", "admin", "system"),
                        ("operator", operator_password, "Default Operator", "operator", "system"),
                        ("supervisor", supervisor_password, "Default Supervisor", "supervisor", "system")
                    ]

                    cursor.executemany("""
                        INSERT INTO users (username, password_hash, full_name, role, created_by)
                        VALUES (?, ?, ?, ?, ?)
                    """, default_users)

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

            # Start database worker thread
            if hasattr(window, 'db_worker'):
                window.db_worker.start()
                logger.info("Database worker thread started")

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