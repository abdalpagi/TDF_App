"""
TDF System - Enhanced Version with Modern UI, Fixed Filters and Role-Based Access Control
Features: Modern Professional UI, Fixed filter functionality, Role-based permissions, Database path configuration, User account management
Updated: Modern typography, enhanced visual design, improved accessibility and readability
"""
# coderabbitai Request Full UI Redesign – Login & Add Ban Pages + Overall Stylin

import sys
# Modern imports for enhanced functionality
import os
import logging
import threading
import json
import hashlib
from contextlib import contextmanager, suppress
from dataclasses import dataclass

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
    AUDIO_AVAILABLE = True
except ImportError:
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
    "server_db_path": r"\\192.168.137.59\rfid\TruckAutoId.mdb",
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
    DANGER = "#DC2626"            # Red for errors
    SURFACE = "#F3F4F6"           # Light gray for surfaces
    BORDER_LIGHT = "#E5E7EB"      # Light border color
    WARNING = "#F59E0B"           # Amber for warnings
    SUCCESS = "#10B981"           # Green for success

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
        # Method body unchanged...
        pass

    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    # Remaining methods unchanged...

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
        # Method body unchanged...
        pass

    def test_server_connection(self):
        """Test server connection - ENSURE THIS METHOD EXISTS"""
        try:
            if not PYODBC_AVAILABLE:
                self.server_available = False
                logger.warning("pyodbc not available for server connection")
                return

            with self._connection_lock, self.get_server_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM VehicleMaster")
                count = cursor.fetchone()[0]
                self.server_available = True
                logger.info(f"Server connected - {count} vehicles in database")
        except Exception as e:
            self.server_available = False
            logger.warning(f"Server connection failed: {e}")

    def init_sqlite(self):
        # Method body unchanged...
        pass

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
        except Exception:
            self.server_available = False
            raise
        finally:
            if connection:
                with suppress(BaseException):
                    connection.close()

    # Remaining methods unchanged...