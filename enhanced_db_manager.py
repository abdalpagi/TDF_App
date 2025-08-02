"""
Enhanced Database Manager - enhanced_db_manager.py
Network-optimized database manager for TDF System with Red Entry Check
COMPLETE VERSION with all required methods and Red Entry functionality
"""

import sqlite3
import time
import threading
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta

import logging
import os
import time
from contextlib import contextmanager


# Keep pyodbc for server Access database
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedDatabaseManager:
    """Enhanced database manager with network share support and Red Entry checking"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.server_db = self.config.get('server_db_path')
        self.sqlite_db = self.config.get('local_sqlite_path')

        # Enhanced connection settings
        self.connection_timeout = self.config.get('connection_timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)

        self.server_available = False
        self._connection_lock = threading.RLock()

        # FIXED: Better connection strings
        self.connection_strings = self._build_robust_connection_strings()

        # Red Entry configuration
        self.red_entry_enabled = self.config.get('red_entry_check_enabled', True)
        self.red_entry_table = self.config.get('red_entry_table_name', 'tblVehicle_RedEntry')
        self.red_entry_reg_column = self.config.get('red_entry_reg_column', 'VRE_RegNo')
        self.red_entry_date_column = self.config.get('red_entry_date_column', 'VRE_InDate')
        self.red_entry_time_column = self.config.get('red_entry_time_column', 'VRE_InTime')
        self.red_entry_id_column = self.config.get('red_entry_id_column', 'VRE_ID')
        self.time_match_margin = self.config.get('time_match_margin_minutes', 5)

        # Initialize databases with safer approach
        self.safe_test_server_connection()
        self.init_ban_records_table()

        logger.info("Enhanced Database Manager with Red Entry checking initialized")

    def _build_robust_connection_strings(self):
        """Build multiple connection string options for better compatibility"""
        if not self.server_db:
            return []

        connection_strings = []

        # Modern driver attempts
        drivers_to_try = [
            "Microsoft Access Driver (*.mdb, *.accdb)",
            "Microsoft Access Driver (*.mdb)",
            "Access ODBC Driver"
        ]

        for driver in drivers_to_try:
            # Basic connection
            basic_conn = f"Driver={{{driver}}};DBQ={self.server_db};"
            connection_strings.append(basic_conn)

            # Enhanced connection with parameters
            enhanced_conn = f"Driver={{{driver}}};DBQ={self.server_db};ExtendedAnsiSQL=1;CharacterSet=UTF8;"
            connection_strings.append(enhanced_conn)

        return connection_strings

    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute operation with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if retryable
                retryable_errors = ['timeout', 'network', 'connection', 'locked', 'busy', 'i/o error']
                is_retryable = any(err in error_str for err in retryable_errors)

                if attempt < self.max_retries and is_retryable:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Database operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Database operation failed after {attempt + 1} attempts: {e}")
                    break

        raise last_exception

    @contextmanager
    def get_sqlite_connection(self):
        """Get SQLite connection with network optimizations"""
        connection = None
        try:
            def create_connection():
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.sqlite_db), exist_ok=True)

                conn = sqlite3.connect(
                    self.sqlite_db,
                    timeout=self.connection_timeout,
                    check_same_thread=False
                )

                # Network optimizations for SQLite
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout

                return conn

            connection = self.execute_with_retry(create_connection)
            yield connection

        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass

    @contextmanager
    def get_server_connection(self):
        """FIXED: Get pyodbc connection to Access server database"""
        connection = None
        try:
            if not PYODBC_AVAILABLE:
                raise Exception("pyodbc not available")

            # Try each connection string
            last_error = None
            for conn_str in self.connection_strings:
                try:
                    connection = pyodbc.connect(
                        conn_str,
                        timeout=min(self.connection_timeout, 10),  # Max 10 seconds
                        autocommit=False
                    )
                    self.server_available = True
                    logger.debug(f"Connection successful with: {conn_str}")
                    break
                except Exception as e:
                    last_error = e
                    logger.debug(f"Connection failed with {conn_str}: {e}")
                    continue

            if not connection:
                self.server_available = False
                raise Exception(f"All connection attempts failed. Last error: {last_error}")

            yield connection

        except Exception as e:
            self.server_available = False
            logger.error(f"Server connection error: {e}")
            raise
        finally:
            if connection:
                try:
                    connection.close()
                except Exception as close_error:
                    logger.debug(f"Error closing connection: {close_error}")

    def safe_test_server_connection(self):
        """SAFE: Test Microsoft Access server database connection"""
        try:
            if not PYODBC_AVAILABLE:
                self.server_available = False
                logger.warning("pyodbc not available - Access server database disabled")
                return

            if not self.server_db or not os.path.exists(self.server_db):
                self.server_available = False
                logger.warning(f"Database file not found: {self.server_db}")
                return

            with self.get_server_connection() as conn:
                cursor = conn.cursor()

                # Test basic connection first
                cursor.execute("SELECT 1 as test_value")
                result = cursor.fetchone()

                if result and result[0] == 1:
                    logger.info("Basic database connection test passed")

                    # Try to get vehicle count
                    try:
                        cursor.execute("SELECT COUNT(*) FROM VehicleMaster")
                        count = cursor.fetchone()[0]
                        self.server_available = True
                        logger.info(f"Access server connected - {count} vehicles in database")
                    except Exception as table_error:
                        # Connection works but tables might be different
                        self.server_available = True
                        logger.warning(f"Connected but table structure different: {table_error}")
                else:
                    self.server_available = False
                    logger.warning("Database connection test failed")

        except Exception as e:
            self.server_available = False
            logger.warning(f"Server connection failed: {e}")

    def check_red_entry_duplicate(self, tanker_number, entry_date=None, entry_time=None):
        """
        ROBUST Red Entry duplicate checking with Access database corruption handling
        Returns: (is_duplicate, red_entry_record, message)
        """
        if not self.red_entry_enabled:
            logger.debug("Red Entry checking is disabled")
            return False, None, "Red Entry check disabled"

        if not self.server_available:
            logger.warning("Server not available for Red Entry check")
            return False, None, "Server unavailable for Red Entry check"

        try:
            with self.get_server_connection() as conn:
                cursor = conn.cursor()

                # OPTIMIZED QUERY: Get recent entries with better error handling
                try:
                    # Query with date filtering to avoid very old corrupted records
                    query = f"""
                        SELECT TOP 10 {self.red_entry_id_column}, {self.red_entry_reg_column}, 
                               {self.red_entry_date_column}, {self.red_entry_time_column}
                        FROM {self.red_entry_table} 
                        WHERE {self.red_entry_reg_column} = ?
                        AND {self.red_entry_date_column} > DateAdd('d', -2, Now())
                        ORDER BY {self.red_entry_id_column} DESC
                    """

                    cursor.execute(query, (tanker_number,))
                    results = cursor.fetchall()

                except Exception as query_error:
                    logger.warning(f"Date-filtered query failed, trying simple query: {query_error}")
                    # Fallback to simple query without date filtering
                    simple_query = f"""
                        SELECT TOP 5 {self.red_entry_id_column}, {self.red_entry_reg_column}, 
                               {self.red_entry_date_column}, {self.red_entry_time_column}
                        FROM {self.red_entry_table} 
                        WHERE {self.red_entry_reg_column} = ?
                        ORDER BY {self.red_entry_id_column} DESC
                    """
                    cursor.execute(simple_query, (tanker_number,))
                    results = cursor.fetchall()

                if not results:
                    logger.debug(f"No Red Entry records found for tanker {tanker_number}")
                    return False, None, "No Red Entry records found"

                # Parse current entry time
                try:
                    if entry_date and entry_time:
                        current_datetime_str = f"{entry_date} {entry_time}"
                        current_datetime = datetime.strptime(current_datetime_str, "%Y-%m-%d %H:%M:%S")
                    else:
                        current_datetime = datetime.now()
                        logger.debug("Using current time for Red Entry comparison")
                except Exception as e:
                    logger.error(f"Error parsing entry datetime: {e}")
                    current_datetime = datetime.now()

                # Check each Red Entry record with ROBUST parsing
                successful_parses = 0
                for record in results:
                    red_id, red_reg, red_date, red_time = record

                    # ROBUST datetime parsing for Access database corruption
                    red_datetime = self._parse_access_datetime_robust(red_date, red_time, red_id)

                    if red_datetime is None:
                        logger.debug(f"Skipping Red Entry {red_id} due to unparseable datetime")
                        continue

                    successful_parses += 1

                    # Calculate time difference
                    try:
                        time_diff = abs((current_datetime - red_datetime).total_seconds() / 60)  # minutes

                        logger.info(
                            f"Red Entry check: {tanker_number} - Current: {current_datetime}, Red: {red_datetime}, Diff: {time_diff:.1f} min")

                        # Check if within time margin
                        if time_diff <= self.time_match_margin:
                            logger.warning(
                                f"DUPLICATE DETECTED: {tanker_number} within {time_diff:.1f} minutes of Red Entry")

                            return True, {
                                'id': red_id,
                                'reg_no': red_reg,
                                'date': red_date,
                                'time': red_time,
                                'parsed_datetime': red_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                'time_diff_minutes': time_diff
                            }, f"Duplicate entry detected - vehicle processed {time_diff:.1f} minutes ago"

                    except Exception as time_calc_error:
                        logger.error(f"Time calculation error for Red Entry {red_id}: {time_calc_error}")
                        continue

                if successful_parses == 0:
                    logger.warning(f"No parseable Red Entry records found for {tanker_number}")
                    return False, None, "No parseable Red Entry records found"
                else:
                    logger.info(
                        f"No duplicate Red Entry found for {tanker_number} within {self.time_match_margin} minutes (checked {successful_parses} records)")
                    return False, None, f"No recent duplicate entries found (checked {successful_parses} records)"

        except Exception as e:
            logger.error(f"Error checking Red Entry for {tanker_number}: {e}")
            # Log the error but don't block verification
            return False, None, f"Red Entry check failed: {str(e)}"

    def _parse_access_datetime_robust(self, red_date, red_time, red_id=None):
        """
        ROBUST Access database datetime parser
        Handles various Access datetime corruption issues
        """
        try:
            # Convert inputs to strings for processing
            date_str = str(red_date).strip() if red_date is not None else ""
            time_str = str(red_time).strip() if red_time is not None else ""

            # Skip obviously invalid data
            if not date_str or not time_str or date_str == "None" or time_str == "None":
                logger.debug(f"Red Entry {red_id}: Empty date/time data")
                return None

            # PATTERN 1: Handle Access OLE Automation date corruption (1899-12-xx)
            if "1899-12-" in date_str or "1899-12-" in time_str:
                logger.debug(f"Red Entry {red_id}: Detected OLE Automation date corruption, using current time")
                # For recent entries with corrupted dates, assume they're very recent
                return datetime.now() - timedelta(minutes=1)

            # PATTERN 2: Extract valid date/time from mixed data
            try:
                # Clean up the strings - remove extra whitespace and split by spaces
                date_parts = date_str.split()
                time_parts = time_str.split()

                # Extract the actual date and time components
                clean_date = date_parts[0] if date_parts else ""
                clean_time = time_parts[0] if time_parts else ""

                # Validate format patterns
                if len(clean_date) >= 10 and len(clean_time) >= 5:
                    # Extract YYYY-MM-DD from date
                    if "-" in clean_date:
                        date_components = clean_date[:10]  # Take first 10 characters
                    else:
                        logger.debug(f"Red Entry {red_id}: Invalid date format: {clean_date}")
                        return None

                    # Extract HH:MM:SS from time
                    if ":" in clean_time:
                        time_components = clean_time[:8]  # Take first 8 characters (HH:MM:SS)
                        # Ensure it has seconds
                        if time_components.count(":") == 1:
                            time_components += ":00"
                    else:
                        logger.debug(f"Red Entry {red_id}: Invalid time format: {clean_time}")
                        return None

                    # Combine and parse
                    datetime_str = f"{date_components} {time_components}"
                    parsed_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

                    # Sanity check: reject dates too far in the past or future
                    now = datetime.now()
                    age_days = abs((now - parsed_datetime).days)

                    if age_days > 365:  # More than 1 year old/future
                        logger.debug(f"Red Entry {red_id}: Date too old/future ({age_days} days): {parsed_datetime}")
                        return None

                    logger.debug(f"Red Entry {red_id}: Successfully parsed datetime: {parsed_datetime}")
                    return parsed_datetime

            except ValueError as parse_error:
                logger.debug(f"Red Entry {red_id}: Standard parsing failed: {parse_error}")

            # PATTERN 3: Handle Access datetime objects directly
            try:
                if hasattr(red_date, 'strftime') and hasattr(red_time, 'strftime'):
                    date_part = red_date.strftime("%Y-%m-%d")
                    time_part = red_time.strftime("%H:%M:%S")

                    combined_str = f"{date_part} {time_part}"
                    parsed = datetime.strptime(combined_str, "%Y-%m-%d %H:%M:%S")

                    # Sanity check
                    age_days = abs((datetime.now() - parsed).days)
                    if age_days <= 365:
                        logger.debug(f"Red Entry {red_id}: Parsed from datetime objects: {parsed}")
                        return parsed

            except Exception as datetime_obj_error:
                logger.debug(f"Red Entry {red_id}: Datetime object parsing failed: {datetime_obj_error}")

            # PATTERN 4: Try to extract datetime using regex
            try:
                import re
                combined = f"{date_str} {time_str}"

                # Look for YYYY-MM-DD HH:MM:SS pattern anywhere in the string
                datetime_pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2}:\d{2})'
                match = re.search(datetime_pattern, combined)

                if match:
                    extracted_datetime = f"{match.group(1)} {match.group(2)}"
                    parsed = datetime.strptime(extracted_datetime, "%Y-%m-%d %H:%M:%S")

                    # Sanity check
                    age_days = abs((datetime.now() - parsed).days)
                    if age_days <= 365:
                        logger.debug(f"Red Entry {red_id}: Extracted via regex: {parsed}")
                        return parsed

            except Exception as regex_error:
                logger.debug(f"Red Entry {red_id}: Regex extraction failed: {regex_error}")

            # PATTERN 5: Last resort - if this is a very recent query, assume recent entry
            logger.warning(f"Red Entry {red_id}: All parsing failed, using approximation")
            logger.warning(f"Red Entry {red_id}: Raw data - Date: '{date_str}', Time: '{time_str}'")

            # For very recent entries that we can't parse, assume they're within the last few minutes
            # This helps catch recent duplicates even with corrupted timestamps
            return datetime.now() - timedelta(minutes=2)

        except Exception as e:
            logger.error(f"Red Entry {red_id}: Critical parsing error: {e}")
            return None

    def test_red_entry_parsing(self, sample_tanker="TEST123"):
        """
        Test Red Entry parsing with a specific tanker
        """
        try:
            logger.info(f"=== TESTING RED ENTRY PARSING FOR {sample_tanker} ===")

            if not self.server_available:
                return False, "Server not available"

            with self.get_server_connection() as conn:
                cursor = conn.cursor()

                # Get some sample records for testing
                query = f"""
                    SELECT TOP 5 {self.red_entry_id_column}, {self.red_entry_reg_column}, 
                           {self.red_entry_date_column}, {self.red_entry_time_column}
                    FROM {self.red_entry_table} 
                    ORDER BY {self.red_entry_id_column} DESC
                """

                cursor.execute(query)
                results = cursor.fetchall()

                if not results:
                    return False, "No Red Entry records found to test"

                successful_parses = 0
                total_records = len(results)
                parsing_results = []

                for record in results:
                    red_id, red_reg, red_date, red_time = record

                    # Test parsing
                    parsed_datetime = self._parse_access_datetime_robust(red_date, red_time, red_id)

                    if parsed_datetime:
                        successful_parses += 1
                        status = "✅ SUCCESS"
                    else:
                        status = "❌ FAILED"

                    parsing_results.append({
                        'id': red_id,
                        'reg': red_reg,
                        'raw_date': str(red_date),
                        'raw_time': str(red_time),
                        'parsed': parsed_datetime.strftime("%Y-%m-%d %H:%M:%S") if parsed_datetime else "FAILED",
                        'status': status
                    })

                # Create detailed report
                success_rate = (successful_parses / total_records * 100) if total_records > 0 else 0

                report = f"""Red Entry Parsing Test Results:

    Success Rate: {successful_parses}/{total_records} ({success_rate:.1f}%)

    Sample Records:
    """
                for result in parsing_results:
                    report += f"""
    ID {result['id']} ({result['reg']}): {result['status']}
      Raw Date: {result['raw_date']}
      Raw Time: {result['raw_time']}
      Parsed: {result['parsed']}
    """

                logger.info(f"Red Entry parsing test: {success_rate:.1f}% success rate")
                return True, report

        except Exception as e:
            error_msg = f"Red Entry parsing test failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    # ADD: Configuration method for Red Entry time margin
    def adjust_red_entry_time_margin(self, new_margin_minutes):
        """
        Adjust the Red Entry time margin for duplicate detection
        """
        try:
            old_margin = self.time_match_margin
            self.time_match_margin = new_margin_minutes

            # Update configuration
            self.config.set('time_match_margin_minutes', new_margin_minutes)
            self.config.save_config()

            logger.info(f"Red Entry time margin changed from {old_margin} to {new_margin_minutes} minutes")
            return True, f"Time margin updated to {new_margin_minutes} minutes"

        except Exception as e:
            error_msg = f"Failed to update time margin: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    # ADD: Method to get Red Entry statistics
    def get_red_entry_statistics(self):
        """
        Get statistics about Red Entry table
        """
        try:
            if not self.server_available:
                return None, "Server not available"

            with self.get_server_connection() as conn:
                cursor = conn.cursor()

                # Get total count
                cursor.execute(f"SELECT COUNT(*) FROM {self.red_entry_table}")
                total_count = cursor.fetchone()[0]

                # Get recent count (last 24 hours)
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {self.red_entry_table}
                    WHERE {self.red_entry_date_column} > DateAdd('d', -1, Now())
                """)
                recent_count = cursor.fetchone()[0]

                # Test parsing on recent records
                cursor.execute(f"""
                    SELECT TOP 20 {self.red_entry_id_column}, {self.red_entry_date_column}, {self.red_entry_time_column}
                    FROM {self.red_entry_table}
                    ORDER BY {self.red_entry_id_column} DESC
                """)
                sample_records = cursor.fetchall()

                parseable_count = 0
                for record in sample_records:
                    red_id, red_date, red_time = record
                    if self._parse_access_datetime_robust(red_date, red_time, red_id):
                        parseable_count += 1

                parse_success_rate = (parseable_count / len(sample_records) * 100) if sample_records else 0

                stats = {
                    'total_records': total_count,
                    'recent_records_24h': recent_count,
                    'sample_tested': len(sample_records),
                    'parseable_records': parseable_count,
                    'parse_success_rate': parse_success_rate,
                    'time_margin_minutes': self.time_match_margin,
                    'enabled': self.red_entry_enabled
                }

                return stats, "Statistics retrieved successfully"

        except Exception as e:
            error_msg = f"Failed to get Red Entry statistics: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    def verify_specific_tanker_no_red_entry(self, tanker_number, operator="System"):
        """
        BACKUP METHOD: Verification without Red Entry (when Red Entry is problematic)
        """
        logger.info(f"=== VERIFICATION WITHOUT RED ENTRY ===")
        logger.info(f"Tanker: {tanker_number}, Operator: {operator}")

        tanker_number_clean = str(tanker_number).strip().upper()

        try:
            # Quick ban record check only
            with sqlite3.connect(self.sqlite_db, timeout=5) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, ban_reason, ban_type, voice_recording, voice_filename
                    FROM ban_records 
                    WHERE UPPER(tanker_number) = UPPER(?) 
                    AND is_active = 1
                    AND (end_date IS NULL OR date(end_date) >= date('now'))
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (tanker_number_clean,))

                ban_record = cursor.fetchone()

                if ban_record:
                    ban_id, ban_reason, ban_type, voice_data, voice_filename = ban_record

                    status_map = {
                        "permanent": ("DENIED", f"PERMANENT BAN: {ban_reason}", "red", True),
                        "temporary": ("DENIED", f"TEMPORARY BAN: {ban_reason}", "red", True),
                        "permission": ("ALLOWED_WITH_PERMISSION", f"PERMISSION REQUIRED: {ban_reason}", "orange", True),
                        "reminder": ("ALLOWED_WITH_WARNING", f"REMINDER: {ban_reason}", "orange", True)
                    }

                    status, reason, decision_color, play_sound = status_map.get(ban_type, (
                        "DENIED", f"BAN ACTIVE: {ban_reason}", "red", True))

                    # Log verification
                    cursor.execute("""
                        INSERT INTO logs (tanker_number, status, reason, operator)
                        VALUES (?, ?, ?, ?)
                    """, (tanker_number_clean, f"{status}_NO_RED_ENTRY", f"No Red Entry check - {reason}", operator))
                    conn.commit()

                    return status, reason, {
                        "tanker_number": tanker_number_clean,
                        "ban_record": {
                            'id': ban_id,
                            'ban_reason': ban_reason,
                            'ban_type': ban_type,
                            'voice_recording': voice_data,
                            'voice_filename': voice_filename
                        },
                        "red_entry_record": None,
                        "decision_color": decision_color,
                        "play_sound": play_sound,
                        "duplicate_detected": False
                    }
                else:
                    # No ban - ALLOWED
                    status = "ALLOWED"
                    reason = f"Vehicle {tanker_number_clean} verified (no Red Entry check) - Access granted"

                    cursor.execute("""
                        INSERT INTO logs (tanker_number, status, reason, operator)
                        VALUES (?, ?, ?, ?)
                    """, (tanker_number_clean, f"{status}_NO_RED_ENTRY", reason, operator))
                    conn.commit()

                    return status, reason, {
                        "tanker_number": tanker_number_clean,
                        "ban_record": None,
                        "red_entry_record": None,
                        "decision_color": "green",
                        "play_sound": False,
                        "duplicate_detected": False
                    }

        except Exception as e:
            logger.error(f"Verification without Red Entry failed for {tanker_number_clean}: {e}")
            return "ERROR", f"Verification failed: {str(e)}", {
                "error": True,
                "tanker_number": tanker_number_clean,
                "duplicate_detected": False
            }
    def get_latest_tanker_from_server(self):
        """FIXED: Get latest tanker entry from Access server database"""
        try:
            if not self.server_available:
                logger.debug("Server not available for latest tanker query")
                return None

            with self.get_server_connection() as conn:
                cursor = conn.cursor()

                # Try primary query
                try:
                    query = """
                       SELECT TOP 1 VT_RegNo, VT_InDate, VT_InTime, VT_ID 
                       FROM VehicleTransactions 
                       ORDER BY VT_ID DESC
                       """
                    cursor.execute(query)
                    result = cursor.fetchone()

                    if result:
                        from main import TankerData
                        return TankerData(
                            tanker_number=str(result[0]).strip(),
                            entry_date=str(result[1]),
                            entry_time=str(result[2])
                        )

                except Exception as table_error:
                    logger.debug(f"VehicleTransactions table query failed: {table_error}")

                    # Try alternative table names
                    alternative_queries = [
                        "SELECT TOP 1 RegNo, InDate, InTime, ID FROM Transactions ORDER BY ID DESC",
                        "SELECT TOP 1 Vehicle_RegNo, Entry_Date, Entry_Time, Transaction_ID FROM Vehicle_Transactions ORDER BY Transaction_ID DESC"
                    ]

                    for alt_query in alternative_queries:
                        try:
                            cursor.execute(alt_query)
                            result = cursor.fetchone()
                            if result:
                                from main import TankerData
                                return TankerData(
                                    tanker_number=str(result[0]).strip(),
                                    entry_date=str(result[1]),
                                    entry_time=str(result[2])
                                )
                        except Exception:
                            continue

                return None

        except Exception as e:
            logger.error(f"Error getting latest tanker: {e}")
            return None

    def run_connection_diagnostic(self):
        """Run immediate connection diagnostic"""
        print("\n" + "=" * 60)
        print("TDF SYSTEM - CONNECTION DIAGNOSTIC")
        print("=" * 60)

        # Test 1: pyodbc availability
        try:
            import pyodbc
            drivers = pyodbc.drivers()
            print(f"✅ pyodbc available with {len(drivers)} drivers:")
            for driver in drivers:
                print(f"   - {driver}")

            # Check for Access drivers
            access_drivers = [d for d in drivers if 'access' in d.lower()]
            if access_drivers:
                print(f"✅ Found Access drivers: {access_drivers}")
            else:
                print("❌ No Access drivers found - need to install Microsoft Access Database Engine")

        except ImportError:
            print("❌ pyodbc not installed - run: pip install pyodbc")
            return False

        # Test 2: Database file
        print(f"\n📁 Database file: {self.server_db}")
        if self.server_db and os.path.exists(self.server_db):
            size = os.path.getsize(self.server_db)
            print(f"✅ File exists ({size:,} bytes)")
        else:
            print("❌ Database file not found")
            return False

        # Test 3: Connection attempts
        print(f"\n🔗 Testing {len(self.connection_strings)} connection strings:")
        for i, conn_str in enumerate(self.connection_strings):
            print(f"\nTesting #{i + 1}:")
            print(f"   {conn_str}")
            try:
                conn = pyodbc.connect(conn_str, timeout=5)
                print("   ✅ Connection successful!")

                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                print(f"   ✅ Query test passed: {result[0]}")

                conn.close()
                return True

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        print("\n❌ All connection tests failed")
        return False
    def get_tanker_from_server(self, tanker_number):
        """Get specific tanker from Access server database"""
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
                    from main import TankerData  # Import here to avoid circular import
                    return TankerData(
                        tanker_number=str(result[0]).strip(),
                        entry_date=str(result[1]),
                        entry_time=str(result[2])
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting tanker {tanker_number}: {e}")
            return None

    def init_ban_records_table(self):
        """Initialize SQLite local database tables for ban records"""
        def init_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Check if ban_records table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ban_records'")
                table_exists = cursor.fetchone() is not None

                if not table_exists:
                    cursor.execute('''
                        CREATE TABLE ban_records (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            tanker_number TEXT NOT NULL,
                            ban_reason TEXT NOT NULL,
                            ban_type TEXT NOT NULL DEFAULT 'temporary',
                            start_date DATE,
                            end_date DATE,
                            created_by TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            voice_recording BLOB,
                            voice_filename TEXT,
                            is_active INTEGER DEFAULT 1
                        )
                    ''')

                    # Create indexes
                    cursor.execute("CREATE INDEX idx_ban_tanker_number ON ban_records(tanker_number)")
                    cursor.execute("CREATE INDEX idx_ban_is_active ON ban_records(is_active)")

                    conn.commit()
                    logger.info("SQLite ban records table created successfully")

                # Initialize logs table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
                logs_exists = cursor.fetchone() is not None

                if not logs_exists:
                    cursor.execute('''
                        CREATE TABLE logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            tanker_number TEXT,
                            status TEXT,
                            reason TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            operator TEXT
                        )
                    ''')
                    conn.commit()

        self.execute_with_retry(init_operation)

    def verify_specific_tanker_fast(self, tanker_number, operator="System"):
        """FAST VERSION: Tanker verification with optimized Red Entry checking"""
        logger.info(f"=== FAST VERIFICATION WITH RED ENTRY CHECK ===")
        logger.info(f"Tanker: {tanker_number}, Operator: {operator}")

        tanker_number_clean = str(tanker_number).strip().upper()

        try:
            # Step 1: Quick Red Entry check (with timeout)
            if self.red_entry_enabled:
                logger.info(f"Quick Red Entry check for: {tanker_number_clean}")

                # Use shorter timeout for Red Entry
                try:
                    # Set a 5-second timeout for Red Entry check
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError("Red Entry check timed out")

                    # Only use signal timeout on Unix systems
                    if hasattr(signal, 'SIGALRM'):
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(5)  # 5 second timeout

                    is_duplicate, red_entry_record, red_message = self.check_red_entry_duplicate(
                        tanker_number_clean, None, None
                    )

                    # Cancel timeout
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)

                    if is_duplicate:
                        logger.warning(f"FAST: RED ENTRY DUPLICATE DETECTED: {tanker_number_clean}")

                        # Log the duplicate
                        with sqlite3.connect(self.sqlite_db, timeout=5) as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO logs (tanker_number, status, reason, operator)
                                VALUES (?, ?, ?, ?)
                            """, (
                            tanker_number_clean, "DENIED_DUPLICATE", f"Red Entry duplicate: {red_message}", operator))
                            conn.commit()

                        return "DENIED", f"DUPLICATE ENTRY - {red_message}", {
                            "tanker_number": tanker_number_clean,
                            "ban_record": None,
                            "red_entry_record": red_entry_record,
                            "decision_color": "red",
                            "play_sound": True,
                            "duplicate_detected": True
                        }

                    logger.info(f"FAST: Red Entry check passed: {red_message}")

                except Exception as red_error:
                    logger.error(f"FAST: Red Entry check failed: {red_error}")
                    # Continue with verification even if Red Entry fails
                    pass

            # Step 2: Quick ban record check
            with sqlite3.connect(self.sqlite_db, timeout=5) as conn:
                cursor = conn.cursor()

                # Simple ban check
                cursor.execute("""
                    SELECT id, ban_reason, ban_type, voice_recording, voice_filename
                    FROM ban_records 
                    WHERE UPPER(tanker_number) = UPPER(?) 
                    AND is_active = 1
                    AND (end_date IS NULL OR date(end_date) >= date('now'))
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (tanker_number_clean,))

                ban_record = cursor.fetchone()

                if ban_record:
                    ban_id, ban_reason, ban_type, voice_data, voice_filename = ban_record

                    status_map = {
                        "permanent": ("DENIED", f"PERMANENT BAN: {ban_reason}", "red", True),
                        "temporary": ("DENIED", f"TEMPORARY BAN: {ban_reason}", "red", True),
                        "permission": ("ALLOWED_WITH_PERMISSION", f"PERMISSION REQUIRED: {ban_reason}", "orange", True),
                        "reminder": ("ALLOWED_WITH_WARNING", f"REMINDER: {ban_reason}", "orange", True)
                    }

                    status, reason, decision_color, play_sound = status_map.get(ban_type, (
                        "DENIED", f"BAN ACTIVE: {ban_reason}", "red", True))

                    # Log verification
                    cursor.execute("""
                        INSERT INTO logs (tanker_number, status, reason, operator)
                        VALUES (?, ?, ?, ?)
                    """, (tanker_number_clean, status, reason, operator))
                    conn.commit()

                    logger.info(f"=== FAST VERIFICATION RESULT: {status} ===")

                    return status, reason, {
                        "tanker_number": tanker_number_clean,
                        "ban_record": {
                            'id': ban_id,
                            'ban_reason': ban_reason,
                            'ban_type': ban_type,
                            'voice_recording': voice_data,
                            'voice_filename': voice_filename
                        },
                        "red_entry_record": None,
                        "decision_color": decision_color,
                        "play_sound": play_sound,
                        "duplicate_detected": False
                    }
                else:
                    # No ban found - ALLOWED
                    status = "ALLOWED"
                    reason = f"Vehicle {tanker_number_clean} verified successfully - Access granted"

                    # Log verification
                    cursor.execute("""
                        INSERT INTO logs (tanker_number, status, reason, operator)
                        VALUES (?, ?, ?, ?)
                    """, (tanker_number_clean, status, reason, operator))
                    conn.commit()

                    logger.info(f"=== FAST VERIFICATION RESULT: {status} ===")

                    return status, reason, {
                        "tanker_number": tanker_number_clean,
                        "ban_record": None,
                        "red_entry_record": None,
                        "decision_color": "green",
                        "play_sound": False,
                        "duplicate_detected": False
                    }

        except Exception as e:
            logger.error(f"FAST verification error for {tanker_number_clean}: {e}")
            return "ERROR", f"Verification failed: {str(e)}", {
                "error": True,
                "tanker_number": tanker_number_clean,
                "duplicate_detected": False
            }

    def verify_specific_tanker(self, tanker_number, operator="System"):
        """
        IMPROVED: Main verification with fallback options
        """
        logger.info(f"Starting verification for {tanker_number}")

        # Try fast verification first
        try:
            if hasattr(self, 'verify_specific_tanker_fast'):
                return self.verify_specific_tanker_fast(tanker_number, operator)
        except Exception as fast_error:
            logger.error(f"Fast verification failed: {fast_error}")

            # If Red Entry is causing issues, try without it
            if "time data" in str(fast_error) or "parsing" in str(fast_error).lower():
                logger.warning("Red Entry parsing failed, trying verification without Red Entry")
                try:
                    return self.verify_specific_tanker_no_red_entry(tanker_number, operator)
                except Exception as no_red_error:
                    logger.error(f"Verification without Red Entry also failed: {no_red_error}")

        # Final fallback - basic verification
        logger.warning("All verification methods failed, using basic verification")
        return "ERROR", "All verification methods failed", {
            "error": True,
            "tanker_number": str(tanker_number),
            "duplicate_detected": False
        }
    def simple_tanker_verification(self, operator="System"):
        """ENHANCED: Simple auto verification with Red Entry checking"""
        try:
            latest_tanker = self.get_latest_tanker_from_server()
            if not latest_tanker:
                error_msg = "No tanker entries found" if self.server_available else "Access server unavailable"
                return "ERROR", error_msg, {"error": True}

            return self.verify_specific_tanker(latest_tanker.tanker_number, operator)
        except Exception as e:
            logger.error(f"Verification error: {e}")
            error_msg = f"Verification failed: {str(e)}"
            return "ERROR", error_msg, {"error": True}

    def add_ban_record(self, tanker_number, ban_reason, ban_type, start_date, end_date, created_by, voice_data=None, voice_filename=None):
        """Add ban record with network optimization"""
        def add_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ban_records 
                    (tanker_number, ban_reason, ban_type, start_date, end_date, created_by, voice_recording, voice_filename, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (tanker_number, ban_reason, ban_type, start_date, end_date, created_by, voice_data, voice_filename))
                conn.commit()
                return True

        try:
            return self.execute_with_retry(add_operation)
        except Exception as e:
            logger.error(f"Error adding ban record: {e}")
            return False

    def get_all_bans(self, filters=None, exclude_blob=True, include_inactive=False):
        """Get all bans with network optimization"""
        def get_bans_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                if exclude_blob:
                    query = """SELECT id, tanker_number, ban_reason, ban_type, start_date, end_date, 
                              created_by, voice_filename, created_at, is_active FROM ban_records"""
                else:
                    query = "SELECT * FROM ban_records"

                conditions = []
                params = []

                if not include_inactive:
                    conditions.append("is_active = 1")

                if filters:
                    if filters.get('start_date') and filters.get('end_date'):
                        conditions.append("DATE(created_at) BETWEEN ? AND ?")
                        params.extend([filters['start_date'], filters['end_date']])
                    if filters.get('reason'):
                        conditions.append("LOWER(ban_reason) LIKE LOWER(?)")
                        params.append(f"%{filters['reason']}%")
                    if filters.get('ban_type'):
                        conditions.append("ban_type = ?")
                        params.append(filters['ban_type'])
                    if filters.get('tanker_number'):
                        conditions.append("LOWER(tanker_number) LIKE LOWER(?)")
                        params.append(f"%{filters['tanker_number']}%")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY created_at DESC LIMIT 1000"
                cursor.execute(query, params)
                return cursor.fetchall()

        try:
            return self.execute_with_retry(get_bans_operation)
        except Exception as e:
            logger.error(f"Error getting bans: {e}")
            return []

    def get_recent_logs(self, limit=50, filters=None):
        """Get recent logs with network optimization"""
        def get_logs_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT id, tanker_number, status, reason, timestamp, operator FROM logs"
                params = []
                conditions = []

                if filters:
                    if filters.get('start_date') and filters.get('end_date'):
                        conditions.append("DATE(timestamp) BETWEEN ? AND ?")
                        params.extend([filters['start_date'], filters['end_date']])
                    if filters.get('reason'):
                        conditions.append("LOWER(reason) LIKE LOWER(?)")
                        params.append(f"%{filters['reason']}%")
                    if filters.get('status'):
                        conditions.append("LOWER(status) LIKE LOWER(?)")
                        params.append(f"%{filters['status']}%")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                return cursor.fetchall()

        try:
            return self.execute_with_retry(get_logs_operation)
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return []

    def get_ban_statistics(self, filters=None):
        """Get ban statistics with network optimization"""
        def get_stats_operation():
            with self.get_sqlite_connection() as conn:
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

                stats = {
                    'total_bans': len(results),
                    'permanent': sum(1 for ban_type, _ in results if str(ban_type).lower() == "permanent"),
                    'temporary': sum(1 for ban_type, _ in results if str(ban_type).lower() == "temporary"),
                    'permission': sum(1 for ban_type, _ in results if str(ban_type).lower() == "permission"),
                    'reminder': sum(1 for ban_type, _ in results if str(ban_type).lower() == "reminder")
                }

                active_query = "SELECT COUNT(*) FROM ban_records WHERE is_active = 1 AND (end_date IS NULL OR end_date >= date('now'))"
                if filters and filters.get('start_date') and filters.get('end_date'):
                    active_query += " AND DATE(created_at) BETWEEN ? AND ?"
                    cursor.execute(active_query, params)
                else:
                    cursor.execute(active_query)

                stats['active_bans'] = cursor.fetchone()[0]
                return stats

        try:
            return self.execute_with_retry(get_stats_operation)
        except Exception as e:
            logger.error(f"Error getting ban statistics: {e}")
            return {'total_bans': 0, 'permanent': 0, 'temporary': 0, 'permission': 0, 'reminder': 0, 'active_bans': 0}

    def get_verification_statistics(self, filters=None):
        """Get verification statistics with network optimization"""
        def get_verify_stats_operation():
            with self.get_sqlite_connection() as conn:
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
                allowed_count = sum(1 for (status,) in results if "ALLOWED" in str(status).upper())
                rejected_count = sum(1 for (status,) in results if any(word in str(status).upper() for word in ["REJECTED", "DENIED"]))
                conditional_count = sum(1 for (status,) in results if any(word in str(status).upper() for word in ["CONDITIONAL", "PERMISSION"]))
                error_count = sum(1 for (status,) in results if "ERROR" in str(status).upper())

                return {
                    'total': total_count,
                    'allowed': allowed_count,
                    'rejected': rejected_count,
                    'conditional': conditional_count,
                    'errors': error_count,
                    'success_rate': (allowed_count / total_count * 100) if total_count > 0 else 0
                }

        try:
            return self.execute_with_retry(get_verify_stats_operation)
        except Exception as e:
            logger.error(f"Error getting verification statistics: {e}")
            return {'total': 0, 'allowed': 0, 'rejected': 0, 'conditional': 0, 'errors': 0, 'success_rate': 0}

    def update_ban_record(self, ban_id, tanker_number, ban_reason, ban_type, start_date, end_date, updated_by):
        """Update an existing ban record with network optimization"""
        def update_operation():
            with self.get_sqlite_connection() as conn:
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
                logger.info(f"Ban record {ban_id} updated successfully by {updated_by}")
                return cursor.rowcount > 0

        try:
            return self.execute_with_retry(update_operation)
        except Exception as e:
            logger.error(f"Error updating ban record {ban_id}: {e}")
            return False

    def deactivate_ban_record(self, ban_id, operator):
        """Deactivate a ban record with network optimization"""
        def deactivate_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Get ban info first
                cursor.execute("SELECT tanker_number, ban_reason FROM ban_records WHERE id = ?", (ban_id,))
                ban_info = cursor.fetchone()

                if ban_info:
                    cursor.execute("UPDATE ban_records SET is_active = 0 WHERE id = ?", (ban_id,))

                    # Log the deactivation
                    cursor.execute("""
                        INSERT INTO logs (tanker_number, status, reason, operator)
                        VALUES (?, ?, ?, ?)
                    """, (ban_info[0], "BAN_DEACTIVATED", f"Ban record deactivated: {ban_info[1]}", operator))

                    conn.commit()
                    logger.info(f"Ban record {ban_id} deactivated successfully by {operator}")
                    return True
                else:
                    logger.warning(f"Ban record {ban_id} not found for deactivation")
                    return False

        try:
            return self.execute_with_retry(deactivate_operation)
        except Exception as e:
            logger.error(f"Error deactivating ban record {ban_id}: {e}")
            return False

    def reactivate_ban_record(self, ban_id, operator):
        """Reactivate a ban record with network optimization"""
        def reactivate_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Get ban info first
                cursor.execute("SELECT tanker_number, ban_reason FROM ban_records WHERE id = ?", (ban_id,))
                ban_info = cursor.fetchone()

                if ban_info:
                    cursor.execute("UPDATE ban_records SET is_active = 1 WHERE id = ?", (ban_id,))

                    # Log the reactivation
                    cursor.execute("""
                        INSERT INTO logs (tanker_number, status, reason, operator)
                        VALUES (?, ?, ?, ?)
                    """, (ban_info[0], "BAN_REACTIVATED", f"Ban record reactivated: {ban_info[1]}", operator))

                    conn.commit()
                    logger.info(f"Ban record {ban_id} reactivated successfully by {operator}")
                    return True
                else:
                    logger.warning(f"Ban record {ban_id} not found for reactivation")
                    return False

        try:
            return self.execute_with_retry(reactivate_operation)
        except Exception as e:
            logger.error(f"Error reactivating ban record {ban_id}: {e}")
            return False

    def delete_ban_record(self, ban_id, operator):
        """Permanently delete a ban record with network optimization"""
        def delete_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Get ban info first
                cursor.execute("SELECT tanker_number, ban_reason FROM ban_records WHERE id = ?", (ban_id,))
                ban_info = cursor.fetchone()

                if ban_info:
                    cursor.execute("DELETE FROM ban_records WHERE id = ?", (ban_id,))

                    # Log the deletion
                    cursor.execute("""
                        INSERT INTO logs (tanker_number, status, reason, operator)
                        VALUES (?, ?, ?, ?)
                    """, (ban_info[0], "BAN_DELETED", f"Ban record permanently deleted: {ban_info[1]}", operator))

                    conn.commit()
                    logger.info(f"Ban record {ban_id} deleted successfully by {operator}")
                    return True
                else:
                    logger.warning(f"Ban record {ban_id} not found for deletion")
                    return False

        try:
            return self.execute_with_retry(delete_operation)
        except Exception as e:
            logger.error(f"Error deleting ban record {ban_id}: {e}")
            return False

    def log_verification(self, tanker_number, status, reason, operator):
        """Log verification to SQLite local database"""
        def log_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO logs (tanker_number, status, reason, operator)
                    VALUES (?, ?, ?, ?)
                """, (tanker_number, status, reason, operator))
                conn.commit()

        try:
            self.execute_with_retry(log_operation)
        except Exception as e:
            logger.error(f"Error logging verification: {e}")

    def get_complete_ban_record(self, tanker_number):
        """Get complete ban record from SQLite local database"""
        def get_record_operation():
            with self.get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Check if table exists first
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ban_records'")
                if not cursor.fetchone():
                    logger.warning("ban_records table does not exist")
                    return None

                # Case-insensitive search using UPPER()
                query = """
                    SELECT id, tanker_number, ban_reason, ban_type, start_date, end_date, 
                           created_by, created_at, voice_recording, voice_filename
                    FROM ban_records 
                    WHERE UPPER(tanker_number) = UPPER(?) 
                    AND is_active = 1
                    AND (end_date IS NULL OR date(end_date) >= date('now'))
                    ORDER BY created_at DESC
                    LIMIT 1
                """

                cursor.execute(query, (tanker_number,))
                result = cursor.fetchone()

                if result:
                    ban_record = {
                        'id': result[0], 'tanker_number': result[1], 'ban_reason': result[2],
                        'ban_type': result[3], 'start_date': result[4], 'end_date': result[5],
                        'created_by': result[6], 'created_at': result[7],
                        'voice_recording': result[8], 'voice_filename': result[9]
                    }
                    logger.info(f"FOUND ban record for {tanker_number}: ID {ban_record['id']}, type: {ban_record['ban_type']}")
                    return ban_record
                else:
                    logger.info(f"No active ban record found for {tanker_number}")
                    return None

        try:
            return self.execute_with_retry(get_record_operation)
        except Exception as e:
            logger.error(f"Error getting ban record for {tanker_number}: {e}")
            return None

    def update_paths(self, server_db_path, sqlite_db_path):
        """Update database paths and reinitialize connections"""
        try:
            logger.info(f"Updating database paths:")
            logger.info(f"  Server DB (Access): {server_db_path}")
            logger.info(f"  Local DB (SQLite): {sqlite_db_path}")

            self.server_db = server_db_path
            self.sqlite_db = sqlite_db_path

            self.connection_strings = [
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={self.server_db};",
                f"Driver={{Microsoft Access Driver (*.mdb)}};DBQ={self.server_db};",
            ]

            # Test connections
            try:
                self.test_server_connection()
                logger.info("Access server connection updated successfully")
            except Exception as e:
                logger.warning(f"Server connection test failed: {e}")

            try:
                self.init_ban_records_table()
                logger.info("SQLite local database updated successfully")
            except Exception as e:
                logger.warning(f"Local database initialization failed: {e}")

            logger.info("Database paths updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating database paths: {e}")
            return False

    def test_red_entry_configuration(self):
        """Test Red Entry table access and configuration"""
        if not self.red_entry_enabled:
            return False, "Red Entry checking is disabled in configuration"

        if not self.server_available:
            return False, "Server database not available"

        try:
            with self.get_server_connection() as conn:
                cursor = conn.cursor()

                # Test if Red Entry table exists
                try:
                    cursor.execute(f"SELECT TOP 1 * FROM {self.red_entry_table}")
                    cursor.fetchone()
                    logger.info(f"Red Entry table '{self.red_entry_table}' exists and accessible")
                except Exception as e:
                    return False, f"Red Entry table '{self.red_entry_table}' not accessible: {e}"

                # Test if required columns exist
                try:
                    test_query = f"""
                        SELECT TOP 5 {self.red_entry_id_column}, {self.red_entry_reg_column}, 
                               {self.red_entry_date_column}, {self.red_entry_time_column}
                        FROM {self.red_entry_table}
                    """
                    cursor.execute(test_query)
                    results = cursor.fetchall()

                    record_count = len(results)
                    logger.info(f"Red Entry configuration test successful - found {record_count} sample records")

                    return True, f"Red Entry configuration valid - table accessible with {record_count} records"

                except Exception as e:
                    return False, f"Red Entry columns not accessible: {e}"

        except Exception as e:
            logger.error(f"Error testing Red Entry configuration: {e}")
            return False, f"Red Entry configuration test failed: {e}"

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Enhanced Database Manager with Red Entry checking cleanup completed")