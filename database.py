"""
Database setup and models for dynamic parking data.
Dynamic data: space availability, working hours, prices, real-time information.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import json

DATABASE_PATH = "parking_data.db"


@dataclass
class ParkingReservation:
    """Model for parking reservations"""
    id: Optional[int] = None
    customer_name: str = ""
    license_plate: str = ""
    reservation_date: str = ""  # YYYY-MM-DD
    reservation_time: str = ""  # HH:MM
    duration_hours: int = 1
    status: str = "pending"  # pending, confirmed, completed, cancelled
    created_at: Optional[str] = None
    admin_notes: str = ""


@dataclass
class ParkingRate:
    """Model for parking rates"""
    id: Optional[int] = None
    rate_type: str = ""  # hourly, daily, special
    price: float = 0.0
    duration_minutes: int = 60
    description: str = ""
    active: bool = True


@dataclass
class ParkingAvailability:
    """Model for space availability"""
    id: Optional[int] = None
    date: str = ""  # YYYY-MM-DD
    time: str = ""  # HH:MM
    available_spaces: int = 0
    total_spaces: int = 100
    last_updated: Optional[str] = None


def init_database():
    """Initialize SQLite database with schema"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Reservations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            license_plate TEXT NOT NULL,
            reservation_date TEXT NOT NULL,
            reservation_time TEXT NOT NULL,
            duration_hours INTEGER DEFAULT 1,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            admin_notes TEXT,
            UNIQUE(license_plate, reservation_date, reservation_time)
        )
    """)

    # Parking rates table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rate_type TEXT NOT NULL,
            price REAL NOT NULL,
            duration_minutes INTEGER NOT NULL,
            description TEXT,
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Space availability table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_availability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            available_spaces INTEGER NOT NULL,
            total_spaces INTEGER DEFAULT 100,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, time)
        )
    """)

    # Audit log for data protection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            user_input TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            contains_pii BOOLEAN DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


class ReservationManager:
    """Handle parking reservations"""

    @staticmethod
    def create_reservation(
        customer_name: str,
        license_plate: str,
        reservation_date: str,
        reservation_time: str,
        duration_hours: int = 1
    ) -> Optional[int]:
        """Create new reservation"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO reservations 
                (customer_name, license_plate, reservation_date, reservation_time, duration_hours)
                VALUES (?, ?, ?, ?, ?)
            """, (customer_name, license_plate, reservation_date, reservation_time, duration_hours))

            reservation_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return reservation_id
        except sqlite3.IntegrityError:
            return None  # Duplicate reservation

    @staticmethod
    def get_reservation(reservation_id: int) -> Optional[ParkingReservation]:
        """Get reservation by ID"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM reservations WHERE id = ?", (reservation_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return ParkingReservation(
                id=row[0],
                customer_name=row[1],
                license_plate=row[2],
                reservation_date=row[3],
                reservation_time=row[4],
                duration_hours=row[5],
                status=row[6],
                created_at=row[7],
                admin_notes=row[8]
            )
        return None

    @staticmethod
    def get_pending_reservations() -> List[ParkingReservation]:
        """Get all pending reservations for admin review"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM reservations WHERE status = 'pending' ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            ParkingReservation(
                id=row[0],
                customer_name=row[1],
                license_plate=row[2],
                reservation_date=row[3],
                reservation_time=row[4],
                duration_hours=row[5],
                status=row[6],
                created_at=row[7],
                admin_notes=row[8]
            )
            for row in rows
        ]

    @staticmethod
    def update_reservation_status(reservation_id: int, status: str, admin_notes: str = ""):
        """Update reservation status (confirm, cancel, complete)"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE reservations SET status = ?, admin_notes = ? WHERE id = ?",
            (status, admin_notes, reservation_id)
        )

        conn.commit()
        conn.close()


class PricingManager:
    """Manage parking rates"""

    @staticmethod
    def init_default_rates():
        """Initialize default parking rates"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        rates = [
            ("hourly", 40.0, 60, "40 грн per hour"),
            ("daily_12h", 300.0, 720, "Daily rate for 12 hours - 300 грн"),
            ("daily_24h", 500.0, 1440, "Full day rate (24 hours) - 500 грн"),
            ("free_drop_off", 0.0, 15, "Free for first 15 minutes (drop-off/pick-up)"),
        ]

        for rate_type, price, duration, description in rates:
            cursor.execute(
                "INSERT OR IGNORE INTO parking_rates (rate_type, price, duration_minutes, description) VALUES (?, ?, ?, ?)",
                (rate_type, price, duration, description)
            )

        conn.commit()
        conn.close()

    @staticmethod
    def get_active_rates() -> List[Dict]:
        """Get all active parking rates"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM parking_rates WHERE active = 1 ORDER BY duration_minutes")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "type": row[1],
                "price": row[2],
                "duration_minutes": row[3],
                "description": row[4],
            }
            for row in rows
        ]

    @staticmethod
    def calculate_parking_cost(duration_minutes: int) -> float:
        """Calculate parking cost based on duration"""
        rates = PricingManager.get_active_rates()

        # Check for exact match (24h, 12h rates)
        for rate in rates:
            if rate["duration_minutes"] == duration_minutes:
                return rate["price"]

        # Calculate hourly
        hourly_rate = next((r["price"] for r in rates if r["type"] == "hourly"), 40.0)
        hours = (duration_minutes + 59) // 60  # Round up to nearest hour
        return hourly_rate * hours


class AvailabilityManager:
    """Manage parking space availability"""

    @staticmethod
    def update_availability(date: str, time: str, available_spaces: int, total_spaces: int = 100):
        """Update availability for a specific date/time"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO parking_availability 
            (date, time, available_spaces, total_spaces, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (date, time, available_spaces, total_spaces))

        conn.commit()
        conn.close()

    @staticmethod
    def get_availability(date: str, time: str = None) -> Dict:
        """Get availability for a specific date/time"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        if time:
            cursor.execute(
                "SELECT * FROM parking_availability WHERE date = ? AND time = ?",
                (date, time)
            )
        else:
            cursor.execute(
                "SELECT * FROM parking_availability WHERE date = ? ORDER BY time",
                (date,)
            )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"message": "No availability data for this date"}

        if time:
            row = rows[0]
            return {
                "date": row[1],
                "time": row[2],
                "available_spaces": row[3],
                "total_spaces": row[4],
                "last_updated": row[5]
            }
        else:
            return {
                "date": date,
                "slots": [
                    {
                        "time": row[2],
                        "available": row[3],
                        "total": row[4]
                    }
                    for row in rows
                ]
            }


class AuditLog:
    """Manage audit logs for data protection"""

    @staticmethod
    def log_interaction(
        action: str,
        user_input: str = "",
        bot_response: str = "",
        contains_pii: bool = False
    ):
        """Log chatbot interaction"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO audit_log (action, user_input, bot_response, contains_pii)
            VALUES (?, ?, ?, ?)
        """, (action, user_input, bot_response, contains_pii))

        conn.commit()
        conn.close()

    @staticmethod
    def get_interaction_logs(limit: int = 100) -> List[Dict]:
        """Get recent interaction logs"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, action, user_input, bot_response, timestamp, contains_pii
            FROM audit_log
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "action": row[1],
                "user_input": row[2],
                "bot_response": row[3],
                "timestamp": row[4],
                "contains_pii": row[5]
            }
            for row in rows
        ]


if __name__ == "__main__":
    init_database()
    PricingManager.init_default_rates()
    # test: Show rates
    rates = PricingManager.get_active_rates()
    print("\nCurrent parking rates:")
    for rate in rates:
        print(f"  {rate['description']} - {rate['price']} грн")

