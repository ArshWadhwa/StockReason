import datetime
from typing import Union

# Static list of NSE Trading Holidays for 2024, 2025, 2026.
# Format: YYYY-MM-DD
NSE_HOLIDAYS = {
    # 2024
    "2024-01-22", # Special Holiday (Ayodhya Ram Mandir)
    "2024-01-26", # Republic Day
    "2024-03-08", # Mahashivratri
    "2024-03-25", # Holi
    "2024-03-29", # Good Friday
    "2024-04-11", # Id-Ul-Fitr (Ramadan)
    "2024-04-17", # Shri Ram Navami
    "2024-05-01", # Maharashtra Day
    "2024-05-20", # General Elections (Mumbai)
    "2024-06-17", # Bakri Id
    "2024-07-17", # Muharram
    "2024-08-15", # Independence Day
    "2024-10-02", # Mahatma Gandhi Jayanti
    "2024-11-01", # Diwali Laxmi Pujan
    "2024-11-15", # Gurunanak Jayanti
    "2024-12-25", # Christmas
    
    # 2025
    "2025-01-26", # Republic Day (Sunday, usually observed/not explicitly a weekday off)
    "2025-02-26", # Mahashivratri
    "2025-03-14", # Holi
    "2025-03-31", # Id-Ul-Fitr
    "2025-04-10", # Mahavir Jayanti
    "2025-04-14", # Dr. Baba Saheb Ambedkar Jayanti
    "2025-04-18", # Good Friday
    "2025-05-01", # Maharashtra Day
    "2025-06-07", # Bakri Id (Saturday)
    "2025-08-15", # Independence Day
    "2025-08-27", # Ganesh Chaturthi
    "2025-10-02", # Mahatma Gandhi Jayanti
    "2025-10-21", # Diwali
    "2025-11-05", # Gurunanak Jayanti
    "2025-12-25", # Christmas
    
    # 2026
    "2026-01-26", # Republic Day
    "2026-03-03", # Holi
    "2026-03-20", # Id-Ul-Fitr
    "2026-04-03", # Good Friday
    "2026-04-14", # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01", # Maharashtra Day
    "2026-08-15", # Independence Day
    "2026-09-14", # Ganesh Chaturthi
    "2026-10-02", # Mahatma Gandhi Jayanti
    "2026-11-09", # Diwali
    "2026-11-24", # Gurunanak Jayanti
    "2026-12-25", # Christmas
}

def is_trading_day(date_obj: Union[datetime.date, datetime.datetime]) -> bool:
    """
    Checks if the given date is an NSE trading day.
    A trading day is a weekday (Monday-Friday) that is not an official NSE holiday.
    """
    # Check if it's a weekend (Saturday=5, Sunday=6)
    if date_obj.weekday() >= 5:
        return False
        
    date_str = date_obj.strftime("%Y-%m-%d")
    if date_str in NSE_HOLIDAYS:
        return False
        
    return True
