LICENSE_PLATE_PATTERN = r'\b[A-Z]{2}\s?\d{4}\s?[A-Z]{2}\b|[А-Я]{2}\s?\d{4}\s?[А-Я]{2}\b'
PHONE_PATTERN = r'(?:\+38)?[\s]?[(]?\d{2,3}[)]?[\s]?\d{3}[\s-]?\d{2}[\s-]?\d{2}'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
NAME_PATTERN = r'\b[А-Яа-яA-Za-z]+\s+[А-Яа-яA-Za-z]+\b'
DATE_PATTERNS = [
    r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
    r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
    r'(\d{2})/(\d{2})/(\d{4})',  # DD/MM/YYYY
    r'(завтра|tomorrow|сьогодні|today|завтра|the day after tomorrow)',  # relative dates
]
TIME_PATTERNS = [
    r'(\d{1,2}):(\d{2})',  # HH:MM or H:MM
    r'(\d{1,2})\.(\d{2})',  # HH.MM
    r'(\d{1,2})\s*:?\s*на\s*ранку',  # Xo in the morning
    r'(\d{1,2})\s*:?\s*увечері',  # Xo in the evening
]