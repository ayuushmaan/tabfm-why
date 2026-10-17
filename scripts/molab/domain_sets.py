"""Domain benchmark manifest: finance + health classification sets (public OpenML).

All targets verified live via fetch_openml before inclusion. 16 sets, 8 per domain.
Anchors (credit-g, diabetes, breast-w, heart-statlog, thyroid) also appear in the
main 16-set roster — kept for continuity between the general and domain tracks.
"""

FINANCE = [
    ("credit-g", 31, "finance"),
    ("australian", 40981, "finance"),
    ("bank-marketing", 1461, "finance"),
    ("default-credit-clients", 42477, "finance"),
    ("banknote-authentication", 1462, "finance"),
    ("creditcard-fraud", 1597, "finance"),
    ("online-shoppers", 45060, "finance"),
    ("german-credit-numeric", 1049, "finance"),
]

HEALTH = [
    ("diabetes", 37, "health"),
    ("breast-w", 15, "health"),
    ("heart-statlog", 53, "health"),
    ("thyroid", 38, "health"),
    ("hepatitis", 55, "health"),
    ("ilpd", 1480, "health"),
    ("breast-cancer-diag", 1510, "health"),
    ("haberman", 43, "health"),
]

ALL = FINANCE + HEALTH
