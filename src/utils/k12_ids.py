"""Shared book/id conventions for this repo.

We use a consistent `book_prefix` format, e.g. `math_9a_rjb`, and derived ids like
`math_9a_rjb_ch1_s2`. Multiple modules (kg merge, benchmark generation, etc.)
need the same book-code ordering and regex parsing, so keep them centralized here.
"""

from __future__ import annotations

import re
from typing import Final

BOOK_CODE_ORDER: Final[list[str]] = [
    "1a",
    "1b",
    "2a",
    "2b",
    "3a",
    "3b",
    "4a",
    "4b",
    "5a",
    "5b",
    "6a",
    "6b",
    "7a",
    "7b",
    "8a",
    "8b",
    "9a",
    "9",
    "9b",
    "bx1",
    "bx2",
    "bx3",
    "xzxbx1",
    "xzxbx2",
    "xzxbx3",
]

BOOK_CODE_ORDER_INDEX: Final[dict[str, int]] = {code: idx for idx, code in enumerate(BOOK_CODE_ORDER)}

PRIMARY_MATH_BOOK_CODES: Final[set[str]] = {"1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b", "5a", "5b", "6a", "6b"}
MIDDLE_SCHOOL_BOOK_CODES: Final[set[str]] = {"7a", "7b", "8a", "8b", "9a", "9", "9b"}
HIGH_SCHOOL_BOOK_CODES: Final[set[str]] = {"bx1", "bx2", "bx3", "xzxbx1", "xzxbx2", "xzxbx3"}

BOOK_CODES: Final[set[str]] = PRIMARY_MATH_BOOK_CODES | MIDDLE_SCHOOL_BOOK_CODES | HIGH_SCHOOL_BOOK_CODES

BOOK_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<subject>[a-z]+)_(?P<book_code>"
    r"1a|1b|2a|2b|3a|3b|4a|4b|5a|5b|6a|6b|7a|7b|8a|8b|9a|9|9b|"
    r"bx1|bx2|bx3|xzxbx1|xzxbx2|xzxbx3"
    r")_rjb$"
)

TYPE_CODE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<subject>[a-z]+)_(?P<book_code>"
    r"1a|1b|2a|2b|3a|3b|4a|4b|5a|5b|6a|6b|7a|7b|8a|8b|9a|9|9b|"
    r"bx1|bx2|bx3|xzxbx1|xzxbx2|xzxbx3"
    r")_rjb(?:_|$)"
)

CHAPTER_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<subject>[a-z]+)_(?P<book_code>"
    r"1a|1b|2a|2b|3a|3b|4a|4b|5a|5b|6a|6b|7a|7b|8a|8b|9a|9|9b|"
    r"bx1|bx2|bx3|xzxbx1|xzxbx2|xzxbx3"
    r")_rjb_ch(?P<chapter>\d+)$"
)

SECTION_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<subject>[a-z]+)_(?P<book_code>"
    r"1a|1b|2a|2b|3a|3b|4a|4b|5a|5b|6a|6b|7a|7b|8a|8b|9a|9|9b|"
    r"bx1|bx2|bx3|xzxbx1|xzxbx2|xzxbx3"
    r")_rjb_ch(?P<chapter>\d+)_s(?P<section>\d+)$"
)

