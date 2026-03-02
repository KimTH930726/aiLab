"""
도메인 상수: Critic 평가 기준 및 임계값.

REQUIRED_KEYWORDS 항목은 문자열(단일 키워드) 또는
튜플(OR 그룹 — 하나만 있어도 통과)을 지원한다.
"""

REQUIRED_KEYWORDS: list = [
    # 고유명사 / 약어: 한/영 공통
    "GPT-4o",
    "Sora",
    "API",
    "o1",
    # 한국어 차용어 OR 영어 원어 (어느 쪽이든 통과)
    ("멀티모달", "multimodal"),
    ("파트너십", "partnership"),
    ("엔터프라이즈", "enterprise", "기업용"),
    ("안전성", "안전", "safety"),
]

PASS_THRESHOLD = 0.7
MIN_LENGTH = 500
MIN_KEYWORD_RATIO = 0.85  # Hard gate: 85% 이상 키워드 커버리지 필수
