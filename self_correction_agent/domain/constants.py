"""
도메인 상수: Critic 평가 기준 및 임계값.
"""

REQUIRED_KEYWORDS = [
    "GPT-4o", "Sora", "API", "safety", "multimodal",
    "o1", "enterprise", "partnership",
]

PASS_THRESHOLD = 0.7
MIN_LENGTH = 500
MIN_KEYWORD_RATIO = 0.85  # Hard gate: 85% 이상 키워드 커버리지 필수
