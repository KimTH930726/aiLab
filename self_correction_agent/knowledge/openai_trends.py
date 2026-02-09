"""
시드 데이터: OpenAI 최신 동향 지식 베이스.

프로덕션에서는 JSON 파일 로드, 웹 크롤링, 또는 외부 API로 교체 가능.
"""

KNOWLEDGE_BASE: list[dict[str, str]] = [
    {
        "id": "1",
        "topic": "GPT-4o",
        "source": "OpenAI Blog 2024",
        "text": (
            "GPT-4o (omni) was released in May 2024 as OpenAI's most advanced "
            "multimodal model. It processes text, image, and audio inputs and "
            "generates text, image, and audio outputs. GPT-4o matches GPT-4 "
            "Turbo performance while being 2x faster and 50% cheaper via the API."
        ),
    },
    {
        "id": "2",
        "topic": "Sora",
        "source": "OpenAI Research 2024",
        "text": (
            "Sora is OpenAI's text-to-video generation model announced in "
            "February 2024. It generates realistic videos up to 60 seconds "
            "from text descriptions using a diffusion transformer architecture. "
            "Sora demonstrates understanding of 3D consistency and object permanence."
        ),
    },
    {
        "id": "3",
        "topic": "API Platform",
        "source": "OpenAI Developer Docs 2024",
        "text": (
            "The OpenAI API platform introduced the Assistants API, improved "
            "function calling, structured outputs (JSON mode), fine-tuning for "
            "GPT-4o-mini, and batch processing. GPT-4o-mini is 60% cheaper "
            "than GPT-3.5-turbo, making the API more accessible to developers."
        ),
    },
    {
        "id": "4",
        "topic": "Safety & Alignment",
        "source": "OpenAI Safety Report 2024",
        "text": (
            "OpenAI invested heavily in safety research including RLHF "
            "improvements, red-teaming exercises, and a Preparedness Framework "
            "for evaluating catastrophic risks. The Superalignment team was "
            "formed to tackle long-term alignment challenges."
        ),
    },
    {
        "id": "5",
        "topic": "ChatGPT Enterprise",
        "source": "OpenAI Product Blog 2024",
        "text": (
            "ChatGPT Enterprise and Team plans offer businesses dedicated "
            "capacity, advanced data privacy controls, admin console features, "
            "unlimited GPT-4 access, and 128K context windows. ChatGPT Edu "
            "extended enterprise capabilities to universities."
        ),
    },
    {
        "id": "6",
        "topic": "GPT Store",
        "source": "OpenAI Blog 2024",
        "text": (
            "The GPT Store launched in January 2024, allowing users to create, "
            "share, and monetize custom GPT applications. It features a no-code "
            "builder interface with actions (API integrations), knowledge "
            "retrieval, and code interpreter capabilities."
        ),
    },
    {
        "id": "7",
        "topic": "o1 Reasoning Model",
        "source": "OpenAI Research 2024",
        "text": (
            "The o1 model series (codenamed 'Strawberry') uses internal "
            "chain-of-thought reasoning, spending more time thinking before "
            "answering. o1 excels at complex math, science, and coding tasks, "
            "achieving PhD-level benchmark performance."
        ),
    },
    {
        "id": "8",
        "topic": "Multimodal Capabilities",
        "source": "OpenAI Technical Report 2024",
        "text": (
            "OpenAI's multimodal stack includes GPT-4V (vision understanding), "
            "DALL-E 3 (image generation), Whisper (speech recognition), and "
            "Advanced Voice Mode with GPT-4o for native audio processing "
            "without intermediate transcription steps."
        ),
    },
    {
        "id": "9",
        "topic": "Open Source & Research",
        "source": "OpenAI GitHub 2024",
        "text": (
            "Open-source releases include Whisper, CLIP, Triton (GPU programming), "
            "and Evals. Research publications on scaling laws, instruction tuning, "
            "and RLHF methods have significantly influenced the broader AI "
            "research community."
        ),
    },
    {
        "id": "10",
        "topic": "Partnerships & Industry",
        "source": "Industry News 2024",
        "text": (
            "OpenAI deepened its Microsoft partnership via Azure OpenAI Service "
            "and Copilot integration. Partnered with Apple for iOS 18 Siri "
            "integration. The company reached $157 billion valuation in its "
            "latest investment round."
        ),
    },
]
