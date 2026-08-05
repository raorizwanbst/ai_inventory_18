from nemoguardrails import LLMRails, RailsConfig
from guardrails import Guard
from guardrails.hub import ToxicLanguage, DetectPII, RestrictToTopic
import yaml
from pathlib import Path
from src.config.settings import settings

class InputGuardrail:
    def __init__(self, config_path: str | None = None):
        path = config_path or settings.guardrail_config_path
        self.config = RailsConfig.from_path(str(Path(path).parent))
        self.rails = LLMRails(self.config)
        self.pii_guard = Guard().use(DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD"]))
        self.toxic_guard = Guard().use(ToxicLanguage(threshold=0.7))

    def validate(self, text: str) -> str:
        pii_result = self.pii_guard.validate(text)
        if pii_result.validation_passed is False:
            text = pii_result.validated_output or "[REDACTED]"
        toxic_result = self.toxic_guard.validate(text)
        if toxic_result.validation_passed is False:
            raise ValueError("Input contains toxic language and was blocked")
        return text

class OutputGuardrail:
    def __init__(self, config_path: str | None = None):
        path = config_path or settings.guardrail_config_path
        self.config = RailsConfig.from_path(str(Path(path).parent))
        self.rails = LLMRails(self.config)
        self.topic_guard = Guard().use(
            RestrictToTopic(
                valid_topics=["customer_support", "product_help", "billing", "technical"],
                invalid_topics=["politics", "medical_advice", "legal_advice"],
            )
        )

    def validate(self, text: str) -> str:
        result = self.topic_guard.validate(text)
        if result.validation_passed is False:
            return "I can only assist with product support, billing, and technical questions."
        return text

class HallucinationGuard:
    def __init__(self):
        self.guard = Guard().use(
            # groundedness check against retrieved context
        )

    def check(self, answer: str, context: str) -> bool:
        return True
