"""Pre-configured decision primitives for production operational workflows."""

from typing import Any, Dict, Optional
from .types import Choice, Noul, Score


def triage_preset() -> Dict[str, Any]:
    """Preset questions for customer support ticket triage and routing."""
    return {
        "intent": Choice(
            instructions="What is the primary customer intent in the message?",
            criteria={
                "refund": "Requesting money back, refund, or duplicate billing reversal",
                "technical_help": "Reporting a bug, API error, 500 downtime, or integration issue",
                "billing_question": "Questions about invoices, subscription plans, or payment methods",
                "cancellation": "Requesting account closure, cancellation, or downgrading",
                "general_info": "Inquiring about documentation, pricing tiers, or how-to guidance",
            },
        ),
        "is_urgent": Noul(
            instructions="Does the customer communicate extreme urgency, critical outage, or impending deadline?",
            criteria={
                "true": "Urgent, production down, emergency, immediate attention needed",
                "false": "Routine question, low priority, general feedback",
            },
        ),
        "frustration": Score(
            instructions="Rate the customer frustration level.",
            criteria=[
                "Calm and polite",
                "Slightly concerned or asking for status",
                "Visibly frustrated or annoyed",
                "Extremely angry, threatening legal action or cancellation",
            ],
        ),
        "churn_risk": Noul(
            instructions="Does the message indicate high risk of the customer leaving or churning?",
            criteria={
                "true": "Threatening to switch to competitors, cancel contract, or stop using product",
                "false": "Committed user asking for help, no mention of leaving",
            },
        ),
    }


def email_preset(custom_categories: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Preset questions for inbound email triage and threat filtering."""
    categories = custom_categories or {
        "billing": "Invoices, payments, credit cards, pricing questions",
        "engineering": "Bug reports, API failures, stack traces, system outages",
        "sales": "Enterprise demos, contract inquiries, volume discounts",
        "security": "Phishing reports, suspicious access, vulnerability disclosures",
        "general": "General questions or uncategorized inquiries",
    }
    return {
        "destination": Choice(
            instructions="Which internal team should handle this email?",
            criteria=categories,
        ),
        "is_spam_or_phishing": Noul(
            instructions="Is this email an unsolicited sales pitch, scam, or phishing attempt?",
            criteria={
                "true": "Spam, promotional blast, credential harvesting, phishing",
                "false": "Legitimate user or customer inquiry",
            },
        ),
        "priority": Score(
            instructions="What priority level should be assigned to this email?",
            criteria=[
                "Low: Newsletter, informational, no action required",
                "Medium: Standard inquiry with 24-48hr SLA",
                "High: Blocking issue affecting paying customer",
                "Critical: Security breach, legal threat, or severe production impact",
            ],
        ),
    }


def moderation_preset() -> Dict[str, Any]:
    """Preset questions for user-generated content and trust & safety moderation."""
    return {
        "policy_violation": Choice(
            instructions="Does this content violate acceptable use policies?",
            criteria={
                "clean": "Content is safe, constructive, and follows community guidelines",
                "harassment": "Direct personal attacks, bullying, threats, or hate speech",
                "spam": "Repetitive links, commercial spam, crypto scams, or bot text",
                "sensitive": "Explicit adult content, graphic violence, or illegal goods",
            },
        ),
        "should_block": Noul(
            instructions="Should this content be immediately blocked from publication?",
            criteria={
                "true": "Clear violation requiring immediate rejection",
                "false": "Safe or borderline content that can be published or reviewed",
            },
        ),
        "severity": Score(
            instructions="Rate the severity of the content risk.",
            criteria=[
                "Safe: Compliant content",
                "Low: Minor profanity or mild uncivil behavior",
                "Medium: Aggressive tone, self-promotion, or borderline spam",
                "High: Severe violation, harassment, or malicious payload",
            ],
        ),
    }


def security_preset() -> Dict[str, Any]:
    """Preset questions for security event triage and anomaly assessment."""
    return {
        "event_type": Choice(
            instructions="Classify the observed security or authentication anomaly.",
            criteria={
                "benign": "Expected user activity, legitimate IP change, or normal login",
                "credential_stuffing": "Rapid succession of failed logins across multiple accounts",
                "brute_force": "Repeated failed attempts targeting a single high-value account",
                "privilege_escalation": "Attempting unauthorized administrative or sudo operations",
                "data_exfiltration": "Abnormal volume of export requests or bulk database downloads",
            },
        ),
        "is_threat": Noul(
            instructions="Does this state represent an active, confirmed malicious security threat?",
            criteria={
                "true": "Active cyber attack, intrusion, or unauthorized compromise",
                "false": "Normal operational glitch, user error, or benign variance",
            },
        ),
        "severity": Score(
            instructions="Rate the incident severity.",
            criteria=[
                "Informational: Logged for audit trail, no action",
                "Warning: Suspicious variance, rate-limit triggered",
                "Elevated: Incident responder paged for triage",
                "Critical: Active breach, immediate token revocation and IP ban",
            ],
        ),
    }
