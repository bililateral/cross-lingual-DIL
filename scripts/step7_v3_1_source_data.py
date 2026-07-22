#!/usr/bin/env python3
"""Frozen source-data and redaction contracts for Step7-v3.1."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import struct
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

import step3_build_seller_profiles as step3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step7_v3_1_source_data_policy.json"
STEP3_SCRIPT = Path(step3.__file__).resolve()
MAX_REDACTION_PASSES = 8
IDENTITY_RESIDUE_CLAIM_SCOPE = (
    "zero_detected_seller_local_aliases_preregistered_high_confidence_global_"
    "handles_audited_fixed_snapshot_seller_or_market_identity_phrases_"
    "identity_context_supported_fixed_snapshot_aliases_and_high_recall_patterns_"
    "not_proof_of_zero_unknown_or_ambiguous_content_collision_identifiers"
)

TRUNCATED_PGP_BLOCK_RE = re.compile(
    r"(?is)-{2,}\s*BEGIN\s+PGP\s+(?:PUBLIC\s+KEY\s+BLOCK|MESSAGE|SIGNATURE)\s*-{2,}"
    r".*?(?:-{2,}\s*END\s+PGP\s+(?:PUBLIC\s+KEY\s+BLOCK|MESSAGE|SIGNATURE)\s*-{2,}|$)"
)

GENERIC_IDENTIFIER_RULES = (
    ("pgp_block_truncated_or_complete", TRUNCATED_PGP_BLOCK_RE),
    ("pgp_block", step3.PGP_BLOCK_RE),
    ("pgp_fingerprint", step3.PGP_FINGERPRINT_RE),
    (
        "email_like_including_truncated_tld",
        re.compile(
            r"(?i)(?<![a-z0-9_+%\-])[a-z0-9][a-z0-9._%+\-]{0,63}"
            r"@[a-z0-9](?:[a-z0-9.\-]{0,252}[a-z0-9])?(?![a-z0-9.\-])"
        ),
    ),
    (
        "email_truncated_after_domain_without_tld",
        re.compile(
            r"(?i)(?<![a-z0-9_+%\-])[a-z0-9][a-z0-9._%+\-]{0,63}"
            r"@[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.(?![a-z0-9])"
        ),
    ),
    (
        "email_missing_at_or_truncated_provider_domain",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"[a-z0-9][a-z0-9._-]{0,63}"
            r"(?:gmail|protonmail|hotmail|yahoo|outlook|icloud|tutanota|mailfence)"
            r"\.(?:c(?:om?)?|n(?:et?)?|o(?:rg?)?|[a-z]{2,12})?(?![a-z0-9])"
        ),
    ),
    (
        "email_missing_at_and_domain_dot",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"[a-z0-9][a-z0-9._-]{2,63}"
            r"(?:gmail|protonmail|hotmail|yahoo|outlook|icloud|tutanota|mailfence)"
            r"(?:c(?:om?)?|n(?:et?)?|o(?:rg?)?)(?![a-z0-9])"
        ),
    ),
    ("email", step3.EMAIL_RE),
    ("jabber", step3.JABBER_RE),
    ("url", step3.URL_RE),
    ("bare_domain", step3.BARE_DOMAIN_RE),
    ("crypto_wallet", step3.CRYPTO_WALLET_RE),
    ("phone_context", step3.PHONE_CONTEXT_RE),
    ("telegram_profile", step3.TELEGRAM_RE),
    ("wickr", step3.WICKR_RE),
    ("wechat_profile", step3.WECHAT_RE),
    ("qq_profile", step3.QQ_RE),
    ("wechat_item", step3.WECHAT_ITEM_RE),
    ("qq_item", step3.QQ_ITEM_RE),
    ("bat", step3.BAT_RE),
    (
        "long_contiguous_identifier_number",
        re.compile(
            r"(?i)(?<![a-z0-9])(?<!\d\.)\d{7,19}(?![a-z0-9])(?!\.\d)"
        ),
    ),
    (
        "international_phone_without_cue",
        re.compile(
            r"(?i)(?<![a-z0-9])(?:x{1,6}\s*)?\+\s*"
            r"\d(?:[\s().-]*\d){5,15}(?:\s*x{1,6})?(?![a-z0-9])"
        ),
    ),
    (
        "contact_cued_phone_or_numeric_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:wh?ats?app|phone|telephone|tel|mobile|call|text|sms|"
            r"contact|reach(?:\s+(?:us|me))?|message(?:\s+(?:us|me))?)"
            r"[^|\n]{0,48}?\+?\d(?:[\s().-]*\d){5,15}(?![a-z0-9])"
        ),
    ),
    (
        "long_id_assignment",
        re.compile(
            r"(?i)(?<![a-z0-9])(?:user(?:name)?|account|contact)?\s*"
            r"id\s*[:=]\s*[a-z0-9][a-z0-9_-]{15,63}(?![a-z0-9])"
        ),
    ),
) + tuple(
    (f"telegram_item_{rule_name}", pattern)
    for rule_name, pattern in step3.TELEGRAM_ITEM_PATTERNS
)

# ``darkmarket`` and ``unionstore`` are real fixed-snapshot identities only in
# their contiguous account spelling.  Their separator variants collide with
# ordinary prose (``the dark market`` and ``Western Union store``), so they
# must never enter a separator-invariant or context-assembled registry.
EXACT_CONTIGUOUS_IDENTITY_TOKENS = frozenset({"darkmarket", "unionstore"})
SEPARATOR_INVARIANT_IDENTITY_CONTENT_COLLISION_DENYLIST = (
    EXACT_CONTIGUOUS_IDENTITY_TOKENS
)
EXACT_CONTIGUOUS_IDENTITY_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:darkmarket|unionstore)(?![a-z0-9])"
)

# Step3 intentionally favors precision and therefore misses some marketplace
# obfuscations.  Step7-v3 needs a stricter *removal* boundary: losing a few cue
# words is preferable to letting a source-model baseline read a contact handle.
OBFUSCATED_CONTACT_RULES = (
    (
        "fixed_snapshot_exact_contiguous_collision_prone_identity",
        EXACT_CONTIGUOUS_IDENTITY_RE,
    ),
    (
        "short_at_fragment_after_mail_cue",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:protonmail|gmail|e-?mail)"
            r"[ \t.:/*_~\-]{1,40}@{1,3}[a-z0-9][a-z0-9_.-]{0,3}(?![a-z0-9])"
        ),
    ),
    (
        "obfuscated_wickr_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"w[\W_]{0,6}[i1!][\W_]{0,6}c[\W_]{0,6}k[\W_]{0,6}r"
            r"(?:[\W_]{0,16}(?:app|id|user(?:name)?|handle))?"
            r"[\W_@]{0,64}[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "obfuscated_telegram_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"t[\W_]{0,4}e[\W_]{0,4}l[\W_]{0,4}e[\W_]{0,4}g"
            r"[\W_]{0,4}r[\W_]{0,4}a[\W_]{0,4}m"
            r"(?:[\W_]{0,16}(?:app|id|user(?:name)?|handle))?"
            r"[\W_@]{0,64}[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "obfuscated_kik_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])k[\W_]{0,6}[i1!][\W_]{0,6}k"
            r"(?:[\W_]{0,2}[i1!])?"
            r"(?:[\W_]{0,16}(?:app|id|user(?:name)?|handle))?"
            r"[\W_@]{0,64}[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "obfuscated_whatsapp_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])w[\W_]{0,4}h?[\W_]{0,4}a[\W_]{0,4}t"
            r"[\W_]{0,4}s?[\W_]{0,4}a[\W_]{0,4}p[\W_]{0,4}p"
            r"(?:[\W_]{0,16}(?:app|id|user(?:name)?|handle|text))?"
            r"[\W_@+]{0,64}(?:[a-z0-9][a-z0-9_.-]{3,63}|\d[\d\s().+-]{6,24}\d)"
        ),
    ),
    (
        "defective_wickr_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:w[\W_]{0,3}(?:[i1!][\W_]{0,3})?c[\W_]{0,3}k[\W_]{0,3}(?:r|e)|"
            r"w[\W_]{0,3}[i1!][\W_]{0,3}k[\W_]{0,3}r|"
            r"v[\W_]{0,2}v[\W_]{0,2}[i1!][\W_]{0,2}c[\W_]{0,2}k[\W_]{0,2}r)"
            r"(?:\s+(?:app|id|user(?:name)?|handle)\s*[:=]?\s*|"
            r"\s*[:=@._/\-]{1,64}\s*)"
            r"@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "defective_whatsapp_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:whtspp|whtapp|whatsap|watsapp)"
            r"[ \t.:,;=@_/\-]{1,64}@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "defective_whatsapp_obfuscated_phone",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:whtspp|whtapp|whatsap|watsapp)"
            r"[^a-z0-9|\n]{1,48}\+?(?:x{0,4})?\d"
            r"(?:[ \t().x\-]*\d){5,15}(?![a-z0-9])"
        ),
    ),
    (
        "truncated_telegram_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])telegr(?:am)?"
            r"[ \t.:,;=@_/\-]{1,48}@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "snapchat_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])snap[\W_]{0,3}chat"
            r"(?:\s+(?:id|user(?:name)?|handle)\s*[:=]?\s*|"
            r"\s*[:=@._\-]{1,32}\s*)"
            r"@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "wick_id_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])wick[ \t._:/\-]{1,12}"
            r"(?:app[ \t._:/\-]{0,8})?(?:id|user(?:name)?|handle)"
            r"[ \t:=@._/\-]{0,24}@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "empty_wick_id_cue",
        re.compile(
            r"(?ix)(?<![a-z0-9])wick[ \t._:/\-]{1,12}"
            r"(?:app[ \t._:/\-]{0,8})?(?:id|user(?:name)?|handle)"
            r"[ \t.:=@_/\-]{0,24}(?![a-z0-9])"
        ),
    ),
    (
        "support_staff_dimitri_identity",
        re.compile(
            r"(?ix)(?<![a-z0-9])support[ \t]+staff[ \t]+dimitri"
            r"(?:[ \t]*\([ \t]*support[ \t]+staff[ \t]*\))?"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "fixed_snapshot_james_shipping_signature",
        re.compile(
            r"(?ix)(?<![a-z0-9])shipping[ \t]+worldwide[ \t]+james"
            r"(?=[ \t]*(?:\|\||$))"
        ),
    ),
    (
        "fixed_snapshot_bare_empire_market_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])on[ \t]+empire"
            r"(?:[ \t]+market)?(?:[ \t]+and[ \t]+grey)?(?![a-z0-9])"
        ),
    ),
    (
        "fixed_snapshot_bare_dream_market_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:on|from|since|at|sur)[ \t]+dream"
            r"(?:[ \t]+(?:m|mark|marke|market))?"
            r"(?:[ \t]+(?:and|et)(?:[ \t]+wallstreet)?)?(?![a-z0-9])"
        ),
    ),
    (
        "fixed_snapshot_bare_dream_history_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:"
            r"when[ \t]+dream(?:[ \t]+mark(?:e(?:t)?)?)?|"
            r"dream[ \t]+ratings?|"
            r"feedback[^|\n]{0,32}\bdream(?:[ \t]+(?:et|and))?"
            r")(?![a-z0-9])"
        ),
    ),
    (
        "fixed_snapshot_empire_server_time_context",
        re.compile(
            r"(?i)(?<![a-z0-9])empire[ \t]+server[ \t]+time(?![a-z0-9])"
        ),
    ),
    (
        "fixed_snapshot_genesis_market_resource_context",
        re.compile(
            r"(?i)(?<![a-z0-9])genesis"
            r"(?=[ \t]+(?:tutorial|resources?|wiki)\b)"
        ),
    ),
    (
        "fixed_snapshot_dark_market_versus_list_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])dark[ \t]+market[ \t]*:[ \t]*"
            r"versus[ \t]*:?(?![a-z0-9])"
        ),
    ),
    (
        "fixed_snapshot_colon_versus_market_list_context",
        re.compile(r"(?i)(?<=:)[ \t]*versus[ \t]*:(?![a-z0-9])"),
    ),
    (
        "fixed_snapshot_evolution_market_history_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:bring[ \t]+our[ \t]+)?expertise"
            r"[ \t]+to[ \t]+evolution(?=[.!]?[ \t]+so[ \t]+what[ \t]+is)"
        ),
    ),
    (
        "fixed_snapshot_hansa_market_seo_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:wells[ \t]+frago[ \t]+)?hansa"
            r"[ \t]+chase[ \t]+bank[ \t]+of[ \t]+america(?![a-z0-9])"
        ),
    ),
    (
        "fixed_snapshot_on_dm_market_context",
        re.compile(r"(?i)(?<![a-z0-9])on[ \t]+dm(?![a-z0-9])"),
    ),
    (
        "fixed_snapshot_market_list_grey_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])grey"
            r"(?=[ \t,;:/_\-]{1,24}(?:verified[ \t]+)?vendor\b)"
        ),
    ),
    (
        "fixed_snapshot_dreammarket_german_closure_compound",
        re.compile(
            r"(?i)(?<![a-z0-9])dreammarket(?:s)?schlie(?:ß|ss)ung"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "context_cued_mixed_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:welcome(?:\s+valued\s+clients)?(?:\s+to)?|"
            r"(?:contact|message)(?:\s+(?:us|me))?(?:\s+(?:on|at|via)){0,2}|"
            r"reach(?:\s+out)?(?:\s+to)?(?:\s+(?:us|me))?"
            r"(?:\s+(?:on|at|via)){0,2}|"
            r"(?:user(?:name)?|account|contact)?\s*id)"
            r"[\W_]{1,32}(?:at[\W_]+)?"
            r"(?=[a-z0-9_.-]{4,64}(?![a-z0-9_.-]))"
            r"(?=[a-z0-9_.-]*[a-z])(?=[a-z0-9_.-]*\d)"
            r"[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "at_handle",
        re.compile(
            r"(?i)(?<![\w@])@{1,3}[ \t.:,;_\-~～…]{0,48}[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "bare_obfuscated_wickr_service",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:wi)?w[\W_]{0,6}[i1!][\W_]{0,6}c[\W_]{0,6}k"
            r"[\W_]{0,6}(?:e[\W_]{0,6})?r(?:me|pro|and|l)?(?![a-z0-9])"
        ),
    ),
    (
        "bare_defective_whatsapp_service",
        re.compile(
            r"(?i)(?<![a-z0-9])(?:whtspp|whtapp|whatsap|watsapp)"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "bare_defective_telegram_service",
        re.compile(
            r"(?i)(?<![a-z0-9])telegr[\W_]{0,4}m(?![a-z0-9])"
        ),
    ),
    (
        "bare_obfuscated_telegram_service",
        re.compile(
            r"(?ix)(?<![a-z0-9])t[\W_]{0,4}e[\W_]{0,4}l[\W_]{0,4}e"
            r"[\W_]{0,4}g[\W_]{0,4}r[\W_]{0,4}a[\W_]{0,4}m(?:s)?(?![a-z0-9])"
        ),
    ),
    (
        "bare_defective_wickr_service",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:w[\W_]{0,3}(?:[i1!][\W_]{0,3})?c[\W_]{0,3}k[\W_]{0,3}(?:r|e)|"
            r"w[\W_]{0,3}[i1!][\W_]{0,3}k[\W_]{0,3}r|"
            r"v[\W_]{0,2}v[\W_]{0,2}[i1!][\W_]{0,2}c[\W_]{0,2}k[\W_]{0,2}r)"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "bare_contact_service",
        re.compile(
            r"(?i)(?<![a-z0-9])(?:k[\W_]{0,4}[i1!][\W_]{0,4}k"
            r"(?:[\W_]{0,2}[i1!])?|"
            r"wh?ats?app|jabber|wechat|wkr|hangouts?|icq|skype|discord|pgp)(?![a-z0-9])"
        ),
    ),
)

# Deliberately separate from the substitution rules above.  This is a broader
# output audit written from the shape of forbidden identifiers, so adding a
# narrow cleaning rule cannot make the final manifest "prove itself" merely by
# reusing that same rule list.
FINAL_CORPUS_AUDIT_RULES = (
    (
        "audit_fixed_snapshot_exact_contiguous_collision_prone_identity",
        # Deliberately independent from ``EXACT_CONTIGUOUS_IDENTITY_RE`` so a
        # production-regex edit cannot silently weaken the publication audit.
        re.compile(
            r"(?i)(?<![a-z0-9])(?:darkmarket|unionstore)(?![a-z0-9])"
        ),
    ),
    (
        "audit_email_like",
        re.compile(
            r"(?i)(?<![a-z0-9_+%\-])[a-z0-9][a-z0-9._%+\-]{0,63}"
            r"@[a-z0-9](?:[a-z0-9.\-]{0,252}[a-z0-9])?(?![a-z0-9.\-])"
        ),
    ),
    (
        "audit_missing_at_mail_provider",
        re.compile(
            r"(?ix)(?<![a-z0-9])[a-z0-9][a-z0-9._-]{0,63}"
            r"(?:gmail|protonmail|hotmail|yahoo|outlook|icloud|tutanota|mailfence)"
            r"\.(?:c(?:om?)?|n(?:et?)?|o(?:rg?)?|[a-z]{2,12})?(?![a-z0-9])"
        ),
    ),
    (
        "audit_email_truncated_after_domain_without_tld",
        re.compile(
            r"(?i)(?<![a-z0-9_+%\-])[a-z0-9][a-z0-9._%+\-]{0,63}"
            r"@[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.(?![a-z0-9])"
        ),
    ),
    (
        "audit_concatenated_mail_provider",
        re.compile(
            r"(?ix)(?<![a-z0-9])[a-z0-9][a-z0-9._-]{2,63}"
            r"(?:gmail|protonmail|hotmail|yahoo|outlook|icloud|tutanota|mailfence)"
            r"(?:c(?:om?)?|n(?:et?)?|o(?:rg?)?)(?![a-z0-9])"
        ),
    ),
    ("audit_url", re.compile(r"(?i)(?:https?://|www\.)[^\s|]{3,}")),
    (
        "audit_pgp_material",
        re.compile(
            r"(?is)(?:BEGIN\s+PGP|END\s+PGP|\b(?:[a-f0-9]{4}[\s:.-]?){10}\b)"
        ),
    ),
    (
        "audit_long_contiguous_identifier",
        re.compile(
            r"(?i)(?<![a-z0-9])(?<!\d\.)\d{7,19}(?![a-z0-9])(?!\.\d)"
        ),
    ),
    (
        "audit_plus_phone",
        re.compile(
            r"(?i)(?<![a-z0-9])(?:x{1,6}\s*)?\+\s*"
            r"\d(?:[\s().-]*\d){5,15}(?:\s*x{1,6})?(?![a-z0-9])"
        ),
    ),
    (
        "audit_contact_cued_phone",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:wh?ats?app|phone|telephone|tel|mobile|call|text|sms|contact|reach|message)"
            r"[^|\n]{0,48}?\+?\d(?:[\s().-]*\d){5,15}(?![a-z0-9])"
        ),
    ),
    (
        "audit_contact_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:w[\W_]{0,4}[i1!]c[\W_]{0,4}k[\W_]{0,4}r|"
            r"t[\W_]{0,3}e[\W_]{0,3}l[\W_]{0,3}e[\W_]{0,3}g[\W_]{0,3}r[\W_]{0,3}a[\W_]{0,3}m(?:s)?|"
            r"k[\W_]{0,4}[i1!][\W_]{0,4}k(?:[\W_]{0,2}[i1!])?|"
            r"wh?ats?app|jabber|xmpp|wechat)"
            r"[^|\n]{0,24}[@]?[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "audit_defective_wickr_service",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:w[\W_]{0,4}(?:[i1!][\W_]{0,4})?c[\W_]{0,4}k[\W_]{0,4}(?:r|e)|"
            r"w[\W_]{0,4}[i1!][\W_]{0,4}k[\W_]{0,4}r|"
            r"v[\W_]{0,3}v[\W_]{0,3}[i1!][\W_]{0,3}c[\W_]{0,3}k[\W_]{0,3}r)"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "audit_snapchat_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])snap[^a-z0-9|\n]{0,4}chat"
            r"(?:[ \t]+(?:id|user(?:name)?|handle)[ \t]*[:=]?[ \t]*|"
            r"[ \t]*[:=@._\-]{1,40}[ \t]*)"
            r"@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "audit_defective_whatsapp_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:whtspp|whtapp|whatsap|watsapp)"
            r"[^a-z0-9|\n]{1,64}@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "audit_bare_defective_whatsapp_service",
        re.compile(
            r"(?i)(?<![a-z0-9])(?:whtspp|whtapp|whatsap|watsapp)"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "audit_truncated_telegram_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])telegr(?:am)?"
            r"[^a-z0-9|\n]{1,48}@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "audit_bare_defective_telegram_service",
        re.compile(
            r"(?i)(?<![a-z0-9])telegr[^a-z0-9|\n]{0,4}m(?![a-z0-9])"
        ),
    ),
    (
        "audit_wick_id_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])wick[ \t._:/\-]{1,16}"
            r"(?:app[ \t._:/\-]{0,12})?(?:id|user(?:name)?|handle)"
            r"[ \t:=@._/\-]{0,32}@?[a-z0-9][a-z0-9_.-]{3,63}(?![a-z0-9])"
        ),
    ),
    (
        "audit_empty_wick_id_cue",
        re.compile(
            r"(?ix)(?<![a-z0-9])wick[^a-z0-9|\n]{1,16}"
            r"(?:app[^a-z0-9|\n]{0,12})?(?:id|user(?:name)?|handle)"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "audit_support_staff_dimitri_identity",
        re.compile(
            r"(?ix)(?<![a-z0-9])support[ \t]+staff[ \t]+dimitri"
            r"(?:[ \t]*\([ \t]*support[ \t]+staff[ \t]*\))?"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "audit_fixed_snapshot_james_shipping_signature",
        re.compile(
            r"(?ix)(?<![a-z0-9])shipping[ \t]+worldwide[ \t]+james"
            r"(?=[ \t]*(?:\|\||$))"
        ),
    ),
    (
        "audit_fixed_snapshot_bare_empire_market_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])on[ \t]+empire"
            r"(?:[ \t]+market)?(?:[ \t]+and[ \t]+grey)?(?![a-z0-9])"
        ),
    ),
    (
        "audit_fixed_snapshot_bare_dream_market_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:on|from|since|at|sur)[ \t]+dream"
            r"(?:[ \t]+(?:m|mark|marke|market))?"
            r"(?:[ \t]+(?:and|et)(?:[ \t]+wallstreet)?)?(?![a-z0-9])"
        ),
    ),
    (
        "audit_fixed_snapshot_bare_dream_history_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])(?:when[ \t]+dream|dream[ \t]+ratings?|"
            r"feedback[^|\n]{0,32}\bdream)(?![a-z0-9])"
        ),
    ),
    (
        "audit_fixed_snapshot_empire_server_time_context",
        re.compile(
            r"(?i)(?<![a-z0-9])empire[ \t]+server[ \t]+time(?![a-z0-9])"
        ),
    ),
    (
        "audit_fixed_snapshot_genesis_market_resource_context",
        re.compile(
            r"(?i)(?<![a-z0-9])genesis"
            r"(?=[ \t]+(?:tutorial|resources?|wiki)\b)"
        ),
    ),
    (
        "audit_fixed_snapshot_dark_market_versus_list_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])dark[ \t]+market[^a-z0-9|\n]{1,12}"
            r"versus(?![a-z0-9])"
        ),
    ),
    (
        "audit_fixed_snapshot_colon_versus_market_list_context",
        re.compile(r"(?i)(?<=:)[ \t]*versus[ \t]*:(?![a-z0-9])"),
    ),
    (
        "audit_fixed_snapshot_evolution_market_history_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])expertise[ \t]+to[ \t]+evolution"
            r"(?=[.!]?[ \t]+so[ \t]+what[ \t]+is)"
        ),
    ),
    (
        "audit_fixed_snapshot_hansa_market_seo_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])hansa[ \t]+chase[ \t]+bank"
            r"[ \t]+of[ \t]+america(?![a-z0-9])"
        ),
    ),
    (
        "audit_fixed_snapshot_on_dm_market_context",
        re.compile(r"(?i)(?<![a-z0-9])on[ \t]+dm(?![a-z0-9])"),
    ),
    (
        "audit_fixed_snapshot_market_list_grey_context",
        re.compile(
            r"(?ix)(?<![a-z0-9])grey"
            r"(?=[^a-z0-9|\n]{1,24}(?:verified[ \t]+)?vendor\b)"
        ),
    ),
    (
        "audit_fixed_snapshot_dreammarket_german_closure_compound",
        re.compile(
            r"(?i)(?<![a-z0-9])dreammarket(?:s)?schlie(?:ß|ss)ung"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "audit_context_cued_mixed_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:welcome(?:\s+valued\s+clients)?(?:\s+to)?|"
            r"(?:contact|message|reach)(?:\s+(?:out|to|us|me|on|at|via)){0,6}|"
            r"(?:user(?:name)?|account|contact)?\s*id)"
            r"[^a-z0-9|\n]{1,40}(?:at[^a-z0-9|\n]+)?"
            r"(?=[a-z0-9_.-]{4,64}(?![a-z0-9_.-]))"
            r"(?=[a-z0-9_.-]*[a-z])(?=[a-z0-9_.-]*\d)"
            r"[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "audit_long_id_assignment",
        re.compile(
            r"(?i)(?<![a-z0-9])(?:user(?:name)?|account|contact)?\s*"
            r"id\s*[:=]\s*[a-z0-9][a-z0-9_-]{15,63}(?![a-z0-9])"
        ),
    ),
)

IDENTIFIER_TOKEN_RE = re.compile(
    r"(?i)(?<![a-z0-9])@?[a-z0-9](?:[a-z0-9_-]{0,126}[a-z0-9])?"
    r"(?:\.[a-z0-9_-]+)*(?![a-z0-9])"
)
CONTEXTUAL_ALIAS_CONTENT_WORD_DENYLIST = frozenset(
    {
        "applestore",
        "digital",
        "fedex",
        "however",
        "master",
        "microsoft",
        "opportunity",
        "premium",
        "prime",
        "private",
        "pfizer",
        "quality",
        "support",
        "yahoo",
    }
)
PROTECTED_IDENTITY_COLLISION_TERMS = frozenset(
    {
        "250mg",
        "25i-nbome",
        "bet365",
        "amazon prime store",
        "dark market",
        "dark-market store",
        "darkmarkets",
        "western union store",
    }
)
PROTECTED_IDENTITY_COLLISION_RAW_COUNTS = {
    "250mg": 77,
    "25i-nbome": 3,
    "bet365": 5,
    "amazon prime store": 2,
    "dark market": 56,
    "dark-market store": 18,
    "darkmarkets": 1,
    "western union store": 3,
}
GLOBAL_IDENTITY_CONTENT_COLLISION_DENYLIST = frozenset(
    {"250mg", "25i-nbome", "bet365"}
)
AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS = frozenset(
    {
        "abcdrug",
        "aeromarket",
        "agartha",
        "agarthamarket",
        "agora",
        "agoramarket",
        "ahemweedshop",
        "alphabay",
        "alphabaymarket",
        "apollon",
        "apollonmarket",
        "benzofam",
        "benzoneil",
        "berlusconi",
        "bestplug",
        "bigbluemarket",
        "bitbazzar",
        "biohaz",
        "biohazar",
        "bioteamz",
        "blueheavens",
        "budstore",
        "cannahome",
        "cannazonmarket",
        "cartelmarket",
        "cartelmarketplace",
        "charlieuk",
        "cocaineuk",
        "cryptomarket",
        "cureman",
        "cyphermarket",
        "cyrillebrono",
        "daevamarket",
        "dark0de",
        "darkc0derebornmarket",
        "darkfox",
        "darkfoxmarket",
        "darkkings",
        "darkm",
        "davidesales",
        "digitalempire",
        "dmmarket",
        "dreammarket",
        "drugusa7",
        "drugskingdom",
        "dutchdrugz",
        "dutchmastermarket",
        "empiremarket",
        "evolutionmarket",
        "expectus",
        "flexcompanyshop",
        "flexowned",
        "foreigshop",
        "fortunemmeds",
        "fullplez",
        "genesismarket",
        "globex",
        "goblinking",
        "hackman",
        "hanf4youshop",
        "hansamarket",
        "happyshoporigi",
        "heinekenexpress",
        "hotshop",
        "icarus",
        "icarusmarket",
        "incognitomarket",
        "ixorex",
        "jerry",
        "johnston",
        "kaiserplug",
        "keraniquet",
        "ketams",
        "legaldrugshop",
        "legitconnect",
        "libertymarket",
        "limemarket",
        "limestone",
        "lizardpro",
        "llama",
        "louisegraham",
        "makershop",
        "maxprisc",
        "milkman",
        "misterbitcoin",
        "monoko",
        "monopolymarket",
        "mrdank",
        "mrcodez",
        "mrgreen",
        "myworld",
        "newsale",
        "nightmaremarket",
        "octapustickets",
        "onlydmtfromtj",
        "opiateconnect",
        "originalgermeds",
        "perfectsales",
        "prestigevendor",
        "ravemart",
        "roxstore",
        "royaldreamdocuments",
        "royalmarket",
        "samsara",
        "shorestore",
        "silkroad",
        "silkroad4",
        "simplylsd",
        "sluts",
        "solutioncenter",
        "sr3",
        "teambiohazard",
        "techhacker",
        "topmoneymaker",
        "tor2door",
        "tor2doormarket",
        "traderoute",
        "ukxan",
        "ukxanworld",
        "universalvendor",
        "usadruglord",
        "usapitcher",
        "valhalla",
        "versusmarket",
        "vicecity",
        "vonadolf",
        "wallstreet",
        "wallstreetmarket",
        "weserunionstore",
        "whitehouse",
        "whitehousemarket",
        "whm",
        "worldmarket",
        "wsm",
        "xenite",
        "youngmoney",
        "yourserviceshop",
    }
)
AUDITED_GLOBAL_IDENTITY_DOT_SEPARATOR_TOKENS = frozenset(
    {"bioteamz", "mrdank", "mrcodez"}
)
IDENTITY_HANDLE_SUFFIXES = (
    "vendor",
    "pharma",
    "store",
    "sales",
    "vends",
    "shop",
    "meds",
    "plug",
)
RATING_SUFFIX_RE = re.compile(
    r"(?i)\s*(?:\(|\[)?\s*\d+(?:\.\d+)?\s*%\s*(?:\)|\])?\s*$"
)

SAFE_FEATURE_NAMES = [
    "same_market_bool",
    "same_source_dataset_bool",
    "clean_category_jaccard",
    "clean_shared_title_bool",
    "clean_shared_description_bool",
    "clean_shared_title_count_capped",
    "clean_shared_description_count_capped",
    "clean_shared_category_count_capped",
    "clean_shared_title_idf_sum",
    "clean_shared_description_idf_sum",
    "clean_shared_title_idf_mean",
    "clean_shared_description_idf_mean",
    "item_count_train_percentile_gap_abs",
    "title_length_median_train_percentile_gap_abs",
    "description_length_median_train_percentile_gap_abs",
    "digit_ratio_mean_train_percentile_gap_abs",
    "punct_ratio_mean_train_percentile_gap_abs",
    "repeated_title_share_train_percentile_gap_abs",
    "repeated_description_share_train_percentile_gap_abs",
    "max_category_share_train_percentile_gap_abs",
]

# These two fields are generated only so the final report can quantify the
# dataset shortcut.  They are never legal inputs to an encoder comparison or
# an M0-eligible pipeline.
SHORTCUT_AUDIT_ONLY_FEATURE_NAMES = [
    "same_market_bool",
    "same_source_dataset_bool",
]
MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES = [
    name for name in SAFE_FEATURE_NAMES if name not in SHORTCUT_AUDIT_ONLY_FEATURE_NAMES
]

NUMERIC_PROFILE_FIELDS = {
    "item_count": ("item_count",),
    "title_length_median": ("title_length_stats", "median"),
    "description_length_median": ("description_length_stats", "median"),
    "digit_ratio_mean": ("style_stats", "digit_ratio_mean"),
    "punct_ratio_mean": ("style_stats", "punct_ratio_mean"),
    "repeated_title_share": ("style_stats", "repeated_title_share"),
    "repeated_description_share": ("style_stats", "repeated_description_share"),
    "max_category_share": ("style_stats", "max_category_share"),
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def bool_value(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def render_csv(rows: list[dict], fieldnames: list[str] | None = None) -> bytes:
    if not rows:
        raise ValueError("Step7-v3 refuses to render an empty CSV")
    fields = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_bytes_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different Step7-v3 artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_csv_immutable(
    path: Path, rows: list[dict], fieldnames: list[str] | None = None
) -> None:
    write_bytes_immutable(path, render_csv(rows, fieldnames))


def write_json_immutable(path: Path, payload: dict) -> None:
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_bytes_immutable(path, rendered)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Atomically refresh an operational manifest that is intentionally mutable."""
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(rendered)
    temporary.replace(path)


def write_jsonl_immutable(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Step7-v3 refuses to render empty JSONL")
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    write_bytes_immutable(path, rendered)


def write_npy_immutable(path: Path, matrix: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(matrix))
    write_bytes_immutable(path, buffer.getvalue())




def validate_policy(policy: dict) -> None:
    """Validate only the frozen source-data contract used by Step7-v3.1."""
    if policy.get("version") != "2026-07-22-step7-v3.1-source-data-v1":
        raise ValueError("Step7-v3.1 source-data policy version drift")
    implementation = policy.get("implementation", {})
    if set(implementation) != {
        "source_data_module",
        "preparation_script",
        "redaction_dependency",
    }:
        raise ValueError("Step7-v3.1 source implementation universe drift")
    expected_implementation_paths = {
        "source_data_module": "scripts/step7_v3_1_source_data.py",
        "preparation_script": "scripts/step7_v3_1_prepare_source_data.py",
        "redaction_dependency": "scripts/step3_build_seller_profiles.py",
    }
    for role, expected_path in expected_implementation_paths.items():
        record = implementation[role]
        if (
            set(record) != {"path", "size_bytes", "sha256"}
            or record["path"] != expected_path
            or int(record["size_bytes"]) <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))
        ):
            raise ValueError(f"Step7-v3.1 source implementation pin drift: {role}")
    required_inputs = {
        "frozen_labels",
        "evidence_labels",
        "seller_profiles",
        "item_identity_signals",
        "component_assignments",
    }
    if set(policy.get("inputs", {})) != required_inputs:
        raise ValueError("Step7-v3.1 source input universe drift")
    boundary = policy["supervision_boundary"]
    expected = boundary["expected_counts"]
    if list(boundary["eligible_split_names"]) != ["train", "valid", "test"]:
        raise ValueError("Step7-v3.1 source split order drift")
    if list(boundary["eligible_labels"]) != ["positive", "negative"]:
        raise ValueError("Step7-v3.1 source label universe drift")
    if sum(int(expected[name]["total"]) for name in ("train", "valid", "test")) != int(
        expected["total"]
    ):
        raise ValueError("Step7-v3.1 source split counts do not sum")
    for split in ("train", "valid", "test"):
        row = expected[split]
        if int(row["positive"]) + int(row["negative"]) != int(row["total"]):
            raise ValueError(f"Step7-v3.1 source label counts drift: {split}")
    fields = [
        "category_concat_top",
        "signature_title_concat",
        "title_concat_top",
        "signature_description_concat",
        "description_concat_top",
    ]
    clean = policy["clean_text_contract"]
    if clean["fields_in_order"] != fields:
        raise ValueError("Step7-v3.1 source field order drift")
    if clean["replacement"] != "single_space_no_identifier_presence_marker":
        raise ValueError("Step7-v3.1 source redaction marker drift")
    if clean["empty_text_fallback"] != "content unavailable":
        raise ValueError("Step7-v3.1 source empty-text fallback drift")
    if policy["safe_pair_features"] != SAFE_FEATURE_NAMES:
        raise ValueError("Step7-v3.1 source safe-feature order drift")
    roles = policy["pair_feature_roles"]
    if roles["shortcut_audit_only_features"] != SHORTCUT_AUDIT_ONLY_FEATURE_NAMES:
        raise ValueError("Step7-v3.1 source shortcut-feature role drift")
    if roles["model_eligible_transfer_features"] != MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES:
        raise ValueError("Step7-v3.1 source transfer-feature role drift")
    outputs = policy["outputs"]
    root = str(outputs["root"]).rstrip("/")
    if not root.startswith("reports/step7_v3_1_full_text_chunked_selection/"):
        raise ValueError("Step7-v3.1 source output root drift")
    if set(outputs) != {
        "root",
        "pair_manifest",
        "field_corpus",
        "train_feature_reference",
        "safe_pair_features",
        "train_labels",
        "valid_labels",
        "preparation_manifest",
        "development_labels_manifest",
    }:
        raise ValueError("Step7-v3.1 source output universe drift")
    if any(
        not str(value).startswith(root + "/")
        for key, value in outputs.items()
        if key != "root"
    ):
        raise ValueError("Step7-v3.1 source output escapes versioned root")
    expected_artifacts = policy["expected_artifacts"]
    if set(expected_artifacts) != {
        "pair_manifest",
        "field_corpus",
        "train_feature_reference",
        "safe_pair_features",
        "train_labels",
        "valid_labels",
    }:
        raise ValueError("Step7-v3.1 expected artifact universe drift")
    for role, record in expected_artifacts.items():
        if set(record) != {"sha256", "size_bytes"}:
            raise ValueError(f"Step7-v3.1 expected artifact schema drift: {role}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
            raise ValueError(f"Step7-v3.1 expected artifact hash drift: {role}")
        if int(record["size_bytes"]) <= 0:
            raise ValueError(f"Step7-v3.1 expected artifact size drift: {role}")


def validate_input_hashes(
    policy: dict, input_names: Iterable[str] | None = None
) -> dict[str, dict]:
    output = {}
    selected = list(input_names) if input_names is not None else list(policy["inputs"])
    for input_name in selected:
        spec = policy["inputs"][input_name]
        path = resolve(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step7-v3 input {input_name}: {path}")
        observed = sha256_file(path)
        expected = str(spec["sha256"]).casefold()
        if observed != expected:
            raise ValueError(
                f"Step7-v3 input hash drift for {input_name}: expected={expected} observed={observed}"
            )
        output[input_name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": observed,
            "size_bytes": path.stat().st_size,
        }
    return output


def validate_content_fidelity_manifest(policy: dict, manifest: dict) -> None:
    quality = policy["clean_text_contract"]["quality_gates"]
    fidelity = manifest.get("content_fidelity", {})
    raw_count = fidelity.get("raw_source_field_character_count")
    clean_count = fidelity.get("clean_model_text_character_count")
    retention = fidelity.get("aggregate_character_retention")
    if (
        fidelity.get("quality_gates_passed") is not True
        or not isinstance(raw_count, int)
        or raw_count <= 0
        or not isinstance(clean_count, int)
        or clean_count <= 0
        or not isinstance(retention, (int, float))
        or not math.isfinite(float(retention))
        or not math.isclose(float(retention), clean_count / raw_count, abs_tol=1e-12)
        or float(retention) < float(quality["minimum_aggregate_character_retention"])
    ):
        raise ValueError("Step7-v3 public content-fidelity gate is invalid")
    fallback_count = fidelity.get("empty_text_fallback_count")
    if not isinstance(fallback_count, int) or fallback_count > int(
        quality["maximum_empty_fallback_count"]
    ):
        raise ValueError("Step7-v3 public empty-content gate is invalid")
    protected = fidelity.get("protected_content_word_retention", {})
    if set(protected) != set(quality["protected_content_words"]):
        raise ValueError("Step7-v3 public protected-word audit universe mismatch")
    for word, record in protected.items():
        raw_word_count = record.get("raw_count")
        clean_word_count = record.get("clean_count")
        word_retention = record.get("retention")
        if (
            not isinstance(raw_word_count, int)
            or raw_word_count <= 0
            or not isinstance(clean_word_count, int)
            or clean_word_count < 0
            or not isinstance(word_retention, (int, float))
            or not math.isclose(
                float(word_retention), clean_word_count / raw_word_count, abs_tol=1e-12
            )
            or float(word_retention) < float(quality["minimum_protected_word_retention"])
        ):
            raise ValueError(
                f"Step7-v3 public protected-word retention is invalid: {word}"
            )
    protected_collisions = fidelity.get(
        "protected_identity_collision_term_retention", {}
    )
    if set(protected_collisions) != set(
        quality["protected_identity_collision_terms"]
    ):
        raise ValueError("Step7-v3 identity-collision fidelity universe mismatch")
    for term, record in protected_collisions.items():
        raw_count = record.get("raw_count")
        clean_count = record.get("clean_count")
        retention = record.get("retention")
        expected_raw_count = quality[
            "expected_protected_identity_collision_raw_counts"
        ][term]
        if (
            not isinstance(raw_count, int)
            or raw_count <= 0
            or raw_count != expected_raw_count
            or not isinstance(clean_count, int)
            or clean_count < 0
            or not isinstance(retention, (int, float))
            or not math.isclose(
                float(retention), clean_count / raw_count, abs_tol=1e-12
            )
            or float(retention)
            < float(
                quality["minimum_protected_identity_collision_term_retention"]
            )
        ):
            raise ValueError(
                f"Step7-v3 identity-collision fidelity gate is invalid: {term}"
            )


def validate_global_identity_audit_manifest(policy: dict, manifest: dict) -> None:
    expected = policy["clean_text_contract"]["global_mixed_alias_expected_audit"]
    registry_count = manifest.get("signal_summary", {}).get(
        "global_identity_token_count"
    )
    audit = manifest.get("signal_summary", {}).get(
        "global_identity_fixed_snapshot_audit", {}
    )
    if audit.get("status") != (
        "pass_registry_is_input_hash_pinned_and_content_collisions_are_preregistered"
    ):
        raise ValueError("Step7-v3 global-identity audit status is invalid")
    if audit.get("scan_scope") != (
        "actual_post_audited_phrase_post_generic_post_local_alias_global_redactions_in_855_seller_five_field_corpus"
    ):
        raise ValueError("Step7-v3 global-identity audit scope is invalid")
    hashed_counts = audit.get("removed_token_sha256_counts", {})
    distinct_count = audit.get("removed_distinct_token_count")
    occurrence_count = audit.get("removed_occurrence_count")
    if (
        not isinstance(distinct_count, int)
        or distinct_count <= 0
        or not isinstance(occurrence_count, int)
        or occurrence_count <= 0
        or len(hashed_counts) != distinct_count
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(token_hash))
            or not isinstance(count, int)
            or count <= 0
            for token_hash, count in hashed_counts.items()
        )
        or sum(hashed_counts.values()) != occurrence_count
        or registry_count != expected["registry_token_count_after_denylist"]
        or distinct_count != expected["removed_distinct_token_count"]
        or occurrence_count != expected["removed_occurrence_count"]
        or canonical_hash(hashed_counts)
        != expected["removed_token_sha256_counts_canonical_sha256"]
    ):
        raise ValueError("Step7-v3 global-identity hashed audit counts are invalid")
    if set(audit.get("content_collision_denylist", [])) != set(
        GLOBAL_IDENTITY_CONTENT_COLLISION_DENYLIST
    ):
        raise ValueError("Step7-v3 global-identity audit denylist drift")
    if audit.get("input_hash_change_requires_full_reaudit") is not True:
        raise ValueError("Step7-v3 global-identity audit is not input-hash scoped")
    if occurrence_count != int(
        manifest.get("redaction_summary", {}).get(
            "global_identifier_token_match_count", -1
        )
    ):
        raise ValueError("Step7-v3 global-identity audit/redaction count mismatch")

    phrase_expected = policy["clean_text_contract"][
        "audited_global_identity_phrase_expected_audit"
    ]
    phrase_registry_count = manifest.get("signal_summary", {}).get(
        "audited_global_identity_phrase_token_count"
    )
    phrase_audit = manifest.get("signal_summary", {}).get(
        "audited_global_identity_phrase_fixed_snapshot_audit", {}
    )
    if phrase_audit.get("status") != (
        "pass_manual_seller_and_market_identity_classification_is_public_input_hash_pinned"
    ):
        raise ValueError("Step7-v3 audited identity-phrase status is invalid")
    if phrase_audit.get("scan_scope") != (
        "actual_first_stage_separator_invariant_phrase_redactions_in_855_seller_five_field_corpus"
    ):
        raise ValueError("Step7-v3 audited identity-phrase scope is invalid")
    phrase_hashed_counts = phrase_audit.get(
        "removed_surface_sha256_counts", {}
    )
    phrase_matched_registry_count = phrase_audit.get(
        "matched_registry_token_count"
    )
    phrase_unmatched_registry_count = phrase_audit.get(
        "unmatched_preventive_registry_token_count"
    )
    phrase_distinct_count = phrase_audit.get("removed_distinct_surface_count")
    phrase_occurrence_count = phrase_audit.get("removed_occurrence_count")
    if (
        not isinstance(phrase_registry_count, int)
        or phrase_registry_count != phrase_expected["registry_token_count"]
        or phrase_audit.get("registry_token_count") != phrase_registry_count
        or phrase_audit.get("registry_tokens_canonical_sha256")
        != phrase_expected["registry_tokens_canonical_sha256"]
        or not isinstance(phrase_matched_registry_count, int)
        or phrase_matched_registry_count
        != phrase_expected["matched_registry_token_count"]
        or phrase_audit.get("matched_registry_tokens_canonical_sha256")
        != phrase_expected["matched_registry_tokens_canonical_sha256"]
        or not isinstance(phrase_unmatched_registry_count, int)
        or phrase_unmatched_registry_count
        != phrase_expected["unmatched_preventive_registry_token_count"]
        or phrase_matched_registry_count + phrase_unmatched_registry_count
        != phrase_registry_count
        or not isinstance(phrase_distinct_count, int)
        or phrase_distinct_count <= 0
        or not isinstance(phrase_occurrence_count, int)
        or phrase_occurrence_count <= 0
        or len(phrase_hashed_counts) != phrase_distinct_count
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(token_hash))
            or not isinstance(count, int)
            or count <= 0
            for token_hash, count in phrase_hashed_counts.items()
        )
        or sum(phrase_hashed_counts.values()) != phrase_occurrence_count
        or phrase_distinct_count
        != phrase_expected["removed_distinct_surface_count"]
        or phrase_occurrence_count != phrase_expected["removed_occurrence_count"]
        or canonical_hash(phrase_hashed_counts)
        != phrase_expected[
            "removed_surface_sha256_counts_canonical_sha256"
        ]
    ):
        raise ValueError(
            "Step7-v3 audited identity-phrase hashed counts are invalid"
        )
    expected_public_input_hashes = {
        name: str(policy["inputs"][name]["sha256"]).casefold()
        for name in (
            "component_assignments",
            "item_identity_signals",
            "seller_profiles",
        )
    }
    if phrase_audit.get("audited_public_input_sha256") != (
        expected_public_input_hashes
    ):
        raise ValueError(
            "Step7-v3 audited identity-phrase input hashes are invalid"
        )
    expected_collision_compacts = sorted(
        compact_identifier(term) for term in PROTECTED_IDENTITY_COLLISION_TERMS
    )
    if (
        phrase_audit.get("protected_content_collision_compacts")
        != expected_collision_compacts
        or phrase_audit.get(
            "registry_is_disjoint_from_protected_content_collisions"
        )
        is not True
        or phrase_audit.get("input_hash_change_requires_full_reaudit") is not True
    ):
        raise ValueError(
            "Step7-v3 audited identity-phrase collision/hash scope is invalid"
        )
    if phrase_occurrence_count != int(
        manifest.get("redaction_summary", {}).get(
            "audited_global_identity_phrase_match_count", -1
        )
    ):
        raise ValueError(
            "Step7-v3 audited identity-phrase audit/redaction count mismatch"
        )

    def valid_hashed_count_map(
        value: object,
        expected_length: int,
        expected_total: int | None = None,
    ) -> bool:
        if not isinstance(value, dict) or len(value) != expected_length:
            return False
        if any(
            re.fullmatch(r"[0-9a-f]{64}", str(key)) is None
            or type(count) is not int
            or count <= 0
            for key, count in value.items()
        ):
            return False
        return expected_total is None or sum(value.values()) == expected_total

    full_expected = policy["clean_text_contract"][
        "full_known_alias_residual_expected_audit"
    ]
    full_census = manifest.get("signal_summary", {}).get(
        "full_known_alias_residual_fixed_snapshot_census", {}
    )
    full_anchor_counts = full_census.get("matched_anchor_sha256_counts")
    full_surface_counts = full_census.get("matched_surface_sha256_counts")
    full_seller_counts = full_census.get(
        "matched_anchor_sha256_seller_counts"
    )
    full_occurrence_count = full_expected["matched_occurrence_count"]
    full_integer_fields = (
        "registry_token_count",
        "scanned_seller_row_count",
        "scanned_segment_count",
        "matched_registry_token_count",
        "matched_occurrence_count",
        "matched_surface_count",
        "confirmed_identity_residual_anchor_count",
        "retained_ambiguous_or_content_collision_anchor_count",
        "retained_ambiguous_or_content_collision_occurrence_count",
    )
    if (
        full_census.get("status")
        != "pass_full_fixed_snapshot_known_alias_census_completed"
        or full_census.get("scan_scope")
        != (
            "serialized_final_model_text_each_seller_row_newline_field_and_"
            "double_pipe_value_scanned_independently"
        )
        or full_census.get("matching_contract")
        != (
            "independent_longest_separator_invariant_exact_then_registry_"
            "anchored_plural_or_identity_suffix"
        )
        or any(type(full_census.get(key)) is not int for key in full_integer_fields)
        or full_census.get("registry_token_count")
        != full_expected["registry_token_count"]
        or manifest.get("signal_summary", {}).get(
            "contextual_global_alias_token_count"
        )
        != full_expected["registry_token_count"]
        or full_census.get("scanned_seller_row_count")
        != full_expected["scanned_seller_row_count"]
        or full_census.get("scanned_segment_count")
        != full_expected["scanned_segment_count"]
        or full_census.get("matched_registry_token_count")
        != full_expected["matched_registry_token_count"]
        or full_census.get("matched_occurrence_count")
        != full_occurrence_count
        or full_census.get("matched_surface_count")
        != full_expected["matched_surface_count"]
        or full_census.get("confirmed_identity_residual_anchor_count")
        != full_expected["confirmed_identity_residual_anchor_count"]
        or full_census.get(
            "retained_ambiguous_or_content_collision_anchor_count"
        )
        != full_expected[
            "retained_ambiguous_or_content_collision_anchor_count"
        ]
        or full_census.get(
            "retained_ambiguous_or_content_collision_occurrence_count"
        )
        != full_expected[
            "retained_ambiguous_or_content_collision_occurrence_count"
        ]
        or full_census.get("confirmed_identity_residual_anchor_count")
        + full_census.get(
            "retained_ambiguous_or_content_collision_anchor_count"
        )
        != full_census.get("matched_registry_token_count")
        or full_census.get(
            "retained_ambiguous_or_content_collision_occurrence_count"
        )
        != full_occurrence_count
        or full_census.get("match_kind_counts")
        != full_expected["match_kind_counts"]
        or sum(full_expected["match_kind_counts"].values())
        != full_occurrence_count
        or not valid_hashed_count_map(
            full_anchor_counts,
            full_expected["matched_registry_token_count"],
            full_occurrence_count,
        )
        or not valid_hashed_count_map(
            full_surface_counts,
            full_expected["matched_surface_count"],
            full_occurrence_count,
        )
        or not valid_hashed_count_map(
            full_seller_counts,
            full_expected["matched_registry_token_count"],
        )
        or set(full_seller_counts) != set(full_anchor_counts)
        or any(
            seller_count > full_anchor_counts[token_hash]
            or seller_count > full_expected["scanned_seller_row_count"]
            for token_hash, seller_count in full_seller_counts.items()
        )
        or canonical_hash(full_anchor_counts)
        != full_expected[
            "matched_anchor_sha256_counts_canonical_sha256"
        ]
        or canonical_hash(full_surface_counts)
        != full_expected[
            "matched_surface_sha256_counts_canonical_sha256"
        ]
        or canonical_hash(full_seller_counts)
        != full_expected[
            "matched_anchor_sha256_seller_counts_canonical_sha256"
        ]
    ):
        raise ValueError("Step7-v3 full known-alias census is invalid")
    if (
        full_census.get("manual_review_contract")
        != (
            "all_residual_known_alias_anchors_are_reviewed_from_fixed_snapshot_"
            "model_text_and_alias_semantics_without_labels_or_evidence_types_"
            "confirmed_identity_anchors_must_move_to_the_unconditional_registry"
        )
        or full_census.get("manual_review_outcome")
        != (
            "pass_no_confirmed_identity_anchor_remains_all_retained_anchors_are_"
            "ambiguous_or_content_collisions"
        )
        or full_census.get("audited_public_input_sha256")
        != expected_public_input_hashes
        or full_census.get("input_hash_change_requires_full_reaudit") is not True
        or full_census.get("unknown_or_ambiguous_identifier_absence_proven")
        is not False
    ):
        raise ValueError("Step7-v3 full known-alias audit scope is invalid")

    embedded_expected = policy["clean_text_contract"][
        "audited_identity_embedded_residual_expected_audit"
    ]
    embedded_census = manifest.get("signal_summary", {}).get(
        "audited_identity_embedded_residual_fixed_snapshot_census", {}
    )
    embedded_anchor_counts = embedded_census.get(
        "matched_anchor_sha256_counts"
    )
    embedded_surface_counts = embedded_census.get(
        "matched_surface_sha256_counts"
    )
    embedded_pair_counts = embedded_census.get(
        "matched_alias_surface_pair_sha256_counts"
    )
    embedded_occurrence_count = embedded_expected["matched_occurrence_count"]
    embedded_integer_fields = (
        "registry_token_count",
        "scanned_seller_row_count",
        "matched_registry_token_count",
        "matched_occurrence_count",
        "matched_alias_surface_pair_count",
        "matched_surface_count",
        "confirmed_identity_residual_count",
        "retained_content_collision_occurrence_count",
    )
    if (
        embedded_census.get("status")
        != "pass_fixed_snapshot_embedded_identity_census_completed"
        or embedded_census.get("scan_scope")
        != (
            "every_ascii_alphanumeric_word_in_serialized_final_model_text_"
            "after_anchored_identity_forms_are_excluded"
        )
        or embedded_census.get("matching_contract")
        != (
            "strict_substring_of_audited_seller_or_market_identity_registry_"
            "without_reusing_redaction_matchers"
        )
        or any(
            type(embedded_census.get(key)) is not int
            for key in embedded_integer_fields
        )
        or embedded_census.get("registry_token_count")
        != embedded_expected["registry_token_count"]
        or embedded_census.get("registry_token_count") != phrase_registry_count
        or embedded_census.get("scanned_seller_row_count")
        != embedded_expected["scanned_seller_row_count"]
        or embedded_census.get("matched_registry_token_count")
        != embedded_expected["matched_registry_token_count"]
        or embedded_census.get("matched_occurrence_count")
        != embedded_occurrence_count
        or embedded_census.get("matched_alias_surface_pair_count")
        != embedded_expected["matched_alias_surface_pair_count"]
        or embedded_census.get("matched_surface_count")
        != embedded_expected["matched_surface_count"]
        or embedded_census.get("confirmed_identity_residual_count")
        != embedded_expected["confirmed_identity_residual_count"]
        or embedded_census.get("retained_content_collision_occurrence_count")
        != embedded_expected["retained_content_collision_occurrence_count"]
        or embedded_census.get("confirmed_identity_residual_count")
        + embedded_census.get("retained_content_collision_occurrence_count")
        != embedded_occurrence_count
        or not valid_hashed_count_map(
            embedded_anchor_counts,
            embedded_expected["matched_registry_token_count"],
            embedded_occurrence_count,
        )
        or not valid_hashed_count_map(
            embedded_pair_counts,
            embedded_expected["matched_alias_surface_pair_count"],
            embedded_occurrence_count,
        )
        or not valid_hashed_count_map(
            embedded_surface_counts,
            embedded_expected["matched_surface_count"],
        )
        or sum(embedded_surface_counts.values()) != embedded_occurrence_count
        or canonical_hash(embedded_anchor_counts)
        != embedded_expected[
            "matched_anchor_sha256_counts_canonical_sha256"
        ]
        or canonical_hash(embedded_surface_counts)
        != embedded_expected[
            "matched_surface_sha256_counts_canonical_sha256"
        ]
        or canonical_hash(embedded_pair_counts)
        != embedded_expected[
            "matched_alias_surface_pair_sha256_counts_canonical_sha256"
        ]
    ):
        raise ValueError("Step7-v3 embedded identity census is invalid")
    if (
        embedded_census.get("manual_review_contract")
        != (
            "all_embedded_candidates_are_reviewed_from_fixed_snapshot_model_text_"
            "and_alias_semantics_without_labels_or_evidence_types_confirmed_"
            "identity_residues_must_be_removed_before_release"
        )
        or embedded_census.get("manual_review_outcome")
        != (
            "pass_all_candidates_are_biohazard_tool_name_wallstreetbet_or_"
            "darkmarkets_content_collisions"
        )
        or embedded_census.get("retained_content_collision_scope")
        != (
            "biohazard_tool_name_wallstreetbet_and_darkmarkets_contains_"
            "darkm_only"
        )
        or embedded_census.get("audited_public_input_sha256")
        != expected_public_input_hashes
        or embedded_census.get("input_hash_change_requires_full_reaudit")
        is not True
        or embedded_census.get("unknown_or_ambiguous_identifier_absence_proven")
        is not False
    ):
        raise ValueError("Step7-v3 embedded identity audit scope is invalid")

    residue_scan = manifest.get("identity_residue_scan", {})
    residue_count_fields = (
        "pattern_residue_count",
        "seller_local_identity_literal_residue_count",
        "seller_local_separator_variant_residue_count",
        "audited_global_identity_phrase_residue_count",
        "known_global_high_confidence_handle_residue_count",
        "context_gated_known_alias_residue_count",
    )
    pattern_counts = residue_scan.get("pattern_residue_count_by_rule")
    if (
        residue_scan.get("status") != "pass"
        or residue_scan.get("claim_scope") != IDENTITY_RESIDUE_CLAIM_SCOPE
        or residue_scan.get("claim_scope")
        != policy["clean_text_contract"]["identity_residue_claim_scope"]
        or residue_scan.get("unknown_identifier_absence_proven") is not False
        or residue_scan.get("scan_scope")
        != "serialized_final_model_text_each_seller_row_independently"
        or type(residue_scan.get("seller_row_count")) is not int
        or residue_scan.get("seller_row_count")
        != full_expected["scanned_seller_row_count"]
        or any(
            type(residue_scan.get(key)) is not int
            or residue_scan.get(key) < 0
            for key in (*residue_count_fields, "total_residue_count")
        )
        or not isinstance(pattern_counts, dict)
        or any(
            not isinstance(rule_name, str)
            or not rule_name
            or type(count) is not int
            or count <= 0
            for rule_name, count in pattern_counts.items()
        )
        or sum(pattern_counts.values())
        != residue_scan.get("pattern_residue_count")
        or sum(residue_scan[key] for key in residue_count_fields)
        != residue_scan.get("total_residue_count")
        or residue_scan.get("total_residue_count") != 0
    ):
        raise ValueError("Step7-v3 final identity-residue scan is invalid")


def validate_expected_model_pin(model_key: str, cfg: dict) -> None:
    expected_hash = str(cfg.get("expected_content_sha256", "")).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"Step7-v3 model content hash is not pinned for {model_key}")
    if int(cfg.get("expected_file_count", 0)) <= 0:
        raise ValueError(f"Step7-v3 model file count is not pinned for {model_key}")
    if int(cfg.get("expected_total_size_bytes", 0)) <= 1024 * 1024:
        raise ValueError(f"Step7-v3 model payload size is not pinned for {model_key}")


def safe_signal_literal(contact_type: str, value: str) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    compact = re.sub(r"\s+", "", token)
    if contact_type == "seller_alias":
        if compact.isdigit():
            return token if len(compact) >= 5 else None
        if re.search(r"[\u3400-\u9fff]", compact):
            return token if len(compact) >= 2 else None
        return token if len(compact) >= 4 else None
    if contact_type in {"qq", "phone"}:
        return token if len(re.sub(r"\D", "", compact)) >= 5 else None
    if contact_type in {"pgp_public_key", "pgp_fingerprint", "crypto_wallet"}:
        return token if len(compact) >= 12 else None
    return token if len(compact) >= 4 else None


def seller_alias_variants(value: object) -> list[str]:
    original = str(value or "").strip()
    if not original:
        return []
    stripped = RATING_SUFFIX_RE.sub("", original).strip(" \t-_|,;:")
    return sorted(
        {candidate for candidate in (original, stripped) if candidate},
        key=lambda candidate: (-len(candidate), candidate.casefold()),
    )


def canonical_identifier_token(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"^[.@_-]+|[.@_-]+$", "", normalized)


def high_confidence_global_token(value: object) -> str | None:
    token = canonical_identifier_token(value)
    compact = re.sub(r"[^a-z0-9]", "", token)
    if not (5 <= len(compact) <= 96):
        return None
    # Cross-seller global removal is deliberately conservative.  Pure aliases
    # use a separate context gate; mixed handles are safe to remove everywhere
    # and cover cloned advertisements such as legitmed007.
    if not (re.search(r"[a-z]", compact) and re.search(r"\d", compact)):
        return None
    return token


def global_identity_tokens(
    literals_by_seller: dict[str, list[str]], profiles: Iterable[dict]
) -> set[str]:
    signal_tokens: set[str] = set()
    for values in literals_by_seller.values():
        for value in values:
            if token := high_confidence_global_token(value):
                signal_tokens.add(token)
    profile_tokens: set[str] = set()
    for profile in profiles:
        for field in ("source_seller_raw", "alias_normalized"):
            for value in seller_alias_variants(profile.get(field, "")):
                if token := high_confidence_global_token(value):
                    profile_tokens.add(token)
    # The fixed-snapshot source union has been audited against the 855-seller
    # corpus.  Three mixed tokens were content collisions.  Separately audited
    # pure or separator-bearing identities are removed by the phrase registry,
    # not silently folded into this mixed-token registry.
    return (profile_tokens | signal_tokens) - set(
        GLOBAL_IDENTITY_CONTENT_COLLISION_DENYLIST
    )


def matches_global_identity_token(
    value: object, registry: set[str] | frozenset[str]
) -> bool:
    """Match an exact known handle or a known handle plus identity suffix.

    The suffix rule is deliberately registry-anchored: a generic token ending
    in ``shop`` is not removed unless its base is already a pinned, label-free
    identity token.
    """
    token = canonical_identifier_token(value)
    if token in registry:
        return True
    if token.endswith("s") and token[:-1] in registry:
        return True
    for suffix in IDENTITY_HANDLE_SUFFIXES:
        base = token[: -len(suffix)].rstrip("_-") if token.endswith(suffix) else ""
        if base in registry:
            return True
    return False


CONTEXTUAL_ALIAS_GENERIC_DENYLIST = frozenset(
    {"seller", "vendor", "market", "unknown", "shop", "store", "team"}
)
CONTEXTUAL_ALIAS_WORD_RE = re.compile(
    r"(?i)(?<![a-z0-9])@?[a-z0-9][a-z0-9_-]{0,95}(?![a-z0-9])"
)
# Unconditional audited identities must also be detectable when a previous
# redaction exposes an alias beginning after a hyphen or underscore inside a
# larger token (for example ``hi-tech programmers hackers`` ->
# ``hi-tech hackers``).  Keeping this scanner separate avoids widening the
# context-gated alias matcher.
UNCONDITIONAL_ALIAS_WORD_RE = re.compile(
    r"(?i)(?<![a-z0-9])@?[a-z0-9]{1,96}(?![a-z0-9])"
)
CONTEXTUAL_ALIAS_PHRASE_GAP_RE = re.compile(r"[ \t._-]{1,8}")
UNCONDITIONAL_ALIAS_PHRASE_GAP_RE = re.compile(r"[ \t_-]{1,8}")
KNOWN_ALIAS_CENSUS_WORD_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
KNOWN_ALIAS_CENSUS_GAP_RE = re.compile(
    # A dot is accepted only inside a contiguous display handle (``mr.code``),
    # never together with whitespace where it could cross a sentence boundary.
    r"(?:[ \t_-]{1,8}|[._-]{1,8})"
)


def compact_identifier(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9]", "", normalized)


def contextual_global_alias_tokens(
    profiles: Iterable[dict],
    identity_literals_by_seller: dict[str, list[str]] | None = None,
) -> set[str]:
    """Build compact, label-free aliases used only under identity context."""
    tokens: set[str] = set()

    def add(compact: str, minimum_length: int) -> None:
        if (
            minimum_length <= len(compact) <= 96
            and compact not in CONTEXTUAL_ALIAS_GENERIC_DENYLIST
            and compact not in CONTEXTUAL_ALIAS_CONTENT_WORD_DENYLIST
            and compact
            not in SEPARATOR_INVARIANT_IDENTITY_CONTENT_COLLISION_DENYLIST
        ):
            tokens.add(compact)

    def add_value(value: object) -> None:
        compact = compact_identifier(value)
        add(compact, 5)
        # Marketplace aliases frequently append a changing year or account
        # number (Daha2020 -> DAHA, DrFRAUD51 -> DrFRAUD).
        without_tail_digits = re.sub(r"\d{1,4}$", "", compact)
        if without_tail_digits != compact:
            add(without_tail_digits, 4)

    for profile in profiles:
        for field in ("source_seller_raw", "alias_normalized"):
            for value in seller_alias_variants(profile.get(field, "")):
                add_value(value)
    # Every signal admitted by ``safe_signal_literal`` enters this *context
    # gated* registry.  Restricting this loop to mixed alpha-numeric handles
    # would silently drop pure identities such as ``OCTAPUSTICKETS`` and would
    # contradict ``direct_signal_usage_contract``.  Only the unconditional
    # global registry remains restricted to high-confidence mixed handles.
    for values in (identity_literals_by_seller or {}).values():
        for value in values:
            add_value(value)
    return tokens


def contextual_alias_deletion_tokens(
    registry: set[str] | frozenset[str],
) -> set[str]:
    """One-character omissions for long alphabetic aliases, cue-gated only."""
    variants: set[str] = set()
    for token in registry:
        if token.isalpha() and 8 <= len(token) <= 32:
            variants.update(token[:index] + token[index + 1 :] for index in range(len(token)))
    return {
        token
        for token in variants
        if len(token) >= 7
        and token not in registry
        and token not in CONTEXTUAL_ALIAS_GENERIC_DENYLIST
        and token not in CONTEXTUAL_ALIAS_CONTENT_WORD_DENYLIST
        and token not in SEPARATOR_INVARIANT_IDENTITY_CONTENT_COLLISION_DENYLIST
    }


CONTEXTUAL_ALIAS_BEFORE_CUE_RE = re.compile(
    r"(?<![a-z0-9])(?:welcome(?:[ \t]+valued[ \t]+clients(?:[ \t]+to)?|[ \t]+to)|"
    r"here[ \t]+at|"
    r"(?:seller|vendor|username|shop|store|team)[ \t]*[:=\-]?|"
    r"(?:find|search)[ \t]+(?:us|me)(?:[ \t]+on)?|"
    r"contact(?:[ \t]+(?:us|me))?(?:[ \t]+via)?)"
    r"[^a-z0-9|\n]{0,24}\Z",
    flags=re.IGNORECASE,
)
CONTEXTUAL_ALIAS_AFTER_CUE_RE = re.compile(
    r"[^a-z0-9|\n]{0,12}(?:shop|store|vendor|seller|team|"
    r"is[ \t]+here(?:[ \t]+again)?|offers?|brings?|provides?|supplies?|sells?)\b",
    flags=re.IGNORECASE,
)


def contextual_alias_match_kind(
    compact: str,
    registry: set[str] | frozenset[str],
    deletion_registry: set[str] | frozenset[str],
) -> str | None:
    if compact in registry:
        return "exact"
    if compact in deletion_registry:
        return "one_character_omission"
    if compact.endswith("s") and compact[:-1] in registry:
        return "known_alias_plural"
    for suffix in IDENTITY_HANDLE_SUFFIXES:
        base = compact[: -len(suffix)].rstrip("_-") if compact.endswith(suffix) else ""
        if base in registry:
            return "known_alias_plus_identity_suffix"
    return None


def anchored_alias_registry_token(
    value: object,
    registry: set[str] | frozenset[str],
) -> str | None:
    """Return the exact registry entry anchoring a compact phrase match."""
    compact = compact_identifier(value)
    if compact in registry:
        return compact
    if compact.endswith("s") and compact[:-1] in registry:
        return compact[:-1]
    for suffix in IDENTITY_HANDLE_SUFFIXES:
        base = compact[: -len(suffix)] if compact.endswith(suffix) else ""
        if base in registry:
            return base
    return None


def full_known_alias_residual_census(
    corpus_rows: Iterable[dict],
    registry: set[str] | frozenset[str],
) -> dict:
    """Census every exact known-alias surface left in serialized model text.

    This intentionally does *not* reuse ``contextual_alias_spans`` or
    ``unconditional_alias_spans``.  It is a fixed-snapshot audit of the full
    label-free alias universe, not another redaction rule.  Each seller row,
    newline-delimited field, and ``||``-delimited value is scanned separately,
    so an alias can never be assembled across a serialization boundary.

    Matching is longest-first at each word boundary.  Exact registry entries
    take precedence over a plural or identity-suffix interpretation, matching
    the production rule's precedence without sharing its implementation.
    Raw aliases are not written to the public manifest; keyed counts are
    SHA-256 hashed after the manual fixed-snapshot review has been completed.
    """
    token_registry = frozenset(str(token).casefold() for token in registry)
    if not token_registry or any(
        not re.fullmatch(r"[a-z0-9]{1,96}", token) for token in token_registry
    ):
        raise ValueError("Step7-v3 full known-alias census registry is invalid")

    maximum_form_length = max(
        len(token) + max(1, *(len(suffix) for suffix in IDENTITY_HANDLE_SUFFIXES))
        for token in token_registry
    )
    anchor_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    seller_sets: dict[str, set[str]] = defaultdict(set)
    scanned_rows = 0
    scanned_segments = 0

    def classify(compact: str) -> tuple[str, str] | None:
        if compact in token_registry:
            return compact, "exact"
        if compact.endswith("s") and compact[:-1] in token_registry:
            return compact[:-1], "known_alias_plural"
        for suffix in IDENTITY_HANDLE_SUFFIXES:
            if compact.endswith(suffix):
                base = compact[: -len(suffix)]
                if base in token_registry:
                    return base, f"known_alias_plus_{suffix}"
        return None

    match_kind_counts: Counter[str] = Counter()
    for row in corpus_rows:
        scanned_rows += 1
        seller_uid = str(row.get("seller_uid", ""))
        if not seller_uid:
            raise ValueError("Step7-v3 full alias census row lacks seller_uid")
        model_text = str(row.get("model_text", ""))
        for segment in re.split(r"\|\||\r?\n", model_text):
            scanned_segments += 1
            words = list(KNOWN_ALIAS_CENSUS_WORD_RE.finditer(segment))
            word_index = 0
            while word_index < len(words):
                compact = ""
                longest: tuple[int, str, str, str] | None = None
                for end_index in range(word_index, len(words)):
                    if end_index > word_index:
                        gap = segment[
                            words[end_index - 1].end() : words[end_index].start()
                        ]
                        if KNOWN_ALIAS_CENSUS_GAP_RE.fullmatch(gap) is None:
                            break
                    compact += words[end_index].group(0).casefold()
                    if len(compact) > maximum_form_length:
                        break
                    classified = classify(compact)
                    if classified is not None:
                        longest = (
                            end_index,
                            classified[0],
                            classified[1],
                            compact,
                        )
                if longest is None:
                    word_index += 1
                    continue
                end_index, anchor, match_kind, compact_surface = longest
                anchor_counts[anchor] += 1
                surface_counts[compact_surface] += 1
                seller_sets[anchor].add(seller_uid)
                match_kind_counts[match_kind] += 1
                word_index = end_index + 1

    hashed_anchor_counts = {
        sha256_text(anchor): int(count)
        for anchor, count in sorted(anchor_counts.items())
    }
    hashed_surface_counts = {
        sha256_text(surface): int(count)
        for surface, count in sorted(surface_counts.items())
    }
    hashed_anchor_seller_counts = {
        sha256_text(anchor): len(seller_sets[anchor]) for anchor in sorted(seller_sets)
    }
    return {
        "status": "pass_full_fixed_snapshot_known_alias_census_completed",
        "scan_scope": (
            "serialized_final_model_text_each_seller_row_newline_field_and_"
            "double_pipe_value_scanned_independently"
        ),
        "matching_contract": (
            "independent_longest_separator_invariant_exact_then_registry_"
            "anchored_plural_or_identity_suffix"
        ),
        "registry_token_count": len(token_registry),
        "scanned_seller_row_count": scanned_rows,
        "scanned_segment_count": scanned_segments,
        "matched_registry_token_count": len(anchor_counts),
        "matched_occurrence_count": int(sum(anchor_counts.values())),
        "matched_surface_count": len(surface_counts),
        "match_kind_counts": {
            key: int(value) for key, value in sorted(match_kind_counts.items())
        },
        "matched_anchor_sha256_counts": hashed_anchor_counts,
        "matched_surface_sha256_counts": hashed_surface_counts,
        "matched_anchor_sha256_seller_counts": hashed_anchor_seller_counts,
        "unknown_or_ambiguous_identifier_absence_proven": False,
    }


def audited_identity_embedded_residual_census(
    corpus_rows: Iterable[dict],
    registry: set[str] | frozenset[str],
) -> dict:
    """Find audited identities embedded inside longer alphanumeric words.

    The ordinary residue scan deliberately uses boundaries to avoid treating
    content words as identities.  This independent second pass inventories the
    opposite failure mode: a confirmed seller or marketplace identity hidden
    inside a longer token.  Exact, plural, and registered identity-suffix forms
    are excluded because the ordinary census already owns those cases.

    Only SHA-256 keyed counts leave this function.  The fixed-snapshot manual
    classification of the resulting candidates is attached by the preparation
    stage and pinned by policy and public-input hashes.
    """
    token_registry = frozenset(str(token).casefold() for token in registry)
    if not token_registry or any(
        not re.fullmatch(r"[a-z0-9]{1,96}", token) for token in token_registry
    ):
        raise ValueError("Step7-v3 embedded identity census registry is invalid")

    aliases_by_first_character: dict[str, list[str]] = defaultdict(list)
    for alias in token_registry:
        aliases_by_first_character[alias[0]].append(alias)
    for aliases in aliases_by_first_character.values():
        aliases.sort(key=lambda value: (-len(value), value))

    anchor_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    scanned_rows = 0
    for row in corpus_rows:
        scanned_rows += 1
        if not str(row.get("seller_uid", "")):
            raise ValueError("Step7-v3 embedded identity census row lacks seller_uid")
        model_text = str(row.get("model_text", ""))
        for word_match in re.finditer(r"(?i)[a-z0-9]+", model_text):
            surface = word_match.group(0).casefold()
            if anchored_alias_registry_token(surface, token_registry) is not None:
                continue
            matched_aliases: set[str] = set()
            for first_character in set(surface):
                for alias in aliases_by_first_character.get(first_character, []):
                    if len(alias) < len(surface) and alias in surface:
                        matched_aliases.add(alias)
            if not matched_aliases:
                continue
            for alias in sorted(matched_aliases):
                anchor_counts[alias] += 1
                # Count candidate identity occurrences grouped by the surface
                # that carried them. One word may contain two distinct audited
                # aliases, so this intentionally follows alias-surface pairs
                # rather than unique word occurrences.
                surface_counts[surface] += 1
                pair_counts[(alias, surface)] += 1

    return {
        "status": "pass_fixed_snapshot_embedded_identity_census_completed",
        "scan_scope": (
            "every_ascii_alphanumeric_word_in_serialized_final_model_text_"
            "after_anchored_identity_forms_are_excluded"
        ),
        "matching_contract": (
            "strict_substring_of_audited_seller_or_market_identity_registry_"
            "without_reusing_redaction_matchers"
        ),
        "registry_token_count": len(token_registry),
        "scanned_seller_row_count": scanned_rows,
        "matched_registry_token_count": len(anchor_counts),
        "matched_occurrence_count": int(sum(anchor_counts.values())),
        "matched_alias_surface_pair_count": len(pair_counts),
        "matched_surface_count": len(surface_counts),
        "matched_anchor_sha256_counts": {
            sha256_text(anchor): int(count)
            for anchor, count in sorted(anchor_counts.items())
        },
        "matched_surface_sha256_counts": {
            sha256_text(surface): int(count)
            for surface, count in sorted(surface_counts.items())
        },
        "matched_alias_surface_pair_sha256_counts": {
            sha256_text(anchor + "\0" + surface): int(count)
            for (anchor, surface), count in sorted(pair_counts.items())
        },
        "unknown_or_ambiguous_identifier_absence_proven": False,
    }


def contextual_alias_spans(
    text: str,
    registry: set[str] | frozenset[str],
    deletion_registry: set[str] | frozenset[str] | None = None,
) -> list[dict]:
    """Find known alias phrases only where the surrounding text signals identity."""
    fuzzy_registry = deletion_registry or frozenset()
    words = list(CONTEXTUAL_ALIAS_WORD_RE.finditer(text))
    spans: list[dict] = []
    for index, first in enumerate(words):
        matched = None
        maximum_word_count = min(3, len(words) - index)
        for word_count in range(maximum_word_count, 0, -1):
            selected = words[index : index + word_count]
            if any(
                not CONTEXTUAL_ALIAS_PHRASE_GAP_RE.fullmatch(
                    text[left.end() : right.start()]
                )
                for left, right in zip(selected, selected[1:])
            ):
                continue
            alias_start = selected[0].start()
            alias_end = selected[-1].end()
            compact = compact_identifier(text[alias_start:alias_end])
            kind = contextual_alias_match_kind(
                compact, registry, fuzzy_registry
            )
            if kind is None:
                continue
            window_start = max(0, alias_start - 96)
            before = text[window_start:alias_start].casefold()
            after = text[alias_end : min(len(text), alias_end + 96)].casefold()
            before_match = CONTEXTUAL_ALIAS_BEFORE_CUE_RE.search(before)
            after_match = CONTEXTUAL_ALIAS_AFTER_CUE_RE.match(after)
            if before_match is not None:
                reason = "before_cue"
                redact_start = window_start + before_match.start()
                redact_end = alias_end
            elif after_match is not None:
                reason = "after_cue"
                redact_start = alias_start
                redact_end = alias_end + after_match.end()
            else:
                continue
            matched = {
                "alias_start": alias_start,
                "alias_end": alias_end,
                "redact_start": redact_start,
                "redact_end": redact_end,
                "compact_alias": compact,
                "match_kind": kind,
                "reason": reason,
            }
            break
        if matched is not None:
            spans.append(matched)
    return spans


def redact_contextual_aliases(
    text: str,
    registry: set[str] | frozenset[str],
    deletion_registry: set[str] | frozenset[str] | None = None,
) -> tuple[str, int]:
    """Remove aliases and their identity cues in one non-cascading pass."""
    matches = contextual_alias_spans(text, registry, deletion_registry)
    spans = [(item["redact_start"], item["redact_end"]) for item in matches]
    match_count = len(matches)
    if not spans:
        return text, 0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    chunks: list[str] = []
    cursor = 0
    for start, end in merged:
        chunks.extend((text[cursor:start], " "))
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks), match_count


def unconditional_alias_spans(
    text: str,
    registry: set[str] | frozenset[str],
) -> list[tuple[int, int]]:
    """Find exact known aliases across harmless display separators.

    This is used for the current seller's own aliases and for the separately
    audited fixed-snapshot identity-phrase registry.  It closes the separator
    gap between an alias such as ``PrestigeVendor`` and a copied display form
    such as ``Prestige Vendor``.  The global registry is deliberately manual,
    source-hash pinned, and disjoint from protected content collisions.
    """
    token_registry = frozenset(str(token).casefold() for token in registry)
    if not token_registry:
        return []
    if any(
        re.fullmatch(r"[a-z0-9]{1,96}", token) is None
        for token in token_registry
    ):
        raise ValueError("Step7-v3 unconditional alias registry is invalid")
    maximum_form_length = max(
        len(token) + max(1, *(len(suffix) for suffix in IDENTITY_HANDLE_SUFFIXES))
        for token in token_registry
    )
    words = list(UNCONDITIONAL_ALIAS_WORD_RE.finditer(text))
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(words):
        matched: tuple[int, int, int] | None = None
        compact = ""
        for end_index in range(index, len(words)):
            if end_index > index:
                broad_gap = text[words[end_index - 1].end() : words[end_index].start()]
                if CONTEXTUAL_ALIAS_PHRASE_GAP_RE.fullmatch(broad_gap) is None:
                    break
            compact += compact_identifier(words[end_index].group(0))
            if len(compact) > maximum_form_length:
                break
            anchor = anchored_alias_registry_token(compact, token_registry)
            if anchor is None:
                continue
            gap_pattern = (
                CONTEXTUAL_ALIAS_PHRASE_GAP_RE
                if anchor in AUDITED_GLOBAL_IDENTITY_DOT_SEPARATOR_TOKENS
                else UNCONDITIONAL_ALIAS_PHRASE_GAP_RE
            )
            if any(
                not gap_pattern.fullmatch(text[left.end() : right.start()])
                for left, right in zip(
                    words[index:end_index], words[index + 1 : end_index + 1]
                )
            ):
                continue
            matched = (
                words[index].start(),
                words[end_index].end(),
                end_index - index + 1,
            )
        if matched is None:
            index += 1
            continue
        spans.append((matched[0], matched[1]))
        index += matched[2]
    return spans


def redact_unconditional_aliases(
    text: str,
    registry: set[str] | frozenset[str],
) -> tuple[str, int, dict[str, int]]:
    spans = unconditional_alias_spans(text, registry)
    if not spans:
        return text, 0, {}
    surface_counts: Counter[str] = Counter(
        compact_identifier(text[start:end]) for start, end in spans
    )
    chunks: list[str] = []
    cursor = 0
    for start, end in spans:
        chunks.extend((text[cursor:start], " "))
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks), len(spans), dict(sorted(surface_counts.items()))


def seller_identity_literals(profile: dict) -> list[str]:
    """Return seller aliases; noisy Step3 signal values are never local blacklists."""
    literals: list[str] = []
    for alias_field in ("source_seller_raw", "alias_normalized"):
        for alias_variant in seller_alias_variants(profile.get(alias_field, "")):
            alias_literal = safe_signal_literal("seller_alias", alias_variant)
            if alias_literal:
                literals.append(alias_literal)
                compact = compact_identifier(alias_literal)
                if (
                    compact != canonical_identifier_token(alias_literal)
                    and len(compact) >= 5
                    and compact not in CONTEXTUAL_ALIAS_CONTENT_WORD_DENYLIST
                    and compact not in GLOBAL_IDENTITY_CONTENT_COLLISION_DENYLIST
                    and compact
                    not in SEPARATOR_INVARIANT_IDENTITY_CONTENT_COLLISION_DENYLIST
                ):
                    literals.append(compact)
    return sorted(set(literals), key=lambda value: (-len(value), value.casefold()))


def seller_identity_phrase_tokens(profile: dict) -> set[str]:
    """Compact exact aliases for separator-invariant seller-local removal."""
    tokens: set[str] = set()
    for alias_field in ("source_seller_raw", "alias_normalized"):
        for alias_variant in seller_alias_variants(profile.get(alias_field, "")):
            compact = compact_identifier(alias_variant)
            if (
                4 <= len(compact) <= 96
                and compact
                not in SEPARATOR_INVARIANT_IDENTITY_CONTENT_COLLISION_DENYLIST
            ):
                tokens.add(compact)
    return tokens


def signal_literals_by_seller(path: Path) -> tuple[dict[str, list[str]], dict]:
    literals: dict[str, set[str]] = defaultdict(set)
    type_counts: Counter[str] = Counter()
    rows = load_csv(path)
    for row in rows:
        seller_uid = str(row.get("seller_uid", "")).strip()
        contact_type = str(row.get("contact_type", "")).strip().casefold()
        if not seller_uid or not contact_type:
            continue
        type_counts[contact_type] += 1
        for field in ("raw_value", "normalized_value"):
            literal = safe_signal_literal(contact_type, row.get(field, ""))
            if literal:
                literals[seller_uid].add(literal)
    return (
        {
            seller_uid: sorted(values, key=lambda value: (-len(value), value.casefold()))
            for seller_uid, values in literals.items()
        },
        {
            "signal_row_count": len(rows),
            "signal_type_counts": dict(sorted(type_counts.items())),
            "seller_with_signal_literal_count": len(literals),
        },
    )


def normalize_redacted_text(text: str) -> str:
    output = re.sub(r"[ \t]+", " ", str(text or ""))
    return re.sub(r"\s*\n\s*", "\n", output).strip()


def identity_literal_pattern(literal: str) -> re.Pattern[str]:
    """Match a known literal exactly, never as a substring of a larger token."""
    value = str(literal)
    if not value:
        raise ValueError("Step7-v3 identity literal cannot be empty")
    prefix = r"(?<!\w)" if value[0].isalnum() or value[0] == "_" else ""
    suffix = r"(?!\w)" if value[-1].isalnum() or value[-1] == "_" else ""
    return re.compile(prefix + re.escape(value) + suffix, flags=re.IGNORECASE)


def redact_identifiers(
    text: str,
    literals: Iterable[str],
    global_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_deletion_registry: set[str] | frozenset[str] | None = None,
    seller_local_phrase_tokens: set[str] | frozenset[str] | None = None,
    audited_global_phrase_tokens: set[str] | frozenset[str] | None = None,
) -> tuple[str, dict]:
    output = str(text or "")
    generic_matches = 0
    literal_matches = 0
    global_token_matches = 0
    global_token_counts: Counter[str] = Counter()
    contextual_alias_matches = 0
    local_phrase_matches = 0
    audited_global_phrase_matches = 0
    audited_global_phrase_counts: Counter[str] = Counter()
    literal_patterns = [
        (literal, identity_literal_pattern(literal)) for literal in literals
    ]
    token_registry = global_tokens or frozenset()
    contextual_registry = contextual_alias_tokens or frozenset()
    local_phrase_registry = seller_local_phrase_tokens or frozenset()
    audited_phrase_registry = audited_global_phrase_tokens or frozenset()
    for redaction_pass_count in range(1, MAX_REDACTION_PASSES + 1):
        before = output
        # Count the manually audited identities before broader rules erase the
        # surrounding contact phrase.  Repeating at the fixed point also finds
        # a phrase exposed only after another identifier has been removed.
        if audited_phrase_registry:
            output, count, surface_counts = redact_unconditional_aliases(
                output, audited_phrase_registry
            )
            audited_global_phrase_matches += int(count)
            audited_global_phrase_counts.update(surface_counts)
        for _rule_name, pattern in (*GENERIC_IDENTIFIER_RULES, *OBFUSCATED_CONTACT_RULES):
            output, count = pattern.subn(" ", output)
            generic_matches += int(count)
        for _literal, pattern in literal_patterns:
            output, count = pattern.subn(" ", output)
            literal_matches += int(count)
        if local_phrase_registry:
            output, count, _surface_counts = redact_unconditional_aliases(
                output, local_phrase_registry
            )
            local_phrase_matches += int(count)
        if token_registry:
            def replace_global_token(match: re.Match[str]) -> str:
                nonlocal global_token_matches
                if matches_global_identity_token(match.group(0), token_registry):
                    global_token_matches += 1
                    global_token_counts[canonical_identifier_token(match.group(0))] += 1
                    return " "
                return match.group(0)

            output = IDENTIFIER_TOKEN_RE.sub(replace_global_token, output)
        if contextual_registry:
            output, count = redact_contextual_aliases(
                output,
                contextual_registry,
                contextual_alias_deletion_registry,
            )
            contextual_alias_matches += int(count)
        output = normalize_redacted_text(output)
        if output == before:
            break
    else:
        raise ValueError(
            f"Step7-v3 redaction did not reach a fixed point in {MAX_REDACTION_PASSES} passes"
        )
    return output, {
        "generic_identifier_match_count": generic_matches,
        "seller_local_alias_match_count": literal_matches,
        "seller_local_alias_phrase_match_count": local_phrase_matches,
        "audited_global_identity_phrase_match_count": audited_global_phrase_matches,
        "audited_global_identity_phrase_counts": dict(
            sorted(audited_global_phrase_counts.items())
        ),
        "global_identifier_token_match_count": global_token_matches,
        "global_identifier_token_counts": dict(sorted(global_token_counts.items())),
        "contextual_alias_match_count": contextual_alias_matches,
        "redaction_pass_count": redaction_pass_count,
    }


def assert_no_identifier_residue(
    text: str,
    literals: Iterable[str],
    seller_uid: str,
    global_tokens: set[str] | frozenset[str] | None = None,
    seller_local_phrase_tokens: set[str] | frozenset[str] | None = None,
    audited_global_phrase_tokens: set[str] | frozenset[str] | None = None,
) -> None:
    for literal in literals:
        if identity_literal_pattern(literal).search(text):
            raise ValueError(
                f"Step7-v3 left a seller-local alias for seller hash="
                f"{hashlib.sha256(seller_uid.encode('utf-8')).hexdigest()[:16]}"
            )
    for rule_name, pattern in (*GENERIC_IDENTIFIER_RULES, *OBFUSCATED_CONTACT_RULES):
        if pattern.search(text):
            raise ValueError(
                f"Step7-v3 left a high-precision identifier pattern: rule={rule_name} "
                f"seller_hash={hashlib.sha256(seller_uid.encode('utf-8')).hexdigest()[:16]}"
            )
    token_registry = global_tokens or frozenset()
    residues = sorted(
        {
            canonical_identifier_token(match.group(0))
            for match in IDENTIFIER_TOKEN_RE.finditer(text)
            if matches_global_identity_token(match.group(0), token_registry)
        }
    )
    if residues:
        raise ValueError(
            "Step7-v3 left a globally known identity token for seller hash="
            f"{hashlib.sha256(seller_uid.encode('utf-8')).hexdigest()[:16]}"
        )
    if unconditional_alias_spans(
        text, seller_local_phrase_tokens or frozenset()
    ):
        raise ValueError(
            "Step7-v3 left a separator-variant seller-local alias for seller hash="
            f"{hashlib.sha256(seller_uid.encode('utf-8')).hexdigest()[:16]}"
        )
    if unconditional_alias_spans(
        text, audited_global_phrase_tokens or frozenset()
    ):
        raise ValueError(
            "Step7-v3 left an audited fixed-snapshot identity phrase for seller hash="
            f"{hashlib.sha256(seller_uid.encode('utf-8')).hexdigest()[:16]}"
        )


def scan_final_corpus_identity_residues(
    corpus_rows: Iterable[dict],
    seller_literals_by_uid: dict[str, list[str]],
    global_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_deletion_registry: set[str] | frozenset[str] | None = None,
    seller_phrase_tokens_by_uid: dict[str, set[str]] | None = None,
    audited_global_phrase_tokens: set[str] | frozenset[str] | None = None,
) -> dict:
    """Independently rescan final, concatenated model text before publication.

    The scan runs on the exact serialized-output representation, not on the
    redaction counters.  It deliberately evaluates each seller row separately
    so separators at the end of one field or row cannot form a fake handle
    together with the next field or seller.
    """
    pattern_counts: Counter[str] = Counter()
    local_literal_residue_count = 0
    global_token_residue_count = 0
    contextual_alias_residue_count = 0
    local_phrase_residue_count = 0
    audited_global_phrase_residue_count = 0
    scanned_rows = 0
    token_registry = global_tokens or frozenset()
    for row in corpus_rows:
        scanned_rows += 1
        seller_uid = str(row["seller_uid"])
        text = str(row["model_text"])
        for rule_name, pattern in FINAL_CORPUS_AUDIT_RULES:
            pattern_counts[rule_name] += sum(1 for _ in pattern.finditer(text))
        for literal in seller_literals_by_uid.get(seller_uid, []):
            local_literal_residue_count += sum(
                1 for _ in identity_literal_pattern(literal).finditer(text)
            )
        local_phrase_residue_count += len(
            unconditional_alias_spans(
                text, (seller_phrase_tokens_by_uid or {}).get(seller_uid, set())
            )
        )
        audited_global_phrase_residue_count += len(
            unconditional_alias_spans(
                text, audited_global_phrase_tokens or frozenset()
            )
        )
        if token_registry:
            global_token_residue_count += sum(
                1
                for match in IDENTIFIER_TOKEN_RE.finditer(text)
                if matches_global_identity_token(match.group(0), token_registry)
            )
        contextual_registry = contextual_alias_tokens or frozenset()
        if contextual_registry:
            contextual_alias_residue_count += len(
                contextual_alias_spans(
                    text,
                    contextual_registry,
                    contextual_alias_deletion_registry,
                )
            )
    pattern_residue_count = int(sum(pattern_counts.values()))
    total = (
        pattern_residue_count
        + int(local_literal_residue_count)
        + int(global_token_residue_count)
        + int(contextual_alias_residue_count)
        + int(local_phrase_residue_count)
        + int(audited_global_phrase_residue_count)
    )
    result = {
        "status": "pass" if total == 0 else "fail",
        "claim_scope": IDENTITY_RESIDUE_CLAIM_SCOPE,
        "unknown_identifier_absence_proven": False,
        "scan_scope": "serialized_final_model_text_each_seller_row_independently",
        "seller_row_count": scanned_rows,
        "pattern_residue_count": pattern_residue_count,
        "pattern_residue_count_by_rule": {
            key: int(value) for key, value in sorted(pattern_counts.items()) if value
        },
        "seller_local_identity_literal_residue_count": int(
            local_literal_residue_count
        ),
        "seller_local_separator_variant_residue_count": int(
            local_phrase_residue_count
        ),
        "audited_global_identity_phrase_residue_count": int(
            audited_global_phrase_residue_count
        ),
        "known_global_high_confidence_handle_residue_count": int(
            global_token_residue_count
        ),
        "context_gated_known_alias_residue_count": int(
            contextual_alias_residue_count
        ),
        "total_residue_count": int(total),
    }
    if total:
        raise ValueError(
            "Step7-v3 final serialized corpus identity scan failed: "
            f"pattern={pattern_residue_count} "
            f"local={local_literal_residue_count} global={global_token_residue_count} "
            f"local_phrase={local_phrase_residue_count} "
            f"audited_global_phrase={audited_global_phrase_residue_count} "
            f"contextual_alias={contextual_alias_residue_count}"
        )
    return result


def normalize_signature(value: str) -> str:
    folded = str(value or "").casefold()
    return " ".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", folded))


def split_concat(value: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"\s*\|\|\s*", str(value or "")) if segment.strip()]


def nested_float(row: dict, path: tuple[str, ...]) -> float:
    value: object = row
    for key in path:
        if not isinstance(value, dict):
            return math.nan
        value = value.get(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def build_clean_seller_record(
    profile: dict,
    clean_cfg: dict,
    global_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_deletion_registry: set[str] | frozenset[str] | None = None,
    audited_global_phrase_tokens: set[str] | frozenset[str] | None = None,
) -> tuple[dict, dict]:
    seller_uid = str(profile["seller_uid"])
    seller_literals = seller_identity_literals(profile)
    seller_phrase_tokens = seller_identity_phrase_tokens(profile)

    clean_fields: dict[str, list[str]] = {}
    diagnostics: Counter[str] = Counter()
    global_token_counts: Counter[str] = Counter()
    audited_global_phrase_counts: Counter[str] = Counter()
    for field in clean_cfg["fields_in_order"]:
        clean_segments = []
        for segment in split_concat(profile.get(field, "")):
            clean_segment, redaction_counts = redact_identifiers(
                segment,
                seller_literals,
                global_tokens=global_tokens,
                contextual_alias_tokens=contextual_alias_tokens,
                contextual_alias_deletion_registry=(
                    contextual_alias_deletion_registry
                ),
                seller_local_phrase_tokens=seller_phrase_tokens,
                audited_global_phrase_tokens=audited_global_phrase_tokens,
            )
            global_token_counts.update(
                redaction_counts.pop("global_identifier_token_counts")
            )
            audited_global_phrase_counts.update(
                redaction_counts.pop("audited_global_identity_phrase_counts")
            )
            diagnostics.update(redaction_counts)
            if clean_segment:
                assert_no_identifier_residue(
                    clean_segment,
                    seller_literals,
                    seller_uid,
                    global_tokens=global_tokens,
                    seller_local_phrase_tokens=seller_phrase_tokens,
                    audited_global_phrase_tokens=audited_global_phrase_tokens,
                )
                clean_segments.append(clean_segment)
        clean_fields[field] = clean_segments

    model_sections = [
        " || ".join(clean_fields[field])
        for field in clean_cfg["fields_in_order"]
        if clean_fields[field]
    ]
    model_text = "\n".join(model_sections).strip()
    empty_after_redaction = not model_text
    if empty_after_redaction:
        model_text = clean_cfg["empty_text_fallback"]

    category_values = clean_fields["category_concat_top"]
    title_values = (
        clean_fields["signature_title_concat"] + clean_fields["title_concat_top"]
    )
    description_values = (
        clean_fields["signature_description_concat"]
        + clean_fields["description_concat_top"]
    )

    def normalized_unique(values: list[str]) -> list[str]:
        return sorted({normalized for value in values if (normalized := normalize_signature(value))})

    numeric = {
        name: nested_float(profile, path)
        for name, path in NUMERIC_PROFILE_FIELDS.items()
    }
    record = {
        "seller_uid": seller_uid,
        "model_text": model_text,
        "clean_categories": normalized_unique(category_values),
        "clean_titles": normalized_unique(title_values),
        "clean_descriptions": normalized_unique(description_values),
        "source_dataset": str(profile.get("source_dataset", "") or ""),
        "source_market_raw": str(profile.get("source_market_raw", "") or ""),
        "numeric_profile": numeric,
    }
    return record, {
        "generic_identifier_match_count": int(diagnostics["generic_identifier_match_count"]),
        "seller_local_alias_match_count": int(
            diagnostics["seller_local_alias_match_count"]
        ),
        "seller_local_alias_phrase_match_count": int(
            diagnostics["seller_local_alias_phrase_match_count"]
        ),
        "audited_global_identity_phrase_match_count": int(
            diagnostics["audited_global_identity_phrase_match_count"]
        ),
        "audited_global_identity_phrase_counts": dict(
            sorted(audited_global_phrase_counts.items())
        ),
        "global_identifier_token_match_count": int(
            diagnostics["global_identifier_token_match_count"]
        ),
        "global_identifier_token_counts": dict(sorted(global_token_counts.items())),
        "contextual_alias_match_count": int(
            diagnostics["contextual_alias_match_count"]
        ),
        "empty_after_redaction": bool(empty_after_redaction),
    }


def train_reference(seller_records: dict[str, dict], train_seller_uids: set[str]) -> dict:
    if not train_seller_uids:
        raise ValueError("Step7-v3 train reference has no sellers")
    missing = sorted(train_seller_uids - set(seller_records))
    if missing:
        raise ValueError(f"Step7-v3 train reference seller missing: {missing[0]}")

    title_df: Counter[str] = Counter()
    description_df: Counter[str] = Counter()
    for seller_uid in sorted(train_seller_uids):
        record = seller_records[seller_uid]
        title_df.update(set(record["clean_titles"]))
        description_df.update(set(record["clean_descriptions"]))

    numeric_references = {}
    for name in NUMERIC_PROFILE_FIELDS:
        values = sorted(
            float(seller_records[seller_uid]["numeric_profile"][name])
            for seller_uid in train_seller_uids
            if math.isfinite(float(seller_records[seller_uid]["numeric_profile"][name]))
        )
        if not values:
            raise ValueError(f"Step7-v3 train reference has no finite values for {name}")
        numeric_references[name] = values
    return {
        "train_seller_count": len(train_seller_uids),
        "train_seller_uid_sha256": canonical_hash(sorted(train_seller_uids)),
        "title_df": dict(sorted(title_df.items())),
        "description_df": dict(sorted(description_df.items())),
        "numeric_references": numeric_references,
    }


def empirical_percentile(reference: list[float], value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.5
    array = np.asarray(reference, dtype=float)
    lower = int(np.searchsorted(array, value, side="left"))
    upper = int(np.searchsorted(array, value, side="right"))
    return float((lower + upper) / (2.0 * len(array)))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def shared_idf(
    shared: set[str],
    document_frequency: dict[str, int],
    train_seller_count: int,
) -> tuple[float, float]:
    if not shared:
        return 0.0, 0.0
    values = [
        math.log((1.0 + train_seller_count) / (1.0 + max(int(document_frequency.get(item, 0)), 2)))
        + 1.0
    for item in sorted(shared)
    ]
    return float(sum(values)), float(sum(values) / len(values))


def build_safe_pair_rows(
    pair_rows: list[dict],
    seller_records: dict[str, dict],
    reference: dict,
) -> list[dict]:
    output = []
    train_seller_count = int(reference["train_seller_count"])
    for pair in pair_rows:
        left = seller_records[pair["seller_uid_left"]]
        right = seller_records[pair["seller_uid_right"]]
        left_categories = set(left["clean_categories"])
        right_categories = set(right["clean_categories"])
        shared_categories = left_categories & right_categories
        shared_titles = set(left["clean_titles"]) & set(right["clean_titles"])
        shared_descriptions = set(left["clean_descriptions"]) & set(right["clean_descriptions"])
        title_idf_sum, title_idf_mean = shared_idf(
            shared_titles, reference["title_df"], train_seller_count
        )
        description_idf_sum, description_idf_mean = shared_idf(
            shared_descriptions, reference["description_df"], train_seller_count
        )
        row = {
            "pair_uid": pair["pair_uid"],
            "same_market_bool": int(
                bool(left["source_market_raw"])
                and left["source_market_raw"] == right["source_market_raw"]
            ),
            "same_source_dataset_bool": int(
                bool(left["source_dataset"])
                and left["source_dataset"] == right["source_dataset"]
            ),
            "clean_category_jaccard": jaccard(left_categories, right_categories),
            "clean_shared_title_bool": int(bool(shared_titles)),
            "clean_shared_description_bool": int(bool(shared_descriptions)),
            "clean_shared_title_count_capped": min(len(shared_titles), 5),
            "clean_shared_description_count_capped": min(len(shared_descriptions), 5),
            "clean_shared_category_count_capped": min(len(shared_categories), 5),
            "clean_shared_title_idf_sum": title_idf_sum,
            "clean_shared_description_idf_sum": description_idf_sum,
            "clean_shared_title_idf_mean": title_idf_mean,
            "clean_shared_description_idf_mean": description_idf_mean,
        }
        for name in NUMERIC_PROFILE_FIELDS:
            left_percentile = empirical_percentile(
                reference["numeric_references"][name], left["numeric_profile"][name]
            )
            right_percentile = empirical_percentile(
                reference["numeric_references"][name], right["numeric_profile"][name]
            )
            row[f"{name}_train_percentile_gap_abs"] = abs(left_percentile - right_percentile)
        if list(row)[1:] != SAFE_FEATURE_NAMES:
            raise AssertionError("Step7-v3 safe feature output order drift")
        output.append(row)
    return output


def validate_public_pair_rows(policy: dict, rows: list[dict]) -> None:
    expected_schema = [
        "pair_uid",
        "split_name",
        "component_id",
        "seller_uid_left",
        "seller_uid_right",
    ]
    if not rows or list(rows[0]) != expected_schema:
        raise ValueError("Step7-v3 public pair manifest schema drift")
    expected_counts = policy["supervision_boundary"]["expected_counts"]
    observed_counts = Counter(row["split_name"] for row in rows)
    expected_by_split = {
        split: int(expected_counts[split]["total"])
        for split in ("train", "valid", "test")
    }
    if observed_counts != Counter(expected_by_split):
        raise ValueError(
            f"Step7-v3 public pair split-count drift: {dict(observed_counts)}"
        )
    pair_uids = [row["pair_uid"] for row in rows]
    if len(pair_uids) != len(set(pair_uids)):
        raise ValueError("Step7-v3 public pair manifest has duplicate pair_uid values")
    unordered_pairs = []
    for row in rows:
        left = str(row["seller_uid_left"] or "").strip()
        right = str(row["seller_uid_right"] or "").strip()
        if not left or not right or left == right:
            raise ValueError("Step7-v3 public pair has an empty or self endpoint")
        if not str(row["component_id"] or "").strip():
            raise ValueError("Step7-v3 public pair has an empty component")
        unordered_pairs.append(tuple(sorted((left, right))))
    if len(unordered_pairs) != len(set(unordered_pairs)):
        raise ValueError("Step7-v3 public pair manifest has reversed/duplicate pairs")
    component_splits: dict[str, set[str]] = defaultdict(set)
    seller_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        component_splits[row["component_id"]].add(row["split_name"])
        seller_splits[row["seller_uid_left"]].add(row["split_name"])
        seller_splits[row["seller_uid_right"]].add(row["split_name"])
    if any(len(splits) != 1 for splits in component_splits.values()):
        raise ValueError("Step7-v3 public pair component crosses splits")
    if any(len(splits) != 1 for splits in seller_splits.values()):
        raise ValueError("Step7-v3 public pair seller crosses splits")
    observed_components = {
        split: sum(splits == {split} for splits in component_splits.values())
        for split in ("train", "valid", "test")
    }
    observed_sellers = {
        split: sum(splits == {split} for splits in seller_splits.values())
        for split in ("train", "valid", "test")
    }
    boundary = policy["supervision_boundary"]
    if observed_components != boundary["expected_component_count_by_split"]:
        raise ValueError("Step7-v3 public pair component-count contract drift")
    if observed_sellers != boundary["expected_seller_count_by_split"]:
        raise ValueError("Step7-v3 public pair seller-count contract drift")


def validate_clean_corpus_rows(rows: list[dict]) -> None:
    expected_schema = [
        "seller_uid",
        "split_name",
        "model_text",
        "model_text_sha256",
    ]
    if not rows or list(rows[0]) != expected_schema:
        raise ValueError("Step7-v3 clean corpus schema drift")
    seller_uids = [row["seller_uid"] for row in rows]
    if seller_uids != sorted(seller_uids) or len(seller_uids) != len(set(seller_uids)):
        raise ValueError("Step7-v3 clean corpus seller index must be sorted and unique")
    for row in rows:
        text = str(row["model_text"])
        if not text.strip():
            raise ValueError("Step7-v3 clean corpus contains empty model text")
        if row["split_name"] not in {"train", "valid", "test"}:
            raise ValueError("Step7-v3 clean corpus contains an invalid split")
        if row["model_text_sha256"] != sha256_text(text):
            raise ValueError("Step7-v3 clean corpus per-seller text hash drift")


def validate_field_corpus_rows(policy: dict, rows: list[dict]) -> None:
    expected = policy["clean_text_contract"]["expected_field_corpus"]
    fields = policy["clean_text_contract"]["fields_in_order"]
    if len(rows) != int(expected["seller_count"]):
        raise ValueError("Step7-v3.1 field corpus seller-count drift")
    seller_uids = [str(row.get("seller_uid", "")) for row in rows]
    if seller_uids != sorted(seller_uids) or len(seller_uids) != len(set(seller_uids)):
        raise ValueError("Step7-v3.1 field corpus seller order/uniqueness drift")
    nonempty_counts = {field: 0 for field in fields}
    for row in rows:
        if list(row) != [
            "seller_uid",
            "split_name",
            "field_texts",
            "field_text_sha256",
            "model_text",
            "model_text_sha256",
        ]:
            raise ValueError("Step7-v3.1 field corpus schema drift")
        if row["split_name"] not in {"train", "valid", "test"}:
            raise ValueError("Step7-v3.1 field corpus split drift")
        if list(row["field_texts"]) != fields or list(row["field_text_sha256"]) != fields:
            raise ValueError("Step7-v3.1 field corpus field order drift")
        for field in fields:
            value = row["field_texts"][field]
            if not isinstance(value, str) or row["field_text_sha256"][field] != sha256_text(value):
                raise ValueError("Step7-v3.1 field text hash drift")
            nonempty_counts[field] += int(bool(value))
        reconstructed = "\n".join(
            row["field_texts"][field]
            for field in fields
            if row["field_texts"][field]
        ).strip()
        if not reconstructed:
            reconstructed = policy["clean_text_contract"]["empty_text_fallback"]
        if reconstructed != row["model_text"]:
            raise ValueError("Step7-v3.1 field corpus reconstruction drift")
        if row["model_text_sha256"] != sha256_text(row["model_text"]):
            raise ValueError("Step7-v3.1 field corpus model-text hash drift")
    if nonempty_counts != expected["nonempty_field_seller_counts"]:
        raise ValueError("Step7-v3.1 field corpus nonempty-field count drift")


def validate_safe_pair_feature_rows(rows: list[dict]) -> None:
    expected_schema = ["pair_uid", *SAFE_FEATURE_NAMES]
    if not rows or list(rows[0]) != expected_schema:
        raise ValueError("Step7-v3 safe pair feature schema drift")
    boolean_names = {"same_market_bool", "same_source_dataset_bool", "clean_shared_title_bool", "clean_shared_description_bool"}
    capped_names = {
        "clean_shared_title_count_capped",
        "clean_shared_description_count_capped",
        "clean_shared_category_count_capped",
    }
    unit_interval_names = {
        "clean_category_jaccard",
        *(name for name in SAFE_FEATURE_NAMES if name.endswith("_train_percentile_gap_abs")),
    }
    for row in rows:
        for name in SAFE_FEATURE_NAMES:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Step7-v3 safe feature is not numeric: {name}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Step7-v3 safe feature is non-finite: {name}")
            if name in boolean_names and value not in {0.0, 1.0}:
                raise ValueError(f"Step7-v3 safe boolean feature is out of range: {name}")
            if name in capped_names and (
                value < 0.0 or value > 5.0 or value != float(int(value))
            ):
                raise ValueError(f"Step7-v3 capped feature is out of range: {name}")
            if name in unit_interval_names and not 0.0 <= value <= 1.0:
                raise ValueError(f"Step7-v3 unit-interval feature is out of range: {name}")
            if name not in boolean_names | capped_names | unit_interval_names and value < 0.0:
                raise ValueError(f"Step7-v3 nonnegative feature is below zero: {name}")


def validate_weight_payload(path: Path) -> None:
    if path.name != "model.safetensors":
        if path.stat().st_size <= 1024 * 1024:
            raise ValueError(f"Step7-v3 model weight file is implausibly small: {path}")
        return
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Step7-v3 safetensors header is truncated: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 2 or header_length > 100 * 1024 * 1024:
            raise ValueError(f"Step7-v3 safetensors header length is invalid: {path}")
        header_bytes = handle.read(header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8").rstrip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Step7-v3 safetensors header is invalid JSON: {path}") from exc
    offsets = [
        value.get("data_offsets")
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    ]
    if not offsets or any(
        not isinstance(offset, list)
        or len(offset) != 2
        or not all(isinstance(item, int) for item in offset)
        or offset[0] < 0
        or offset[1] < offset[0]
        for offset in offsets
    ):
        raise ValueError(f"Step7-v3 safetensors data offsets are invalid: {path}")
    expected_size = 8 + header_length + max(offset[1] for offset in offsets)
    if expected_size != path.stat().st_size:
        raise ValueError(
            f"Step7-v3 safetensors payload is truncated or has trailing bytes: {path}; "
            f"expected={expected_size} observed={path.stat().st_size}"
        )


def validate_sentence_transformer_layout(model_key: str, cfg: dict) -> dict:
    model_dir = resolve(cfg["local_path"])
    modules_path = model_dir / "modules.json"
    pooling_path = model_dir / "1_Pooling" / "config.json"
    config_path = model_dir / "config.json"
    for path in (modules_path, pooling_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Step7-v3 local model file missing for {model_key}: {path}")
    weight_files = [
        path
        for name in ("model.safetensors", "pytorch_model.bin")
        for path in model_dir.rglob(name)
        if path.is_file() and path.stat().st_size > 1024 * 1024
    ]
    if not weight_files:
        raise FileNotFoundError(
            f"Step7-v3 local model weights are missing for {model_key}: {model_dir}. "
            "A tokenizer/config-only directory is not executable."
        )
    for weight_file in weight_files:
        validate_weight_payload(weight_file)
    modules = load_json(modules_path)
    pooling = load_json(pooling_path)
    observed_pooling = None
    if bool(pooling.get("pooling_mode_cls_token")) and not bool(
        pooling.get("pooling_mode_mean_tokens")
    ):
        observed_pooling = "cls"
    elif bool(pooling.get("pooling_mode_mean_tokens")) and not bool(
        pooling.get("pooling_mode_cls_token")
    ):
        observed_pooling = "mean"
    if observed_pooling != cfg["expected_pooling"]:
        raise ValueError(
            f"Step7-v3 pooling mismatch for {model_key}: "
            f"expected={cfg['expected_pooling']} observed={observed_pooling}"
        )
    has_dense = any(str(module.get("type", "")).endswith(".Dense") for module in modules)
    if has_dense != bool(cfg["expected_dense_module"]):
        raise ValueError(
            f"Step7-v3 dense-head mismatch for {model_key}: "
            f"expected={cfg['expected_dense_module']} observed={has_dense}"
        )
    return {
        "model_key": model_key,
        "local_path": str(model_dir.relative_to(ROOT)).replace("\\", "/"),
        "pooling": observed_pooling,
        "has_dense_module": has_dense,
        "modules_sha256": sha256_file(modules_path),
        "pooling_config_sha256": sha256_file(pooling_path),
        "model_config_sha256": sha256_file(config_path),
    }




def directory_inventory_fingerprint(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Step7-v3 model directory is missing: {path}")
    records = []
    candidates = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    for item in sorted(
        candidates, key=lambda candidate: str(candidate.relative_to(path)).replace("\\", "/")
    ):
        relative = str(item.relative_to(path)).replace("\\", "/")
        if ".cache" in item.parts or "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        records.append({"path": relative, "size_bytes": item.stat().st_size})
    if not records:
        raise ValueError(f"Step7-v3 model directory is empty: {path}")
    return {
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "inventory_sha256": canonical_hash(records),
    }


def model_content_fingerprint(path: Path) -> dict:
    """Hash every runtime model file, excluding downloader/cache metadata."""
    if not path.is_dir():
        raise FileNotFoundError(f"Step7-v3 model directory is missing: {path}")
    records = []
    candidates = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    for item in sorted(
        candidates, key=lambda candidate: str(candidate.relative_to(path)).replace("\\", "/")
    ):
        if ".cache" in item.parts or "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        records.append(
            {
                "path": str(item.relative_to(path)).replace("\\", "/"),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    if not records:
        raise ValueError(f"Step7-v3 model directory is empty: {path}")
    return {
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "content_sha256": canonical_hash(records),
        "files": records,
    }


def validate_model_content_pin(model_key: str, cfg: dict) -> dict:
    validate_expected_model_pin(model_key, cfg)
    observed = model_content_fingerprint(resolve(cfg["local_path"]))
    checks = {
        "content_sha256": str(cfg["expected_content_sha256"]).casefold(),
        "file_count": int(cfg["expected_file_count"]),
        "total_size_bytes": int(cfg["expected_total_size_bytes"]),
    }
    for key, expected in checks.items():
        if observed[key] != expected:
            raise ValueError(
                f"Step7-v3 model payload drift for {model_key}: "
                f"field={key} expected={expected} observed={observed[key]}"
            )
    return observed
