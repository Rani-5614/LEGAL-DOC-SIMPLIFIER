# models/legal_analyzer.py
"""
Document intelligence helpers:
  1. detect_document_type()  - weighted keyword/phrase scoring with confidence
  2. analyze_risks()         - clause-level risk detection with plain-language
                               explanations and suggestions based on the real text
  3. extract_deadlines()     - expiry / renewal / notice-period / due-date detection
"""
import re
from datetime import datetime, date, timedelta

# ------------------------------------------------------------------
# 1. DOCUMENT TYPE DETECTION
# ------------------------------------------------------------------
# (pattern, weight). Strong, specific phrases carry more weight than generic words.
DOC_TYPE_RULES = {
    'Employment Document': [
        (r'employment agreement', 8), (r'offer letter', 8), (r'appointment letter', 8),
        (r'employee', 2), (r'employer', 2), (r'probation', 4), (r'notice period', 3),
        (r'salary', 3), (r'ctc', 3), (r'designation', 3), (r'resignation', 4),
        (r'non-compete', 3), (r'gratuity', 4), (r'provident fund', 4), (r'leave policy', 3),
        (r'termination of employment', 5), (r'job description', 3),
    ],
    'Property/Land Document': [
        (r'sale deed', 9), (r'lease deed', 9), (r'rental agreement', 9), (r'rent agreement', 9),
        (r'leave and license', 9), (r'landlord', 4), (r'tenant', 4), (r'lessor', 4), (r'lessee', 4),
        (r'security deposit', 4), (r'property', 2), (r'plot', 2), (r'flat', 1),
        (r'registered', 1), (r'sub-registrar', 6), (r'encumbrance', 5), (r'title deed', 7),
        (r'possession', 2), (r'monthly rent', 5), (r'premises', 3), (r'real estate', 4),
    ],
    'Financial Document': [
        (r'loan agreement', 9), (r'promissory note', 9), (r'sanction letter', 8),
        (r'borrower', 4), (r'lender', 4), (r'interest rate', 4), (r'emi', 4), (r'principal', 2),
        (r'collateral', 4), (r'guarantor', 4), (r'mortgage', 4), (r'credit', 1), (r'repayment', 4),
        (r'hypothecation', 6), (r'bank', 1), (r'cheque', 3), (r'prepayment', 4), (r'default interest', 5),
    ],
    'Family Document': [
        (r'divorce', 8), (r'decree of divorce', 10), (r'custody', 6), (r'maintenance', 3),
        (r'alimony', 6), (r'matrimonial', 6), (r'marriage', 3), (r'will and testament', 9),
        (r'testator', 6), (r'last will', 8), (r'succession', 5), (r'adoption', 6),
        (r'husband', 2), (r'wife', 2), (r'minor child', 4), (r'mutual consent', 4),
    ],
    'Corporate Document': [
        (r'articles of association', 10), (r'memorandum of association', 10),
        (r'shareholders? agreement', 10), (r'board of directors', 5), (r'shareholder', 4),
        (r'share capital', 5), (r'equity', 2), (r'founder', 3), (r'incorporated', 3),
        (r'annual general meeting', 6), (r'company secretary', 5), (r'joint venture', 6),
        (r'vesting', 4), (r'term sheet', 7), (r'directors?', 2),
    ],
    'Government/Legal Notice': [
        (r'legal notice', 10), (r'notice under section', 8), (r'show cause', 8),
        (r'hereby called upon', 7), (r'summons', 7), (r'court order', 6), (r'judgment', 5),
        (r'decree', 4), (r'writ petition', 8), (r'petitioner', 4), (r'respondent', 4),
        (r'hon\'?ble court', 6), (r'tribunal', 4), (r'within 15 days', 3), (r'within 30 days', 3),
        (r'cheque dishonou?r', 8), (r'section 138', 8),
    ],
    'Intellectual Property': [
        (r'patent', 6), (r'trademark', 6), (r'copyright', 6), (r'licen[cs]e agreement', 5),
        (r'royalty', 5), (r'intellectual property', 7), (r'trade secret', 5), (r'licensor', 4),
        (r'licensee', 4), (r'assignment of rights', 5), (r'moral rights', 5), (r'infring', 4),
    ],
    'Criminal Document': [
        (r'first information report', 10), (r'\bfir\b', 7), (r'chargesheet', 9), (r'charge sheet', 9),
        (r'bail', 7), (r'accused', 6), (r'offen[cs]e', 3), (r'ipc', 4), (r'bns', 4), (r'crpc', 4),
        (r'police station', 5), (r'complainant', 3), (r'arrest', 5), (r'criminal', 4),
        (r'cognizable', 6), (r'anticipatory bail', 9),
    ],
    'Insurance Document': [
        (r'insurance policy', 9), (r'policyholder', 7), (r'insured', 4), (r'insurer', 5),
        (r'premium', 5), (r'sum assured', 7), (r'claim settlement', 6), (r'nominee', 3),
        (r'coverage', 3), (r'exclusions?', 3), (r'renewal premium', 6), (r'waiting period', 5),
    ],
    'Education Document': [
        (r'admission', 3), (r'student', 3), (r'tuition', 5), (r'university', 3), (r'college', 3),
        (r'scholarship', 5), (r'bond', 2), (r'academic year', 5), (r'course fee', 5),
    ],
    'Healthcare Document': [
        (r'patient', 4), (r'hospital', 3), (r'medical', 2), (r'consent for (surgery|treatment)', 9),
        (r'diagnosis', 4), (r'discharge summary', 8), (r'physician', 3), (r'treatment', 2),
    ],
    'Civil Document': [
        (r'civil suit', 9), (r'plaintiff', 5), (r'defendant', 5), (r'injunction', 5), (r'tort', 4),
        (r'negligence', 3), (r'damages', 2), (r'consumer complaint', 7), (r'decree holder', 5),
    ],
    # Broad catch-all - intentionally low weights so specific types win.
    'Contract Agreement': [
        (r'service agreement', 8), (r'non-disclosure agreement', 9), (r'\bnda\b', 6),
        (r'master services agreement', 9), (r'terms and conditions', 5), (r'this agreement', 3),
        (r'agreement', 1.5), (r'contract', 1.5), (r'indemnif', 2), (r'governing law', 2),
        (r'force majeure', 2), (r'confidential information', 3), (r'scope of work', 4),
        (r'deliverables', 3), (r'party of the (first|second) part', 3), (r'consultant', 2),
        (r'vendor', 2), (r'independent contractor', 4),
    ],
}


def detect_document_type(text):
    """
    Returns dict: {type, confidence (0-100), alternatives: [(type, score)], matched_terms: [..]}
    """
    low = (text or '').lower()
    if not low.strip():
        return {'type': 'General Legal Document', 'confidence': 0, 'alternatives': [], 'matched_terms': []}

    scores, matched = {}, {}
    for dtype, rules in DOC_TYPE_RULES.items():
        total, terms = 0.0, []
        for pattern, weight in rules:
            hits = len(re.findall(pattern, low))
            if hits:
                # diminishing returns so repeated generic words don't dominate
                total += weight * (1 + min(hits - 1, 4) * 0.35)
                terms.append(re.sub(r'\\b|\\|\(|\)|\?|\[.*?\]', '', pattern))
        scores[dtype] = total
        matched[dtype] = terms

    # Title bias: first 300 chars usually state the document type
    head = low[:300]
    for dtype, rules in DOC_TYPE_RULES.items():
        for pattern, weight in rules:
            if weight >= 7 and re.search(pattern, head):
                scores[dtype] += weight * 0.8

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    if best_score < 3:
        return {'type': 'General Legal Document', 'confidence': 20, 'alternatives': [],
                'matched_terms': []}

    second = ranked[1][1] if len(ranked) > 1 else 0
    margin = (best_score - second) / best_score if best_score else 0
    strength = min(best_score / 25.0, 1.0)
    confidence = int(round(min(98, 35 + 40 * strength + 25 * margin)))

    alternatives = [(t, round(s, 1)) for t, s in ranked[1:4] if s >= 3]
    return {'type': best, 'confidence': confidence, 'alternatives': alternatives,
            'matched_terms': matched[best][:8]}


# Detailed lawyer-search profile per document type
# 'osm_terms' are used to match OpenStreetMap names/tags; 'search_terms' for map search links.
SPECIALIZATION_PROFILE = {
    'Contract Agreement': {'primary': 'Contract & Business Law', 'related': ['Corporate Law', 'Commercial Law']},
    'Employment Document': {'primary': 'Employment & Labour Law', 'related': ['Labour Law', 'Industrial Disputes']},
    'Property/Land Document': {'primary': 'Property & Real Estate Law', 'related': ['Land Law', 'Registration & Conveyancing']},
    'Financial Document': {'primary': 'Banking & Finance Law', 'related': ['Debt Recovery', 'Tax Law']},
    'Family Document': {'primary': 'Family Law', 'related': ['Divorce Law', 'Succession & Wills']},
    'Corporate Document': {'primary': 'Corporate & Company Law', 'related': ['Securities Law', 'Mergers & Acquisitions']},
    'Government/Legal Notice': {'primary': 'Litigation & Administrative Law', 'related': ['Constitutional Law', 'Civil Litigation']},
    'Intellectual Property': {'primary': 'Intellectual Property Law', 'related': ['Patent Law', 'Trademark & Copyright']},
    'Criminal Document': {'primary': 'Criminal Law', 'related': ['Bail & Criminal Defence', 'Criminal Procedure']},
    'Civil Document': {'primary': 'Civil Litigation', 'related': ['Tort Law', 'Consumer Law']},
    'Insurance Document': {'primary': 'Insurance Law', 'related': ['Consumer Law', 'Contract Law']},
    'Education Document': {'primary': 'Education Law', 'related': ['Administrative Law', 'Consumer Law']},
    'Healthcare Document': {'primary': 'Medical & Healthcare Law', 'related': ['Consumer Law', 'Medical Negligence']},
    'General Legal Document': {'primary': 'General Legal Practice', 'related': ['Civil Law', 'Advocate']},
}


def get_specialization_profile(doc_type):
    return SPECIALIZATION_PROFILE.get(doc_type, SPECIALIZATION_PROFILE['General Legal Document'])


# ------------------------------------------------------------------
# 2. PERSONALISED RISK ANALYSIS
# ------------------------------------------------------------------
# Each rule: key, label, severity (1-3), regex, simple explanation, suggestion
RISK_RULES = [
    {
        'key': 'unlimited_liability', 'label': 'Unlimited liability', 'severity': 3,
        'pattern': r'unlimited liability|without (any )?limit(ation)? (of|on) liability|liable for all (losses|damages)',
        'explain': 'You could be forced to pay for ALL losses, with no upper cap on the amount.',
        'suggest': 'Ask for a cap on liability (for example, limited to the fees paid under this document).',
    },
    {
        'key': 'indemnity', 'label': 'Indemnity (you cover the other side\'s losses)', 'severity': 3,
        'pattern': r'indemnif|hold harmless|save harmless',
        'explain': 'You may have to pay back the other party for losses or legal costs, even for things you did not directly cause.',
        'suggest': 'Make the indemnity mutual, limit it to losses caused by your own breach or negligence, and add a maximum amount.',
    },
    {
        'key': 'penalty', 'label': 'Penalty / liquidated damages', 'severity': 3,
        'pattern': r'liquidated damages|penalt(y|ies)|forfeit(ure)?|forfeited',
        'explain': 'A fixed penalty or loss of money/property applies if you miss a condition, whether or not the other side actually lost anything.',
        'suggest': 'Check the amount is reasonable. Ask for a grace period and a clause that penalties apply only for actual, proven loss.',
    },
    {
        'key': 'unilateral_termination', 'label': 'One-sided termination', 'severity': 3,
        'pattern': r'terminate[^.]{0,80}(at any time|without (any )?(cause|reason|notice)|sole discretion)|without cause',
        'explain': 'The other party can end this arrangement whenever they like, without giving a reason or notice.',
        'suggest': 'Request a reasonable written notice period (e.g. 30 days) and make the right to terminate mutual.',
    },
    {
        'key': 'non_compete', 'label': 'Non-compete / restriction on work', 'severity': 3,
        'pattern': r'non-?compete|restraint of trade|shall not (engage|work|be employed)[^.]{0,80}(competitor|similar)',
        'explain': 'You may be stopped from working in your field or for competitors, sometimes even after the agreement ends.',
        'suggest': 'Limit the duration, area and scope. In India, post-termination non-competes are generally unenforceable under Section 27 of the Contract Act, but get a lawyer to confirm.',
    },
    {
        'key': 'auto_renewal', 'label': 'Automatic renewal', 'severity': 2,
        'pattern': r'auto(matic(ally)?)?[- ]?renew|renew(s|ed)? automatically|deemed to be renewed|evergreen',
        'explain': 'The agreement renews by itself, and you can get locked in again if you miss the cancellation window.',
        'suggest': 'Set a calendar reminder before the notice deadline, or ask that renewal needs your written consent.',
    },
    {
        'key': 'irrevocable', 'label': 'Irrevocable / exclusive commitment', 'severity': 2,
        'pattern': r'irrevocabl|exclusive(ly)? (right|licen[cs]e|basis)|sole and exclusive',
        'explain': 'You give up the ability to change your mind or to work with anyone else on this subject.',
        'suggest': 'Ask for a time limit and an exit route (for example, if the other side fails to perform).',
    },
    {
        'key': 'waiver', 'label': 'Waiver of rights', 'severity': 2,
        'pattern': r'waive[sd]?\b|waiver of|relinquish',
        'explain': 'You agree to give up certain legal rights, such as the right to claim compensation or go to court.',
        'suggest': 'Confirm exactly which rights are waived. Do not sign away rights you may need later.',
    },
    {
        'key': 'arbitration', 'label': 'Arbitration / far-away jurisdiction', 'severity': 2,
        'pattern': r'arbitrat|exclusive jurisdiction|courts? (of|at|in) [A-Z][a-z]+ (shall|will|alone)',
        'explain': 'Disputes may be decided privately by an arbitrator, or only in courts in a specific (possibly distant) city.',
        'suggest': 'Check the city and who picks the arbitrator. Ask for a neutral or convenient location and a mutually chosen arbitrator.',
    },
    {
        'key': 'interest_late_fee', 'label': 'High interest / late fees', 'severity': 2,
        'pattern': r'(late (payment )?fee|default interest|penal interest|interest (at|of)[^.]{0,30}(%|per cent|percent))',
        'explain': 'Extra charges or interest are added if you pay late, and these can grow quickly.',
        'suggest': 'Check the exact percentage and period. Negotiate a grace period and a lower rate.',
    },
    {
        'key': 'security_deposit', 'label': 'Security deposit / forfeiture of money', 'severity': 1,
        'pattern': r'security deposit|earnest money|advance (payment|amount)',
        'explain': 'Money is held by the other party and may not be returned in some situations.',
        'suggest': 'Get the refund conditions and the refund timeline in writing; keep receipts.',
    },
    {
        'key': 'confidentiality', 'label': 'Confidentiality obligations', 'severity': 1,
        'pattern': r'confidential(ity)?|non-?disclosure',
        'explain': 'You must keep certain information secret, and sharing it may lead to legal action.',
        'suggest': 'Check how long the secrecy lasts and what counts as "confidential".',
    },
    {
        'key': 'ip_assignment', 'label': 'Transfer of intellectual property', 'severity': 2,
        'pattern': r'assign(s|ed)? (all )?(rights|intellectual property)|work (made )?for hire|ownership of (all )?(work|deliverables)',
        'explain': 'What you create may belong to the other party, not to you.',
        'suggest': 'Keep ownership of your pre-existing work and limit the transfer to what is specifically paid for.',
    },
    {
        'key': 'unilateral_change', 'label': 'Right to change terms unilaterally', 'severity': 2,
        'pattern': r'(reserves? the right|may|can) (to )?(modify|amend|change|alter|revise)[^.]{0,60}(at any time|without (prior )?notice|sole discretion)',
        'explain': 'The other party can change the rules later without your agreement.',
        'suggest': 'Ask that any change needs your written consent, or at least advance notice with a right to exit.',
    },
    {
        'key': 'short_deadline', 'label': 'Very short time limit', 'severity': 2,
        'pattern': r'within\s+(\d{1,2}|one|two|three|five|seven)\s*\(?\d*\)?\s*(days?|hours?)',
        'explain': 'You have only a short window to act (for example, to reply, pay, or object), and missing it can have consequences.',
        'suggest': 'Note the exact date right away and set a reminder. Ask for more time if it is too short.',
    },
    {
        'key': 'personal_guarantee', 'label': 'Personal guarantee', 'severity': 3,
        'pattern': r'personal(ly)? guarant|guarantor|jointly and severally',
        'explain': 'Your personal assets (not just a company\'s) may be used to repay the debt or loss.',
        'suggest': 'Limit the guarantee to a fixed amount and period, and avoid "jointly and severally" if you can.',
    },
]

SEVERITY_LABEL = {1: 'Low', 2: 'Medium', 3: 'High'}

# Extra, type-specific advice (added only for the detected type)
TYPE_SPECIFIC_ADVICE = {
    'Employment Document': 'Check notice period, probation terms, non-compete, bond/recovery clauses and what happens to unpaid salary on exit.',
    'Property/Land Document': 'Verify title, registration, lock-in period, rent escalation and refund of the security deposit.',
    'Financial Document': 'Compare the interest rate, processing fees, prepayment charges and the consequences of default.',
    'Contract Agreement': 'Make sure scope of work, payment schedule, liability cap and exit rights are all clearly written.',
    'Family Document': 'Review maintenance, custody and property terms carefully; these are hard to change after signing.',
    'Government/Legal Notice': 'Respond within the stated deadline. Missing it can weaken your legal position.',
    'Criminal Document': 'Do not make statements without a lawyer. Time-sensitive: contact a criminal lawyer immediately.',
    'Insurance Document': 'Read exclusions, waiting periods and claim-intimation timelines closely.',
    'Intellectual Property': 'Confirm exactly what rights you keep, territory, duration and royalties.',
    'Corporate Document': 'Check vesting, exit/drag-along rights, dilution and reserved matters.',
}


def _split_sentences(text):
    parts = re.split(r'(?<=[.;:])\s+|\n+', text or '')
    return [p.strip() for p in parts if p and len(p.strip()) > 15]


def analyze_risks(text, doc_type=None, risk_level='Medium'):
    """
    Returns {
      'findings': [{label, severity, severity_label, explanation, suggestion, excerpt}],
      'summary': str, 'top_actions': [str]
    }
    Every finding is tied to an excerpt that actually appears in the user's document.
    """
    sentences = _split_sentences(text)
    findings, seen = [], set()

    for rule in RISK_RULES:
        rx = re.compile(rule['pattern'], re.IGNORECASE)
        for s in sentences:
            if rx.search(s):
                key = rule['key']
                if key in seen:
                    break
                seen.add(key)
                excerpt = s if len(s) <= 260 else s[:257].rsplit(' ', 1)[0] + '...'
                suggestion = rule['suggest']
                # Personalise suggestion using actual values in the sentence
                nums = re.findall(r'(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d+)?|\d+\s?%|\d+\s*(?:days?|months?|years?)', s, re.I)
                if nums:
                    suggestion += f' (Your document mentions: {", ".join(dict.fromkeys(n.strip() for n in nums[:3]))}.)'
                findings.append({
                    'key': key,
                    'label': rule['label'],
                    'severity': rule['severity'],
                    'severity_label': SEVERITY_LABEL[rule['severity']],
                    'explanation': rule['explain'],
                    'suggestion': suggestion,
                    'excerpt': excerpt,
                })
                break

    findings.sort(key=lambda f: f['severity'], reverse=True)

    high = sum(1 for f in findings if f['severity'] == 3)
    med = sum(1 for f in findings if f['severity'] == 2)
    if findings:
        summary = (f"This document is rated {risk_level} risk. We found {len(findings)} clause(s) worth a closer look"
                   f" ({high} high, {med} medium). Each one is explained below in plain language with a suggested action.")
    else:
        summary = (f"This document is rated {risk_level} risk. No commonly risky clauses were detected, "
                   "but still read the full document before signing.")

    top_actions = [f"{f['label']}: {f['suggestion'].split(' (Your')[0]}" for f in findings[:3]]
    if doc_type in TYPE_SPECIFIC_ADVICE:
        top_actions.append(f"For a {doc_type.lower()}: {TYPE_SPECIFIC_ADVICE[doc_type]}")

    return {'findings': findings, 'summary': summary, 'top_actions': top_actions}


# ------------------------------------------------------------------
# 3. DEADLINE / EXPIRY EXTRACTION
# ------------------------------------------------------------------
MONTHS = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
    'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}
WORD_NUM = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
            'nine': 9, 'ten': 10, 'twelve': 12, 'fifteen': 15, 'thirty': 30, 'sixty': 60, 'ninety': 90}

_MON = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
DATE_PATTERNS = [
    # 15th March 2026 / 15 March, 2026
    (re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(' + _MON + r')\.?,?\s+(\d{4})\b', re.I), 'dmy_text'),
    # March 15, 2026
    (re.compile(r'\b(' + _MON + r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b', re.I), 'mdy_text'),
    # 15/03/2026, 15-03-2026, 15.03.2026 (Indian day-first)
    (re.compile(r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b'), 'dmy_num'),
    # 2026-03-15
    (re.compile(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'), 'ymd_num'),
]

DEADLINE_KEYWORDS = [
    # (category, regex of trigger words, priority)
    ('Expiry', r'expir(?:e|y|es|ed|ing|ation)|valid (?:until|till|upto|up to|through)|ends? on|end date|lapse', 'high'),
    ('Renewal', r'renew(?:al|s|ed)?|extension|extend(?:ed)?', 'high'),
    ('Notice Period', r'notice|intimat|vacate|give notice|prior written', 'high'),
    ('Payment Due', r'due (?:on|by|date)|payable (?:on|by|before)|pay(?:ment)? (?:on or before|before|by)|instalment|installment|emi|rent (?:is )?(?:due|payable)|premium due', 'high'),
    ('Response Deadline', r'reply|respond|response|appear|file (?:a )?(?:reply|appeal|petition)|hearing|show cause', 'high'),
    ('Start / Effective Date', r'effective (?:from|date|on)|commenc|start(?:s|ing)? (?:on|from)|dated|w\.e\.f', 'low'),
    ('Lock-in / Term', r'lock-?in|term of|tenure|period of', 'medium'),
]


def _make_date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def _find_dates(text):
    """Return list of (date, start, end)."""
    out = []
    for rx, kind in DATE_PATTERNS:
        for m in rx.finditer(text):
            g = m.groups()
            d = None
            if kind == 'dmy_text':
                d = _make_date(g[2], MONTHS.get(g[1].lower().rstrip('.')), g[0])
            elif kind == 'mdy_text':
                d = _make_date(g[2], MONTHS.get(g[0].lower().rstrip('.')), g[1])
            elif kind == 'dmy_num':
                d = _make_date(g[2], g[1], g[0])
            elif kind == 'ymd_num':
                d = _make_date(g[0], g[1], g[2])
            if d and 1990 <= d.year <= 2100:
                out.append((d, m.start(), m.end()))
    # remove overlaps (keep first)
    out.sort(key=lambda x: x[1])
    cleaned, last_end = [], -1
    for item in out:
        if item[1] >= last_end:
            cleaned.append(item)
            last_end = item[2]
    return cleaned


def _classify_near(text, start, end):
    """Pick the category whose trigger word sits closest to the date/period (within the same sentence)."""
    lo = max(0, start - 120)
    hi = min(len(text), end + 60)
    window = text[lo:hi]
    low = window.lower()
    pos = start - lo
    # keep to the current sentence where possible
    left = max(low.rfind('. ', 0, pos), low.rfind('\n', 0, pos))
    if left != -1:
        low = low[left + 1:]
        pos -= left + 1
    right = low.find('. ', pos)
    if right != -1:
        low = low[:right]
    best, best_dist = None, 10 ** 6
    for cat, rx, prio in DEADLINE_KEYWORDS:
        for m in re.finditer(rx, low):
            dist = abs(m.start() - pos)
            if dist < best_dist:
                best, best_dist = (cat, prio), dist
    return best or ('Important Date', 'medium')


def _clean_ctx(s):
    s = re.sub(r'\s+', ' ', s).strip()
    return s if len(s) <= 220 else s[:217] + '...'


def extract_deadlines(text, reference_date=None):
    """
    Detects absolute dates (with context) and relative periods (notice period, within X days, etc.).
    Returns list of dicts:
      {title, category, priority, date (ISO or None), period_text (e.g. '30 days'), context, kind: 'date'|'period'}
    """
    if not text:
        return []
    ref = reference_date or date.today()
    results, used = [], set()

    # --- Absolute dates -------------------------------------------------
    for d, start, end in _find_dates(text):
        ctx = _clean_ctx(text[max(0, start - 110): min(len(text), end + 90)])
        cat, prio = _classify_near(text, start, end)
        key = (d, cat)
        if key in used:
            continue
        used.add(key)
        results.append({
            'title': f'{cat}', 'category': cat, 'priority': prio, 'date': d.isoformat(),
            'period_text': None, 'context': ctx, 'kind': 'date',
        })

    # --- Relative periods ------------------------------------------------
    period_rx = re.compile(
        r'(?P<n>\d{1,3}|' + '|'.join(WORD_NUM.keys()) + r')\s*(?:\(\d+\)\s*)?(?P<unit>day|week|month|year)s?\b',
        re.IGNORECASE)
    seen_period = set()
    for m in period_rx.finditer(text):
        n_raw = m.group('n').lower()
        n = int(n_raw) if n_raw.isdigit() else WORD_NUM.get(n_raw, 0)
        if n <= 0:
            continue
        unit = m.group('unit').lower()
        ctx_raw = text[max(0, m.start() - 110): min(len(text), m.end() + 90)]
        near = text[max(0, m.start() - 70): m.end() + 40].lower()
        # keep only periods that sound like obligations
        if not re.search(r'notice|within|prior|before|renew|lock-?in|terminate|grace|cure|expire|period of|term of|tenure|validity|cancel|payable|pay', near):
            continue
        cat, prio = _classify_near(text, m.start(), m.end())
        if cat in ('Start / Effective Date', 'Important Date'):
            cat, prio = 'Time Limit', 'medium'
        period_text = f'{n} {unit}{"s" if n != 1 else ""}'
        key = (period_text, cat, ctx_raw[:40])
        if key in seen_period:
            continue
        seen_period.add(key)
        results.append({
            'title': cat, 'category': cat, 'priority': prio, 'date': None,
            'period_text': period_text, 'context': _clean_ctx(ctx_raw), 'kind': 'period',
        })

    # Sort: dated first (soonest), then periods; high priority first
    order = {'high': 0, 'medium': 1, 'low': 2}
    results.sort(key=lambda r: (r['date'] is None, r['date'] or '9999', order[r['priority']]))
    return results[:25]


def days_until(iso_date, today=None):
    if not iso_date:
        return None
    today = today or date.today()
    try:
        d = datetime.strptime(iso_date, '%Y-%m-%d').date() if isinstance(iso_date, str) else iso_date
    except ValueError:
        return None
    return (d - today).days
