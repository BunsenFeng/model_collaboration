"""
IFEval instruction-following checkers.
Each checker takes (response: str, kwargs: dict) and returns bool.
Score per prompt = fraction of instructions satisfied.
"""

import re
import json

# ── helpers ──────────────────────────────────────────────────────────────────

def _count_words(text):
    return len(text.split())

def _count_sentences(text):
    # split on .  !  ? followed by whitespace or end
    sentences = re.split(r'[.!?]+(?:\s|$)', text.strip())
    return len([s for s in sentences if s.strip()])

def _paragraphs(text):
    return [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]

def _apply_relation(count, relation, target):
    if relation in ('at least', 'MORE THAN', 'more than'):
        return count >= target
    if relation in ('less than', 'at most', 'LESS THAN'):
        return count <= target
    if relation in ('exactly', 'EXACTLY'):
        return count == target
    # fallback
    return count >= target

# ── checkers ─────────────────────────────────────────────────────────────────

def check_punctuation_no_comma(response, kwargs):
    return ',' not in response


def check_length_number_words(response, kwargs):
    return _apply_relation(_count_words(response), kwargs['relation'], kwargs['num_words'])


def check_length_number_sentences(response, kwargs):
    return _apply_relation(_count_sentences(response), kwargs['relation'], kwargs['num_sentences'])


def check_length_number_paragraphs(response, kwargs):
    return _apply_relation(len(_paragraphs(response)), 'at least', kwargs['num_paragraphs']) and \
           _apply_relation(len(_paragraphs(response)), 'less than', kwargs['num_paragraphs'] + 1)


def check_length_nth_paragraph_first_word(response, kwargs):
    paras = _paragraphs(response)
    n = kwargs['nth_paragraph']  # 1-indexed
    first_word = kwargs['first_word'].lower()
    if n > len(paras):
        return False
    words = paras[n - 1].split()
    return bool(words) and words[0].lower() == first_word


def check_keywords_existence(response, kwargs):
    text = response.lower()
    return all(kw.lower() in text for kw in kwargs['keywords'])


def check_keywords_forbidden_words(response, kwargs):
    text = response.lower()
    return all(fw.lower() not in text for fw in kwargs['forbidden_words'])


def check_keywords_frequency(response, kwargs):
    keyword = kwargs['keyword'].lower()
    count = response.lower().count(keyword)
    return _apply_relation(count, kwargs['relation'], kwargs['frequency'])


def check_keywords_letter_frequency(response, kwargs):
    letter = kwargs['letter'].lower()
    count = response.lower().count(letter)
    return _apply_relation(count, kwargs['let_relation'], kwargs['let_frequency'])


def check_detectable_format_number_highlighted_sections(response, kwargs):
    # *text* but not **text** (bold)
    highlights = re.findall(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', response)
    return len(highlights) >= kwargs['num_highlights']


def check_detectable_format_number_bullet_lists(response, kwargs):
    bullets = re.findall(r'^\s*[-*•]\s+.+', response, re.MULTILINE)
    return len(bullets) >= kwargs['num_bullets']


def check_detectable_format_title(response, kwargs):
    return bool(re.search(r'^#{1,6}\s+\S+', response, re.MULTILINE))


def check_detectable_format_multiple_sections(response, kwargs):
    splitter = re.escape(kwargs['section_spliter'])
    matches = re.findall(rf'{splitter}\s*\d+', response, re.IGNORECASE)
    return len(matches) >= kwargs['num_sections']


def check_detectable_format_json_format(response, kwargs):
    text = response.strip()
    # strip markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        json.loads(text)
        return True
    except Exception:
        return False


_CONSTRAINED_RESPONSES = [
    'my answer is yes.',
    'my answer is no.',
    'my answer is maybe.',
]

def check_detectable_format_constrained_response(response, kwargs):
    text = response.lower()
    return any(cr in text for cr in _CONSTRAINED_RESPONSES)


def check_startend_end_checker(response, kwargs):
    return response.strip().endswith(kwargs['end_phrase'])


def check_startend_quotation(response, kwargs):
    text = response.strip()
    return text.startswith('"') and text.endswith('"')


def check_change_case_english_capital(response, kwargs):
    letters = [c for c in response if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def check_change_case_english_lowercase(response, kwargs):
    letters = [c for c in response if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def check_change_case_capital_word_frequency(response, kwargs):
    words = response.split()
    if not words:
        return False
    cap_count = sum(1 for w in words if w and w[0].isupper())
    return _apply_relation(cap_count, kwargs['capital_relation'], kwargs['capital_frequency'])


def check_combination_repeat_prompt(response, kwargs):
    prompt = kwargs.get('prompt_to_repeat', '').strip()
    return prompt and prompt in response


def check_combination_two_responses(response, kwargs):
    return '******' in response


def check_detectable_content_number_placeholders(response, kwargs):
    placeholders = re.findall(r'\[[^\]]+\]', response)
    return len(placeholders) >= kwargs['num_placeholders']


def check_detectable_content_postscript(response, kwargs):
    marker = kwargs.get('postscript_marker', 'P.S.')
    return marker in response


def check_language_response_language(response, kwargs):
    try:
        from langdetect import detect
        detected = detect(response)
        # map full language names to ISO codes
        lang_map = {
            'arabic': 'ar', 'russian': 'ru', 'german': 'de', 'italian': 'it',
            'vietnamese': 'vi', 'urdu': 'ur', 'tamil': 'ta', 'bengali': 'bn',
            'gujarati': 'gu', 'finnish': 'fi', 'korean': 'ko', 'bulgarian': 'bg',
            'swahili': 'sw', 'persian': 'fa', 'punjabi': 'pa', 'nepali': 'ne',
            'marathi': 'mr', 'telugu': 'te', 'kannada': 'kn', 'thai': 'th',
            'portuguese': 'pt', 'hindi': 'hi',
        }
        target = kwargs['language'].lower()
        target_code = lang_map.get(target, target)
        return detected == target_code
    except Exception:
        return True  # can't check without langdetect, give benefit of the doubt


# ── dispatch table ────────────────────────────────────────────────────────────

_CHECKERS = {
    'punctuation:no_comma':                           check_punctuation_no_comma,
    'length_constraints:number_words':                check_length_number_words,
    'length_constraints:number_sentences':            check_length_number_sentences,
    'length_constraints:number_paragraphs':           check_length_number_paragraphs,
    'length_constraints:nth_paragraph_first_word':    check_length_nth_paragraph_first_word,
    'keywords:existence':                             check_keywords_existence,
    'keywords:forbidden_words':                       check_keywords_forbidden_words,
    'keywords:frequency':                             check_keywords_frequency,
    'keywords:letter_frequency':                      check_keywords_letter_frequency,
    'detectable_format:number_highlighted_sections':  check_detectable_format_number_highlighted_sections,
    'detectable_format:number_bullet_lists':          check_detectable_format_number_bullet_lists,
    'detectable_format:title':                        check_detectable_format_title,
    'detectable_format:multiple_sections':            check_detectable_format_multiple_sections,
    'detectable_format:json_format':                  check_detectable_format_json_format,
    'detectable_format:constrained_response':         check_detectable_format_constrained_response,
    'startend:end_checker':                           check_startend_end_checker,
    'startend:quotation':                             check_startend_quotation,
    'change_case:english_capital':                    check_change_case_english_capital,
    'change_case:english_lowercase':                  check_change_case_english_lowercase,
    'change_case:capital_word_frequency':             check_change_case_capital_word_frequency,
    'combination:repeat_prompt':                      check_combination_repeat_prompt,
    'combination:two_responses':                      check_combination_two_responses,
    'detectable_content:number_placeholders':         check_detectable_content_number_placeholders,
    'detectable_content:postscript':                  check_detectable_content_postscript,
    'language:response_language':                     check_language_response_language,
}


def score_ifeval_response(response, instruction_id_list, kwargs_list):
    """
    Returns fraction of instructions satisfied (0.0–1.0).
    """
    if not instruction_id_list:
        return 1.0
    results = []
    for instr_id, kw in zip(instruction_id_list, kwargs_list):
        checker = _CHECKERS.get(instr_id)
        if checker is None:
            results.append(True)  # unknown instruction — give benefit of the doubt
            continue
        try:
            results.append(checker(response, kw))
        except Exception:
            results.append(False)
    return sum(results) / len(results)
