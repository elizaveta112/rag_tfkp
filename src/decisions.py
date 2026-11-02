"""Model-visible option verdicts mapped to the public answer protocol.

No mathematical inference, label lookup, rationale parsing, or answer correction.
Every selected option is an explicit boolean decision made by Qwen.
"""
MODEL_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'checks': {'type': 'array', 'maxItems': 6, 'items': {
            'type': 'object', 'properties': {
                'id': {'type': 'string'},
                'reason': {'type': 'string'},
                'correct': {'type': 'boolean'}},
            'required': ['id', 'reason', 'correct'], 'additionalProperties': False}},
        'rationale': {'type': 'string'},
        'abstain': {'type': 'boolean'},
        'citations': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 4}},
    'required': ['id', 'checks', 'rationale', 'abstain', 'citations'],
    'additionalProperties': False}


def to_public_answer(result, question):
    if not isinstance(result, dict) or set(result) != set(MODEL_RESPONSE_SCHEMA['required']):
        raise ValueError('Return id, checks, rationale, abstain, citations; no answer_ids in model response')
    if result['id'] != question['id'] or type(result['abstain']) is not bool:
        raise ValueError('Wrong question ID or abstain type')
    if not isinstance(result['rationale'], str) or not 1 <= len(result['rationale'].strip()) <= 600:
        raise ValueError('Keep the final rationale within 1–600 characters; checks already contain option reasons')
    checks = result['checks']
    if not isinstance(checks, list) or len(checks) > len(question['options']):
        raise ValueError('checks must be one list with at most one entry per option')
    options = [o['id'] for o in question['options']]
    decisions = {}
    for check in checks:
        if not isinstance(check, dict) or set(check) != {'id', 'reason', 'correct'}:
            raise ValueError('Each check requires id, reason, correct')
        key = check['id']
        if not isinstance(key, str) or key not in options or key in decisions:
            raise ValueError('Unknown or duplicate option ID in checks')
        if type(check['correct']) is not bool:
            raise ValueError('correct must be a JSON boolean, not text or an integer')
        if not isinstance(check['reason'], str) or not 1 <= len(check['reason'].strip()) <= 160:
            raise ValueError('Each option reason must contain 1–160 characters')
        decisions[key] = check['correct']
    selected = [key for key in options if decisions.get(key) is True]
    if result['abstain']:
        if selected or result['citations']:
            raise ValueError('abstain=true requires no true checks and empty citations')
    elif set(decisions) != set(options) or not selected:
        raise ValueError('For an answer, check every actual option ID exactly once and select at least one')
    return {'id': result['id'], 'answer_ids': selected, 'abstain': result['abstain'],
            'rationale': result['rationale'], 'citations': result['citations']}
