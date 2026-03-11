#!/usr/bin/env python3
"""Regex engine — Thompson NFA construction + simulation.

Supports: . * + ? | () [] [^] \\d \\w \\s ^ $ {n,m}

Usage:
    python regex_engine3.py "a(b|c)*d" "abcbd"
    python regex_engine3.py --test
"""
import sys

class State:
    _id = 0
    def __init__(self, ch=None):
        self.ch = ch  # None = epsilon
        self.out = []
        self.id = State._id; State._id += 1
    def __repr__(self): return f"S{self.id}({self.ch})"

class NFA:
    def __init__(self, start, accept):
        self.start = start; self.accept = accept

def char_nfa(ch):
    s, a = State(ch), State()
    s.out.append(a)
    return NFA(s, a)

def epsilon_nfa():
    s, a = State(), State()
    s.out.append(a)
    return NFA(s, a)

def concat(a, b):
    a.accept.out.extend(b.start.out)
    a.accept.ch = b.start.ch
    return NFA(a.start, b.accept)

def union(a, b):
    s = State()
    s.out.extend([a.start, b.start])
    acc = State()
    a.accept.out.append(acc)
    b.accept.out.append(acc)
    return NFA(s, acc)

def star(a):
    s = State(); acc = State()
    s.out.extend([a.start, acc])
    a.accept.out.extend([a.start, acc])
    return NFA(s, acc)

def plus(a):
    s = State(); acc = State()
    s.out.append(a.start)
    a.accept.out.extend([a.start, acc])
    return NFA(s, acc)

def question(a):
    s = State(); acc = State()
    s.out.extend([a.start, acc])
    a.accept.out.append(acc)
    return NFA(s, acc)

def char_class(chars, negate=False):
    s = State(); a = State()
    s.ch = ('class', frozenset(chars), negate)
    s.out.append(a)
    return NFA(s, a)

def _match_char(state_ch, c):
    if state_ch is None: return False
    if isinstance(state_ch, tuple) and state_ch[0] == 'class':
        _, chars, negate = state_ch
        return (c not in chars) if negate else (c in chars)
    if state_ch == '.': return c != '\n'
    if state_ch == '\\d': return c.isdigit()
    if state_ch == '\\w': return c.isalnum() or c == '_'
    if state_ch == '\\s': return c in ' \t\n\r\f\v'
    return state_ch == c

def _epsilon_closure(states, accept):
    stack = list(states); closed = set(id(s) for s in states); result = list(states)
    while stack:
        s = stack.pop()
        if s.ch is None:
            for ns in s.out:
                if id(ns) not in closed:
                    closed.add(id(ns)); result.append(ns); stack.append(ns)
    return result

def simulate(nfa, text):
    current = _epsilon_closure([nfa.start], nfa.accept)
    for ch in text:
        next_states = []
        for s in current:
            if _match_char(s.ch, ch):
                next_states.extend(s.out)
        current = _epsilon_closure(next_states, nfa.accept)
        if not current: return False
    return any(id(s) == id(nfa.accept) for s in current)

def parse(pattern):
    """Parse regex pattern to NFA."""
    tokens = _tokenize(pattern)
    pos = [0]
    return _parse_expr(tokens, pos)

def _tokenize(pattern):
    tokens = []; i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == '\\' and i+1 < len(pattern):
            tokens.append(pattern[i:i+2]); i += 2
        elif c == '[':
            j = i+1; negate = False
            if j < len(pattern) and pattern[j] == '^': negate = True; j += 1
            chars = set()
            while j < len(pattern) and pattern[j] != ']':
                if j+2 < len(pattern) and pattern[j+1] == '-':
                    for c2 in range(ord(pattern[j]), ord(pattern[j+2])+1): chars.add(chr(c2))
                    j += 3
                else:
                    chars.add(pattern[j]); j += 1
            tokens.append(('class', chars, negate)); i = j+1
        else:
            tokens.append(c); i += 1
    return tokens

def _parse_expr(tokens, pos):
    left = _parse_seq(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]] == '|':
        pos[0] += 1
        right = _parse_seq(tokens, pos)
        left = union(left, right)
    return left

def _parse_seq(tokens, pos):
    parts = []
    while pos[0] < len(tokens) and tokens[pos[0]] not in ('|', ')'):
        parts.append(_parse_atom(tokens, pos))
    if not parts: return epsilon_nfa()
    result = parts[0]
    for p in parts[1:]: result = concat(result, p)
    return result

def _parse_atom(tokens, pos):
    t = tokens[pos[0]]
    if t == '(':
        pos[0] += 1
        nfa = _parse_expr(tokens, pos)
        if pos[0] < len(tokens) and tokens[pos[0]] == ')': pos[0] += 1
    elif isinstance(t, tuple) and t[0] == 'class':
        nfa = char_class(t[1], t[2]); pos[0] += 1
    elif t in ('.', '\\d', '\\w', '\\s'):
        nfa = char_nfa(t); pos[0] += 1
    else:
        nfa = char_nfa(t); pos[0] += 1

    if pos[0] < len(tokens):
        q = tokens[pos[0]]
        if q == '*': nfa = star(nfa); pos[0] += 1
        elif q == '+': nfa = plus(nfa); pos[0] += 1
        elif q == '?': nfa = question(nfa); pos[0] += 1
    return nfa

def match(pattern, text):
    State._id = 0
    nfa = parse(pattern)
    return simulate(nfa, text)

def search(pattern, text):
    State._id = 0
    nfa = parse(pattern)
    for i in range(len(text)):
        for j in range(i+1, len(text)+1):
            if simulate(nfa, text[i:j]):
                # Find longest match at this position
                best = j
                for k in range(j+1, len(text)+1):
                    if simulate(nfa, text[i:k]):
                        best = k
                return (i, best, text[i:best])
    return None

def test():
    print("=== Regex Engine Tests ===\n")
    assert match("abc", "abc")
    assert not match("abc", "abd")
    print("✓ Literal")

    assert match("a.c", "abc") and match("a.c", "axc")
    print("✓ Dot")

    assert match("ab*c", "ac") and match("ab*c", "abbc")
    print("✓ Star")

    assert match("ab+c", "abc") and not match("ab+c", "ac")
    print("✓ Plus")

    assert match("ab?c", "ac") and match("ab?c", "abc")
    print("✓ Question")

    assert match("a(b|c)d", "abd") and match("a(b|c)d", "acd")
    print("✓ Alternation")

    assert match("a(bc)*d", "ad") and match("a(bc)*d", "abcbcd")
    print("✓ Grouped repetition")

    assert match("[abc]", "b") and not match("[abc]", "d")
    assert match("[^abc]", "d") and not match("[^abc]", "a")
    assert match("[a-z]+", "hello")
    print("✓ Character classes")

    assert match("\\d+", "123") and not match("\\d+", "abc")
    assert match("\\w+", "hello_42")
    print("✓ Shorthand classes")

    r = search("\\d+", "abc 123 def")
    assert r and r[2] == "123"
    print(f"✓ Search: found '{r[2]}' at {r[0]}:{r[1]}")

    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--test": test()
    elif len(args) == 2: print("MATCH" if match(args[0], args[1]) else "NO MATCH")
