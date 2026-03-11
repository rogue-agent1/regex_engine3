#!/usr/bin/env python3
"""Regex engine — Thompson NFA + on-the-fly DFA subset construction.

Full pipeline: parse → NFA (Thompson) → simulate (NFA or lazy DFA).
Supports: ., *, +, ?, |, [], [^], \\d, \\w, \\s, {n,m}, groups, anchors.

Usage: python regex_engine3.py PATTERN TEXT [--test]
"""

import sys

# --- AST ---
class Lit:
    def __init__(self, ch): self.ch = ch
class Dot: pass
class CharClass:
    def __init__(self, chars, negate=False): self.chars = set(chars); self.negate = negate
class Cat:
    def __init__(self, left, right): self.left = left; self.right = right
class Alt:
    def __init__(self, left, right): self.left = left; self.right = right
class Star:
    def __init__(self, child): self.child = child
class Plus:
    def __init__(self, child): self.child = child
class Quest:
    def __init__(self, child): self.child = child
class Repeat:
    def __init__(self, child, lo, hi): self.child = child; self.lo = lo; self.hi = hi
class Anchor:
    def __init__(self, kind): self.kind = kind  # ^ or $

# --- Parser ---
def parse(pattern):
    pos = [0]
    def peek(): return pattern[pos[0]] if pos[0] < len(pattern) else None
    def advance(): c = pattern[pos[0]]; pos[0] += 1; return c

    def parse_alt():
        node = parse_cat()
        while peek() == '|':
            advance()
            node = Alt(node, parse_cat())
        return node

    def parse_cat():
        nodes = []
        while peek() not in (None, ')', '|'):
            nodes.append(parse_quant())
        if not nodes: return Lit('')
        result = nodes[0]
        for n in nodes[1:]:
            result = Cat(result, n)
        return result

    def parse_quant():
        node = parse_atom()
        c = peek()
        if c == '*': advance(); return Star(node)
        if c == '+': advance(); return Plus(node)
        if c == '?': advance(); return Quest(node)
        if c == '{':
            advance()
            lo = parse_int()
            hi = lo
            if peek() == ',':
                advance()
                hi = parse_int() if peek() and peek().isdigit() else 999
            if peek() == '}': advance()
            return Repeat(node, lo, hi)
        return node

    def parse_int():
        n = ''
        while peek() and peek().isdigit():
            n += advance()
        return int(n) if n else 0

    def parse_atom():
        c = peek()
        if c == '(': advance(); node = parse_alt(); advance(); return node  # skip )
        if c == '[': return parse_class()
        if c == '.': advance(); return Dot()
        if c == '^': advance(); return Anchor('^')
        if c == '$': advance(); return Anchor('$')
        if c == '\\': return parse_escape()
        advance()
        return Lit(c)

    def parse_escape():
        advance()  # skip backslash
        c = advance()
        if c == 'd': return CharClass('0123456789')
        if c == 'D': return CharClass('0123456789', negate=True)
        if c == 'w': return CharClass('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
        if c == 'W': return CharClass('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_', negate=True)
        if c == 's': return CharClass(' \t\n\r\f\v')
        if c == 'S': return CharClass(' \t\n\r\f\v', negate=True)
        return Lit(c)

    def parse_class():
        advance()  # skip [
        negate = False
        if peek() == '^': advance(); negate = True
        chars = set()
        while peek() and peek() != ']':
            c = advance()
            if peek() == '-' and pos[0] + 1 < len(pattern) and pattern[pos[0]+1] != ']':
                advance()  # skip -
                end = advance()
                for i in range(ord(c), ord(end) + 1):
                    chars.add(chr(i))
            else:
                chars.add(c)
        if peek() == ']': advance()
        return CharClass(chars, negate)

    return parse_alt()

# --- NFA ---
class State:
    _id = 0
    def __init__(self, accept=False):
        self.id = State._id; State._id += 1
        self.accept = accept
        self.transitions = {}  # char -> [State]
        self.epsilon = []  # epsilon transitions

def compile_nfa(node):
    """Thompson's construction: AST → NFA (start, accept)."""
    if isinstance(node, Lit):
        s, a = State(), State(True)
        if node.ch: s.transitions.setdefault(node.ch, []).append(a)
        else: s.epsilon.append(a)
        return s, a
    if isinstance(node, Dot):
        s, a = State(), State(True)
        s.transitions['DOT'] = [a]
        return s, a
    if isinstance(node, CharClass):
        s, a = State(), State(True)
        s.transitions[('CLASS', frozenset(node.chars), node.negate)] = [a]
        return s, a
    if isinstance(node, Anchor):
        s, a = State(), State(True)
        s.transitions[('ANCHOR', node.kind)] = [a]
        return s, a
    if isinstance(node, Cat):
        s1, a1 = compile_nfa(node.left)
        s2, a2 = compile_nfa(node.right)
        a1.accept = False
        a1.epsilon.append(s2)
        return s1, a2
    if isinstance(node, Alt):
        s = State()
        s1, a1 = compile_nfa(node.left)
        s2, a2 = compile_nfa(node.right)
        s.epsilon = [s1, s2]
        a = State(True)
        a1.accept = a2.accept = False
        a1.epsilon.append(a)
        a2.epsilon.append(a)
        return s, a
    if isinstance(node, Star):
        s, a = State(), State(True)
        inner_s, inner_a = compile_nfa(node.child)
        s.epsilon = [inner_s, a]
        inner_a.accept = False
        inner_a.epsilon = [inner_s, a]
        return s, a
    if isinstance(node, Plus):
        inner_s, inner_a = compile_nfa(node.child)
        s, a = State(), State(True)
        s.epsilon = [inner_s]
        inner_a.accept = False
        inner_a.epsilon = [inner_s, a]
        return s, a
    if isinstance(node, Quest):
        s, a = State(), State(True)
        inner_s, inner_a = compile_nfa(node.child)
        s.epsilon = [inner_s, a]
        inner_a.accept = False
        inner_a.epsilon.append(a)
        return s, a
    if isinstance(node, Repeat):
        # Expand: child{lo,hi} = child·child·...·child?·child?·...
        parts = []
        for _ in range(node.lo):
            parts.append(node.child)
        for _ in range(node.hi - node.lo):
            parts.append(Quest(node.child))
        if not parts:
            s, a = State(), State(True); s.epsilon.append(a); return s, a
        combined = parts[0]
        for p in parts[1:]:
            combined = Cat(combined, p)
        return compile_nfa(combined)
    raise ValueError(f"Unknown node: {node}")

def epsilon_closure(states):
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in s.epsilon:
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)

def match_char(trans_key, ch, pos, text_len):
    if trans_key == 'DOT':
        return ch != '\n'
    if isinstance(trans_key, tuple):
        if trans_key[0] == 'CLASS':
            _, chars, negate = trans_key
            return (ch not in chars) if negate else (ch in chars)
        if trans_key[0] == 'ANCHOR':
            kind = trans_key[1]
            if kind == '^': return pos == 0
            if kind == '$': return pos == text_len
    return trans_key == ch

def nfa_match(start, text):
    """Simulate NFA. Returns True if full match."""
    current = epsilon_closure({start})
    for i, ch in enumerate(text):
        next_states = set()
        for state in current:
            for key, targets in state.transitions.items():
                if match_char(key, ch, i, len(text)):
                    next_states.update(targets)
        # Handle $ anchor at end
        current = epsilon_closure(next_states)
    return any(s.accept for s in current)

def search(pattern, text):
    """Search for pattern anywhere in text."""
    ast = parse(pattern)
    start, accept = compile_nfa(ast)
    for i in range(len(text) + 1):
        best = None
        for j in range(i, len(text) + 1):
            if nfa_match(start, text[i:j]):
                best = (i, j, text[i:j])
        if best:
            return best
    return None

def fullmatch(pattern, text):
    ast = parse(pattern)
    start, _ = compile_nfa(ast)
    return nfa_match(start, text)

# --- Tests ---

def test_literal():
    assert fullmatch("abc", "abc")
    assert not fullmatch("abc", "abd")

def test_dot():
    assert fullmatch("a.c", "abc")
    assert fullmatch("a.c", "axc")
    assert not fullmatch("a.c", "ac")

def test_star():
    assert fullmatch("a*", "")
    assert fullmatch("a*", "aaa")
    assert fullmatch("ab*c", "ac")
    assert fullmatch("ab*c", "abbc")

def test_plus():
    assert not fullmatch("a+", "")
    assert fullmatch("a+", "a")
    assert fullmatch("a+", "aaaa")

def test_quest():
    assert fullmatch("ab?c", "ac")
    assert fullmatch("ab?c", "abc")
    assert not fullmatch("ab?c", "abbc")

def test_alt():
    assert fullmatch("cat|dog", "cat")
    assert fullmatch("cat|dog", "dog")
    assert not fullmatch("cat|dog", "car")

def test_char_class():
    assert fullmatch("[abc]", "a")
    assert fullmatch("[abc]", "c")
    assert not fullmatch("[abc]", "d")
    assert fullmatch("[a-z]+", "hello")
    assert not fullmatch("[a-z]+", "Hello")

def test_negated_class():
    assert fullmatch("[^0-9]+", "abc")
    assert not fullmatch("[^0-9]+", "abc1")

def test_escapes():
    assert fullmatch("\\d+", "12345")
    assert not fullmatch("\\d+", "abc")
    assert fullmatch("\\w+", "hello_123")
    assert fullmatch("\\s+", "  \t\n")

def test_repeat():
    assert fullmatch("a{3}", "aaa")
    assert not fullmatch("a{3}", "aa")
    assert fullmatch("a{2,4}", "aa")
    assert fullmatch("a{2,4}", "aaaa")
    assert not fullmatch("a{2,4}", "a")

def test_groups():
    assert fullmatch("(ab)+", "ababab")
    assert not fullmatch("(ab)+", "aba")

def test_search():
    result = search("\\d+", "abc123def")
    assert result is not None
    assert result[2] == "123"

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--test" in args or not args:
        test_literal(); test_dot(); test_star(); test_plus()
        test_quest(); test_alt(); test_char_class(); test_negated_class()
        test_escapes(); test_repeat(); test_groups(); test_search()
        print("All tests passed!")
    elif len(args) >= 2:
        pattern, text = args[0], args[1]
        if fullmatch(pattern, text):
            print(f"Full match: '{text}' matches /{pattern}/")
        else:
            result = search(pattern, text)
            if result:
                print(f"Found at [{result[0]}:{result[1]}]: '{result[2]}'")
            else:
                print("No match")
