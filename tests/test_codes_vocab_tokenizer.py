import pytest

from codeformer.data.codes import CodeHierarchy, CodeSpec, format_codes, parse_codes
from codeformer.data.tokenizer import SimpleTokenizer
from codeformer.data.vocab import SPECIALS, Vocab, code_token

SPEC = CodeSpec(6, "-")


class TestParseCodes:
    def test_string(self):
        assert parse_codes("A-B-C-D-E-F", SPEC) == tuple("ABCDEF")

    def test_list_and_whitespace(self):
        assert parse_codes([" A", "B ", "C", "D", "E", "F"], SPEC) == tuple("ABCDEF")

    @pytest.mark.parametrize("bad", ["A-B-C", "A-B-C-D-E-F-G", "A--C-D-E-F", ""])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_codes(bad, SPEC)

    def test_roundtrip(self):
        assert format_codes(parse_codes("1-2-3-4-5-6", SPEC), SPEC) == "1-2-3-4-5-6"

    def test_custom_delimiter(self):
        s = CodeSpec(3, "/")
        assert parse_codes("a/b/c", s) == ("a", "b", "c")


class TestHierarchy:
    def test_children_and_membership(self):
        h = CodeHierarchy(3)
        h.add_many([("a", "x", "1"), ("a", "y", "2"), ("b", "x", "3")])
        assert h.children(()) == {"a", "b"}
        assert h.children(("a",)) == {"x", "y"}
        assert h.children(("b", "x")) == {"3"}
        assert ("a", "y", "2") in h and ("a", "y", "3") not in h
        assert h.children(("zzz",)) == set()

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            CodeHierarchy(3).add(("a", "b"))

    def test_serialisation(self):
        h = CodeHierarchy.from_list(2, [["a", "b"], ["a", "c"]])
        assert CodeHierarchy.from_list(2, h.to_list()).children(("a",)) == {"b", "c"}


class TestVocab:
    def test_specials_first_and_stable(self):
        v = Vocab()
        assert tuple(v.token(i) for i in range(len(SPECIALS))) == SPECIALS
        assert v.pad_id == 0

    def test_level_tagged_codes_do_not_collide(self):
        v = Vocab()
        v.extend([], [("01", "01")])
        assert v.code_id(0, "01") != v.code_id(1, "01")
        assert v.code_of(v.code_id(1, "01")) == (1, "01")

    def test_unknown_token_maps_to_unk(self):
        assert Vocab().id("nope") == Vocab().unk_id

    def test_min_freq(self):
        v = Vocab()
        v.extend([["a", "a", "b"]], [], min_freq=2)
        assert "a" in v and "b" not in v

    def test_roundtrip_and_extension_preserves_ids(self):
        v = Vocab()
        v.extend([["hello", "world"]], [("A", "B")])
        before = {t: v.id(t) for t in v.to_list()}
        v2 = Vocab.from_list(v.to_list())
        v2.extend([["new"]], [("A", "C")])
        assert all(v2.id(t) == i for t, i in before.items())
        assert code_token(1, "C") in v2

    def test_corrupt_vocab_rejected(self):
        with pytest.raises(ValueError):
            Vocab.from_list(["x", "y"])

    def test_code_ids_for_level(self):
        v = Vocab()
        v.extend([["word"]], [("A", "B"), ("A", "C")])
        assert len(v.code_ids_for_level(1)) == 2
        assert len(v.code_ids_for_level(0)) == 1


def test_tokenizer():
    t = SimpleTokenizer()
    assert t.tokenize("Hello, World-42!") == ["hello", ",", "world", "-", "42", "!"]
    assert SimpleTokenizer(lowercase=False).tokenize("AbC") == ["AbC"]
    assert t.tokenize("") == []
