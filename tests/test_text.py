"""Units, prompt terms and hallucination filters."""
from livecaptions.config.schema import FilterCfg, TermsCfg
from livecaptions.engines.base import Piece, Segment
from livecaptions.filters import hallucination as hf
from livecaptions.segmentation.prompt import PromptBuilder, split_terms, terms2line
from livecaptions.segmentation.units import Unit, kept_prefix, segments2units, units2text


def make_units(keys):
    return [Unit(k, k, i, i + 1) for i, k in enumerate(keys)]


def test_units_split_words_and_cjk():
    seg = Segment(0.0, 2.0, [Piece(b' Hello,', 0.0, 0.5), Piece(b' world', 0.5, 1.0), Piece('。突'.encode(), 1.0, 1.5),
                             Piece('触'.encode(), 1.5, 2.0)])
    units = segments2units([seg], 10.0)
    assert [u.key for u in units] == ['hello', 'world', '突', '触']
    assert units2text(units) == 'Hello, world。突触'
    assert units[0].t0 == 10.0 and units[-1].t1 == 12.0


def test_units_partial_utf8_tokens():
    data = '递'.encode()
    seg = Segment(0, 1, [Piece(data[:2], 0.0, 0.4), Piece(data[2:], 0.4, 0.8)])
    units = segments2units([seg], 0.0)
    assert [u.key for u in units] == ['递']
    assert units[0].t0 == 0.0 and units[0].t1 == 0.8


def test_units_segment_index():
    segs = [Segment(0, 1, [Piece(b' a', 0, 1)]), Segment(1, 2, [Piece(b' b', 1, 2)])]
    assert [u.segment for u in segments2units(segs, 0)] == [0, 1]


def test_kept_prefix():
    kept = make_units(list('abcdef'))
    assert kept_prefix(make_units(list('abcdefgh')), kept) == 6
    assert kept_prefix(make_units(list('abXdefgh')), kept) == 6       # a misheard unit inside the kept text
    assert kept_prefix(make_units(list('abcdeZgh')), kept) == 6       # only the last unit differs
    assert kept_prefix(make_units(list('bcdefgh')), kept) == 5        # the first unit dropped


def test_terms_line_and_split():
    assert split_terms('突触 递质、抑制,  synapse') == ['突触', '递质', '抑制', 'synapse']
    assert terms2line(['突触', '递质']) == '突触、递质。'
    assert terms2line(['axon', 'dendrite']) == 'axon, dendrite.'


class FakeTokenizer:
    def tokenize(self, text):
        return list(range(len(text)))


def test_prompt_budget():
    pb = PromptBuilder(FakeTokenizer())
    cfg = TermsCfg(term_tokens=8, prompt_tokens=12)
    pb.set_terms('aaa bbb ccc', cfg, translating=False)
    assert pb.terms_cut and len(pb.term_tokens) <= 8   # 'aaa, bbb.' fits, the third term does not
    assert len(pb.build('x' * 100, cfg)) == 12
    pb.set_terms('aaa', cfg, translating=True)
    assert pb.term_tokens == []                          # no terms when translating


def test_blacklist_and_exact():
    fc = FilterCfg()
    assert hf.is_blacklisted('请不吝点赞 订阅 转发', fc)
    assert hf.is_blacklisted('Thank you.', fc)
    assert not hf.is_blacklisted('Thank you for coming to the lecture', fc)
    fc.blacklist_on = False
    assert not hf.is_blacklisted('请不吝点赞', fc)


def test_cut_repetitions():
    units = make_units(['the', 'sun', 'and'] + ['the', 'sun'] * 6)
    out = hf.cut_repetitions(units, 4)
    assert [u.key for u in out] == ['the', 'sun', 'and'] + ['the', 'sun'] * 4


def test_terms_only_and_kana():
    assert hf.is_terms_only(make_units(['抑', '制']), ['抑制', '突触'])
    assert not hf.is_terms_only(make_units(['抑', '制', '了']), ['抑制'])
    units = [Unit('はい', 'はい', 0, 1), Unit('好', '好', 1, 2)]
    assert [u.key for u in hf.drop_kana(units)] == ['好']


def test_unlikely_speech_and_density():
    fc = FilterCfg()
    assert hf.is_unlikely_speech(Segment(0, 1, no_speech_prob=0.9, avg_logprob=-1.5), fc)
    assert not hf.is_unlikely_speech(Segment(0, 1, no_speech_prob=0.9, avg_logprob=-0.2), fc)
    assert hf.is_too_dense(make_units(['x'] * 40), 1.0, fc)
    assert not hf.is_too_dense(make_units(['x'] * 5), 1.0, fc)
