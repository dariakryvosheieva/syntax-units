import csv, json, random, stanza

from lemminflect import getInflection
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency


random.seed(42)

nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma",
    tokenize_pretokenized=False,
    use_gpu=True,
    verbose=False,
)
detok = TreebankWordDetokenizer()


AUX_LEMMAS = {
    "be",
    "do",
    "have",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
}

DETERMINERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "some",
    "any",
    "each",
    "every",
    "either",
    "neither",
    "no",
    "all",
    "both",
    "many",
    "few",
    "several",
    "much",
    "little",
    "another",
    "such",
    "what",
    "whose",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
}

BOUNDARY_UPOS = {"VERB", "AUX", "ADP", "SCONJ", "CCONJ", "PUNCT", "INTJ", "SYM"}


def build_wordnet_pool(pos: str, min_freq=2.0, max_lemmas=None):
    wn_pos = wn.NOUN if pos == "n" else wn.VERB
    pool = set()
    for syn in wn.all_synsets(pos=wn_pos):
        for l in syn.lemmas():
            name = l.name().lower()
            # Skip multi-word, underscore, hyphen, numerals, etc.
            if "_" in name or not name.isalpha():
                continue
            if pos == "v" and name in AUX_LEMMAS:
                continue
            if zipf_frequency(name, "en") < min_freq:
                continue
            pool.add(name)
            if max_lemmas and len(pool) >= max_lemmas:
                break
        if max_lemmas and len(pool) >= max_lemmas:
            break
    return pool


def get_candidates(pool, orig_word, orig_freq, orig_len, tolerance=1.0, len_tol=3):
    # Find candidates within frequency and length tolerance
    candidates = []
    for word in pool:
        if word == orig_word:
            continue
        # Only allow single alphabetic words
        if not word.isalpha():
            continue
        freq = zipf_frequency(word, "en")
        if freq < 2.0:  # skip rare words
            continue
        if abs(freq - orig_freq) <= tolerance and abs(len(word) - orig_len) <= len_tol:
            candidates.append(word)
    return candidates


def has_det_or_prep(i, tokens, upos):
    k = i - 1
    while k >= 0:
        if upos[k] in BOUNDARY_UPOS or tokens[k] in {",", ";", ":"}:
            break
        if upos[k] == "DET" or tokens[k].lower() in DETERMINERS:
            return True
        if upos[k] == "ADP":
            return True
        k -= 1
    return False


def is_head_noun(i, upos, tokens):
    k = i + 1
    while k < len(tokens):
        if upos[k] in BOUNDARY_UPOS or tokens[k] in {",", ";", ":"}:
            break
        if upos[k] == "NOUN":
            # ignore following gerunds (progressive modifiers)
            if tokens[k].lower().endswith("ing"):
                k += 1
                continue
            return False
        k += 1
    return True


def stanza_tokens(sentence):
    doc = nlp(sentence)
    out = []
    for s in doc.sentences:
        for w in s.words:
            feats = {
                kv.split("=")[0]: kv.split("=")[1]
                for kv in (w.feats or "").split("|")
                if kv
            }
            out.append((w.text, w.upos, w.lemma.lower(), w.xpos, feats))
    return out


def needs_an(word):
    w = word.lower()
    if w.startswith(("honest", "hour", "heir", "honor", "herb")):
        return True
    if w.startswith(("un", "uni", "use", "user", "uk", "ur")):
        return False
    return w[0] in "aeiou"


def substitute_word(sentence, pool, pos):
    tokens, upos, lemmas, xpos, feats = zip(*stanza_tokens(sentence))
    target_upos = "NOUN" if pos == "n" else "VERB"

    for i, (tok, up, lem, xp, ft) in enumerate(zip(tokens, upos, lemmas, xpos, feats)):
        if up != target_upos:
            continue

        if pos == "v" and lem in AUX_LEMMAS:
            continue

        if pos == "n":
            if not has_det_or_prep(i, tokens, upos):
                continue
            if not is_head_noun(i, upos, tokens):
                continue

        # Ensure we only try replacing lemmas that exist in WordNet pool
        if lem not in pool:
            continue

        # Skip fixed quantifier "a lot of"
        if (
            pos == "n"
            and lem == "lot"
            and i > 0
            and tokens[i - 1].lower() in {"a", "an"}
            and i + 1 < len(tokens)
            and tokens[i + 1].lower() == "of"
        ):
            continue

        if pos == "n":
            ptb = "NNS" if ft.get("Number") == "Plur" else "NN"
        else:
            ptb = xp

        orig_freq = zipf_frequency(lem, "en")
        if orig_freq < 2.0:
            continue
        orig_len = len(lem)

        candidates = get_candidates(
            pool, lem, orig_freq, orig_len, tolerance=1.0, len_tol=3
        )
        if not candidates:
            continue
        random.shuffle(candidates)

        for candidate in candidates:
            new_tokens = list(tokens)
            new_form = getInflection(candidate, tag=ptb)
            if not new_form:
                continue
            surface = new_form[0]

            if zipf_frequency(surface, "en") < 2.0:
                continue

            new_tokens[i] = surface

            if pos == "n":
                # Adjust determiner a/an
                if i > 0 and new_tokens[i - 1].lower() in {"a", "an"}:
                    new_tokens[i - 1] = "an" if needs_an(surface) else "a"
                elif ptb == "NN":
                    # If no preceding determiner, insert 'the'
                    has_det = False
                    k = i - 1
                    while k >= 0:
                        if upos[k] in BOUNDARY_UPOS or tokens[k] in {",", ";", ":"}:
                            break
                        if upos[k] == "DET" or tokens[k].lower() in DETERMINERS:
                            has_det = True
                            break
                        k -= 1
                    if not has_det:
                        new_tokens.insert(i, "the")

            return detok.detokenize(new_tokens)

    return None


def transform_blimp_file(in_path, out_path, pos):
    records = [json.loads(l) for l in open(in_path)]
    pool = build_wordnet_pool(pos)

    with open(out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        max_words = 0
        all_good = []
        all_bad = []

        for minimal_pair in records:
            good = minimal_pair["sentence_good"]
            bad = substitute_word(good, pool, pos)

            if not bad:
                continue

            good_words = good.split()
            bad_words = bad.split()

            max_words = max(max_words, len(good_words), len(bad_words))
            all_good.append(good_words)
            all_bad.append(bad_words)

        header = ["stim" + str(i + 1) for i in range(max_words + 2)]
        writer.writerow(header)
        for i, (good_words, bad_words) in enumerate(zip(all_good, all_bad)):
            writer.writerow([2 * i + 1] + good_words + ["+"])
            writer.writerow([2 * i + 2] + bad_words + ["-"])


if __name__ == "__main__":
    pos = "v"  # "n"
    from glob import glob

    for fn in glob("raw/blimp/*.jsonl"):
        out = fn.replace("raw/blimp", f"processed/blimp-{pos}-sub")
        out = out.replace(".jsonl", f"-{pos}-sub.csv")
        transform_blimp_file(fn, out, pos)
