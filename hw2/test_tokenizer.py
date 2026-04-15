import pickle
from collections import Counter


VOCAB_PATH = "tokenizer_results/TinyStoriesV2-GPT4-train_vocab.pkl"
MERGES_PATH = "tokenizer_results/TinyStoriesV2-GPT4-train_merges.pkl"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def byte_symbols(text: str):
    # list of 1-byte bytes tokens: [b'h', b'e', ...]
    bs = text.encode("utf-8")
    return [bytes([b]) for b in bs]

def apply_merges(symbols, merges, max_merges=None):
    """
    symbols: list[bytes] where each bytes can be length>=1 (after merges)
    merges: list[tuple[bytes, bytes]]
    """
    if max_merges is None:
        max_merges = len(merges)

    for (a, b) in merges[:max_merges]:
        i = 0
        out = []
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)   # <-- important: merged token is concatenated bytes
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols

def main():
    vocab = load_pickle(VOCAB_PATH)
    merges = load_pickle(MERGES_PATH)

    print("=== Loaded ===")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges count: {len(merges)}")
    print("First 5 merges:", merges[:5])

    # --- Encode test ---
    text = "hello world"
    symbols = byte_symbols(text)
    merged = apply_merges(symbols, merges, max_merges=500)

    print("\n=== Encode Test ===")
    print("Before tokens:", len(symbols), "After tokens:", len(merged))
    print("Before (first 30):", symbols[:30])
    print("After  (first 30):", merged[:30])

    # sanity: decode back (should always work for byte-level if you just concat)
    recon = b"".join(merged).decode("utf-8")
    print("\nRound-trip ok:", recon == text)

    # show if expected frequent merges appear
    sample = ["the the the", " and the ", "there", "apple apples"]
    print("\n=== Token count reduction samples ===")
    for t in sample:
        s = byte_symbols(t)
        m = apply_merges(s, merges, max_merges=2000)
        print(f"{t!r}: {len(s)} -> {len(m)}")

if __name__ == "__main__":
    main()