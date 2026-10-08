import contextlib
import json
import pathlib
import sys

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / 'pipeline'))

from encode_text import embed_text, load_text_encoder

# torch and faiss each bundle libomp on macOS, so this process imports torch and never faiss.
# app.py holds the indexes and sends one JSON-encoded query per line on stdin. this worker
# answers with one JSON line per query, {model_key: embedding}, on stdout.

model_keys = ['fashion_clip', 'clip_vit_b32']

def load_encoders():
    encoders = {}

    for model_key in model_keys:
        try:
            encoders[model_key] = load_text_encoder(model_key)
        except Exception as e:
            sys.exit(f"embed_worker: could not load the {model_key} text encoder, {type(e).__name__}: {e}")

    return encoders

def embed_query(encoders, query):
    try:
        return {model_key: embed_text(encoder, [query])[0].tolist() for model_key, encoder in encoders.items()}
    except Exception as e:
        return {'error': f"{type(e).__name__}: {e}"}

if __name__ == '__main__':
    protocol_out = sys.stdout

    # encode_text prints load info to stdout, which would break the line protocol, so it goes to stderr
    with contextlib.redirect_stdout(sys.stderr):
        encoders = load_encoders()

    protocol_out.write('ready\n')
    protocol_out.flush()

    for line in sys.stdin:
        query = json.loads(line)

        with contextlib.redirect_stdout(sys.stderr):
            reply = embed_query(encoders, query)

        protocol_out.write(json.dumps(reply) + '\n')
        protocol_out.flush()
