# Articulator

Articulator is a browser-based semantic writing studio for turning rough thoughts into richer, more useful starting prompts. It combines lexical relationships, optional on-device sentence embeddings, and optional free hosted text generation.

## Run it

No build step or API key is required. Serve the repository root over HTTP, for example:

```bash
python3 -m http.server 4173 --bind 0.0.0.0
```

Then open `http://localhost:4173` (or the corresponding preview URL).

## How it works

- **Word relationships:** on Expand or Refresh map, Articulator sends a few topic words to the free Datamuse API. It groups returned synonyms, broader concepts, more specific ideas, and associations into a clickable meaning map. These are WordNet-style lexical relationships, not a strict WordNet lookup.
- **Embeddings (optional):** choose **Enable** to download the quantized `Xenova/all-MiniLM-L6-v2` model into the browser cache. It embeds the idea and candidate links locally, then ranks the links by cosine similarity. Your text is not sent to the embedding model service.
- **AI expansion (optional):** choosing **Expand this idea** lazy-loads Puter.js and requests an expansion from its `liquid/lfm-2.5-1.2b-instruct:free` model. This requires no app-managed API key. The full seed and selected meaning links are sent to the hosted service only after that action. Provider availability and its free-use policy can change; if AI is unreachable, Articulator creates a useful local draft instead.
- **Privacy and saved ideas:** the editable draft and Idea shelf use this browser's local storage. Word lookups send only extracted topic words to Datamuse; AI use is opt-in.

The app is a static HTML/CSS/JavaScript project with no server-side data store.
