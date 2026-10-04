# Articulator

Articulator is a browser-based idea and vocabulary workbench. Its Ultra Workbench explores semantic contrasts and bridges with sourced lexical relations, transparent on-device vector ranking, and an opt-in AI wildcard lane; the original prompt-writing Studio and local Idea shelf remain available.

## Run it

No build step or API key is required. Serve the repository root over HTTP, for example:

```bash
python3 -m http.server 4173 --bind 0.0.0.0
```

Then open `http://localhost:4173` (or the corresponding preview URL).

## How it works

- **Ultra Workbench:** enter one or more `Toward (A) − Away from (B)` comparisons. Choose a contrast, shared-ground, or midpoint lens, then trace candidates through Open English WordNet senses/relations and optional Datamuse meaning-like and text-association trails. Results show definitions, relation provenance, and a cosine-style alignment score; scores are rankings, not probabilities or factual/causal claims.
- **Embeddings:** pressing **Explore this direction** (or requesting AI wildcards) loads the quantized `Xenova/all-MiniLM-L6-v2` model on demand. Text embedding, matrix pooling, vector operations, and candidate ranking run in the browser. Search terms go to lexical sources only after the user starts a search; they are not sent to the embedding model service.
- **AI (optional):** **Get AI wildcards** is a separate action that sends the chosen concepts and optional context hint to the hosted model through Puter.js to suggest speculative terms. Suggestions are marked unverified and locally ranked; lexical results do not depend on AI. In the original Studio, **Expand this idea** separately sends the full seed and selected meaning links to that hosted service after the user chooses Expand.
- **Prompt Studio and Idea shelf:** the original Studio remains available for developing/copying a prompt; its older meaning map still uses Datamuse WordNet-style relations, not a strict WordNet query. Saved ideas stay in this browser's local storage.
- **Provider status:** Open English WordNet is accessed through its public JSON API. Datamuse currently works without an API key, but its maintainers say requests will require a key starting January 1, 2027; Datamuse associations may need a replacement/configuration after that date. Provider availability can change.

The app is a static HTML/CSS/JavaScript project with no server-side data store. There is no MCP dependency.

## Tests

The pure vector helpers can be checked without a browser or installed dependencies:

```bash
node --test tests/workbench-math.test.js
```

See [the semantic direction workbench implementation note](docs/semantic-direction-calculator.md) for the ranking methods, source boundaries, licensing, and current limitations.
