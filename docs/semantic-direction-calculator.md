# Semantic direction workbench

- **Status:** Implemented in the Ultra Articulator Workbench
- **Scope:** Exploratory vocabulary search using live lexical sources and on-device embeddings

## What it does

The default Workbench searches a semantic direction described by one or more paired concepts:

```text
Toward (A) − Away from (B)
```

For example, `orthodox − religion` asks which retrieved candidates align with the embedding-space direction from *religion* toward *orthodox*. Add up to four pairs when they express the same kind of contrast. Choose among three lenses:

- **Contrast:** normalize each `embed(A) − embed(B)` delta, average the deltas, then normalize the result.
- **Shared ground:** rank a candidate by the average, across pairs, of its lower cosine similarity to each side of the pair. This is not vector subtraction.
- **Blend:** normalize the mean of the A and B embeddings and rank candidates by cosine similarity to that midpoint.

These operations rank model representations; they do not literally remove a meaning, establish a dictionary sense, or prove a real-world claim. Scores are alignment values, not probabilities or truth scores. The optional context hint is included in the local embedding text to help with ambiguous inputs, but it does not select or verify a WordNet sense.

## Sources and provenance

The Workbench keeps candidate retrieval separate from embedding ranking. Results display source, relation trail, definition/example when returned, and OEWN synset IDs where available. A merged word can retain evidence from multiple source terms and synsets.

- **Open English WordNet (OEWN):** uses the documented public JSON API at `https://en-word.net/api/lemma/{lemma}` and `https://en-word.net/api/synset/{synset-id}` to retrieve input senses, synset members, glosses, examples, and selected lexical relations. This is a live online lookup, not a bundled local dictionary or a full-vocabulary vector index. OEWN's 2025 release is CC BY 4.0; the interface links to OEWN and its license. OEWN derives from Princeton WordNet; a redistributed local subset or index should preserve OEWN attribution/license terms and applicable upstream Princeton notices.
- **Datamuse:** optional `ml` meaning-like results and `rel_trg` text associations broaden the candidate field. These are Datamuse's search/association signals, not verified WordNet relations. The `md=dp` response can include Wiktionary/WordNet-derived definitions; each is labeled as returned via Datamuse. Datamuse says requests will require an API key starting January 1, 2027, so the association trail may need configuration or replacement after that date.
- **AI wildcards:** a distinct, explicitly requested Puter.js action can suggest speculative vocabulary. These suggestions are marked unverified and ranked locally; they are not dictionary facts and do not feed the lexical results.

When the user presses **Explore**, the entered terms are sent to the selected lexical services. Embedding and vector scoring run in the browser. The AI receives concepts and the optional context hint only after the user selects **Get AI wildcards**. No MCP service is required. See the links in the Workbench for provider documentation and licensing.

## Vector helpers and tests

The math is independent of OEWN and Datamuse. In the browser the pure functions are exposed as `window.ArticulatorVectorMath`; Node can import them from `workbench.js`. The API includes:

```js
poolEmbeddingMatrix(matrix, { batchIndex, axis, mask, weights, normalize })
normalizeVector(vector)
subtractVectors(a, b)
meanVectors(vectors, { weights, normalize })
combineDirections(deltas, { weights, normalizeEach, normalizeResult })
cosine(a, b)
rankCandidates(index, direction, { topK, minScore, excludeTerms, filter })
```

`poolEmbeddingMatrix` accepts vectors, 2D matrices, and `[batch,tokens,dimension]` tensors. Batch selection, matrix axis, token masks, and weights are explicit. Malformed, non-finite, and dimension-incompatible inputs produce errors; normalization and candidate ranking reject unusable zero vectors instead of silently flattening or skipping bad vectors. `rankCandidates` accepts arrays, `{records: [...]}`, or iterables of dictionary entries with `word` (or `text`) and `vector`; it supports exact normalized exclusions, a score threshold, and caller-supplied filters.

Run the math tests with Node.js:

```bash
node --test tests/workbench-math.test.js
```

## Existing Articulator features

The original prompt-writing **Studio** and browser-local **Idea shelf** remain available from the sidebar. A Workbench result can be set as a new A concept or sent to Studio as a sourced exploration note. The Workbench is implemented in `workbench.js`; the existing Studio behavior remains in `app.js`.

## Current limitations and possible next steps

- OEWN and Datamuse lookups require a network connection, and direct API calls remain subject to provider availability and browser CORS policy. If OEWN is blocked or unavailable, the interface reports that source failure and can still use available Datamuse results. A versioned local OEWN subset/index would improve availability and support offline lookups, but is not currently bundled.
- Candidate discovery follows selected lexical relations and Datamuse trails; it is not an exhaustive search over every English word. Scores are relative to the retrieved field.
- The MiniLM model provides one contextualized representation for each input string. Human review is still needed for polysemy, phrase interpretation, model bias, and surprising associations.
- Datamuse's announced API-key change should be addressed before January 1, 2027 if that trail is to remain enabled without interruption.
