(() => {
  const OEWN_API = 'https://en-word.net/api';
  const DATAMUSE_API = 'https://api.datamuse.com/words';
  const MAX_PAIRS = 4;
  const MAX_CANDIDATES = 280;
  const MAX_OEWN_TARGETS = 72;
  const MAX_VISIBLE = 12;
  const MAX_EXPANDED = 24;
  const VECTOR_EPSILON = 1e-9;

  const RELATION_GROUPS = {
    similar: 'synonyms',
    hypernym: 'taxonomy',
    hyponym: 'taxonomy',
    instance_hypernym: 'taxonomy',
    instance_hyponym: 'taxonomy',
    member_holonym: 'taxonomy',
    substance_holonym: 'taxonomy',
    part_holonym: 'taxonomy',
    member_meronym: 'taxonomy',
    substance_meronym: 'taxonomy',
    part_meronym: 'taxonomy',
    has_domain_topic: 'taxonomy',
    has_domain_region: 'taxonomy',
    has_domain_usage: 'taxonomy',
    domain_topic: 'taxonomy',
    domain_region: 'taxonomy',
    domain_usage: 'taxonomy',
    antonym: 'opposites',
    derivation: 'morphology',
    attribute: 'morphology',
    also: 'morphology',
    pertainym: 'morphology',
    participle: 'morphology',
    verb_group: 'morphology',
    cause: 'morphology',
    entailment: 'morphology',
  };

  const RELATION_LABELS = {
    similar: 'similar adjective',
    hypernym: 'broader concept',
    hyponym: 'more specific concept',
    instance_hypernym: 'broader class',
    instance_hyponym: 'specific instance',
    member_holonym: 'whole that includes this',
    substance_holonym: 'whole substance',
    part_holonym: 'whole that contains this',
    member_meronym: 'member of',
    substance_meronym: 'substance of',
    part_meronym: 'part of',
    has_domain_topic: 'domain topic',
    has_domain_region: 'regional domain',
    has_domain_usage: 'usage domain',
    domain_topic: 'topic domain',
    domain_region: 'regional domain',
    domain_usage: 'usage domain',
    antonym: 'antonym',
    derivation: 'derived word form',
    attribute: 'attribute relation',
    also: 'related sense',
    pertainym: 'pertains to',
    participle: 'participle form',
    verb_group: 'verb group',
    cause: 'causes',
    entailment: 'entails',
    member: 'same WordNet synset',
    meaninglike: 'meaning-like match',
    association: 'associated in text',
  };

  const LENS_COPY = {
    contrast: 'Ranks candidates by their alignment with the A − B direction. This is a relative vector score, not a truth score.',
    shared: 'Ranks candidates by their weaker similarity to each side of every pair. Strong shared-ground candidates need to fit both concepts.',
    blend: 'Ranks candidates near the normalized midpoint of the paired concepts. Useful for hybrids, bridges, and adjacent fields.',
  };

  const elements = typeof document === 'undefined' ? {} : {
    pairs: document.getElementById('wbPairs'),
    addPair: document.getElementById('wbAddPair'),
    context: document.getElementById('wbContext'),
    run: document.getElementById('wbRunButton'),
    ai: document.getElementById('wbAiButton'),
    status: document.getElementById('wbStatus'),
    results: document.getElementById('wbResults'),
    resultCount: document.getElementById('wbResultCount'),
    resultSubtitle: document.getElementById('wbResultSubtitle'),
    summary: document.getElementById('wbSummaryStrip'),
    senseNotes: document.getElementById('wbSenseNotes'),
    methodText: document.getElementById('wbMethodText'),
    lensExplanation: document.getElementById('wbLensExplanation'),
    fieldNote: document.getElementById('wbFieldNote'),
    copy: document.getElementById('wbCopyButton'),
    showMore: document.getElementById('wbShowMore'),
    resultsFooter: document.getElementById('wbResultsFooter'),
  };

  const state = {
    lens: 'contrast',
    filter: 'aligned',
    candidates: [],
    aiCandidates: [],
    senses: [],
    pairSnapshot: [],
    contextSnapshot: '',
    sourceSummary: null,
    resultSignature: null,
    resultLens: null,
    resultUsesTrails: true,
    visibleCount: MAX_VISIBLE,
    runId: 0,
    controller: null,
    busy: false,
  };

  const embeddingCache = new Map();

  function normalizeText(value) {
    return String(value ?? '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
      .toLowerCase().replace(/[’']/g, '').replace(/[^\p{L}\p{N}]+/gu, ' ').trim().replace(/\s+/g, ' ');
  }

  function embeddingKey(value) {
    return String(value ?? '').normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ');
  }

  function cleanDefinition(value) {
    return String(value ?? '').replace(/\s+/g, ' ').replace(/^\s+|\s+$/g, '').slice(0, 320);
  }

  function asNumericVector(value) {
    let vector;
    if (value?.data && value?.dims) {
      if (value.dims.length !== 1) throw new Error('Pool an embedding matrix explicitly before using it as a vector.');
      vector = poolEmbeddingMatrix(value, { normalize: false });
    } else if (ArrayBuffer.isView(value)) {
      vector = Float32Array.from(value, Number);
    } else if (Array.isArray(value) && value.length && (typeof value[0] === 'number' || typeof value[0] === 'bigint')) {
      vector = Float32Array.from(value, Number);
    } else {
      throw new Error('Expected a vector or an explicitly pooled embedding matrix.');
    }
    if (!vector.length || Array.from(vector).some((item) => !Number.isFinite(item))) throw new Error('Expected a non-empty vector of finite numbers.');
    return vector;
  }

  function matrixDescriptor(matrix) {
    if (matrix?.data && (Array.isArray(matrix.dims) || ArrayBuffer.isView(matrix.dims))) {
      const dims = Array.from(matrix.dims, Number);
      const data = Array.from(matrix.data, Number);
      const expected = dims.reduce((product, dimension) => product * dimension, 1);
      if (!dims.length || dims.some((dimension) => !Number.isInteger(dimension) || dimension <= 0) || data.length !== expected) {
        throw new Error('Embedding matrix shape does not match its data.');
      }
      return { dims, data };
    }
    if (ArrayBuffer.isView(matrix)) return { dims: [matrix.length], data: Array.from(matrix, Number) };
    if (!Array.isArray(matrix) || !matrix.length) throw new Error('Embedding matrix is empty.');
    if (typeof matrix[0] === 'number') return { dims: [matrix.length], data: matrix.map(Number) };
    if (ArrayBuffer.isView(matrix[0])) {
      const rows = matrix.map((row) => Array.from(row, Number));
      const width = rows[0]?.length || 0;
      if (!width || rows.some((row) => row.length !== width)) throw new Error('Embedding matrix rows have inconsistent dimensions.');
      return { dims: [rows.length, width], data: rows.flat() };
    }
    if (Array.isArray(matrix[0])) {
      const first = matrix[0];
      if (!first.length) throw new Error('Embedding matrix contains an empty row.');
      if (typeof first[0] === 'number') {
        const width = first.length;
        if (matrix.some((row) => !Array.isArray(row) || row.length !== width || row.some((item) => !Number.isFinite(Number(item))))) {
          throw new Error('Embedding matrix rows have inconsistent dimensions or non-finite values.');
        }
        return { dims: [matrix.length, width], data: matrix.flat().map(Number) };
      }
      const tokens = matrix.map((batch) => {
        if (!Array.isArray(batch) || !batch.length || !Array.isArray(batch[0])) throw new Error('Unsupported embedding tensor shape.');
        const width = batch[0].length;
        if (batch.some((row) => !Array.isArray(row) || row.length !== width)) throw new Error('Embedding tensor rows have inconsistent dimensions.');
        return batch.flat().map(Number);
      });
      const sequence = matrix[0].length;
      const width = matrix[0][0].length;
      if (matrix.some((batch) => batch.length !== sequence)) throw new Error('Embedding tensor batches have inconsistent sequence lengths.');
      return { dims: [matrix.length, sequence, width], data: tokens.flat() };
    }
    throw new Error('Unsupported embedding matrix format.');
  }

  /**
   * Convert [d], [rows,d], or [batch,tokens,d] embeddings to one vector.
   * Matrix axis, batch item, masks, and weights are explicit so independent
   * examples are never silently averaged together.
   */
  function poolEmbeddingMatrix(matrix, {
    batchIndex = 0,
    axis = 0,
    mask = null,
    weights = null,
    normalize = true,
  } = {}) {
    const { dims, data } = matrixDescriptor(matrix);
    const expectedSize = dims.reduce((product, dimension) => product * dimension, 1);
    if (!dims.length || dims.some((dimension) => !Number.isInteger(dimension) || dimension <= 0) || expectedSize !== data.length) throw new Error('Embedding matrix shape does not match its data.');
    if (data.some((item) => !Number.isFinite(item))) throw new Error('Embedding matrix contains non-finite values.');
    if (dims.length === 1) return normalize ? normalizeVector(data) : Float32Array.from(data);

    let rows;
    let width;
    if (dims.length === 2) {
      if (axis !== 0 && axis !== 1) throw new Error('For a 2D embedding matrix, axis must be 0 or 1.');
      const rowCount = dims[0];
      const columnCount = dims[1];
      if (axis === 0) {
        rows = Array.from({ length: rowCount }, (_, row) => data.slice(row * columnCount, (row + 1) * columnCount));
      } else {
        rows = Array.from({ length: columnCount }, (_, row) => Array.from({ length: rowCount }, (_, column) => data[column * columnCount + row]));
      }
      width = rows[0]?.length || 0;
    } else if (dims.length === 3) {
      if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= dims[0]) throw new Error('batchIndex is outside the embedding tensor.');
      const sequenceLength = dims[1];
      width = dims[2];
      const offset = batchIndex * sequenceLength * width;
      rows = Array.from({ length: sequenceLength }, (_, row) => data.slice(offset + row * width, offset + (row + 1) * width));
    } else {
      throw new Error('Only vectors, 2D matrices, and 3D [batch,tokens,dimension] tensors are supported.');
    }

    if (!rows.length || !width) throw new Error('Cannot pool an empty embedding matrix.');
    const rowWeights = rows.map((_, index) => {
      if (mask && !mask[index]) return 0;
      const weight = weights ? Number(weights[index] ?? 0) : 1;
      return Number.isFinite(weight) && weight > 0 ? weight : 0;
    });
    const totalWeight = rowWeights.reduce((sum, weight) => sum + weight, 0);
    if (!totalWeight) throw new Error('The embedding matrix has no unmasked rows to pool.');
    const vector = new Float32Array(width);
    rows.forEach((row, rowIndex) => {
      const weight = rowWeights[rowIndex] / totalWeight;
      for (let dimension = 0; dimension < width; dimension += 1) {
        vector[dimension] += Number(row[dimension]) * weight;
      }
    });
    return normalize ? normalizeVector(vector) : vector;
  }

  function normalizeVector(value) {
    const vector = ArrayBuffer.isView(value) ? Array.from(value, Number) : Array.from(value || [], Number);
    if (!vector.length || vector.some((item) => !Number.isFinite(item))) throw new Error('Cannot normalize an empty or non-finite vector.');
    const norm = Math.sqrt(vector.reduce((sum, item) => sum + item * item, 0));
    if (norm < VECTOR_EPSILON) throw new Error('Cannot normalize a zero-length vector.');
    return Float32Array.from(vector, (item) => item / norm);
  }

  function subtractVectors(aValue, bValue) {
    const a = asNumericVector(aValue);
    const b = asNumericVector(bValue);
    if (a.length !== b.length) throw new Error('Cannot subtract vectors with different dimensions.');
    return Float32Array.from(a, (item, index) => item - b[index]);
  }

  function meanVectors(vectors, { weights = null, normalize = false } = {}) {
    if (!Array.isArray(vectors) || !vectors.length) throw new Error('At least one vector is required to calculate a mean.');
    const rows = vectors.map(asNumericVector);
    const dimension = rows[0].length;
    if (rows.some((row) => row.length !== dimension)) throw new Error('Cannot average vectors with different dimensions.');
    const rowWeights = rows.map((_, index) => {
      const value = weights ? Number(weights[index] ?? 0) : 1;
      return Number.isFinite(value) && value > 0 ? value : 0;
    });
    const total = rowWeights.reduce((sum, weight) => sum + weight, 0);
    if (!total) throw new Error('Vector weights must sum to a positive value.');
    const result = new Float32Array(dimension);
    rows.forEach((row, rowIndex) => {
      const weight = rowWeights[rowIndex] / total;
      for (let dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex += 1) {
        result[dimensionIndex] += row[dimensionIndex] * weight;
      }
    });
    return normalize ? normalizeVector(result) : result;
  }

  function combineDirections(deltas, { weights = null, normalizeEach = true, normalizeResult = true } = {}) {
    const usable = [];
    const usableWeights = [];
    (deltas || []).forEach((delta, index) => {
      const vector = asNumericVector(delta);
      if (Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) < VECTOR_EPSILON) return;
      usable.push(normalizeEach ? normalizeVector(vector) : vector);
      usableWeights.push(weights ? Number(weights[index] ?? 0) : 1);
    });
    if (!usable.length) throw new Error('The comparisons do not form a usable direction.');
    return meanVectors(usable, { weights: usableWeights, normalize: normalizeResult });
  }

  function cosine(aValue, bValue) {
    const a = asNumericVector(aValue);
    const b = asNumericVector(bValue);
    if (a.length !== b.length) throw new Error('Cannot compare vectors with different dimensions.');
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let index = 0; index < a.length; index += 1) {
      dot += a[index] * b[index];
      normA += a[index] * a[index];
      normB += b[index] * b[index];
    }
    if (normA < VECTOR_EPSILON || normB < VECTOR_EPSILON) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  function rankCandidates(index, direction, { topK = MAX_EXPANDED, minScore = -1, excludeTerms = [], filter = null } = {}) {
    const entries = Array.isArray(index) ? index : Array.isArray(index?.records) ? index.records : index?.[Symbol.iterator] ? [...index] : [];
    const directionVector = asNumericVector(direction);
    if (Math.sqrt(directionVector.reduce((sum, value) => sum + value * value, 0)) < VECTOR_EPSILON) throw new Error('The ranking direction cannot be a zero vector.');
    if (!Number.isFinite(minScore)) throw new Error('minScore must be a finite number.');
    const excluded = new Set(Array.from(excludeTerms || [], normalizeText).filter(Boolean));
    const scored = [];
    entries.forEach((entry, indexInCorpus) => {
      const word = String(entry?.word ?? entry?.text ?? '').trim();
      if (!word || !entry?.vector || excluded.has(normalizeText(word))) return;
      if (filter && !filter(entry)) return;
      const vector = asNumericVector(entry.vector);
      if (vector.length !== directionVector.length) throw new Error(`Candidate “${word}” has a vector dimension that does not match the direction.`);
      if (Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) < VECTOR_EPSILON) throw new Error(`Candidate “${word}” has a zero vector.`);
      const score = cosine(vector, directionVector);
      if (!Number.isFinite(score) || score < minScore) return;
      scored.push({ ...entry, word, score, indexInCorpus });
    });
    scored.sort((a, b) => b.score - a.score || normalizeText(a.word).localeCompare(normalizeText(b.word)) || a.indexInCorpus - b.indexInCorpus);
    const limit = Number.isFinite(topK) ? Math.max(0, Math.floor(topK)) : scored.length;
    return scored.slice(0, limit);
  }

  function setStatus(message, type = '') {
    if (!elements.status) return;
    elements.status.textContent = message;
    elements.status.classList.toggle('is-error', type === 'error');
    elements.status.classList.toggle('is-success', type === 'success');
    elements.status.classList.toggle('is-loading', type === 'loading');
  }

  function currentPairs() {
    return [...elements.pairs.querySelectorAll('.wb-pair')].map((row) => ({
      toward: row.querySelector('[data-side="toward"]')?.value.trim() || '',
      away: row.querySelector('[data-side="away"]')?.value.trim() || '',
    }));
  }

  function updatePairControls() {
    const rows = [...elements.pairs.querySelectorAll('.wb-pair')];
    rows.forEach((row, index) => {
      row.dataset.pairIndex = String(index);
      row.querySelector('.wb-pair-number').textContent = `COMPARISON ${String(index + 1).padStart(2, '0')}`;
      const remove = row.querySelector('[data-remove-pair]');
      remove.disabled = rows.length === 1;
      remove.setAttribute('aria-label', `Remove comparison ${index + 1}`);
    });
    elements.addPair.disabled = rows.length >= MAX_PAIRS;
    elements.addPair.innerHTML = rows.length >= MAX_PAIRS
      ? '<span aria-hidden="true">＋</span> Maximum of four comparisons'
      : '<span aria-hidden="true">＋</span> Add a parallel comparison';
  }

  function pairTemplate(index, toward = '', away = '') {
    return `<div class="wb-pair" data-pair-index="${index}">
      <div class="wb-pair-top"><span class="wb-pair-number">COMPARISON ${String(index + 1).padStart(2, '0')}</span><button class="wb-remove-pair" type="button" data-remove-pair aria-label="Remove comparison ${index + 1}">×</button></div>
      <div class="wb-pair-fields">
        <label class="wb-concept-field"><span>TOWARD <b>A</b></span><input class="wb-concept" data-side="toward" type="text" maxlength="100" value="${escapeHTML(toward)}" placeholder="e.g. orthodox" autocomplete="off" /></label>
        <button class="wb-swap" type="button" data-swap-pair aria-label="Swap toward and away concepts" title="Swap direction">⇄</button>
        <label class="wb-concept-field"><span>AWAY FROM <b>B</b></span><input class="wb-concept" data-side="away" type="text" maxlength="100" value="${escapeHTML(away)}" placeholder="e.g. religion" autocomplete="off" /></label>
      </div>
    </div>`;
  }

  function markStale() {
    if (!state.resultSignature) return false;
    const wasStale = elements.results.classList.contains('is-stale');
    const stale = getCurrentSignature() !== state.resultSignature;
    elements.results.classList.toggle('is-stale', stale);
    if (stale) {
      setStatus(state.busy ? 'Inputs changed during this run · results will be marked stale.' : 'Inputs changed · run Explore again to refresh these results.');
      elements.resultSubtitle.textContent = 'Showing results for the previous settings · refresh to recalculate.';
    } else if (wasStale) {
      elements.resultSubtitle.textContent = 'Displayed results match the current settings.';
      setStatus('Inputs match the displayed result.', 'success');
    }
    return stale;
  }

  function readTrailOptions() {
    return new Set([...document.querySelectorAll('[data-wb-trail]:checked')].map((input) => input.dataset.wbTrail));
  }

  function getUniqueInputTerms(pairs) {
    const seen = new Set();
    const terms = [];
    pairs.flatMap((pair) => [pair.toward, pair.away]).forEach((text) => {
      const key = normalizeText(text);
      if (!key || seen.has(key)) return;
      seen.add(key);
      terms.push(text);
    });
    return terms.slice(0, MAX_PAIRS * 2);
  }

  function isCurrentRun(runId) {
    return runId === state.runId;
  }

  async function fetchJson(url, { signal = null, timeout = 8500 } = {}) {
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener('abort', abort, { once: true });
    }
    const timer = window.setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {
        method: 'GET',
        mode: 'cors',
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      if (!response.ok) throw new Error(`Source returned ${response.status}`);
      return await response.json();
    } finally {
      window.clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
    }
  }

  async function mapLimit(items, limit, mapper) {
    const results = new Array(items.length);
    let nextIndex = 0;
    const workerCount = Math.min(Math.max(1, limit), items.length);
    await Promise.all(Array.from({ length: workerCount }, async () => {
      while (nextIndex < items.length) {
        const index = nextIndex++;
        try { results[index] = { status: 'fulfilled', value: await mapper(items[index], index) }; }
        catch (reason) { results[index] = { status: 'rejected', reason }; }
      }
    }));
    return results;
  }

  function makeCandidateAccumulator() {
    const byWord = new Map();
    return {
      add(candidate) {
        const word = String(candidate?.word || '').trim();
        const key = normalizeText(word);
        if (!word || !key || word.length > 70 || /[<>\n\r]/.test(word)) return;
        let existing = byWord.get(key);
        if (!existing) {
          existing = {
            id: key,
            word,
            vector: null,
            definitions: [],
            examples: [],
            evidence: [],
            partsOfSpeech: [],
            aiRationale: '',
          };
          byWord.set(key, existing);
        }
        const addDefinition = (text, source, synsetId = '', pos = '') => {
          const definition = cleanDefinition(text);
          const sourceLabel = source || candidate.provider || 'Lexical source';
          const senseId = synsetId || '';
          if (!definition || existing.definitions.some((saved) => normalizeText(saved.text) === normalizeText(definition) && saved.synsetId === senseId && saved.source === sourceLabel)) return;
          existing.definitions.push({ text: definition, source: sourceLabel, synsetId: senseId, pos: pos || '' });
        };
        if (candidate.definition) addDefinition(candidate.definition, candidate.definitionSource, candidate.synsetId, candidate.pos);
        (candidate.definitions || []).forEach((item) => addDefinition(
          typeof item === 'string' ? item : item.text,
          typeof item === 'string' ? null : item.source,
          typeof item === 'string' ? candidate.synsetId : item.synsetId || candidate.synsetId,
          typeof item === 'string' ? candidate.pos : item.pos || candidate.pos,
        ));
        (candidate.examples || []).forEach((example) => {
          const cleanExample = cleanDefinition(example);
          if (cleanExample && !existing.examples.includes(cleanExample)) existing.examples.push(cleanExample);
        });
        (candidate.evidence || []).forEach((item) => {
          const signature = `${item.provider}|${item.relation}|${normalizeText(item.sourceTerm)}|${item.synsetId || ''}`;
          if (!existing.evidence.some((saved) => saved.signature === signature)) existing.evidence.push({ ...item, signature });
        });
        if (candidate.provider || candidate.relation) {
          const evidence = {
            provider: candidate.provider || 'Word source',
            relation: candidate.relation || 'related entry',
            sourceTerm: candidate.sourceTerm || '',
            synsetId: candidate.synsetId || '',
            group: candidate.group || '',
            url: candidate.url || '',
          };
          const signature = `${evidence.provider}|${evidence.relation}|${normalizeText(evidence.sourceTerm)}|${evidence.synsetId}`;
          if (!existing.evidence.some((saved) => saved.signature === signature)) existing.evidence.push({ ...evidence, signature });
        }
        (candidate.partsOfSpeech || []).forEach((pos) => {
          if (pos && !existing.partsOfSpeech.includes(pos)) existing.partsOfSpeech.push(pos);
        });
        if (candidate.pos && !existing.partsOfSpeech.includes(candidate.pos)) existing.partsOfSpeech.push(candidate.pos);
        if (candidate.aiRationale && !existing.aiRationale) existing.aiRationale = cleanDefinition(candidate.aiRationale);
      },
      values() { return [...byWord.values()]; },
    };
  }

  function relationGroup(field) {
    return RELATION_GROUPS[field] || null;
  }

  function relationLabel(field) {
    return RELATION_LABELS[field] || field.replace(/_/g, ' ');
  }

  function addSynsetCandidates(accumulator, synset, { sourceTerm, relation = 'same WordNet synset', group = 'synonyms' } = {}) {
    if (!synset || !Array.isArray(synset.members)) return;
    const definition = Array.isArray(synset.definition) ? synset.definition.join('; ') : synset.definition;
    const examples = Array.isArray(synset.example) ? synset.example : synset.example ? [synset.example] : [];
    for (const member of synset.members) {
      const word = member?.lemma;
      if (!word) continue;
      accumulator.add({
        word,
        provider: 'Open English WordNet',
        relation,
        sourceTerm,
        group,
        synsetId: synset.id || '',
        definition,
        definitionSource: 'OEWN definition',
        examples,
        pos: member.poskey || synset.partOfSpeech || '',
        url: `https://en-word.net/view/lemma/${encodeURIComponent(String(word).replace(/\s+/g, '_'))}`,
      });
    }
  }

  function relationTargetIds(synset, selectedTrails) {
    const targets = [];
    for (const [field, value] of Object.entries(synset || {})) {
      const group = relationGroup(field);
      if (!group || !selectedTrails.has(group) || !Array.isArray(value)) continue;
      for (const item of value) {
        const id = typeof item === 'string' ? item : item?.target_synset;
        if (typeof id !== 'string' || !/^\d{8}-[nvars]$/.test(id)) continue;
        targets.push({ id, field, group });
      }
    }
    return targets;
  }

  async function fetchOewnCandidates(terms, selectedTrails, signal) {
    const accumulator = makeCandidateAccumulator();
    const senseNotes = [];
    const rootResults = await mapLimit(terms, 6, async (term) => {
      const lemma = term.trim().replace(/\s+/g, '_');
      const url = `${OEWN_API}/lemma/${encodeURIComponent(lemma)}`;
      const data = await fetchJson(url, { signal });
      return { term, synsets: Array.isArray(data) ? data : [] };
    });
    let successfulLookups = 0;
    const targetMap = new Map();

    for (const result of rootResults) {
      if (result.status !== 'fulfilled') continue;
      successfulLookups += 1;
      const { term, synsets } = result.value;
      synsets.slice(0, 10).forEach((synset) => {
        const definition = Array.isArray(synset.definition) ? synset.definition.join('; ') : synset.definition;
        const examples = Array.isArray(synset.example) ? synset.example : synset.example ? [synset.example] : [];
        senseNotes.push({ term, definition: cleanDefinition(definition), pos: synset.partOfSpeech || '', synsetId: synset.id || '', example: examples[0] || '' });
        if (selectedTrails.has('synonyms')) addSynsetCandidates(accumulator, synset, { sourceTerm: term, relation: 'same-sense member', group: 'synonyms' });
        relationTargetIds(synset, selectedTrails).forEach((target) => {
          if (!targetMap.has(target.id)) {
            if (targetMap.size >= MAX_OEWN_TARGETS) return;
            targetMap.set(target.id, { id: target.id, links: [] });
          }
          const entry = targetMap.get(target.id);
          const signature = `${target.field}|${target.group}|${normalizeText(term)}`;
          if (!entry.links.some((link) => link.signature === signature)) entry.links.push({ ...target, sourceTerm: term, signature });
        });
      });
    }

    const targetResults = await mapLimit([...targetMap.values()], 8, async (target) => {
      const synset = await fetchJson(`${OEWN_API}/synset/${encodeURIComponent(target.id)}`, { signal });
      return { synset, target };
    });
    let relationSynsets = 0;
    targetResults.forEach((result) => {
      if (result.status !== 'fulfilled') return;
      relationSynsets += 1;
      const { synset, target } = result.value;
      (target.links || []).forEach((link) => addSynsetCandidates(accumulator, synset, {
        sourceTerm: link.sourceTerm,
        relation: relationLabel(link.field),
        group: link.group,
      }));
    });

    return {
      candidates: accumulator.values(),
      senseNotes: senseNotes.slice(0, 24),
      successfulLookups,
      inputCount: terms.length,
      relationSynsets,
      failedLookups: rootResults.filter((result) => result.status === 'rejected').length,
      failedTargets: targetResults.filter((result) => result.status === 'rejected').length,
    };
  }

  function parseDatamuseDefinitions(entry) {
    const definitions = [];
    const parts = [];
    (Array.isArray(entry.defs) ? entry.defs : []).forEach((value) => {
      const line = String(value);
      const split = line.indexOf('\t');
      const pos = split >= 0 ? line.slice(0, split).trim() : '';
      const definition = cleanDefinition(split >= 0 ? line.slice(split + 1) : line);
      if (definition) definitions.push({ text: definition, source: 'Wiktionary / WordNet via Datamuse', pos });
    });
    (Array.isArray(entry.tags) ? entry.tags : []).forEach((tag) => {
      if (['n', 'v', 'adj', 'adv'].includes(tag)) parts.push(tag === 'n' ? 'noun' : tag === 'v' ? 'verb' : tag === 'adj' ? 'adjective' : 'adverb');
    });
    return { definitions, parts };
  }

  async function fetchDatamuseCandidates(terms, selectedTrails, signal) {
    const queries = [];
    if (selectedTrails.has('meaninglike')) terms.forEach((term) => queries.push({ term, param: 'ml', relation: 'meaning-like match', group: 'meaninglike' }));
    if (selectedTrails.has('association')) terms.forEach((term) => queries.push({ term, param: 'rel_trg', relation: 'associated in text', group: 'association' }));
    const responses = await mapLimit(queries, 8, async (query) => {
      const url = new URL(DATAMUSE_API);
      url.searchParams.set(query.param, query.term);
      url.searchParams.set('max', '18');
      url.searchParams.set('md', 'dp');
      const data = await fetchJson(url.toString(), { signal, timeout: 6500 });
      return { query, entries: Array.isArray(data) ? data : [] };
    });
    const accumulator = makeCandidateAccumulator();
    let successfulQueries = 0;
    responses.forEach((response) => {
      if (response.status !== 'fulfilled') return;
      successfulQueries += 1;
      const { query, entries } = response.value;
      entries.forEach((entry) => {
        if (!entry?.word) return;
        const { definitions, parts } = parseDatamuseDefinitions(entry);
        accumulator.add({
          word: entry.word,
          provider: 'Datamuse',
          relation: query.relation,
          sourceTerm: query.term,
          group: query.group,
          definitions,
          partsOfSpeech: parts,
          url: 'https://www.datamuse.com/api/',
        });
      });
    });
    return { candidates: accumulator.values(), successfulQueries, queryCount: queries.length, failedQueries: responses.filter((response) => response.status === 'rejected').length };
  }

  function mergeCandidateLists(...lists) {
    const accumulator = makeCandidateAccumulator();
    lists.flat().forEach((item) => accumulator.add(item));
    return accumulator.values();
  }

  function limitCandidateField(candidates, limit) {
    const providerBuckets = [
      candidates.filter((candidate) => candidate.evidence.some((item) => item.provider === 'Open English WordNet')),
      candidates.filter((candidate) => candidate.evidence.some((item) => item.provider === 'Datamuse')),
    ];
    const cursors = [0, 0];
    const selected = [];
    const seen = new Set();
    while (selected.length < limit) {
      let added = false;
      providerBuckets.forEach((bucket, bucketIndex) => {
        if (selected.length >= limit) return;
        while (cursors[bucketIndex] < bucket.length && seen.has(bucket[cursors[bucketIndex]].id)) cursors[bucketIndex] += 1;
        const candidate = bucket[cursors[bucketIndex]++];
        if (!candidate) return;
        seen.add(candidate.id);
        selected.push(candidate);
        added = true;
      });
      if (!added) break;
    }
    candidates.forEach((candidate) => {
      if (selected.length < limit && !seen.has(candidate.id)) {
        selected.push(candidate);
        seen.add(candidate.id);
      }
    });
    return selected;
  }

  function getLensVectors(pairs, inputVectorMap) {
    return pairs.map((pair) => ({
      toward: inputVectorMap.get(normalizeText(pair.toward)),
      away: inputVectorMap.get(normalizeText(pair.away)),
    }));
  }

  function makeDirection(pairVectors, lens) {
    if (lens === 'shared') return null;
    if (lens === 'blend') {
      return meanVectors(pairVectors.flatMap((pair) => [pair.toward, pair.away]), { normalize: true });
    }
    const deltas = pairVectors.map((pair) => subtractVectors(pair.toward, pair.away));
    return combineDirections(deltas, { normalizeEach: true, normalizeResult: true });
  }

  function scoreForLens(candidateVector, pairVectors, lens, direction) {
    if (lens === 'shared') {
      const pairScores = pairVectors.map((pair) => Math.min(cosine(candidateVector, pair.toward), cosine(candidateVector, pair.away)));
      return pairScores.reduce((sum, score) => sum + score, 0) / pairScores.length;
    }
    return cosine(candidateVector, direction);
  }

  function candidateHasGroup(candidate, group) {
    return candidate.evidence.some((item) => item.group === group);
  }

  function rankWorkbenchCandidates(candidates, pairVectors, pairs, lens, vectorMap, topK = MAX_EXPANDED) {
    const excluded = pairs.flatMap((pair) => [pair.toward, pair.away]);
    const direction = lens === 'shared' ? null : makeDirection(pairVectors, lens);
    const entries = candidates.map((candidate) => ({ ...candidate, vector: candidate.vector || vectorMap.get(embeddingKey(candidate.word)) })).filter((item) => item.vector);
    const scored = entries.map((candidate) => ({
      ...candidate,
      score: scoreForLens(candidate.vector, pairVectors, lens, direction),
    })).filter((candidate) => Number.isFinite(candidate.score) && !excluded.map(normalizeText).includes(normalizeText(candidate.word)));
    const unique = new Map();
    scored.forEach((candidate) => {
      const key = normalizeText(candidate.word);
      if (!key) return;
      const current = unique.get(key);
      if (!current || candidate.score > current.score) unique.set(key, candidate);
      else {
        current.evidence.push(...candidate.evidence.filter((evidence) => !current.evidence.some((item) => item.signature === evidence.signature)));
        current.definitions.push(...candidate.definitions.filter((definition) => !current.definitions.some((item) => normalizeText(item.text) === normalizeText(definition.text))));
      }
    });
    const ranked = rankCandidates([...unique.values()], lens === 'shared' ? pairVectors[0].toward : direction, {
      topK: unique.size,
      minScore: -1,
      excludeTerms: excluded,
      filter: (entry) => Number.isFinite(entry.score),
    });
    // The shared-ground lens uses a custom score (minimum similarity to both poles),
    // so preserve that score and use the same stable sorting contract.
    const final = lens === 'shared'
      ? [...unique.values()].sort((a, b) => b.score - a.score || normalizeText(a.word).localeCompare(normalizeText(b.word)))
      : ranked.map((item) => ({ ...item, score: unique.get(normalizeText(item.word))?.score ?? item.score }));
    return { results: final.slice(0, topK), direction };
  }

  function posLabel(value) {
    const pos = String(value || '').toLowerCase();
    if (pos === 'n') return 'noun';
    if (pos === 'v') return 'verb';
    if (pos === 'a' || pos === 's') return 'adjective';
    if (pos === 'r' || pos === 'adv') return 'adverb';
    return value || '';
  }

  function getSelectedEvidence(candidate) {
    const unique = [];
    candidate.evidence.forEach((item) => {
      const label = `${item.provider} · ${item.relation}${item.sourceTerm ? ` of “${item.sourceTerm}”` : ''}`;
      if (!unique.includes(label)) unique.push(label);
    });
    return unique;
  }

  function evidenceClass(candidate) {
    if (candidate.aiRationale) return 'ai';
    if (candidate.evidence.some((item) => item.provider === 'Open English WordNet')) return 'oewn';
    return 'datamuse';
  }

  function renderCandidateCard(candidate, index) {
    const definitions = candidate.definitions.slice(0, 2);
    const allEvidence = getSelectedEvidence(candidate);
    const evidence = allEvidence.slice(0, 3);
    const resultLens = state.resultLens || state.lens;
    const scoreName = resultLens === 'shared' ? 'shared alignment' : resultLens === 'blend' ? 'blend alignment' : 'direction alignment';
    const definitionMarkup = definitions.length
      ? definitions.map((definition) => `<p class="wb-definition"><span class="wb-definition-label">${escapeHTML(definition.source)}${definition.pos ? ` · ${escapeHTML(posLabel(definition.pos))}` : ''}</span>${escapeHTML(definition.text)}</p>`).join('')
      : candidate.aiRationale
        ? `<p class="wb-definition wb-ai-rationale"><span class="wb-definition-label">AI WILDCARD · UNVERIFIED</span>${escapeHTML(candidate.aiRationale)}</p>`
        : '<p class="wb-definition wb-no-definition">No definition was returned for this entry.</p>';
    const exampleMarkup = candidate.examples.slice(0, 1).map((example) => `<p class="wb-example">“${escapeHTML(example)}”</p>`).join('');
    const evidenceMarkup = evidence.length
      ? evidence.map((item) => `<span class="wb-evidence-tag">${escapeHTML(item)}</span>`).join('')
      : '<span class="wb-evidence-tag">AI-generated suggestion · not verified</span>';
    const sources = candidate.evidence.map((item) => item.provider);
    const trailBoundary = sources.includes('Datamuse') && sources.includes('Open English WordNet')
      ? 'Open English WordNet trails connect lexical senses; Datamuse matches and text associations are not WordNet relations.'
      : sources.includes('Datamuse')
        ? 'Datamuse meaning-like matches and text associations are not verified WordNet relations.'
        : 'Open English WordNet relations connect lexical senses; they are not real-world evidence.';
    const oewnLink = candidate.evidence.find((item) => item.provider === 'Open English WordNet')?.url
      || `https://en-word.net/view/lemma/${encodeURIComponent(candidate.word.replace(/\s+/g, '_'))}`;
    const externalLink = candidate.aiRationale ? 'https://developer.puter.com/' : sources.includes('Open English WordNet') ? oewnLink : 'https://www.datamuse.com/api/';
    const sourceActionLabel = candidate.aiRationale ? 'AI provider ↗' : 'Source ↗';
    const score = Number.isFinite(candidate.score) ? `${candidate.score >= 0 ? '+' : ''}${candidate.score.toFixed(3)}` : '—';
    const pos = candidate.partsOfSpeech.slice(0, 2).map(posLabel).filter(Boolean).join(' · ');
    return `<article class="wb-result-card wb-source-${evidenceClass(candidate)}" style="--card-index:${index}">
      <div class="wb-result-card-top"><span class="wb-rank">${String(index + 1).padStart(2, '0')}</span><span class="wb-result-source-label">${candidate.aiRationale ? 'AI WILDCARD · UNVERIFIED' : evidenceClass(candidate) === 'oewn' ? 'OPEN ENGLISH WORDNET' : 'DATAMUSE · ASSOCIATION'}</span><span class="wb-score" title="${escapeHTML(scoreName)}; not a probability">${score}<small>ALIGN</small></span></div>
      <h3>${escapeHTML(candidate.word)}${pos ? `<small>${escapeHTML(pos)}</small>` : ''}</h3>
      <div class="wb-definition-stack">${definitionMarkup}${exampleMarkup}</div>
      <div class="wb-evidence-list">${evidenceMarkup}</div>
      <details class="wb-result-details"><summary>Why it appeared</summary><div><p>${escapeHTML(candidate.aiRationale ? 'This term was generated by the optional AI action. It is not a dictionary fact; its vector score only describes model alignment.' : `${trailBoundary} The lexical trail and embedding score are separate signals; the score ranks this text embedding against the selected lens.`)}</p>${!candidate.aiRationale && allEvidence.length ? `<p>Recorded lexical trail: ${escapeHTML(allEvidence.join('; '))}</p>` : ''}${candidate.evidence.some((item) => item.synsetId) ? `<small>Synset ID${candidate.evidence.filter((item) => item.synsetId).length > 1 ? 's' : ''}: ${escapeHTML([...new Set(candidate.evidence.map((item) => item.synsetId).filter(Boolean))].join(', '))}</small>` : ''}</div></details>
      <div class="wb-result-actions"><button type="button" data-set-toward="${escapeHTML(candidate.word)}">Set as A</button><button type="button" data-send-note="${escapeHTML(candidate.id)}">Draft a note</button><a href="${escapeHTML(externalLink)}" target="_blank" rel="noreferrer">${sourceActionLabel}</a></div>
    </article>`;
  }

  function visibleCandidates() {
    if (state.filter === 'ai') return state.aiCandidates;
    if (state.filter === 'lateral') return state.candidates.filter((candidate) => candidateHasGroup(candidate, 'association') || candidateHasGroup(candidate, 'meaninglike'));
    if (state.filter === 'aligned') return state.candidates.filter((candidate) => candidate.score > 0);
    return state.candidates;
  }

  function renderResults() {
    const candidates = visibleCandidates();
    const filtered = [...candidates].sort((a, b) => b.score - a.score || normalizeText(a.word).localeCompare(normalizeText(b.word)));
    const shown = filtered.slice(0, state.visibleCount);
    elements.resultCount.textContent = `${filtered.length} ${filtered.length === 1 ? 'candidate' : 'candidates'}`;
    elements.copy.disabled = !filtered.length;
    elements.resultsFooter.hidden = filtered.length <= MAX_VISIBLE;
    elements.showMore.textContent = state.visibleCount >= filtered.length ? 'Show fewer' : `Show 12 more · ${filtered.length - state.visibleCount} remaining`;
    if (!shown.length) {
      const emptyText = state.filter === 'ai'
        ? 'No AI wildcards yet. Use the opt-in button to ask for speculative, offbeat candidates.'
        : state.filter === 'lateral'
          ? 'No lateral associations in this run. Enable “Meaning-like” or “Text associations” and explore again.'
          : state.filter === 'aligned'
            ? 'No retrieved term has positive alignment with this lens. Check All candidates to see the rest of the ranked field.'
            : 'No candidates were retrieved. Try a broader phrase, turn on another trail, or check the source connections.';
      elements.results.innerHTML = `<div class="wb-empty-state compact-empty"><span class="wb-empty-orbit" aria-hidden="true">∅</span><h3>Nothing in this field yet</h3><p>${escapeHTML(emptyText)}</p></div>`;
      return;
    }
    elements.results.innerHTML = shown.map(renderCandidateCard).join('');
  }

  function updateTabs() {
    document.querySelectorAll('[data-wb-filter]').forEach((button) => {
      const filter = button.dataset.wbFilter;
      const selected = filter === state.filter;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-selected', String(selected));
      if (filter === 'ai') button.hidden = !state.aiCandidates.length;
    });
  }

  function renderSenseNotes(senses, terms) {
    if (!senses.length) {
      elements.senseNotes.hidden = true;
      elements.senseNotes.innerHTML = '';
      return;
    }
    const unique = new Map();
    senses.forEach((sense) => {
      const key = `${normalizeText(sense.term)}|${sense.synsetId}`;
      if (!unique.has(key)) unique.set(key, sense);
    });
    const values = [...unique.values()].slice(0, 8);
    elements.senseNotes.hidden = false;
    elements.senseNotes.innerHTML = `<div class="wb-sense-note-heading"><strong>Input senses found in OEWN</strong><span>Raw input embeddings are used; choose a context hint if a sense does not fit.</span></div><div class="wb-sense-note-list">${values.map((sense) => {
      const sourceUrl = `https://en-word.net/view/lemma/${encodeURIComponent(String(sense.term).replace(/\s+/g, '_'))}`;
      return `<article><span>${escapeHTML(sense.term)}${sense.pos ? ` · ${escapeHTML(posLabel(sense.pos))}` : ''}</span><p>${escapeHTML(sense.definition || 'No gloss for this sense.')}</p>${sense.example ? `<small>Example: “${escapeHTML(sense.example)}”</small>` : ''}<a class="wb-sense-source" href="${escapeHTML(sourceUrl)}" target="_blank" rel="noreferrer">OEWN synset ${escapeHTML(sense.synsetId || '')} ↗</a></article>`;
    }).join('')}</div>`;
  }

  function updateSummary(sourceSummary, candidates) {
    const pieces = [];
    if (sourceSummary.oewn) pieces.push(`OEWN · ${sourceSummary.oewn.senseCount} senses · ${sourceSummary.oewn.relationSynsets} related synsets`);
    if (sourceSummary.datamuse) pieces.push(`Datamuse · ${sourceSummary.datamuse.candidates} candidates`);
    if (sourceSummary.oewn?.failed || sourceSummary.datamuse?.failed) pieces.push('some source requests failed');
    pieces.push(`${candidates.length} unique terms · ranked on-device`);
    elements.summary.hidden = false;
    elements.summary.innerHTML = pieces.map((piece, index) => `<span>${index ? '<i aria-hidden="true">·</i>' : ''}${escapeHTML(piece)}</span>`).join('');
  }

  function getCurrentSignature(pairs = currentPairs(), context = elements.context.value, lens = state.lens, trails = readTrailOptions(), includeTrails = state.resultUsesTrails) {
    return JSON.stringify({ pairs: pairs.map((pair) => [embeddingKey(pair.toward), embeddingKey(pair.away)]), context: embeddingKey(context), lens, trails: includeTrails ? [...trails].sort() : null });
  }

  function updateMethodText(pairs = state.pairSnapshot, lens = state.lens) {
    if (!pairs.length) return;
    const sample = pairs.map((pair) => `(${pair.toward} − ${pair.away})`).join(' + ');
    const lensDescription = lens === 'shared'
      ? 'Shared ground ranks by the average, across pairs, of the lower of the two cosine similarities. It is not a direction vector.'
      : lens === 'blend'
        ? 'Blend normalizes the mean of the A and B embeddings, then ranks candidates by cosine similarity to that midpoint.'
        : 'Contrast normalizes each A − B difference, averages the directions, and ranks by cosine alignment. With one pair, that is exactly the normalized A − B direction.';
    elements.methodText.textContent = `${lensDescription} Current input: ${sample}. Scores are model alignment, not probabilities, factual claims, or causal evidence.`;
  }

  function setBusy(isBusy) {
    state.busy = isBusy;
    elements.run.disabled = isBusy;
    elements.ai.disabled = isBusy;
    elements.showMore.disabled = isBusy;
    document.querySelectorAll('[data-wb-filter]').forEach((button) => { button.disabled = isBusy; });
    const stale = !isBusy && state.resultSignature && getCurrentSignature() !== state.resultSignature;
    const label = elements.run.querySelector('span:nth-child(2)');
    if (label) label.textContent = isBusy ? 'Following the trails…' : stale ? 'Refresh exploration' : 'Explore this direction';
    elements.run.classList.toggle('is-loading', isBusy);
  }

  function trailSelectionValid(trails) {
    return trails.size > 0;
  }

  async function runExploration() {
    const pairs = currentPairs();
    const invalid = pairs.findIndex((pair) => !pair.toward || !pair.away);
    if (invalid >= 0) {
      const row = elements.pairs.querySelectorAll('.wb-pair')[invalid];
      const missing = !pairs[invalid].toward ? 'toward' : 'away';
      row?.querySelector(`[data-side="${missing}"]`)?.focus();
      setStatus(`Comparison ${invalid + 1} needs both a Toward and an Away from concept.`, 'error');
      return;
    }
    const trails = readTrailOptions();
    if (!trailSelectionValid(trails)) {
      setStatus('Choose at least one lexical trail before exploring.', 'error');
      return;
    }

    const runId = ++state.runId;
    state.controller?.abort();
    state.controller = new AbortController();
    const signal = state.controller.signal;
    const runLens = state.lens;
    const context = elements.context.value.trim();
    state.resultUsesTrails = true;
    const runSignature = getCurrentSignature(pairs, context, runLens, trails, true);
    state.candidates = [];
    state.aiCandidates = [];
    state.senses = [];
    state.sourceSummary = null;
    state.resultSignature = null;
    state.filter = 'aligned';
    state.visibleCount = MAX_VISIBLE;
    elements.copy.disabled = true;
    elements.resultsFooter.hidden = true;
    elements.summary.hidden = true;
    elements.senseNotes.hidden = true;
    updateTabs();
    setBusy(true);
    elements.results.classList.remove('is-stale');
    elements.results.innerHTML = '<div class="wb-progress-state"><span class="loading-spinner" aria-hidden="true"></span><div><strong>Following lexical trails…</strong><small>Fetching senses and relations while the local model prepares.</small></div></div>';
    const lookupSources = [];
    if (['synonyms', 'taxonomy', 'opposites', 'morphology'].some((trail) => trails.has(trail))) lookupSources.push('Open English WordNet senses and relations');
    if (trails.has('meaninglike') || trails.has('association')) lookupSources.push('Datamuse associations');
    setStatus(`Looking up ${lookupSources.join(' and ')}…`, 'loading');
    elements.resultSubtitle.textContent = 'Building a candidate field from sourced lexical relations.';
    elements.fieldNote.textContent = 'Following the selected lexical trails. Retrieved words will be ranked locally; each relation and definition remains separately attributed.';
    updateMethodText(pairs, runLens);

    const terms = getUniqueInputTerms(pairs);
    const candidatePromise = collectCandidateField(terms, trails, signal);
    const modelPromise = loadEmbeddingModel();
    try {
      const [candidateResult, modelResult] = await Promise.allSettled([candidatePromise, modelPromise]);
      if (!isCurrentRun(runId)) return;
      if (modelResult.status === 'rejected') throw modelResult.reason;
      if (candidateResult.status === 'rejected') throw candidateResult.reason;
      const collected = candidateResult.value;
      const candidates = limitCandidateField(collected.candidates, MAX_CANDIDATES);
      if (!candidates.length) {
        state.candidates = [];
        state.senses = collected.senses;
        state.sourceSummary = collected.summary;
        state.pairSnapshot = pairs.map((pair) => ({ ...pair }));
        state.contextSnapshot = context;
        state.resultSignature = runSignature;
        state.resultLens = runLens;
        updateMethodText(pairs, runLens);
        renderSenseNotes(collected.senses, terms);
        updateSummary(collected.summary, candidates);
        renderResults();
        elements.resultSubtitle.textContent = 'No unused candidates were returned from the selected lexical trails.';
        elements.fieldNote.textContent = 'The sources returned no unused terms for this pass. Try shorter or broader concepts, another relation trail, or a different context hint.';
        const stale = getCurrentSignature() !== runSignature;
        if (stale) {
          elements.results.classList.add('is-stale');
          elements.resultSubtitle.textContent = 'Showing the previous settings · refresh to recalculate.';
        }
        const apiWorked = collected.summary.oewn?.successfulLookups || collected.summary.datamuse?.successfulQueries;
        const emptyStatus = apiWorked ? 'Sources responded, but produced no unused candidates. Try broader concepts or add a trail.' : 'No word source is reachable right now. Check your connection and retry.';
        setStatus(stale ? `${emptyStatus} Inputs changed during this run.` : emptyStatus, 'error');
        return;
      }

      setStatus(`Retrieved ${candidates.length} terms · encoding one batch with MiniLM on this device…`, 'loading');
      const inputTexts = pairs.flatMap((pair) => [
        context ? `${pair.toward}. Context: ${context}.` : pair.toward,
        context ? `${pair.away}. Context: ${context}.` : pair.away,
      ]);
      const candidateTexts = candidates.map((candidate) => context ? `${candidate.word}. Context: ${context}.` : candidate.word);
      const uniqueTextMap = new Map();
      [...inputTexts, ...candidateTexts].forEach((text) => {
        const key = embeddingKey(text);
        if (key && !uniqueTextMap.has(key)) uniqueTextMap.set(key, text);
      });
      const texts = [...uniqueTextMap.values()];
      let vectors;
      const missing = texts.filter((text) => !embeddingCache.has(embeddingKey(text)));
      if (missing.length) {
        const encoded = await embedTexts(missing);
        missing.forEach((text, index) => embeddingCache.set(embeddingKey(text), poolEmbeddingMatrix(encoded[index], { normalize: true })));
      }
      vectors = new Map(texts.map((text) => [embeddingKey(text), embeddingCache.get(embeddingKey(text))]));
      if (!isCurrentRun(runId)) return;

      const pairVectors = pairs.map((pair) => ({
        toward: vectors.get(embeddingKey(context ? `${pair.toward}. Context: ${context}.` : pair.toward)),
        away: vectors.get(embeddingKey(context ? `${pair.away}. Context: ${context}.` : pair.away)),
      }));
      if (pairVectors.some((pair) => !pair.toward || !pair.away)) throw new Error('The model did not return every concept vector.');
      const scored = candidates.map((candidate) => ({ ...candidate, vector: vectors.get(embeddingKey(context ? `${candidate.word}. Context: ${context}.` : candidate.word)) })).filter((candidate) => candidate.vector);
      const { results, direction } = rankWorkbenchCandidates(scored, pairVectors, pairs, runLens, vectors, MAX_CANDIDATES);
      state.candidates = results;
      state.senses = collected.senses;
      state.pairSnapshot = pairs.map((pair) => ({ ...pair }));
      state.contextSnapshot = context;
      state.sourceSummary = collected.summary;
      state.resultSignature = runSignature;
      state.resultLens = runLens;
      updateMethodText(pairs, runLens);
      renderSenseNotes(collected.senses, terms);
      updateSummary(collected.summary, candidates);
      renderResults();
      const resultCount = results.length;
      const stale = getCurrentSignature() !== runSignature;
      elements.results.classList.toggle('is-stale', stale);
      elements.resultSubtitle.textContent = `${resultCount} ranked candidates · ${runLens === 'contrast' ? 'more like A than B' : runLens === 'shared' ? 'shared-ground scan' : 'concept midpoint scan'}${stale ? ' · settings changed during this run' : ''}.`;
      const sourceMessages = [];
      if (collected.summary.oewn && !collected.summary.oewn.successfulLookups) sourceMessages.push('OEWN unavailable');
      else if (collected.summary.oewn?.failed) sourceMessages.push('some OEWN requests failed');
      if (collected.summary.datamuse && !collected.summary.datamuse.successfulQueries) sourceMessages.push('Datamuse unavailable');
      else if (collected.summary.datamuse?.failed) sourceMessages.push('some Datamuse requests failed');
      const statusSuffix = sourceMessages.length ? ` · ${sourceMessages.join(' · ')}; using what was available` : '';
      setStatus(`Done · ${resultCount} candidates ranked locally${statusSuffix}${stale ? ' · inputs changed; refresh to update' : ''}.`, stale ? '' : resultCount ? 'success' : '');
      const pairSummary = pairs.map((pair) => `${pair.toward} / ${pair.away}`).join('; ');
      const lensSummary = runLens === 'shared'
        ? `Shared-ground scan across both concepts in each pair: ${pairSummary}.`
        : runLens === 'blend'
          ? `Midpoint scan blending both concepts in each pair: ${pairSummary}.`
          : `Current contrast direction: ${pairs.map((pair) => `${pair.toward} minus ${pair.away}`).join('; ')}.`;
      elements.fieldNote.textContent = resultCount
        ? `${lensSummary} Best-scoring terms are relative to the retrieved candidate field. Open a result for its definition and lexical trail; use “Set as A” to branch. A high score does not prove equivalence or a real-world claim.`
        : 'No candidates remain after filtering out the input terms. Try shorter or more concrete concepts.';
      if (direction) elements.results.dataset.directionDimension = String(direction.length);
    } catch (error) {
      if (!isCurrentRun(runId)) return;
      const message = error?.message || 'Could not complete this exploration.';
      setStatus(message.includes('load') || message.includes('model') ? `The local embedding model could not run: ${message}` : message, 'error');
      elements.results.innerHTML = `<div class="wb-empty-state compact-empty"><span class="wb-empty-orbit" aria-hidden="true">!</span><h3>Couldn’t finish this pass</h3><p>${escapeHTML(message)} Your inputs are still here; retry or switch off a source that is unavailable.</p></div>`;
      elements.resultSubtitle.textContent = 'The workbench keeps source failures visible instead of inventing results.';
      elements.fieldNote.textContent = 'This pass did not produce ranked results. The inputs are unchanged; retry after checking the source or model connection.';
    } finally {
      if (isCurrentRun(runId)) setBusy(false);
    }
  }

  async function collectCandidateField(terms, trails, signal) {
    const jobs = [];
    const selectedWordnet = ['synonyms', 'taxonomy', 'opposites', 'morphology'].some((trail) => trails.has(trail));
    const selectedDatamuse = trails.has('meaninglike') || trails.has('association');
    if (selectedWordnet) jobs.push(fetchOewnCandidates(terms, trails, signal));
    else jobs.push(Promise.resolve({ candidates: [], senseNotes: [], successfulLookups: 0, relationSynsets: 0 }));
    if (selectedDatamuse) jobs.push(fetchDatamuseCandidates(terms, trails, signal));
    else jobs.push(Promise.resolve({ candidates: [], successfulQueries: 0, queryCount: 0 }));
    const [oewnResult, datamuseResult] = await Promise.allSettled(jobs);
    const oewn = oewnResult.status === 'fulfilled' ? oewnResult.value : { candidates: [], senseNotes: [], successfulLookups: 0, relationSynsets: 0, failed: true };
    const datamuse = datamuseResult.status === 'fulfilled' ? datamuseResult.value : { candidates: [], successfulQueries: 0, queryCount: 0, failed: true };
    const candidates = mergeCandidateLists(oewn.candidates, datamuse.candidates);
    const inputKeys = new Set(terms.map(normalizeText));
    const cleanCandidates = candidates.filter((candidate) => !inputKeys.has(normalizeText(candidate.word)));
    return {
      candidates: cleanCandidates,
      senses: oewn.senseNotes || [],
      summary: {
        oewn: selectedWordnet ? { successfulLookups: oewn.successfulLookups, senseCount: oewn.senseNotes?.length || 0, relationSynsets: oewn.relationSynsets || 0, failed: Boolean(oewn.failed || oewn.failedLookups || oewn.failedTargets) } : null,
        datamuse: selectedDatamuse ? { successfulQueries: datamuse.successfulQueries, candidates: datamuse.candidates?.length || 0, queryCount: datamuse.queryCount || 0, failed: Boolean(datamuse.failed || datamuse.failedQueries) } : null,
      },
    };
  }

  function activeResultText() {
    const candidates = visibleCandidates();
    if (!candidates.length) return '';
    const pairs = state.pairSnapshot.map((pair) => `A: ${pair.toward} | B: ${pair.away}`).join('\n');
    const header = `Articulator Workbench · ${state.resultLens || state.lens} lens\n${pairs}\nScores are model alignment, not probabilities or facts.`;
    const body = [...candidates].sort((a, b) => b.score - a.score).slice(0, MAX_EXPANDED).map((candidate, index) => {
      const definition = candidate.definitions[0]?.text || candidate.aiRationale || 'No definition available.';
      const sources = getSelectedEvidence(candidate).slice(0, 3).join('; ') || 'AI-generated, unverified';
      const sourceUrls = [...new Set(candidate.evidence.map((item) => item.url).filter(Boolean))].join('; ') || 'No lexical source link (AI-generated suggestion)';
      return `${index + 1}. ${candidate.word} (${candidate.score >= 0 ? '+' : ''}${candidate.score.toFixed(3)})\n   ${definition}\n   Trail: ${sources}\n   Source link: ${sourceUrls}`;
    }).join('\n');
    return `${header}\n\n${body}`;
  }

  async function copyResults() {
    const text = activeResultText();
    if (!text) return;
    await copyText(text, 'Workbench notes copied');
  }

  function setFilter(filter) {
    state.filter = filter;
    state.visibleCount = MAX_VISIBLE;
    updateTabs();
    renderResults();
  }

  function markInputsChanged() {
    markStale();
    if (!state.busy) elements.run.querySelector('span:nth-child(2)').textContent = state.resultSignature && getCurrentSignature() !== state.resultSignature ? 'Refresh exploration' : 'Explore this direction';
  }

  function sendTermToStudio(word, candidate) {
    const pair = state.pairSnapshot[0] || currentPairs()[0];
    const definition = candidate?.definitions?.[0]?.text || '';
    const source = candidate?.definitions?.[0]?.source || '';
    const shorten = (value, limit) => String(value || '').length > limit ? `${String(value).slice(0, limit - 1)}…` : String(value || '');
    const lexicalTrail = candidate?.evidence?.length ? candidate.evidence.slice(0, 2).map((item) => `${item.provider}: ${item.relation}${item.sourceTerm ? ` of ${item.sourceTerm}` : ''}`).join('; ') : 'no lexical source trail';
    const sourceLink = candidate?.evidence?.find((item) => item.url)?.url || 'no lexical source link';
    const toward = shorten(pair?.toward || 'A', 30);
    const away = shorten(pair?.away || 'B', 30);
    const note = candidate?.aiRationale
      ? `Explore “${shorten(word, 60)}” vs A “${toward}” and B “${away}”. AI-generated wildcard; not a dictionary entry. Speculative rationale: “${shorten(candidate.aiRationale, 130)}”. Alignment is not factual evidence; verify claims independently.`
      : `Explore “${shorten(word, 60)}” vs A “${toward}” and B “${away}”. Source: ${shorten(sourceLink, 90)}. ${definition ? `Definition (${shorten(source || 'source unspecified', 30)}): “${shorten(definition, 120)}”.` : 'No sourced definition is available.'} Trail: ${shorten(lexicalTrail, 70)}. Alignment is not a fact; verify claims independently.`;
    const idea = document.getElementById('ideaInput');
    idea.value = note.slice(0, 600);
    idea.dispatchEvent(new Event('input', { bubbles: true }));
    setWorkspaceView('studio');
    idea.focus();
    showToast(candidate?.aiRationale ? 'An unverified AI wildcard note is ready in Studio' : 'A sourced exploration note is ready in Studio');
  }

  function setFirstToward(word) {
    const firstRow = elements.pairs.querySelector('.wb-pair');
    const toward = firstRow?.querySelector('[data-side="toward"]');
    const away = firstRow?.querySelector('[data-side="away"]');
    if (!toward || !away) return;
    const previous = toward.value.trim();
    toward.value = word;
    if (previous && normalizeText(previous) === normalizeText(away.value)) away.value = '';
    toward.focus();
    markInputsChanged();
  }

  function renderAiCandidates(items, vectors, pairVectors, pairs, lens, context, runSignature) {
    const candidateMap = makeCandidateAccumulator();
    items.forEach((item) => candidateMap.add({ word: item.term, aiRationale: item.rationale, provider: 'AI suggestion', relation: 'speculative wildcard', group: 'ai' }));
    const candidates = candidateMap.values().map((candidate) => ({ ...candidate, vector: vectors.get(embeddingKey(context ? `${candidate.word}. Context: ${context}.` : candidate.word)) })).filter((candidate) => candidate.vector);
    const { results } = rankWorkbenchCandidates(candidates, pairVectors, pairs, lens, vectors, MAX_EXPANDED);
    state.aiCandidates = results.map((candidate) => ({ ...candidate, aiRationale: items.find((item) => normalizeText(item.term) === normalizeText(candidate.word))?.rationale || '' }));
    state.resultSignature = runSignature;
    state.resultLens = lens;
    updateMethodText(pairs, lens);
    updateTabs();
    setFilter('ai');
    const stale = getCurrentSignature() !== runSignature;
    elements.results.classList.toggle('is-stale', stale);
    elements.resultSubtitle.textContent = `AI-generated wildcards · unverified suggestions, locally ranked against the ${lens} lens.${stale ? ' Settings changed during this request.' : ''}`;
    setStatus(`AI returned ${state.aiCandidates.length} wildcard terms. Treat their rationales as suggestions, not facts.${stale ? ' Inputs changed during the request; refresh to update.' : ''}`, stale ? '' : 'success');
  }

  async function getAiWildcards() {
    const pairs = currentPairs();
    const invalid = pairs.findIndex((pair) => !pair.toward || !pair.away);
    if (invalid >= 0) {
      const row = elements.pairs.querySelectorAll('.wb-pair')[invalid];
      row?.querySelector(!pairs[invalid].toward ? '[data-side="toward"]' : '[data-side="away"]')?.focus();
      setStatus(`Comparison ${invalid + 1} needs both concepts first.`, 'error');
      return;
    }
    const runId = ++state.runId;
    const runLens = state.lens;
    const context = elements.context.value.trim();
    state.resultUsesTrails = false;
    const runSignature = getCurrentSignature(pairs, context, runLens, [], false);
    state.controller?.abort();
    state.controller = new AbortController();
    state.aiCandidates = [];
    state.resultSignature = null;
    state.pairSnapshot = pairs.map((pair) => ({ ...pair }));
    state.contextSnapshot = context;
    state.filter = 'ai';
    updateTabs();
    elements.copy.disabled = true;
    elements.resultsFooter.hidden = true;
    elements.summary.hidden = true;
    elements.senseNotes.hidden = true;
    elements.results.classList.remove('is-stale');
    setBusy(true);
    setStatus(`Sending the concepts${context ? ' and context hint' : ''} to the optional hosted AI for speculative vocabulary…`, 'loading');
    elements.resultSubtitle.textContent = 'AI wildcard search is separate from lexical results and is explicitly opt-in.';
    elements.fieldNote.textContent = 'AI suggestions are speculative vocabulary, not sourced definitions or verified relations. Their vector alignment is calculated locally after generation.';
    elements.results.innerHTML = '<div class="wb-progress-state"><span class="loading-spinner" aria-hidden="true"></span><div><strong>Asking for lateral candidates…</strong><small>The AI suggestions will be clearly marked and ranked locally.</small></div></div>';
    try {
      const puter = await loadPuter();
      if (!isCurrentRun(runId)) return;
      const promptPairs = pairs.map((pair) => `Toward: ${pair.toward} | Away from: ${pair.away}`).join('\n');
      const response = await puter.ai.chat([
        { role: 'system', content: 'You are a divergent vocabulary partner. Generate candidates, not facts. Do not invent definitions, citations, statistics, or claims. Return valid JSON only.' },
        { role: 'user', content: `For this semantic search direction, propose 12 unusual but plausible words or short concepts that may occupy a similar direction. Explore cross-disciplinary metaphors, old/technical vocabulary, material or sensory analogies, tensions, and bridge concepts. Avoid simply repeating the inputs.\n${promptPairs}\n${context ? `Context hint: ${context}\n` : ''}Return JSON exactly as {"items":[{"term":"short word or phrase","rationale":"one short speculative reason, explicitly not a factual definition"}]}. Keep each term under 5 words.` },
      ], { model: AI_MODEL });
      if (!isCurrentRun(runId)) return;
      const raw = extractChatText(response).trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '');
      let parsed;
      try { parsed = JSON.parse(raw); } catch { throw new Error('The AI response was not valid candidate JSON.'); }
      const items = (Array.isArray(parsed?.items) ? parsed.items : []).filter((item) => typeof item?.term === 'string' && item.term.trim()).slice(0, 12).map((item) => ({ term: item.term.trim().slice(0, 100), rationale: String(item.rationale || 'AI-generated association; not verified.').trim().slice(0, 250) }));
      if (!items.length) throw new Error('The AI did not return usable wildcard terms.');
      state.pairSnapshot = pairs.map((pair) => ({ ...pair }));
      state.contextSnapshot = context;
      setStatus('Encoding generated terms on-device…', 'loading');
      await loadEmbeddingModel();
      const withContext = (text) => context ? `${text}. Context: ${context}.` : text;
      const texts = pairs.flatMap((pair) => [withContext(pair.toward), withContext(pair.away)]).concat(items.map((item) => withContext(item.term)));
      const unique = new Map();
      texts.forEach((text) => { if (!unique.has(embeddingKey(text))) unique.set(embeddingKey(text), text); });
      const missing = [...unique.values()].filter((text) => !embeddingCache.has(embeddingKey(text)));
      if (missing.length) {
        const encoded = await embedTexts(missing);
        missing.forEach((text, index) => embeddingCache.set(embeddingKey(text), poolEmbeddingMatrix(encoded[index], { normalize: true })));
      }
      const vectors = new Map([...unique.values()].map((text) => [embeddingKey(text), embeddingCache.get(embeddingKey(text))]));
      const pairVectors = pairs.map((pair) => ({ toward: vectors.get(embeddingKey(withContext(pair.toward))), away: vectors.get(embeddingKey(withContext(pair.away))) }));
      renderAiCandidates(items, vectors, pairVectors, pairs, runLens, context, runSignature);
    } catch (error) {
      if (!isCurrentRun(runId)) return;
      setStatus(error?.message || 'The optional AI service is unavailable. Your lexical search still works.', 'error');
      elements.results.innerHTML = '<div class="wb-empty-state compact-empty"><span class="wb-empty-orbit" aria-hidden="true">✦</span><h3>Wildcards are optional</h3><p>The hosted suggestion service did not complete. Use the local lexical search for sourced candidates.</p></div>';
      elements.fieldNote.textContent = 'The optional AI service did not complete. Your concepts remain available for a sourced lexical search.';
    } finally {
      if (isCurrentRun(runId)) setBusy(false);
    }
  }

  function handleResultAction(event) {
    const setButton = event.target.closest('[data-set-toward]');
    if (setButton) {
      setFirstToward(setButton.dataset.setToward);
      return;
    }
    const noteButton = event.target.closest('[data-send-note]');
    if (noteButton) {
      const candidate = [...state.candidates, ...state.aiCandidates].find((item) => item.id === noteButton.dataset.sendNote);
      if (candidate) sendTermToStudio(candidate.word, candidate);
    }
  }

  function handlePairEvent(event) {
    const row = event.target.closest('.wb-pair');
    if (event.target.matches('[data-remove-pair]')) {
      if (elements.pairs.querySelectorAll('.wb-pair').length <= 1) return;
      row.remove();
      updatePairControls();
      markInputsChanged();
      return;
    }
    if (event.target.matches('[data-swap-pair]')) {
      const toward = row.querySelector('[data-side="toward"]');
      const away = row.querySelector('[data-side="away"]');
      [toward.value, away.value] = [away.value, toward.value];
      toward.focus();
      markInputsChanged();
    }
  }

  function initialize() {
    if (!elements.pairs || !elements.run) return;
    updatePairControls();
    elements.run.addEventListener('click', runExploration);
    elements.ai.addEventListener('click', getAiWildcards);
    elements.addPair.addEventListener('click', () => {
      const rows = elements.pairs.querySelectorAll('.wb-pair');
      if (rows.length >= MAX_PAIRS) return;
      elements.pairs.insertAdjacentHTML('beforeend', pairTemplate(rows.length));
      updatePairControls();
      elements.pairs.querySelector('.wb-pair:last-child [data-side="toward"]')?.focus();
      markInputsChanged();
    });
    elements.pairs.addEventListener('click', handlePairEvent);
    elements.pairs.addEventListener('input', (event) => {
      if (event.target.matches('.wb-concept')) markInputsChanged();
    });
    elements.context.addEventListener('input', markInputsChanged);
    document.querySelectorAll('[data-wb-lens]').forEach((button) => {
      button.addEventListener('click', () => {
        state.lens = button.dataset.wbLens;
        document.querySelectorAll('[data-wb-lens]').forEach((item) => {
          const selected = item === button;
          item.classList.toggle('is-selected', selected);
          item.setAttribute('aria-pressed', String(selected));
        });
        elements.lensExplanation.textContent = LENS_COPY[state.lens];
        updateMethodText(currentPairs());
        markInputsChanged();
      });
    });
    document.querySelectorAll('[data-wb-trail]').forEach((input) => input.addEventListener('change', markInputsChanged));
    document.querySelectorAll('[data-wb-filter]').forEach((button) => {
      button.addEventListener('click', () => setFilter(button.dataset.wbFilter));
    });
    elements.results.addEventListener('click', handleResultAction);
    elements.copy.addEventListener('click', copyResults);
    elements.showMore.addEventListener('click', () => {
      const filtered = visibleCandidates();
      state.visibleCount = state.visibleCount >= filtered.length ? MAX_VISIBLE : Math.min(state.visibleCount + MAX_VISIBLE, filtered.length);
      renderResults();
    });
    document.getElementById('wbLensHelp').addEventListener('click', () => {
      elements.lensExplanation.textContent = `${LENS_COPY[state.lens]} Select a different lens to explore the same candidates through another mathematical view.`;
      elements.lensExplanation.focus?.();
    });
    elements.pairs.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        runExploration();
      }
    });
    state.pairSnapshot = currentPairs();
    updateMethodText(state.pairSnapshot);
  }

  const vectorMath = Object.freeze({
    poolEmbeddingMatrix,
    normalizeVector,
    subtractVectors,
    meanVectors,
    combineDirections,
    cosine,
    rankCandidates,
  });
  if (typeof module !== 'undefined' && module.exports) module.exports = vectorMath;
  else if (typeof window !== 'undefined') window.ArticulatorVectorMath = vectorMath;

  initialize();
})();
