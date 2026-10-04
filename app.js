const DEFAULT_IDEA = 'How can public spaces help people feel more connected?';
const DRAFT_KEY = 'articulator:draft:v1';
const SHELF_KEY = 'articulator:shelf:v1';
const AI_MODEL = 'liquid/lfm-2.5-1.2b-instruct:free';
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
const PUTER_SCRIPT_URL = 'https://js.puter.com/v2/';

const MODES = {
  deepen: {
    label: 'Deepen',
    instruction: 'deepen the thought by surfacing context, human experience, causes, constraints, and tangible examples',
  },
  broaden: {
    label: 'Broaden',
    instruction: 'broaden the landscape by connecting adjacent disciplines, systems, timescales, and less obvious perspectives',
  },
  reframe: {
    label: 'Reframe',
    instruction: 'offer fresh and respectful reframings, alternate interpretations, and useful tensions without losing the original intent',
  },
};

const RELATIONS = [
  { id: 'synonyms', param: 'rel_syn', label: 'NEAR MEANINGS', relation: 'synonyms', color: 'syn' },
  { id: 'broader', param: 'rel_gen', label: 'BROADER CONCEPTS', relation: 'broader ideas', color: 'broader' },
  { id: 'specific', param: 'rel_spc', label: 'SPECIFIC EXAMPLES', relation: 'more specific ideas', color: 'specific' },
  { id: 'associated', param: 'rel_trg', label: 'CONNECTED IDEAS', relation: 'strongly associated ideas', color: 'related' },
];

const STOP_WORDS = new Set((
  'a an and are as at be been being by can could did do does for from had has have how i if in into is it its may might more most my of on or our should so some than that the their them then there these they this those to too up us was we were what when where which who why will with would you your about after again all also any because before between both but each feel felt get give goes good got help here her him his how idea just less like many make me much no not now often other own people person quite really same she so take than think through time under use used using very want way well while work world'
).split(' '));

const MODE_LABELS = { deepen: 'Deepen the thought', broaden: 'Broaden the landscape', reframe: 'Reframe the idea' };

const elements = {
  idea: document.getElementById('ideaInput'),
  charCount: document.getElementById('charCount'),
  autosave: document.getElementById('autosaveStatus'),
  audience: document.getElementById('audienceSelect'),
  tone: document.getElementById('toneSelect'),
  generate: document.getElementById('generateButton'),
  generateText: document.querySelector('.primary-button-text'),
  loadEmbedding: document.getElementById('loadEmbeddingButton'),
  embeddingDescription: document.getElementById('embeddingDescription'),
  modelStatus: document.getElementById('modelStatus'),
  outputSubtitle: document.getElementById('outputSubtitle'),
  loadingStatus: document.getElementById('loadingStatus'),
  loadingMessage: document.getElementById('loadingMessage'),
  outputBadge: document.getElementById('outputBadge'),
  expanded: document.getElementById('expandedText'),
  contextList: document.getElementById('contextList'),
  contextCount: document.getElementById('contextCount'),
  meaningGroups: document.getElementById('meaningGroups'),
  mapSubtitle: document.getElementById('mapSubtitle'),
  meaningSource: document.getElementById('meaningSource'),
  includedCount: document.getElementById('includedCount'),
  promptPreview: document.getElementById('promptPreview'),
  togglePrompt: document.getElementById('togglePromptButton'),
  copyPrompt: document.getElementById('copyPromptButton'),
  copyExpansion: document.getElementById('copyExpansionButton'),
  refreshMap: document.getElementById('refreshMapButton'),
  saveIdea: document.getElementById('saveIdeaButton'),
  shelfCount: document.getElementById('shelfCount'),
  savedList: document.getElementById('savedList'),
  toast: document.getElementById('toast'),
  toastMessage: document.getElementById('toastMessage'),
  helpDialog: document.getElementById('helpDialog'),
  shelfDialog: document.getElementById('shelfDialog'),
};

const exampleExpanded = elements.expanded.textContent.trim();
const exampleContexts = [...elements.contextList.querySelectorAll('.context-item')].map((item) => ({
  title: item.querySelector('h3')?.textContent.trim() || '',
  detail: item.querySelector('p')?.textContent.trim() || '',
}));

const state = {
  mode: 'deepen',
  mapData: [],
  mapSource: 'example',
  mapSeed: DEFAULT_IDEA,
  expanded: exampleExpanded,
  contexts: exampleContexts,
  generatedSeed: DEFAULT_IDEA,
  generatedSettings: { mode: 'deepen', audience: 'an AI language model', tone: 'clear and curious' },
  outputLabel: 'A SAMPLE EXPLORATION',
  modelReady: false,
  generationId: 0,
};

let embedder = null;
let embeddingLoadPromise = null;
let puterLoadPromise = null;
let toastTimer = null;
let draftTimer = null;
let lastMapRequest = 0;

function safeReadJSON(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function safeWriteJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);
}

function getSelectedTerms() {
  return [...elements.meaningGroups.querySelectorAll('.meaning-chip.is-included')]
    .map((chip) => ({
      word: chip.dataset.term || chip.textContent.replace(/^[+✓]\s*/, '').trim(),
      source: chip.dataset.source || '',
      group: chip.dataset.group || '',
    }))
    .filter((item) => item.word);
}

function updateCharCount() {
  elements.charCount.textContent = String(elements.idea.value.length);
}

function saveDraftSoon() {
  elements.autosave.classList.add('is-saving');
  elements.autosave.lastChild.textContent = 'Saving';
  window.clearTimeout(draftTimer);
  draftTimer = window.setTimeout(() => {
    const saved = safeWriteJSON(DRAFT_KEY, {
      idea: elements.idea.value,
      mode: state.mode,
      audience: elements.audience.value,
      tone: elements.tone.value,
    });
    elements.autosave.classList.remove('is-saving');
    elements.autosave.lastChild.textContent = saved ? 'Saved' : 'Not saved';
  }, 250);
}

function getSettings() {
  return {
    mode: state.mode,
    audience: elements.audience.value,
    tone: elements.tone.value,
  };
}

function isCurrentOutput() {
  const idea = elements.idea.value.trim();
  const settings = getSettings();
  return idea === state.generatedSeed
    && settings.mode === state.generatedSettings.mode
    && settings.audience === state.generatedSettings.audience
    && settings.tone === state.generatedSettings.tone;
}

function syncStaleState() {
  const currentIdea = elements.idea.value.trim();
  if (state.mapSource === 'example' && currentIdea !== DEFAULT_IDEA) {
    state.mapSource = 'none';
    state.mapSeed = null;
    state.mapData = [];
    elements.meaningGroups.innerHTML = '<div class="meaning-empty">Refresh the map or expand this seed to look up words for your idea.</div>';
    elements.mapSubtitle.textContent = 'Your thought changed · refresh to look up its word neighborhood.';
    elements.meaningSource.textContent = 'Example links cleared · refresh for this seed';
    elements.includedCount.textContent = '0 words woven in';
  } else if (state.mapSource === 'live' && state.mapSeed && currentIdea !== state.mapSeed) {
    elements.mapSubtitle.textContent = 'This word map is for an earlier seed · refresh it for the current thought.';
    elements.meaningSource.textContent = 'Previous seed · refresh to get matching links';
  } else if (state.mapSource === 'live' && state.mapSeed && currentIdea === state.mapSeed) {
    elements.mapSubtitle.textContent = 'Live word relationships are connected to this seed.';
    elements.meaningSource.textContent = 'Live Datamuse links · click a word to include or remove it';
  }
  if (state.mapSource === 'example' && currentIdea === DEFAULT_IDEA && isCurrentOutput()) {
    elements.outputSubtitle.textContent = 'One thought, with more room around it.';
    elements.outputBadge.textContent = 'A SAMPLE EXPLORATION';
    return;
  }
  if (!state.generatedSeed) {
    elements.outputSubtitle.textContent = 'Your expanded thought will appear here when you are ready.';
    return;
  }
  if (!isCurrentOutput()) {
    elements.outputSubtitle.textContent = 'Your seed or direction changed · expand again to refresh the context.';
    elements.outputBadge.textContent = 'DRAFT NEEDS A REFRESH';
  } else {
    elements.outputSubtitle.textContent = state.mapSource === 'example' ? 'One thought, with more room around it.' : 'The idea, expanded with meaning and context.';
    if (state.mapSource !== 'example' && state.expanded) {
      elements.outputBadge.textContent = state.outputLabel || 'EXPANDED WITH FREE AI';
    }
  }
}

function createPrompt() {
  const idea = elements.idea.value.trim();
  if (!idea) return 'Add a thought to begin building your prompt.';

  const isSameSeed = idea === state.generatedSeed;
  const expanded = isSameSeed && state.expanded
    ? state.expanded
    : 'No expanded context yet. Develop the starting idea before drafting the response.';
  const terms = getSelectedTerms();
  const termLine = terms.length
    ? `Meaning links to explore when relevant: ${terms.map((item) => item.word).join(', ')}.`
    : 'Use useful semantic connections where they genuinely clarify the idea; do not force unrelated concepts.';
  const modeLabel = MODE_LABELS[state.mode] || MODE_LABELS.deepen;

  return [
    'Act as a thoughtful writing partner. Help me develop the idea below into useful, specific context for further work. Preserve the original intent, avoid inventing facts, and distinguish questions or assumptions from established information.',
    '',
    'STARTING IDEA',
    `“${idea}”`,
    '',
    'EXPANDED CONTEXT',
    expanded,
    '',
    termLine,
    `Direction: ${modeLabel}.`,
    `Audience: ${elements.audience.value}. Tone: ${elements.tone.value}.`,
    '',
    'Make the response clear, concrete, and easy to build on. Include useful perspectives, examples or questions where they fit. Keep the central idea recognizable.',
  ].join('\n');
}

function renderPrompt() {
  elements.promptPreview.textContent = createPrompt();
}

function updateIncludedCount() {
  const count = getSelectedTerms().length;
  elements.includedCount.textContent = `${count} ${count === 1 ? 'word' : 'words'} woven in`;
}

function refreshPromptFromState() {
  renderPrompt();
  updateIncludedCount();
  saveDraftSoon();
}

function setMode(mode) {
  if (!MODES[mode]) return;
  state.mode = mode;
  document.querySelectorAll('.direction-option').forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle('is-selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  syncStaleState();
  refreshPromptFromState();
}

function setChipState(chip, included) {
  chip.classList.toggle('is-included', included);
  chip.setAttribute('aria-pressed', String(included));
  const mark = chip.querySelector('.chip-plus');
  if (mark) mark.textContent = included ? '✓' : '+';
}

function makeEmptyResults() {
  state.expanded = '';
  state.contexts = [];
  state.generatedSeed = null;
  state.mapSource = 'none';
  state.mapSeed = null;
  state.mapData = [];
  state.outputLabel = 'READY WHEN YOU ARE';
  elements.expanded.textContent = 'Your thought will take shape here. Expand an idea to add useful context, connected meanings, and a prompt you can take anywhere.';
  elements.outputBadge.textContent = 'READY WHEN YOU ARE';
  elements.outputSubtitle.textContent = 'Your expanded thought will appear here when you are ready.';
  elements.contextCount.textContent = '00';
  elements.contextList.innerHTML = '<div class="meaning-empty">A few useful context directions will appear here after you expand your idea.</div>';
  elements.meaningGroups.innerHTML = '<div class="meaning-empty">Your idea’s word neighborhood will appear here after a lookup. You can still create a prompt without it.</div>';
  elements.mapSubtitle.textContent = 'A word can open a whole neighborhood of ideas.';
  elements.meaningSource.textContent = 'Generate an idea to look up live lexical links';
  elements.includedCount.textContent = '0 words woven in';
  renderPrompt();
}

function restoreDraft() {
  const saved = safeReadJSON(DRAFT_KEY, null);
  if (!saved || typeof saved !== 'object') return;
  if (typeof saved.idea === 'string' && saved.idea.length <= 600) elements.idea.value = saved.idea;
  if (MODES[saved.mode]) state.mode = saved.mode;
  if ([...elements.audience.options].some((option) => option.value === saved.audience)) elements.audience.value = saved.audience;
  if ([...elements.tone.options].some((option) => option.value === saved.tone)) elements.tone.value = saved.tone;

  if (elements.idea.value.trim() !== DEFAULT_IDEA
    || state.mode !== 'deepen'
    || elements.audience.value !== 'an AI language model'
    || elements.tone.value !== 'clear and curious') {
    makeEmptyResults();
  }
  document.querySelectorAll('.direction-option').forEach((button) => {
    const selected = button.dataset.mode === state.mode;
    button.classList.toggle('is-selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
}

function showToast(message) {
  elements.toastMessage.textContent = message;
  elements.toast.classList.add('is-visible');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove('is-visible'), 2600);
}

async function copyText(text, label) {
  if (!text) {
    showToast('There is nothing to copy yet');
    return;
  }
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const fallback = document.createElement('textarea');
      fallback.value = text;
      fallback.style.position = 'fixed';
      fallback.style.opacity = '0';
      document.body.appendChild(fallback);
      fallback.select();
      document.execCommand('copy');
      fallback.remove();
    }
    showToast(label);
  } catch {
    showToast('Clipboard access was blocked by the browser');
  }
}

function openDialog(dialog) {
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else dialog.setAttribute('open', '');
}

function closeDialog(dialog) {
  if (typeof dialog.close === 'function' && dialog.open) dialog.close();
  else dialog.removeAttribute('open');
}

function extractKeywords(text) {
  const tokens = [...text.toLowerCase().matchAll(/[a-z][a-z'-]{1,30}/g)];
  const counts = new Map();
  for (const match of tokens) {
    const word = match[0].replace(/^['-]+|['-]+$/g, '');
    if (word.length < 3 || STOP_WORDS.has(word)) continue;
    const item = counts.get(word) || { word, count: 0, index: match.index || 0 };
    item.count += 1;
    counts.set(word, item);
  }
  const candidates = [...counts.values()];
  candidates.sort((a, b) => {
    const scoreA = (Math.min(a.word.length, 11) * 1.25) + Math.log1p(a.count) * 3;
    const scoreB = (Math.min(b.word.length, 11) * 1.25) + Math.log1p(b.count) * 3;
    return scoreB - scoreA || a.index - b.index;
  });
  return candidates.slice(0, 3).sort((a, b) => a.index - b.index).map((item) => item.word);
}

async function queryDatamuse(seed, relation) {
  const url = new URL('https://api.datamuse.com/words');
  url.searchParams.set(relation.param, seed);
  url.searchParams.set('max', '7');
  url.searchParams.set('md', 'p');
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 6500);
  try {
    const response = await fetch(url.toString(), { signal: controller.signal, mode: 'cors' });
    if (!response.ok) throw new Error(`Word API returned ${response.status}`);
    const result = await response.json();
    if (!Array.isArray(result)) return [];
    return result.map((entry) => typeof entry.word === 'string' ? entry.word.trim() : '')
      .filter((word) => word && word.toLowerCase() !== seed.toLowerCase() && word.length < 45 && !/[<>\n\r]/.test(word));
  } finally {
    window.clearTimeout(timer);
  }
}

async function fetchMeaningMap(idea) {
  const requestId = ++lastMapRequest;
  const seeds = extractKeywords(idea);
  if (!seeds.length) {
    return { requestId, seeds: [], groups: RELATIONS.map((relation) => ({ ...relation, items: [] })), anyResponse: false };
  }

  const jobs = [];
  for (const relation of RELATIONS) {
    for (const seed of seeds) {
      jobs.push(queryDatamuse(seed, relation).then((words) => ({ relationId: relation.id, seed, words })));
    }
  }
  const settled = await Promise.allSettled(jobs);
  const groups = RELATIONS.map((relation) => ({ ...relation, items: [] }));
  let anyResponse = false;
  for (const result of settled) {
    if (result.status !== 'fulfilled') continue;
    anyResponse = true;
    const group = groups.find((item) => item.id === result.value.relationId);
    if (!group) continue;
    for (const word of result.value.words) {
      const exists = group.items.some((item) => item.word.toLowerCase() === word.toLowerCase());
      if (!exists && group.items.length < 9) group.items.push({ word, source: result.value.seed, score: null });
    }
  }
  if (requestId !== lastMapRequest) return { requestId, stale: true, seeds, groups, anyResponse };
  return { requestId, seeds, groups, anyResponse };
}

function renderMeaningMap(groups, { sample = false, anyResponse = true, preserveSelection = true, noSeeds = false } = {}) {
  if (!groups?.length || (!sample && groups.every((group) => !group.items.length))) {
    const emptyMessage = noSeeds
      ? 'There were no clear topic words to look up. Try adding a few concrete nouns or concepts.'
      : anyResponse
        ? 'No close word relationships appeared for this thought. Try a different phrase, or continue with the idea you have.'
        : 'The free word graph could not be reached right now. Your idea and prompt are still available; try refreshing the map later.';
    elements.meaningGroups.innerHTML = `<div class="meaning-empty">${emptyMessage}</div>`;
    elements.meaningSource.textContent = noSeeds ? 'No topic word found in this seed' : anyResponse ? 'No lexical matches for this seed' : 'Word API offline · try again later';
    updateIncludedCount();
    renderPrompt();
    return;
  }

  const selected = preserveSelection
    ? new Set(getSelectedTerms().map((item) => item.word.toLowerCase()))
    : new Set(groups.flatMap((group) => group.items.filter((item) => item.included).map((item) => item.word.toLowerCase())));
  const markup = groups.map((group) => {
    const chips = group.items.map((item) => {
      const included = selected.has(item.word.toLowerCase());
      const score = typeof item.score === 'number'
        ? `<span class="similarity-score" title="Cosine similarity from local sentence embeddings">${item.score.toFixed(2)}</span>`
        : '';
      return `<button class="meaning-chip${included ? ' is-included' : ''}" type="button" data-term="${escapeHTML(item.word)}" data-source="${escapeHTML(item.source)}" data-group="${escapeHTML(group.id)}" aria-pressed="${included}" title="${escapeHTML(`${group.relation} of ${item.source}`)}"><span class="chip-plus">${included ? '✓' : '+'}</span>${escapeHTML(item.word)}${score}</button>`;
    }).join('');
    const content = chips || '<span class="meaning-group-empty">No links found for this relation.</span>';
    return `<div class="meaning-group"><div class="meaning-group-label"><span class="meaning-dot dot-${escapeHTML(group.color)}"></span><span>${escapeHTML(group.label)}</span><small>${escapeHTML(group.relation)}</small></div><div class="meaning-chips">${content}</div></div>`;
  }).join('');
  elements.meaningGroups.innerHTML = markup;
  elements.meaningSource.textContent = sample
    ? 'Example links · click a word to weave it into your prompt'
    : 'Live Datamuse links · click a word to include or remove it';
  updateIncludedCount();
  renderPrompt();
}

function cosineSimilarity(a, b) {
  if (!a?.length || !b?.length || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function getTensorVectors(output, expectedCount) {
  if (Array.isArray(output)) {
    return output.map((item) => Array.from(item?.data || item || []));
  }
  if (!output?.data) throw new Error('The embedding model returned an unexpected result.');
  const data = Array.from(output.data);
  const dims = Array.isArray(output.dims) ? output.dims : [];
  const dimension = dims.length ? dims[dims.length - 1] : Math.floor(data.length / Math.max(1, expectedCount));
  if (!dimension) throw new Error('The embedding model did not return any vectors.');
  const count = Math.floor(data.length / dimension);
  return Array.from({ length: Math.min(count, expectedCount) }, (_, index) => data.slice(index * dimension, (index + 1) * dimension));
}

async function embedTexts(texts) {
  if (!embedder) throw new Error('Local embeddings are not loaded.');
  const output = await embedder(texts, { pooling: 'mean', normalize: true });
  const vectors = getTensorVectors(output, texts.length);
  if (vectors.length !== texts.length) throw new Error('Could not match vectors to the meaning links.');
  return vectors;
}

async function rankMeaningMap() {
  if (!state.modelReady || !state.mapData.length || state.mapSource !== 'live') return;
  const requestId = lastMapRequest;
  const candidates = state.mapData.flatMap((group) => group.items.map((item) => ({ ...item, groupId: group.id })));
  if (!candidates.length) return;
  elements.modelStatus.textContent = 'Comparing your thought with nearby meanings…';
  try {
    const texts = [elements.idea.value.trim() || state.generatedSeed || '', ...candidates.map((item) => `${item.source}: ${item.word}`)];
    const vectors = await embedTexts(texts);
    if (requestId !== lastMapRequest) return;
    const ideaVector = vectors[0];
    const scored = new Map();
    candidates.forEach((item, index) => {
      scored.set(`${item.groupId}:${item.word.toLowerCase()}`, cosineSimilarity(ideaVector, vectors[index + 1]));
    });
    state.mapData = state.mapData.map((group) => ({
      ...group,
      items: group.items.map((item) => ({
        ...item,
        score: scored.get(`${group.id}:${item.word.toLowerCase()}`) ?? null,
      })).sort((a, b) => (b.score ?? -1) - (a.score ?? -1)),
    }));
    renderMeaningMap(state.mapData, { anyResponse: true });
    elements.modelStatus.textContent = 'Ready · meaning links ranked against your full thought on this device.';
  } catch (error) {
    elements.modelStatus.textContent = error?.message || 'Could not compare the local vectors. Try again.';
  }
}

async function loadEmbeddingModel() {
  if (embedder) return embedder;
  if (embeddingLoadPromise) return embeddingLoadPromise;

  elements.loadEmbedding.disabled = true;
  elements.loadEmbedding.textContent = 'Loading…';
  elements.modelStatus.textContent = 'Loading the local semantic model…';
  elements.embeddingDescription.textContent = 'Downloading MiniLM · runs in this browser';

  embeddingLoadPromise = (async () => {
    try {
      const transformers = await import(TRANSFORMERS_URL);
      if (transformers.env) {
        transformers.env.allowLocalModels = false;
        transformers.env.useBrowserCache = true;
      }
      const pipeline = transformers.pipeline || transformers.default?.pipeline;
      if (typeof pipeline !== 'function') throw new Error('The embedding library could not be initialized.');
      embedder = await pipeline('feature-extraction', EMBEDDING_MODEL, {
        quantized: true,
        progress_callback: (progress) => {
          if (!progress || progress.status !== 'progress') return;
          const percent = Number.isFinite(progress.progress) ? Math.round(progress.progress) : null;
          elements.loadEmbedding.textContent = percent === null ? 'Loading…' : `${percent}%`;
          elements.modelStatus.textContent = percent === null
            ? 'Downloading the optional model to this browser…'
            : `Downloading the optional model · ${percent}%`;
        },
      });
      state.modelReady = true;
      elements.loadEmbedding.disabled = false;
      elements.loadEmbedding.textContent = 'Ready';
      elements.loadEmbedding.title = 'The local embedding model is ready';
      elements.embeddingDescription.textContent = 'MiniLM vectors · on-device and ready';
      elements.modelStatus.textContent = 'Ready · your text stays on-device while meaning links are ranked.';
      if (state.mapSource === 'live') await rankMeaningMap();
      return embedder;
    } catch (error) {
      embedder = null;
      elements.loadEmbedding.disabled = false;
      elements.loadEmbedding.textContent = 'Retry';
      elements.embeddingDescription.textContent = 'MiniLM could not be loaded · retry when online';
      elements.modelStatus.textContent = error?.message
        ? `Could not load the model: ${error.message}`
        : 'Could not reach the model host. Check your connection and retry.';
      throw error;
    } finally {
      embeddingLoadPromise = null;
    }
  })();
  return embeddingLoadPromise;
}

function localFallback(idea, mode, selectedTerms) {
  const links = selectedTerms.length ? `Connected ideas such as ${formatList(selectedTerms.slice(0, 4))} can add useful texture` : 'The people involved, the setting, and the forces around it can add useful texture';
  const endings = {
    deepen: `Start with “${idea}” and look beneath the surface: who is affected, what shapes their experience, and what conditions could change the outcome? ${links}. Keep the original intention visible while making room for specific situations, constraints, and questions that would help someone explore it thoughtfully.`,
    broaden: `Take “${idea}” as a starting point rather than a boundary. Connect it to adjacent systems, disciplines, and scales—from everyday experiences to the wider conditions that shape them. ${links}; use those connections to reveal fresh opportunities and perspectives without losing the central thread.`,
    reframe: `Keep the heart of “${idea},” then look at it from more than one angle. What changes when the idea is viewed by a different person, in a different setting, or through a useful counterpoint? ${links}. Let the reframing create productive questions, not replace the original thought.`,
  };
  return endings[mode] || endings.deepen;
}

function localFallbackContexts(mode, selectedTerms) {
  const selected = selectedTerms.slice(0, 3);
  const termHint = selected.length ? ` Could ${formatList(selected)} reveal an overlooked angle?` : '';
  if (mode === 'broaden') return [
    { title: 'Adjacent fields', detail: 'Which neighboring disciplines, communities, or systems see this issue differently?' },
    { title: 'Scale & time', detail: 'How might the idea change at the personal, local, and wider-system level?' },
    { title: 'Unexpected links', detail: `What surprising connection could make the idea more generative?${termHint}` },
  ];
  if (mode === 'reframe') return [
    { title: 'Another point of view', detail: 'How might someone with different needs or lived experience describe the same situation?' },
    { title: 'A productive tension', detail: 'What is the strongest counterpoint, trade-off, or assumption worth testing?' },
    { title: 'A new frame', detail: `What changes if the core question is asked in a different way?${termHint}` },
  ];
  return [
    { title: 'People & lived experience', detail: 'Who is affected, and what do they need, notice, or experience in this situation?' },
    { title: 'Context & conditions', detail: 'Which surrounding systems, constraints, or everyday details shape what is possible?' },
    { title: 'Examples & trade-offs', detail: `What concrete situation would make the idea clearer—and what should be tested?${termHint}` },
  ];
}

function buildAIConversation(idea, selectedTerms, settings) {
  const links = selectedTerms.length
    ? selectedTerms.map((item) => `- ${item.word}${item.source ? ` (linked from “${item.source}”)` : ''}`).join('\n')
    : '- No words selected. Use your own judgment to find relevant connections.';
  const userMessage = [
    `Starting thought: ${idea}`,
    `Direction: ${MODES[settings.mode].instruction}.`,
    `Audience: ${settings.audience}.`,
    `Tone: ${settings.tone}.`,
    'Optional lexical links chosen by the writer:',
    links,
    '',
    'Expand the seed into a useful, richer thought. Preserve its intent, add context without making unsupported factual claims, and use selected links only where they fit naturally. Then offer exactly three concise, distinct context directions that help the writer keep exploring.',
    'Return only valid JSON, with this exact shape: {"expanded":"two to four sentences","directions":[{"title":"short label","detail":"one useful question or perspective"},{"title":"short label","detail":"one useful question or perspective"},{"title":"short label","detail":"one useful question or perspective"}]}. No markdown fences or extra keys.',
  ].join('\n');
  const systemMessage = 'You are Articulator, a careful and imaginative idea-development partner. Keep the writer’s intent intact. Add useful context, human perspectives, and specificity; do not pretend that possibilities are facts. Be concise, concrete, and non-repetitive. Return valid JSON only.';
  return [
    { role: 'system', content: systemMessage },
    { role: 'user', content: userMessage },
  ];
}

function loadPuter() {
  if (window.puter?.ai?.chat) return Promise.resolve(window.puter);
  if (puterLoadPromise) return puterLoadPromise;
  puterLoadPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-puter-client]');
    const script = existing || document.createElement('script');
    const timeout = window.setTimeout(() => {
      if (!window.puter?.ai?.chat) script.remove();
      reject(new Error('The free AI service took too long to load.'));
    }, 14000);
    const finish = () => {
      window.clearTimeout(timeout);
      if (window.puter?.ai?.chat) resolve(window.puter);
      else {
        script.remove();
        reject(new Error('The free AI library loaded without its chat service.'));
      }
    };
    script.addEventListener('load', finish, { once: true });
    script.addEventListener('error', () => {
      window.clearTimeout(timeout);
      script.remove();
      reject(new Error('Could not reach the free AI service.'));
    }, { once: true });
    if (!existing) {
      script.src = PUTER_SCRIPT_URL;
      script.async = true;
      script.dataset.puterClient = 'true';
      script.referrerPolicy = 'no-referrer';
      document.head.appendChild(script);
    } else if (window.puter?.ai?.chat) {
      window.clearTimeout(timeout);
      resolve(window.puter);
    }
  }).finally(() => {
    puterLoadPromise = null;
  });
  return puterLoadPromise;
}

function extractChatText(response) {
  if (typeof response === 'string') return response;
  const content = response?.message?.content ?? response?.choices?.[0]?.message?.content ?? response?.text ?? response?.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map((part) => typeof part === 'string' ? part : (part?.text || '')).join('');
  return '';
}

function parseAIResult(raw, idea, mode, selectedTerms) {
  const cleaned = raw.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '');
  let parsed = null;
  try {
    parsed = JSON.parse(cleaned);
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (match) {
      try { parsed = JSON.parse(match[0]); } catch { /* Use a clear local fallback below. */ }
    }
  }
  const expanded = typeof parsed?.expanded === 'string' && parsed.expanded.trim()
    ? parsed.expanded.trim()
    : cleaned;
  const directions = Array.isArray(parsed?.directions)
    ? parsed.directions.filter((item) => typeof item?.title === 'string' && typeof item?.detail === 'string')
      .slice(0, 3).map((item) => ({ title: item.title.trim(), detail: item.detail.trim() }))
    : [];
  return {
    expanded: expanded && !expanded.startsWith('{')
      ? expanded
      : localFallback(idea, mode, selectedTerms),
    contexts: directions.length === 3 ? directions : localFallbackContexts(mode, selectedTerms),
  };
}

function setContexts(contexts) {
  const safeContexts = (contexts || []).slice(0, 3);
  elements.contextCount.textContent = String(safeContexts.length).padStart(2, '0');
  if (!safeContexts.length) {
    elements.contextList.innerHTML = '<div class="meaning-empty">No context directions were returned this time.</div>';
    return;
  }
  elements.contextList.innerHTML = safeContexts.map((item, index) => `
    <article class="context-item">
      <span class="context-index">${String.fromCharCode(65 + index)}</span>
      <div><h3>${escapeHTML(item.title)}</h3><p>${escapeHTML(item.detail)}</p></div>
      <span class="context-chevron" aria-hidden="true">↗</span>
    </article>`).join('');
}

function formatList(items) {
  const values = items.map((item) => typeof item === 'string' ? item : item.word).filter(Boolean);
  if (values.length < 2) return values[0] || '';
  if (values.length === 2) return `${values[0]} and ${values[1]}`;
  return `${values.slice(0, -1).join(', ')}, and ${values.at(-1)}`;
}

async function requestAI(idea, selectedTerms, settings) {
  const puter = await loadPuter();
  const conversation = buildAIConversation(idea, selectedTerms, settings);
  const request = puter.ai.chat(conversation, { model: AI_MODEL });
  let timeoutId;
  try {
    const timeout = new Promise((_, reject) => {
      timeoutId = window.setTimeout(() => reject(new Error('The AI model took too long to respond.')), 35000);
    });
    const response = await Promise.race([request, timeout]);
    const text = extractChatText(response);
    if (!text.trim()) throw new Error('The AI returned an empty response.');
    return parseAIResult(text, idea, settings.mode, selectedTerms);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function chooseInitialLinks(groups) {
  const chosen = [];
  const priorities = ['synonyms', 'broader', 'specific', 'associated'];
  priorities.forEach((id) => {
    const group = groups.find((item) => item.id === id);
    if (group?.items?.length) chosen.push(group.items[0].word.toLowerCase());
  });
  const max = 4;
  const chosenSet = new Set(chosen.slice(0, max));
  return groups.map((group) => ({
    ...group,
    items: group.items.map((item) => ({ ...item, included: chosenSet.has(item.word.toLowerCase()) })),
  }));
}

function showLoading(message) {
  elements.loadingMessage.textContent = message;
  elements.loadingStatus.hidden = false;
}

function hideLoading() {
  elements.loadingStatus.hidden = true;
}

function setGenerating(isGenerating) {
  elements.generate.disabled = isGenerating;
  elements.refreshMap.disabled = isGenerating;
  elements.generateText.textContent = isGenerating ? 'Following the idea…' : 'Expand this idea';
  elements.generate.querySelector('.button-spark').textContent = isGenerating ? '◌' : '✳';
}

async function generateIdea() {
  const idea = elements.idea.value.trim();
  if (!idea) {
    elements.idea.focus();
    showToast('Add a thought before expanding it');
    return;
  }

  const generationId = ++state.generationId;
  const generationSettings = getSettings();
  setGenerating(true);
  showLoading('Finding a word neighborhood…');
  elements.outputSubtitle.textContent = 'Following relationships, then adding context.';
  elements.outputBadge.textContent = 'WORKING ON YOUR IDEA';
  saveDraftSoon();

  try {
    const mapResult = await fetchMeaningMap(idea);
    if (generationId !== state.generationId || mapResult.stale) return;
    const initialGroups = chooseInitialLinks(mapResult.groups);
    state.mapData = initialGroups.map((group) => ({
      ...group,
      items: group.items.map((item) => ({ word: item.word, source: item.source, score: null })),
    }));
    state.mapSource = 'live';
    state.mapSeed = idea;
    renderMeaningMap(initialGroups, { anyResponse: mapResult.anyResponse, preserveSelection: false, noSeeds: !mapResult.seeds.length });
    elements.mapSubtitle.textContent = mapResult.seeds.length
      ? `Live word relationships from ${mapResult.seeds.join(', ')}.`
      : 'No clear topic word was found · the prompt can still be expanded.';
    if (state.modelReady && state.mapData.length) await rankMeaningMap();

    const selectedTerms = getSelectedTerms();
    showLoading('Adding context with the free AI…');
    let result;
    let outputLabel = 'EXPANDED WITH FREE AI';
    let aiFailed = false;
    try {
      result = await requestAI(idea, selectedTerms, generationSettings);
    } catch (error) {
      result = {
        expanded: localFallback(idea, generationSettings.mode, selectedTerms),
        contexts: localFallbackContexts(generationSettings.mode, selectedTerms),
      };
      outputLabel = 'LOCAL DRAFT · AI UNAVAILABLE';
      aiFailed = true;
    }
    if (generationId !== state.generationId) return;

    state.expanded = result.expanded;
    state.contexts = result.contexts;
    state.generatedSeed = idea;
    state.generatedSettings = generationSettings;
    state.outputLabel = outputLabel;
    elements.expanded.textContent = result.expanded;
    elements.outputBadge.textContent = outputLabel;
    elements.outputSubtitle.textContent = 'The idea, expanded with meaning and context.';
    setContexts(result.contexts);
    renderPrompt();
    elements.meaningSource.textContent = mapResult.anyResponse
      ? 'Live Datamuse links · click a word to include or remove it'
      : 'Word API offline · draft created without live links';
    updateIncludedCount();
    if (aiFailed) showToast('AI unavailable · your local draft is ready');
    else showToast('Your idea has more room now');
  } catch (error) {
    if (generationId !== state.generationId) return;
    const selectedTerms = getSelectedTerms();
    state.expanded = localFallback(idea, generationSettings.mode, selectedTerms);
    state.contexts = localFallbackContexts(generationSettings.mode, selectedTerms);
    state.generatedSeed = idea;
    state.generatedSettings = generationSettings;
    state.outputLabel = 'LOCAL DRAFT';
    elements.expanded.textContent = state.expanded;
    elements.outputBadge.textContent = state.outputLabel;
    setContexts(state.contexts);
    renderPrompt();
    showToast(error?.message || 'A local draft was created instead');
  } finally {
    if (generationId === state.generationId) {
      hideLoading();
      setGenerating(false);
      syncStaleState();
    }
  }
}

async function refreshOnlyMap() {
  const idea = elements.idea.value.trim();
  if (!idea) {
    showToast('Add a thought before refreshing the map');
    return;
  }
  const before = elements.refreshMap.innerHTML;
  elements.refreshMap.disabled = true;
  elements.refreshMap.innerHTML = '<span class="loading-spinner" aria-hidden="true"></span><span>Looking up</span>';
  elements.meaningSource.textContent = 'Looking up live word relationships…';
  try {
    const result = await fetchMeaningMap(idea);
    if (result.stale) return;
    const selected = new Set(getSelectedTerms().map((item) => item.word.toLowerCase()));
    state.mapData = result.groups.map((group) => ({
      ...group,
      items: group.items.map((item) => ({ ...item, score: null, included: selected.has(item.word.toLowerCase()) })),
    }));
    state.mapSource = 'live';
    state.mapSeed = idea;
    renderMeaningMap(state.mapData, { anyResponse: result.anyResponse, noSeeds: !result.seeds.length });
    elements.mapSubtitle.textContent = result.seeds.length
      ? `Live word relationships from ${result.seeds.join(', ')}.`
      : 'No clear topic word was found.';
    if (state.modelReady) await rankMeaningMap();
    showToast(result.anyResponse ? 'Meaning map refreshed' : 'The word graph is unavailable right now');
  } catch {
    showToast('Could not refresh the meaning map');
  } finally {
    elements.refreshMap.disabled = elements.generate.disabled;
    elements.refreshMap.innerHTML = before;
  }
}

function getShelf() {
  const saved = safeReadJSON(SHELF_KEY, []);
  return Array.isArray(saved) ? saved.filter((item) => item && typeof item.id === 'string') : [];
}

function updateShelfCount() {
  elements.shelfCount.textContent = String(getShelf().length);
}

function saveCurrentIdea() {
  const idea = elements.idea.value.trim();
  if (!idea) {
    showToast('Add a thought before saving it');
    return;
  }
  const shelf = getShelf();
  const existingIndex = shelf.findIndex((item) => item.idea === idea);
  const item = {
    id: existingIndex >= 0 ? shelf[existingIndex].id : `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    idea,
    expanded: isCurrentOutput() ? state.expanded : '',
    contexts: isCurrentOutput() ? state.contexts : [],
    mode: state.mode,
    audience: elements.audience.value,
    tone: elements.tone.value,
    savedAt: Date.now(),
  };
  if (existingIndex >= 0) shelf[existingIndex] = item;
  else shelf.unshift(item);
  const saved = safeWriteJSON(SHELF_KEY, shelf.slice(0, 50));
  updateShelfCount();
  showToast(saved ? 'Idea saved to your shelf' : 'This browser could not save the idea');
}

function renderShelf() {
  const items = getShelf();
  updateShelfCount();
  if (!items.length) {
    elements.savedList.innerHTML = '<div class="saved-empty">Your shelf is empty for now.<br />Save an idea from the studio and it will be kept on this device.</div>';
    return;
  }
  elements.savedList.innerHTML = items.map((item) => {
    const date = new Date(item.savedAt || Date.now()).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    return `<article class="saved-item" data-id="${escapeHTML(item.id)}"><button class="saved-item-main" type="button" data-load-saved="${escapeHTML(item.id)}"><strong>${escapeHTML(item.idea)}</strong><small>${escapeHTML(date)} · ${escapeHTML(MODES[item.mode]?.label || 'Idea')}</small></button><button class="delete-saved" type="button" data-delete-saved="${escapeHTML(item.id)}">Remove</button></article>`;
  }).join('');
}

function loadSavedIdea(id) {
  const item = getShelf().find((saved) => saved.id === id);
  if (!item) return;
  elements.idea.value = item.idea;
  state.mode = MODES[item.mode] ? item.mode : 'deepen';
  elements.audience.value = [...elements.audience.options].some((option) => option.value === item.audience) ? item.audience : 'an AI language model';
  elements.tone.value = [...elements.tone.options].some((option) => option.value === item.tone) ? item.tone : 'clear and curious';
  document.querySelectorAll('.direction-option').forEach((button) => {
    const selected = button.dataset.mode === state.mode;
    button.classList.toggle('is-selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  if (item.expanded) {
    state.expanded = item.expanded;
    state.generatedSeed = item.idea;
    state.generatedSettings = getSettings();
    state.contexts = Array.isArray(item.contexts) && item.contexts.length
      ? item.contexts
      : localFallbackContexts(state.mode, []);
    state.outputLabel = 'SAVED IDEA';
    elements.expanded.textContent = item.expanded;
    elements.outputBadge.textContent = 'SAVED IDEA';
    setContexts(state.contexts);
    elements.meaningGroups.innerHTML = '<div class="meaning-empty">Refresh the map to find live word relationships for this saved idea.</div>';
    elements.meaningSource.textContent = 'Refresh the map to explore this idea’s meaning links';
    state.mapSource = 'none';
    state.mapSeed = null;
    state.mapData = [];
  } else {
    makeEmptyResults();
  }
  updateCharCount();
  syncStaleState();
  renderPrompt();
  saveDraftSoon();
  closeDialog(elements.shelfDialog);
  setWorkspaceView('studio', false);
  document.getElementById('main').scrollIntoView({ behavior: 'smooth', block: 'start' });
  showToast('Idea opened in the studio');
}

function setWorkspaceView(view, shouldScroll = true) {
  const showWorkbench = view !== 'studio';
  const workbench = document.getElementById('workbenchMain');
  const studio = document.getElementById('main');
  const workbenchNav = document.getElementById('workbenchNav');
  const studioNav = document.getElementById('studioNav');
  if (workbench) workbench.hidden = !showWorkbench;
  if (studio) studio.hidden = showWorkbench;
  [
    [workbenchNav, showWorkbench],
    [studioNav, !showWorkbench],
  ].forEach(([button, active]) => {
    if (!button) return;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  const breadcrumb = document.getElementById('workspaceBreadcrumb');
  const title = document.getElementById('workspaceTitle');
  if (breadcrumb) breadcrumb.textContent = showWorkbench ? 'WORKBENCH' : 'STUDIO';
  if (title) title.textContent = showWorkbench ? 'Semantic direction' : 'New exploration';
  if (shouldScroll) {
    const target = showWorkbench ? workbench : studio;
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function initialize() {
  restoreDraft();
  setWorkspaceView('workbench', false);
  updateCharCount();
  updateShelfCount();
  renderPrompt();
  updateIncludedCount();
  document.querySelectorAll('.meaning-chip').forEach((chip) => {
    chip.setAttribute('aria-pressed', String(chip.classList.contains('is-included')));
  });
  syncStaleState();

  elements.idea.addEventListener('input', () => {
    updateCharCount();
    syncStaleState();
    renderPrompt();
    saveDraftSoon();
  });

  [elements.audience, elements.tone].forEach((select) => select.addEventListener('change', () => {
    syncStaleState();
    renderPrompt();
    saveDraftSoon();
  }));

  document.querySelectorAll('.direction-option').forEach((button) => {
    button.addEventListener('click', () => setMode(button.dataset.mode));
  });

  document.querySelectorAll('.example-seed').forEach((button) => {
    button.addEventListener('click', () => {
      elements.idea.value = button.dataset.seed || '';
      elements.idea.focus();
      updateCharCount();
      syncStaleState();
      renderPrompt();
      saveDraftSoon();
    });
  });

  elements.generate.addEventListener('click', generateIdea);
  elements.idea.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      generateIdea();
    }
  });

  elements.meaningGroups.addEventListener('click', (event) => {
    const chip = event.target.closest('.meaning-chip');
    if (!chip) return;
    setChipState(chip, !chip.classList.contains('is-included'));
    refreshPromptFromState();
  });

  elements.loadEmbedding.addEventListener('click', async () => {
    if (state.modelReady) {
      if (state.mapSource === 'live') await rankMeaningMap();
      else showToast('Embeddings are ready · generate an idea to rank its links');
      return;
    }
    try {
      await loadEmbeddingModel();
      showToast('On-device embeddings are ready');
    } catch {
      showToast('Could not load the local model · check your connection');
    }
  });

  elements.refreshMap.addEventListener('click', refreshOnlyMap);
  elements.copyExpansion.addEventListener('click', () => copyText(state.expanded, 'Expanded idea copied'));
  elements.copyPrompt.addEventListener('click', () => copyText(createPrompt(), 'LLM prompt copied'));
  elements.togglePrompt.addEventListener('click', () => {
    const collapsed = elements.promptPreview.dataset.collapsed === 'true';
    elements.promptPreview.dataset.collapsed = String(!collapsed);
    elements.togglePrompt.setAttribute('aria-expanded', String(collapsed));
    elements.togglePrompt.innerHTML = collapsed ? 'Show less <span aria-hidden="true">⌃</span>' : 'View full prompt <span aria-hidden="true">⌄</span>';
  });
  elements.saveIdea.addEventListener('click', saveCurrentIdea);

  document.getElementById('helpButton').addEventListener('click', () => openDialog(elements.helpDialog));
  document.getElementById('howItWorksNav').addEventListener('click', () => openDialog(elements.helpDialog));
  document.getElementById('shelfNav').addEventListener('click', () => {
    renderShelf();
    openDialog(elements.shelfDialog);
  });
  document.querySelector('.brand').addEventListener('click', (event) => {
    event.preventDefault();
    setWorkspaceView('workbench');
  });
  document.getElementById('workbenchNav').addEventListener('click', () => setWorkspaceView('workbench'));
  document.getElementById('studioNav').addEventListener('click', () => setWorkspaceView('studio'));
  document.querySelectorAll('[data-close-dialog]').forEach((button) => {
    button.addEventListener('click', () => closeDialog(button.closest('dialog')));
  });

  elements.savedList.addEventListener('click', (event) => {
    const loadButton = event.target.closest('[data-load-saved]');
    if (loadButton) {
      loadSavedIdea(loadButton.dataset.loadSaved);
      return;
    }
    const deleteButton = event.target.closest('[data-delete-saved]');
    if (!deleteButton) return;
    const next = getShelf().filter((item) => item.id !== deleteButton.dataset.deleteSaved);
    safeWriteJSON(SHELF_KEY, next);
    renderShelf();
    showToast('Idea removed from your shelf');
  });

  [elements.helpDialog, elements.shelfDialog].forEach((dialog) => {
    dialog.addEventListener('click', (event) => {
      if (event.target === dialog) closeDialog(dialog);
    });
  });
}

initialize();
