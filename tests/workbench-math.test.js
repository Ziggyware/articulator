const test = require('node:test');
const assert = require('node:assert/strict');
const {
  poolEmbeddingMatrix,
  normalizeVector,
  meanVectors,
  combineDirections,
  cosine,
  rankCandidates,
} = require('../workbench.js');

test('pools a 2D embedding matrix along its row axis', () => {
  const pooled = poolEmbeddingMatrix([[1, 2], [3, 4], [5, 6]], { normalize: false });
  assert.deepEqual(Array.from(pooled), [3, 4]);
});

test('pools one batch of a 3D token embedding and respects the token mask', () => {
  const tensor = {
    dims: Int32Array.from([2, 3, 2]),
    data: Float32Array.from([1, 2, 4, 6, 100, 200, 2, 4, 4, 8, 10, 20]),
  };
  const pooled = poolEmbeddingMatrix(tensor, { batchIndex: 1, mask: [1, 1, 0], normalize: false });
  assert.deepEqual(Array.from(pooled), [3, 6]);
});

test('supports weighted pooling and unit normalization', () => {
  const pooled = poolEmbeddingMatrix([[2, 0], [0, 4]], { weights: [3, 1] });
  assert.ok(Math.abs(pooled[0] - 0.8320503) < 1e-6);
  assert.ok(Math.abs(pooled[1] - 0.5547002) < 1e-6);
  assert.ok(Math.abs(Math.hypot(...pooled) - 1) < 1e-6);
});

test('mean and pooled direction helpers normalize explicitly', () => {
  const average = meanVectors([[1, 0], [0, 1]], { normalize: true });
  assert.ok(Math.abs(average[0] - Math.SQRT1_2) < 1e-6);
  const direction = combineDirections([[8, 0], [0, 3]]);
  assert.ok(Math.abs(direction[0] - Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(direction[1] - Math.SQRT1_2) < 1e-6);
});

test('rejects malformed and zero vectors instead of silently flattening them', () => {
  assert.throws(() => normalizeVector([0, 0]), /zero-length/);
  assert.throws(() => poolEmbeddingMatrix([[1, 2], [3]], { normalize: false }), /inconsistent dimensions/);
  assert.throws(() => poolEmbeddingMatrix([[1, 2], [3, 4]], { axis: 7 }), /axis/);
});

test('filters a dictionary against a direction with exclusions and predicates', () => {
  const dictionary = [
    { word: 'toward', vector: [1, 0], source: 'input' },
    { word: 'north', vector: [0, 1], kind: 'lexical' },
    { word: 'bridge', vector: [0.8, 0.2], kind: 'lexical' },
    { word: 'away', vector: [-1, 0], kind: 'lexical' },
    { word: 'speculation', vector: [0.99, 0.01], kind: 'ai' },
  ];
  const ranked = rankCandidates(dictionary, [1, 0], {
    topK: 5,
    minScore: 0,
    excludeTerms: ['toward'],
    filter: (entry) => entry.kind === 'lexical',
  });
  assert.deepEqual(ranked.map((entry) => entry.word), ['bridge', 'north']);
  assert.ok(ranked[0].score > ranked[1].score);
  assert.ok(Math.abs(cosine([1, 0], ranked[0].vector) - ranked[0].score) < 1e-7);
  assert.equal(rankCandidates(new Set([{ text: 'east', vector: [1, 0] }]), [1, 0])[0].word, 'east');
  assert.throws(() => rankCandidates([{ word: 'bad dimension', vector: [1] }], [1, 0]), /dimension/);
  assert.throws(() => rankCandidates([], [0, 0]), /zero vector/);
});
