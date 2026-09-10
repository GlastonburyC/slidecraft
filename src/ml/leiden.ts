/**
 * Leiden community detection over a k-nearest-neighbour graph.
 *
 * The point is discovering classes rather than being told them: cluster the
 * patch embeddings, look at what each cluster picked out, and promote the ones
 * that mean something into named classes. It is the same route a single-cell
 * analysis takes — components, neighbour graph, Leiden — for the same reason,
 * which is that nobody knows the categories in advance.
 *
 * NOT YET CORRECT — not wired into the app.
 *
 * On three well-separated synthetic blobs this finds seven communities with
 * modularity 0.553, where the true three-way partition scores 0.667. Local
 * moving is right (it is optimal on two disjoint cliques), and the fault is in
 * the aggregation rounds: they fail to merge the sub-communities that local
 * moving leaves behind, so the result is a local optimum that later rounds
 * never escape. Fixing that is the remaining work.
 *
 * Leiden rather than Louvain. Louvain can leave a community internally
 * disconnected: nodes assigned together that are not reachable within their
 * own community, which is exactly the artefact you would then go and try to
 * interpret. Leiden's refinement step rules that out by construction, and the
 * extra phase is most of the code below.
 */

export interface Graph {
  /** Neighbour indices, concatenated; `offsets[i]` to `offsets[i + 1]`. */
  neighbours: Int32Array;
  weights: Float32Array;
  offsets: Int32Array;
  n: number;
  /** Twice the total edge weight, the normaliser modularity needs. */
  totalWeight: number;
}

/**
 * A symmetric k-nearest-neighbour graph in component space.
 *
 * Symmetrised by union rather than intersection: a patch on the edge of a
 * cluster is often a neighbour of the core without the core naming it back,
 * and dropping those leaves rims of tissue stranded as singletons.
 */
export function knnGraph(points: Float32Array, n: number, dim: number, k = 15): Graph {
  const kk = Math.max(1, Math.min(k, n - 1));
  const pairs = new Map<number, number>();

  const distances = new Float64Array(n);
  const order = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { distances[j] = Infinity; order[j] = j; continue; }
      let sum = 0;
      for (let c = 0; c < dim; c++) {
        const d = points[i * dim + c] - points[j * dim + c];
        sum += d * d;
      }
      distances[j] = sum;
      order[j] = j;
    }

    // Partial selection: only the k smallest matter, so a full sort is waste.
    const idx = Array.from(order).sort((a, b) => distances[a] - distances[b]);
    for (let t = 0; t < kk; t++) {
      const j = idx[t];
      const a = Math.min(i, j);
      const b = Math.max(i, j);
      pairs.set(a * n + b, 1);
    }
  }

  const degree = new Int32Array(n);
  for (const key of pairs.keys()) {
    const a = Math.floor(key / n);
    const b = key % n;
    degree[a]++;
    degree[b]++;
  }

  const offsets = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) offsets[i + 1] = offsets[i] + degree[i];
  const neighbours = new Int32Array(offsets[n]);
  const weights = new Float32Array(offsets[n]);
  const cursor = Int32Array.from(offsets.subarray(0, n));

  for (const key of pairs.keys()) {
    const a = Math.floor(key / n);
    const b = key % n;
    neighbours[cursor[a]] = b;
    weights[cursor[a]++] = 1;
    neighbours[cursor[b]] = a;
    weights[cursor[b]++] = 1;
  }

  let totalWeight = 0;
  for (let i = 0; i < weights.length; i++) totalWeight += weights[i];
  return { neighbours, weights, offsets, n, totalWeight };
}

function strengths(g: Graph): Float64Array {
  const out = new Float64Array(g.n);
  for (let i = 0; i < g.n; i++) {
    for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) out[i] += g.weights[e];
  }
  return out;
}

/** Deterministic shuffle, so the same graph always gives the same partition. */
function shuffled(n: number, seed: number): Int32Array {
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  let s = (seed || 1) >>> 0;
  for (let i = n - 1; i > 0; i--) {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    const j = s % (i + 1);
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  return order;
}

/**
 * Move nodes to the neighbouring community that most improves modularity,
 * repeatedly, until nothing moves. This is the phase Louvain and Leiden share.
 */
export function localMove(
  g: Graph,
  community: Int32Array,
  resolution: number,
  seed: number,
): boolean {
  const strength = strengths(g);
  const communityStrength = new Float64Array(g.n);
  for (let i = 0; i < g.n; i++) communityStrength[community[i]] += strength[i];

  const m2 = g.totalWeight || 1;
  const linksTo = new Map<number, number>();
  let moved = false;

  for (let pass = 0; pass < 30; pass++) {
    let movedThisPass = false;
    for (const i of shuffled(g.n, seed + pass)) {
      const from = community[i];
      linksTo.clear();
      for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) {
        const c = community[g.neighbours[e]];
        linksTo.set(c, (linksTo.get(c) ?? 0) + g.weights[e]);
      }

      communityStrength[from] -= strength[i];
      let best = from;
      let bestGain = (linksTo.get(from) ?? 0) - (resolution * communityStrength[from] * strength[i]) / m2;

      for (const [c, w] of linksTo) {
        if (c === from) continue;
        const gain = w - (resolution * communityStrength[c] * strength[i]) / m2;
        if (gain > bestGain + 1e-12) {
          bestGain = gain;
          best = c;
        }
      }

      communityStrength[best] += strength[i];
      if (best !== from) {
        community[i] = best;
        movedThisPass = true;
        moved = true;
      }
    }
    if (!movedThisPass) break;
  }
  return moved;
}

/**
 * Split each community into well-connected parts.
 *
 * The step that makes this Leiden. Every node starts alone, and merges only
 * into a part it is sufficiently connected to *within its own community* — so
 * a community that was really two things joined through a third community
 * comes apart here instead of surviving as one badly-connected blob.
 */
function refine(
  g: Graph,
  community: Int32Array,
  resolution: number,
  seed: number,
): Int32Array {
  const refined = new Int32Array(g.n);
  for (let i = 0; i < g.n; i++) refined[i] = i;

  const strength = strengths(g);
  const partStrength = Float64Array.from(strength);
  const m2 = g.totalWeight || 1;
  const linksTo = new Map<number, number>();

  for (const i of shuffled(g.n, seed)) {
    linksTo.clear();
    let own = 0;
    for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) {
      const j = g.neighbours[e];
      // Only within the community this node already belongs to.
      if (community[j] !== community[i]) continue;
      linksTo.set(refined[j], (linksTo.get(refined[j]) ?? 0) + g.weights[e]);
      own += g.weights[e];
    }
    if (own === 0) continue;

    partStrength[refined[i]] -= strength[i];
    let best = refined[i];
    let bestGain = 0;
    for (const [part, w] of linksTo) {
      if (part === refined[i]) continue;
      const gain = w - (resolution * partStrength[part] * strength[i]) / m2;
      if (gain > bestGain + 1e-12) {
        bestGain = gain;
        best = part;
      }
    }
    refined[i] = best;
    partStrength[best] += strength[i];
  }
  return refined;
}

/** Collapse a partition into a smaller graph whose nodes are its communities. */
function aggregate(g: Graph, partition: Int32Array): { graph: Graph; map: Int32Array } {
  const relabel = new Map<number, number>();
  const map = new Int32Array(g.n);
  for (let i = 0; i < g.n; i++) {
    let id = relabel.get(partition[i]);
    if (id === undefined) {
      id = relabel.size;
      relabel.set(partition[i], id);
    }
    map[i] = id;
  }

  const n = relabel.size;
  const edges = new Map<number, number>();
  for (let i = 0; i < g.n; i++) {
    const a = map[i];
    for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) {
      const b = map[g.neighbours[e]];
      if (a > b) continue; // each undirected edge once
      const key = a * n + b;
      edges.set(key, (edges.get(key) ?? 0) + g.weights[e]);
    }
  }

  const degree = new Int32Array(n);
  for (const key of edges.keys()) {
    const a = Math.floor(key / n);
    const b = key % n;
    degree[a]++;
    if (b !== a) degree[b]++;
  }
  const offsets = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) offsets[i + 1] = offsets[i] + degree[i];
  const neighbours = new Int32Array(offsets[n]);
  const weights = new Float32Array(offsets[n]);
  const cursor = Int32Array.from(offsets.subarray(0, n));

  let totalWeight = 0;
  for (const [key, w] of edges) {
    const a = Math.floor(key / n);
    const b = key % n;
    if (b === a) {
      /**
       * A self-loop counts twice toward a node's degree.
       *
       * When a community collapses to a node, its internal edges become that
       * node's self-loop — and an undirected edge contributes to the degree
       * from both ends. Storing it once understates the strength of exactly
       * the communities that are most cohesive, so the modularity gain from
       * keeping them together is undervalued and the algorithm splits things
       * that belong together.
       */
      neighbours[cursor[a]] = a;
      weights[cursor[a]++] = 2 * w;
      totalWeight += 2 * w;
      continue;
    }
    neighbours[cursor[a]] = b;
    weights[cursor[a]++] = w;
    neighbours[cursor[b]] = a;
    weights[cursor[b]++] = w;
    totalWeight += 2 * w;
  }

  return { graph: { neighbours, weights, offsets, n, totalWeight }, map };
}

export interface Clustering {
  /** Cluster index per point, numbered by size so 0 is the largest. */
  labels: Int32Array;
  clusters: number;
  sizes: number[];
  modularity: number;
}

export function modularityOf(g: Graph, labels: Int32Array, resolution = 1): number {
  const m2 = g.totalWeight || 1;
  const strength = strengths(g);
  const inside = new Map<number, number>();
  const total = new Map<number, number>();

  for (let i = 0; i < g.n; i++) {
    total.set(labels[i], (total.get(labels[i]) ?? 0) + strength[i]);
    for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) {
      if (labels[g.neighbours[e]] === labels[i]) {
        inside.set(labels[i], (inside.get(labels[i]) ?? 0) + g.weights[e]);
      }
    }
  }

  let q = 0;
  for (const w of inside.values()) q += w / m2;
  for (const s of total.values()) q -= (resolution * (s / m2) * s) / m2;
  return q;
}

/**
 * `resolution` sets how fine the partition is: higher splits more. It is the
 * one parameter worth exposing, because the right number of clusters is a
 * judgement about the tissue rather than something the data settles.
 */
export function leiden(g: Graph, resolution = 1, seed = 1): Clustering {
  let level: Graph = g;
  let community = new Int32Array(g.n);
  for (let i = 0; i < g.n; i++) community[i] = i;
  // Maps every original node to its community at the current level.
  let mapping = Int32Array.from(community);

  for (let round = 0; round < 12; round++) {
    const moved = localMove(level, community, resolution, seed + round);
    const refined = refine(level, community, resolution, seed + round);

    const { graph, map } = aggregate(level, refined);
    if (graph.n === level.n) {
      // Nothing collapsed: the partition has stopped changing.
      if (!moved) break;
    }

    // Carry the original nodes forward through this level's relabelling.
    const next = new Int32Array(g.n);
    for (let i = 0; i < g.n; i++) next[i] = map[mapping[i]];
    mapping = next;

    // The aggregated graph starts from the *unrefined* communities, which is
    // what lets Leiden keep the coarse structure while having split it.
    const seeded = new Int32Array(graph.n);
    for (let i = 0; i < level.n; i++) seeded[map[i]] = community[i];
    const relabelled = new Map<number, number>();
    for (let i = 0; i < graph.n; i++) {
      let id = relabelled.get(seeded[i]);
      if (id === undefined) {
        id = relabelled.size;
        relabelled.set(seeded[i], id);
      }
      seeded[i] = id;
    }

    level = graph;
    community = seeded;
    if (graph.n <= 1) break;
  }

  // Number clusters by size, so "cluster 0" is always the biggest.
  const finalLabels = new Int32Array(g.n);
  for (let i = 0; i < g.n; i++) finalLabels[i] = community[mapping[i]];
  const counts = new Map<number, number>();
  for (let i = 0; i < g.n; i++) counts.set(finalLabels[i], (counts.get(finalLabels[i]) ?? 0) + 1);
  const bySize = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const rank = new Map(bySize.map(([id], i) => [id, i]));
  for (let i = 0; i < g.n; i++) finalLabels[i] = rank.get(finalLabels[i])!;

  return {
    labels: finalLabels,
    clusters: bySize.length,
    sizes: bySize.map(([, size]) => size),
    modularity: modularityOf(g, finalLabels, resolution),
  };
}
