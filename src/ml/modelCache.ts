/**
 * Downloads model weights once and keeps them in the Cache Storage API.
 *
 * Weights are tens of megabytes and immutable, so re-fetching them on every
 * reload would be the slowest part of the app. Cache Storage handles bodies of
 * this size far better than IndexedDB and survives reloads.
 */

const CACHE_NAME = "slidecraft-models-v1";
const TOKEN_KEY = "slidecraft.hfToken";

/**
 * Hugging Face access token, held only in this browser.
 *
 * Gated repos (UNI, Virchow2, CONCH, GigaPath) need `Authorization: Bearer` to
 * fetch at all. It lives in localStorage rather than anywhere in the project,
 * so it cannot be committed, and it is sent only to huggingface.co — never
 * attached to whatever arbitrary URL a model spec happens to carry.
 */
export function getHfToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setHfToken(token: string | null): void {
  try {
    if (token && token.trim()) localStorage.setItem(TOKEN_KEY, token.trim());
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage blocked; the session still works, it just will not persist */
  }
}

/** Only huggingface.co gets the token, whatever a spec's URL says. */
function authHeaders(url: string): HeadersInit | undefined {
  const token = getHfToken();
  if (!token) return undefined;
  try {
    const host = new URL(url).hostname;
    if (host !== "huggingface.co" && !host.endsWith(".huggingface.co")) return undefined;
  } catch {
    return undefined;
  }
  return { Authorization: `Bearer ${token}` };
}

export interface DownloadProgress {
  part: string;
  received: number;
  total: number;
}

async function cache(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // private mode, or storage denied
  }
}

/** True when every URL is already cached, so the UI can say "ready" honestly. */
export async function isCached(urls: string[]): Promise<boolean> {
  const c = await cache();
  if (!c) return false;
  const hits = await Promise.all(urls.map((u) => c.match(u)));
  return hits.every(Boolean);
}

/**
 * Fetch with progress, serving from cache when present. Returns the raw bytes,
 * which is what onnxruntime-web wants for `InferenceSession.create`.
 */
export async function fetchWeights(
  url: string,
  part: string,
  expectedBytes: number,
  onProgress?: (p: DownloadProgress) => void,
): Promise<Uint8Array> {
  const c = await cache();

  const cached = c ? await c.match(url) : undefined;
  if (cached) {
    onProgress?.({ part, received: expectedBytes, total: expectedBytes });
    return new Uint8Array(await cached.arrayBuffer());
  }

  if (url.startsWith("slidecraft-local:")) {
    // Imported weights only ever live in the cache; there is nothing to fetch.
    throw new Error(
      `The imported weights for "${part}" are no longer in local storage. ` +
        `Import the .onnx file again.`,
    );
  }

  const res = await fetch(url, { headers: authHeaders(url) });
  if (res.status === 401 || res.status === 403) {
    throw new Error(
      `Access denied for ${part}. This model is gated: accept its terms on Hugging Face, ` +
        `then add your access token in the model menu.`,
    );
  }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} fetching ${part}`);

  const total = Number(res.headers.get("content-length")) || expectedBytes;
  const reader = res.body?.getReader();

  let bytes: Uint8Array;
  if (!reader) {
    bytes = new Uint8Array(await res.arrayBuffer());
  } else {
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.({ part, received, total });
    }
    bytes = new Uint8Array(received);
    let at = 0;
    for (const ch of chunks) { bytes.set(ch, at); at += ch.length; }
  }

  // Store a copy; a consumed Response cannot be cached.
  if (c) {
    try {
      await c.put(url, new Response(bytes.slice().buffer, {
        headers: { "content-type": "application/octet-stream", "content-length": String(bytes.length) },
      }));
    } catch { /* over quota: still usable this session */ }
  }
  return bytes;
}

/**
 * Ask the browser not to evict our cache under storage pressure. Model weights
 * are tens of megabytes and expensive to refetch; without this the browser is
 * free to drop them at any time.
 */
export async function requestPersistence(): Promise<boolean> {
  try {
    if (!navigator.storage?.persist) return false;
    if (await navigator.storage.persisted()) return true;
    return await navigator.storage.persist();
  } catch {
    return false;
  }
}

export async function cacheUsage(): Promise<{ usage: number; quota: number } | null> {
  try {
    const est = await navigator.storage?.estimate?.();
    return est ? { usage: est.usage ?? 0, quota: est.quota ?? 0 } : null;
  } catch {
    return null;
  }
}

export async function clearModelCache(): Promise<void> {
  try { await caches.delete(CACHE_NAME); } catch { /* nothing to do */ }
}
