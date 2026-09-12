/**
 * A minimal ZIP writer: stored entries, no compression.
 *
 * Here rather than as a dependency because what it has to do is small and
 * completely specified, and because the alternative pulls a compression library
 * into a bundle that already ships 37 MB of WebAssembly. Stored entries are the
 * right choice anyway — the payload is float arrays, which deflate barely
 * shrinks, and a zarr store read straight out of the archive wants its members
 * addressable rather than packed.
 *
 * Zip64 is not implemented. Anything approaching 4 GB has no business being
 * assembled in a browser tab, and the caller checks the size before it gets
 * here rather than producing an archive that is quietly wrong past the limit.
 */

export interface ZipEntry {
  name: string;
  data: Uint8Array;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** Total bytes an archive of these entries will occupy. */
export function zipSize(entries: ZipEntry[]): number {
  let total = 0;
  for (const e of entries) {
    const n = new TextEncoder().encode(e.name).length;
    total += 30 + n + e.data.length; // local header + name + payload
    total += 46 + n;                 // central directory record
  }
  return total + 22;                 // end of central directory
}

export function makeZip(entries: ZipEntry[]): Uint8Array {
  const enc = new TextEncoder();
  const named = entries.map((e) => ({
    name: enc.encode(e.name),
    data: e.data,
    crc: crc32(e.data),
  }));

  const out = new Uint8Array(zipSize(entries));
  const view = new DataView(out.buffer);
  let at = 0;
  const offsets: number[] = [];

  const u32 = (v: number) => { view.setUint32(at, v >>> 0, true); at += 4; };
  const u16 = (v: number) => { view.setUint16(at, v & 0xffff, true); at += 2; };

  for (const e of named) {
    offsets.push(at);
    u32(0x04034b50);            // local file header
    u16(20);                    // version needed
    u16(0);                     // flags
    u16(0);                     // method: stored
    u16(0); u16(0);             // mod time, mod date — fixed, so the output of
                                // two identical exports is byte-identical
    u32(e.crc);
    u32(e.data.length);         // compressed size
    u32(e.data.length);         // uncompressed size
    u16(e.name.length);
    u16(0);                     // extra length
    out.set(e.name, at); at += e.name.length;
    out.set(e.data, at); at += e.data.length;
  }

  const dirStart = at;
  named.forEach((e, i) => {
    u32(0x02014b50);            // central directory header
    u16(20); u16(20); u16(0); u16(0);
    u16(0); u16(0);
    u32(e.crc);
    u32(e.data.length);
    u32(e.data.length);
    u16(e.name.length);
    u16(0); u16(0); u16(0); u16(0);
    u32(0);                     // external attributes
    u32(offsets[i]);
    out.set(e.name, at); at += e.name.length;
  });

  // Measured before the record is written, not during it: `at` advances as the
  // fields go down, so reading it inline reports the directory twelve bytes
  // longer than it is. Readers that trust the count then walk off the end.
  const dirSize = at - dirStart;

  u32(0x06054b50);              // end of central directory
  u16(0); u16(0);
  u16(named.length); u16(named.length);
  u32(dirSize);
  u32(dirStart);
  u16(0);
  return out;
}
