// Minimal ZIP reader/writer, "stored" entries only (no compression).
// Enough for records: standard tools (unzip, Python's zipfile, Finder,
// Explorer) open the files, and we can read back what we wrote.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

const DOS_DATE = (0 << 9) | (1 << 5) | 1; // 1980-01-01 in MS-DOS date format

export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const asBytes = (data) =>
  typeof data === 'string' ? new TextEncoder().encode(data)
    : data instanceof Uint8Array ? data
      : ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : new Uint8Array(data);

/** @param entries [{name, data: string | ArrayBuffer | TypedArray}] -> Uint8Array */
export function writeZip(entries) {
  const encoder = new TextEncoder();
  const files = entries.map(({ name, data }) => {
    const bytes = asBytes(data);
    return { name: encoder.encode(name), bytes, crc: crc32(bytes) };
  });
  const localSize = files.reduce((s, f) => s + 30 + f.name.length + f.bytes.length, 0);
  const centralSize = files.reduce((s, f) => s + 46 + f.name.length, 0);
  if (localSize + centralSize > 0xffffffff || files.length > 0xffff) throw new Error('record too large for a non-ZIP64 archive');

  const out = new Uint8Array(localSize + centralSize + 22);
  const view = new DataView(out.buffer);
  let p = 0;
  const offsets = [];
  for (const f of files) {
    offsets.push(p);
    view.setUint32(p, 0x04034b50, true);
    view.setUint16(p + 4, 20, true);          // version needed
    view.setUint16(p + 6, 0x0800, true);      // flags: UTF-8 names
    view.setUint16(p + 8, 0, true);           // method: stored
    view.setUint16(p + 10, 0, true);          // time 00:00
    view.setUint16(p + 12, DOS_DATE, true);   // 1980-01-01; records carry their own timestamps
    view.setUint32(p + 14, f.crc, true);
    view.setUint32(p + 18, f.bytes.length, true);
    view.setUint32(p + 22, f.bytes.length, true);
    view.setUint16(p + 26, f.name.length, true);
    view.setUint16(p + 28, 0, true);
    out.set(f.name, p + 30);
    out.set(f.bytes, p + 30 + f.name.length);
    p += 30 + f.name.length + f.bytes.length;
  }
  const centralStart = p;
  files.forEach((f, i) => {
    view.setUint32(p, 0x02014b50, true);
    view.setUint16(p + 4, 20, true);
    view.setUint16(p + 6, 20, true);
    view.setUint16(p + 8, 0x0800, true);
    view.setUint16(p + 10, 0, true);
    view.setUint16(p + 12, 0, true);
    view.setUint16(p + 14, DOS_DATE, true);
    view.setUint32(p + 16, f.crc, true);
    view.setUint32(p + 20, f.bytes.length, true);
    view.setUint32(p + 24, f.bytes.length, true);
    view.setUint16(p + 28, f.name.length, true);
    // extra, comment, disk, internal attrs: zero
    view.setUint32(p + 38, 0, true);
    view.setUint32(p + 42, offsets[i], true);
    out.set(f.name, p + 46);
    p += 46 + f.name.length;
  });
  view.setUint32(p, 0x06054b50, true);
  view.setUint16(p + 8, files.length, true);
  view.setUint16(p + 10, files.length, true);
  view.setUint32(p + 12, p - centralStart, true);
  view.setUint32(p + 16, centralStart, true);
  return out;
}

/** Read a stored-only ZIP. Returns Map(name -> Uint8Array view). */
export function readZip(input) {
  const bytes = asBytes(input);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = bytes.length - 22;
  while (eocd >= 0 && view.getUint32(eocd, true) !== 0x06054b50) eocd--;
  if (eocd < 0) throw new Error('not a ZIP file');
  const count = view.getUint16(eocd + 10, true);
  let p = view.getUint32(eocd + 16, true);
  const decoder = new TextDecoder();
  const files = new Map();
  for (let i = 0; i < count; i++) {
    if (view.getUint32(p, true) !== 0x02014b50) throw new Error('corrupt ZIP central directory');
    const method = view.getUint16(p + 10, true);
    const size = view.getUint32(p + 20, true);
    const nameLen = view.getUint16(p + 28, true);
    const extraLen = view.getUint16(p + 30, true);
    const commentLen = view.getUint16(p + 32, true);
    const local = view.getUint32(p + 42, true);
    const name = decoder.decode(bytes.subarray(p + 46, p + 46 + nameLen));
    if (method !== 0) throw new Error(`${name}: compressed entries are not supported`);
    const dataStart = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    files.set(name, bytes.subarray(dataStart, dataStart + size));
    p += 46 + nameLen + extraLen + commentLen;
  }
  return files;
}
