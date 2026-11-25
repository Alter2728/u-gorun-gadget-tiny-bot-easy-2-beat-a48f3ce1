// Tiny regime-adaptive long-only bot for the bars-replay protocol.
// It reads a hello line, then answers every batch with one reply of orders
// decided from that batch's bar closes; orders fill at the next bar open.
import { createInterface } from 'node:readline';

type SymInfo = { symbol: string; price_scale: number; qty_scale: number };
type BarMsg = { symbol: string; c: string; c_depth: string[] };
type FillMsg = { client_id: string; symbol: string; side: string; status: string; fill_bar: number };
type PosMsg = { symbol: string; qty: string; cost_quote_units: string };

let symbols: SymInfo[] = [];
let quoteScale = 6;
let warmup = 0;
let maxWeightBps = 2500;
let maxOrders = 16;
let minNotional = 1000000n;

const closes = new Map<string, number[]>();
const depthUsdc = new Map<string, number[]>();
const meta = new Map<string, { entrySeq: number; peak: number; kind: string }>();
const pendingKind = new Map<string, string>();
const cooldownUntil = new Map<string, number>();
let trades = 0;
let orderSeq = 0;

function px(sym: SymInfo, raw: string): number {
  return Number(BigInt(raw)) / 10 ** sym.price_scale;
}
function ema(vals: number[], span: number): number {
  const k = 2 / (span + 1);
  let e = vals[0];
  for (let i = 1; i < vals.length; i++) e = vals[i] * k + e * (1 - k);
  return e;
}
function mom(vals: number[], k: number): number {
  if (vals.length <= k) return 0;
  return Math.log(vals[vals.length - 1] / vals[vals.length - 1 - k]);
}
function vol24(vals: number[]): number {
  const n = Math.min(24, vals.length - 1);
  if (n < 3) return 0.01;
  const rs: number[] = [];
  for (let i = vals.length - n; i < vals.length; i++) rs.push(Math.log(vals[i] / vals[i - 1]));
  const m = rs.reduce((a, b) => a + b, 0) / rs.length;
  return Math.sqrt(rs.reduce((a, b) => a + (b - m) * (b - m), 0) / rs.length);
}
function rsi14(vals: number[]): number {
  if (vals.length < 15) return 50;
  let g = 0;
  let l = 0;
  for (let i = vals.length - 14; i < vals.length; i++) {
    const d = vals[i] - vals[i - 1];
    if (d > 0) g += d; else l -= d;
  }
  if (l === 0) return 100;
  return 100 - 100 / (1 + g / l);
}
function impactBps(sym: string, amountUsdc: number): number {
  const d = depthUsdc.get(sym);
  if (!d || d.length === 0) return 50;
  let s = 0;
  for (const x of d) if (x > 0) s += amountUsdc / x;
  return s * 10000;
}
function slipCap(sym: string, amountUsdc: number): number {
  return Math.min(1000, Math.max(250, Math.round(350 + impactBps(sym, amountUsdc))));
}
function nextId(): string {
  orderSeq += 1;
  return 'o' + orderSeq;
}

type Order =
  | { side: 'buy'; client_id: string; symbol: string; notional_quote_units: string; max_slippage_bps: number }
  | { side: 'sell'; client_id: string; symbol: string; qty: string; max_slippage_bps: number };

function decide(seq: number, cash: bigint, equity: bigint, positions: PosMsg[]): Order[] {
  if (seq < warmup) return [];
  const eq = Number(equity) / 10 ** quoteScale;
  const cashUsdc = Number(cash) / 10 ** quoteScale;
  const held = new Map<string, { qty: bigint; cost: bigint }>();
  for (const p of positions) held.set(p.symbol, { qty: BigInt(p.qty), cost: BigInt(p.cost_quote_units) });

  type Info = {
    sym: SymInfo; last: number; ef: number; es: number; trend: number; dip: number;
    m6: number; m12: number; m24: number; vol: number; rsi: number; up: boolean;
    upbar: boolean; score: number; n: number;
  };
  const infos: Info[] = [];
  for (const sym of symbols) {
    const vals = closes.get(sym.symbol) ?? [];
    if (vals.length < 16) continue;
    const last = vals[vals.length - 1];
    const ef = ema(vals, 8);
    const es = ema(vals, 24);
    const trend = ef / es - 1;
    const vol = vol24(vals);
    const m12 = mom(vals, 12);
    const m24 = mom(vals, 24);
    const score = (trend + 0.6 * m12 + 0.5 * m24) / (vol * 4 + 0.004);
    infos.push({
      sym, last, ef, es, trend, dip: last / ef - 1,
      m6: mom(vals, 6), m12, m24, vol, rsi: rsi14(vals),
      up: last > es && trend > 0, upbar: last > vals[vals.length - 2], score, n: vals.length,
    });
  }
  if (infos.length === 0) return [];
  const bySym = new Map(infos.map((i) => [i.sym.symbol, i]));

  for (const [s, m] of meta) {
    const inf = bySym.get(s);
    if (inf && inf.last > m.peak) m.peak = inf.last;
  }

  // Exits first: when any sell goes out, buys wait for the next batch so the
  // sell proceeds are actually in cash before they fund a rotation.
  const sells: Order[] = [];
  for (const [s, h] of held) {
    const inf = bySym.get(s);
    if (!inf) continue;
    const m = meta.get(s) ?? { entrySeq: seq, peak: inf.last, kind: 'tr' };
    const heldBars = seq - m.entrySeq;
    const value = (Number(h.qty) / 10 ** inf.sym.qty_scale) * inf.last;
    const cost = Number(h.cost) / 10 ** quoteScale;
    const pnl = cost > 0 ? value / cost - 1 : 0;
    let exit = false;
    if (m.kind === 'mr') {
      if (pnl < -0.02 || pnl > 0.03 || heldBars >= 18 || (heldBars >= 2 && inf.rsi > 62)) exit = true;
    } else {
      const trail = Math.min(0.06, Math.max(0.022, 1.4 * inf.vol * Math.sqrt(Math.max(1, heldBars))));
      if (pnl < -0.035) exit = true;
      if (heldBars >= 2 && inf.last < m.peak * (1 - trail)) exit = true;
      if (heldBars >= 1 && inf.trend < 0) exit = true;
      if (heldBars >= 30 && pnl < 0.005) exit = true;
    }
    if (exit) {
      sells.push({ side: 'sell', client_id: nextId(), symbol: s, qty: h.qty.toString(), max_slippage_bps: slipCap(s, value) });
      cooldownUntil.set(s, seq + 6);
    }
  }
  if (sells.length > 0) return sells.slice(0, maxOrders);

  const m24s = infos.map((i) => i.m24).sort((a, b) => a - b);
  const med24 = m24s[Math.floor(m24s.length / 2)];
  const gross = med24 > 0.005 ? 0.85 : med24 > -0.02 ? 0.6 : 0.32;

  const tradedBars = seq - warmup;
  let thr = 0.3;
  let rsiMr = 32;
  if (tradedBars > 24 && trades < 6) { thr *= 0.4; rsiMr = 40; }
  const force = tradedBars > 40 && trades < 9;

  const cands: { inf: Info; kind: string }[] = [];
  for (const inf of infos) {
    const s = inf.sym.symbol;
    if (held.has(s)) continue;
    if ((cooldownUntil.get(s) ?? -1) > seq) continue;
    let kind: string | null = null;
    if (inf.n >= 25 && inf.up && inf.m24 > 0 && inf.score > thr && ((inf.dip < -0.004 && inf.upbar) || inf.score > 3.0)) kind = 'tr';
    else if (inf.rsi < rsiMr && inf.upbar && inf.m6 < -0.01) kind = 'mr';
    else if (force && inf.m12 > -0.02) kind = 'mr';
    if (kind) cands.push({ inf, kind });
  }
  cands.sort((a, b) => (b.inf.score - a.inf.score) || (a.inf.sym.symbol < b.inf.sym.symbol ? -1 : 1));

  const weights = [0.24, 0.2, 0.15, 0.11];
  let heldValue = 0;
  for (const [s, h] of held) {
    const inf = bySym.get(s);
    if (inf) heldValue += (Number(h.qty) / 10 ** inf.sym.qty_scale) * inf.last;
  }
  let room = gross * eq - heldValue;
  let cashLeft = cashUsdc * 0.98;
  const buys: Order[] = [];
  const slots = Math.max(0, 4 - held.size);
  for (let r = 0; r < Math.min(slots, cands.length); r++) {
    const { inf, kind } = cands[r];
    let w = weights[Math.min(r, weights.length - 1)];
    if (kind === 'mr') w = Math.min(w, 0.1);
    const weightRoom = (maxWeightBps / 10000) * eq;
    const amt = Math.min(w * eq, room, cashLeft, weightRoom);
    const units = BigInt(Math.floor(amt * 10 ** quoteScale));
    if (units < minNotional || amt < 50) continue;
    const id = nextId();
    pendingKind.set(id, kind);
    buys.push({ side: 'buy', client_id: id, symbol: inf.sym.symbol, notional_quote_units: units.toString(), max_slippage_bps: slipCap(inf.sym.symbol, amt) });
    room -= amt;
    cashLeft -= amt;
  }
  return buys.slice(0, maxOrders);
}

const rl = createInterface({ input: process.stdin });
rl.on('line', (line: string) => {
  if (!line.trim()) return;
  const msg = JSON.parse(line);
  if (msg.type === 'hello') {
    symbols = msg.symbols;
    quoteScale = msg.capital.quote_scale;
    warmup = msg.warmup_bars ?? 0;
    maxWeightBps = msg.caps.max_weight_bps;
    maxOrders = msg.caps.max_orders_per_bar;
    minNotional = BigInt(msg.fees.min_notional_quote_units);
    return;
  }
  if (msg.type === 'batch') {
    for (const f of msg.fills as FillMsg[]) {
      if (f.status !== 'filled') continue;
      trades += 1;
      if (f.side === 'buy') {
        const vals = closes.get(f.symbol) ?? [];
        meta.set(f.symbol, { entrySeq: f.fill_bar, peak: vals.length ? vals[vals.length - 1] : 0, kind: pendingKind.get(f.client_id) ?? 'tr' });
      } else {
        meta.delete(f.symbol);
      }
    }
    for (const b of msg.bars as BarMsg[]) {
      const sym = symbols.find((s) => s.symbol === b.symbol);
      if (!sym) continue;
      const arr = closes.get(b.symbol) ?? [];
      arr.push(px(sym, b.c));
      if (arr.length > 400) arr.splice(0, arr.length - 400);
      closes.set(b.symbol, arr);
      depthUsdc.set(b.symbol, b.c_depth.map((d) => Number(BigInt(d)) / 1e6));
    }
    const orders = decide(msg.seq, BigInt(msg.account.cash_quote_units), BigInt(msg.account.equity_quote_units), msg.account.positions);
    process.stdout.write(JSON.stringify({ type: 'reply', seq: msg.seq, orders }) + '\n');
    return;
  }
});
