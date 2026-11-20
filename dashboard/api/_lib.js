// Same pipeline as src/features.py + src/predict.py, in plain JavaScript.
const model = require("./_model.json");

// Settings come from Vercel environment variables (with the project defaults).
function settings() {
  const num = (v, d) => (v !== undefined && v !== "" && !Number.isNaN(Number(v)) ? Number(v) : d);
  return {
    low: num(process.env.LOW_LIMIT, 0.3),
    high: num(process.env.HIGH_LIMIT, 0.7),
    maxAmount: num(process.env.MAX_AMOUNT, 1000000),
    modelName: process.env.MODEL_NAME || "Random Forest",
    appEnv: process.env.APP_ENV || "production",
  };
}

const TYPES = ["online", "pos", "atm"];

function validate(body, cfg) {
  const t = body || {};
  const errors = [];
  const amount = Number(t.amount);
  const hour = Number(t.hour);
  const count = Number(t.txn_count_1h);
  const avg = Number(t.avg_spending);
  const loc = Number(t.new_location);
  if (!(amount > 0 && amount <= cfg.maxAmount)) errors.push(`amount must be between 1 and ${cfg.maxAmount}`);
  if (!(Number.isInteger(hour) && hour >= 0 && hour <= 23)) errors.push("hour must be a whole number from 0 to 23");
  if (!(Number.isInteger(count) && count >= 1 && count <= 50)) errors.push("txn_count_1h must be a whole number from 1 to 50");
  if (!(avg > 0 && avg <= cfg.maxAmount)) errors.push(`avg_spending must be between 1 and ${cfg.maxAmount}`);
  if (!(loc === 0 || loc === 1)) errors.push("new_location must be 0 or 1");
  if (!TYPES.includes(t.txn_type)) errors.push("txn_type must be online, pos or atm");
  return { errors, txn: { amount, hour, txn_count_1h: count, avg_spending: avg, new_location: loc, txn_type: t.txn_type } };
}

// features.py: amount_ratio, is_night, one-hot txn_type
function buildFeatures(t) {
  return {
    amount: t.amount,
    hour: t.hour,
    txn_count_1h: t.txn_count_1h,
    avg_spending: t.avg_spending,
    new_location: t.new_location,
    amount_ratio: Math.round((t.amount / t.avg_spending) * 1000) / 1000,
    is_night: t.hour >= 23 || t.hour <= 4 ? 1 : 0,
    type_atm: t.txn_type === "atm" ? 1 : 0,
    type_online: t.txn_type === "online" ? 1 : 0,
    type_pos: t.txn_type === "pos" ? 1 : 0,
  };
}

function scaled(features) {
  const x = model.features.map((name) => {
    const s = model.scale[name];
    const v = s ? (features[name] - s.mean) / s.std : features[name];
    return Math.fround(v); // scikit-learn compares float32 values
  });
  return x;
}

function walk(tree, x) {
  let n = 0;
  while (tree.f[n] !== -1) n = x[tree.f[n]] <= tree.t[n] ? tree.l[n] : tree.r[n];
  return tree.t[n];
}

function fraudProbability(txn) {
  const x = scaled(buildFeatures(txn));
  let sum = 0;
  for (const tree of model.trees) sum += walk(tree, x);
  return sum / model.trees.length;
}

function riskLevel(p, cfg) {
  if (p < cfg.low) return "Low";
  if (p < cfg.high) return "Medium";
  return "High";
}

// Plain-language signals, the same patterns the data generator uses for fraud.
function signals(txn) {
  const f = buildFeatures(txn);
  const out = [];
  if (f.amount_ratio >= 5) out.push({ tone: "bad", text: `Amount is ${f.amount_ratio.toFixed(1)}× the card's usual spending` });
  else if (f.amount_ratio >= 2) out.push({ tone: "warn", text: `Amount is ${f.amount_ratio.toFixed(1)}× the card's usual spending` });
  else out.push({ tone: "ok", text: "Amount is in line with usual spending" });
  if (f.is_night) out.push({ tone: "bad", text: "Late-night transaction (11 PM – 4 AM)" });
  else out.push({ tone: "ok", text: "Made during normal hours" });
  if (txn.txn_count_1h >= 4) out.push({ tone: "bad", text: `${txn.txn_count_1h} transactions within the last hour` });
  else if (txn.txn_count_1h === 3) out.push({ tone: "warn", text: "3 transactions within the last hour" });
  else out.push({ tone: "ok", text: "Normal transaction frequency" });
  if (txn.new_location === 1) out.push({ tone: "bad", text: "Made from a new location" });
  else out.push({ tone: "ok", text: "Usual location" });
  return out;
}

module.exports = { settings, validate, fraudProbability, riskLevel, signals, buildFeatures, model };
