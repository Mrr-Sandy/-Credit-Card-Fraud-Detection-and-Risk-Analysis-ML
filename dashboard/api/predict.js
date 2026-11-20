const { settings, validate, fraudProbability, riskLevel, signals, buildFeatures } = require("./_lib");

module.exports = (req, res) => {
  const cfg = settings();
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST with a JSON body" });
  }
  const { errors, txn } = validate(req.body, cfg);
  if (errors.length) return res.status(400).json({ error: "Invalid input", details: errors });

  const p = fraudProbability(txn);
  res.status(200).json({
    fraud_probability: Math.round(p * 1000) / 1000,
    risk_level: riskLevel(p, cfg),
    model: cfg.modelName,
    thresholds: { low: cfg.low, high: cfg.high },
    amount_ratio: buildFeatures(txn).amount_ratio,
    signals: signals(txn),
  });
};
