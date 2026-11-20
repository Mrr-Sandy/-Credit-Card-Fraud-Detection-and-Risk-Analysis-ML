const { settings } = require("./_lib");

// Settings the page needs, read from the Vercel environment variables.
module.exports = (req, res) => {
  const cfg = settings();
  res.setHeader("Cache-Control", "no-store");
  res.status(200).json({
    low: cfg.low,
    high: cfg.high,
    maxAmount: cfg.maxAmount,
    model: cfg.modelName,
    env: cfg.appEnv,
  });
};
