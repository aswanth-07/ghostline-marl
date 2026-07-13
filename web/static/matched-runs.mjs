function runIdentity(record) {
  if (!record || typeof record !== "object") return null;
  const tier = Number(record.tier);
  const seed = Number(record.seed);
  if (!Number.isInteger(tier) || tier < 1 || tier > 6) return null;
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) return null;
  return { tier, seed, key: `${tier}:${seed}` };
}

export function matchedRunSnapshot(history) {
  const human = history?.human ?? null;
  const agent = history?.agent ?? null;
  const humanIdentity = runIdentity(human);
  const agentIdentity = runIdentity(agent);

  if (!human && !agent) {
    return {
      state: "empty",
      matched: false,
      message: "Complete human and agent runs on the same tier and seed.",
      human,
      agent,
    };
  }
  if (!human || !agent) {
    const present = humanIdentity ?? agentIdentity;
    return {
      state: "awaiting",
      matched: false,
      message: present
        ? `Awaiting the other controller on T${present.tier} / seed ${present.seed}.`
        : "Awaiting a second run with a valid tier and seed.",
      human,
      agent,
    };
  }
  if (!humanIdentity || !agentIdentity) {
    return {
      state: "refused",
      matched: false,
      message: "Comparison refused: one run has no valid tier-and-seed identity.",
      human,
      agent,
    };
  }
  if (humanIdentity.key !== agentIdentity.key) {
    return {
      state: "refused",
      matched: false,
      message: `Comparison refused: human T${humanIdentity.tier} / ${humanIdentity.seed} and agent T${agentIdentity.tier} / ${agentIdentity.seed} are different contracts.`,
      human,
      agent,
    };
  }
  return {
    state: "matched",
    matched: true,
    message: `Matched contract: T${humanIdentity.tier} / seed ${humanIdentity.seed}.`,
    human,
    agent,
  };
}
