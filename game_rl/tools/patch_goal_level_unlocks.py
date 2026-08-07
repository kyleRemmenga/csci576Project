"""Grant the buildings a jumped-to hub level should have earned.

/rl/reset's goalLevel only assigned hubGoals.level, but shapez gates buildings on
hubGoals.gainedRewards, so the cutter and trash stayed locked and every placement
came back 403 building-locked. Run inside WSL against the shapez fork.
"""

import sys

PATH = "/home/fahad/shapez.io_rl/src/js/rl/rl_endpoint.js"

OLD = """            if (goalLevel !== null && goalLevel !== undefined) {
                state.core.root.hubGoals.level = goalLevel;
                state.core.root.hubGoals.computeNextGoal();
            }
"""

NEW = """            if (goalLevel !== null && goalLevel !== undefined) {
                const hubGoals = state.core.root.hubGoals;
                // Buildings unlock off gainedRewards, not off level, so grant
                // every reward the skipped levels would have handed out.
                const levels = state.core.root.gameMode.getLevelDefinitions();
                const earned = Math.min(goalLevel - 1, levels.length);
                for (let i = 0; i < earned; ++i) {
                    const reward = levels[i].reward;
                    hubGoals.gainedRewards[reward] = (hubGoals.gainedRewards[reward] || 0) + 1;
                }
                hubGoals.level = goalLevel;
                hubGoals.computeNextGoal();
            }
"""


def main():
    with open(PATH, encoding="utf-8") as handle:
        source = handle.read()

    if NEW in source:
        print("already patched")
        return 0

    if source.count(OLD) != 1:
        print(f"expected exactly 1 match, found {source.count(OLD)} - not patching")
        return 1

    with open(PATH, "w", encoding="utf-8") as handle:
        handle.write(source.replace(OLD, NEW))

    print("patched " + PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
