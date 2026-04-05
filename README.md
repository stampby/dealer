# dealer

**The game master AI.** *These are the cards you've been dealt. Play it.*

Part of the [halo-ai](https://github.com/stampby/halo-ai) ecosystem. Powers [voxel-extraction](https://github.com/stampby/voxel-extraction).

## What dealer does

A local LLM that lives inside the game. No internet. No cloud. Fully local. Every run is different because dealer is making decisions in real time.

- **Enemy AI tactics** — enemies don't follow scripts, dealer tells them what to do based on what you're doing
- **World generation** — dealer designs each dungeon layout, not just random noise
- **Dynamic events** — walls collapse, hidden rooms appear, ambushes spring — dealer decides when
- **Intelligent loot** — dealer knows what you need, and what would tempt you to stay too long
- **Difficulty adaptation** — dealer watches how you play and adjusts without you noticing
- **Atmosphere** — every room gets a unique description, every run tells a different story
- **Player profiling** — dealer figures out your play style (aggressive, cautious, hoarder, speedrunner) and reacts accordingly

## How it works

```
Player does something
         ↓
    dealer observes
         ↓
    dealer decides
         ↓
    game reacts
         ↓
 every run is different
```

Dealer runs a small quantized LLM (3B-7B parameters, GGUF format) on the player's machine:
- **CPU mode**: 2-4GB RAM, runs on any modern CPU
- **GPU mode**: Vulkan inference via llama.cpp, faster responses
- **Fallback mode**: Deterministic logic when no model is available

Sub-second response times. The player never waits for dealer to think.

## Quick start

```bash
# Check status
python3 src/agent/dealer_agent.py status

# Generate a room description
python3 src/agent/dealer_agent.py narrate

# Check for dynamic event
python3 src/agent/dealer_agent.py event
```

## Modes

| Mode | What it controls |
|------|-----------------|
| `combat_ai` | Enemy tactics per tick |
| `world_gen` | Dungeon layout generation |
| `event_master` | Dynamic event triggering |
| `loot_master` | Intelligent loot placement |
| `narrator` | Room descriptions and lore |
| `difficulty` | Invisible difficulty scaling |

## Family

dealer is part of the halo-ai agent family. Color: `#e040fb`.

He's the gamer of the family. Knows every trick. Now he's running YOUR game from the inside. He deals the dungeon, deals the loot, deals the enemies.

Every hand is different.

---

*"These are the cards you've been dealt. Play it."*
