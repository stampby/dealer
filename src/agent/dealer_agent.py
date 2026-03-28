#!/usr/bin/env python3
"""
dealer — the game master AI.

A local LLM that lives inside the game. Controls enemy behavior,
generates worlds, adapts difficulty, creates dynamic events.
Every run is different because dealer is making decisions in real time.

Ships with the game. No internet. No cloud. Fully local.
The player never knows what dealer will do next.

Part of the halo-ai ecosystem.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[dealer] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dealer")


class DealerMode(Enum):
    WORLD_GEN = "world_gen"          # Generating the world/dungeon
    COMBAT_AI = "combat_ai"          # Controlling enemies in real time
    EVENT_MASTER = "event_master"    # Triggering dynamic events
    LOOT_MASTER = "loot_master"      # Deciding loot placement
    NARRATOR = "narrator"            # Room descriptions, lore, dialogue
    DIFFICULTY = "difficulty"        # Adapting difficulty


@dataclass
class PlayerProfile:
    """What dealer knows about the player this run."""
    health_pct: float = 1.0
    deaths: int = 0
    kills: int = 0
    rooms_cleared: int = 0
    total_rooms: int = 0
    loot_value: int = 0
    carry_weight_pct: float = 0.0
    time_elapsed: float = 0.0
    time_remaining: float = 900.0
    favorite_weapon: str = ""
    play_style: str = "unknown"  # aggressive, cautious, explorer, speedrunner
    damage_taken_last_60s: int = 0
    damage_dealt_last_60s: int = 0


@dataclass
class WorldState:
    """Current state of the game world."""
    biome: str = "undercroft"
    difficulty: int = 1
    rooms: list[dict] = field(default_factory=list)
    active_enemies: int = 0
    events_triggered: list[str] = field(default_factory=list)
    boss_alive: bool = True
    extraction_available: bool = True
    panic_level: float = 0.0  # 0-1, how tense things are


class Dealer:
    """
    The game master. Deals the hand. Every run is different.

    Uses a small local LLM (3B-7B params, GGUF quantized) to make
    intelligent decisions about the game in real time. Runs on CPU
    or player's GPU via llama.cpp.
    """

    BANNER = r"""
    ╔═══════════════════════════════════════╗
    ║          🎴  D E A L E R  🎴          ║
    ║       the game master AI              ║
    ║                                       ║
    ║  "Every hand is different."           ║
    ╚═══════════════════════════════════════╝
    """

    # System prompts for each mode
    SYSTEM_PROMPTS = {
        DealerMode.WORLD_GEN: (
            "You are dealer, the dungeon master AI for a voxel dungeon crawler. "
            "Generate dungeon layouts as JSON. Each room has a theme, purpose, "
            "enemy count, loot tier, and connections. Create interesting, "
            "varied dungeons that tell a story through their layout. "
            "Respond ONLY with valid JSON."
        ),
        DealerMode.COMBAT_AI: (
            "You are dealer, controlling enemies in a voxel dungeon crawler. "
            "Given the player's state and enemy positions, decide enemy tactics. "
            "Options: 'aggressive' (rush player), 'defensive' (hold position), "
            "'flank' (circle around), 'retreat' (fall back), 'ambush' (wait), "
            "'call_reinforcements', 'use_ability'. "
            "Respond with a JSON array of actions, one per enemy."
        ),
        DealerMode.EVENT_MASTER: (
            "You are dealer, the event master for a voxel dungeon crawler. "
            "Based on the current game state, decide if a dynamic event should "
            "trigger. Events: wall_collapse, hidden_room, enemy_ambush, "
            "loot_cache, trap_activation, environmental_hazard, npc_encounter, "
            "boss_taunt, shortcut_reveal, resource_drop. "
            "Respond with JSON: {event, description, severity}. "
            "Say {event: 'none'} if no event should trigger."
        ),
        DealerMode.LOOT_MASTER: (
            "You are dealer, the loot master. Given the player's inventory, "
            "playstyle, and current needs, decide what loot to place. "
            "Be fair but interesting — sometimes give what they need, "
            "sometimes tempt them with heavy high-value items when they're "
            "already overloaded. Make them choose. "
            "Respond with JSON: [{item, rarity, weight, value}]"
        ),
        DealerMode.NARRATOR: (
            "You are dealer, the narrator of a voxel dungeon crawler. "
            "Write atmospheric, short room descriptions (1-2 sentences max). "
            "Dark, gritty, immersive. No flowery language. "
            "Reference the voxel world — blocky geometry, cube-shaped stones, "
            "angular shadows. Make the player feel the weight of the place."
        ),
        DealerMode.DIFFICULTY: (
            "You are dealer, the difficulty controller. Given the player's "
            "performance data, decide difficulty adjustments. "
            "Scale: -2 (much easier) to +2 (much harder). "
            "Values: enemy_health_mult, enemy_damage_mult, enemy_count_mult, "
            "loot_quality_bonus, event_frequency. "
            "Respond with JSON. Be subtle — the player should never feel "
            "the difficulty is being manipulated."
        ),
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        server_url: str = "http://127.0.0.1:8081",
        use_embedded: bool = True,
    ):
        """
        Initialize dealer.

        Args:
            model_path: Path to GGUF model file (for embedded mode)
            server_url: URL of llama.cpp server (for server mode)
            use_embedded: If True, start llama.cpp as subprocess
        """
        self.model_path = model_path or self._find_model()
        self.server_url = server_url
        self.use_embedded = use_embedded
        self.player = PlayerProfile()
        self.world = WorldState()
        self._server_process = None
        self._request_count = 0
        self._avg_response_ms = 0.0

        log.info(self.BANNER)

    def _find_model(self) -> str:
        """Find a suitable GGUF model on the system."""
        search_paths = [
            Path.home() / ".local" / "share" / "dealer" / "models",
            Path.home() / "models",
            Path("/opt/dealer/models"),
            Path("/srv/ai/models"),
        ]
        for base in search_paths:
            if base.exists():
                for gguf in base.glob("*.gguf"):
                    size_gb = gguf.stat().st_size / (1024 ** 3)
                    # Prefer small models (1-4GB = 3B-7B quantized)
                    if 0.5 < size_gb < 5.0:
                        log.info("Found model: %s (%.1fGB)", gguf.name, size_gb)
                        return str(gguf)
        log.warning("No GGUF model found — dealer will use fallback logic")
        return ""

    # ── LLM Interface ──────────────────────────────────────────

    async def _query_llm(
        self, mode: DealerMode, user_prompt: str, max_tokens: int = 256,
    ) -> str:
        """Send a prompt to the local LLM and get a response."""
        system_prompt = self.SYSTEM_PROMPTS[mode]

        start = time.monotonic()

        if self.model_path and self.use_embedded:
            response = await self._query_embedded(system_prompt, user_prompt, max_tokens)
        elif self.server_url:
            response = await self._query_server(system_prompt, user_prompt, max_tokens)
        else:
            response = self._fallback_response(mode)

        elapsed_ms = (time.monotonic() - start) * 1000
        self._request_count += 1
        self._avg_response_ms = (
            self._avg_response_ms * (self._request_count - 1) + elapsed_ms
        ) / self._request_count

        log.debug("LLM response (%.0fms): %s", elapsed_ms, response[:100])
        return response

    async def _query_server(
        self, system: str, user: str, max_tokens: int,
    ) -> str:
        """Query llama.cpp server via HTTP API."""
        import aiohttp

        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.9,
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("LLM query failed: %s — using fallback", e)

        return self._fallback_response_for_prompt(system)

    async def _query_embedded(
        self, system: str, user: str, max_tokens: int,
    ) -> str:
        """Query via embedded llama.cpp subprocess."""
        # Uses llama-cli or llama-server started as subprocess
        # For now, delegate to server mode (server auto-started)
        return await self._query_server(system, user, max_tokens)

    def _fallback_response(self, mode: DealerMode) -> str:
        """Deterministic fallback when LLM is unavailable."""
        import random
        match mode:
            case DealerMode.COMBAT_AI:
                tactics = ["aggressive", "defensive", "flank", "retreat", "ambush"]
                return json.dumps([{"tactic": random.choice(tactics)}])
            case DealerMode.EVENT_MASTER:
                if random.random() < 0.15:
                    events = ["wall_collapse", "loot_cache", "enemy_ambush", "trap_activation"]
                    return json.dumps({"event": random.choice(events), "severity": "medium"})
                return json.dumps({"event": "none"})
            case DealerMode.NARRATOR:
                descs = [
                    "The walls press in. Water drips from somewhere above.",
                    "Dust hangs in the air. Something was here recently.",
                    "The stone floor is worn smooth by ancient feet.",
                    "A cold draft pushes through cracks in the voxels.",
                    "Shadows pool in the corners. The lantern barely reaches.",
                ]
                return random.choice(descs)
            case DealerMode.DIFFICULTY:
                return json.dumps({
                    "enemy_health_mult": 1.0,
                    "enemy_damage_mult": 1.0,
                    "enemy_count_mult": 1.0,
                    "loot_quality_bonus": 0,
                    "event_frequency": 1.0,
                })
            case _:
                return "{}"

    def _fallback_response_for_prompt(self, system: str) -> str:
        """Match system prompt to a mode for fallback."""
        for mode, prompt in self.SYSTEM_PROMPTS.items():
            if prompt == system:
                return self._fallback_response(mode)
        return "{}"

    # ── Game Master Functions ──────────────────────────────────

    async def decide_enemy_tactics(
        self, enemies: list[dict], player_state: dict,
    ) -> list[dict]:
        """Decide what each enemy should do this tick."""
        self._update_player(player_state)

        prompt = json.dumps({
            "enemies": enemies,
            "player": {
                "health_pct": self.player.health_pct,
                "position": player_state.get("position", [0, 0, 0]),
                "is_sprinting": player_state.get("is_sprinting", False),
                "is_extracting": player_state.get("is_extracting", False),
                "weapon": self.player.favorite_weapon,
            },
            "panic_level": self.world.panic_level,
        })

        response = await self._query_llm(DealerMode.COMBAT_AI, prompt, max_tokens=128)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return [{"tactic": "aggressive"} for _ in enemies]

    async def check_for_event(self) -> Optional[dict]:
        """Check if a dynamic event should trigger right now."""
        prompt = json.dumps({
            "rooms_cleared": self.player.rooms_cleared,
            "total_rooms": self.player.total_rooms,
            "time_remaining": self.player.time_remaining,
            "health_pct": self.player.health_pct,
            "carry_weight_pct": self.player.carry_weight_pct,
            "boss_alive": self.world.boss_alive,
            "recent_events": self.world.events_triggered[-5:],
            "panic_level": self.world.panic_level,
        })

        response = await self._query_llm(DealerMode.EVENT_MASTER, prompt, max_tokens=128)
        try:
            event = json.loads(response)
            if event.get("event") != "none":
                self.world.events_triggered.append(event["event"])
                log.info("Event triggered: %s", event)
                return event
        except json.JSONDecodeError:
            pass
        return None

    async def generate_room_description(self, room: dict) -> str:
        """Generate atmospheric text for a room."""
        prompt = json.dumps({
            "room_role": room.get("role", "combat"),
            "room_size": room.get("size", "medium"),
            "biome": self.world.biome,
            "enemies_present": room.get("enemy_count", 0) > 0,
            "loot_present": room.get("has_loot", False),
            "is_boss_room": room.get("is_boss", False),
        })

        return await self._query_llm(DealerMode.NARRATOR, prompt, max_tokens=64)

    async def decide_loot(self, room: dict) -> list[dict]:
        """Decide what loot to place in a room."""
        prompt = json.dumps({
            "room_role": room.get("role", "loot"),
            "player_inventory_value": self.player.loot_value,
            "carry_weight_pct": self.player.carry_weight_pct,
            "play_style": self.player.play_style,
            "difficulty": self.world.difficulty,
            "time_remaining": self.player.time_remaining,
        })

        response = await self._query_llm(DealerMode.LOOT_MASTER, prompt, max_tokens=256)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    async def adjust_difficulty(self) -> dict:
        """Evaluate player performance and adjust difficulty."""
        prompt = json.dumps({
            "health_pct": self.player.health_pct,
            "deaths": self.player.deaths,
            "kills": self.player.kills,
            "rooms_cleared": self.player.rooms_cleared,
            "damage_taken_recent": self.player.damage_taken_last_60s,
            "damage_dealt_recent": self.player.damage_dealt_last_60s,
            "play_style": self.player.play_style,
            "time_pct_remaining": self.player.time_remaining / 900.0,
        })

        response = await self._query_llm(DealerMode.DIFFICULTY, prompt, max_tokens=128)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"enemy_health_mult": 1.0, "enemy_damage_mult": 1.0}

    async def generate_world_layout(self, pack_config: dict) -> dict:
        """Generate a unique dungeon layout for this run."""
        prompt = json.dumps({
            "pack": pack_config.get("pack_name", "The Undercroft"),
            "difficulty": self.world.difficulty,
            "min_rooms": pack_config.get("min_rooms", 8),
            "max_rooms": pack_config.get("max_rooms", 20),
            "available_room_types": ["combat", "loot", "boss", "puzzle", "rest", "trap"],
            "enemy_types": pack_config.get("enemy_types", ["skeleton", "crawler"]),
            "theme": pack_config.get("description", "dark underground dungeon"),
        })

        response = await self._query_llm(DealerMode.WORLD_GEN, prompt, max_tokens=512)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    # ── Player Profiling ───────────────────────────────────────

    def _update_player(self, state: dict) -> None:
        """Update dealer's knowledge of the player."""
        self.player.health_pct = state.get("health_pct", 1.0)
        self.player.kills = state.get("kills", 0)
        self.player.loot_value = state.get("loot_value", 0)
        self.player.carry_weight_pct = state.get("carry_weight_pct", 0.0)
        self.player.time_remaining = state.get("time_remaining", 900.0)

        # Detect play style from behavior
        dmg_dealt = state.get("damage_dealt_last_60s", 0)
        dmg_taken = state.get("damage_taken_last_60s", 0)

        if dmg_dealt > 100 and dmg_taken > 50:
            self.player.play_style = "aggressive"
        elif dmg_dealt < 20 and self.player.rooms_cleared > 0:
            self.player.play_style = "cautious"
        elif self.player.carry_weight_pct > 0.8:
            self.player.play_style = "hoarder"
        elif state.get("rooms_explored_fast", 0) > 3:
            self.player.play_style = "speedrunner"
        else:
            self.player.play_style = "explorer"

    # ── Status ─────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "agent": "dealer",
            "status": "ready" if self.model_path else "fallback",
            "model": Path(self.model_path).name if self.model_path else "none",
            "server": self.server_url,
            "requests": self._request_count,
            "avg_response_ms": round(self._avg_response_ms, 1),
            "player_style": self.player.play_style,
            "panic_level": self.world.panic_level,
        }

    def get_marketplace_listing(self) -> dict:
        return {
            "name": "dealer",
            "display_name": "Dealer",
            "tagline": "The Game Master AI",
            "description": (
                "A local LLM that lives inside the game. Controls enemy "
                "behavior, generates worlds, adapts difficulty, creates "
                "dynamic events. Every run is different. Ships with the "
                "game — no internet required."
            ),
            "icon": "dealer.svg",
            "color": "#e040fb",
            "category": "gaming",
            "capabilities": [
                "Dynamic enemy AI tactics",
                "Procedural world generation",
                "Real-time difficulty adaptation",
                "Dynamic event triggering",
                "Intelligent loot placement",
                "Atmospheric room narration",
                "Player behavior profiling",
            ],
            "requires": ["llama.cpp"],
            "optional": ["gpu_inference"],
        }


async def main():
    import sys

    dealer = Dealer()
    log.info("Status: %s", json.dumps(dealer.status(), indent=2))

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "status":
            print(json.dumps(dealer.status(), indent=2))
        elif command == "marketplace":
            print(json.dumps(dealer.get_marketplace_listing(), indent=2))
        elif command == "narrate":
            desc = await dealer.generate_room_description({
                "role": "combat", "size": "large", "enemy_count": 3
            })
            print(desc)
        elif command == "event":
            event = await dealer.check_for_event()
            print(json.dumps(event, indent=2) if event else "No event")
        else:
            print("Usage: dealer [status|marketplace|narrate|event]")
    else:
        print("Usage: dealer [status|marketplace|narrate|event]")


if __name__ == "__main__":
    asyncio.run(main())
