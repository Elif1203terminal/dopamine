import json
import random
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import math
from itertools import groupby
import matplotlib.cm as cm

class RewardPathwayAgent:

    
    def _save_memory(self, memory_path='memory.json'):
        """Save current learned patterns to disk."""
        try:
            with open(memory_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
            print(f"[MEMORY] Saved memory to {memory_path}.")
        except Exception as e:
            print(f"[MEMORY ERROR] Could not save memory: {e}")


    def __init__(self, maze_size=(10, 10), memory_file="agent_memory.json"):
        """Initialize the agent with a larger maze and more sophisticated memory structures"""
        self.maze_size = maze_size
        self.memory_file = memory_file
        self.memory = self._load_memory()
        self.current_position = (0, 0)
        self.goal_position = (maze_size[0]-1, maze_size[1]-1)
        self.maze = self._generate_maze()
        
        # Learning parameters
        self.exploration_rate = 0.8  # Start with high exploration
        self.decay_rate = 0.95       # Gradually reduce exploration
        self.min_exploration = 0.1   # Never go below this exploration rate
        self.curiosity_burst_chance = 0.05  # Occasional random exploration bursts
        
        # Tracking current trial
        self.current_path = []
        self.visited_positions = set()
        self.attempt_history = []
        self.current_patterns = []
        self.decision_reasons = []
        # Vision field (3x3 grid centered on agent)
        self.vision_range = 1  # This defines radius (1 = 3x3)
        self.visible_tiles = {}  # {(x, y): terrain}
        # Pattern abstraction parameters
        self.micro_patterns = {}  # For small movement sequences (2-4 steps)
        self.macro_patterns = {}  # For larger movement sequences (5-10 steps)
        
        # Position context awareness
        self.position_success_map = defaultdict(lambda: {"success": 0, "failure": 0})
        
        # Meta-learning
        self.heuristics = {
            "wall_following": 0.0,
            "straight_lines": 0.0,
            "exploration": 0.0,
            "backtracking": 0.0
        }
        
        # World model - internal representation of the maze
        self.world_model = self._initialize_world_model()
        self.confidence_map = np.zeros(maze_size)  # confidence in our world model
        
        # Visualization data
        self.heatmap = np.zeros(maze_size)
        self.learning_curve = []
        
        # Self-reflection data
        self.regret_history = []
        self.insight_log = []
        
    def _initialize_world_model(self):
        """Create an initial world model with probabilities of walls/paths"""
        # 0 = unknown, 1 = path, 2 = wall
        model = np.zeros(self.maze_size)
        # Start and goal are known to be paths
        model[0, 0] = 1
        model[self.maze_size[0]-1, self.maze_size[1]-1] = 1
        return model
        
    def _load_memory(self):
        """Load agent memory from file or create new memory structure"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
            except:
                memory = self._initialize_memory()
        else:
            memory = self._initialize_memory()

        # Ensure vision_patterns key is always present
        if "vision_patterns" not in memory:
            memory["vision_patterns"] = {}

        return memory
            
    def _initialize_memory(self):
        """Create a new memory structure with hierarchical pattern storage"""
        return {
            "patterns": {},
            "meta": {
                "total_trials": 0,
                "successful_trials": 0,
                "total_steps": 0,
                "discoveries": [],
                "high_reward_patterns": []
            },
            "micro_patterns": {},
            "macro_patterns": {},
            "position_patterns": {},
            "heuristics": {},
            "world_model_history": [],
            "meta_patterns": {},
            "symmetries": {},
            "decision_reasoning": [],
            "insights": [],
            "regrets": []
        }
     
    def _generate_maze(self):
        """
        Generate a maze with consistent semantic structure.
        One guaranteed path from S to G contains terrain gradient:
        S -> . -> + -> > -> G
        """
        width, height = self.maze_size
        self.maze = np.full((width, height), '#', dtype=str)

        # Step 1: Carve a straight safe path from S to G
        x, y = 0, 0
        self.current_position = (x, y)
        path = [(x, y)]
        while (x, y) != (width - 1, height - 1):
            if x < width - 1 and (random.random() < 0.5 or y == height - 1):
                x += 1
            elif y < height - 1:
                y += 1
            path.append((x, y))

        self.goal_position = (x, y)

        # Step 2: Lay terrain gradient over safe path
        terrain_sequence = ['.'] * (len(path) // 3) + ['+'] * (len(path) // 3) + ['>'] * (len(path) // 3)
        while len(terrain_sequence) < len(path) - 2:
            terrain_sequence.append('.')  # pad with neutral
        for i, (px, py) in enumerate(path[1:-1]):
            self.maze[px][py] = terrain_sequence[i]

        # Step 3: Set start and goal explicitly
        sx, sy = path[0]
        gx, gy = path[-1]
        self.maze[sx][sy] = 'S'
        self.maze[gx][gy] = 'G'

        # Step 4: Scatter misleading terrain elsewhere
        for _ in range(width * height // 3):
            rx, ry = random.randint(0, width - 1), random.randint(0, height - 1)
            if self.maze[rx][ry] == '#':
                terrain = random.choices(['~', '!', '+', '.'], weights=[0.3, 0.3, 0.2, 0.2])[0]
                self.maze[rx][ry] = terrain

        return self.maze


    def _is_maze_solvable(self, maze):
        """Check if there is a path from start to goal using BFS"""
        from collections import deque
        visited = set()
        queue = deque([self.current_position])

        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal_position:
                return True
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze_size[0] and 0 <= ny < self.maze_size[1]:
                    if maze[nx][ny] != '#' and (nx, ny) not in visited:
                        queue.append((nx, ny))
        return False
    
    def _would_block_path(self, maze, x, y):
        """Check if placing a wall here would block paths completely"""
        # Simple check: ensure there's at least one adjacent open cell
        adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        blocked_count = 0
        
        for adj_x, adj_y in adjacent:
            if not (0 <= adj_x < self.maze_size[0] and 0 <= adj_y < self.maze_size[1]):
                blocked_count += 1
            elif maze[adj_x][adj_y] == '#':
                blocked_count += 1
                
        # If three or more sides are blocked, don't place a wall here
        return blocked_count >= 3
        
    def _is_valid_move(self, move):
        """Check if a move is valid in the current maze state"""
        x, y = self.current_position
        
        if move == "U":
            new_x, new_y = x-1, y
        elif move == "D":
            new_x, new_y = x+1, y
        elif move == "L":
            new_x, new_y = x, y-1
        elif move == "R":
            new_x, new_y = x, y+1
        
        # Check maze boundaries
        if not (0 <= new_x < self.maze_size[0] and 0 <= new_y < self.maze_size[1]):
            return False
            
        # Check for walls
        if self.maze[new_x][new_y] == '#':
            return False
            
        return True
    
    def _get_next_position(self, position, move):
        """Get the next position after a move"""
        x, y = position
        
        if move == "U":
            return (x-1, y)
        elif move == "D":
            return (x+1, y)
        elif move == "L":
            return (x, y-1)
        elif move == "R":
            return (x, y+1)
        
        return position  # No change if invalid move
        
    def _compress_path(self, path):
        """Compress a path into a pattern representation"""
        if not path:
            return ""
            
        compressed = []
        current_move = path[0]
        count = 1
        
        for move in path[1:]:
            if move == current_move:
                count += 1
            else:
                compressed.append(f"{count}{current_move}")
                current_move = move
                count = 1
                
        compressed.append(f"{count}{current_move}")
        return "-".join(compressed)
    def _terrain_arc_score(self, positions):
        """Score a path segment based on past terrain arc outcomes"""
        if "terrain_arcs" not in self.memory or len(positions) < 3:
            return 0.0

        arc = "->".join(self.maze[x][y] for (x, y) in positions if 0 <= x < self.maze_size[0] and 0 <= y < self.maze_size[1])
        if arc in self.memory["terrain_arcs"]:
            arc_data = self.memory["terrain_arcs"][arc]
            success_rate = arc_data["successes"] / max(1, arc_data["seen"])
            return arc_data["reward"] * success_rate
        return 0.0      
        
    def _extract_patterns(self, path):
        """Extract multiple levels of patterns from a path"""
        patterns = {}
        
        # Extract micro-patterns (2-4 steps)
        for size in range(2, min(5, len(path)+1)):
            for i in range(len(path) - size + 1):
                micro = self._compress_path(path[i:i+size])
                if micro not in patterns:
                    patterns[micro] = {"level": "micro", "count": 1, "position": i}
                else:
                    patterns[micro]["count"] += 1
        
        # Extract macro-patterns (5-10 steps)
        for size in range(5, min(11, len(path)+1)):
            for i in range(len(path) - size + 1):
                macro = self._compress_path(path[i:i+size])
                if macro not in patterns:
                    patterns[macro] = {"level": "macro", "count": 1, "position": i}
                else:
                    patterns[macro]["count"] += 1
                    
        return patterns
    
    def _extract_meta_patterns(self):
        """Extract higher-level concepts from observed patterns"""
        if len(self.memory["patterns"]) < 5:
            return  # Need more data to find meta patterns
            
        patterns = list(self.memory["patterns"].items())
        
        # Look for symmetrical patterns
        for p1, data1 in patterns:
            for p2, data2 in patterns:
                if p1 != p2 and self._are_symmetrical(p1, p2):
                    # Record this symmetry relationship
                    if p1 not in self.memory["symmetries"]:
                        self.memory["symmetries"][p1] = []
                    if p2 not in self.memory["symmetries"][p1]:
                        self.memory["symmetries"][p1].append(p2)
        
        # Look for patterns that form loops
        for pattern, data in patterns:
            if self._forms_loop(pattern):
                self.memory["meta_patterns"][pattern] = "loop"
                
        # Identify wall-following patterns
        for pattern, data in patterns:
            if self._is_wall_following(pattern):
                self.memory["meta_patterns"][pattern] = "wall_following"
                
        # Identify exploration vs exploitation patterns
        successful_patterns = [p for p, d in patterns if d["successes"] > d["failures"]]
        for pattern in successful_patterns:
            if len(pattern) > 15:  # Longer patterns
                self.memory["meta_patterns"][pattern] = "exploitation"
                
        # Look for recurring subpatterns across successful paths
        subpatterns = self._find_common_subpatterns(successful_patterns, min_freq=2)
        for subpattern, freq in subpatterns:
            if len(subpattern) >= 3:  # Meaningful subpatterns
                self.memory["meta_patterns"][subpattern] = "recurring_success"
                
    def _find_common_subpatterns(self, patterns, min_freq=2):
        """Find common subpatterns across a list of patterns"""
        all_subpatterns = []
        
        for pattern in patterns:
            parts = pattern.split("-")
            for i in range(len(parts)):
                for j in range(i+1, min(i+5, len(parts)+1)):
                    subpattern = "-".join(parts[i:j])
                    all_subpatterns.append(subpattern)
                    
        # Count frequencies
        counter = Counter(all_subpatterns)
        return [(p, c) for p, c in counter.items() if c >= min_freq]
                
    def _is_wall_following(self, pattern):
        """Detect if a pattern represents wall-following behavior"""
        # Wall following often alternates between two directions
        parts = pattern.split("-")
        if len(parts) < 3:
            return False
            
        # Check for alternating pattern like R-U-R-U or similar
        directions = [p[-1] for p in parts]  # Extract just the direction
        alternating = True
        for i in range(2, len(directions)):
            if directions[i] != directions[i-2]:
                alternating = False
                break
                
        return alternating
        
    def _forms_loop(self, pattern):
        """Check if a pattern forms a loop (returns to same position)"""
        parts = pattern.split("-")
        x, y = 0, 0
        
        # Simulate the pattern
        for part in parts:
            count = int(part[:-1])
            direction = part[-1]
            
            for _ in range(count):
                if direction == "U":
                    y -= 1
                elif direction == "D":
                    y += 1
                elif direction == "L":
                    x -= 1
                elif direction == "R":
                    x += 1
                    
        # If we end up back at (0,0), it's a loop
        return x == 0 and y == 0
        
    def _are_symmetrical(self, p1, p2):
        """Check if two patterns are symmetrical (opposite or rotated)"""
        # Simple opposition check (U↔D, L↔R)
        opposites = {"U": "D", "D": "U", "L": "R", "R": "L"}
        
        parts1 = p1.split("-")
        parts2 = p2.split("-")
        
        if len(parts1) != len(parts2):
            return False
            
        # Check for direct opposition
        all_opposite = True
        for i in range(len(parts1)):
            count1 = int(parts1[i][:-1])
            dir1 = parts1[i][-1]
            count2 = int(parts2[i][:-1])
            dir2 = parts2[i][-1]
            
            if count1 != count2 or dir2 != opposites.get(dir1):
                all_opposite = False
                break
                
        return all_opposite
        
    def _calculate_reward(self, path, success, steps_taken):
        """Calculate reward based on multiple factors with balanced temporal considerations"""
        compressed_path = self._compress_path(path)
        patterns = self._extract_patterns(path)
        reward = 0
        # Terrain-based reward logic
        terrain_reward = 0
        for pos in self.visited_positions:
            x, y = pos
            if not (0 <= x < self.maze_size[0] and 0 <= y < self.maze_size[1]):
                continue
            tile = self.maze[x][y]
            if tile == '~':
                terrain_reward -= 0.5
            elif tile == '+':
                terrain_reward += 1.0
            elif tile == '!':
                terrain_reward -= 1.5
            elif tile == '>':
                terrain_reward += 0.8
            elif tile == 'G':
                terrain_reward += 10.0
        reward += terrain_reward
        # Base rewards
        if success:
            # Success reward with efficiency bonus
            efficiency_factor = max(0.5, 1.0 - (steps_taken / (self.maze_size[0] + self.maze_size[1])))
            reward += 10 * efficiency_factor
        
        # Check for learning from failure
        if compressed_path in self.memory["patterns"]:
            pattern_data = self.memory["patterns"][compressed_path]
            
            if pattern_data["failures"] > 0 and success:
                # Extra reward for overcoming previous failures
                reward += min(5, pattern_data["failures"]) * 2
                self._log_insight(f"Learned from {pattern_data['failures']} previous failures", reward_impact=2)

            if pattern_data["failures"] > 3 and not success:
                # Penalty for repeating known bad patterns
                reward -= 2
                self._log_regret(f"Repeated a known bad pattern ({pattern_data['failures']} failures)", -2)
        else:
            # Novelty bonus for trying a new pattern
            reward += 1
            
        # === Enhanced Discovery-Driven Reward ===
        novel_position_bonus = 0
        low_confidence_bonus = 0

        for pos in self.visited_positions:
            x, y = pos
            if pos not in self.position_success_map or sum(self.position_success_map[pos].values()) < 3:
                novel_position_bonus += 1
            if self.confidence_map[x, y] < 0.3:
                low_confidence_bonus += (0.5 - self.confidence_map[x, y])  # greater bonus for greater uncertainty

        # Boost curiosity-driven exploration
        reward += novel_position_bonus * 0.4     # double original
        reward += low_confidence_bonus * 0.6     # new: reinforce navigating ambiguous terrain

        
        # Reward pattern abstraction
        unique_patterns = len(patterns)
        reward += unique_patterns * 0.3
        
        # Add temporal context - balance short vs long-term rewards
        if len(self.learning_curve) > 20:
            recent_efficiency = sum([e["steps"] for e in self.learning_curve[-10:]])/10
            if steps_taken < recent_efficiency:
                # Bonus for improving efficiency over time
                reward *= 1.2
                self._log_insight("Improving efficiency over recent average", reward_impact=0.2)
            
        # Add regret mechanism - penalize extended unsuccessful paths
        if not success and steps_taken > self.maze_size[0] + self.maze_size[1]:
            # Calculate opportunity cost of poor choices
            regret_penalty = 0.5 * (steps_taken / (self.maze_size[0] + self.maze_size[1]))
            reward -= regret_penalty
            self._log_regret(f"Inefficient path wasted {steps_taken} steps", -regret_penalty)
        
        # Reward world model improvement
        model_improvement = self._calculate_model_improvement()
        reward += model_improvement * 0.5
        
        # Apply temporal discounting (shorter paths are better)
        discount_factor = 0.9 ** (steps_taken / 10)
        reward *= discount_factor
        
        return reward
    
    def _log_insight(self, insight_text, reward_impact=0):
        """Log a positive insight for self-reflection"""
        self.insight_log.append({
            "trial": self.memory["meta"]["total_trials"],
            "insight": insight_text,
            "reward_impact": reward_impact,
            "position": self.current_position,
            "steps_taken": len(self.current_path)
        })
        
        # Add to persistent memory
        self.memory["insights"].append({
            "trial": self.memory["meta"]["total_trials"],
            "insight": insight_text,
            "reward_impact": reward_impact
        })
        
    def _log_regret(self, regret_text, reward_impact=0):
        """Log a regret for self-reflection"""
        self.regret_history.append({
            "trial": self.memory["meta"]["total_trials"],
            "regret": regret_text,
            "reward_impact": reward_impact,
            "position": self.current_position,
            "steps_taken": len(self.current_path)
        })
        
        # Add to persistent memory
        self.memory["regrets"].append({
            "trial": self.memory["meta"]["total_trials"],
            "regret": regret_text,
            "reward_impact": reward_impact
        })
    
    def _calculate_model_improvement(self):
        """Calculate how much the world model improved in this trial"""
        # Simple metric: count of newly discovered cells
        new_cells = len(self.visited_positions) - self._count_previously_known_cells()
        return new_cells * 0.1
        
    def _count_previously_known_cells(self):
        """Count how many cells we already knew about before this trial"""
        known_count = 0
        for pos in self.visited_positions:
            x, y = pos
            if self.world_model[x, y] != 0:  # If not unknown
                known_count += 1
        return known_count

    def _update_world_model(self, new_position):
        """Update the agent's internal world model based on observations"""
        x, y = new_position
        
        # Update current position based on tile type
        tile = self.maze[x][y]
        if tile in {'+', '>', 'G'}:
            self.world_model[x, y] = 3  # Mark as rewarding
        elif tile in {'!', '~'}:
            self.world_model[x, y] = -1  # Mark as danger
        else:
            self.world_model[x, y] = 1  # Regular path

        self.confidence_map[x, y] += 0.5
        
        # Update confidence in the model
        self.confidence_map = np.minimum(self.confidence_map, 1.0)  # Cap confidence at 1.0
        
        # Check adjacent cells for walls
        adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for adj_x, adj_y in adjacent:
            if not (0 <= adj_x < self.maze_size[0] and 0 <= adj_y < self.maze_size[1]):
                continue  # Skip out of bounds
                
            # Test if this is valid - if not, it must be a wall
            if not self._is_valid_move_to(self.current_position, (adj_x, adj_y)):
                self.world_model[adj_x, adj_y] = 2  # Mark as wall
                self.confidence_map[adj_x, adj_y] += 0.2

    def _update_vision_field(self):
        """Update the 3x3 vision field around the current position"""
        x, y = self.current_position
        self.visible_tiles.clear()
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze_size[0] and 0 <= ny < self.maze_size[1]:
                    self.visible_tiles[(nx, ny)] = self.maze[nx][ny]

    def _is_valid_move_to(self, from_pos, to_pos):
        """Check if a move from one position to another is valid"""
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Must be adjacent
        if abs(x2-x1) + abs(y2-y1) != 1:
            return False
            
        # Check maze boundaries
        if not (0 <= x2 < self.maze_size[0] and 0 <= y2 < self.maze_size[1]):
            return False
            
        # Check for walls
        if self.maze[x2][y2] == '#':
            return False
            
        return True

    def _reflect_on_decision(self, move, outcome, reward=None):
        """Record reasoning behind decisions and outcomes"""
        self.last_move = move
        x, y = self.current_position
        new_pos = self._get_next_position(self.current_position, move)
        
        reasoning = {
            "trial": self.memory["meta"]["total_trials"],
            "step": len(self.current_path),
            "move": move,
            "from_position": (x, y),
            "to_position": new_pos,
            "pattern_matching": [],
            
            "position_data": {},
            "outcome": outcome
        }
        # Add vision-based reasoning
        if new_pos in self.visible_tiles:
            reasoning["visible_terrain"] = self.visible_tiles[new_pos]
            reasoning["vision_score"] = (
                1.0 if self.visible_tiles[new_pos] == '+'
                else 0.8 if self.visible_tiles[new_pos] == '>'
                else -1.5 if self.visible_tiles[new_pos] == '!'
                else -0.5 if self.visible_tiles[new_pos] == '~'
                else 10.0 if self.visible_tiles[new_pos] == 'G'
                else 0.0
            )
        else:
            reasoning["visible_terrain"] = "unknown"
            reasoning["vision_score"] = 0.0
        
        # Record pattern matching that led to this decision
        for pattern in self.current_patterns:
            if pattern in self.memory["patterns"]:
                success_rate = self.memory["patterns"][pattern]["successes"] / max(1, self.memory["patterns"][pattern]["seen"])
                reasoning["pattern_matching"].append({
                    "pattern": pattern,
                    "success_rate": success_rate
                })
        
        # Record position data
        pos_str = f"{new_pos[0]},{new_pos[1]}"
        if pos_str in self.position_success_map:
            pos_data = self.position_success_map[pos_str]
            success_rate = pos_data["success"] / max(1, sum(pos_data.values()))
            reasoning["position_data"] = {
                "success_rate": success_rate,
                "visits": sum(pos_data.values())
            }
            
        # Record world model confidence
        if 0 <= new_pos[0] < self.maze_size[0] and 0 <= new_pos[1] < self.maze_size[1]:
            reasoning["world_model_confidence"] = float(self.confidence_map[new_pos[0], new_pos[1]])
        
        if reward is not None:
            reasoning["reward"] = reward
            
        # Store reasoning for later meta-analysis
        self.decision_reasons.append(reasoning)
        self.memory["decision_reasoning"].append(reasoning)
        
    def _update_memory(self, path, success, reward, steps_taken):
        """Update agent memory with new experience"""
        compressed_path = self._compress_path(path)

        # Track success/failure counts
        pattern_entry = self.memory["patterns"].get(compressed_path, {
         "seen": 0,
            "successes": 0,
            "failures": 0,
            "avg_steps": 0.0,
            "reward": 0.0
        })

        pattern_entry["seen"] += 1
        if success:
            pattern_entry["successes"] += 1
            self.memory["meta"]["successful_trials"] = self.memory["meta"].get("successful_trials", 0) + 1
        else:
            pattern_entry["failures"] += 1

        # Update average reward and steps
        pattern_entry["reward"] += reward
        pattern_entry["avg_steps"] = (
            (pattern_entry["avg_steps"] * (pattern_entry["seen"] - 1)) + steps_taken
        ) / pattern_entry["seen"]

        self.memory["patterns"][compressed_path] = pattern_entry

        if hasattr(self, "visible_tiles") and hasattr(self, "last_move"):
            vision_key_parts = []
            for pos, val in sorted(self.visible_tiles.items()):
                vision_key_parts.append(f"({pos[0]}, {pos[1]}):{val}")
            vision_key = "|".join(vision_key_parts) + f"__move:{self.last_move}"

            vision_memory = self.memory.get("vision_patterns", {})
            entry = vision_memory.get(vision_key, {
                "seen": 0,
                "reward": 0.0
            })

            entry["seen"] += 1
            entry["reward"] += reward
            vision_memory[vision_key] = entry
            self.memory["vision_patterns"] = vision_memory

        # --- Terrain Arc Abstraction ---
        terrain_arc = []
        for pos in self.visited_positions:
            x, y = pos
            tile = self.maze[x][y]
            terrain_arc.append(tile)

        # Compress into arc string
        arc = "->".join(terrain_arc)

        # Initialize terrain_arcs memory
        if "terrain_arcs" not in self.memory:
            self.memory["terrain_arcs"] = {}
        
        if arc not in self.memory["terrain_arcs"]:
            self.memory["terrain_arcs"][arc] = {
                "seen": 1,
                "successes": 1 if success else 0,
                "failures": 0 if success else 1,
                "reward": reward,
                "avg_steps": steps_taken
            }
        else:
            arc_data = self.memory["terrain_arcs"][arc]
            arc_data["seen"] += 1
            if success:
                arc_data["successes"] += 1
            else:
                arc_data["failures"] += 1
            arc_data["reward"] = 0.7 * arc_data["reward"] + 0.3 * reward
            arc_data["avg_steps"] = (arc_data["avg_steps"] * (arc_data["seen"] - 1) + steps_taken) / arc_data["seen"]
        
            # Extract semantic terrain pattern
        terrain_path = []
        penalties_avoided = []
        for pos in self.visited_positions:
            x, y = pos
            tile = self.maze[x][y]
            terrain_path.append(tile)
            if tile in ['!', '~']:
                penalties_avoided.append((x, y))
        
        terrain_str = "-".join(terrain_path)
        
        # Initialize terrain_patterns memory
        if "terrain_patterns" not in self.memory:
            self.memory["terrain_patterns"] = {}
        
        if terrain_str not in self.memory["terrain_patterns"]:
            self.memory["terrain_patterns"][terrain_str] = {
                "seen": 1,
                "reward": reward,
                "steps": steps_taken,
                "avoided_penalties": penalties_avoided
            }        
        else:
            pattern_data = self.memory["terrain_patterns"][terrain_str]
            pattern_data["seen"] += 1
            pattern_data["reward"] = 0.7 * pattern_data["reward"] + 0.3 * reward
            pattern_data["steps"] = int((pattern_data["steps"] + steps_taken) / 2)
            pattern_data["avoided_penalties"] += penalties_avoided
        patterns = self._extract_patterns(path)
        
        # Update overall pattern statistics
        if compressed_path not in self.memory["patterns"]:
            self.memory["patterns"][compressed_path] = {
                "seen": 1,
                "successes": 1 if success else 0,
                "failures": 0 if success else 1,
                "avg_steps": steps_taken,
                "reward": reward
            }
        else:
            pattern = self.memory["patterns"][compressed_path]
            pattern["seen"] += 1
            if success:
                pattern["successes"] += 1
            else:
                pattern["failures"] += 1
            # Update running average of steps
            pattern["avg_steps"] = (pattern["avg_steps"] * (pattern["seen"] - 1) + steps_taken) / pattern["seen"]
            # Use exponential moving average for reward
            pattern["reward"] = 0.7 * pattern["reward"] + 0.3 * reward
        
        # Update micro and macro patterns
        for pattern_str, data in patterns.items():
            level = data["level"]
            level_key = f"{level}_patterns"
            
            if pattern_str not in self.memory[level_key]:
                self.memory[level_key][pattern_str] = {
                    "seen": 1,
                    "successes": 1 if success else 0,
                    "failures": 0 if success else 1,
                    "reward": reward,
                    "positions": [data["position"]]
                }
            else:
                pattern_data = self.memory[level_key][pattern_str]
                pattern_data["seen"] += 1
                if success:
                    pattern_data["successes"] += 1
                else:
                    pattern_data["failures"] += 1
                pattern_data["reward"] = 0.7 * pattern_data["reward"] + 0.3 * reward
                pattern_data["positions"].append(data["position"])
                if len(pattern_data["positions"]) > 20:  # Keep only recent positions
                    pattern_data["positions"] = pattern_data["positions"][-20:]
        
        # Update position success map
        for pos in self.visited_positions:
            pos_str = f"{pos[0]},{pos[1]}"
            if success:
                self.position_success_map[pos_str]["success"] += 1
            else:
                self.position_success_map[pos_str]["failure"] += 1
        
        # Update meta information
        self.memory["meta"]["total_trials"] += 1
        if success:
            self.memory["meta"]["successful_trials"] += 1
        self.memory["meta"]["total_steps"] += steps_taken
        
        # Record high reward patterns
        if reward > 5:
            high_reward = {
                "pattern": compressed_path,
                "reward": reward,
                "success": success,
                "trial": self.memory["meta"]["total_trials"]
            }
            self.memory["meta"]["high_reward_patterns"].append(high_reward)
            # Keep only top 20 high reward patterns
            self.memory["meta"]["high_reward_patterns"].sort(key=lambda x: x["reward"], reverse=True)
            self.memory["meta"]["high_reward_patterns"] = self.memory["meta"]["high_reward_patterns"][:20]
        
        # Save world model snapshot
        world_model_snapshot = {
            "trial": self.memory["meta"]["total_trials"],
            "explored_percentage": np.count_nonzero(self.world_model) / (self.maze_size[0] * self.maze_size[1]) * 100,
            "success": success
        }
        self.memory["world_model_history"].append(world_model_snapshot)
        
        # Extract meta-patterns
        self._extract_meta_patterns()
        
        # Save memory to file
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def _simulate_path(self, start_pos, moves, max_steps=None):
        """Mentally simulate a path through the world model"""
        if max_steps is None:
            max_steps = len(moves)
            
        current_pos = start_pos
        path = []
        positions = [current_pos]
        
        for i, move in enumerate(moves[:max_steps]):
            new_pos = self._get_next_position(current_pos, move)
            x, y = new_pos
            
            # Check if we would hit a wall or boundary in our world model
            if not (0 <= x < self.maze_size[0] and 0 <= y < self.maze_size[1]):
                return positions, path, False  # Hit boundary
                
            if self.world_model[x, y] == 2:  # Known wall
                return positions, path, False  # Hit wall
                
            current_pos = new_pos
            path.append(move)
            positions.append(current_pos)
            
            # Check if we reached the goal
            if current_pos == self.goal_position:
                return positions, path, True  # Success
                
        return positions, path, None  # Neither success nor failure
    
    def _choose_next_move(self):
        """Choose the next move based on memory, world model, and pattern recognition"""
        valid_moves = [move for move in ["U", "D", "L", "R"] if self._is_valid_move(move)]
        if not valid_moves:
            return None

        if (random.random() < self.exploration_rate) or (random.random() < self.curiosity_burst_chance):
            move = random.choice(valid_moves)
            self.current_patterns = []
            return move

        current_path = self.current_path.copy()
        current_pos = self.current_position
        best_move = None
        best_score = float('-inf')
        move_reasoning = {}

        if self.memory["meta"]["total_trials"] > 10:
            best_path_to_goal = self._find_path_to_goal()
            if best_path_to_goal and len(best_path_to_goal) > 0:
                self.current_patterns = ["world_model_path"]
                return best_path_to_goal[0]

        for move in valid_moves:
            test_path = current_path + [move]
            test_compressed = self._compress_path(test_path)
            new_pos = self._get_next_position(current_pos, move)

            lookback = list(self.visited_positions)[-2:] if len(self.visited_positions) >= 2 else list(self.visited_positions)
            lookahead = lookback + [new_pos]
            terrain_arc_score = self._terrain_arc_score(lookahead)

            pattern_score = 0
            position_score = 0
            goal_distance_score = 0
            exploration_score = 0
            meta_pattern_score = 0
            vision_score = 0

            # Vision terrain type
            if new_pos in self.visible_tiles:
                terrain = self.visible_tiles[new_pos]
                if terrain == '+':
                    vision_score = 1.0
                elif terrain == '>':
                    vision_score = 0.8
                elif terrain == '!':
                    vision_score = -1.5
                elif terrain == '~':
                    vision_score = -0.5
                elif terrain == 'G':
                    vision_score = 10.0

            # === Vision Pattern Memory Integration ===
            vision_key_parts = []
            for pos, val in sorted(self.visible_tiles.items()):
                vision_key_parts.append(f"({pos[0]}, {pos[1]}):{val}")
            vision_key = "|".join(vision_key_parts) + f"__move:{move}"

            if vision_key in self.memory.get("vision_patterns", {}):
                vision_pattern = self.memory["vision_patterns"][vision_key]
                reward_boost = vision_pattern["reward"] / max(1, vision_pattern["seen"])
                vision_score += reward_boost

            # Micro-pattern match
            for size in range(min(4, len(test_path)), 1, -1):
                micro = self._compress_path(test_path[-size:])
                if micro in self.memory["micro_patterns"]:
                    pattern_data = self.memory["micro_patterns"][micro]
                    success_rate = pattern_data["successes"] / max(1, pattern_data["seen"])
                    pattern_score += pattern_data["reward"] * success_rate

                    if micro in self.memory["meta_patterns"]:
                        meta_type = self.memory["meta_patterns"][micro]
                        if meta_type == "loop" and success_rate < 0.3:
                            meta_pattern_score -= 2
                        elif meta_type == "wall_following" and success_rate > 0.7:
                            meta_pattern_score += 1
                        elif meta_type == "recurring_success":
                            meta_pattern_score += 1.5

            # Position-based score
            pos_str = f"{new_pos[0]},{new_pos[1]}"
            if pos_str in self.position_success_map:
                pos_data = self.position_success_map[pos_str]
                visits = sum(pos_data.values())
                if visits > 0:
                    success_rate = pos_data["success"] / max(1, visits)
                    position_score = (success_rate - 0.5) * min(visits / 5, 1)

            goal_distance = abs(new_pos[0] - self.goal_position[0]) + abs(new_pos[1] - self.goal_position[1])
            max_distance = self.maze_size[0] + self.maze_size[1]
            goal_distance_score = 1 - (goal_distance / max_distance)

            # Combined move score with vision pattern bias
            final_score = (
                pattern_score * 0.30 +
                position_score * 0.15 +
                goal_distance_score * 0.10 +
                exploration_score * 0.15 +
                meta_pattern_score * 0.10 +
                terrain_arc_score * 0.15 +
                vision_score * 0.25
            )

            move_reasoning[move] = {
                "pattern_score": pattern_score,
                "position_score": position_score,
                "goal_distance_score": goal_distance_score,
                "exploration_score": exploration_score,
                "meta_pattern_score": meta_pattern_score,
                "terrain_arc_score": terrain_arc_score,
                "vision_score": vision_score,
                "final_score": final_score
            }

            final_score += random.random() * 0.1  # Add randomness

            if final_score > best_score:
                best_score = final_score
                best_move = move

        self.current_patterns = []
        if best_move in move_reasoning:
            for factor, value in move_reasoning[best_move].items():
                if value > 0:
                    self.current_patterns.append(f"{factor}:{value:.2f}")

        if best_move is None or best_score <= 0:
            best_move = random.choice(valid_moves)
            self.current_patterns = ["random_choice"]

        if not hasattr(self, "visual_context_history"):
            self.visual_context_history = []

        self.visual_context_history.append({
            "position": self.current_position,
            "visible_tiles": dict(self.visible_tiles),
            "chosen_move": best_move
        })

        return best_move

    
    def _find_path_to_goal(self):
        """Try to find a path to the goal using the world model"""
        # Simple breadth-first search through known paths
        start = self.current_position
        goal = self.goal_position
        
        # If world model is too sparse, don't try pathfinding yet
        known_cells = np.count_nonzero(self.world_model)
        if known_cells < (self.maze_size[0] * self.maze_size[1]) * 0.3:
            return None
        
        # Queue for BFS: (position, path_to_here)
        queue = [(start, [])]
        visited = set([start])
        
        while queue:
            (x, y), path = queue.pop(0)
            
            # Check if we reached the goal
            if (x, y) == goal:
                return path
                
            # Try each direction
            for move, (dx, dy) in zip(["U", "D", "L", "R"], [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nx, ny = x + dx, y + dy
                
                # Skip if outside maze or already visited
                if not (0 <= nx < self.maze_size[0] and 0 <= ny < self.maze_size[1]):
                    continue
                if (nx, ny) in visited:
                    continue
                    
                # Skip if we know this is a wall
                if self.world_model[nx, ny] == 2:  # Wall
                    continue
                    
                # Skip if unknown (unless we're allowing exploration)
                if self.world_model[nx, ny] == 0 and random.random() > self.exploration_rate:
                    continue
                    
                # Add to queue
                queue.append(((nx, ny), path + [move]))
                visited.add((nx, ny))
                
        # No path found
        return None
    
    def _update_exploration_rate(self, success):
        """Update exploration rate based on success and current stage"""
        if success:
            # Decrease exploration faster after success
            self.exploration_rate *= self.decay_rate
        else:
            # Decrease exploration more slowly after failure
            self.exploration_rate *= (self.decay_rate + 0.02)
            
        # Ensure exploration doesn't go below minimum
        self.exploration_rate = max(self.min_exploration, self.exploration_rate)
        
        # Occasionally add curiosity burst if we're stuck in local optima
        if len(self.learning_curve) >= 10:
            recent_rewards = [entry.get("reward", 0) for entry in self.learning_curve[-10:]]
            if max(recent_rewards) - min(recent_rewards) < 1.0:  # Little variation in rewards
                self.exploration_rate = min(0.6, self.exploration_rate * 1.2)  # Boost exploration
                self._log_insight("Increased exploration to escape local optimum", 0)
    
    def _detect_loop(self, path, threshold=3):
        """Detect if the agent is stuck in a movement loop"""
        if len(path) < threshold * 2:
            return False
            
        # Check for repeating sequences
        for i in range(1, threshold + 1):
            if path[-i:] == path[-2*i:-i]:
                return True
                
        # Check for position loops (revisiting same positions)
        positions = []
        current_pos = (0, 0)  # Start position
        
        for move in path:
            current_pos = self._get_next_position(current_pos, move)
            positions.append(current_pos)
            
        # Check for position repeats in last N moves
        position_counts = Counter(positions[-15:])
        if position_counts.most_common(1)[0][1] >= 4:  # Same position 4+ times in last 15 moves
            return True
                
        return False
    
    def _update_heatmap(self):
        """Update the position heatmap"""
        x, y = self.current_position
        self.heatmap[x][y] += 1
    
    def _print_maze(self):
        """Print the current maze layout with start (S) and goal (G) marked"""
        for x in range(self.maze_size[0]):
            row = ""
            for y in range(self.maze_size[1]):
                if (x, y) == (0, 0):
                    row += "S "
                elif (x, y) == self.goal_position:
                    row += "G "
                else:
                    row += self.maze[x][y] + " "
            print(row)
        print("-" * 40)

    def run_trial(self):
        """Run a single trial of maze navigation"""
        # Reset for new trial
        self.current_position = (0, 0)
        self.current_path = []
        self.visited_positions = set([self.current_position])
        self.decision_reasons = []
        self._print_maze()

        max_steps = self.maze_size[0] * self.maze_size[1] * 4  # Allow more steps for larger mazes
        steps_taken = 0
        result = "timeout"
        
        while steps_taken < max_steps:
            self._update_heatmap()
            
            # Check if goal reached
            x, y = self.current_position
            if self.current_position == self.goal_position or self.maze[x][y] in {'+', '>', 'G'}:
                result = "success"
                break
                
            # Detect movement loops
            if self._detect_loop(self.current_path):
                result = "loop"
                self._log_regret("Got stuck in a movement loop", -1)
                break
                
            # Choose and make next move
            next_move = self._choose_next_move()
            if next_move is None:
                result = "dead_end"
                break
                
            # Update position based on move
            old_position = self.current_position
            self.current_position = self._get_next_position(self.current_position, next_move)
                
            # Update path and visited positions
            self.current_path.append(next_move)
            self.visited_positions.add(self.current_position)
            steps_taken += 1
            
            # Update world model with new information
            self._update_vision_field()
            self._update_world_model(self.current_position)
            
            # Reflect on this decision
            self._reflect_on_decision(next_move, "in_progress")
            
        # Update final decision outcome
        if self.decision_reasons:
            self.decision_reasons[-1]["outcome"] = result
        
        # Calculate success and reward
        success = (result == "success")
        reward = self._calculate_reward(self.current_path, success, steps_taken)
        
        # Update memory and learning parameters
        self._update_memory(self.current_path, success, reward, steps_taken)
        self._update_exploration_rate(success)
        
        # Print trial results
        compressed_path = self._compress_path(self.current_path)
        print(f"Trial {self.memory['meta']['total_trials']} result: {result}")
        print(f"Steps: {steps_taken}, Path: {compressed_path}")
        print(f"Reward: {reward:.2f}, Exploration rate: {self.exploration_rate:.2f}")
        print("-" * 40)
        
        # Update learning curve
        self.learning_curve.append({
            "trial": self.memory["meta"]["total_trials"],
            "result": result,
            "steps": steps_taken,
            "reward": reward
        })
        # === NEW: Learn from visual context history ===
        if hasattr(self, "visual_context_history"):
            for context in self.visual_context_history:
                tiles = context["visible_tiles"]
                move = context["chosen_move"]

                # Build a key like: down:+|right:>__move:R
                context_key_parts = [f"{direction}:{terrain}" for direction, terrain in tiles.items()]
                key = "|".join(sorted(context_key_parts)) + f"__move:{move}"

                if "vision_patterns" not in self.memory:
                    self.memory["vision_patterns"] = {}

                if key not in self.memory["vision_patterns"]:
                    self.memory["vision_patterns"][key] = {"seen": 0, "reward": 0.0}

                self.memory["vision_patterns"][key]["seen"] += 1
                self.memory["vision_patterns"][key]["reward"] += reward
        return success, reward, steps_taken, result
        
    def visualize_learning(self):
        """Visualize the agent's learning progress"""
        # Create figure with subplots - added an extra plot for the world model
        fig, axs = plt.subplots(3, 2, figsize=(15, 18))
        
        # Plot 1: Learning curve (reward over time)
        trials = [entry["trial"] for entry in self.learning_curve]
        rewards = [entry["reward"] for entry in self.learning_curve]
        success_trials = [i for i, entry in enumerate(self.learning_curve) if entry["result"] == "success"]
        success_rewards = [rewards[i] for i in success_trials]
        
        axs[0, 0].plot(trials, rewards, 'b-', alpha=0.6)
        axs[0, 0].scatter([trials[i] for i in success_trials], success_rewards, color='green', marker='o')
        axs[0, 0].set_title('Reward Over Time')
        axs[0, 0].set_xlabel('Trial')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)
        
        # Plot 2: Steps to completion
        steps = [entry["steps"] for entry in self.learning_curve]
        success_steps = [steps[i] for i in success_trials]
        
        axs[0, 1].plot(trials, steps, 'r-', alpha=0.6)
        axs[0, 1].scatter([trials[i] for i in success_trials], success_steps, color='green', marker='o')
        axs[0, 1].set_title('Steps per Trial')
        axs[0, 1].set_xlabel('Trial')
        axs[0, 1].set_ylabel('Steps')
        axs[0, 1].grid(True)
        
        # Plot 3: Maze heatmap with semantic terrain colors
        maze_img = np.zeros((self.maze_size[0], self.maze_size[1], 3))

        for x in range(self.maze_size[0]):
            for y in range(self.maze_size[1]):
                tile = self.maze[x][y]
                if tile == '#':
                    maze_img[x, y] = [0, 0, 0]  # Wall - black
                elif tile == '.':
                    maze_img[x, y] = [0.8, 0.8, 0.8]  # Neutral path - light gray
                elif tile == '~':
                    maze_img[x, y] = [0.5, 0.3, 0.2]  # Mud - brown
                elif tile == '+':
                    maze_img[x, y] = [0.2, 0.6, 1.0]  # Reward - blue
                elif tile == '!':
                    maze_img[x, y] = [1.0, 0.2, 0.2]  # Penalty - red
                elif tile == '>':
                    maze_img[x, y] = [1.0, 1.0, 0.3]  # Goal-adjacent - yellow
                elif tile == 'S':
                    maze_img[x, y] = [0, 1, 0]  # Start - green
                elif tile == 'G':
                    maze_img[x, y] = [1, 0, 0]  # Goal - red

        # Overlay visit frequency as blue tint
        max_val = np.max(self.heatmap) if np.max(self.heatmap) > 0 else 1
        normalized_heatmap = self.heatmap / max_val
        for x in range(self.maze_size[0]):
            for y in range(self.maze_size[1]):
                maze_img[x, y, 2] += 0.3 * normalized_heatmap[x, y]
        axs[1, 0].imshow(maze_img)
        axs[1, 0].set_title('Maze with Visit Heatmap')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        
        # Plot 4: Pattern distribution
        if self.memory["patterns"]:
            patterns = sorted(self.memory["patterns"].items(), key=lambda x: x[1]["reward"], reverse=True)[:10]
            pattern_names = [p[0] if len(p[0]) < 15 else p[0][:12]+"..." for p in patterns]
            pattern_rewards = [p[1]["reward"] for p in patterns]
            
            axs[1, 1].barh(pattern_names, pattern_rewards, color='purple')
            axs[1, 1].set_title('Top 10 Patterns by Reward')
            axs[1, 1].set_xlabel('Reward')
            
        # Plot 5: World Model Visualization
        world_model_img = np.zeros((self.maze_size[0], self.maze_size[1], 3))
        
        # Color the world model: green=known path, red=known wall, gray=unknown
        for x in range(self.maze_size[0]):
            for y in range(self.maze_size[1]):
                if self.world_model[x, y] == 1:  # Known path
                    # Intensity based on confidence
                    intensity = self.confidence_map[x, y]
                    world_model_img[x, y] = [0, intensity, 0]
                elif self.world_model[x, y] == 2:  # Known wall
                    intensity = self.confidence_map[x, y]
                    world_model_img[x, y] = [intensity, 0, 0]
                else:  # Unknown
                    world_model_img[x, y] = [0.3, 0.3, 0.3]
        
        # Mark start and goal
        world_model_img[0, 0] = [0, 1, 1]  # Cyan for start
        world_model_img[self.goal_position[0], self.goal_position[1]] = [1, 1, 0]  # Yellow for goal
        
        axs[2, 0].imshow(world_model_img)
        axs[2, 0].set_title('World Model')
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])
        
        # Plot 6: Insights and Regrets
        insight_trials = [insight["trial"] for insight in self.insight_log]
        insight_impacts = [insight["reward_impact"] for insight in self.insight_log]
        
        regret_trials = [regret["trial"] for regret in self.regret_history]
        regret_impacts = [regret["reward_impact"] for regret in self.regret_history]
        
        # Combined plot
        axs[2, 1].scatter(insight_trials, insight_impacts, color='green', label='Insights', alpha=0.7)
        axs[2, 1].scatter(regret_trials, regret_impacts, color='red', label='Regrets', alpha=0.7)
        axs[2, 1].set_title('Self-Reflection: Insights vs Regrets')
        axs[2, 1].set_xlabel('Trial')
        axs[2, 1].set_ylabel('Reward Impact')
        axs[2, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axs[2, 1].legend()
        axs[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('enhanced_learning_visualization.png')
        plt.close()
        
    def print_summary(self):
        """Print a summary of what the agent has learned with added reflection"""
        print("\n===== AGENT LEARNING SUMMARY =====")
        print(f"Total trials: {self.memory['meta']['total_trials']}")
        print(f"Successful trials: {self.memory['meta']['successful_trials']}")
        success_rate = self.memory['meta']['successful_trials'] / max(1, self.memory['meta']['total_trials'])
        print(f"Success rate: {success_rate:.2%}")
        print(f"Total patterns discovered: {len(self.memory['patterns'])}")
        print(f"World model completion: {np.count_nonzero(self.world_model) / (self.maze_size[0] * self.maze_size[1]):.1%}")
        self._save_memory()


        # Top patterns
        if self.memory["patterns"]:
            print("\nTop 5 highest reward patterns:")
            patterns = sorted(self.memory["patterns"].items(), key=lambda x: x[1]["reward"], reverse=True)[:5]
            for i, (pattern, data) in enumerate(patterns, 1):
                print(f"{i}. {pattern}: Reward={data['reward']:.2f}, Success={data['successes']}/{data['seen']}")
                
        # Learning from failure
        print("\nTop patterns that overcame failure:")
        failure_learners = []
        for pattern, data in self.memory["patterns"].items():
            if data["failures"] > 0 and data["successes"] > 0:
                recovery_ratio = data["successes"] / max(1, data["failures"])
                failure_learners.append((pattern, data, recovery_ratio))
                
        failure_learners.sort(key=lambda x: x[2], reverse=True)
        for i, (pattern, data, ratio) in enumerate(failure_learners[:5], 1):
            print(f"{i}. {pattern}: Recovered {data['successes']} times after {data['failures']} failures")
        
        # Meta-patterns discovered
        if self.memory["meta_patterns"]:
            print("\nMeta-patterns discovered:")
            meta_types = {}
            for pattern, meta_type in self.memory["meta_patterns"].items():
                if meta_type not in meta_types:
                    meta_types[meta_type] = 0
                meta_types[meta_type] += 1
            
            for meta_type, count in meta_types.items():
                print(f"- {meta_type}: {count} patterns")
                
        # Key insights
        if self.memory["insights"]:
            print("\nKey insights:")
            insights = sorted(self.memory["insights"], key=lambda x: x["reward_impact"], reverse=True)[:5]
            for i, insight in enumerate(insights, 1):
                print(f"{i}. Trial {insight['trial']}: {insight['insight']} (impact: {insight['reward_impact']})")
                
        # Key regrets
        if self.memory["regrets"]:
            print("\nKey learning from mistakes:")
            regrets = sorted(self.memory["regrets"], key=lambda x: x["reward_impact"])[:5]
            for i, regret in enumerate(regrets, 1):
                print(f"{i}. Trial {regret['trial']}: {regret['regret']} (impact: {regret['reward_impact']})")
        
        # Self-reflection on learning process
        print("\nLearning Process Reflection:")
        if len(self.learning_curve) >= 10:
            first_10_success = sum(1 for entry in self.learning_curve[:10] if entry["result"] == "success")
            last_10_success = sum(1 for entry in self.learning_curve[-10:] if entry["result"] == "success")
            
            first_10_steps = [entry["steps"] for entry in self.learning_curve[:10] if entry["result"] == "success"]
            last_10_steps = [entry["steps"] for entry in self.learning_curve[-10:] if entry["result"] == "success"]
            
            avg_first_steps = sum(first_10_steps) / max(1, len(first_10_steps))
            avg_last_steps = sum(last_10_steps) / max(1, len(last_10_steps))
            
            print(f"- Success rate improvement: {first_10_success/10:.1%} → {last_10_success/10:.1%}")
            
            if len(first_10_steps) > 0 and len(last_10_steps) > 0:
                print(f"- Efficiency improvement: {avg_first_steps:.1f} → {avg_last_steps:.1f} steps")
                
            if self.memory["meta"]["successful_trials"] > 0:
                print(f"- Found first success on trial #{next((i for i, e in enumerate(self.learning_curve) if e['result'] == 'success'), 'N/A')}")
                
        print("\nVisualization saved as 'enhanced_learning_visualization.png'")
        print("====================================")

def run_simulation(maze_size=(10, 10), trials=100, verbose=True):
    """Run a complete simulation with the agent"""
    agent = RewardPathwayAgent(maze_size=maze_size)
    
    if verbose:
        print(f"Starting simulation with {trials} trials on a {maze_size[0]}x{maze_size[1]} maze")
        print("Initial maze configuration:")
        
        # Print the maze
        for x in range(maze_size[0]):
            row = ""
            for y in range(maze_size[1]):
                if (x, y) == (0, 0):
                    row += "S "  # Start
                elif (x, y) == (maze_size[0]-1, maze_size[1]-1):
                    row += "G "  # Goal
                else:
                    row += agent.maze[x][y] + " "
            print(row)
    
    print("\nRunning trials...")
    for i in range(trials):
        agent.run_trial()
        
        # Periodically visualize progress
        if verbose and ((i+1) % 20 == 0 or i == trials-1):
            agent.visualize_learning()
            print(f"Completed {i+1}/{trials} trials")
    
    agent.print_summary()
    return agent

def analyze_performance(maze_sizes=[(5, 5), (10, 10), (15, 15)], trials_per_size=100):
    """Analyze agent performance across different maze sizes"""
    results = {}
    
    for size in maze_sizes:
        print(f"\nTesting maze size {size[0]}x{size[1]}")
        agent = run_simulation(maze_size=size, trials=trials_per_size, verbose=False)
        
        # Collect statistics
        results[f"{size[0]}x{size[1]}"] = {
            "success_rate": agent.memory["meta"]["successful_trials"] / agent.memory["meta"]["total_trials"],
            "avg_steps_to_success": np.mean([entry["steps"] for entry in agent.learning_curve if entry["result"] == "success"]) 
                if any(entry["result"] == "success" for entry in agent.learning_curve) else float('inf'),
            "patterns_found": len(agent.memory["patterns"]),
            "meta_patterns": len(agent.memory["meta_patterns"]),
            "world_model_completion": np.count_nonzero(agent.world_model) / (size[0] * size[1])
        }
    
    # Plot comparison
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Success rate by maze size
    sizes = list(results.keys())
    success_rates = [results[size]["success_rate"] for size in sizes]
    
    axs[0].bar(sizes, success_rates, color='green')
    axs[0].set_title('Success Rate by Maze Size')
    axs[0].set_xlabel('Maze Size')
    axs[0].set_ylabel('Success Rate')
    axs[0].grid(True, axis='y')
    
    # Average steps to success
    avg_steps = [results[size]["avg_steps_to_success"] for size in sizes]
    avg_steps = [step if step != float('inf') else 0 for step in avg_steps]  # Handle no successes
    
    axs[1].bar(sizes, avg_steps, color='blue')
    axs[1].set_title('Average Steps to Success by Maze Size')
    axs[1].set_xlabel('Maze Size')
    axs[1].set_ylabel('Steps')
    axs[1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('maze_size_comparison.png')
    plt.close()
    
    # Print detailed results
    print("\n===== PERFORMANCE ANALYSIS =====")
    for size, stats in results.items():
        print(f"\nMaze Size: {size}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Avg Steps to Success: {stats['avg_steps_to_success']:.1f}")
        print(f"Patterns Discovered: {stats['patterns_found']}")
        print(f"Meta-patterns Extracted: {stats['meta_patterns']}")
        print(f"World Model Completion: {stats['world_model_completion']:.1%}")
    
    print("\nComparison visualization saved as 'maze_size_comparison.png'")
    print("=============================")
    
    return results

if __name__ == "__main__":
    # Run the simulation with a 10x10 maze for 100 trials
    agent = run_simulation(maze_size=(10, 10), trials=100)
    
    # Optional: Analyze performance across different maze sizes
    # analyze_performance()
