from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sim.envs.base.mujoco_env import MujocoEnv, MujocoCfg


@dataclass
class OptimizeParam:
    """Definition of a parameter to optimize"""
    name: str
    range: Tuple[float, float]  # [min, max]
    init_val: float = None  # none -> random initialization
    mutation_range: Tuple[float, float] = (0.8, 1.2)  # relative mutation bounds


class ParameterOptimizer:
    def __init__(
        self,
        base_cfg: MujocoCfg,
        parameters: List[OptimizeParam],
        n_iterations: int = 50,
        population_size: int = 20,
        elite_frac: float = 0.2,
    ):
        self.base_cfg = base_cfg
        self.parameters = parameters
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_frac))

    def _get_cfg_value(self, cfg: MujocoCfg, param_path: str) -> Any:
        """Get a value from nested config using dot notation"""
        value = cfg
        for part in param_path.split('.'):
            value = getattr(value, part)
        return value

    def _set_cfg_value(self, cfg: MujocoCfg, param_path: str, value: Any) -> None:
        """Set a value in nested config using dot notation"""
        parts = param_path.split('.')
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _create_parameters(self) -> List[Dict[str, float]]:
        """Create a population of parameter sets"""
        population = []
        
        for _ in range(self.population_size):
            params = {}
            for param in self.parameters:
                if param.init_val is not None:
                    # Initialize near the provided value
                    value = param.init_val * np.random.uniform(0.9, 1.1)
                else:
                    # Random initialization within bounds
                    value = np.random.uniform(param.range)
                params[param.name] = np.clip(value, param.range[0], param.range[1])
            population.append(params)
            
        return population

    def _evaluate_parameters(self, params: Dict[str, float], n_episodes: int = 3) -> float:
        """Evaluate a parameter set over multiple episodes"""
        # Create a copy of the base config
        cfg = MujocoCfg()
        cfg.__dict__.update(self.base_cfg.__dict__)
        
        # Update config with new parameters
        for param_name, value in params.items():
            self._set_cfg_value(cfg, param_name, value)
        
        env = MujocoEnv(cfg, render=False)
        total_reward = 0.0
        
        for _ in range(n_episodes):
            env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # For testing stability, use zero actions
                action = np.zeros(env.num_joints)
                _, reward, done, _ = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        env.close()
        return total_reward / n_episodes

    def _create_new_population(
        self,
        population: List[Dict[str, float]],
        scores: np.ndarray
    ) -> List[Dict[str, float]]:
        """Create new population from elite performers"""
        # Sort by score
        elite_idx = np.argsort(scores)[-self.n_elite:]
        elite_population = [population[i] for i in elite_idx]
        
        # Create new population through mutation of elite members
        new_population = []
        while len(new_population) < self.population_size:
            # Select random elite member
            parent = elite_population[np.random.randint(self.n_elite)]
            
            # Mutate parameters
            child = {}
            for param in self.parameters:
                # Apply random mutation within specified range
                mutation = np.random.uniform(*param.mutation_range)
                value = parent[param.name] * mutation
                child[param.name] = np.clip(value, param.range[0], param.range[1])
            
            new_population.append(child)
            
        return new_population

    def optimize(self) -> Tuple[Dict[str, float], List[float]]:
        """Run parameter optimization"""
        best_score = float('-inf')
        best_params = None
        history = []
        
        # Initial population
        population = self._create_parameters()
        
        for iteration in range(self.n_iterations):
            # Evaluate population
            scores = np.array([
                self._evaluate_parameters(params)
                for params in population
            ])
            
            # Track best performer
            max_score = np.max(scores)
            if max_score > best_score:
                best_score = max_score
                best_params = population[np.argmax(scores)]
            
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            print(f"Best score: {best_score:.2f}")
            print("Best parameters:")
            for name, value in best_params.items():
                print(f"  {name}: {value:.3f}")
            
            history.append(best_score)
            
            # Create new population
            population = self._create_new_population(population, scores)
            
        return best_params, history



from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sim.envs.base.mujoco_env import MujocoEnv, MujocoCfg


@dataclass
class OptimizeParam:
    """Definition of a parameter to optimize"""
    name: str  # Full parameter path (e.g., 'gains.kp_scale')
    min_val: float
    max_val: float
    init_val: float = None  # If None, will be randomly initialized
    mutation_strength: float = 0.3  # Controls the strength of mutations


class ParameterOptimizer:
    def __init__(
        self,
        base_cfg: MujocoCfg,
        parameters: List[OptimizeParam],
        n_iterations: int = 50,
        population_size: int = 40,  # Increased population size
        elite_frac: float = 0.1,    # Reduced elite fraction
        exploration_prob: float = 0.2,  # Probability of random exploration
        n_episodes: int = 5,  # More episodes for better evaluation
        adaptive_mutation: bool = True,
    ):
        self.base_cfg = base_cfg
        self.parameters = parameters
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_frac))
        self.exploration_prob = exploration_prob
        self.n_episodes = n_episodes
        self.adaptive_mutation = adaptive_mutation
        self.mutation_strength = 1.0  # Will be adapted during optimization
        
        # Initialize parameter ranges for normalization
        self.param_ranges = {
            param.name: (param.max_val - param.min_val)
            for param in parameters
        }

    def _get_cfg_value(self, cfg: MujocoCfg, param_path: str) -> Any:
        """Get a value from nested config using dot notation"""
        value = cfg
        for part in param_path.split('.'):
            value = getattr(value, part)
        return value

    def _set_cfg_value(self, cfg: MujocoCfg, param_path: str, value: Any) -> None:
        """Set a value in nested config using dot notation"""
        parts = param_path.split('.')
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _create_parameters(self) -> List[Dict[str, float]]:
        """Create a population of parameter sets with improved initialization"""
        population = []
        
        # Create structured initial population
        for _ in range(self.population_size):
            params = {}
            if np.random.random() < self.exploration_prob:
                # Random exploration
                for param in self.parameters:
                    value = np.random.uniform(param.min_val, param.max_val)
                    params[param.name] = value
            else:
                # Systematic initialization around init values
                for param in self.parameters:
                    if param.init_val is not None:
                        # Log-uniform sampling around init value
                        log_range = np.log([0.1, 10.0])  # Sample in range 0.1x to 10x
                        log_multiplier = np.random.uniform(*log_range)
                        value = param.init_val * np.exp(log_multiplier)
                    else:
                        # Latin Hypercube style sampling
                        segment = (param.max_val - param.min_val) / self.population_size
                        value = param.min_val + segment * (np.random.random() + _)
                    
                    params[param.name] = np.clip(value, param.min_val, param.max_val)
            
            population.append(params)
            
        return population

    def _evaluate_parameters(self, params: Dict[str, float]) -> float:
        """Evaluate a parameter set with improved robustness"""
        cfg = MujocoCfg()
        cfg.__dict__.update(self.base_cfg.__dict__)
        
        for param_name, value in params.items():
            self._set_cfg_value(cfg, param_name, value)
        
        env = MujocoEnv(cfg, render=False)
        rewards = []
        
        try:
            for _ in range(self.n_episodes):
                env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    # Sinusoidal test actions for better stability evaluation
                    t = episode_length * env.cfg.sim.dt
                    action = 0.5 * np.sin(2 * np.pi * t)
                    _, reward, done, info = env.step(action * np.ones(env.num_joints))
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Early termination for unstable episodes
                    if info.get('fall', False):
                        episode_reward *= 0.5  # Penalty for falling
                        break
                    
                rewards.append(episode_reward)
            
            # Calculate score with stability consideration
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            stability_penalty = 0.1 * std_reward  # Penalize inconsistent performance
            
            return mean_reward - stability_penalty
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return float('-inf')
        finally:
            env.close()

    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Perform crossover between two parents"""
        child = {}
        for param in self.parameters:
            # Interpolate between parents with random weight
            weight = np.random.random()
            value = weight * parent1[param.name] + (1 - weight) * parent2[param.name]
            child[param.name] = np.clip(value, param.min_val, param.max_val)
        return child

    def _mutate(self, params: Dict[str, float], generation: int) -> Dict[str, float]:
        """Apply mutation with adaptive strength"""
        mutated = {}
        for param in self.parameters:
            value = params[param.name]
            
            # Combine multiplicative and additive mutation
            if np.random.random() < 0.5:
                # Multiplicative mutation (good for parameters with large ranges)
                mutation = np.random.lognormal(0, param.mutation_strength * self.mutation_strength)
                value *= mutation
            else:
                # Additive mutation (good for fine-tuning)
                range_size = self.param_ranges[param.name]
                mutation = np.random.normal(0, param.mutation_strength * self.mutation_strength * range_size)
                value += mutation
            
            mutated[param.name] = np.clip(value, param.min_val, param.max_val)
        
        return mutated

    def _create_new_population(
        self,
        population: List[Dict[str, float]],
        scores: np.ndarray,
        generation: int
    ) -> List[Dict[str, float]]:
        """Create new population with improved diversity"""
        # Sort by score
        sorted_indices = np.argsort(scores)
        elite_idx = sorted_indices[-self.n_elite:]
        elite_population = [population[i] for i in elite_idx]
        
        # Adapt mutation strength based on improvement
        if self.adaptive_mutation and generation > 0:
            improvement = (scores[elite_idx[-1]] - scores[elite_idx[0]]) / abs(scores[elite_idx[0]])
            if improvement < 0.01:  # Small improvement
                self.mutation_strength *= 1.2  # Increase exploration
            elif improvement > 0.05:  # Large improvement
                self.mutation_strength *= 0.8  # Reduce exploration
            self.mutation_strength = np.clip(self.mutation_strength, 0.5, 2.0)
        
        new_population = elite_population.copy()
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            if np.random.random() < self.exploration_prob:
                # Random exploration
                params = {
                    param.name: np.random.uniform(param.min_val, param.max_val)
                    for param in self.parameters
                }
                new_population.append(params)
            else:
                # Tournament selection
                tournament_size = 3
                parent1 = max(
                    np.random.choice(population, tournament_size),
                    key=lambda x: scores[population.index(x)]
                )
                parent2 = max(
                    np.random.choice(population, tournament_size),
                    key=lambda x: scores[population.index(x)]
                )
                
                # Crossover and mutation
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, generation)
                new_population.append(child)
        
        return new_population

    def optimize(self) -> Tuple[Dict[str, float], List[float]]:
        """Run parameter optimization with improved logging"""
        best_score = float('-inf')
        best_params = None
        history = []
        
        population = self._create_parameters()
        
        for iteration in range(self.n_iterations):
            # Evaluate population
            scores = np.array([
                self._evaluate_parameters(params)
                for params in population
            ])
            
            # Update best performer
            max_score = np.max(scores)
            if max_score > best_score:
                best_score = max_score
                best_params = population[np.argmax(scores)]
            
            # Logging
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            print(f"Best score: {best_score:.2f}")
            print(f"Current mutation strength: {self.mutation_strength:.2f}")
            print("Best parameters:")
            for name, value in best_params.items():
                print(f"  {name}: {value:.3f}")
            
            history.append(best_score)
            
            # Create new population
            population = self._create_new_population(population, scores, iteration)
        
        return best_params, history

if __name__ == "__main__":
    cfg = MujocoCfg()
    parameters = [
        OptimizeParam(
            name='gains.kp_scale',
            min_val=0.1,
            max_val=5.0,
            init_val=1.0,
        ),
        OptimizeParam(
            name='gains.kd_scale',
            min_val=0.1,
            max_val=5.0,
            init_val=1.0,
        ),
        # OptimizeParam(
        #     name='gains.tau_scale',
        #     min_val=1.0,
        #     max_val=20.0,
        #     init_val=4.0,
        # ),
        # OptimizeParam(
        #     name='extras.tau_factor',
        #     min_val=1.0,
        #     max_val=20.0,
        #     init_val=1.0,
        # ),
    ]
    optimizer = ParameterOptimizer(
        cfg,
        parameters,
        n_iterations=200,
        population_size=5,
    )
    best_params, history = optimizer.optimize()
    print("\nOptimization complete!")
    print("Best parameters found:")
    for name, value in best_params.items():
        print(f"{name}: {value:.3f}")
