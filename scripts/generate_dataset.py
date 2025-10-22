"""
Generates dataset using beam search to find shortest path.
For example 100 states for each random walk distance K from 1 to 40.
Random walk of length K != optimal distance.

Usage: python scripts/generate_dataset.py --group_id 54 --target_id 0 --model_id 333 --epoch 8192 --output_path experiments/datasets/cube_3x3_labeled.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilgrim.model import Pilgrim
from pilgrim.searcher import Searcher
from pilgrim.utils import generate_inverse_moves
from tqdm import tqdm


def load_cube_group_and_target(group_id, target_id, device):
    """Load cube group data and target state."""
    # Load cube data from generators file
    generator_path = f"generators/p{group_id:03d}.json"

    with open(generator_path, "r") as f:
        data = json.load(f)
        all_moves = torch.tensor(data["actions"], dtype=torch.int64, device=device)
        action_names = data["names"]

    inverse_moves = torch.tensor(
        generate_inverse_moves(action_names), dtype=torch.int64, device=device
    )

    # Load target state
    target_path = f"targets/p{group_id:03d}-t{target_id:03d}.pt"

    target = torch.load(target_path, map_location=device)
    if target.dim() > 1:
        target = target.squeeze()

    print(f"Loaded cube group {group_id}, target {target_id}")
    print(f"Number of moves: {len(action_names)}")
    print(f"Target shape: {target.shape}")

    return all_moves, action_names, inverse_moves, target


def generate_random_walks_at_distance(
    all_moves, inverse_moves, target, K, device, num_states=100
):
    """
    Generate random walks of length K. Copied from trainer.py.

    Args:
        all_moves: All possible moves tensor
        inverse_moves: Inverse moves mapping
        target: Target state (solved cube)
        K: Exact distance to generate
        device: Device to use
        num_states: Number of states to generate

    Returns:
        Tensor of states at distance K
    """
    states = target.repeat(num_states, 1)
    last_moves = torch.full((num_states,), -1, dtype=torch.int64, device=device)
    n_gens = all_moves.size(0)

    for step in range(K):
        possible_moves = torch.ones(
            (num_states, n_gens), dtype=torch.bool, device=device
        )
        # avoid inverse moves if we have a previous move (not -1)
        valid_states = last_moves >= 0
        if valid_states.any():
            possible_moves[valid_states, inverse_moves[last_moves[valid_states]]] = (
                False
            )
        next_moves = torch.multinomial(possible_moves.float(), 1).squeeze()
        # if num_states == 1
        if next_moves.dim() == 0:
            next_moves = next_moves.unsqueeze(0)
        states = torch.gather(states, 1, all_moves[next_moves])
        last_moves = next_moves

    return states


def generate_states_by_distance(
    all_moves, inverse_moves, target, device, K_max=40, states_per_K=100
):
    """
    Generate states at specific distances from the target using random walks.

    Args:
        all_moves: All possible moves tensor
        inverse_moves: Inverse moves mapping
        target: Target state
        K_max: Maximum distance to generate states for
        states_per_K: Number of states to generate for each distance
        device: Device to use for computation

    Returns:
        Dictionary with states and their distances
    """
    states_by_distance = {}

    for K in range(1, K_max + 1):
        print(f"Generating {states_per_K} states at distance {K}...")

        # Generate random walks of exactly length K
        states = generate_random_walks_at_distance(
            all_moves,
            inverse_moves,
            target,
            K,
            device,
            states_per_K,
        )

        states_by_distance[K] = states
        print(f"Generated {len(states)} states at distance {K}")

    return states_by_distance


def load_trained_model(group_id, target_id, model_id, epoch, device):
    """Load a pre-trained model for beam search."""
    # Load model info
    log_dir = "logs"
    with open(
        f"{log_dir}/model_p{int(group_id):03d}-t{int(target_id):03d}_{model_id}.json",
        "r",
    ) as json_file:
        info = json.load(json_file)

    # Derive important group parameters from the loaded data
    generator_path = f"generators/p{group_id:03d}.json"
    with open(generator_path, "r") as f:
        data = json.load(f)
        all_moves = torch.tensor(data["actions"], dtype=torch.int64, device=device)
        state_size = all_moves.size(1)

    target_path = f"targets/p{group_id:03d}-t{target_id:03d}.pt"
    V0 = torch.load(target_path, weights_only=True, map_location=device)
    num_classes = torch.unique(V0).size(0)

    # Load model and weights
    model = Pilgrim(
        num_classes=num_classes,
        state_size=state_size,
        hd1=info["hd1"],
        hd2=info["hd2"],
        nrd=info["nrd"],
        activation_function=info.get("activation", "relu"),
        use_batch_norm=info.get("use_batch_norm", True),
    )
    model.load_state_dict(
        torch.load(
            f"weights/p{int(group_id):03d}-t{int(target_id):03d}_{model_id}_e{epoch:05d}.pth",
            weights_only=False,
            map_location=device,
        )
    )
    model.eval()

    # Fix float16
    model = model.half()
    model.dtype = torch.float16

    if V0.min() < 0:
        model.z_add = -V0.min().item()

    return model.to(device)


def label_states_with_beam_search(
    states_by_distance,
    model,
    all_moves,
    target,
    device,
    beam_width,
    num_attempts,
    num_steps,
):
    """
    Label states using beam search to find shortest paths.

    Args:
        states_by_distance: Dictionary of states by distance
        model: Pre-trained model for beam search
        all_moves: All possible moves tensor
        target: Target state
        device: Device to use
        beam_width: Beam search width
        num_attempts: Number of search attempts
        num_steps: Number of search steps

    Returns:
        Labeled dataset with states, distances, found lengths, and solution flags
    """
    print("Starting beam search labeling...")

    searcher = Searcher(model, all_moves, target, device=device, verbose=0)

    labeled_data = {
        "states": [],
        "walk_distances": [],
        "found_lengths": [],
        "solved_flags": [],
        "solution_times": [],
        "solution_moves": [],
    }

    total_states = sum(len(states) for states in states_by_distance.values())
    processed = 0

    print(f"Starting beam search labeling for {total_states} states...")
    print(f"Beam width: {beam_width}, Max attempts: {num_attempts}")

    for K, states in states_by_distance.items():
        print(f"Labeling {len(states)} states at distance {K}...")

        for i, state in enumerate(tqdm(states, desc=f"Distance {K}")):
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_states} states...")

            start_time = time.time()

            # Run beam search
            result = searcher.get_solution(
                state=state,
                B=beam_width,
                num_attempts=num_attempts,
                num_steps=num_steps,
            )
            dt = time.time() - start_time
            # print(f"Beam search time: {dt:.2f}s")

            # Extract solution info
            moves, attempts = result[:2]
            solved = moves is not None
            length = len(moves) if solved else -1

            # Store results
            labeled_data["states"].append(state.cpu())
            labeled_data["walk_distances"].append(K)
            labeled_data["found_lengths"].append(length)
            labeled_data["solved_flags"].append(solved)
            labeled_data["solution_times"].append(dt)
            labeled_data["solution_moves"].append(moves.tolist() if solved else [])

            processed += 1

    # Convert to tensors
    labeled_data["states"] = torch.stack(labeled_data["states"])
    labeled_data["walk_distances"] = torch.tensor(labeled_data["walk_distances"])
    labeled_data["found_lengths"] = torch.tensor(labeled_data["found_lengths"])
    labeled_data["solved_flags"] = torch.tensor(labeled_data["solved_flags"])
    labeled_data["solution_times"] = torch.tensor(labeled_data["solution_times"])

    return labeled_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labeled dataset for 3x3 cube"
    )
    parser.add_argument("--group_id", type=int, help="Group ID.")
    parser.add_argument("--target_id", type=int, default=0, help="Target ID.")
    parser.add_argument(
        "--dataset", type=str, default="rnd", help="Type of dataset, 'santa' or 'rnd'."
    )
    parser.add_argument("--model_id", type=int, required=True, help="Model ID.")
    parser.add_argument(
        "--epoch", type=int, required=True, help="Number of epochs to train model."
    )
    parser.add_argument("--B", type=int, default=2**18, help="Beam size")
    parser.add_argument(
        "--num_attempts", type=int, default=1, help="Number of allowed restarts."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of allowed steps in one beam search run.",
    )
    parser.add_argument(
        "--K_max", type=int, default=40, help="Maximum distance to generate"
    )
    parser.add_argument(
        "--states_per_K", type=int, default=100, help="States per distance"
    )
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument(
        "--verbose", type=int, default=0, help="Use tqdm if verbose > 0."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="experiments/datasets/cube_3x3_labeled.pt",
        help="Output path for labeled dataset",
    )

    args = parser.parse_args()

    # Set device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda", args.device_id)
    else:
        device = torch.device("cpu")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Start dataset generation with {device}.")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Group ID: {args.group_id}")
    print(f"Target ID: {args.target_id}")
    print(f"Model ID: {args.model_id}")
    print(f"Epoch: {args.epoch}")
    print(f"K_max: {args.K_max}")
    print(f"States per K: {args.states_per_K}")
    print(f"Total states: {args.K_max * args.states_per_K}")
    print(f"Beam width: {args.B}")
    print(f"Device: {device}")

    # Load group data and target
    print("\n1. Loading group data and target...")
    all_moves, action_names, inverse_moves, target = load_cube_group_and_target(
        args.group_id, args.target_id, device
    )

    # Load trained model
    print("\n2. Loading trained model...")
    model = load_trained_model(
        args.group_id, args.target_id, args.model_id, args.epoch, device
    )

    # Generate states by distance
    print("\n3. Generating states by distance...")
    states_by_distance = generate_states_by_distance(
        all_moves,
        inverse_moves,
        target,
        device,
        args.K_max,
        args.states_per_K,
    )

    # Label states with beam search
    print("\n4. Labeling states with beam search...")
    labeled_data = label_states_with_beam_search(
        states_by_distance=states_by_distance,
        model=model,
        all_moves=all_moves,
        target=target,
        device=device,
        beam_width=args.B,
        num_attempts=args.num_attempts,
        num_steps=args.num_steps,
    )

    # Save dataset
    print(f"\n5. Saving dataset to {args.output_path}...")
    torch.save(labeled_data, args.output_path)

    print(f"Total states: {len(labeled_data['states'])}")
    print(f"Solved states: {labeled_data['solved_flags'].sum().item()}")
    print(f"Solve rate: {labeled_data['solved_flags'].float().mean().item():.2%}")

    # Filter out failed searches for length statistics
    valid_lengths = labeled_data["found_lengths"][labeled_data["found_lengths"] >= 0]
    if len(valid_lengths) > 0:
        print(f"Average found length: {valid_lengths.float().mean().item():.2f}")
        print(f"Min found length: {valid_lengths.min().item()}")
        print(f"Max found length: {valid_lengths.max().item()}")
    else:
        print("No valid solutions found")

    print(
        f"Average solution time: {labeled_data['solution_times'].float().mean().item():.2f}s"
    )
    for K in range(1, min(11, args.K_max + 1)):  # Show first 10 distances
        mask = labeled_data["walk_distances"] == K
        if mask.any():
            solved_rate = labeled_data["solved_flags"][mask].float().mean().item()
            valid_mask = mask & (labeled_data["found_lengths"] >= 0)
            if valid_mask.any():
                avg_length = (
                    labeled_data["found_lengths"][valid_mask].float().mean().item()
                )
                min_length = labeled_data["found_lengths"][valid_mask].min().item()
                max_length = labeled_data["found_lengths"][valid_mask].max().item()
                print(
                    f"  Distance {K:2d}: {mask.sum().item():3d} states, "
                    f"solve rate: {solved_rate:6.2%}, "
                    f"avg length: {avg_length:5.2f} "
                    f"(min: {min_length:2d}, max: {max_length:2d})"
                )
            else:
                print(
                    f"  Distance {K:2d}: {mask.sum().item():3d} states, solve rate: {solved_rate:6.2%}, no valid solutions"
                )

    walk_distances = labeled_data["walk_distances"]
    found_lengths = labeled_data["found_lengths"]
    valid_mask = found_lengths >= 0

    if valid_mask.any():
        # Compare walk distances vs found lengths
        differences = found_lengths[valid_mask] - walk_distances[valid_mask]
        shorter_count = (differences < 0).sum().item()
        same_count = (differences == 0).sum().item()
        longer_count = (differences > 0).sum().item()

        print("Random walk vs optimal distance comparison:")
        print(
            f"Shorter optimal path found: {shorter_count:4d} ({shorter_count / valid_mask.sum().item():.1%})"
        )
        print(
            f"Same length: {same_count:4d} ({same_count / valid_mask.sum().item():.1%})"
        )
        print(
            f"Longer optimal path: {longer_count:4d} ({longer_count / valid_mask.sum().item():.1%})"
        )

        if shorter_count > 0:
            avg_improvement = (-differences[differences < 0]).float().mean().item()
            print(f"Average improvement when shorter: {avg_improvement:.2f} moves")

    print(f"\nDataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
