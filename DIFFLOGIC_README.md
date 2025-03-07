# DiffLogicCA Image Sampling for Pytti

This is an experimental implementation of Differentiable Logic Cellular Automata (DiffLogicCA) for the Pytti text-to-image system. Based on the research paper "Differentiable Logic Cellular Automata: From Game of Life to pattern generation with learned recurrent circuits".

## What is DiffLogicCA?

DiffLogicCA combines two powerful concepts:
1. Neural Cellular Automata (NCA) - Systems of cells that update based on their neighbors
2. Differentiable Logic Gate Networks - Neural networks using binary logic gates that remain differentiable

This implementation creates a novel image sampling method that uses binary (0/1) states for each cell, processed through learned logic circuits. The system can evolve patterns over time, creating emergent behaviors from simple rules.

## Features

- **Binary representation**: Images are encoded as binary states.
- **Differentiable Logic Gates**: 16 different logic operations (AND, OR, XOR, etc.) implemented in a differentiable way.
- **Perception Circuits**: Process neighborhoods of cells to extract patterns.
- **Update Circuits**: Evolve cell states over time based on perception.
- **Natural emergence**: Complex patterns emerge from simple rules.

## Usage

1. **Simple test**: Run `test_difflogic.py` to see the system in action.
2. **Growing patterns**: Watch patterns emerge from a simple seed.
3. **Image evolution**: See how the system transforms input images.

## Implementation Details

The implementation follows the architecture described in the paper:

1. **Cell State**: Each cell has a binary state vector (channels).
2. **Perception Stage**: Cells perceive their neighborhood using logic circuits.
3. **Update Stage**: Cells update their state based on perception and current state.
4. **Visualization**: The first 3 channels are interpreted as RGB values.

## Benefits for Pytti

- **Unique sampling method**: Creates different kinds of patterns than traditional methods.
- **Emergent behaviors**: Can produce unexpected and interesting results.
- **Efficient binary operations**: Potentially faster than continuous operations.
- **Controllable growth**: Patterns grow naturally from seeds.

## Limitations

- This is an experimental implementation and may not be as stable as other image models.
- Training the logic circuits requires special techniques not yet fully implemented.
- The system may produce unpredictable results.

## Future Work

- Implement training for specific image targets
- Add asynchronous updates for more natural evolution
- Integrate with Pytti's prompt system
- Add specialized gates for forgetting or maintaining state

## Credits

Based on research by Pietro Miotti, Eyvind Niklasson, Ettore Randazzo, and Alexander Mordvintsev at Google's Paradigms of Intelligence Team.

Integrated with the Pytti text-to-image system developed by H. R. (@sportsracer48). 