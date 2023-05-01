module CompGVFs

# using Plots
# using RecipesBase, Colors
# using MinimalRLCore
include("Approximators.jl")

# Simple learning Code
include("GVFs.jl")
include("Learn.jl")

include("Learner.jl")

include("CompGVFParameters.jl")

# What is the difference between a learner and an agent?
# A learner is concerned with a single learning update and a horde/demon
# An Agent may have multiple learners. One to learn the agent's behabior, and others
# to learn predictions.

# Environment Code
include("Environments.jl")

include("exp_utils.jl")

# Experiments
include("exp.jl")


end
