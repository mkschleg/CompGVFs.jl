module CompGVFs

# using Plots
# using RecipesBase, Colors
# using MinimalRLCore


# Simple learning Code
include("GVFs.jl")
include("Learn.jl")

# Environment Code
include("Environments.jl")

include("exp_utils.jl")

# Experiments
include("exp.jl")


end
