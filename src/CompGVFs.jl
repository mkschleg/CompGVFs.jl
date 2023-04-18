module CompGVFs

# using Plots
# using RecipesBase, Colors
# using MinimalRLCore
# organization is key!



# Simple learning Code
# this is now JUST GVFQuestions
include("GVFQuestions.jl")
include("Approximators.jl")

struct GVF{F}
    question::GVFQuestion
    answer::F # A function approximator
end

struct Horde
    gvfs::Vector{GVF}
end

include("Learn.jl")

# Environment Code
include("Environments.jl")

include("exp_utils.jl")

# # Experiments
include("exp.jl")


end
