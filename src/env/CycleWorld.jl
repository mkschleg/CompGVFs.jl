
import MinimalRLCore

mutable struct CycleWorld <: MinimalRLCore.AbstractEnvironment
    chain_length::Int64
    agent_state::Int64
    partially_observable::Bool
    CycleWorld(chain_length::Int64; 
	       rng=Random.GLOBAL_RNG, 
	       partially_observable=true) =
		   new(chain_length,
		       0,
		       partially_observable)
end

# ╔═╡ d864d151-5d71-4851-a8ec-72e415573476
function MinimalRLCore.reset!(env::CycleWorld, rng=nothing; kwargs...)
    env.agent_state = 0
end

# ╔═╡ c0c9cf24-7294-4e26-8d0f-f978c9996e04
MinimalRLCore.get_actions(env::CycleWorld) = Set(1:1)

# ╔═╡ e52b2a09-e4f7-4756-96a8-6dce21a3eaac
function MinimalRLCore.environment_step!(env::CycleWorld,
                                         action, 
                                         rng=nothing; 
                                         kwargs...)
    env.agent_state = (env.agent_state + 1) % env.chain_length
end

# ╔═╡ 4bffba42-2d29-46ac-9bdd-59f348d92a30
MinimalRLCore.get_reward(env::CycleWorld) = 0 # -> get the reward of the environment

# ╔═╡ 7b266d56-2a4d-40de-9a9f-3a9680bebacc
fully_observable_state(env::CycleWorld) = [env.agent_state+1]

# ╔═╡ a2abdbd6-c2a4-443f-b724-6bf574436ee7
function partially_observable_state(env::CycleWorld)
    state = zeros(1)
    if env.agent_state == 0
	state[1] = 1
    end
    return state
end

# ╔═╡ 924b6945-3826-4138-aa00-206ddb9840a5
function partially_observable_state(state::Int)
    state = zeros(1)	
    if state == 0
	state[1] = 1
    end
    return state
end

# ╔═╡ 31115f9d-c987-4b11-9b02-ee7942606f7a
function MinimalRLCore.get_state(env::CycleWorld) # -> get state of agent
    if env.partially_observable
	return partially_observable_state(env)
    else
	return fully_observable_state(env)
    end
end

# ╔═╡ f18749cd-0160-44ed-b91f-30a6a9dfad8f
function MinimalRLCore.is_terminal(env::CycleWorld) # -> determines if the agent_state is terminal
    return false
end

# ╔═╡ 1775dc2a-91ea-4303-a5c7-b260dd06278d
function Base.show(io::IO, env::CycleWorld)
    model = fill("0", env.chain_length)
    model[1] = "1"
    println(io, join(model, ' '))
    model = fill("-", env.chain_length)
    model[env.agent_state + 1] = "^"
    println(io, join(model, ' '))
    # println(env.agent_state)
end
